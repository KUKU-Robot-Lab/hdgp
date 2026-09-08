"""손 full-joint 법칙과 액션한계 구성을 **수치로 실행**한다 — 소스 문자열 검사가 못 잡는 것을 잡는다.

왜 필요한가: 16환경 스모크는 손이 물체에서 멀어 파지 경로가 한 번도 안 돈다. 법칙이 실제로
어떻게 움직이는지는 학습을 돌려야만 보이는데, 그때는 이미 GPU 를 며칠 쓴 뒤다. 그래서 두
메서드 소스를 stub self 로 직접 실행한다.

여기서 잠그는 것(09.08 사용자 확정 "SimToolReal 처럼 풀 조인트"):
  ① a=−1 → lo, a=+1 → hi, a=0 → 중앙 (α=1 이면 즉시). SimToolReal action_utils.py:61-69 와 같은 꼴.
  ② α<1 이면 관절 목표 EMA `q* ← α·raw + (1−α)·q*_prev`, 출발점은 리셋 자세(`_syn_target`).
  ③ 범위 = soft limit ∩ 프로필 override — **좁히기만** 한다. 넓히려 해도 soft limit 이 이긴다.
  ④ 폭 0 액션 칸·리셋 자세 범위 밖은 **부팅에서 죽는다**(하한 0 이 유일한 손등 방어선이라 조용히 못 넘어간다).
  ⑤ `_syn_close` 는 진단용 정규화 목표 (q*−lo)/(hi−lo) 다.

실행:
    cd hdgp && PYTHONPATH=source/openarm python3 -m pytest \
        source/openarm/openarm/agnostic/tasks/grasp_fj/tests/test_fj_hand_law.py -q
"""

from __future__ import annotations

import ast
import re
import textwrap
import types
from pathlib import Path

import pytest
import torch

_ENV_SRC = (Path(__file__).resolve().parent.parent / "grasp_fj_env.py").read_text(encoding="utf-8")

N = 3                                  # env 수
_NAMES = ("j_a_1", "j_a_3", "j_b_3", "j_b_4")   # 테솔로 축소판: 외전 1개 + 굴곡 3개(_3/_4 대칭)
_LO = torch.tensor([-0.4, -1.571, -1.571, -1.571])
_HI = torch.tensor([0.6, 1.571, 1.571, 1.571])
_Q0 = torch.tensor([0.0, -0.5, 0.0, 0.0])        # 리셋 자세: j_a_3 만 −0.5 pre-curl(엄지 _3 축소판)


def _methods():
    """제어 법칙 메서드 소스를 그대로 뽑아 실행 가능한 함수로 만든다."""
    tree = ast.parse(_ENV_SRC)
    out = {}
    for name in ("_hand_targets", "_build_hand_action_range", "_hand_mask"):
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == name)
        src = textwrap.dedent("\n".join(_ENV_SRC.split("\n")[node.lineno - 1:node.end_lineno]))
        ns = {"torch": torch}
        exec(src, ns)  # noqa: S102 — 소스 자신의 제어 법칙
        out[name] = ns[name]
    return out


class _Hand:
    """`_hand_targets`/`_build_hand_action_range` 가 읽는 것만 갖춘 stub."""

    def __init__(self, *, ema=0.1, direct=True, override=None, q0=None, lo=None, hi=None):
        m = _methods()
        self._ht = types.MethodType(m["_hand_targets"], self)
        self._build_hand_action_range = types.MethodType(m["_build_hand_action_range"], self)
        self._hand_mask = types.MethodType(m["_hand_mask"], self)
        self.cfg = types.SimpleNamespace(hand_direct=direct, hand_ema=ema, synergy_close_speed=0.005,
                                         hand_reset_clamp_max_rad=0.6)
        self.device = "cpu"
        self._syn_lo = (_LO if lo is None else torch.tensor(lo)).clone()
        self._syn_hi = (_HI if hi is None else torch.tensor(hi)).clone()
        nh = len(self._syn_lo)
        self._syn_open = torch.zeros(nh)
        self._syn_grip = torch.ones(nh)
        self._syn_movable = torch.zeros(nh, dtype=torch.bool)
        self._syn_ids = list(range(nh))
        _q0 = (_Q0 if q0 is None else torch.tensor(q0)).clone()
        self._syn_target = _q0.unsqueeze(0).repeat(N, 1)          # 부모 리셋: _syn_target = q0
        self._syn_close = torch.zeros(N, nh)
        self.profile = types.SimpleNamespace(
            name="stub", hand_joint_names=_NAMES[:nh],
            hand_action_limit_override=dict(override or {}))
        def _find(rx, preserve_order=False):
            # IsaacLab resolve_matching_names: **fullmatch**, 한 키도 못 잡으면 ValueError(빈 리스트 아님).
            ids = [i for i, n in enumerate(_NAMES[:nh]) if re.fullmatch(rx, n)]
            if not ids:
                raise ValueError("Not all regular expressions are matched!")
            return ids, [_NAMES[i] for i in ids]
        _hard = torch.stack([self._syn_lo, self._syn_hi], dim=1).unsqueeze(0)     # (1, nh, 2) = soft(factor 1.0)
        self.robot = types.SimpleNamespace(
            data=types.SimpleNamespace(default_joint_pos=_q0.unsqueeze(0), joint_pos_limits=_hard),
            find_joints=_find)
        self._build_hand_action_range()
        if direct:
            self._syn_target = self._hand_reset_q.unsqueeze(0).repeat(N, 1)   # B `_reset_idx` 가 심는 값

    def step(self, a):
        t = self._ht(a)
        self._syn_target = t              # A 의 `_hand_command` 가 하는 일
        return t

    def run(self, a, n):
        for _ in range(n):
            t = self.step(a)
        return t


def _act(*vals):
    return torch.tensor(vals, dtype=torch.float).unsqueeze(0).repeat(N, 1)


# ---------------------------------------------------------------- ① 선형 매핑
def test_action_endpoints_map_to_the_action_limits_when_ema_is_one():
    """★a=−1 → lo, a=+1 → hi, a=0 → 중앙. α=1 이면 평활 없이 즉시."""
    h = _Hand(ema=1.0)
    t = h.step(_act(-1, -1, -1, -1))
    assert torch.allclose(t[0], h._act_lo, atol=1e-6)
    t = h.step(_act(1, 1, 1, 1))
    assert torch.allclose(t[0], h._act_hi, atol=1e-6)
    t = h.step(_act(0, 0, 0, 0))
    assert torch.allclose(t[0], 0.5 * (h._act_lo + h._act_hi), atol=1e-6)


def test_out_of_range_actions_are_clipped_not_extrapolated():
    h = _Hand(ema=1.0)
    t = h.step(_act(-7, 9, -7, 9))
    assert torch.allclose(t[0], torch.stack([h._act_lo[0], h._act_hi[1], h._act_lo[2], h._act_hi[3]]), atol=1e-6)


# ---------------------------------------------------------------- ② EMA
def test_joint_target_ema_reproduces_the_moving_average_law_from_the_reset_pose():
    """★`q* ← α·raw + (1−α)·q*_prev`, q*_{-1} = 리셋 자세(SimToolReal prev_targets = joint_pos)."""
    a = 0.1
    h = _Hand(ema=a)
    raw = h._act_hi.clone()                     # a=+1
    prev = _Q0.clone()
    for i in range(1, 31):
        t = h.step(_act(1, 1, 1, 1))
        prev = a * raw + (1.0 - a) * prev
        assert torch.allclose(t[0], prev, atol=1e-6), i
    # 63% 를 10 스텝(0.17s), 90% 를 22 스텝 — SimToolReal handMovingAverage 0.1 그대로.
    frac = float((t[0, 3] - _Q0[3]) / (raw[3] - _Q0[3]))
    assert 0.90 < frac < 0.99


def test_first_step_after_reset_does_not_jump_when_reset_pose_is_inside_the_range():
    """리셋 자세가 범위 안이면 a=0 첫 스텝의 이동은 α·(중앙−q0) 뿐이다 — clamp 로 튀지 않는다."""
    h = _Hand(ema=0.1)
    t = h.step(_act(0, 0, 0, 0))
    mid = 0.5 * (h._act_lo + h._act_hi)
    assert torch.allclose(t[0], _Q0 + 0.1 * (mid - _Q0), atol=1e-6)


# ---------------------------------------------------------------- ③ 범위 = soft limit ∩ override
def test_default_range_is_the_soft_limit_and_all_joints_are_movable():
    """override 가 없으면 SimToolReal 그대로 — soft limit 전폭, 죽은 칸 0."""
    h = _Hand(override={})
    assert torch.allclose(h._act_lo, _LO) and torch.allclose(h._act_hi, _HI)
    assert bool(h._syn_movable.all()), "full-joint 는 20칸 전부 가동이다"


def test_override_only_narrows_never_widens():
    """★교집합이다. 하한을 URDF 아래로, 상한을 URDF 위로 적으면 soft limit 이 이긴다."""
    h = _Hand(override={r"j_b_[34]$": (0.0, None), r"j_a_1$": (-9.0, 9.0)})
    assert h._act_lo.tolist() == pytest.approx([-0.4, -1.571, 0.0, 0.0])
    assert h._act_hi.tolist() == pytest.approx([0.6, 1.571, 1.571, 1.571])


def test_override_regex_that_matches_nothing_is_a_boot_error():
    """오타가 조용히 '전폭' 으로 돌면 손등 방어선이 사라진다 — 시끄럽게 죽어야 한다."""
    with pytest.raises(RuntimeError, match="아무것도 못 잡았다"):
        _Hand(override={r"j_z_9$": (0.0, None)})


# ---------------------------------------------------------------- ④ 부팅 fail-loud
def test_zero_width_action_slot_is_a_boot_error():
    with pytest.raises(RuntimeError, match="폭 0 액션 칸"):
        _Hand(override={r"j_b_3$": (1.571, None)})      # 하한을 상한까지 올림


def test_reset_pose_outside_the_range_is_clamped_into_the_seed_not_rejected():
    """★엄지 _3 축소판: 프로필 리셋 −0.5 인데 하한 0 → B 는 리셋에서 0 으로 clamp 해 관절 상태·EMA 시드에 심는다.
    첫 스텝(a=0)의 이동은 α·(중앙−0) 뿐이다 — 0.5 rad 튐이 없다.
    """
    h = _Hand(override={r"j_a_3$": (0.0, None)})
    assert h._hand_reset_q.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    t = h.step(_act(0, 0, 0, 0))
    mid = 0.5 * (h._act_lo + h._act_hi)
    assert torch.allclose(t[0], h._hand_reset_q + 0.1 * (mid - h._hand_reset_q), atol=1e-6)


def test_reset_pose_far_outside_the_range_is_a_boot_error():
    """clamp 이동량이 `hand_reset_clamp_max_rad` 를 넘으면 범위와 다른 자세다 → 부팅 거부."""
    with pytest.raises(RuntimeError, match="벗어난다"):
        _Hand(override={r"j_a_3$": (0.2, None)})           # −0.5 → 0.2 = 0.7 rad > 0.6


# ---------------------------------------------------------------- ⑤ 진단 정규화
def test_syn_close_is_the_normalized_target_for_diagnostics():
    h = _Hand(ema=1.0, override={r"j_b_[34]$": (0.0, None)})
    h.step(_act(1, 1, 1, 1))
    assert torch.allclose(h._syn_close[0], torch.ones(4), atol=1e-6)
    h.step(_act(-1, -1, -1, -1))
    assert torch.allclose(h._syn_close[0], torch.zeros(4), atol=1e-6)


def test_synergy_path_is_untouched_when_hand_direct_is_off():
    """`hand_direct=False` 면 `_build_hand_action_range` 는 A 의 open→grip 끝점만 남기고 손대지 않는다."""
    h = _Hand(direct=False, override={r"j_b_[34]$": (0.0, None)})
    assert torch.equal(h._act_lo, h._syn_open) and torch.equal(h._act_hi, h._syn_grip)
    assert not bool(h._syn_movable.any()), "시너지 모드에서는 가동 마스크를 건드리지 않는다"


# ---------------------------------------------------------------- 실제 프로필 대조 (URDF 없이)
def test_tesollo_right_override_floors_exactly_the_twelve_distal_flexions():
    """프로필(관절명 소유자)의 override 가 `_3/_4` 10개를 모두 잡고, 그 외는 안 건드린다."""
    from openarm.agnostic.tasks.grasp_fj.robot_profiles import PROFILES
    p = PROFILES["tesollo_right"]
    ov = dict(p.hand_action_limit_override)
    hit = {}
    for n in p.hand_joint_names:
        for rx, (lo, hi) in ov.items():
            if re.fullmatch(rx, n):                      # IsaacLab 과 같은 의미
                assert n not in hit, f"{n} 이 규칙 두 개에 걸린다"
                hit[n] = (lo, hi)
    assert set(hit) == {n for n in p.hand_joint_names if n.endswith(("_3", "_4"))}
    assert all(v == (0.0, None) for v in hit.values()), "엄지 _3 포함 전부 하한 0(사용자 재확정)"
    # 리셋(open) 자세가 하한 아래인 관절은 B 리셋이 clamp 한다 — 이동량이 cfg 상한(0.6) 안이어야 한다.
    for n, q in zip(p.hand_joint_names, p.hand_open_pose):
        if n in hit and q < hit[n][0]:
            assert hit[n][0] - q <= 0.6, (n, q, hit[n])


# ---------------------------------------------------------------- 목표당 스텝 예산
def _clock():
    """`_restart_goal_clock` 소스를 stub self 로 실행 가능하게 만든다."""
    node = next(n for n in ast.walk(ast.parse(_ENV_SRC))
                if isinstance(n, ast.FunctionDef) and n.name == "_restart_goal_clock")
    src = textwrap.dedent("\n".join(_ENV_SRC.split("\n")[node.lineno - 1:node.end_lineno]))
    ns = {"torch": torch}
    exec(src, ns)  # noqa: S102 — 소스 자신의 제어 법칙
    return ns["_restart_goal_clock"]


class _Clock:
    """`episode_length_buf` 는 IsaacLab 에서 **long** 이다 — dtype 을 실제와 맞춘다."""

    def __init__(self, restart, lens, success):
        self.cfg = types.SimpleNamespace(goal_clock_restart_step=restart)
        self.episode_length_buf = torch.tensor(lens, dtype=torch.long)
        self._success_now = torch.tensor(success, dtype=torch.bool)
        self.extras = {}
        self._fn = types.MethodType(_clock(), self)

    def run(self):
        self._fn()
        return self.episode_length_buf.tolist()


def test_goal_clock_restart_only_touches_successful_envs():
    """★성공한 env 만 시계가 되돌아간다. 나머지는 한 스텝도 안 건드린다."""
    c = _Clock(2, [10, 250, 599, 400], [False, True, True, False])
    assert c.run() == [10, 2, 2, 400]
    assert float(c.extras["task/goal_clock_restart"]) == pytest.approx(0.5)


def test_goal_clock_restart_is_a_noop_when_off():
    """−1 = 끔 = 현행(에피소드당 예산). 버퍼도 extras 도 안 건드린다."""
    c = _Clock(-1, [10, 250, 599], [False, True, True])
    assert c.run() == [10, 250, 599]
    assert "task/goal_clock_restart" not in c.extras


def test_goal_clock_restart_keeps_long_dtype_and_is_in_place():
    """dtype 이 float 로 바뀌면 time-out 술어(`buf >= max-1`)가 조용히 어긋난다."""
    c = _Clock(2, [10, 250], [False, True])
    before = c.episode_length_buf.data_ptr()
    c.run()
    assert c.episode_length_buf.dtype == torch.long
    assert c.episode_length_buf.data_ptr() == before, "in-place 여야 부모가 같은 버퍼를 본다"


def test_goal_clock_restart_dodges_every_small_buffer_consumer():
    """★값이 2 인 이유의 회귀 울타리 — 0/1 에 반응하는 소비자 넷을 전부 피한다.

    · 액션·관측·**물체**(10스텝) 지연 flush 는 `episode_length_buf == 0` 에서 터진다.
    · `_fresh` 리셋 진단은 `<= 1` 이다 — 오염되면 `reset/arm_q_dev_max` 가 눈이 먼다.
    """
    cfg_src = (Path(__file__).resolve().parent.parent / "grasp_fj_env_cfg.py").read_text(encoding="utf-8")
    leaf = cfg_src[cfg_src.index("class GraspFJTesolloRightEnvCfg"):]
    r = int(re.search(r"goal_clock_restart_step: int = (-?\d+)", leaf).group(1))
    assert r >= 2, f"0/1 은 지연 flush 와 _fresh 진단을 깬다 (got {r})"
    lens = _Clock(r, [5], [True]).run()
    assert lens[0] > 1, "리셋 진단(_fresh <= 1)에 걸린다"

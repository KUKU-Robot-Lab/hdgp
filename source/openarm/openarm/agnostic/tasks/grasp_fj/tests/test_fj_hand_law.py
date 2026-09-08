"""손 폐쇄 법칙을 **수치로 실행**한다 — 소스 문자열 검사가 못 잡는 것을 잡는다.

왜 필요한가: 16환경 스모크에서 `close_gate` 가 0(손이 물체에서 24.5cm) 이라 폐쇄 경로가
한 번도 안 돌았다. 게이트가 열렸을 때 실제로 어떻게 닫히는지는 학습을 돌려야만 보이는데,
그때는 이미 GPU 를 며칠 쓴 뒤다. 그래서 두 메서드 소스를 stub self 로 직접 실행한다.

여기서 잠그는 것:
  ① `synergy_close_ema = 0` 이면 **현행 램프와 완전히 동일**하다(포크 회귀 방어).
  ② `ema > 0` + `speed ≥ 1` 이면 `c ← α·cmd + (1−α)·c` 다(SimToolReal handMovingAverage).
  ③ `close_gate` 와 `blocked` 가 **두 모드 모두에서** 산다 — EMA 가 순서 교환 가능하다는 주장의 증명.
  ④ 푸는 방향은 게이트·blocked 를 항상 통과한다(잘못 오므린 상태 탈출).

실행:
    cd hdgp && PYTHONPATH=source/openarm python3 -m pytest \
        source/openarm/openarm/agnostic/tasks/grasp_fj/tests/test_fj_hand_law.py -q
"""

from __future__ import annotations

import ast
import textwrap
import types
from pathlib import Path

import pytest
import torch

_ENV_SRC = (Path(__file__).resolve().parent.parent / "grasp_fj_env.py").read_text(encoding="utf-8")

N, NH = 3, 4                      # 3 env · 손 관절 4 (전부 가동으로 둔다)
_OPEN = torch.tensor([0.0, 0.0, 0.0, 0.0])
_GRIP = torch.tensor([1.0, 1.0, 1.0, 1.0])


def _methods():
    """제어 법칙 메서드 소스를 그대로 뽑아 실행 가능한 함수로 만든다."""
    tree = ast.parse(_ENV_SRC)
    out = {}
    for name in ("_hand_targets", "_hand_step", "_closure_to_target",
                 "_build_hand_action_range", "_hand_mask"):
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == name)
        src = textwrap.dedent("\n".join(_ENV_SRC.split("\n")[node.lineno - 1:node.end_lineno]))
        ns = {"torch": torch}
        exec(src, ns)  # noqa: S102 — 소스 자신의 제어 법칙
        out[name] = ns[name]
    return out


class _Hand:
    """`_hand_targets`/`_hand_step` 이 읽는 것만 갖춘 stub."""

    def __init__(self, *, ema=0.0, speed=0.005, gate=1.0, blocked=False, close0=0.0,
                 mode="synergy", open_pose=None, grip_pose=None,
                 locked=(), wide=(), span=2.0, wide_span=2.0):
        m = _methods()
        # `_hand_targets` 안에서 `self._hand_step(...)` 등을 부르므로 **바인드**해야 한다.
        self._ht = types.MethodType(m["_hand_targets"], self)
        self._hand_step = types.MethodType(m["_hand_step"], self)
        self._closure_to_target = types.MethodType(m["_closure_to_target"], self)
        self._build_hand_action_range = types.MethodType(m["_build_hand_action_range"], self)
        self._hand_mask = types.MethodType(m["_hand_mask"], self)
        self.cfg = types.SimpleNamespace(hand_direct=True, synergy_close_speed=speed,
                                         synergy_close_ema=ema, hand_range_mode=mode,
                                         hand_shape_span_rad=span, hand_wide_span_rad=wide_span)
        self.device = "cpu"
        self._locked_names, self._wide_names = set(locked), set(wide)
        self._syn_open = _OPEN.clone() if open_pose is None else torch.tensor(open_pose)
        self._syn_grip = _GRIP.clone() if grip_pose is None else torch.tensor(grip_pose)
        nh = len(self._syn_open)
        self._syn_close = torch.full((N, nh), float(close0))
        self._syn_lo = torch.full((nh,), -2.0)      # 넉넉 — 여기서는 한계 clamp 를 시험하지 않는다
        self._syn_hi = torch.full((nh,), 2.0)
        self._syn_movable = (self._syn_grip - self._syn_open).abs() > 1e-4
        self._close_gate = torch.full((N,), float(gate))
        self._blocked = bool(blocked)
        _names = tuple(f"j{i}" for i in range(nh))
        self.profile = types.SimpleNamespace(
            name="stub", hand_joint_names=_names,
            hand_locked_joint_regex="|".join(sorted(self._locked_names)),
            hand_wide_shape_joint_regex="|".join(sorted(self._wide_names)))
        self._syn_ids = list(range(nh))
        # `_hand_mask` 가 부르는 articulation 스텁 — 정규식 대신 이름 집합으로 해석한다.
        _sel = {"|".join(sorted(self._locked_names)): sorted(self._locked_names),
                "|".join(sorted(self._wide_names)): sorted(self._wide_names)}
        self.robot = types.SimpleNamespace(
            find_joints=lambda rx, preserve_order=False: (
                [_names.index(n) for n in _sel.get(rx, [])], list(_sel.get(rx, []))))
        self._build_hand_action_range()
        if close0 is not None:
            self._syn_close[:] = float(close0)

    def _hand_blocked(self):
        return torch.full((N, int(self._syn_movable.sum())), self._blocked, dtype=torch.bool)

    @property
    def _c(self):
        """가동 관절 하나의 폐쇄도(테스트 편의)."""
        return float(self._syn_close[0, 0])

    def step(self, a):
        return self._ht(a)

    def run(self, a, n):
        for _ in range(n):
            t = self.step(a)
        return t


def _act(v: float) -> torch.Tensor:
    """폐쇄도 목표 v ∈ [0,1] 를 내는 액션(cmd_j = 0.5(a+1))."""
    return torch.full((N, NH), 2.0 * v - 1.0)


# ---------------------------------------------------------------- ① 램프 회귀
def test_ema_zero_is_bit_identical_to_the_ramp():
    """★`synergy_close_ema = 0` 은 오늘과 한 글자도 다르지 않아야 한다."""
    h = _Hand(ema=0.0, speed=0.005)
    for i in range(1, 21):
        h.step(_act(1.0))
        assert torch.allclose(h._syn_close, torch.full((N, NH), 0.005 * i)), i


def test_ramp_speed_is_the_sweep_axis():
    """c1/c2 가 실제로 갈린다 — 20 스텝 뒤 폐쇄도가 speed 에 비례한다."""
    for speed, want in ((0.005, 0.10), (0.020, 0.40)):
        h = _Hand(ema=0.0, speed=speed)
        h.run(_act(1.0), 20)
        assert float(h._syn_close[0, 0]) == pytest.approx(want, abs=1e-6), speed


# ---------------------------------------------------------------- ② EMA
def test_ema_reproduces_the_moving_average_law():
    """★`speed ≥ 1` 이면 clamp 가 항등이므로 `c ← α·cmd + (1−α)·c` 여야 한다."""
    a, cmd = 0.1, 1.0
    h = _Hand(ema=a, speed=1.0)
    c = 0.0
    for i in range(1, 31):
        h.step(_act(cmd))
        c = a * cmd + (1.0 - a) * c
        assert float(h._syn_close[0, 0]) == pytest.approx(c, abs=1e-6), i
    # 63% 를 10 스텝(0.17s), 90% 를 22 스텝 — 램프 0.005(200 스텝)의 20배 빠르다.
    assert 0.60 < c < 0.99


def test_ema_targets_are_affine_in_closure_so_joint_space_ema_is_identical():
    """★폐쇄도 EMA == 관절공간 EMA 라는 주장의 수치 증명(lerp 가 c 에 아핀)."""
    a = 0.1
    h = _Hand(ema=a, speed=1.0)
    tgt_prev = torch.lerp(_OPEN, _GRIP, torch.zeros(NH))
    raw = torch.lerp(_OPEN, _GRIP, torch.ones(NH))          # cmd = 1.0 의 관절 목표
    for _ in range(15):
        tgt = h.step(_act(1.0))
        tgt_prev = a * raw + (1.0 - a) * tgt_prev           # 관절공간 EMA
        assert torch.allclose(tgt[0], tgt_prev, atol=1e-6)


# ---------------------------------------------------------------- ③ 게이트·blocked 가 두 모드 모두에서 산다
@pytest.mark.parametrize("ema,speed", [(0.0, 0.005), (0.1, 1.0)])
def test_close_gate_blocks_closing_in_both_modes(ema, speed):
    """★게이트 0 이면 어느 모드에서도 폐쇄가 **전혀** 진행되지 않는다."""
    h = _Hand(ema=ema, speed=speed, gate=0.0)
    h.run(_act(1.0), 50)
    assert float(h._syn_close.abs().max()) == pytest.approx(0.0, abs=1e-6)   # float32


@pytest.mark.parametrize("ema,speed", [(0.0, 0.005), (0.1, 1.0)])
def test_close_gate_scales_but_never_reverses(ema, speed):
    """게이트는 [0,1] 배율이다 — 절반이면 절반만 진행하고 부호는 안 바뀐다."""
    full = _Hand(ema=ema, speed=speed, gate=1.0); full.run(_act(1.0), 10)
    half = _Hand(ema=ema, speed=speed, gate=0.5); half.run(_act(1.0), 10)
    assert 0.0 < float(half._syn_close[0, 0]) < float(full._syn_close[0, 0])


@pytest.mark.parametrize("ema,speed", [(0.0, 0.005), (0.1, 1.0)])
def test_blocked_holds_closing_in_both_modes(ema, speed):
    """★`blocked` 는 접촉 항 0개인 이 보상에서 감쌈을 만드는 유일한 장치 — EMA 를 켜도 살아야 한다."""
    h = _Hand(ema=ema, speed=speed, blocked=True, close0=0.3)
    h.run(_act(1.0), 50)
    assert float(h._syn_close[0, 0]) == pytest.approx(0.3, abs=1e-6), "막혔는데 더 조였다"   # float32


# ---------------------------------------------------------------- ④ 푸는 방향은 항상 통과
@pytest.mark.parametrize("ema,speed", [(0.0, 0.005), (0.1, 1.0)])
def test_release_always_passes_gate_and_blocked(ema, speed):
    """잘못 오므린 상태에서 못 빠져나오면 정책이 갇힌다."""
    h = _Hand(ema=ema, speed=speed, gate=0.0, blocked=True, close0=0.8)
    h.run(_act(0.0), 60)
    assert float(h._syn_close[0, 0]) < 0.8, "게이트 0 · blocked 인데 못 풀었다"


def test_closure_stays_in_unit_range():
    for ema, speed in ((0.0, 0.005), (0.1, 1.0)):
        h = _Hand(ema=ema, speed=speed)
        h.run(_act(1.0), 400)
        assert 0.0 <= float(h._syn_close.min()) and float(h._syn_close.max()) <= 1.0 + 1e-6
        h2 = _Hand(ema=ema, speed=speed, close0=1.0)
        h2.run(_act(0.0), 400)
        assert float(h2._syn_close.min()) >= -1e-6


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
    from pathlib import Path as _P
    cfg_src = (_P(__file__).resolve().parent.parent / "grasp_fj_env_cfg.py").read_text(encoding="utf-8")
    import re as _re
    leaf = cfg_src[cfg_src.index("class GraspFJTesolloRightEnvCfg"):]
    r = int(_re.search(r"goal_clock_restart_step: int = (-?\d+)", leaf).group(1))
    assert r >= 2, f"0/1 은 지연 flush 와 _fresh 진단을 깬다 (got {r})"
    lens = _Clock(r, [5], [True]).run()
    assert lens[0] > 1, "리셋 진단(_fresh <= 1)에 걸린다"


# ---------------------------------------------------------------- per_role 범위 해제
# tesollo_right 실측 배열의 축소판: j0 = 외전(폭 0, 하드웨어는 ±0.6), j1..j3 = 굴곡.
_FROZEN_OPEN = [0.0, 0.0, 0.0, 0.0]
_FROZEN_GRIP = [0.0, 1.9, 1.8, 1.8]


def _h(mode, **kw):
    """j0 = 외전(폭 0). 반폭 기본 2.0 = 관절한계까지 = 옛 거동(테스트 대부분이 이걸 가정)."""
    return _Hand(mode=mode, open_pose=_FROZEN_OPEN, grip_pose=_FROZEN_GRIP, **kw)


def test_synergy_mode_leaves_the_frozen_joint_dead():
    """★현행(기본) — 폭 0 관절은 액션을 줘도 목표가 안 움직인다. 이것이 고치려는 상태다."""
    h = _h("synergy", ema=0.0, speed=1.0)
    t0 = h.step(_act(0.0))[0, 0].item()
    t1 = h.run(_act(1.0), 30)[0, 0].item()
    assert t0 == pytest.approx(0.0) and t1 == pytest.approx(0.0), "폭 0 인데 움직였다"


def test_per_role_frees_only_the_zero_width_joints():
    """★폭 0 인 것만 관절 한계로 풀린다. 굴곡 관절 끝점은 **그대로**다."""
    h = _h("per_role", ema=0.0, speed=1.0)
    assert h._shape_mask.tolist() == [True, False, False, False]
    assert h._act_lo.tolist() == pytest.approx([-2.0, 0.0, 0.0, 0.0])   # 외전만 한계 lo
    assert h._act_hi.tolist() == pytest.approx([2.0, 1.9, 1.8, 1.8])    # 굴곡은 grip 그대로


def test_per_role_reset_closure_maps_to_the_open_pose():
    """★푼 관절은 폐쇄도 0 이 중립이 아니라 **한쪽 끝**이다.

    `_close_home` 을 안 심으면 리셋 직후 손이 벌어진 채로 시작한다 — 조용한 자세 오류.
    """
    h = _h("per_role", ema=0.0, speed=1.0)
    assert h._close_home[0].item() == pytest.approx(0.5)      # (0−(−2))/4
    assert h._close_home[1:].abs().max().item() == pytest.approx(0.0)
    tgt = h._closure_to_target(h._close_home.unsqueeze(0).expand(N, -1))
    assert torch.allclose(tgt[0], torch.tensor(_FROZEN_OPEN), atol=1e-6), "리셋 자세가 open 이 아니다"


def test_per_role_frozen_joint_now_moves_both_ways():
    """벌리고(+) 오므리는(−) 양방향 다 나와야 손 모양을 바꿀 수 있다."""
    h = _h("per_role", ema=0.0, speed=1.0, close0=0.5)
    assert h.run(_act(1.0), 5)[0, 0].item() > 0.5, "한쪽으로 못 벌어졌다"
    h2 = _h("per_role", ema=0.0, speed=1.0, close0=0.5)
    assert h2.run(_act(0.0), 5)[0, 0].item() < -0.5, "반대쪽으로 못 벌어졌다"


@pytest.mark.parametrize("ema,speed", [(0.0, 0.005), (0.1, 1.0)])
def test_close_gate_is_exempt_for_shape_joints_only(ema, speed):
    """★게이트 0(정렬 전)에서도 **외전은 움직이고 굴곡은 멈춘다**.

    게이트는 "정렬 전에 오므리지 마라"는 밸브다. 외전은 오므림이 아니라 **접근 중에 바꿔야
    하는 손 모양**이고, 옆에서 접근할 때 손을 벌리는 것이 이 판에서 풀어주려는 자유도다.
    """
    h = _h("per_role", ema=ema, speed=speed, gate=0.0, close0=0.5)
    flex0 = h._syn_close[0, 1].item()            # 굴곡 시작 폐쇄도(여기서는 0.5)
    t = h.run(_act(1.0), 40)
    assert t[0, 0].item() > 0.5 + 1e-3, "게이트 0 인데 외전이 안 움직였다(면제 실패)"
    assert h._syn_close[0, 1].item() == pytest.approx(flex0, abs=1e-6), (
        "게이트 0 인데 굴곡이 더 조여졌다")


@pytest.mark.parametrize("ema,speed", [(0.0, 0.005), (0.1, 1.0)])
def test_blocked_still_holds_shape_joints(ema, speed):
    """면제한 것은 게이트뿐 — `blocked` 는 외전에도 그대로 걸린다(막히면 그만 민다)."""
    h = _h("per_role", ema=ema, speed=speed, blocked=True, close0=0.5)
    h.run(_act(1.0), 40)
    assert h._syn_close[0, 0].item() == pytest.approx(0.5, abs=1e-6), "막혔는데 계속 벌렸다"


def test_per_role_needs_no_joint_name_literals():
    """★판별은 이름이 아니라 **폭**이다 — env 에 로봇 관절명을 박으면 계약 테스트가 죽는다."""
    blk = textwrap.dedent("\n".join(
        _ENV_SRC.split("\n")[
            next(n.lineno for n in ast.walk(ast.parse(_ENV_SRC))
                 if isinstance(n, ast.FunctionDef) and n.name == "_build_hand_action_range") - 1:]))
    head = blk[:blk.index("def _closure_to_target")]
    for banned in ("thumb", "index", "middle", "ring", "pinky", "_1", "_2"):
        assert f'"{banned}"' not in head, f"관절명/역할 리터럴 '{banned}' 이 박혔다"
    assert "(gr - op).abs() <= 1e-4" in head, "폭 기반 판별이어야 한다"


# ---------------------------------------------------------------- 3등급 외전 범위
# j0 = 잠금 외전(간격 좁은 손가락) · j1 = 좁게 푸는 외전 · j2 = 넓게 푸는 외전(엄지) · j3 = 굴곡
_T3_OPEN = [0.0, 0.0, 0.0, 0.0]
_T3_GRIP = [0.0, 0.0, 0.0, 1.8]


def _h3(**kw):
    return _Hand(mode="per_role", open_pose=_T3_OPEN, grip_pose=_T3_GRIP,
                 wide=("j2",), span=0.035, wide_span=0.35, **kw)


def test_no_dead_action_slot_in_per_role():
    """★액션 20칸을 선언했으면 20칸이 다 무언가를 해야 한다.

    벌림을 0 으로 잠그면 그 칸은 정책이 무슨 값을 내든 아무 일도 안 일어나는 **죽은 칸**이
    된다 — 우리가 시너지에서 고치려던 바로 그 결함이다. SimToolReal 도 잠그지 않고 ±2° 를
    남겼다(`iiwa14_left_sharpa_adjusted_restricted.urdf`: AA 를 ±20°→±2° 로 줄였을 뿐 0 이 아니다).
    """
    h = _h3()
    w = (h._act_hi - h._act_lo)
    assert float(w.min()) > 1e-6, f"죽은 액션 칸이 있다: {w.tolist()}"


def test_two_span_tiers_are_applied():
    """좁은 반폭(손가락)과 넓은 반폭(엄지)이 따로 걸린다."""
    h = _h3()
    assert h._act_lo[0].item() == pytest.approx(-0.035) and h._act_hi[0].item() == pytest.approx(0.035)
    assert h._act_lo[1].item() == pytest.approx(-0.035) and h._act_hi[1].item() == pytest.approx(0.035)
    assert h._act_lo[2].item() == pytest.approx(-0.35) and h._act_hi[2].item() == pytest.approx(0.35)
    # 굴곡은 여전히 open→grip
    assert h._act_lo[3].item() == pytest.approx(0.0) and h._act_hi[3].item() == pytest.approx(1.8)


def test_span_is_clipped_by_the_hardware_limit():
    """반폭이 관절 한계를 넘으면 한계가 이긴다 — 지령이 물리적으로 불가능한 곳을 못 가리킨다."""
    h = _Hand(mode="per_role", open_pose=[0.0], grip_pose=[0.0], wide=("j0",), wide_span=99.0)
    assert h._act_lo[0].item() >= float(h._syn_lo[0]) - 1e-6
    assert h._act_hi[0].item() <= float(h._syn_hi[0]) + 1e-6


def test_freed_abduction_cannot_reach_the_crossing_angle():
    """★실측 근거: 인접 손끝이 만나는 각 0.094 rad(5.4°) · 콜라이더가 닿는 각 0.024 rad(1.4°).

    자기충돌이 꺼져 있어(08.29 다물체 대책) 물리가 교차를 못 막는다 — 각도가 유일한 방어다.
    좁은 반폭은 교차각보다 **작아야** 한다. SimToolReal 도 같은 논리로 ±0.035(그들 교차각
    5.6~11.2°의 1/3)를 골랐고, 그들 학습 환경(IsaacGym)에서는 비트마스크 전이 오염 때문에
    근위·중위 마디끼리 실제로 충돌하지 않아 이 절단이 **유일한** 방어였다.
    """
    from pathlib import Path as _P
    cfg_src = (_P(__file__).resolve().parent.parent / "grasp_fj_env_cfg.py").read_text(encoding="utf-8")
    import re as _re
    span = float(_re.search(r"hand_shape_span_rad: float = ([0-9.]+)", cfg_src).group(1))
    assert span < 0.094, f"손끝 교차각(0.094 rad)보다 크다: {span}"
    # 검증기도 같은 상한을 강제해야 한다(hydra 오버라이드 방어).
    assert "0.08" in cfg_src and "hand_shape_span_rad" in cfg_src


def test_flexion_never_opens_the_back_of_hand():
    """★테솔로 `_3`/`_4` 는 [-1.571, +1.571] **대칭**이다 — a=−1 이 손등 −90° 를 지령한다.

    SHARPA 는 URDF 에서 PIP/DIP 하한이 **0.000**, MCP_FE 는 −10° 뿐이라 전 범위 [-1,1]
    매핑이 안전했다. 우리는 그 매핑을 그대로 쓰면 안 되고, 굴곡의 하한은 보정된 open 자세다.
    """
    h = _h3()
    flex = ~h._shape_mask
    assert bool(flex.any())
    for i in range(len(h._act_lo)):
        if bool(flex[i]):
            assert h._act_lo[i].item() == pytest.approx(_T3_OPEN[i]), "굴곡 하한이 open 을 벗어났다"
    # ★반폭을 아무리 키워도 굴곡은 안 넓어진다 — freed 가 "폭 0" 에서만 참이라 구조적으로 막힌다.
    big = _Hand(mode="per_role", open_pose=_T3_OPEN, grip_pose=_T3_GRIP,
                wide=("j2",), span=1.5, wide_span=1.5)
    fl = ~big._shape_mask
    for i in range(len(big._act_lo)):
        if bool(fl[i]):
            assert big._act_lo[i].item() == pytest.approx(_T3_OPEN[i])
            assert big._act_hi[i].item() == pytest.approx(_T3_GRIP[i])
    # 그래도 보루는 코드에 남아 있어야 한다 — 미래 편집이 이 성질을 깨면 부팅에서 죽는다.
    import ast as _ast, textwrap as _tw
    node = next(n for n in _ast.walk(_ast.parse(_ENV_SRC))
                if isinstance(n, _ast.FunctionDef) and n.name == "_build_hand_action_range")
    blk = _tw.dedent("\n".join(_ENV_SRC.split("\n")[node.lineno - 1:node.end_lineno]))
    assert "_flex = ~freed" in blk and "손등 과신전" in blk and "raise RuntimeError" in blk

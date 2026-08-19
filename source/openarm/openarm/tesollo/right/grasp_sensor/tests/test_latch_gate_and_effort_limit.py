"""latch 진입 게이트(A3) + 손 effort_limit(A4) 계약 (2026-08-19).

배경:
  A3 — latch 는 비가역 + approach/grasp shaping 차단 스위치. 구 "아무 3지" 진입은
       엄지 없는 얕은 latch 를 허용해 shaping 을 조기·영구 소멸시켰다
       (Stage1 실측 wrap_at_latch 0.03~0.05 고착). 신: 4지 + 엄지 접촉 AND.
  A4 — effort_limit 미설정(URDF 7.5N·m) + squeeze 비용 0 → 전 손관절 3~5N·m 상시,
       thumb_1 이 하드스톱을 넘어 -0.94rad 까지 밀림(실측 viol=1.00). 실기 연속토크
       ~1.5N·m 정합으로 과압착 레짐을 물리적으로 제거한다.

cfg 는 isaaclab→pxr 를 끌어와 Isaac 앱 없이 import 가 안 되므로 ast/문자열 계약으로 잡는다.
compute_lift_readiness 는 순수 torch 라 직접 실행 검증한다.
"""
import ast
import re
from pathlib import Path

import torch

from openarm.tesollo.right.grasp_sensor.grasp_right_utils import compute_lift_readiness

_BASE = Path(__file__).resolve().parents[1]
_CFG_SRC = (_BASE / "grasp_right_env_cfg.py").read_text()
_ENV_SRC = (_BASE / "grasp_right_env.py").read_text()


def _cfg_literals():
    tree = ast.parse(_CFG_SRC)
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "GraspRightEnvCfg")
    out = {}
    for node in cls.body:
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            try:
                out[node.target.id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    return out


# ---------------------------------------------------------------------------
# A3: latch 진입 게이트
# ---------------------------------------------------------------------------

def test_latch_threshold_matches_success_threshold():
    """latch 시점 = success 그립 요건 충족 시점 (얕은 latch 차단의 핵심)."""
    c = _cfg_literals()
    assert c["lift_start_min_grip_fingers"] == c["success_min_grip_fingers"] == 4


def test_latch_requires_thumb_contact_in_env():
    """env 가 엄지 접촉(finger 0)을 required_contact 로 전달해야 한다."""
    assert re.search(
        r"required_contact\s*=\s*_any_grip_contact\[:,\s*0\]", _ENV_SRC
    ), "latch 호출에 엄지 required_contact 가 없다 (A3 회귀)"


def test_compute_lift_readiness_blocks_thumbless_latch():
    """4지 count 를 채워도 엄지가 없으면 latch 하지 않는다 (직접 실행)."""
    n = 4
    hold = torch.zeros(n)
    latched = torch.zeros(n, dtype=torch.bool)
    contacts = torch.tensor([4, 4, 5, 3])            # count
    thumb = torch.tensor([False, True, True, True])  # 엄지 접촉
    for _ in range(8):                               # hold_steps 만큼 유지
        hold, _, latched = compute_lift_readiness(
            num_contacts=contacts,
            is_grasp_phase=~latched,
            previous_hold_count=hold,
            previous_latched=latched,
            min_contacts=4,
            hold_steps=8,
            required_contact=thumb,
        )
    assert latched.tolist() == [False, True, True, False], latched.tolist()


def test_latch_is_still_irreversible_once_earned():
    """정당하게 latch 된 뒤 접촉이 흔들려도 래치는 유지된다 (기존 계약 보존)."""
    hold = torch.tensor([8.0])
    latched = torch.tensor([True])
    hold, _, latched = compute_lift_readiness(
        num_contacts=torch.tensor([0]),
        is_grasp_phase=~latched,
        previous_hold_count=hold,
        previous_latched=latched,
        min_contacts=4,
        hold_steps=8,
        required_contact=torch.tensor([False]),
    )
    assert bool(latched[0])


# ---------------------------------------------------------------------------
# P1: latch 절벽 제거 — grasp shaping 은 latch 후에도 유지
# ---------------------------------------------------------------------------

def test_reward_has_no_latch_gate() -> None:
    """2026-08-20 재설계: reward 에서 latch 게이트를 **전부** 제거했다.

    3연속 fresh 실패(2지 국소최적 / 엄지 회피 latch 차단 / 손끝 파지)가 모두
    "latch 절벽 + 곱셈 게이트" 라는 같은 뿌리였다. latch 는 이제 제어 트리거(수직 램프)
    ·ADR·로깅 전용이고, reward 는 물리 상태(거리/접촉/높이/기울기)만 본다.
    레퍼런스: grasp_v2/right 가 같은 실패(3271 epoch 성공 0) 후 같은 결론에 도달했다.
    """
    core = (_BASE / "grasp_reward.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in core.split("\n") if not l.lstrip().startswith("#"))
    for gate in ("lift_gate", "pre_lift_gate", "lifted_gate", "lift_latched"):
        assert gate not in code, f"reward 에 latch 게이트가 되살아났다: {gate}"


def test_mcp_frozen_after_latch():
    """래치 후 MCP 동결 — 리프트 중 손가락을 더 말아 인벨롭을 깨는 것을 막는다.

    실측(test4 ep6500): 래치 시점 dst 1.52 로 인벨롭 성립 → 리프트 20~40스텝에
    MCP 0.767→0.843 조임 → dst 0.71 붕괴, 손끝만 남음(tip 4.54→4.96).
    PIP/DIP 는 접촉 동결로 잠겨 있어 MCP 만 가능한 경로였다.
    래치 **전** 무게이트는 유지해야 한다(근위 마디 밀착 = 감쌈 생성 메커니즘).
    """
    assert "g2 = torch.zeros_like(tip_c)" not in _ENV_SRC, "MCP 가 다시 무상시 무게이트다"
    assert re.search(
        r"g2\s*=\s*_any_c\s*\*\s*self\.lift_ready_latched_buf", _ENV_SRC
    ), "MCP 래치-후 동결이 없다"
    # 래치 전에는 걸리지 않아야 한다 → latched 곱이 반드시 있어야 함(위 assert 가 보장)
    assert "_any_c = (mid_c + dist_c + tip_c).clamp(max=1.0)" in _ENV_SRC


# ---------------------------------------------------------------------------
# A4: 손 effort_limit
# ---------------------------------------------------------------------------

def test_hand_actuators_have_realistic_effort_limit():
    """4개 손 액추에이터 그룹 전부 effort_limit_sim=1.5 (실기 연속토크 정합).

    미설정이면 URDF 7.5N·m 로 돌아가 과압착 레짐이 부활한다.
    """
    for group in ("_1", "_2", "_3", "_4"):
        block = re.search(
            r'joint_names_expr=\["r_hj_\[a-z\]\+' + group + r'"\].*?\)',
            _CFG_SRC, re.S,
        )
        assert block, f"손 그룹 {group} 액추에이터 블록을 못 찾음"
        assert "effort_limit_sim=1.5" in block.group(0), (
            f"손 그룹 {group} 에 effort_limit_sim=1.5 가 없다 (A4 회귀)"
        )

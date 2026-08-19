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

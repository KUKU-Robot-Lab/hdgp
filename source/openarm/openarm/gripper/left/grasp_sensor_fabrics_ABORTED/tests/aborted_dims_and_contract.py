"""차원·reward 계약 고정 (Isaac 불필요).

right/grasp_sensor 는 constants 주석의 산술(146/183)이 실제 값(154/191)과 어긋난 채 남아
있다 — 주석은 코드가 아니라 검증되지 않기 때문이다. 여기서는 **테스트가 차원을 고정**한다.

cfg 는 isaaclab→pxr 를 끌어와 Isaac 앱 없이는 import 가 안 된다. 그래서 dataclass 필드
기본값을 ast 로 직접 읽는다(우측 tests 와 동일 관례).
"""

import ast
import inspect
from pathlib import Path

import pytest

from openarm.common.grasp_v2_contract import GRASP_V2_REWARD_TERMS
from openarm.gripper.left.grasp_sensor_fabrics_ABORTED import grasp_left_constants as C
from openarm.gripper.left.grasp_sensor_fabrics_ABORTED import grasp_reward

_CFG_SRC = Path(__file__).resolve().parents[1] / "grasp_left_env_cfg.py"


def _cfg_literals() -> dict:
    tree = ast.parse(_CFG_SRC.read_text(encoding="utf-8"))
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "GraspLeftGripperEnvCfg")
    out = {}
    for node in cls.body:
        if not isinstance(node, ast.AnnAssign) or node.value is None:
            continue
        try:
            out[node.target.id] = ast.literal_eval(node.value)
        except ValueError:
            pass
    return out


# ---------------------------------------------------------------------------
# 차원
# ---------------------------------------------------------------------------
def test_action_is_six_dof_pose_plus_one_gripper():
    assert C.NUM_PALM_ACTION == 6
    assert C.NUM_GRIPPER_ACTION == 1
    assert C.NUM_ACTIONS == 7


def test_observation_dims_match_term_breakdown():
    # 문서화된 항목 합과 상수가 일치해야 한다 (주석 drift 차단)
    assert C.NUM_OBSERVATIONS == sum(C._OBS_TERMS)
    assert C.NUM_CRITIC_EXTRAS == sum(C._CRITIC_EXTRA_TERMS)
    assert C.NUM_CRITIC_OBSERVATIONS == C.NUM_OBSERVATIONS + C.NUM_CRITIC_EXTRAS


def test_observation_dims_are_stable_values():
    # 값이 바뀌면 학습된 체크포인트가 전부 무효가 된다 — 의도적 변경일 때만 이 테스트를 고칠 것.
    assert C.NUM_OBSERVATIONS == 48
    assert C.NUM_CRITIC_OBSERVATIONS == 62


def test_gripper_has_two_joints_but_one_dof():
    """gripper_2 는 USD PhysX mimic 이라 지령 자유도는 1 이다."""
    assert C.NUM_GRIPPER_JOINTS == 2
    assert C.NUM_GRIPPER_DOF == 1


def test_episode_structure():
    assert C.EPISODE_STEPS == C.GRASP_PHASE_STEPS + C.LIFT_PHASE_STEPS
    assert C.EPISODE_STEPS == 600


# ---------------------------------------------------------------------------
# reward 계약
# ---------------------------------------------------------------------------
def test_reward_returns_full_v2_term_contract():
    """8-term 계약을 지켜야 기존 TFEvents 파싱·분석 도구가 그대로 붙는다."""
    src = inspect.getsource(grasp_reward.compute_gripper_grasp_reward_terms)
    for term in GRASP_V2_REWARD_TERMS:
        assert f'"{term}"' in src, f"reward term 누락: {term}"


def test_cfg_defines_every_referenced_reward_weight():
    cfg = _cfg_literals()
    required = [
        "approach_weight", "approach_sharpness", "approach_xy_penalty_weight",
        "grasp_xy_threshold", "approach_tilt_penalty_weight", "grasp_upright_threshold_deg",
        "grasp_weight", "grasp_opposition_credit", "grasp_squeeze_credit",
        "lift_reward_weight", "lift_success_height", "lift_height_ref",
        "stabilize_weight", "stabilize_action_sharpness", "stability_reward_weight",
        "success_bonus_weight", "post_lift_contact_loss_weight", "action_smooth_weight",
        "cup_xy_disp_limit",
    ]
    missing = [k for k in required if k not in cfg]
    assert not missing, f"cfg 에 reward 파라미터 누락: {missing}"


def test_grasp_quality_credits_sum_within_one():
    """가중합이 1.0 을 넘으면 grasp 최대치가 grasp_weight 를 초과해 항목 비율이 깨진다."""
    cfg = _cfg_literals()
    total = cfg["grasp_opposition_credit"] + cfg["grasp_squeeze_credit"]
    assert 0.0 <= total <= 1.0


def test_latch_and_success_gates_require_both_fingers():
    """게이트 완화 금지 — 1지 접촉 래치는 부실 파지 국소최적을 만든다."""
    cfg = _cfg_literals()
    assert cfg["lift_start_min_contacts"] == 2
    assert cfg["success_min_contacts"] == 2
    assert cfg["grasp_ready_hold_steps"] >= 1


def test_post_lift_contact_loss_is_a_penalty():
    cfg = _cfg_literals()
    assert cfg["post_lift_contact_loss_weight"] < 0.0
    assert cfg["action_smooth_weight"] < 0.0


@pytest.mark.parametrize("name", ["lift_reward_weight", "success_bonus_weight"])
def test_grasp_does_not_dominate_lift_or_success(name):
    """reward-audit Check 1: grasp 만으로 수렴하는 국소최적을 만들지 않는다."""
    cfg = _cfg_literals()
    assert cfg["grasp_weight"] < cfg[name] * 3.0

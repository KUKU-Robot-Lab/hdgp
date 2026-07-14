"""distillation obs 계약: student 차원이 teacher에서 물체 privileged state만 뺀 값인가.

grasp_right_env.py 전체는 isaaclab/fabrics 없이 import 할 수 없으므로,
차원 계약은 constants + env 소스의 obs concat 목록을 직접 대조해 검증한다.
"""

from __future__ import annotations

import ast
from pathlib import Path

_TASK_DIR = Path(__file__).resolve().parents[1]
_ENV_SRC = _TASK_DIR / "grasp_right_env.py"
_CONSTANTS_SRC = _TASK_DIR / "grasp_right_constants.py"


def _int_constants() -> dict[str, int]:
    """constants 모듈은 preset 을 상대 import 해서 단독 로드가 안 된다 → AST 로 읽는다."""
    tree = ast.parse(_CONSTANTS_SRC.read_text())
    values: dict[str, int] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, int):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                values[target.id] = node.value.value
    return values


C = _int_constants()

# teacher policy obs 중 물체 privileged 항목 (student 는 이걸 못 본다)
_OBJECT_PRIVILEGED_DIMS = {
    "object_pos_noisy": 3,
    "object_rot_noisy": 4,
    "object_scale": 1,
    # onehot(N_obj)은 차원이 물체 수에 따라 변해 NUM_OBS_BASE 에 안 들어있다
}


def _obs_terms(func_name: str) -> list[str]:
    """env 소스에서 지정 함수 안의 torch.cat 항목 이름들을 뽑는다."""
    tree = ast.parse(_ENV_SRC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for call in ast.walk(node):
                is_cat = (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "cat"
                )
                if is_cat and isinstance(call.args[0], ast.List):
                    return [
                        elt.attr
                        for elt in call.args[0].elts
                        if isinstance(elt, ast.Attribute)
                    ]
    raise AssertionError(f"{func_name} 안에서 torch.cat 을 못 찾았다")


def test_student_obs_is_teacher_minus_object_state():
    # teacher(193) 에서 물체 privileged 8D 를 빼면 student(185)
    privileged_total = sum(_OBJECT_PRIVILEGED_DIMS.values())

    assert C["NUM_STUDENT_OBS"] == C["NUM_OBS_BASE"] - privileged_total


def test_student_obs_omits_every_object_privileged_term():
    student_terms = _obs_terms("compute_student_policy_observations")

    for privileged in (*_OBJECT_PRIVILEGED_DIMS, "multi_object_idx_onehot"):
        assert privileged not in student_terms, (
            f"student obs 에 물체 privileged 항목 '{privileged}' 이 남아있다 — "
            "vision student 가 물체 정보를 공짜로 받게 된다"
        )


def test_student_obs_keeps_goal_and_proprio():
    # object_goal 은 고정 절대점이라 실기에서도 알 수 있다 → student 에 남긴다
    student_terms = _obs_terms("compute_student_policy_observations")

    for kept in (
        "robot_dof_pos_noisy", "robot_dof_vel_noisy",
        "hand_pos_noisy", "hand_vel_noisy",
        "object_goal", "actions",
        "fabric_q", "fabric_qd", "fabric_qdd",
    ):
        assert kept in student_terms


def test_teacher_obs_dim_matches_action_dim():
    """teacher obs 는 actions 를 그대로 싣는다 — action 차원이 바뀌면 함께 움직인다.

    193/247/185 는 action 11D 시절 값. abduction 5축 자유화로 16D 가 되면서 각각 +5.
    07.14 접촉력 obs 추가: actor/student +15(tip xyz), critic +25(tip15+distal5+middle5).
    distillation 이식이 (action 변경 없이) 이 값을 건드리면 여기서 잡힌다.
    """
    assert C["NUM_OBS_BASE"] == 193 + 5 + 15
    assert C["NUM_CRITIC_OBS_BASE"] == 247 + 5 + 25
    assert C["NUM_STUDENT_OBS"] == 185 + 5 + 15

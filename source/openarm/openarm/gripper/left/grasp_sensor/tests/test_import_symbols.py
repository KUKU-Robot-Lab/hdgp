"""env 모듈이 참조하는 이름이 실제로 존재하는지 정적 검증 (Isaac 불필요).

왜 필요한가
-----------
`openarm/tasks/__init__.py` 의 자동 등록은 `except (ModuleNotFoundError, ImportError): pass`
로 감싸여 있어 **임포트 에러가 조용히 삼켜진다**. 오타 하나로 태스크가 등록되지 않고,
증상은 "task not found" 뿐이라 원인을 추적하기 어렵다.

env 모듈 자체는 isaaclab/fabrics_sim 을 끌어와 여기서 import 할 수 없으므로,
소스를 ast 로 읽어 **from-import 한 이름이 대상 모듈에 실제로 있는지** 확인한다.
"""

import ast
from pathlib import Path

import pytest

from openarm.gripper.left.grasp_sensor import grasp_left_constants, grasp_left_preset, grasp_reward

_PKG = Path(__file__).resolve().parents[1]
_LOCAL_MODULES = {
    "grasp_left_constants": grasp_left_constants,
    "grasp_left_preset": grasp_left_preset,
    "grasp_reward": grasp_reward,
}
_FABRIC_SRC = (
    _PKG.parents[4] / "FABRICS/src/fabrics_sim/fabrics/openarm_tesollo_pose_fabric.py"
)


def _relative_imports(source: Path) -> list[tuple[str, str]]:
    """(모듈명, 심볼) 목록 — `from .mod import a, b` 형태만."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            for alias in node.names:
                out.append((node.module, alias.name))
    return out


@pytest.mark.parametrize("src_name", ["grasp_left_env.py", "grasp_left_env_cfg.py"])
def test_relative_imports_resolve(src_name):
    missing = []
    for module, symbol in _relative_imports(_PKG / src_name):
        mod = _LOCAL_MODULES.get(module)
        if mod is None:
            continue  # 이 테스트가 import 할 수 없는 모듈(cfg 등)은 건너뛴다
        if not hasattr(mod, symbol):
            missing.append(f"{module}.{symbol}")
    assert not missing, f"{src_name} 이 참조하는 이름이 없다: {missing}"


def test_env_imports_the_gripper_fabric_class():
    """`OpenArmGripperLeftPoseFabric` 이 fabric 모듈에 실제로 정의돼 있어야 한다."""
    env_src = (_PKG / "grasp_left_env.py").read_text(encoding="utf-8")
    assert "OpenArmGripperLeftPoseFabric" in env_src

    tree = ast.parse(_FABRIC_SRC.read_text(encoding="utf-8"))
    classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    assert "OpenArmGripperLeftPoseFabric" in classes
    # 기존 소비자(pour/grasp_v1/grasp_sensor)가 쓰는 클래스도 그대로 남아 있어야 한다
    assert {"OpenArmTeoslloPoseFabric", "OpenArmTeoslloLeftPoseFabric"} <= classes


def test_env_calls_the_two_finger_reward():
    env_src = (_PKG / "grasp_left_env.py").read_text(encoding="utf-8")
    assert "compute_gripper_grasp_reward_terms" in env_src
    assert hasattr(grasp_reward, "compute_gripper_grasp_reward_terms")


def test_gripper_target_goes_to_one_joint_only():
    """gripper_2 는 USD PhysX mimic — 둘 다 지령하면 mimic 제약과 드라이브가 싸운다."""
    env_src = (_PKG / "grasp_left_env.py").read_text(encoding="utf-8")
    assert "self.gripper_cmd_index" in env_src
    assert "joint_ids=[self.gripper_cmd_index]" in env_src

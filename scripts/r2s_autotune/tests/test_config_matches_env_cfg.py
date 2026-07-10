"""autotune config의 actuator group이 RL env_cfg와 일치하는지 대조한다.

group 이름이 어긋나면 get_actuator_params()가 조용히 default로 fallback한다.
에러가 나지 않으므로 이 테스트가 없으면 calibration이 적용되지 않아도 알 수 없다.
"""

import ast
from pathlib import Path

import pytest

from r2s_autotune.config import load_config
from r2s_autotune.paths import HDGP_ROOT

RH56F1_ENV_CFG = (
    HDGP_ROOT / "source/openarm/openarm/rh56f1/right/grasp_v1/grasp_right_env_cfg.py"
)
RH56F1_AUTOTUNE_CFG = Path(__file__).resolve().parents[1] / "configs" / "bi_rh56f1.yaml"


def _literal(node: ast.AST) -> object | None:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def _extract_actuator_groups(source_path: Path) -> dict[str, dict]:
    """env_cfg의 actuators={...} 블록에서 group 이름, regex, 기본 gain을 뽑는다.

    두 형태를 모두 다룬다:
        **_actuator_params("name", 400.0, 80.0)
        stiffness=400.0, damping=60.0
    """
    tree = ast.parse(source_path.read_text())
    groups: dict[str, dict] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            name = _literal(key) if key is not None else None
            if not isinstance(name, str) or not isinstance(value, ast.Call):
                continue
            func = value.func
            if not (isinstance(func, ast.Name) and func.id == "ImplicitActuatorCfg"):
                continue

            entry: dict = {"joint_names_expr": None, "stiffness": None, "damping": None}
            for keyword in value.keywords:
                if keyword.arg == "joint_names_expr":
                    entry["joint_names_expr"] = _literal(keyword.value)
                elif keyword.arg in ("stiffness", "damping"):
                    entry[keyword.arg] = _literal(keyword.value)
                elif keyword.arg is None and isinstance(keyword.value, ast.Call):
                    # **_actuator_params("group", stiffness, damping)
                    args = [_literal(a) for a in keyword.value.args]
                    if len(args) >= 3:
                        entry["stiffness"], entry["damping"] = args[1], args[2]
            groups[name] = entry

    return groups


@pytest.fixture(scope="module")
def env_cfg_groups() -> dict[str, dict]:
    if not RH56F1_ENV_CFG.is_file():
        pytest.skip(f"env_cfg not found: {RH56F1_ENV_CFG}")
    groups = _extract_actuator_groups(RH56F1_ENV_CFG)
    if not groups:
        pytest.skip("no ImplicitActuatorCfg groups parsed from env_cfg")
    return groups


def test_group_names_match_env_cfg(env_cfg_groups):
    config = load_config(RH56F1_AUTOTUNE_CFG)

    assert set(config.groups) == set(env_cfg_groups)


def test_joint_expressions_match_env_cfg(env_cfg_groups):
    config = load_config(RH56F1_AUTOTUNE_CFG)

    for name, group in config.groups.items():
        expected = env_cfg_groups[name]["joint_names_expr"]
        assert list(group.joint_names_expr) == list(expected), f"regex drift in group '{name}'"


def test_default_gains_match_env_cfg(env_cfg_groups):
    config = load_config(RH56F1_AUTOTUNE_CFG)

    for name, group in config.groups.items():
        entry = env_cfg_groups[name]
        assert group.stiffness == entry["stiffness"], f"stiffness drift in group '{name}'"
        assert group.damping == entry["damping"], f"damping drift in group '{name}'"


def test_configs_cover_all_movable_joints(config_path):
    # load_config가 coverage 위반 시 JointContractError를 던진다.
    config = load_config(config_path)

    claimed = sum(len(g.joint_names_expr) for g in config.groups.values())
    assert claimed > 0
    assert config.tune_groups
    assert all(config.groups[name].is_tunable for name in config.tune_groups)

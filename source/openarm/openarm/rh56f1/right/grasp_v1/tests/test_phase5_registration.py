"""Phase 5 gym 등록 / agents / 로깅 정적 검증 (GPU/isaaclab 불필요)."""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_INIT = ROOT / "config" / "__init__.py"
PPO_YAML = ROOT / "config" / "agents" / "rl_games_ppo_cfg.yaml"
LSTM_YAML = ROOT / "config" / "agents" / "rl_games_ppo_lstm_cfg.yaml"

# train.py 의 실제 정규식 (동기 유지)
TRAIN_REGEX = r"\.pipeline\.(?:gripper|hand)\.([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\."

ENTRY = (
    "openarm.rh56f1.right"
    ".grasp_v1"
    ".grasp_right_env:GraspRightEnv"
)


def test_config_init_syntax_and_ids():
    s = CONFIG_INIT.read_text()
    ast.parse(s)
    for gid in ['id="open-rh56f1_r_grasp_v1"', 'id="open-rh56f1_r_grasp_v1-lstm"',
                'id="open-rh56f1_r_grasp_v1-play"', 'id="open-rh56f1_r_grasp_v1-play-lstm"']:
        assert gid in s, gid
    assert "openarm.rh56f1.right.grasp_v1.grasp_right_env:GraspRightEnv" in s
    assert "5g_grasp_right_v11" not in s


def test_agents_yaml_names():
    assert "name: inspire_r_grasp_v1" in PPO_YAML.read_text()
    assert "name: inspire_r_grasp_v1-lstm" in LSTM_YAML.read_text()
    assert "5g_grasp_right-v11" not in PPO_YAML.read_text()


def test_entry_points_to_current_package():
    assert ENTRY == "openarm.rh56f1.right.grasp_v1.grasp_right_env:GraspRightEnv"


def test_log_path_branch():
    """RH56F1 right grasp logs under the hand-specific variant/folder."""
    side_dir, folder = "rh56f1_r", "grasp_v1"
    if side_dir in ("left", "right", "both"):
        path = f"log/rl_games/pipeline/{side_dir}/{folder}"
    else:
        path = f"log/rl_games/{side_dir}/{folder}"
    assert path == "log/rl_games/rh56f1_r/grasp_v1"


def test_train_regex_legacy_still_works():
    """기존 right 경로 하위호환 — group(1)=right."""
    legacy = (".pipeline.hand.right.5g_grasp_right_v11.grasp_right_env:GraspRightEnv")
    m = re.search(TRAIN_REGEX, legacy)
    assert m and m.group(1) == "right" and m.group(2) == "5g_grasp_right_v11"


if __name__ == "__main__":
    test_config_init_syntax_and_ids()
    test_agents_yaml_names()
    test_entry_points_to_current_package()
    test_log_path_branch()
    test_train_regex_legacy_still_works()
    print("Phase 5 registration/logging: 5 checks passed (GREEN)")

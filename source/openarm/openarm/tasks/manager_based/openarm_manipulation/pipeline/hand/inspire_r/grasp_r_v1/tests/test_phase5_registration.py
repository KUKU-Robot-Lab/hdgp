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
    "openarm.tasks.manager_based.openarm_manipulation"
    ".pipeline.hand.inspire_r.grasp_r_v1"
    ".grasp_right_env:GraspRightEnv"
)


def test_config_init_syntax_and_ids():
    s = CONFIG_INIT.read_text()
    ast.parse(s)
    for gid in ['id="inspire_r_grasp_v1"', 'id="inspire_r_grasp_v1-lstm"',
                'id="inspire_r_grasp_v1-play"', 'id="inspire_r_grasp_v1-play-lstm"']:
        assert gid in s, gid
    # entry_point 이 현재 경로(inspire_r.grasp_r_v1)를 가리켜야 (옛 right.5g_grasp_right_v11 아님)
    assert "pipeline.hand.inspire_r.grasp_r_v1" in s
    assert "5g_grasp_right_v11" not in s
    assert "right.5g_grasp" not in s


def test_agents_yaml_names():
    assert "name: inspire_r_grasp_v1" in PPO_YAML.read_text()
    assert "name: inspire_r_grasp_v1-lstm" in LSTM_YAML.read_text()
    assert "5g_grasp_right-v11" not in PPO_YAML.read_text()


def test_train_regex_resolves_inspire_r():
    """train.py 정규식이 entry_point 에서 variant=inspire_r, folder=grasp_r_v1 추출."""
    m = re.search(TRAIN_REGEX, ENTRY)
    assert m is not None
    assert m.group(1) == "inspire_r"
    assert m.group(2) == "grasp_r_v1"


def test_log_path_branch():
    """명명형 variant 는 pipeline 접두 없이 log/rl_games/<variant>/<folder>."""
    side_dir, folder = "inspire_r", "grasp_r_v1"
    if side_dir in ("left", "right", "both"):
        path = f"log/rl_games/pipeline/{side_dir}/{folder}"
    else:
        path = f"log/rl_games/{side_dir}/{folder}"
    assert path == "log/rl_games/inspire_r/grasp_r_v1"


def test_train_regex_legacy_still_works():
    """기존 right 경로 하위호환 — group(1)=right."""
    legacy = (".pipeline.hand.right.5g_grasp_right_v11.grasp_right_env:GraspRightEnv")
    m = re.search(TRAIN_REGEX, legacy)
    assert m and m.group(1) == "right" and m.group(2) == "5g_grasp_right_v11"


if __name__ == "__main__":
    test_config_init_syntax_and_ids()
    test_agents_yaml_names()
    test_train_regex_resolves_inspire_r()
    test_log_path_branch()
    test_train_regex_legacy_still_works()
    print("Phase 5 registration/logging: 5 checks passed (GREEN)")

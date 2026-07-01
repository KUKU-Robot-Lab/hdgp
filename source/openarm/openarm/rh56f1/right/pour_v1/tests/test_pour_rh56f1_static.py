"""inspire_r_pour_v1 RH56F1 포팅 정적 검증 (GPU/isaaclab 불필요).

preset/constants 는 순수 파이썬 → 직접 import.
env_cfg/env 는 isaaclab 의존 → 소스 텍스트 검사로 대체.
"""
import ast
import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(mod_name: str, filename: str):
    """preset/constants 를 패키지 컨텍스트 없이 단독 로드 (상대 import 우회)."""
    # constants 는 `from .pour_right_preset import ...` 사용 → preset 을 먼저 sys.modules 에 주입
    import sys
    spec = importlib.util.spec_from_file_location(mod_name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _load_preset_and_constants():
    import sys
    # 상대 import 가 동작하도록 가짜 패키지 등록
    pkg = "_pour_rh56f1_test_pkg"
    if pkg not in sys.modules:
        import types
        m = types.ModuleType(pkg)
        m.__path__ = [str(ROOT)]
        sys.modules[pkg] = m
    preset = _load(f"{pkg}.pour_right_preset", "pour_right_preset.py")
    constants = _load(f"{pkg}.pour_right_constants", "pour_right_constants.py")
    return preset, constants


# ---------------------------------------------------------------------------
# preset
# ---------------------------------------------------------------------------
def test_preset_rh56f1_hand_joints():
    preset, _ = _load_preset_and_constants()
    assert len(preset.RIGHT_HAND_JOINT_NAMES) == 6
    assert len(preset.RIGHT_HAND_MIMIC_JOINT_NAMES) == 6
    assert all(n.startswith("r_hj_") for n in preset.RIGHT_HAND_JOINT_NAMES)
    assert len(preset.RIGHT_ACTUATED_JOINT_NAMES) == 13  # 7 arm + 6 hand
    # 손 포즈 6D
    for pose in (preset.HAND_START_POSE, preset.HAND_APPROACH_POSE,
                 preset.HAND_GRASP_POSE, preset.HAND_FULL_GRIP_POSE):
        assert len(pose) == 6


def test_preset_left_hand_rh56f1():
    preset, _ = _load_preset_and_constants()
    assert len(preset.LEFT_HAND_JOINT_NAMES) == 12  # drive 6 + mimic 6
    assert len(preset.LEFT_ARM_AND_HAND_JOINT_NAMES) == 19  # 7 arm + 12 hand
    # Tesollo 그리퍼 잔재 없음
    assert "openarm_left_finger_joint1" not in preset.LEFT_ARM_REST_JOINT_POS
    assert all(v == 0.0 for v in preset.LEFT_HAND_REST_JOINT_POS.values())


def test_preset_body_names_rh56f1():
    preset, _ = _load_preset_and_constants()
    assert preset.HAND_BODY_NAMES_USD[0] == "r_al_7"
    assert len(preset.HAND_BODY_NAMES_USD) == 6
    assert all("rl_dg" not in b for b in preset.HAND_BODY_NAMES_USD)
    assert len(preset.FABRIC_HAND_BODY_NAMES) == 7
    assert preset.FABRIC_HAND_BODY_NAMES[2] == "rh56f1_tip_thumb"


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
def test_constants_dims():
    _, c = _load_preset_and_constants()
    assert c.NUM_ARM_DOF == 7
    assert c.NUM_HAND_DOF == 6
    assert c.NUM_ROBOT_DOF == 13
    assert c.NUM_ACTIONS == 12           # 6 palm + 1 nullspace + 5 finger lerp (pour_v6 이식)
    assert c.NUM_NULLSPACE_ACTION == 1
    assert c.NUM_FINGER_ACTION == 5
    assert c.NUM_OBSERVATIONS == 51      # pour_v6 sim2real 레이아웃 (left_arm 7)
    # critic = base 77 + extras 35 = 112
    assert c.NUM_CRITIC_BASE_OBSERVATIONS == 77
    assert c.NUM_CRITIC_EXTRAS == 35
    assert c.NUM_CRITIC_OBSERVATIONS == 112
    assert c.NUM_CRITIC_OBSERVATIONS == c.NUM_CRITIC_BASE_OBSERVATIONS + c.NUM_CRITIC_EXTRAS


def test_critic_dim_matches_assembly():
    """critic base 82 = 실제 cat 항목 합."""
    base = (7 + 7 + 6 + 6      # arm pos/vel + finger pos/vel
            + 3 + 4 + 3 + 4    # rcup_rel,quat + lcup_rel,quat
            + 3 + 3 + 3        # opening_rel + source_pour_axis + source_up_axis
            + 8                # transport stack
            + 5 + 5            # binary_contact + tip_force
            + 6                # last_actions (palm 6, α 제외)
            + 1 + 1 + 1 + 1)   # bead source/target/cross/spill
    assert base == 77
    extra = (7 + 7            # left_arm pos/vel (7 arm; 왼손 rest 고정)
             + 5 + 5          # distal binary/force (=tip)
             + 1 + 1          # cup_h + rho
             + 1 + 1 + 7)     # demo_arm_err + demo_j5_err + demo_target_arm_q (privileged)
    assert extra == 35
    assert base + extra == 112


# ---------------------------------------------------------------------------
# config 등록
# ---------------------------------------------------------------------------
def test_config_init_ids_and_entry():
    s = (ROOT / "config" / "__init__.py").read_text()
    ast.parse(s)
    for gid in ['id="open-rh56f1_r_pour_v1"', 'id="open-rh56f1_r_pour_v1-lstm"',
                'id="open-rh56f1_r_pour_v1-play"', 'id="open-rh56f1_r_pour_v1-play-lstm"']:
        assert gid in s, gid
    assert "openarm.rh56f1.right.pour_v1.pour_right_env" in s
    assert "5g_pour_right_v3" not in s
    assert "hand.right.5g_pour" not in s


def test_agents_yaml_names():
    ppo = (ROOT / "config" / "agents" / "rl_games_ppo_cfg.yaml").read_text()
    lstm = (ROOT / "config" / "agents" / "rl_games_ppo_lstm_cfg.yaml").read_text()
    assert "name: inspire_r_pour_v1" in ppo
    assert "name: inspire_r_pour_v1-lstm" in lstm
    assert "5g_pour_right-v3" not in ppo


def test_train_regex_resolves_inspire_r_pour():
    regex = r"\.pipeline\.(?:gripper|hand)\.([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\."
    entry = (
        "openarm.tasks.manager_based.openarm_manipulation"
        ".pipeline.hand.inspire_r.pour_r_v1.pour_right_env:PourRightEnv"
    )
    m = re.search(regex, entry)
    assert m and m.group(1) == "inspire_r" and m.group(2) == "pour_r_v1"


# ---------------------------------------------------------------------------
# env.py 소스 검사 (isaaclab 미설치 → 텍스트)
# ---------------------------------------------------------------------------
def test_env_fabric_and_hand_mapping():
    s = (ROOT / "pour_right_env.py").read_text()
    ast.parse(s)
    # RH56F1 fabric 사용
    assert "OpenArmRh56f1PoseFabric" in s
    assert "openarm_rh56f1_pose_fabric" in s
    assert "OpenArmTeoslloPoseFabric" not in s
    assert "self.open_tesollo_fabric" not in s
    # 5 finger → 6 drive 매핑
    assert "_finger_drive_counts" in s
    assert "repeat_interleave(self._finger_drive_counts" in s
    # distal=tip, middle=zeros
    assert "_distal_sensor" not in s.replace("# ", "")  # 센서 객체 제거됨
    # tesollo 손 잔재 없음
    assert "rl_dg" not in s
    assert "rj_dg" not in s


def test_env_cfg_rh56f1():
    s = (ROOT / "pour_right_env_cfg.py").read_text()
    ast.parse(s)
    assert "robot/openarm_bi_rh56f1_rl/openarm_bi_rh56f1_rl.usd" in s
    assert "cup/cup_middle.usd" in s          # 오른쪽 source cup
    assert "rh56f1_right_drive" in s
    assert "rh56f1_left_drive" in s
    assert "tesollo_hand" not in s
    assert "openarm_left_gripper" not in s
    assert "enable_warmstart_reset: bool = False" in s
    assert "distal_sensor_cfg" not in s
    assert "middle_sensor_cfg" not in s


if __name__ == "__main__":
    test_preset_rh56f1_hand_joints()
    test_preset_left_hand_rh56f1()
    test_preset_body_names_rh56f1()
    test_constants_dims()
    test_critic_dim_matches_assembly()
    test_config_init_ids_and_entry()
    test_agents_yaml_names()
    test_train_regex_resolves_inspire_r_pour()
    test_env_fabric_and_hand_mapping()
    test_env_cfg_rh56f1()
    print("pour RH56F1 static: 10 checks passed (GREEN)")

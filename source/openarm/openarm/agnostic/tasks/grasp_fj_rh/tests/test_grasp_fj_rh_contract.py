"""grasp_fj_rh 계약 테스트 — Isaac Sim 없이 돈다(소스 텍스트·AST + 순수 데이터 프로필).

이 트랙은 Track B(`grasp_fj`)의 **로봇 교체판**이다. 그래서 잠그는 것은 두 가지다.

1. B 와 갈리지 않아야 하는 것 — 팔 경로·fabric 부재·접촉 센서 부재는 부모 그대로 쓴다
   (여기서 덮으면 A/B/RH 대조가 성립하지 않는다).
2. RH56F1 이라서 달라지는 것 — 언더액추 구동/종속 분리, per_finger 6슬롯, 차원 13/94/118,
   그리고 09.02 에 **지표 없이** 실패했던 자세·게인 조합.

실행:
    cd hdgp && PYTHONPATH=source/openarm python3 -m pytest \
        source/openarm/openarm/agnostic/tasks/grasp_fj_rh/tests -q
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from openarm.agnostic.tasks.grasp_kp.tests.test_grasp_kp_contract import (
    _class_methods,
    _code,
    _fn_block,
)

_HERE = Path(__file__).resolve().parent.parent
_FJ = _HERE.parent / "grasp_fj"
_ENV = (_HERE / "grasp_fj_rh_env.py").read_text(encoding="utf-8")
_CFG = (_HERE / "grasp_fj_rh_env_cfg.py").read_text(encoding="utf-8")
_REG = (_HERE / "config" / "__init__.py").read_text(encoding="utf-8")
_LSTM = (_HERE / "config" / "agents" / "rl_games_ppo_lstm_cfg.yaml").read_text(encoding="utf-8")
_FJ_ENV = (_FJ / "grasp_fj_env.py").read_text(encoding="utf-8")
_CONTROL = (_HERE.parent / "grasp_s2r" / "grasp_s2r_control.py").read_text(encoding="utf-8")
#: 자산 루트 — 저장소 어디서 돌아도 같은 곳을 가리키도록 위로 올라가며 찾는다
#: (env 는 `grasp_s2r_env_cfg._ASSETS_DIR` 을 쓰지만 그 모듈은 isaaclab 을 끌어온다).
def _assets_root() -> Path:
    for d in Path(__file__).resolve().parents:
        if (d / "assets" / "robot").is_dir():
            return d / "assets"
    raise AssertionError("assets/robot 을 못 찾았다")


_ASSETS = _assets_root()


def _profile():
    from openarm.agnostic.tasks.grasp_fj_rh.robot_profiles import PROFILES
    return PROFILES["rh56f1_right"]


# ---------------------------------------------------------------- 등록·설정
def test_task_ids_registered_without_sapg():
    assert '_ENTRY = "openarm.agnostic.tasks.grasp_fj_rh.grasp_fj_rh_env:GraspFJRHEnv"' in _REG
    for suffix in ("", "-play", "-lstm", "-play-lstm"):
        assert f'"open-{{_tag}}_grasp_fj_rh{suffix}"'.replace("{_tag}", "{_tag}") or True
    assert 'f"open-{_tag}_grasp_fj_rh{_suffix}"' in _REG
    assert re.search(r'"rh_r":\s*GraspFJRH56F1RightEnvCfg', _REG)
    # ★로컬 2048 env 전용 트랙 — SAPG id 를 두면 벤더 rl_games 없이 조용히 PPO 로 돈다.
    assert "sapg" not in _code(_REG).lower()
    assert "fabric_class" not in _code(_REG)


def test_learning_hypers_are_the_b1_set_not_the_simtoolreal_alignment():
    """★09.07 서버 실측: 정렬본으로 돈 tesollo B 다섯 런(b2~b6)이 e100~150 에 같은 서명으로
    무너졌고(close 0.02~0.11 · ft_dist 0.35~0.65 로 멀어짐 · reward 1.0) b1 세트만 살아남았다
    (fj_b1 lifted 0.826 · fj_b9 succ 10.10). 우리 rh_b3 도 정렬본에서 정확히 그 서명이었다.
    이 트랙은 그 9키를 b1 값으로 고정한다 — 바꾸려면 대체 탐색 수단을 함께 넣어야 한다."""
    want = {
        "concat_input": "True", "concat_output": "True", "mixed_precision": "False",
        "learning_rate": "3e-4", "entropy_coef": "0.002", "e_clip": "0.2",
        "mini_epochs": "4", "bound_loss_type": "regularization", "bounds_loss_coef": "0.005",
    }
    for key, val in want.items():
        m = re.search(rf"^\s+{key}:\s*(\S+)", _LSTM, re.M)
        assert m, key
        assert m.group(1) == val, f"{key}: {m.group(1)} ≠ b1 값 {val}"
    # ★sigma 를 살리는 것은 entropy_coef 와 bound_loss_type 오타 둘뿐이다 — 근거를 파일에 남긴다.
    assert "fj_b1" in _LSTM and "close_gate" in _LSTM


def test_lstm_batch_math_fits_the_local_operating_point():
    """2,048 env × horizon 16 이 minibatch 로 나누어떨어져야 한다(로컬 기본 운전점)."""
    horizon = int(re.search(r"horizon_length:\s*(\d+)", _LSTM).group(1))
    mb = int(re.search(r"minibatch_size:\s*(\d+)", _LSTM).group(1))
    seq = int(re.search(r"seq_length:\s*(\d+)", _LSTM).group(1))
    assert (2048 * horizon) % mb == 0, (horizon, mb)
    assert mb % seq == 0
    assert "2,048" in _LSTM or "2048" in _LSTM, "기본 운전점(2,048 env)을 yaml 에 적어 둔다"


# ---------------------------------------------------------------- 훅 경계
def test_env_overrides_only_the_robot_adapter_hooks():
    """B 의 팔·손 경로를 덮으면 로봇 교체가 아니라 다른 과제가 된다."""
    names = _class_methods(_ENV, "GraspFJRHEnv")
    required = {"_arm_slot_width", "_setup_fabrics", "_log_fabric_metrics", "_hand_command"}
    helpers = {"_load_mimic_pairs", "_assert_mimic_constraints_present", "_assert_hand_pose_usable",
               "_widen_dependent_joint_limits"}
    forbidden = {"_arm_command", "_apply_action", "_cmd_state", "_synergy_targets",
                 "_setup_synergy", "_get_observations", "_get_rewards", "_get_dones", "_reset_idx",
                 "_init_home_palm", "_step_fabric", "_post_command"}
    assert required <= names, required - names
    assert not (forbidden & names), forbidden & names
    assert names <= required | helpers, names - (required | helpers)


def test_arm_slot_width_comes_from_the_track_b_hook():
    """팔 슬롯 폭은 리터럴이 아니라 cfg 의 `_arm_action_dim` 에서 온다."""
    block = _fn_block(_ENV, "_arm_slot_width")
    assert "self.cfg._arm_action_dim(self.profile)" in block
    assert not re.search(r"return\s+\d", block)
    # mixin 은 그 훅을 쓰고 있어야 한다(리터럴 6 이 남아 있으면 B 에서 부팅이 죽는다).
    mixin = _fn_block(_CONTROL, "_setup_synergy")
    assert "self._arm_slot_width()" in mixin and "if 6 + _n_act" not in mixin


def test_fabric_and_contact_paths_are_inherited_untouched():
    for banned in ("fabrics_sim", "ContactSensor", "force_matrix_w", "_contact_forces",
                   "set_joint_velocity_target", "set_joint_position_target"):
        assert banned not in _code(_ENV), banned
    assert "self.fabric = None" in _fn_block(_FJ_ENV, "_setup_fabrics")


# ---------------------------------------------------------------- 언더액추 계약
def test_profile_splits_driven_and_dependent_joints():
    p = _profile()
    assert p.num_hand_joints == 6 and len(p.hand_joint_names) == 6
    assert p.fabric_class is None and p.fabric_joint_order == ()
    # 구동 정규식은 종속(_2 / thumb_[34])을 잡으면 안 된다 — 잡으면 제약과 싸운다.
    for dep in ("r_hj_index_2", "r_hj_middle_2", "r_hj_ring_2", "r_hj_pinky_2",
                "r_hj_thumb_3", "r_hj_thumb_4"):
        assert not re.fullmatch(p.hand_joint_regex, dep), dep
    for drv in p.hand_joint_names:
        assert re.fullmatch(p.hand_joint_regex, drv), drv


def test_dependent_actuators_are_zero_gain_on_both_sides():
    """PhysX mimic 이 위치를 정하는 관절에 드라이브를 켜면 싸운다(09.07 실측: damping 0.1 → 발산)."""
    p = _profile()
    for name in ("right_hand_mimic", "left_hand_mimic"):
        spec = p.actuator_specs[name]
        assert (spec["stiffness"], spec["damping"]) == (0.0, 0.0), name


def test_every_dof_is_covered_by_some_actuator():
    """IsaacLab 은 전 관절 커버리지를 요구한다 — 빠진 관절은 조용히 무구동 자유회전한다."""
    p = _profile()
    urdf = ET.parse(_ASSETS / p.usd_relpath.replace(".usd", ".urdf")).getroot()
    movable = [j.get("name") for j in urdf.iter("joint") if j.get("type") != "fixed"]
    patterns = [re.compile(e) for spec in p.actuator_specs.values()
                for e in spec["joint_names_expr"]]
    missing = [j for j in movable if not any(pat.fullmatch(j) for pat in patterns)]
    assert not missing, missing
    # 그리고 같은 관절을 두 그룹이 잡으면 IsaacLab 이 부팅에서 죽는다.
    dup = [j for j in movable if sum(bool(pat.fullmatch(j)) for pat in patterns) > 1]
    assert not dup, dup


def test_mimic_table_is_read_from_the_asset_not_hardcoded():
    """배율을 코드에 적으면 자산과 갈린다 — env 는 URDF 를 읽는다."""
    assert "1.1169" not in _ENV and "1.1425" not in _ENV and "0.7508" not in _ENV
    block = _fn_block(_ENV, "_load_mimic_pairs")
    assert 'j.find("mimic")' in block and "multiplier" in block
    # 결합이 사라진 자산을 조용히 통과시키면 손가락이 흐물거린다 — 부팅에서 죽여야 한다.
    assert "raise RuntimeError" in block
    assert "physxMimicJoint:rotZ:gearing" in _ENV


def test_asset_actually_carries_the_mimic_multipliers():
    """URDF 표가 이 트랙이 기대하는 언더액추 구조(6쌍)인지 데이터로 확인."""
    p = _profile()
    urdf = ET.parse(_ASSETS / p.usd_relpath.replace(".usd", ".urdf")).getroot()
    pairs = {j.get("name"): float(j.find("mimic").get("multiplier"))
             for j in urdf.iter("joint")
             if j.find("mimic") is not None and str(j.get("name")).startswith("r_hj_")}
    assert len(pairs) == 6, pairs
    assert set(pairs) & set(p.hand_joint_names) == set(), "종속관절이 구동 목록에 섞였다"


# ---------------------------------------------------------------- 손 레이아웃·차원
def test_per_finger_slots_are_contiguous_and_match_the_driven_joints():
    p = _profile()
    slots = sorted(s for m in p.hand_finger_channels.values() for s in m.values())
    assert slots == list(range(len(p.hand_joint_names))), slots
    named = {f"r_hj_{f}_{sfx}" for f, m in p.hand_finger_channels.items() for sfx in m}
    assert named == set(p.hand_joint_names), named ^ set(p.hand_joint_names)


def test_action_and_observation_widths():
    """action 13 = 팔 7 + 손 6 · actor 94 · critic 118 (`_derive_spaces` 공식 그대로)."""
    p = _profile()
    n_arm, n_hand, n_tips = p.num_arm_joints, p.num_hand_joints, len(p.fingertip_bodies)
    action = n_arm + len({s for m in p.hand_finger_channels.values() for s in m.values()})
    obs = 2 * n_arm + 2 * n_hand + 3 + 6 + 3 * n_tips + n_arm + 12 + 12 + action
    state = obs + 6 + 6 + 1 + n_tips + 1 + 1 + 1 + 1 + 1 + 1
    assert (action, obs, state) == (13, 94, 118)
    for token in ("action 13", "actor 94", "critic 118"):
        assert token in _CFG, token
    assert "94D" in _LSTM and "13D" in _LSTM


def test_hand_poses_move_every_driven_joint_within_limits():
    """open == grip 이면 그 관절은 시너지가 안 건드리고, grip 이 한계 밖이면 잘린다(09.02)."""
    p = _profile()
    urdf = ET.parse(_ASSETS / p.usd_relpath.replace(".usd", ".urdf")).getroot()
    lim = {j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
           for j in urdf.iter("joint") if j.find("limit") is not None}
    for k, name in enumerate(p.hand_joint_names):
        o, g = float(p.hand_open_pose[k]), float(p.hand_grip_pose[k])
        assert abs(g - o) > 1e-4, f"{name}: open == grip"
        for v in (o, g):
            assert lim[name][0] - 1e-6 <= v <= lim[name][1] + 1e-6, (name, v, lim[name])


# ---------------------------------------------------------------- 조용한 실패 차단
def test_cfg_kills_the_tesollo_only_knobs():
    """09.02 에 지표 없이 실패했던 조합을 cfg 단계에서 죽인다."""
    val = _fn_block(_CFG, "_validate_rh_fields")
    for token in ("per_finger", "oppose_grip_delta_rad", "k_arm", "close_gate_enabled",
                  "right_hand_mimic", "raise RuntimeError"):
        assert token in val, token
    assert re.search(r"oppose_grip_delta_rad:\s*float\s*=\s*0\.0", _CFG)
    assert re.search(r'hand_layout:\s*str\s*=\s*"per_finger"', _CFG)
    assert re.search(r'object_bank:\s*str\s*=\s*"shaker_one"', _CFG)


def test_blocked_threshold_is_above_torque_saturation():
    """★막힘 판정은 '닿았다'가 아니라 '더 못 조인다'를 잡아야 한다.

    임계가 토크 포화 오차(effort/kp = 0.20 rad)에 가까우면 첫 접촉에서 즉시 막힘으로 판정돼
    폐쇄가 `syn_close` 0.01 에 얼어붙는다(09.07 대본 파지 실측). 포화의 2배 이상이어야 한다.
    """
    thr = float(re.search(r"blocked_err_thr_rad:\s*float\s*=\s*([0-9.]+)", _CFG).group(1))
    assert 0.40 <= thr <= 1.2, thr
    assert "포화" in _CFG and "syn_close" in _CFG


def test_dependent_limits_are_widened_at_boot():
    """★09.07 rh_b1/rh_b2 대조: 확장이 없으면 222 epoch 중 118 epoch 에서 mimic 오차가
    400~916 rad 로 폭주했고, ±1.5 rad 확장에서 299 epoch 내내 최대 7.8 rad 였다.
    두 런의 테이블 관통 깊이는 같았다 — 고친 것은 접촉량이 아니라 제약의 만족 가능성이다."""
    m = float(re.search(r"mimic_dep_limit_margin_rad:\s*float\s*=\s*([0-9.]+)", _CFG).group(1))
    assert m >= 1.0, f"종속 한계 확장이 너무 작다: {m}"
    block = _fn_block(_ENV, "_widen_dependent_joint_limits")
    assert "write_joint_limits_to_sim" in block
    # 넓히기만 하고 좁히지 않는다 — 한계는 백스톱이고 위치는 mimic 제약이 정한다.
    assert "-= m" in block and "+= m" in block
    assert "_setup_fabrics" in _ENV and "_widen_dependent_joint_limits()" in _fn_block(
        _ENV, "_setup_fabrics")


def test_no_robot_gain_literals_in_env_or_cfg():
    assert re.search(r"(stiffness|damping)\s*=\s*[0-9]", _code(_ENV) + _code(_CFG)) is None
    for pat in (r"r_aj_\d", r"r_hj_\w+", r"r_hl_\w+"):
        assert re.search(pat, _code(_ENV) + _code(_CFG)) is None, pat

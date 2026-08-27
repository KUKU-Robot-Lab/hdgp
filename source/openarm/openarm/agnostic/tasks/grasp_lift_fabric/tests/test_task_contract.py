"""grasp_lift_fabric 계약 — 이 트랙은 `grasp_s2r` 를 상속하고 **손 제어만** 다르다.

08.27 전면 재작성. 구 계약 107건은 폐기된 설계(절대 palm 매핑·λμνρ 계층·코리더
래치·ADR 7축·tip IK)를 잠그고 있었고, **18건이 Isaac 게이트 뒤에서 조용히 skip**
되면서 그중 다수가 실제로는 ERROR/FAIL 상태였다. 구 파일 자신이 그 사고를 기록해
뒀다 — "게이트 뒤라 구현이 통째로 없는데도 통과로 보였다".

그래서 여기 계약은 **전부 Isaac 게이트 밖**(소스 텍스트 + 순수 데이터 프로필)이다.
물리·IK 는 probe 와 부팅 스모크의 몫이다.

실행:
    cd hdgp && PYTHONPATH=source/openarm python3 -m pytest \
        source/openarm/openarm/agnostic/tasks/grasp_lift_fabric/tests -q
"""

from __future__ import annotations

import pathlib
import re

import pytest

_TRACK = pathlib.Path(__file__).resolve().parents[1]
_SIB = _TRACK.parent / "grasp_s2r"


def _src(p: pathlib.Path) -> str:
    assert p.exists(), f"{p} 가 없다"
    return p.read_text(encoding="utf-8")


ENV = _src(_TRACK / "grasp_lift_fabric_env.py")
CFG = _src(_TRACK / "grasp_lift_fabric_env_cfg.py")
REG = _src(_TRACK / "config" / "__init__.py")
PROF = _src(_TRACK / "robot_profiles.py")
SIB_ENV = _src(_SIB / "grasp_s2r_env.py")
SIB_CTL = _src(_SIB / "grasp_s2r_control.py")
SIB_CFG = _src(_SIB / "grasp_s2r_env_cfg.py")


def _profiles():
    from openarm.agnostic.tasks.grasp_s2r import robot_profiles as rp
    return rp.PROFILES


# ======================================================================
# 1. 자매 상속 — 이 트랙의 정체성
# ======================================================================

def test_env_inherits_sibling_and_does_not_copy_it():
    """복제가 아니라 상속이어야 한다(08.26 사용자 확정: import 가 드리프트를 막는다)."""
    assert "from ..grasp_s2r.grasp_s2r_env import GraspS2REnv" in ENV
    assert re.search(r"class GraspLiftFabricEnv\(GraspS2REnv\)", ENV)
    assert "from ..grasp_s2r.grasp_s2r_env_cfg import GraspS2REnvCfg" in CFG
    assert re.search(r"class GraspLiftFabricEnvCfg\(GraspS2REnvCfg\)", CFG)
    # 상속본이 자매 env 를 통째로 베끼면 줄 수가 폭증한다. 손 제어만 덮는 크기여야 한다.
    assert len(ENV.splitlines()) < 400, "env 가 커졌다 — 자매 코드를 복제하고 있지 않은지"


def test_track_defines_no_reward_math():
    """보상은 자매 것(`grasp_s2r_rewards`)만 쓴다 — 우리 파일엔 수식이 없어야 한다."""
    for bad in ("def compute_", "approach_weight", "grasp_weight",
                "lift_weight", "success_weight", "wrap_retention"):
        assert bad not in ENV, f"env 에 보상 수식/가중치 잔재: {bad}"
        assert bad not in CFG, f"cfg 에 보상 가중치 잔재: {bad}"
    assert "grasp_s2r_rewards" in SIB_ENV


def test_dead_reward_modules_stay_deleted():
    """λμνρ 계층 게이트·graded envelope 는 폐기됐다. 부활하면 두 트랙이 갈린다."""
    for name in ("rewards_stage.py", "rewards_tip.py", "rewards.py"):
        assert not (_TRACK / name).exists(), f"{name} 가 되살아났다"


def test_absolute_palm_mapping_is_not_resurrected():
    """★팔 액션은 **홈+델타**다.

    절대 매핑(`palm = scale(a, 박스전체)`)은 저장소 공통 σ=1.0 과 곱해지면 매 스텝
    작업공간 전역에서 목표를 재추첨해 접근이 랜덤워크가 된다(08.27 실측: 클램프 전
    요청량 0.33~0.36 m/step 상시 포화). 이 트랙이 갈아엎힌 직접 원인이다.
    """
    assert "_kuka_absolute" not in ENV
    assert "_pre_physics_step" not in ENV, "팔 액션 경로를 덮으면 델타 규약이 깨진다"
    assert "self._home_palm.unsqueeze(0) + delta" in SIB_ENV


# ======================================================================
# 2. 상속 seam — 자매가 리팩터하면 **여기서** 깨져야 한다
# ======================================================================

@pytest.mark.parametrize("sym", [
    "def _setup_synergy", "def _synergy_targets", "def _close_progress",
    "def _setup_fabrics", "def _report_home_cage", "def _syn_to_fab",
])
def test_overridden_hooks_exist_in_sibling(sym):
    assert sym in SIB_CTL or sym in SIB_ENV, f"자매에 {sym} 가 없다 — 오버라이드가 죽었다"


@pytest.mark.parametrize("sym", [
    "self._syn_target", "self._syn_vel", "self._syn_ids", "self._syn_close",
    "self._syn_lo", "self._syn_hi", "self._syn_movable",
    "self._wrap_idx", "self._r_cage", "self._finger_names", "self._fab_t",
])
def test_inherited_state_names_still_exist(sym):
    """우리 오버라이드가 채우거나 읽는 부모 상태 이름."""
    assert sym in SIB_CTL or sym in SIB_ENV, f"자매 상태 {sym} 가 사라졌다"


def test_hand_targets_reach_pd_outside_fabric():
    """손은 fabric 밖 — 부모 `_apply_action` 이 관절 목표를 직접 PD 에 하달한다."""
    assert "set_joint_position_target(self._syn_target, joint_ids=self._syn_ids)" in SIB_CTL
    assert "hand_velocity_ff_scale" in SIB_CTL


# ======================================================================
# 3. 손 액션 매핑 — 이 트랙 고유
# ======================================================================

def test_hand_action_maps_home_to_open_and_plus_one_to_flexion_limit():
    """`a=−1` → 홈(펴짐) · `a=+1` → **굴곡 한계**.

    구 매핑 `a∈[-1,1] → [관절 lo, hi]` 는 `_3`/`_4` 한계가 대칭 ±90° 라 홈이 한계
    중앙이었다 — 액션 범위의 절반이 손등 쪽 **역굴곡**을 지시했다.
    """
    assert "0.5 * (a_hand.clamp(-1.0, 1.0) + 1.0)" in ENV
    assert "self._hand_home_free + u * (self._flex_limit - self._hand_home_free)" in ENV
    # 구 결함(한계 직접 보간)이 돌아오면 안 된다.
    assert not re.search(r"_hand_lo\s*\+.*\(_hand_hi\s*-\s*_hand_lo\)", ENV)


def test_flexion_sign_is_measured_at_boot_by_fk_and_fails_loud():
    """굴곡 부호는 **부팅 FK 실측**이다 — 좌우 미러를 상수로 박지 않는다.

    판정은 **대향 그룹 거리 감소**. palm +x 성분 규약은 엄지에서 오판한다(FK 실측).
    """
    m = re.search(r"def _measure_flex_signs.*?(?=\n    def )", ENV, re.S)
    assert m, "_measure_flex_signs 가 없다"
    body = m.group(0)
    assert "_fingertip_taskmap" in body, "FK 로 재야 한다"
    assert "contact_group_a" in body and "contact_group_b" in body
    assert "raise RuntimeError" in body, "판별 불가 시 fail-loud"
    assert "_flex_limit = torch.where(self._flex_sign > 0" in body
    # 가동폭 0 자유 관절도 fail-loud (액션을 줘도 안 움직이는 관절)
    assert "가동폭 0" in body


def test_flexion_measurement_runs_after_fabric_exists():
    """`_measure_flex_signs` 는 fabric FK 를 쓰므로 `super()._setup_fabrics()` **뒤**."""
    m = re.search(r"def _setup_fabrics.*?(?=\n    def )", ENV, re.S)
    assert m
    # docstring 안의 언급이 순서 판정을 오염시키므로 **코드만** 남긴다.
    body = re.sub(r'"""[\s\S]*?"""', "", m.group(0))
    i_home = body.index("_apply_hand_home_override")
    i_super = body.index("super()._setup_fabrics()")
    i_flex = body.index("_measure_flex_signs")
    assert i_home < i_super < i_flex, (
        "순서 계약: 홈 오버라이드 → super()._setup_fabrics() → 굴곡 부호 실측")


def test_hand_home_override_precedes_fabric_rest_pose():
    """홈 오버라이드가 fabric cspace rest·리셋 q0 보다 **먼저** 적용돼야 셋이 안 갈린다."""
    assert "default_config.copy_(self.fabric_q)" in SIB_CTL
    assert "hand_home_override" in CFG and "pinky_1" in CFG
    assert "_dj[:, _jn.index(_nm)] = float(_val)" in ENV


def test_action_dim_is_derived_from_free_joints_not_synergy_channels():
    p = _profiles()["tesollo_right"]
    frozen = tuple(n.replace("{side}", "r") for n in _frozen_override())
    assert all(n in p.hand_joint_names for n in frozen), "고정 관절 이름 오타"
    n_free = len(p.hand_joint_names) - len(frozen)
    assert n_free == 13, f"자유 관절 {n_free} — 고정 목록이 바뀌었다"
    # cfg 는 obs/state 공식을 복제하지 않고 **차분**만 더한다(부모 공식 유지).
    assert "delta = (6 + n_free) - self.action_space" in CFG
    assert "self.observation_space += delta" in CFG
    assert "self.state_space += delta" in CFG


def _frozen_override() -> tuple[str, ...]:
    m = re.search(r"frozen_hand_joints_override:.*?=\s*\((.*?)\)\n", CFG, re.S)
    assert m, "frozen_hand_joints_override 선언이 없다"
    return tuple(re.findall(r'"([^"]+)"', m.group(1)))


def test_frozen_set_covers_abduction_and_pinky():
    """`_1` 전부 + pinky_2 + thumb_2 — 08.26 사용자 지시."""
    names = {n.replace("{side}_hj_", "") for n in _frozen_override()}
    assert {"thumb_1", "index_1", "middle_1", "ring_1", "pinky_1"} <= names
    assert "pinky_2" in names and "thumb_2" in names


# ======================================================================
# 4. 손 보조 게이트는 없다 (08.27 사용자 지시)
# ======================================================================

@pytest.mark.parametrize("flag", ["synergy_contact_freeze", "couple_four_fingers"])
def test_synergy_only_mechanisms_are_off(flag):
    """누산 delta 위에서만 뜻이 있는 시너지 기구 — 우리 절대 매핑엔 표현 불가."""
    assert re.search(rf"{flag}: bool = False", CFG), f"{flag} 가 켜져 있다"


def test_close_gate_stays_on_because_reward_design_is_shared():
    """★close_gate 는 **보상 설계**의 일부다(자매 6632002 부터 `grasp` 가 곱한다).

    임계가 부팅 FK 로 실측되는 `r_cage` 하나뿐이라 로봇 비의존이므로, 로봇 특수성
    제거 대상이 아니다 — 끄면 우리 보상이 자매와 갈린다(사용자 원칙 08.27).
    """
    assert re.search(r"close_gate_enabled: bool = True", CFG)
    assert "close_gate.clamp(0.0, 1.0) * grasp_quality" in _src(
        _SIB / "grasp_s2r_rewards.py"), "자매 grasp 항이 더는 close_gate 를 안 쓴다"


def test_close_gate_is_not_applied_to_our_hand_action():
    """게이트는 **보상 쪽만** 받는다 — 손 액션 경로는 이 트랙 고유다."""
    m = re.search(r"def _synergy_targets.*?(?=\n    def |\Z)", ENV, re.S)
    assert m and "_close_gate" not in m.group(0)


def test_close_progress_is_measured_joint_not_command():
    """지령을 재면 손이 테이블에 눌려 펴져도 만점이 나온다(자매 72ac912 실측).

    우리는 `_close_progress` 를 **상속**하므로 자매가 되돌리면 여기서 잡힌다.
    """
    m = re.search(r"def _close_progress.*?(?=\n    def )", SIB_CTL, re.S)
    assert m and "self.robot.data.joint_pos[:, self._syn_ids]" in m.group(0), (
        "자매 폐쇄도가 지령 기반으로 돌아갔다")
    # 우리는 분모를 자유 관절 ∩ (open≠grip) 로 좁힌다.
    assert "self._syn_movable = _free_mask & self._syn_movable" in ENV


def test_no_approach_freeze_in_this_track():
    """접근 중 손가락 동결은 제거됐다 — 정책이 `a=−1` 로 손을 열어 둔다."""
    for bad in ("freeze_fingers_until_approach", "finger_release_dist_m",
                "_hand_freeze_targets", "_solve_freeze_pose", "_fingers_free"):
        assert bad not in ENV and bad not in CFG, f"동결 잔재: {bad}"


def test_close_gate_flag_is_honored_by_parent():
    """게이트를 끄면 부모가 실제로 상수 1 을 넣는지(플래그가 죽어 있으면 안 된다)."""
    assert "if bool(self.cfg.close_gate_enabled):" in SIB_ENV
    assert "self._close_gate = torch.ones(self.num_envs" in SIB_ENV


# ======================================================================
# 5. 감쌈 분모 — 가용 손가락만
# ======================================================================

def test_unusable_fingers_declared_and_removed_from_wrap_denominator():
    """소지는 `_1`/`_2` 가 둘 다 고정이라 감쌈이 **원리적으로 불가**하다.

    분모에 남기면 wrap_frac 상한이 0.75 로 깎이고, 벌어진 채 굳은 손가락이 컵에
    걸린 것이 접촉으로 세어진다(08.27 실측).
    """
    assert re.search(r'hand_unusable_fingers: tuple\[str, \.\.\.\] = \("pinky",\)', CFG)
    assert "hand_unusable_fingers" in ENV
    assert "self._wrap_idx = torch.tensor(_keep" in ENV
    assert "raise RuntimeError" in ENV.split("hand_unusable_fingers")[1][:900]


def test_wrap_denominator_source_in_sibling_is_group_b_cap_envelope():
    assert "f in p.contact_group_b and f in p.envelope_fingers" in SIB_ENV


# ======================================================================
# 6. 자매에서 상속받는 세팅 (전면 동기의 실체)
# ======================================================================

@pytest.mark.parametrize("field", [
    "palm_delta_xyz", "palm_delta_rot_deg",
    "palm_cmd_rate_limit_m", "palm_cmd_rate_limit_rot_deg",
    "palm_box_z_min_override", "palm_slew_pos", "palm_slew_rot_deg",
    "fabrics_dt", "fabric_decimation", "fabrics_damping_gain",
    "fabric_velocity_ff_scale",
])
def test_arm_command_fields_are_never_overridden_here(field):
    """★★팔 지령 규약은 **한 글자도 갈리면 안 된다**(08.27 사용자: "제일 중요").

    `palm_cmd_step_raw`(클램프 전 요청량)는 리미터 포화율의 유일한 근거이고, 그
    값이 두 트랙에서 같은 의미를 가지려면 델타 박스·리미터·fabric 시간축이 전부
    같아야 한다. 여기서 하나라도 덮으면 자매와 수치를 비교할 근거가 사라진다.
    """
    assert field not in CFG, f"{field} 를 이 트랙이 덮고 있다 — 자매 값을 상속하라"


def test_arm_command_path_is_not_reimplemented_here():
    """팔 액션·리미터·fabric 적분은 상속 그대로 — 오버라이드 자체를 금지한다."""
    for sym in ("_pre_physics_step", "palm_targets", "_prev_palm_cmd",
                "_delta_lo", "_delta_hi", "_step_fabric", "_apply_action",
                "_palm_cmd_step_raw"):
        assert sym not in ENV, f"팔 경로 재정의 발견: {sym}"


def test_arm_action_is_home_plus_delta_with_slew():
    assert "palm_delta_xyz" in SIB_CFG and "palm_delta_rot_deg" in SIB_CFG
    assert "palm_cmd_rate_limit_m: float = 0.02" in SIB_CFG
    assert "palm_cmd_rate_limit_rot_deg: float = 2.9" in SIB_CFG
    # 클램프 **전** 원값 로깅 — 리미터 포화율의 유일한 근거.
    assert "_palm_cmd_step_raw" in SIB_ENV


def test_home_is_never_clipped_out_of_the_box():
    """박스가 홈을 잘라내면 `a=0`(홈)의 의미가 깨진다."""
    assert "torch.minimum(self._palm_lo, self._home_palm)" in SIB_ENV
    assert "torch.maximum(self._palm_hi, self._home_palm)" in SIB_ENV


def test_failures_terminate_and_only_timeout_truncates():
    """실패를 truncated 로 내보내면 rl_games 가 bootstrap 보너스를 준다 —
    "잘 잡았다가 넘어뜨리기"가 자기강화된다(자매 트랙 실측 붕괴)."""
    m = re.search(r"def _get_dones.*?(?=\n    def )", SIB_ENV, re.S)
    assert m
    body = m.group(0)
    assert re.search(r"terminated\s*=.*tipped", body, re.S)
    assert re.search(r"truncated\s*=\s*self\.episode_length_buf", body)


def test_goal_is_measured_from_settled_height_not_spawn():
    assert "settled[:, 2] = float(self.cfg.table_surface_z)" in SIB_ENV
    assert "self.goal_pos[env_ids] = settled + torch.tensor" in SIB_ENV


def test_adr_off_and_single_cup():
    assert "enable_adr: bool = False" in SIB_CFG
    assert "cup_big_rl.usd" in SIB_CFG
    assert "replicate_physics=True" in SIB_CFG


def test_spawn_center_comes_from_profile_and_gap_is_fail_loud_here():
    """자매 스폰 중심은 **자매 홈 케이지** 역산값이다. 우리는 pinky_1=0 홈이라
    부팅에서 다시 검산해야 한다 — 경고가 아니라 정지."""
    assert "p.object_spawn_center[0] + offs[:, 0]" in SIB_ENV
    m = re.search(r"def _report_home_cage.*?(?=\n    def |\Z)", ENV, re.S)
    assert m and "raise RuntimeError" in m.group(0)
    # ★중심 간격만 보면 안 된다 — 컵은 축당 ±spawn_range 로 흩어진다. 최악 스폰의
    #   여유를 숫자로 남겨야 "가끔 리셋에서 박히는" 산발 사고를 원인까지 읽을 수 있다.
    assert "2.0 ** 0.5" in m.group(0) and "spawn_range" in m.group(0)


# ======================================================================
# 7. 등록
# ======================================================================

def test_registration_uses_our_profile_registry_and_keeps_play_ids():
    assert "from .. import robot_profiles as _rp" in REG
    assert '"-play"' in REG and '"-play-lstm"' in REG
    ids = re.findall(r'f"open-\{_tag\}_grasp_lift_fab\{_suffix\}"', REG)
    assert ids, "id 규약(open-<tag>_<r|l>_grasp_lift_fab)이 깨졌다"
    # run_naming 정규식 `^(open-\w+)_([rbl])_(.+)$` 두 번째 슬롯이 r/l/b 여야 한다.
    tags = re.findall(r'"(\w+_[rl])":', REG)
    assert set(tags) == {"sens_r", "sens_l", "bis_r"}


def test_agent_yaml_matches_sibling_except_name():
    for f in ("rl_games_ppo_cfg.yaml", "rl_games_ppo_lstm_cfg.yaml"):
        ours = _src(_TRACK / "config" / "agents" / f)
        theirs = _src(_SIB / "config" / "agents" / f)
        assert ours.replace("agn_grasp_lift_fab", "agn_grasp_s2r") == theirs, (
            f"{f} 가 자매와 갈렸다 — 이름 말고 바꾸려면 근거를 남기고 이 계약을 고쳐라")
    assert not (_TRACK / "config" / "agents" / "rl_games_ppo_paper_cfg.yaml").exists()


# ======================================================================
# 8. 자매 파일 불가침
# ======================================================================

# ======================================================================
# 7-b. USD 교체 — 이 트랙의 존재 이유
# ======================================================================

def test_new_robot_is_a_field_diff_not_a_new_profile_literal():
    """자산 교체는 자매 프로필의 **필드 차분**이어야 한다 — 통째로 다시 쓰면 드리프트."""
    assert "dataclasses.replace(" in PROF
    assert '_SENSOR_R = _s2r.PROFILES["tesollo_right"]' in PROF
    # 자매 레지스트리에 **등록**은 하되 기존 키는 절대 덮지 않는다.
    assert "_s2r.PROFILES[_n] = _p" in PROF
    assert "덮으려 한다 — 금지" in PROF


def test_palm_normal_axis_is_not_made_robot_specific():
    """★손바닥 법선은 `palm_ee` **+x** 이고 자산이 바뀌어도 같다(사용자 확정 08.27).

    한때 `modules/robots.py` 의 `palmar_axis_local` 을 palm 법선으로 오독해 자산별
    축 치환을 넣었다가 되돌렸다 — 그 필드는 **손가락 마디 링크**의 손바닥면 방향
    (마디별 dict)이지 palm body 의 법선이 아니다. 자매의 `_palm_ee_R()`(열 0 = 법선)
    을 그대로 상속한다.
    """
    assert "_palm_ee_R" not in ENV, "palm 법선축을 이 트랙에서 재정의하고 있다"
    assert "PALM_NORMAL_COL" not in ENV and "PALM_NORMAL_COL" not in PROF
    assert "열 0 = 손바닥 법선(+x)" in SIB_CTL, "자매의 법선 규약이 바뀌었다"


def test_bis_profile_swaps_only_asset_dependent_fields():
    from openarm.agnostic.tasks.grasp_lift_fabric import robot_profiles as rp
    s, b = rp.PROFILES["tesollo_right"], rp.PROFILES["bis_right"]
    # 자산이 바꾸는 것
    assert "bi_s" in b.usd_relpath and b.fabric_robot_dir == "openarm_tesollo_bi_s"
    assert b.palm_box_min != s.palm_box_min          # palm 오프셋 54.8mm 차
    # 자산이 바꾸지 **않는** 것 — 이름 규약·시너지 자세·접촉 그룹은 상속
    for f in ("hand_joint_names", "hand_open_pose", "hand_grip_pose",
              "hand_channel_of_joint", "contact_group_a", "contact_group_b",
              "finger_sensor_bodies", "fingertip_bodies", "palm_body",
              "arm_joint_regex", "hand_joint_regex", "fabric_joint_order",
              "palm_rot_center_deg", "palm_rot_half_deg"):
        assert getattr(b, f) == getattr(s, f), f"{f} 가 자산 교체로 바뀌면 안 된다"
    # 반대편 팔 구성이 다르다 — 없는 관절 이름이 남으면 Articulation 조립이 죽는다
    assert not [k for k in b.init_joint_pos if "gripper" in k]
    assert "left_gripper" not in b.actuator_specs
    assert len([k for k in b.init_joint_pos if k.startswith("l_hj_")]) == 20


def test_track_only_imports_sibling_never_writes_it():
    """자매 트랙은 다른 세션 소유다. 우리 소스는 import 만 해야 한다."""
    for src in (ENV, CFG, REG):
        assert "grasp_sensor" not in src, "폐기된 grasp_sensor 의존이 남아 있다"
    # 자매 모듈에 monkey-patch 하면 안 된다 — 읽기(PROFILES[...])는 허용, 대입은 금지.
    for src in (ENV, CFG, REG):
        assert not re.search(r"^\s*(_rp|_s2r|grasp_s2r)[\w.]*\s*=\s*[^=]", src, re.M), (
            "자매 모듈 심볼에 대입하고 있다 — 자매 파일은 불가침")
        assert "setattr(_rp" not in src

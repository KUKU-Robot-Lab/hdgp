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

@pytest.mark.parametrize("flag", [
    "close_gate_enabled", "synergy_contact_freeze", "couple_four_fingers"])
def test_hand_side_gates_are_off(flag):
    assert re.search(rf"{flag}: bool = False", CFG), f"{flag} 가 켜져 있다"


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


# ======================================================================
# 7. 등록
# ======================================================================

def test_registration_uses_sibling_profiles_and_keeps_play_ids():
    assert "from ...grasp_s2r import robot_profiles as _rp" in REG
    assert '"-play"' in REG and '"-play-lstm"' in REG
    ids = re.findall(r'f"open-\{_tag\}_grasp_lift_fab\{_suffix\}"', REG)
    assert ids, "id 규약(open-<tag>_<r|l>_grasp_lift_fab)이 깨졌다"
    # run_naming 정규식 `^(open-\w+)_([rbl])_(.+)$` 두 번째 슬롯이 r/l/b 여야 한다.
    tags = re.findall(r'"(sens_[rl])":', REG)
    assert set(tags) == {"sens_r", "sens_l"}


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

def test_track_only_imports_sibling_never_writes_it():
    """자매 트랙은 다른 세션 소유다. 우리 소스는 import 만 해야 한다."""
    for src in (ENV, CFG, REG):
        assert "grasp_sensor" not in src, "폐기된 grasp_sensor 의존이 남아 있다"
    # 자매 모듈에 monkey-patch 하면 안 된다 — 읽기(PROFILES[...])는 허용, 대입은 금지.
    for src in (ENV, CFG, REG):
        assert not re.search(r"^\s*(_rp|_s2r|grasp_s2r)[\w.]*\s*=\s*[^=]", src, re.M), (
            "자매 모듈 심볼에 대입하고 있다 — 자매 파일은 불가침")
        assert "setattr(_rp" not in src

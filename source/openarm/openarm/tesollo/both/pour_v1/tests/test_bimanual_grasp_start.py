"""[both/pour_v1] 양손 물리 파지 시작 계약 테스트 (정적 — Isaac 불필요).

pour_v1 은 `both/pour_sensor` 포크지만 **왼팔이 receiver 컵을 실제로 쥔 상태에서
에피소드를 시작**한다는 점이 다르다. 이 파일은 그 차이가 유지되는지를 못 박는다.

의도적으로 소스 텍스트를 검사하는 테스트가 섞여 있다 — 물리 동작은 Isaac 없이는
확인할 수 없지만, "kinematic-follow 로 되돌아갔다" 같은 구조적 회귀는 텍스트로 잡힌다.
"""
from __future__ import annotations

import io
import math
import os
import tokenize
from pathlib import Path

import numpy as np

from openarm.tesollo.both.pour_v1.pour_right_constants import NUM_ACTIONS, NUM_OBSERVATIONS
from openarm.tesollo.both.pour_v1.pour_right_preset import (
    LEFT_ARM_AND_HAND_JOINT_NAMES,
    LEFT_ARM_JOINT_NAMES,
    LEFT_ARM_REST_JOINT_POS,
    LEFT_HAND_BODY_NAME_CANDIDATES,
    LEFT_HAND_JOINT_NAMES,
    LEFT_TARGET_CUP_QUAT_WXYZ,
    RIGHT_HAND_JOINT_NAMES,
)

TASK_DIR = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (TASK_DIR / name).read_text()


def _code(name: str) -> str:
    """`#` 주석만 제거하고 나머지 원문(공백·문자열 포함)은 보존한다.

    이 파일의 구조 검사들이 우리가 쓴 설명 주석에 걸려 오탐을 내지 않게 한다
    (예: "구 코드는 `else -1` 이었다" 라는 주석 때문에 회귀 검사가 실패하던 문제).

    독스트링은 일부러 남긴다 — 제거하려 했더니 줄 시작의 **딕셔너리 키 문자열**
    (`"right_arm_proximal":` 등)까지 독스트링으로 오인해 삭제되는 버그가 있었다.
    구조 검사에 방해되는 것은 주석뿐이므로 여기서 멈추는 게 맞다.
    """
    src = _read(name)
    lines = src.splitlines(keepends=True)
    # 주석 토큰의 (행, 열) 위치를 모아 해당 열 이후를 잘라낸다 (문자열 안 '#' 는 안전).
    cuts: dict[int, int] = {}
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            row, col = tok.start
            cuts[row] = min(cuts.get(row, col), col)
    for row, col in cuts.items():
        lines[row - 1] = lines[row - 1][:col].rstrip() + "\n"
    return "".join(lines)


def _nows(text: str) -> str:
    """모든 공백을 제거한다. `_code()` 는 토큰 사이 공백을 보존하지 않으므로
    구조 검사(정규식·부분문자열)는 공백 무관 비교로 해야 한다."""
    return "".join(text.split())


# ---------------------------------------------------------------------------
# 왼손 관절 구성 — DG-5FS 20관절
# ---------------------------------------------------------------------------
def test_left_hand_has_twenty_joints_mirroring_right() -> None:
    # Arrange / Act / Assert
    assert len(LEFT_HAND_JOINT_NAMES) == 20
    assert len(RIGHT_HAND_JOINT_NAMES) == 20
    # 좌우가 접두사만 다른 완전 미러여야 warm state 열 순서가 호환된다
    assert [n[1:] for n in LEFT_HAND_JOINT_NAMES] == [n[1:] for n in RIGHT_HAND_JOINT_NAMES]


def test_left_gripper_joints_are_gone() -> None:
    """구 2-DOF 그리퍼(l_hj_gripper_*)가 남아 있으면 bi_s_rl USD 에서 해석 실패한다."""
    assert not any("gripper" in n for n in LEFT_ARM_AND_HAND_JOINT_NAMES)
    assert len(LEFT_ARM_AND_HAND_JOINT_NAMES) == len(LEFT_ARM_JOINT_NAMES) + 20


def test_left_rest_pose_covers_every_hand_joint() -> None:
    """spawn 기본 자세에 20관절이 모두 있어야 한다 (누락 시 0 으로 조용히 채워짐)."""
    for name in LEFT_HAND_JOINT_NAMES:
        assert name in LEFT_ARM_REST_JOINT_POS, f"{name} 누락"


def test_left_thumb_sign_mirrors_right() -> None:
    """왼손 thumb 은 오른손과 부호가 반대여야 한다 (left/grasp_v1 규약과 동일)."""
    assert LEFT_ARM_REST_JOINT_POS["l_hj_thumb_2"] > 0
    assert LEFT_ARM_REST_JOINT_POS["l_hj_thumb_3"] > 0


# ---------------------------------------------------------------------------
# receiver 컵 — 물리 강체 + upright 스폰
# ---------------------------------------------------------------------------
def test_left_cup_is_dynamic_not_kinematic() -> None:
    """receiver 컵이 kinematic 으로 되돌아가면 '물리 파지'가 아니게 된다."""
    # `left_target_cup_cfg` 블록만 떼어 본다 — table_cfg 는 kinematic 이 정상이므로
    # 파일 전체를 검사하면 오탐이 난다.
    cfg = _code("pour_right_env_cfg.py")
    start = cfg.index("left_target_cup_cfg")
    block = cfg[start : cfg.index("beads_cfg", start)]
    assert "kinematic_enabled=False" in block
    assert "kinematic_enabled=True" not in block
    # 중력을 끄면 손이 놓쳐도 컵이 떠 있어 낙하 판정이 무의미해진다
    assert "disable_gravity=False" in block


def test_kinematic_follow_is_removed() -> None:
    """매 스텝 컵 pose 를 써 넣는 follow 경로가 부활하지 않았는지 확인."""
    env = _code("pour_right_env.py")
    assert "_get_left_cup_follow_pose" not in env
    assert "_left_cup_follow_offset" not in env


def test_left_cup_spawn_quat_is_upright() -> None:
    """DG-5FS 마운트 yaw 로 구 R_y90 보정이 깨졌던 회귀를 막는다."""
    w, x, y, z = LEFT_TARGET_CUP_QUAT_WXYZ
    # 쿼터니언 → 회전행렬의 z축
    zz = 1.0 - 2.0 * (x * x + y * y)
    assert math.isclose(zz, 1.0, abs_tol=1e-6), f"컵이 upright 가 아니다 (world z dot={zz:.4f})"
    assert math.isclose(np.linalg.norm(LEFT_TARGET_CUP_QUAT_WXYZ), 1.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# 좌/우 이중 warm bank
# ---------------------------------------------------------------------------
def test_left_warm_bank_config_exists() -> None:
    cfg = _read("pour_right_env_cfg.py")
    assert "left_warm_state_paths" in cfg
    assert "grasp_warm_tesollo_left.hdf5" in cfg
    # 오른팔 뱅크도 좌우 구분 이름으로 분리돼야 혼동이 없다
    assert "grasp_warm_tesollo_right.hdf5" in cfg


def test_reset_applies_left_warm_on_every_path() -> None:
    """reset 경로 4곳이 모두 왼팔 warm 헬퍼를 쓰는지 (경로별 표류 방지)."""
    env = _code("pour_right_env.py")
    # _reset_idx 는 warm 분기와 pregrasp 분기에서 **같은 샘플**을 재사용하므로
    # 관절 쓰기는 4회지만 샘플링은 3회다 (일관성 유지 목적 — 두 분기가 다른 자세면 안 된다).
    assert env.count("_write_left_warm_joints(") == 5   # 정의 1 + 호출 4
    # 정의 1 + 리셋 3 + 겹침 재추첨 1 = 5.
    # 재추첨(2026-08-18)은 겹친 페어만 왼쪽 pick 을 다시 뽑으므로 호출이 하나 늘었다.
    assert env.count("_sample_left_warm(") == 5
    assert env.count("_place_left_cup(") == 4           # 정의 1 + 호출 3


def test_left_arm_no_longer_forced_to_rest_in_warm_reset() -> None:
    """구 코드의 `left_arm_dof_indices = left_arm_zero_pos` 직접 대입은
    degrade 분기(헬퍼 내부) 한 곳에만 남아야 한다."""
    env = _nows(_code("pour_right_env.py"))
    assert env.count("self.left_arm_dof_indices]=self.left_arm_zero_pos[0]") == 1


# ---------------------------------------------------------------------------
# 왼컵 낙하 종료
# ---------------------------------------------------------------------------
def test_left_cup_drop_termination_wired() -> None:
    cfg = _read("pour_right_env_cfg.py")
    env = _read("pour_right_env.py")
    for key in ("left_cup_drop_enable", "left_cup_drop_dist_m",
                "left_cup_drop_z_m", "left_cup_drop_penalty"):
        assert key in cfg, f"{key} 누락"
    # 종료 조건과 보상 페널티 양쪽에 반영돼야 한다
    assert "| left_cup_dropped" in env
    assert "left_cup_drop_penalty * self._left_cup_dropped.float()" in env
    # 진단 로깅에도 포함
    assert '("left_cup_dropped", left_cup_dropped)' in env


# ---------------------------------------------------------------------------
# 차원 불변 — 단계형 학습(frozen→learned) 체크포인트 인계 조건
# ---------------------------------------------------------------------------
def test_action_dim_is_invariant_for_staged_training() -> None:
    """왼손을 position-hold 로 둔 이유가 바로 이 불변성이다.

    frozen(1단계) → learned(2단계) 로 넘어갈 때 **action 차원**이 같아야 체크포인트가
    그대로 인계된다. frozen 은 action[12:15] 를 무시할 뿐 차원을 바꾸지 않는다.
    """
    assert NUM_ACTIONS == 15


def test_actor_obs_drops_frozen_left_hand() -> None:
    """왼팔 obs = arm 7관절만 (왼손 20관절은 동결이라 행동 정보가 없다).

    구 pour_sensor 는 왼팔 9(arm7+그리퍼2)를 넣어 55 였다. pour_v1 에서 같은 코드를 두면
    왼손 20관절이 흘러들어 obs 가 91 로 터진다(실행으로 확인된 버그).
    """
    assert NUM_OBSERVATIONS == 51        # 7+7+5+7+7+3+3+3+3+6
    env = _code("pour_right_env.py")
    # 왼팔 obs 는 arm-only 인덱스를 써야 한다
    assert "self.robot.data.joint_pos[:, self.left_arm_only_dof_indices]" in env
    assert "self.robot.data.joint_pos[:, self.left_arm_dof_indices]" not in env


def test_frozen_mode_ignores_left_action_without_changing_dim() -> None:
    env = _read("pour_right_env.py")
    assert 'if _recv_mode == "frozen":' in env
    # frozen 이 per-env rest 를 쓰는지 — 구 코드의 전역 FK 상수를 쓰면 warm 파지가 깨진다
    assert "_new_left_target = self._left_tcp_rest_env.clone()" in env


def test_left_tcp_rest_is_per_env() -> None:
    """왼팔 rest 는 warm 파지자세라 env 마다 다르다."""
    env = _read("pour_right_env.py")
    assert "_left_tcp_rest_env" in env
    assert "_left_tcp_rest_captured" in env
    # 클램프 박스도 per-env 여야 한다
    assert "self._left_tcp_min_env, self._left_tcp_max_env" in env


# ---------------------------------------------------------------------------
# 안전장치 — 조용한 실패 금지
# ---------------------------------------------------------------------------
def test_body_index_resolution_raises_instead_of_minus_one() -> None:
    """이름을 못 찾으면 예외를 던져야 한다.

    구 코드는 `body_names.index(x) if x in body_names else -1` 형태였고, -1 은
    파이썬에서 **마지막 body** 를 뜻하므로 에러 없이 엉뚱한 링크를 잡았다.
    (산문에 그 패턴이 등장하므로 문자열 부재 검사 대신 **구조**를 검사한다.)
    """
    env = _nows(_code("pour_right_env.py"))
    # 좌우 palm 모두 resolver 경유
    assert "self.palm_body_index:int=self._resolve_body_index(" in env
    assert "self._left_hand_body_index=self._resolve_body_index(" in env
    # resolver 는 실패 시 반드시 예외
    body = env[env.index("def_resolve_body_index("):]
    body = body[: body.index("def__init__(")]
    assert "raiseRuntimeError(" in body, "_resolve_body_index 가 예외를 던지지 않는다"
    assert len(LEFT_HAND_BODY_NAME_CANDIDATES) >= 2


def test_uses_bi_s_rl_usd() -> None:
    cfg = _read("pour_right_env_cfg.py")
    assert "openarm_tesollo_bi_s_rl" in cfg
    assert "openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.usd" not in cfg


def test_z_lock_reference_is_low_passed() -> None:
    """물리 파지된 왼컵의 진동이 오른팔 목표로 직결되지 않게 한다."""
    cfg = _read("pour_right_env_cfg.py")
    env = _read("pour_right_env.py")
    assert "pour_spout_z_lock_lpf_alpha" in cfg
    assert "_update_spout_z_lock_ref" in env
    assert "self._spout_z_lock_ref + self.cfg.pour_z_margin" in env


# ---------------------------------------------------------------------------
# grasp_v1 물리 정합 — "수집 물리 = 소비 물리"
#   warm state 를 만든 환경(grasp_v1)과 재생하는 환경(pour_v1)의 물리가 다르면
#   리셋 직후 파지가 어긋난다. 아래는 그 정합을 코드 레벨에서 잠근다.
# ---------------------------------------------------------------------------
_GRASP_R_CFG = (
    TASK_DIR.parents[1] / "right" / "grasp_v1" / "grasp_right_env_cfg.py"
)
_GRASP_L_CFG = (
    TASK_DIR.parents[1] / "left" / "grasp_v1" / "grasp_left_env_cfg.py"
)


def _actuator_gains(text: str) -> dict[str, tuple[float, float]]:
    """`ImplicitActuatorCfg` 블록에서 {group: (stiffness, damping)} 을 뽑는다."""
    import re

    out: dict[str, tuple[float, float]] = {}
    text = _nows(text)
    for m in re.finditer(
        r'"(?P<name>\w+)":ImplicitActuatorCfg\((?P<body>[^()]*?)\),',
        text,
    ):
        body = m.group("body")
        k = re.search(r"stiffness=([0-9.]+)", body)
        d = re.search(r"damping=([0-9.]+)", body)
        if k and d:
            out[m.group("name")] = (float(k.group(1)), float(d.group(1)))
    return out


def test_both_hands_use_grasp_v1_grasping_hand_gains() -> None:
    """pour_v1 은 양손이 파지하므로 좌우 모두 grasp_v1 의 '파지 손' 게인을 쓴다.

    grasp_v1 은 파지 손만 5/2 로 두고 유휴 손은 400/60 으로 굳혀둔다
    (S1~S4 실측: k=400 은 effort limit 7.5N·m 대비 토크 포화 영역 = bang-bang).
    """
    pour = _actuator_gains(_code("pour_right_env_cfg.py"))
    grasp = _actuator_gains(_code(str(_GRASP_R_CFG)))
    ref = grasp["tesollo_hand_curl"]                 # grasp_v1 파지 손 기준값
    assert ref == (5.0, 2.0), f"grasp_v1 파지 손 게인이 바뀌었다: {ref}"
    for side in ("", "left_"):
        for part in ("abduction", "curl", "pip", "dip"):
            g = f"tesollo_{side}hand_{part}"
            assert g in pour, f"{g} 그룹 누락"
            assert pour[g] == ref, f"{g}={pour[g]} != grasp_v1 {ref}"


def test_arm_groups_match_calibration_naming() -> None:
    """r2s_autotune 캘리브는 `{side}_arm_proximal/elbow/wrist` 그룹명을 기대한다.

    이름이 다르면 게인 주입이 **조용히 fallback** 된다(문서화된 함정).
    pour_v1 은 양팔이 능동이므로 좌우 모두 3분할해야 한다.
    """
    pour = _actuator_gains(_code("pour_right_env_cfg.py"))
    for side in ("right", "left"):
        for part in ("proximal", "elbow", "wrist"):
            assert f"{side}_arm_{part}" in pour, f"{side}_arm_{part} 그룹 누락"
    # 단일 그룹으로 되돌아가면 부위별 friction 을 넣을 수 없다
    assert "openarm_right_arm" not in pour
    assert "openarm_left_arm" not in pour


def test_arm_friction_matches_real2sim_measurement() -> None:
    """팔 Coulomb 마찰은 07.29 real2sim 실측값 — pour_v1 엔 아예 없었다."""
    import re

    pour = _nows(_code("pour_right_env_cfg.py"))
    grasp_r = _nows(_code(str(_GRASP_R_CFG)))
    grasp_l = _nows(_code(str(_GRASP_L_CFG)))
    expected = {"proximal": 0.213, "elbow": 0.493, "wrist": 0.151}
    for side, ref_text in (("right", grasp_r), ("left", grasp_l)):
        for part, val in expected.items():
            m = re.search(
                rf'"{side}_arm_{part}":\s*ImplicitActuatorCfg\(.*?friction=([0-9.]+)',
                pour,
                re.S,
            )
            assert m, f"{side}_arm_{part} 에 friction 이 없다"
            assert float(m.group(1)) == val, f"{side}_arm_{part} friction={m.group(1)} != {val}"
            # grasp_v1 쪽 값이 바뀌면 이 테스트가 먼저 깨져야 한다
            assert f"friction={val}" in ref_text, f"grasp_v1 {side} friction {val} 이 사라졌다"


def test_source_cup_mass_follows_grasp_v1() -> None:
    """수집↔소비 force-ratio 정합. (별도 파일 테스트와 중복이지만 여기 의도를 남긴다)"""
    import re

    pour = _code("pour_right_env_cfg.py")
    m = re.search(r"source_cup_fixed_mass:float\|None=([0-9.]+)", _nows(pour))
    assert m, "source_cup_fixed_mass 기본값 없음"
    grasp = _nows(_code(str(_GRASP_R_CFG)))
    g = re.search(r"_BASE_OBJECT_MASS:float=([0-9.]+)", grasp)
    assert g, "grasp_v1 _BASE_OBJECT_MASS 없음"
    assert float(m.group(1)) == float(g.group(1))


def test_palm_workspace_intentionally_differs_from_grasp() -> None:
    """palm 워크스페이스는 **따라가지 않는다** — 따라가면 deep tilt 가 막힌다.

    pour 박스는 깊은 tilt 시 palm 이 target 너머로 스윙해야 해서 x_min 을 -0.30 까지
    열어둔 값이다(test11). grasp 박스(0.20)로 되돌리면 tilt 가 plateau 한다.
    이 테스트는 "정합" 작업이 이 부분까지 번지는 것을 막는 안전장치다.
    """
    pour = _nows(_code("pour_right_preset.py"))
    assert "-0.30,-0.55,0.10" in pour, "pour deep-tilt palm 박스가 바뀌었다"


def test_source_cup_asset_intentionally_differs_from_grasp() -> None:
    """컵 자산도 따라가지 않는다 — grasp 는 '속이 찬 원통', pour 는 '속 빈 SDF 컵'.

    grasp 자산으로 바꾸면 비드가 컵 안에 들어갈 수 없어 붓기 자체가 불가능해진다.
    """
    pour = _code("pour_right_env_cfg.py")
    assert "cup/cup_big_sdf.usd" in pour
    assert "cup_big_rl.usd" not in pour


def test_pour_sensor_is_untouched_by_this_fork() -> None:
    """RA-L 대조군이 살아 있어야 구/신 비교가 가능하다."""
    sensor = TASK_DIR.parent / "pour_sensor" / "pour_right_env_cfg.py"
    text = sensor.read_text()
    assert "openarm_tesollo_sensor_rl" in text, "pour_sensor 가 오염됐다"
    assert "kinematic_enabled=True" in text, "pour_sensor 왼컵이 오염됐다"
    assert "left_warm_state_paths" not in text


# ---------------------------------------------------------------------------
# 비드 파이프라인 — "비드가 든 컵을 쥔 채 시작"
#   pour 의 손은 warm 자세로 **동결된 수동 스프링**(재조임 없음)이고 강성이 5.0 이다.
#   빈 컵 파지를 텔레포트한 뒤 비드를 넣으면 하중을 흡수하지 못해 컵을 놓칠 수 있다.
#   그래서 비드는 **수집 시점**에 채우고, 그 상태를 warm HDF5 로 넘겨 pour 가 복원한다.
# ---------------------------------------------------------------------------
_GRASP_R_ENV = TASK_DIR.parents[1] / "right" / "grasp_v1" / "grasp_right_env.py"
_GRASP_R_CACHE = TASK_DIR.parents[1] / "right" / "grasp_v1" / "warm_state_cache.py"
_COLLECT_CLI = TASK_DIR.parents[5] / "scripts" / "warm_states" / "collect_grasp_v1_warm_states.py"


def test_bead_definition_is_shared_between_collect_and_pour() -> None:
    """비드 물성이 양쪽에서 갈리면 같은 좌표를 복원해도 동역학이 달라진다."""
    pour = _code("pour_right_env_cfg.py")
    grasp = _code(str(_GRASP_R_ENV.parent / "grasp_right_env_cfg.py"))
    assert "openarm.common.bead_assets" in pour
    assert "openarm.common.bead_assets" in grasp
    # pour 안에 비드 물성이 재정의돼 있으면(fork 되면) 정합이 깨진다
    assert "static_friction=0.1" not in pour, "비드 물성이 pour 에 되돌아왔다"


def test_grasp_collection_bead_flag_defaults_off() -> None:
    """학습에는 영향이 없어야 한다 (수집 CLI 에서만 켠다)."""
    grasp = _nows(_code(str(_GRASP_R_ENV.parent / "grasp_right_env_cfg.py")))
    assert "collect_with_beads:bool=False" in grasp


def test_grasp_exports_bead_state_and_robot_usd() -> None:
    env = _code(str(_GRASP_R_ENV))
    cache = _code(str(_GRASP_R_CACHE))
    # 비드 상태를 export 에 넘기는가
    assert "bead_state=" in env
    assert "_bead_state_env_local" in env
    # 캐시가 선택 데이터셋으로 기록하는가
    assert 'create_dataset("bead_state"' in cache
    # 자산 출처도 함께 남겨야 이번 USD 사고가 재발하지 않는다
    assert '"robot_usd"' in env


def test_pour_restores_bead_state_and_skips_delayed_spawn() -> None:
    env = _code("pour_right_env.py")
    bank = _code("warm_state_bank.py")
    assert "bead_state" in bank, "warm bank 가 bead_state 를 모른다"
    assert "_warmstart_bead_state" in env
    # 복원 시 재소환을 막아야 한다 (이미 컵 안에 있음)
    assert "self._beads_spawned[env_ids] = True" in env
    # 없으면 기존 경로로 degrade
    assert "self._hide_beads(env_ids)" in env


def test_bead_count_mismatch_degrades_loudly() -> None:
    """개수가 다르면 조용히 잘못 복원하지 말고 기존 경로로 내려가야 한다."""
    env = _code("pour_right_env.py")
    assert "비드 개수 불일치" in env
    bank = _code("warm_state_bank.py")
    assert "비드 개수 불일치" in bank


def test_collect_cli_exposes_with_beads() -> None:
    cli = _code(str(_COLLECT_CLI))
    assert "--with_beads" in cli
    assert "env.collect_with_beads=true" in cli
    # 좌/우 프리셋이 있어야 순차 수집이 된다
    assert "tesollo_right" in cli
    assert "tesollo_left" in cli


# ---------------------------------------------------------------------------
# 컵 다양화 — cyl 제거(치환) + source/receiver 스케일 DR
# ---------------------------------------------------------------------------
def test_grasp_objects_are_cups_only_and_count_preserved() -> None:
    """cyl 3종을 **컵 변형으로 치환**했다 — 삭제가 아니다.

    `multi_object_idx_onehot` 폭이 `len(_object_names)` 에서 파생되어 물체 수가 곧
    actor obs 차원(146 = 138 + onehot 8)이다. 삭제하면 obs 146→143 이 되어 학습된
    체크포인트가 무효가 된다. 8종을 유지하면 obs 불변 + 컵 다양성 4→7 증가.
    """
    import re

    for side, fname in (("right", "grasp_right_env_cfg.py"), ("left", "grasp_left_env_cfg.py")):
        cfg = _code(str(TASK_DIR.parents[1] / side / "grasp_v1" / fname))
        ids = re.findall(r'"id":\s*"([^"]+)"', cfg)
        assert len(ids) == 8, f"{side}: 물체 {len(ids)}종 (8이어야 obs 불변)"
        assert not any("cyl" in i for i in ids), f"{side}: cyl 잔존 {ids}"
        # pour source 기준 spec 1 이 s100 이어야 매핑이 유지된다
        assert ids[1] == "cup_big_s100", f"{side}: spec1={ids[1]} (s100 이어야)"
        # shaker 는 **바닥 플러그를 넣은 사본**이어야 비드를 담는다 (원본은 양쪽 뚫린 관)
        assert ids[4] == "shaker_closed", f"{side}: spec4={ids[4]} (shaker_closed 이어야)"
        cfg_raw = _read(str(TASK_DIR.parents[1] / side / "grasp_v1" / fname))
        assert "shaker_closed_rl.usd" in cfg_raw
        assert "shaker_body_rl.usd" not in _code(str(TASK_DIR.parents[1] / side / "grasp_v1" / fname))


def test_pour_v1_cup_scale_dr_enabled_both_sides() -> None:
    cfg = _nows(_code("pour_right_env_cfg.py"))
    assert "source_cup_scale_set:tuple[float,...]=(0.85,1.0,1.15,1.30)" in cfg
    assert "left_target_cup_scale_set:tuple[float,...]=(0.85,1.0,1.15,1.30)" in cfg
    # 스케일↔grasp spec 매핑이 좌우 모두 있어야 파지자세가 컵 크기와 맞는다
    assert "source_warm_spec_map:tuple[int,...]=(0,1,2,3)" in cfg
    assert "left_warm_spec_map:tuple[int,...]=(0,1,2,3)" in cfg


def test_left_warm_sampling_matches_receiver_scale() -> None:
    """receiver 컵이 물리 파지로 바뀌었으므로 스케일-파지자세 매칭이 필수다."""
    env = _code("pour_right_env.py")
    assert "_tgt_spec_env" in env
    assert "_left_warm_spec_pools" in env
    # 미태깅 캐시 + scale_set 조합은 조용히 통과시키면 안 된다
    assert "미태깅(object_spec_idx 전부 -1)" in env


def test_left_spec_filter_disabled_under_scale_set() -> None:
    """receiver 컵 scale_set 을 쓰면 좌팔 단일 spec 필터를 **꺼야** 한다.

    필터가 spec 1 만 남기면 spec 풀 구성이 0/2/3 을 못 찾아 로드가 실패한다.
    우팔은 같은 규약을 이미 갖고 있었는데 좌팔에 빠져 있었고, E0-3 공존 게이트가
    실제로 이 버그를 잡았다(ValueError: 왼팔 warm 캐시에 spec 0 상태가 없다).
    """
    env = _nows(_code("pour_right_env.py"))
    # 좌/우 모두 scale_set 활성 시 필터를 비운다
    assert "ifself._tgt_spec_envisnotNone:_lf=()" in env
    assert "ifself._src_spec_envisnotNone:_spec_filter=()" in env


# ---------------------------------------------------------------------------
# 좌/우 컵 겹침 방지 (2026-08-18)
# ---------------------------------------------------------------------------
# 근본원인: grasp 리프트가 joint7 만 0.31rad 돌려 잡은 컵을 몸쪽으로 스윙시킨다.
#   좌/우 뱅크를 독립 샘플링해 합치면 두 컵이 겹치고, PhysX 가 침투를 밀어내며
#   파지를 뜯어낸다(스폰 ∓0.10 실측: 겹침 44.3%, 접촉 0.81개, 64env 중 11 완주).
# 대책 2단: (1) grasp 스폰을 ∓0.20 으로 벌림  (2) 남은 꼬리는 리셋 시 재추첨.

def _grasp_cfg_code(side: str) -> str:
    """grasp cfg 원문(주석 제거). `_code` 는 TASK_DIR 기준 상대경로를 받는다."""
    path = _GRASP_R_CFG if side == "right" else _GRASP_L_CFG
    assert path.is_file(), f"{path} 없음"
    return _code(os.path.relpath(path, TASK_DIR))


def test_grasp_spawn_y_is_separated_for_bimanual():
    """좌/우 스폰 y 가 ∓0.20 이어야 한다 — 겹침 0.05% 를 만든 실측값."""
    assert _nows("object_spawn_y_center: float = -0.20") in _nows(_grasp_cfg_code("right"))
    assert _nows("object_spawn_y_center: float = 0.20") in _nows(_grasp_cfg_code("left"))


def test_redraw_is_wired_into_both_warm_reset_paths():
    """warmstart cache 와 deep-tilt boot **둘 다** 재추첨을 호출해야 한다.

    한쪽만 걸면 그 경로에서만 겹침이 살아남아 원인 추적이 어려워진다.
    """
    code = _code("pour_right_env.py")
    assert code.count("self._redraw_overlapping_pairs(") == 2, (
        "재추첨 호출이 2곳(_reset_from_warmstart_cache, _reset_from_deep_tilt_bank)이 아니다"
    )
    assert "def _redraw_overlapping_pairs(" in code


def test_redraw_uses_per_env_scaled_radii():
    """겹침 판정에 **per-env 스케일 반영 반경**을 써야 한다.

    컵 스케일이 (0.85~1.30) 로 섞이므로 nominal 반경으로 재면 큰 컵의 겹침을 놓친다.
    """
    code = _code("pour_right_env.py")
    body = code.split("def _redraw_overlapping_pairs(")[1].split("\n    def ")[0]
    assert "_src_inner_r_env" in body and "_tgt_inner_r_env" in body
    assert "cup_wall_thickness_m" in body, "내부 반경만 쓰면 컵 벽 두께를 빠뜨린다"


def test_redraw_has_bounded_retries():
    """재추첨은 유한 시도여야 한다(무한 루프 금지) + 미해소 시 침묵하지 않아야."""
    code = _code("pour_right_env.py")
    body = code.split("def _redraw_overlapping_pairs(")[1].split("\n    def ")[0]
    assert "left_right_cup_redraw_tries" in body
    assert "while " not in body, "무한 루프 위험 — for + tries 로 제한할 것"
    assert "WARN" in body, "미해소분을 조용히 넘기면 안 된다"


def test_j1_yaw_spread_is_removed():
    """j1 회전 보정은 **제거**했다 — 비활성만 두면 누군가 값을 올린다.

    실측: `l_aj_1 +0.15` → Δpalm=[-0.057, **0.000**, -0.034]. j1 은 베이스 요가 아니다
    (URDF axis 는 joint-local, 부모 마운트 rpy=(-pi/2,0,0)). 관절은 j1 으로 돌리고
    좌표는 z축 회전으로 옮겨 손과 컵이 분리됐다. 되살리려면 IK 로 palm pose 를 풀 것.
    """
    for name in ("pour_right_env.py", "pour_right_env_cfg.py"):
        code = _code(name)
        assert "warm_arm_yaw_spread_rad" not in code, f"{name}: 폐기된 cfg 필드가 남아 있다"
        assert "_spread_warm_bank_yaw" not in code, f"{name}: 폐기된 회전 함수가 남아 있다"


def test_pour_target_is_not_reanchored_to_measured_pose():
    """붓기 목표는 **실측 pose 에 재앵커하면 안 된다** (2026-08-18).

    구 pour_sensor: `pour_point_target = rim_env + delta` — `rim_env` 가 실측 주둥이라,
    팔이 하중으로 뒤처지면 목표가 그 뒤처진 위치에 다시 앵커된다 → 지연이 **속도**로
    바뀌어 위치가 선형 누적된다(관절 괴리는 0.04rad 포화인데 컵 x 는 0.33→0.077,
    zero-action out_x 사망 104/128 · 생존 0).

    `grasp_v1` 은 같은 fabric 을 쓰면서 목표를 절대 앵커(`pregrasp_palm_pose_buf + delta`)로
    잡아 이 루프가 없다. pour 는 20cm 이상 이동해야 하므로 고정 앵커 대신 **action 만
    적분하는 명령 상태**(`_cmd_spout_env`)를 쓴다 — plant 를 되읽지 않는다.

    ⚠ `fabric_q` 를 매 스텝 실제 관절로 동기화하는 대안도 시도했으나, fabric 이 앞서
      적분할 여지가 없어 **팔이 명령을 전혀 못 따라갔다**(+y 3cm 명령 280스텝에 palm
      이동 −11mm, 이동률 0.001). 폭주를 부동으로 바꾼 셈이라 폐기했다.
      fabric 은 open-loop 로 두고 목표만 고치는 것이 맞다.
    """
    env = _code("pour_right_env.py")
    n = _nows(env)
    assert _nows("pour_point_target = self._cmd_spout_env") in n, \
        "붓기 목표가 명령 상태에서 오지 않는다"
    assert _nows("pour_point_target = rim_env + delta") not in n, \
        "실측 주둥이에 재앵커하는 구 방식이 남아 있다 (드리프트 재발)"
    assert "_cmd_spout_valid[env_ids] = False" in env, \
        "리셋에서 명령 상태를 무효화하지 않으면 이전 에피소드 목표가 새 에피소드로 샌다"
    # fabric 은 open-loop 여야 한다 — 동기화하면 팔이 못 움직인다(위 ⚠ 참조).
    assert _nows("self.fabric_q[:, :NUM_ARM_DOF] = self.robot.data.joint_pos") not in n, \
        "fabric_q 를 실제 관절로 동기화하면 위치 명령 추종이 죽는다"

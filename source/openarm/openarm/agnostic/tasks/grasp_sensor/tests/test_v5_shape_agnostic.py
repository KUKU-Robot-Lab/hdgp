"""v5 형상 비의존 재설계 계약 (08.26).

전부 소스 텍스트 검사 — Isaac 불요. 목적: "다양한 컵으로 바꿔도 잡는 정책"의
구조적 전제(물체 상수 0 · obs 무형상 · 단일 소스 · 커리큘럼 분리)를 회귀로 고정한다.
"""

from __future__ import annotations

import re
from pathlib import Path

_TASK_DIR = Path(__file__).resolve().parent.parent
_CFG = (_TASK_DIR / "grasp_sensor_env_cfg.py").read_text(encoding="utf-8")
_ENV = (_TASK_DIR / "grasp_sensor_env.py").read_text(encoding="utf-8")
_REW = (_TASK_DIR / "rewards_tip_cyl.py").read_text(encoding="utf-8")


def _code(src: str) -> str:
    """주석 제거 — 주석 속 옛 상수 언급은 위반이 아니다."""
    return "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("#"))


def test_no_shape_constants_in_cfg():
    """특정 물체(cup_big)의 치수·원점·물림점 상수가 cfg 에 없어야 한다."""
    code = _code(_CFG)
    for banned in ("stage_gc_local_override", "stage_gc_opposition_frac",
                   "object_origin_offset_z", "object_spawn_z",
                   "tip_r_center", "side_radius", "grasp_z_offset",
                   "stage_success_envelope_min", "envelope_gate_saturation",
                   "0.0773"):
        assert banned not in code, f"형상 의존 상수 잔재: {banned}"


def test_object_bank_is_single_source():
    """물체 정보는 object_bank 단일 소스 — cfg 가 import·파생하고 env 가 가드한다."""
    assert "from openarm.agnostic.modules import object_bank" in _CFG
    code = _code(_CFG)
    assert re.search(r"object_bank:\s*str\s*=", code), "object_bank 필드 부재"
    assert "bank.rigid_body_name" in code, "접촉 필터가 뱅크에서 파생되지 않는다"
    assert '"baseLink"' not in code, "rigid body 이름 하드코딩 잔재"
    assert "requires_replicate_physics_off" in code, "replicate_physics 파생 부재"
    env_code = _code(_ENV)
    assert "assert_spawned_after_clone" in env_code, "clone 순서 가드 부재"
    assert re.search(r"clone_environments[\s\S]{0,900}RigidObject\(self\.cfg\.object_cfg\)",
                     env_code), "물체 생성이 clone 이후가 아니다"
    assert "_object_rest_z" in env_code, "per-spec rest_z 텐서 부재"


def test_resolve_cfg_called_in_env_init():
    """hydra 오버라이드 함정 — env __init__ 이 resolve_cfg 를 재호출해야 한다."""
    assert re.search(r"def resolve_cfg\(", _code(_CFG)), "resolve_cfg 자유함수 부재"
    m = re.search(r"def __init__\(self, cfg[\s\S]{0,700}?super\(\).__init__", _ENV)
    assert m and "resolve_cfg(cfg)" in m.group(0), (
        "env __init__ 이 super() 이전에 resolve_cfg 를 부르지 않는다")


def test_obs_has_no_object_identity():
    """policy obs 에 물체 정체성(scale·onehot·치수·질량)이 없어야 한다(s2r + 일반화)."""
    code = _code(_CFG)
    assert "onehot" not in code, "onehot 이 cfg 에 생겼다 — obs 오염 경로"
    m = re.search(r"cfg\.observation_space\s*=\s*\(([\s\S]*?)\)", code)
    assert m, "observation_space 식 부재"
    assert "bank" not in m.group(1) and "scale" not in m.group(1), (
        "observation_space 가 물체 뱅크에서 파생된다")


def test_obs_carries_issued_palm_command():
    """policy obs 에 리미터 **통과 후** palm 지령 6D 가 있어야 한다.

    `actions` 는 리미터 전 요청이라, 상한이 물리는 스텝에서는 실제로 내려간 지령과
    다르다. 절대 목표 규약에서 정책이 직전 지령을 모르면 과거·현재를 비교할 수 없다.
    """
    env_code = _code(_ENV)
    m = re.search(r"obs = torch\.cat\(\[([\s\S]*?)\], dim=1\)", env_code)
    assert m, "obs 결합식 부재"
    assert "palm_cmd_n" in m.group(1), "obs 에 palm 지령이 없다"
    # 지령은 액션과 같은 정규화 좌표여야 둘을 직접 견줄 수 있다.
    assert re.search(
        r"palm_cmd_n\s*=\s*\(2\.0 \* \(self\.palm_targets - self\._palm_lo\)",
        env_code), "palm 지령이 액션과 같은 [-1,1] 박스 정규화가 아니다"
    cfg_code = _code(_CFG)
    m = re.search(r"cfg\.observation_space\s*=\s*\(([\s\S]*?)\)", cfg_code)
    assert "+ 6" in m.group(1).replace("\n", " "), (
        "observation_space 에 지령 6D 가 반영되지 않았다")


def test_gc_and_gate_are_hand_derived():
    """파지중심·λ 임계는 부팅 FK 손 기하 파생 — 물체 실측 상수 금지."""
    env_code = _code(_ENV)
    assert "_setup_grasp_geometry" in env_code
    assert "_r_cage" in env_code, "케이지 반경 파생 부재"
    assert "stage_gate_approach_scale" in env_code, "λ 임계가 scale 파생이 아니다"
    cfg_code = _code(_CFG)
    m = re.search(r"stage_gate_approach_m:\s*float\s*=\s*([0-9.]+)", cfg_code)
    assert m and float(m.group(1)) == 0.0, (
        "stage_gate_approach_m 기본값이 0.0(부팅 파생 신호)이 아니다")


def test_z_dead_default_identity():
    """공유 수식의 z 데드밴드는 getattr 기본 0.0 — 자매 트랙에서 구식과 항등."""
    code = _code(_REW)
    assert re.search(r'getattr\(cfg,\s*"stage_approach_z_dead",\s*0\.0\)', code), (
        "z_dead getattr 기본값이 0.0 이 아니다 — 자매 하위호환 파손")
    assert re.search(r"relu\(_dz_gc\s*-\s*_zdead\)", code), "수직 힌지 수식 부재"
    m = re.search(r"stage_approach_z_dead:\s*float\s*=\s*([0-9.]+)", _code(_CFG))
    assert m and float(m.group(1)) > 0.0, "이 트랙에서 z_dead 가 활성이 아니다"


def test_wrap_deep_share_denominator():
    """wrap 과 deep 의 분모가 같아야 한다(가용 손가락) — 4 vs 3 혼합 왜곡 재발 방지."""
    m_w = re.search(r"wrap_c=\([^\)]*\)\[:, self\.(\w+)\]", _ENV)
    m_d = re.search(r"deep_c=\([^\)]*\)\[:, self\.(\w+)\]", _ENV)
    assert m_w and m_d, "wrap_c/deep_c 배선 부재"
    assert m_w.group(1) == m_d.group(1) == "_wrap_idx", (
        f"분모 불일치: wrap={m_w.group(1)} deep={m_d.group(1)}")


def test_lift_success_single_source():
    """승급 판정 리터럴 금지 — stage_gate_lift_m·success_envelope_min 단일 소스."""
    m = re.search(r"self\._lift_success_now = \([\s\S]*?\n        \)", _ENV)
    assert m, "_lift_success_now 블록 부재"
    blk = m.group(0)
    assert "stage_gate_lift_m" in blk and "success_envelope_min" in blk, (
        "승급 판정이 cfg 단일 소스를 안 쓴다")
    assert "0.05" not in _code(blk), "승급 판정에 리터럴 0.05 잔재"
    assert "_corridor_latch" in blk, "승급에 코리더 무위반 요구가 빠졌다"


def test_start_curriculum_contract():
    """접근 거리 역커리큘럼(§1.8) — cspace rest 불변·만렙=홈 항등·fail-loud 파생."""
    cfg_code = _code(_CFG)
    assert re.search(r"start_pose_frac:\s*tuple", cfg_code), "start_pose_frac 부재"
    env_code = _code(_ENV)
    assert "_setup_start_curriculum" in env_code
    assert "_q_pregrasp_arm" in env_code
    # 리셋이 fabric cspace rest(default_config)를 건드리면 도달영역이 바뀐다.
    m = re.search(r"def _reset_idx[\s\S]*?write_root_state_to_sim", _ENV)
    assert m and "default_config" not in _code(m.group(0)), (
        "_reset_idx 가 fabric cspace rest 를 건드린다")
    # 만렙(start_frac=1)은 lerp(q_pre, q_home, 1) = q_home — 구 리셋과 항등 구조.
    assert re.search(r"torch\.lerp\(\s*self\._q_pregrasp_arm", _ENV), (
        "시작 자세가 lerp(프리그래스프, 홈) 구조가 아니다")
    # 지령 버퍼 씨딩 — 절대 액션 매핑의 짝(상태만 옮기면 첫 스텝에 박스 중심으로 튐).
    assert "_prev_palm_cmd[env_ids] = _seed[:, :3]" in _ENV, "지령 버퍼 씨딩 부재"


def test_hand_control_is_synergy_only():
    """v5: 손 제어 경로는 synergy 단일 — 죽은 분기·모드 스위치 잔재 금지."""
    for banned in ("hand_control", "self._hand_fabric", "tip_per_finger=self",
                   "compute_grasp_sensor_rewards", "hand_targets"):
        assert banned not in _code(_ENV), f"구 손 제어 경로 잔재: {banned}"
    assert "hand_control" not in _code(_CFG), "cfg 에 hand_control 스위치 잔재"

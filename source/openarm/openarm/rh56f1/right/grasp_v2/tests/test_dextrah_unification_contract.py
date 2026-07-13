from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _text(filename: str) -> str:
    return (ROOT / filename).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1) finger_curl_reg 기준 (77357f0 이식): HAND_FULL_GRIP_POSE 로 되돌아가면
#    "빈손 주먹=무패널티, 물체 잡으면 페널티" 역유인이 재발한다.
# ---------------------------------------------------------------------------
def test_curl_reg_anchors_on_open_pose_not_full_grip() -> None:
    env = _text("grasp_right_env.py")
    body = env.split("# 4) finger_curl_reg", 1)[1].split("# 5) palm orientation", 1)[0]

    assert "self.hand_open_pose" in body, "curl 기준이 open_pose 여야 한다"
    assert "self.hand_full_grip_pose" not in body, (
        "curl 기준이 FULL_GRIP 으로 회귀 — 물체 회피가 최적인 역유인 재발"
    )
    assert "clamp(max=float(self.cfg.finger_curl_dist_max))" in body, (
        "발산 방지 clamp 누락 — 물리 발산 시 제곱 증폭으로 리턴 폭주 위험"
    )
    cfg = _text("grasp_right_env_cfg.py")
    assert "finger_curl_dist_max:     float = 14.0" in cfg


# ---------------------------------------------------------------------------
# 2) 접근 자세 분기 (cd29c62 이식): 이름 기반 고정 분기. 물체 높이/회전 기반
#    규칙(78592a3, rh56f1에 먼저 넣었던 것)은 ADR 회전이 커지면 스스로 꺼지는
#    자기모순이 있어 폐기했다 — 잔재가 남아있으면 안 된다.
# ---------------------------------------------------------------------------
def test_approach_branch_is_name_based_not_height_based() -> None:
    env = _text("grasp_right_env.py")
    utils = _text("grasp_right_utils.py")
    preset = _text("grasp_right_preset.py")
    cfg = _text("grasp_right_env_cfg.py")

    assert "compute_palm_pose_id" in env
    assert "compute_palm_pose_id" in utils
    assert 'SIDE_APPROACH_OBJECT_NAMES = ("cup",)' in preset
    assert "side_approach_object_names" in cfg

    # 폐기된 높이 기반 분기 잔재(실제 코드)가 남아있으면 안 된다 — 주석 속 역사적
    # 언급("구 compute_flat_object_mask 는 ~")은 정상 문서화이므로 def 기준으로 확인.
    assert "def compute_flat_object_mask" not in utils
    assert "compute_flat_object_mask(" not in env  # 호출부
    assert "FLAT_OBJECT_HEIGHT_THRESHOLD" not in cfg
    assert "FLAT_OBJECT_HEIGHT_THRESHOLD" not in preset
    assert "_compute_topdown_mask" not in env


def test_approach_branch_topdown_is_default() -> None:
    # side(cup)만 90°, 나머지 전부 top-down(180°) — cd29c62 반전 방향 확인.
    utils = _text("grasp_right_utils.py")
    body = utils.split("def compute_palm_pose_id", 1)[1]
    assert "torch.zeros_like(object_idx)" in body  # side → 0
    assert "torch.ones_like(object_idx)" in body   # 그 외(top-down) → 1
    # is_side 가 True 인 분기가 0(side)을 반환해야 하며, 기본(else)이 1(top-down).
    assert body.index("torch.zeros_like(object_idx)") < body.index("torch.ones_like(object_idx)")


# ---------------------------------------------------------------------------
# 3) pregrasp offset — clearance 비례 (9f0e4f7 이식): 고정 offset 회귀 시
#    ADR 회전 상승에서 PhysX depenetration 폭주 잠복 위험 재발.
# ---------------------------------------------------------------------------
def test_pregrasp_offset_is_clearance_based_not_fixed() -> None:
    env = _text("grasp_right_env.py")
    cfg = _text("grasp_right_env_cfg.py")
    preset = _text("grasp_right_preset.py")

    assert "_compute_pregrasp_offset" in env
    assert "object_clearance" in env
    assert "self.object_clearance[obj_idx]" in env

    # 폐기된 고정 offset 테이블/IK 캐시 잔재.
    assert "pregrasp_offset_by_pose" not in env
    assert "PREGRASP_OFFSET_TOPDOWN" not in env
    assert "PREGRASP_OFFSET_TOPDOWN" not in preset
    assert "cache_pregrasp_reset" not in cfg
    assert "_cache_q_arm" not in env
    assert "_build_pregrasp_cache" not in env


# ---------------------------------------------------------------------------
# 4) palm action — 절대 pose (1aa9dcc 이식): anchor+delta 회귀 시 credit
#    assignment 붕괴("가만히 있기" 국소최적) 재발.
# ---------------------------------------------------------------------------
def test_palm_action_is_absolute_pose_not_delta() -> None:
    cfg = _text("grasp_right_env_cfg.py")
    env = _text("grasp_right_env.py")

    assert "palm_rate_xyz_per_step:     float = 0.04" in cfg
    assert "palm_rate_rot_deg_per_step: float = 8.0" in cfg
    assert "palm_delta_xyz" not in cfg, "delta cfg 가 되살아났다"
    assert "palm_delta_rot_deg" not in cfg

    assert "scale(palm_action, self.palm_mins_env, self.palm_maxs_env)" in env, (
        "palm action 이 절대 pose 가 아님 (delta 로 되돌아갔다)"
    )
    assert "self.delta_mins" not in env
    assert "self.delta_maxs" not in env
    assert "self.palm_rate_limits" in env
    # rate limit: 목표는 절대 pose, 실제 이동은 palm_pose_targets 를 rate 이내로.
    assert "(palm_pose - self.palm_pose_targets).clamp(" in env


# ---------------------------------------------------------------------------
# 5) G-frame 은 의도적으로 이식하지 않음 — RH56F1 실측(probe)으로 tesollo 의
#    "가짜 top-down" 프레임버그가 없음을 확인했기 때문(local+Z=법선, Allegro
#    규약과 이미 일치). 향후 실수로 이식되면 안 되는 표시자만 확인.
# ---------------------------------------------------------------------------
def test_g_frame_intentionally_not_ported() -> None:
    env = _text("grasp_right_env.py")
    utils = _text("grasp_right_utils.py")
    assert "g_pose_to_fabric_quat" not in env
    assert "g_pose_to_fabric_quat" not in utils
    assert "PALM_GRASP_FRAME_ROT" not in _text("grasp_right_preset.py")

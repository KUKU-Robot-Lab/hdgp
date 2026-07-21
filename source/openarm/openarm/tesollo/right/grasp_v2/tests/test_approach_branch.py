"""접근 자세 분기(cup=side, 그 외 top-down) + abduction 자유화 검증.

분기가 틀리면 접근 자세가 통째로 뒤집히고, abduction 부호가 틀리면 자기충돌
검사가 꺼진 상태에서 손가락이 서로 관통하므로 둘 다 명시적으로 고정한다.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

PKG = Path(__file__).resolve().parents[1]
LEFT_PKG = PKG.parents[1] / "left" / "grasp_v2"
REPO = PKG.parents[5]   # …/hdgp


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


r_utils = _load("_branch_r_utils", PKG / "grasp_right_utils.py")
l_utils = _load("_branch_l_utils", LEFT_PKG / "grasp_left_utils.py")
r_preset = _load("_branch_r_preset", PKG / "grasp_right_preset.py")
l_preset = _load("_branch_l_preset", LEFT_PKG / "grasp_left_preset.py")
synergy = _load("_branch_synergy", PKG / "tesollo_hand_synergy.py")


# ---------------------------------------------------------------------------
# 접근 자세 분기 — cup 만 side(0), 나머지 전부 top-down(1)
# ---------------------------------------------------------------------------
def test_cup_is_side_everything_else_topdown():
    names = ["cup", "small_8_cyl", "small_8_cuboid", "large_5_cyl", "75443"]
    side_idx = torch.tensor([names.index("cup")], dtype=torch.long)
    obj_idx = torch.arange(len(names), dtype=torch.long)

    pose_id = r_utils.compute_palm_pose_id(obj_idx, side_idx)

    assert pose_id.tolist() == [0, 1, 1, 1, 1]


def test_branch_does_not_depend_on_spawn_rotation():
    """물체 이름 기반이라 ADR 회전이 커져도 분기가 흔들리지 않는다.

    lstm_test2 의 높이 규칙은 원통이 누우면 "납작"에서 빠져 topdown_frac 이
    0.025 → 0.0015 로 자멸했다. 회전이 입력에 아예 없음을 고정한다.
    """
    side_idx = torch.tensor([0], dtype=torch.long)
    obj_idx = torch.tensor([0, 1, 1, 1], dtype=torch.long)

    first = r_utils.compute_palm_pose_id(obj_idx, side_idx)
    second = r_utils.compute_palm_pose_id(obj_idx, side_idx)

    assert torch.equal(first, second)
    assert first.tolist() == [0, 1, 1, 1]


def test_left_mirror_uses_same_branch_rule():
    side_idx = torch.tensor([0], dtype=torch.long)
    obj_idx = torch.tensor([0, 1], dtype=torch.long)

    assert l_utils.compute_palm_pose_id(obj_idx, side_idx).tolist() == [0, 1]


def test_side_object_names_are_cups():
    # cup + cup_big(pour 실물컵) 은 세로 원통이라 side approach 로 감싸 잡는다. 좌우 동일.
    assert r_preset.SIDE_APPROACH_OBJECT_NAMES == ("cup", "cup_big")
    assert l_preset.SIDE_APPROACH_OBJECT_NAMES == ("cup", "cup_big")


# ---------------------------------------------------------------------------
# grasp 프레임 (G) — 진짜 top-down 이 나오는가
# ---------------------------------------------------------------------------
def _R_world_palm(preset, utils, g_euler_deg):
    """G 규약 euler → 실제 palm 회전행렬 (fabric 에 넘어가는 quaternion 을 되풀어서)."""
    C = torch.tensor(preset.PALM_GRASP_FRAME_ROT)
    pose = torch.tensor([[0.0, 0.0, 0.0] + [math.radians(v) for v in g_euler_deg]])
    q = utils.g_pose_to_fabric_quat(pose, C)[0, 3:]          # xyzw
    x, y, z, w = q
    return torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def test_grasp_frame_is_proper_rotation():
    for preset in (r_preset, l_preset):
        C = torch.tensor(preset.PALM_GRASP_FRAME_ROT)
        assert torch.allclose(C.T @ C, torch.eye(3), atol=1e-6)
        assert abs(float(torch.det(C)) - 1.0) < 1e-6


def test_topdown_actually_faces_the_table():
    """핵심 회귀 테스트 — lstm_test3 의 "가짜 top-down" 재발 방지.

    palm 법선(로컬 +X)이 world -Z 여야 손바닥이 테이블을 본다. 실패한 구현은
    법선이 (0,+1,0) 수평이고 손가락만 아래로 꽂혀 있었다(실측 확인).
    """
    for preset, utils in ((r_preset, r_utils), (l_preset, l_utils)):
        R = _R_world_palm(preset, utils, preset.PREGRASP_G_EULER_TOPDOWN)
        normal, finger = R[:, 0], R[:, 2]
        assert normal[2] < -0.99, f"법선이 아래를 안 봄: {normal.tolist()}"
        assert abs(float(finger[2])) < 0.01, f"손가락이 수평이 아님: {finger.tolist()}"


def test_topdown_normal_is_invariant_to_heading():
    """ez(손가락 방위각)를 아무리 돌려도 법선은 -Z 를 유지해야 한다."""
    for ez in (-135.0, -45.0, 0.0, 45.0, 90.0, 180.0):
        R = _R_world_palm(r_preset, r_utils, [ez, 0.0, 180.0])
        assert R[2, 0] < -0.99, f"ez={ez} 에서 법선이 무너짐"


def test_side_pose_matches_legacy_exactly():
    """cup(side) 자세는 G 규약 전환 후에도 기존과 완전히 같아야 한다 (회귀 방지).

    기존: right (ez=90, ey=0, ex=90) / left (ez=-90, ey=0, ex=-90) — P 규약 euler.
    """
    for preset, utils, legacy in (
        (r_preset, r_utils, (90.0, 0.0, 90.0)),
        (l_preset, l_utils, (-90.0, 0.0, -90.0)),
    ):
        R_new = _R_world_palm(preset, utils, preset.PREGRASP_G_EULER_SIDE)
        R_old = r_utils.euler_zyx_to_matrix(
            torch.tensor([[math.radians(v) for v in legacy]])
        )[0]
        assert torch.allclose(R_new, R_old, atol=1e-5), \
            f"side 자세가 바뀜:\n{R_new}\nvs\n{R_old}"


def test_pregrasp_lies_inside_its_own_palm_box():
    """경계가 pregrasp 자세를 잘라내면 안 된다 — 이 사고가 이미 두 번 났다.

    (1) left lstm_test1: +90 하드코드가 left 경계에 0° 로 clamp → 90° 뒤틀림
    (2) lstm_test3: 진짜 top-down 에 필요한 ey=90 이 경계 [-45,45] 밖
    각 pregrasp 가 자기 박스 안에 5° 이상 여유를 두고 들어있는지 고정한다.
    """
    MPA = 45.0          # cfg.max_pose_angle
    MARGIN = 5.0
    for preset in (r_preset, l_preset):
        for pose_name, pregrasp, center in (
            ("side", preset.PREGRASP_G_EULER_SIDE, preset.PALM_G_EULER_CENTER_SIDE),
            ("top-down", preset.PREGRASP_G_EULER_TOPDOWN, preset.PALM_G_EULER_CENTER_TOPDOWN),
        ):
            lo = preset.palm_pose_mins(MPA, center)
            hi = preset.palm_pose_maxs(MPA, center)
            for k in range(3):
                v = math.radians(pregrasp[k])
                assert lo[3 + k] + math.radians(MARGIN) <= v <= hi[3 + k] - math.radians(MARGIN), \
                    f"{pose_name} euler[{k}] 가 경계에 붙었다: {math.degrees(v):.1f}° " \
                    f"∈ [{math.degrees(lo[3+k]):.1f}, {math.degrees(hi[3+k]):.1f}]"


def test_quaternion_roundtrip_matches_euler_path():
    """quaternion 경로가 euler 경로와 같은 회전을 내는지 (비특이 자세에서)."""
    C = torch.tensor(r_preset.PALM_GRASP_FRAME_ROT)
    for g in ([0.0, 0.0, 180.0], [30.0, -20.0, 160.0], [0.0, 0.0, -90.0]):
        pose = torch.tensor([[0.1, 0.2, 0.3] + [math.radians(v) for v in g]])
        R_direct = r_utils.euler_zyx_to_matrix(pose[:, 3:6])[0] @ C.T
        R_quat = _R_world_palm(r_preset, r_utils, g)
        assert torch.allclose(R_direct, R_quat, atol=1e-5)


# ---------------------------------------------------------------------------
# abduction 자유화
# ---------------------------------------------------------------------------
def test_abduction_ranges_stay_on_one_side_of_zero():
    """index 는 음수(right)/양수(left), pinky 는 양수(right)/음수(left) 로만 열린다.

    enabled_self_collisions=False 이므로 이 범위가 손가락 관통을 막는 유일한 방어선이다.
    thumb 두 축(벌림·대향)은 전 범위 — 엄지는 물체·자세에 따라 어디든 가야 한다.
    """
    # 축 순서: thumb_1(벌림), thumb_2(대향), index_1, pinky_1, pinky_2
    for preset, is_right in ((r_preset, True), (l_preset, False)):
        lo = preset.HAND_ABDUCTION_LIMITS_MIN
        hi = preset.HAND_ABDUCTION_LIMITS_MAX
        names = preset.HAND_ABDUCTION_JOINT_NAMES

        assert lo[0] < 0.0 < hi[0], names[0]          # thumb_1: 전 범위
        assert abs(hi[1] - lo[1]) > 3.0, names[1]     # thumb_2: 대향 전 범위(π)

        if is_right:
            assert lo[2] < 0.0 and math.isclose(hi[2], 0.0, abs_tol=1e-9), names[2]
        else:
            assert math.isclose(lo[2], 0.0, abs_tol=1e-9) and hi[2] > 0.0, names[2]

        for k in (3, 4):  # pinky_1, pinky_2
            if is_right:
                assert math.isclose(lo[k], 0.0, abs_tol=1e-9) and hi[k] > 0.0, names[k]
            else:
                assert math.isclose(hi[k], 0.0, abs_tol=1e-9) and lo[k] < 0.0, names[k]


def test_thumb_opposition_is_free():
    """엄지 대향(thumb_2)이 고정되면 top-down 파지가 원리적으로 불가능하다.

    -1.57 고정은 side 접근 전용 튜닝값이었고, 세 pose(APPROACH/GRASP/FULL_GRIP)에서
    값이 같아 시너지 진행도가 항상 0 → 정책이 절대 못 움직였다.
    그 결과 어떤 palm 높이에서도 물체를 못 들어올렸다(리프트 최선 +1.7cm, 대부분 음수).
    """
    for preset in (r_preset, l_preset):
        names = preset.HAND_ABDUCTION_JOINT_NAMES
        assert any("thumb_2" in n for n in names), "thumb_2(대향)가 자유화되지 않았다"
        i = [k for k, n in enumerate(names) if "thumb_2" in n][0]
        span = abs(preset.HAND_ABDUCTION_LIMITS_MAX[i] - preset.HAND_ABDUCTION_LIMITS_MIN[i])
        assert span > 3.0, f"thumb_2 범위가 {span:.2f} rad — 대향 전 범위(π)여야 한다"


def test_abduction_limits_are_left_right_mirrored():
    for i in range(4):
        assert math.isclose(
            l_preset.HAND_ABDUCTION_LIMITS_MIN[i],
            -r_preset.HAND_ABDUCTION_LIMITS_MAX[i],
            abs_tol=1e-6,
        )
        assert math.isclose(
            l_preset.HAND_ABDUCTION_LIMITS_MAX[i],
            -r_preset.HAND_ABDUCTION_LIMITS_MIN[i],
            abs_tol=1e-6,
        )


def test_abduction_local_indices_match_joint_names():
    """인덱스가 틀리면 엉뚱한 관절(예: thumb_2 curl)을 덮어써 파지가 깨진다."""
    for preset, joints in (
        (r_preset, r_preset.RIGHT_HAND_JOINT_NAMES),
        (l_preset, l_preset.LEFT_HAND_JOINT_NAMES),
    ):
        for local_idx, name in zip(
            preset.HAND_ABDUCTION_LOCAL_INDICES, preset.HAND_ABDUCTION_JOINT_NAMES
        ):
            assert joints[local_idx] == name


def test_abduction_action_maps_to_limits():
    lo = torch.tensor(r_preset.HAND_ABDUCTION_LIMITS_MIN)
    hi = torch.tensor(r_preset.HAND_ABDUCTION_LIMITS_MAX)

    at_min = r_utils.compute_abduction_targets(torch.full((1, 5), -1.0), lo, hi)
    at_max = r_utils.compute_abduction_targets(torch.full((1, 5), +1.0), lo, hi)
    at_mid = r_utils.compute_abduction_targets(torch.zeros(1, 5), lo, hi)

    assert torch.allclose(at_min[0], lo, atol=1e-6)
    assert torch.allclose(at_max[0], hi, atol=1e-6)
    assert torch.allclose(at_mid[0], 0.5 * (lo + hi), atol=1e-6)


def test_abduction_action_clamps_out_of_range_input():
    lo = torch.tensor(r_preset.HAND_ABDUCTION_LIMITS_MIN)
    hi = torch.tensor(r_preset.HAND_ABDUCTION_LIMITS_MAX)

    out = r_utils.compute_abduction_targets(torch.full((1, 5), 5.0), lo, hi)

    assert torch.allclose(out[0], hi, atol=1e-6)


def test_synergy_basis_cannot_drive_abduction_joints():
    """자유화가 왜 별도 action 축이어야 하는지 고정.

    시너지 진행도는 (q* - open)/(grip - open) 인데 세 pose 에서 값이 같은 관절
    (open == grip)은 스팬이 0 이라 진행도가 항상 0 이다. basis 에 성분이 있어도
    (thumb_2 는 PC5 에 0.598) 움직이지 않는다 — 이것이 thumb_2 가 -1.57 에 못 박혀
    있던 이유다. 따라서 시너지 밖 별도 축으로 빼야 한다.
    """
    synergy_mod = _load("_branch_synergy2", PKG / "tesollo_hand_synergy.py")
    anchor = torch.tensor(synergy_mod.HAND_SYNERGY_ANCHOR)      # == HAND_APPROACH_POSE
    grip = torch.tensor(r_preset.HAND_FULL_GRIP_POSE)

    for local_idx, name in zip(
        r_preset.HAND_ABDUCTION_LOCAL_INDICES, r_preset.HAND_ABDUCTION_JOINT_NAMES
    ):
        span = abs(float(grip[local_idx] - anchor[local_idx]))
        assert span < 1e-6, (
            f"{name}: open→grip 스팬이 {span:.3f} — 스팬이 있으면 시너지가 이 관절을 "
            f"움직이므로 별도 abduction 축과 충돌한다"
        )


# ---------------------------------------------------------------------------
# pregrasp clearance — 스폰 겹침(→ depenetration 폭주) 방지
# ---------------------------------------------------------------------------
def _clearances():
    import json
    tbl = json.loads((REPO / "assets" / "object_bbox.json").read_text())
    return {n: math.sqrt(sum(float(v) ** 2 for v in h)) for n, h in tbl.items()}


def test_pregrasp_clears_every_object_in_both_poses():
    """어떤 물체도, 어떤 회전에서도 palm 위치를 침범하지 못해야 한다.

    top-down palm z = (회전 후 half_z) + FINGER_REACH 이므로, 회전이 어떻든 palm 은
    물체 top 보다 FINGER_REACH 만큼 위에 있다 → 겹침이 원리적으로 불가능.
    side(cup)는 clearance(회전 무관 최대 반경) + 여유를 쓴다.
    고정 offset 시절엔 palm 이 물체중심 9.2cm 라 153종 중 48종이 palm 을 덮었고,
    회전 ADR 36 부터 depenetration 폭주 → 리턴 -4.9e7 붕괴가 났다.
    """
    clr = _clearances()
    for preset in (r_preset, l_preset):
        # top-down: 정의상 물체 top 위 FINGER_REACH → 겹침 없음
        assert preset.PREGRASP_TOPDOWN_FINGER_REACH > 0.03, \
            "FINGER_REACH 가 palm 두께보다 작으면 물체와 겹친다"

        # side: palm 이 회전 무관 최대 반경 밖에 있어야 한다
        for name, c in clr.items():
            sy = c + preset.PREGRASP_SIDE_CLEARANCE
            sz = preset.PREGRASP_SIDE_Z
            d_side = math.sqrt(sy * sy + sz * sz)
            assert d_side > c, f"{name}: side palm({d_side:.3f}) 이 물체 반경({c:.3f}) 안"


def test_pregrasp_is_within_finger_reach():
    """palm 이 물체 top 위 FINGER_REACH 에 와야 손가락(~10cm)이 물체에 닿는다.

    clearance(대각선)를 쓰던 시절엔 직립 물체에서 물체 top 을 최대 3.2cm 과대평가해
    palm 이 물체 top 위 9.3cm 에 떴다 → 손가락을 굽혀도 안 닿았다
    (실측: contact/tip 0.00~0.09, object_height 음수 — 한 번도 못 잡음).
    """
    FINGER_LEN = 0.10
    for preset in (r_preset, l_preset):
        reach = preset.PREGRASP_TOPDOWN_FINGER_REACH
        assert reach < FINGER_LEN, \
            f"FINGER_REACH({reach}) 가 손가락 길이({FINGER_LEN}) 이상 — 굽혀도 안 닿는다"


def test_curl_penalty_is_bounded():
    """finger_curl_reg 가 무계이면 물리 발산이 리턴을 -1e7 규모로 터뜨린다.

    [08-21 v2] right 는 하이브리드 reward(approach/grasp=grasp_v1 인라인, lift/goal=
    DEXTRAH)로 finger_curl_reg 자체가 없다 — 발산 방어는 total 계산 직후 직접 건
    torch.nan_to_num(nan/posinf/neginf→0)이 대신한다. left 는 아직 이식 전이라
    구 clamp 유지 검증.
    """
    src = (PKG / "grasp_right_env.py").read_text()
    reward_body = src.split("def _get_rewards", 1)[1].split("return total", 1)[0]
    assert "torch.nan_to_num(" in reward_body, \
        "right: total 발산 방어(nan_to_num)가 없다"
    left_src = (LEFT_PKG / "grasp_left_env_cfg.py").read_text()
    assert "finger_curl_dist_max" in left_src
    left_env_src = (LEFT_PKG / "grasp_left_env.py").read_text()
    assert "clamp(max=float(self.cfg.finger_curl_dist_max))" in left_env_src


def test_finger_curl_reg_anchors_on_open_pose_not_fist():
    """finger_curl_reg 기준이 FULL_GRIP(주먹)이면 정책이 물체에서 도망간다.

    DEXTRAH 원본 curled_q 는 편 자세(+thumb 대향)이고, 의도는 "과도한 말림 방지"다.
    FULL_GRIP 을 기준으로 쓰면 부호가 뒤집혀 '빈손 주먹'이 페널티 0 이 되고,
    물체를 잡으면 손가락이 막혀 페널티가 붙는다 → 접근 자체가 손해가 된다.
    lstm_test3 실증: hand_to_object 0.256(ep50) → 0.028(ep100).

    [08-21] right 는 grasp_v1 staged reward 이식으로 finger_curl_reg 자체가 없어
    이 위험이 구조적으로 소거됐다(항이 없으니 기준도 없음). left 만 검증.
    """
    src = (LEFT_PKG / "grasp_left_env.py").read_text()
    i = src.index("finger_curl_dist = (")
    expr = src[i:i + 200]
    assert "hand_open_pose" in expr, "left: curl 기준이 열린 자세가 아님"
    assert "hand_full_grip_pose" not in expr, \
        "left: curl 기준이 FULL_GRIP(주먹) — 물체 회피를 보상한다"
    right_src = (PKG / "grasp_right_env.py").read_text()
    assert "finger_curl_dist = (" not in right_src, \
        "right: finger_curl_reg 가 되살아났다(grasp_v1 이식 계약 위반)"


def test_every_object_can_reach_the_goal_in_topdown():
    """top-down 파지로 모든 물체가 goal tol 안에 들어와야 한다.

    top-down 에서 palm 은 물체 top 위 FINGER_REACH 이므로, 물체가 올라갈 수 있는
    최대 높이 = palm 박스 z 상한 - (half_z + FINGER_REACH).
    박스 상한이 0.65 였을 때는 물체 절반이 goal tol 밖이라 성공이 물리적으로
    불가능했다(in_success 0.000). IK 실측상 팔은 z≈0.74 까지 간다 → 상한 0.75.
    """
    import json
    tbl = json.loads((REPO / "assets" / "object_bbox.json").read_text())
    GOAL_Z = 0.65      # cfg.object_goal_pos[2]
    TOL = 0.10         # cfg.object_goal_tol

    for preset in (r_preset, l_preset):
        z_max = preset.PALM_POS_MAXS[2]
        worst_name, worst_gap = None, -1.0
        for name, half in tbl.items():
            half_z = float(half[2])          # 직립 기준 (회전하면 palm 도 같이 올라감)
            obj_z_max = z_max - (half_z + preset.PREGRASP_TOPDOWN_FINGER_REACH)
            gap = GOAL_Z - obj_z_max
            if gap > worst_gap:
                worst_name, worst_gap = name, gap
        assert worst_gap < TOL, (
            f"{worst_name}: top-down 으로 goal 에 {worst_gap*100:.1f}cm 까지밖에 못 감 "
            f"(tol {TOL*100:.0f}cm) — palm 박스 z 상한 {z_max} 가 너무 낮다"
        )


def test_curl_weight_matches_tesollo_hand_scale():
    """finger_curl_reg weight 는 손 구조에 맞춰야 한다.

    DEXTRAH 는 Allegro(16관절, 파지 굽힘 ~1.0rad)에서 ‖q-anchor‖²≈12 → 실효 -0.12/step.
    tesollo 는 20관절·1.8rad 라 ‖·‖²≈37 → 같은 weight(-0.01) 로 -0.37/step (3배).
    그 결과 "손을 굽히는 것 자체가 순손실"이 되어 정책이 손을 펴고 물체에서 도망갔다
    (curl -0.367 → -0.005 와 hand_to_object 급락이 동시 발생).

    [08-21] right 는 grasp_v1 staged reward 이식으로 finger_curl_reg ADR 항목 자체가
    없다(reward_weights ADR 그룹 전체 제거). left 만 검증.
    """
    src = (LEFT_PKG / "grasp_left_env_cfg.py").read_text()
    assert '"finger_curl_reg":          (-0.003, -0.003)' in src, \
        "left: curl weight 가 tesollo 스케일(-0.003)이 아님"


def test_abduction_curriculum_starts_at_grasping_pose():
    """ADR range_scale=0 에서 abduction 은 HAND_APPROACH_POSE 에 고정돼야 한다.

    probe 실증: palm 이 물체 위 4cm, thumb_2=-90°(= APPROACH_POSE 값), abduction 중립일 때
    리프트 +17.6cm (grip 3.50). abduction 을 벌리면(+1) 리프트 0 — 즉 abduction 을
    처음부터 열면 파지를 방해한다. 그래서 scale 0 에서 시작해 기본 파지를 배운 뒤
    (in_success > 0.4 로 ADR 상승) 세밀 제어를 연다. scale=0 이면 실효 action 이
    11D = DEXTRAH 원본(6 palm + 5 PCA)과 같아진다.
    """
    for pkg, side in ((PKG, "right"), (LEFT_PKG, "left")):
        cfg_src = (pkg / f"grasp_{side}_env_cfg.py").read_text()
        env_src = (pkg / f"grasp_{side}_env.py").read_text()
        assert '"abduction": {' in cfg_src and '"range_scale": (0.0, 1.0)' in cfg_src, \
            f"{side}: ADR abduction 커리큘럼이 없다"
        assert 'self._adr("abduction", "range_scale"' in env_src, \
            f"{side}: env 가 range_scale 을 안 쓴다"
        assert "self.abduction_neutral" in env_src, \
            f"{side}: 중립값(HAND_APPROACH_POSE) 앵커가 없다"

    # 중립값이 실제로 APPROACH_POSE 이고 thumb_2 = -90° 인지
    for preset in (r_preset, l_preset):
        ap = preset.HAND_APPROACH_POSE
        idxs = preset.HAND_ABDUCTION_LOCAL_INDICES
        names = preset.HAND_ABDUCTION_JOINT_NAMES
        for i, nm in zip(idxs, names):
            if "thumb_2" in nm:
                assert abs(abs(ap[i]) - 1.57) < 0.01, \
                    f"{nm} 중립이 ±90° 가 아님: {ap[i]:.3f} — probe 최적 파지 자세와 다르다"

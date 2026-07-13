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


def test_side_object_names_is_cup_only():
    assert r_preset.SIDE_APPROACH_OBJECT_NAMES == ("cup",)
    assert l_preset.SIDE_APPROACH_OBJECT_NAMES == ("cup",)


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

    enabled_self_collisions=False 이므로 이 범위가 손가락 관통을 막는 유일한
    방어선이다.
    """
    for preset, is_right in ((r_preset, True), (l_preset, False)):
        lo = preset.HAND_ABDUCTION_LIMITS_MIN
        hi = preset.HAND_ABDUCTION_LIMITS_MAX
        names = preset.HAND_ABDUCTION_JOINT_NAMES

        assert lo[0] < 0.0 < hi[0], names[0]  # thumb_1: 전 범위

        if is_right:
            assert lo[1] < 0.0 and math.isclose(hi[1], 0.0, abs_tol=1e-9), names[1]
        else:
            assert math.isclose(lo[1], 0.0, abs_tol=1e-9) and hi[1] > 0.0, names[1]

        for k in (2, 3):  # pinky_1, pinky_2
            if is_right:
                assert math.isclose(lo[k], 0.0, abs_tol=1e-9) and hi[k] > 0.0, names[k]
            else:
                assert math.isclose(hi[k], 0.0, abs_tol=1e-9) and lo[k] < 0.0, names[k]


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

    at_min = r_utils.compute_abduction_targets(torch.full((1, 4), -1.0), lo, hi)
    at_max = r_utils.compute_abduction_targets(torch.full((1, 4), +1.0), lo, hi)
    at_mid = r_utils.compute_abduction_targets(torch.zeros(1, 4), lo, hi)

    assert torch.allclose(at_min[0], lo, atol=1e-6)
    assert torch.allclose(at_max[0], hi, atol=1e-6)
    assert torch.allclose(at_mid[0], 0.5 * (lo + hi), atol=1e-6)


def test_abduction_action_clamps_out_of_range_input():
    lo = torch.tensor(r_preset.HAND_ABDUCTION_LIMITS_MIN)
    hi = torch.tensor(r_preset.HAND_ABDUCTION_LIMITS_MAX)

    out = r_utils.compute_abduction_targets(torch.full((1, 4), 5.0), lo, hi)

    assert torch.allclose(out[0], hi, atol=1e-6)


def test_synergy_basis_cannot_drive_abduction_joints():
    """자유화가 왜 별도 action 축이어야 하는지 고정.

    basis 열이 0 이면 q* = anchor 라 진행도가 항상 0 이다 — open/grip 스팬을
    벌려도 관절이 안 움직인다. basis 를 교체하면 이 테스트가 깨지고, 그때는
    자유화 설계를 다시 검토해야 한다.
    """
    basis = torch.tensor(synergy.HAND_SYNERGY_BASIS)  # (5, 20)

    for local_idx in r_preset.HAND_ABDUCTION_LOCAL_INDICES:
        col = basis[:, local_idx].abs().max()
        if local_idx == 0:  # thumb_1 은 PC4/PC5 성분이 있으나 스팬 0 이라 죽어 있었다
            assert col > 0.1
        else:
            assert col < 0.05


# ---------------------------------------------------------------------------
# pregrasp clearance — 스폰 겹침(→ depenetration 폭주) 방지
# ---------------------------------------------------------------------------
def _clearances():
    import json
    tbl = json.loads((REPO / "assets" / "object_bbox.json").read_text())
    return {n: math.sqrt(sum(float(v) ** 2 for v in h)) for n, h in tbl.items()}


def test_pregrasp_clears_every_object_in_both_poses():
    """어떤 물체도, 어떤 회전에서도 palm 위치를 침범하지 못해야 한다.

    고정 offset(구버전)은 palm 이 물체중심 9.2cm 에 있어 153종 중 48종이 palm 을
    덮었다 → 회전 ADR 36 부터 PhysX depenetration 폭주 → 리턴 -4.9e7 붕괴.
    clearance 비례로 바꾼 뒤에는 위반이 0 이어야 한다.
    """
    clr = _clearances()
    for preset in (r_preset, l_preset):
        for name, c in clr.items():
            # top-down: palm = 물체중심 + (xy, c + FINGER_CLEARANCE)
            tx, ty = preset.PREGRASP_TOPDOWN_XY
            tz = c + preset.PREGRASP_TOPDOWN_CLEARANCE
            d_top = math.sqrt(tx * tx + ty * ty + tz * tz)
            assert d_top > c, f"{name}: top-down palm({d_top:.3f}) 이 물체 반경({c:.3f}) 안"

            # side: palm = 물체중심 + (offset_x, ±(c + PALM_CLEARANCE), SIDE_Z)
            sy = c + preset.PREGRASP_SIDE_CLEARANCE
            sz = preset.PREGRASP_SIDE_Z
            d_side = math.sqrt(sy * sy + sz * sz)
            assert d_side > c, f"{name}: side palm({d_side:.3f}) 이 물체 반경({c:.3f}) 안"


def test_pregrasp_keeps_minimum_margin():
    """여유가 palm 두께 수준(3cm) 이상이어야 실제 메시가 안 닿는다."""
    clr = _clearances()
    worst = min(
        (math.sqrt(sum(v * v for v in (
            r_preset.PREGRASP_TOPDOWN_XY[0],
            r_preset.PREGRASP_TOPDOWN_XY[1],
            c + r_preset.PREGRASP_TOPDOWN_CLEARANCE,
        ))) - c)
        for c in clr.values()
    )
    assert worst >= 0.03, f"top-down 최소 여유 {worst*100:.1f}cm < 3cm"


def test_curl_penalty_is_bounded():
    """finger_curl_reg 가 무계이면 물리 발산이 리턴을 -1e7 규모로 터뜨린다."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_branch_r_cfg_src", PKG / "grasp_right_env_cfg.py"
    )
    src = (PKG / "grasp_right_env_cfg.py").read_text()
    assert "finger_curl_dist_max" in src
    env_src = (PKG / "grasp_right_env.py").read_text()
    assert "clamp(max=float(self.cfg.finger_curl_dist_max))" in env_src

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


def test_topdown_euler_is_mirrored():
    assert r_preset.PREGRASP_EULER_EX_TOPDOWN_DEG == 180.0
    assert l_preset.PREGRASP_EULER_EX_TOPDOWN_DEG == -180.0


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

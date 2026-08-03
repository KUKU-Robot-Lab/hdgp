import pytest

from r2s_autotune.paths import asset_manifest
from r2s_autotune.joint_contract import (
    JointContractError,
    load_manifest,
    normalize_joint_names,
    resolve_group_joints,
    verify_group_coverage,
)

TESOLLO_MANIFEST = asset_manifest("openarm_tesollo_sensor_rl")
RH56F1_MANIFEST = asset_manifest("openarm_bi_rh56f1_rl")


def test_movable_joints_exclude_fixed_joints():
    manifest = load_manifest(RH56F1_MANIFEST)

    # 38 DOF + 07.29 헤드 카메라 USD의 pan/tilt 2개.
    assert len(manifest.movable_joints) == 40
    assert {"head_j_pan", "head_j_tilt"} <= set(manifest.movable_joints)
    assert all(not j.endswith("_mount") for j in manifest.movable_joints)


def test_control_joints_are_subset_of_movable():
    manifest = load_manifest(RH56F1_MANIFEST)

    assert set(manifest.control_joints) <= set(manifest.movable_joints)
    # mimic 관절은 action으로 구동하지 않지만 actuator는 붙는다.
    assert "r_hj_thumb_3" in manifest.movable_joints
    assert "r_hj_thumb_3" not in manifest.control_joints


def test_normalize_maps_legacy_tesollo_names_to_canonical():
    manifest = load_manifest(TESOLLO_MANIFEST)

    legacy = ["openarm_right_joint1", "rj_dg_1_2", "openarm_left_finger_joint1"]
    canonical = normalize_joint_names(legacy, manifest)

    assert canonical == ("r_aj_1", "r_hj_thumb_2", "l_hj_gripper_1")


def test_normalize_passes_through_canonical_names():
    manifest = load_manifest(TESOLLO_MANIFEST)

    assert normalize_joint_names(["r_aj_1", "r_hj_thumb_2"], manifest) == ("r_aj_1", "r_hj_thumb_2")


def test_normalize_rejects_unknown_names():
    manifest = load_manifest(TESOLLO_MANIFEST)

    with pytest.raises(JointContractError, match="neither canonical nor manifest mapping"):
        normalize_joint_names(["not_a_joint"], manifest)


def test_resolve_group_uses_fullmatch_not_substring():
    movable = ["r_hj_thumb_1", "r_hj_thumb_1_extra", "r_hj_thumb_2"]

    assert resolve_group_joints(["r_hj_thumb_1"], movable) == ("r_hj_thumb_1",)


def test_resolve_group_rejects_expression_matching_nothing():
    with pytest.raises(JointContractError, match="matches nothing"):
        resolve_group_joints(["r_hj_pinky_9"], ["r_hj_pinky_1"])


def test_resolve_group_rejects_overlapping_expressions():
    with pytest.raises(JointContractError, match="overlap within group"):
        resolve_group_joints(["r_aj_[1-2]", "r_aj_1"], ["r_aj_1", "r_aj_2"])


def test_coverage_reports_uncovered_joints():
    manifest = load_manifest(RH56F1_MANIFEST)

    report = verify_group_coverage({"arms": ["r_aj_[1-7]", "l_aj_[1-7]"]}, manifest)

    assert not report.is_complete
    assert "r_hj_thumb_1" in report.uncovered
    assert report.overlapping == ()


def test_coverage_reports_overlap_between_groups():
    manifest = load_manifest(RH56F1_MANIFEST)

    report = verify_group_coverage(
        {"a": ["r_aj_[1-7]"], "b": ["r_aj_[1-3]"]},
        manifest,
    )

    assert "r_aj_1" in report.overlapping

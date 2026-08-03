"""부위별 calibration 파일 규약.

지켜야 할 것은 하나다: **측정하지 않은 값이 학습에 흘러들지 않는다.**
autotune 결과는 튜닝하지 않은 group까지 담으므로, 그대로 쓰면 손 강성이 400에서
config placeholder 30으로 떨어진다 — 에러 없이. extract가 그 경계를 만든다.
"""

import json

import pytest
import yaml

from r2s_autotune.calibration_parts import (
    CalibrationPartError,
    extract_part,
    main,
    merge_parts,
)
from r2s_autotune.paths import HDGP_ROOT

from conftest import CONFIG_DIR

SENSOR_PART_CONFIGS = [
    "tesollo_sensor_right_arm.yaml",
    "tesollo_sensor_right_hand.yaml",
    "tesollo_sensor_left_arm.yaml",
]


def _result():
    """튜닝한 팔 group과, 튜닝하지 않아 placeholder가 남은 손 group이 섞인 결과."""
    return {
        "schema_version": 1,
        "robot_asset": "openarm_tesollo_sensor_rl",
        "source_dataset": "/home/user/r2s/right_track.hdf5",
        "fit_error": {"total": 0.0286},
        "autotune": {"tune_groups": ["arm_proximal", "arm_wrist"]},
        "groups": {
            "arm_proximal": {"stiffness": 67.6, "damping": 6.38, "joint_friction": 0.21},
            "arm_wrist": {"stiffness": 12.0, "damping": 2.15, "joint_friction": 0.15},
            "tesollo_hand_curl": {"stiffness": 30.0, "damping": 5.0},
        },
    }


def test_extract_keeps_only_the_groups_that_were_actually_tuned():
    part = extract_part(_result(), part="right_arm")

    assert set(part["groups"]) == {"arm_proximal", "arm_wrist"}
    # 이게 핵심: 측정한 적 없는 손 값이 따라 나오면 학습에서 강성이 조용히 무너진다.
    assert "tesollo_hand_curl" not in part["groups"]


def test_extract_renames_groups_and_records_the_rename():
    part = extract_part(
        _result(),
        part="right_arm",
        rename={"arm_proximal": "right_arm_proximal", "arm_wrist": "right_arm_wrist"},
    )

    assert set(part["groups"]) == {"right_arm_proximal", "right_arm_wrist"}
    assert part["groups"]["right_arm_proximal"]["stiffness"] == 67.6
    assert part["provenance"]["renamed_groups"]["arm_proximal"] == "right_arm_proximal"


def test_extract_carries_provenance_so_a_value_can_be_traced_back():
    part = extract_part(_result(), part="right_arm", measured_on="2026-07-29",
                        source_path="logs/x.json")

    provenance = part["provenance"]
    assert provenance["source_dataset"] == "/home/user/r2s/right_track.hdf5"
    assert provenance["measured_on"] == "2026-07-29"
    assert provenance["autotune_result"] == "logs/x.json"


def test_extract_can_retarget_another_asset_without_hiding_where_it_was_measured():
    part = extract_part(_result(), part="right_arm", asset="openarm_tesollo_bi_rl")

    assert part["robot_asset"] == "openarm_tesollo_bi_rl"
    assert part["provenance"]["measured_with_asset"] == "openarm_tesollo_sensor_rl"


def test_extract_refuses_a_group_absent_from_the_result():
    with pytest.raises(CalibrationPartError, match="결과에 없는 group"):
        extract_part(_result(), part="right_arm", groups=["arm_elbow"])


def test_extract_refuses_a_result_of_another_schema():
    result = _result() | {"schema_version": 2}

    with pytest.raises(CalibrationPartError, match="schema_version"):
        extract_part(result, part="right_arm")


def _part(asset, name, groups):
    return {
        "schema_version": 1,
        "robot_asset": asset,
        "part": name,
        "groups": groups,
    }


def test_merge_joins_disjoint_parts():
    arm = _part("a", "right_arm", {"right_arm_wrist": {"stiffness": 12.0, "damping": 2.2}})
    hand = _part("a", "right_hand", {"tesollo_hand_curl": {"stiffness": 88.0, "damping": 9.0}})

    merged = merge_parts([arm, hand], sources=["arm.json", "hand.json"])

    assert set(merged["groups"]) == {"right_arm_wrist", "tesollo_hand_curl"}
    assert merged["robot_asset"] == "a"
    assert [entry["part"] for entry in merged["merged_from"]] == ["right_arm", "right_hand"]


def test_merge_refuses_an_overlapping_group_instead_of_letting_one_win():
    first = _part("a", "p1", {"right_arm_wrist": {"stiffness": 12.0, "damping": 2.2}})
    second = _part("a", "p2", {"right_arm_wrist": {"stiffness": 400.0, "damping": 80.0}})

    with pytest.raises(CalibrationPartError, match="양쪽에 있다"):
        merge_parts([first, second], sources=["p1.json", "p2.json"])


def test_merge_refuses_parts_from_different_assets():
    first = _part("sensor", "p1", {"g": {"stiffness": 1.0, "damping": 1.0}})
    second = _part("bi", "p2", {"h": {"stiffness": 1.0, "damping": 1.0}})

    with pytest.raises(CalibrationPartError, match="robot_asset"):
        merge_parts([first, second])


def test_merge_refuses_a_group_missing_damping():
    broken = _part("a", "p", {"g": {"stiffness": 1.0}})

    with pytest.raises(CalibrationPartError, match="damping"):
        merge_parts([broken], sources=["p.json"])


def test_cli_extract_then_merge_round_trips(tmp_path):
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(_result()))
    part_path = tmp_path / "right_arm.json"
    merged_path = tmp_path / "merged.json"

    assert main(["extract", "--result", str(result_path), "--part", "right_arm",
                 "--output", str(part_path)]) == 0
    assert main(["merge", str(part_path), "--output", str(merged_path)]) == 0

    merged = json.loads(merged_path.read_text())
    assert set(merged["groups"]) == {"arm_proximal", "arm_wrist"}
    assert "tesollo_hand_curl" not in merged["groups"]


# ── 부위 config 간 정합 ──────────────────────────────────────────────────────
# 부위를 나눠 돌리는 전제는 "group 경계가 부위마다 같다"는 것이다. 한쪽 config에서
# regex가 흔들리면 합쳤을 때 관절이 겹치거나 비는데, 그건 병합 시점에는 안 보인다.


def _groups_of(name):
    raw = yaml.safe_load((CONFIG_DIR / name).read_text())
    return {group: tuple(body["joint_names_expr"]) for group, body in raw["groups"].items()}


def test_sensor_part_configs_declare_the_same_group_map():
    maps = {name: _groups_of(name) for name in SENSOR_PART_CONFIGS}
    reference_name, reference = next(iter(maps.items()))

    for name, groups in maps.items():
        assert groups == reference, f"{name}의 group 경계가 {reference_name}와 다르다"


def test_each_sensor_part_config_tunes_a_different_part():
    tuned = {}
    for name in SENSOR_PART_CONFIGS:
        raw = yaml.safe_load((CONFIG_DIR / name).read_text())
        tuned[name] = set(raw["tune_groups"])

    names = list(tuned)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            assert not (tuned[first] & tuned[second]), f"{first}와 {second}가 같은 group을 튜닝한다"


def test_stored_right_arm_calibration_holds_only_arm_groups():
    """자산 옆에 저장된 실측본에 손 group이 섞이면 안 된다."""
    path = (
        HDGP_ROOT / "assets/robot/openarm_tesollo_sensor_rl/calibration/right_arm.json"
    )
    if not path.is_file():
        pytest.skip(f"not generated yet: {path}")

    payload = json.loads(path.read_text())

    assert payload["schema_version"] == 1
    assert set(payload["groups"]) == {
        "right_arm_proximal",
        "right_arm_elbow",
        "right_arm_wrist",
    }

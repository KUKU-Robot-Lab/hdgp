"""내보낸 calibration JSON을 hdgp의 실제 loader가 읽을 수 있는지 검증한다.

새 loader를 만들지 않기로 했으므로(가이드 §10 Task 0), export schema가 기존
real2sim_actuator_cfg.py와 어긋나면 학습 시점에야 발견된다. 여기서 잡는다.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from r2s_autotune.calibration_io import GroupCalibration, load_calibration
from r2s_autotune.compute_tracking_error import ErrorBreakdown
from r2s_autotune.export_best_calibration import export_best_calibration
from r2s_autotune.paths import HDGP_ROOT
from r2s_autotune.sample_candidates import Candidate

HDGP_LOADER = (
    HDGP_ROOT / "source/openarm/openarm/rh56f1/right/grasp_v1/real2sim_actuator_cfg.py"
)


@pytest.fixture(scope="module")
def hdgp_loader():
    if not HDGP_LOADER.is_file():
        pytest.skip(f"hdgp loader not found: {HDGP_LOADER}")
    spec = importlib.util.spec_from_file_location("hdgp_real2sim", HDGP_LOADER)
    module = importlib.util.module_from_spec(spec)
    # dataclass가 cls.__module__을 sys.modules에서 되찾으므로 exec 전에 등록해야 한다.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def exported(tmp_path):
    candidates = [
        Candidate(0, {"rh56f1_right_flexion": GroupCalibration(400.0, 60.0, 0.0)}),
        Candidate(1, {"rh56f1_right_flexion": GroupCalibration(520.0, 48.0, 0.3)}),
    ]
    errors = ErrorBreakdown(
        total=np.array([1.0, 0.25]),
        mse_q=np.array([0.9, 0.2]),
        mse_dq=np.array([0.1, 0.05]),
        delay_penalty=np.array([0.0, 0.0]),
    )
    path = tmp_path / "best.json"
    calibration = export_best_calibration(
        path, "openarm_bi_rh56f1_rl", "/data/track.hdf5", candidates, errors,
        ["rh56f1_right_flexion"],
    )
    return path, calibration


def test_export_selects_the_lowest_error_candidate(exported):
    _, calibration = exported

    assert calibration.groups["rh56f1_right_flexion"].stiffness == 520.0


def test_export_records_improvement_over_seed(exported):
    path, _ = exported
    payload = json.loads(path.read_text())

    assert payload["autotune"]["best_candidate_index"] == 1
    assert payload["autotune"]["improvement_ratio"] == pytest.approx(0.75)


def test_export_writes_schema_version_1(exported):
    path, _ = exported
    payload = json.loads(path.read_text())

    assert payload["schema_version"] == 1
    assert payload["robot_asset"] == "openarm_bi_rh56f1_rl"
    assert payload["source_dataset"] == "/data/track.hdf5"


def test_exported_json_round_trips_through_our_loader(exported):
    path, calibration = exported

    reloaded = load_calibration(path)

    assert reloaded.groups == calibration.groups


def test_hdgp_loader_reads_exported_json(exported, hdgp_loader):
    path, _ = exported

    groups = hdgp_loader.load_real2sim_calibration(str(path))

    assert groups["rh56f1_right_flexion"].stiffness == 520.0
    assert groups["rh56f1_right_flexion"].damping == 48.0


def test_hdgp_get_actuator_params_uses_exported_gains(exported, hdgp_loader):
    path, _ = exported
    groups = hdgp_loader.load_real2sim_calibration(str(path))

    params = hdgp_loader.get_actuator_params("rh56f1_right_flexion", groups, 400.0, 60.0)

    assert params["stiffness"] == 520.0
    assert params["damping"] == 48.0
    assert params["friction"] == pytest.approx(0.3)


def test_hdgp_loader_falls_back_silently_for_unknown_group(exported, hdgp_loader):
    """이 조용한 fallback이 group 이름 오타를 숨긴다. test_config_matches_env_cfg가 방어한다."""
    path, _ = exported
    groups = hdgp_loader.load_real2sim_calibration(str(path))

    params = hdgp_loader.get_actuator_params("typo_group_name", groups, 400.0, 60.0)

    assert params == {"stiffness": 400.0, "damping": 60.0}


def test_loader_rejects_unsupported_schema_version(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 2, "groups": {"a": {"stiffness": 1, "damping": 1}}}))

    with pytest.raises(ValueError, match="unsupported calibration schema_version"):
        load_calibration(path)


def test_export_rejects_error_count_mismatch(tmp_path):
    errors = ErrorBreakdown(np.array([1.0]), np.array([1.0]), np.array([0.0]), np.array([0.0]))

    with pytest.raises(ValueError, match="does not match error count"):
        export_best_calibration(
            tmp_path / "x.json", "asset", "ds",
            [Candidate(0, {}), Candidate(1, {})], errors, [],
        )

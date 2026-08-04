import json

import numpy as np
import pytest
import torch

from scripts.eval_s2r.providers import CameraFileProvider, make_provider
from scripts.eval_s2r.transforms import compose_local_pose

# 스키마 리터럴(task-2-brief.md 계약 고정). 값 자체는 축소했지만 키 구조는 원본 그대로.
_T_LOCAL_CAM_0 = [[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.3], [0.0, 0.0, 0.0, 1.0]]
_T_LOCAL_CAM_1 = [[0.0, -1.0, 0.0, 0.4], [1.0, 0.0, 0.0, 0.1], [0.0, 0.0, 1.0, 0.35], [0.0, 0.0, 0.0, 1.0]]
_T_LOCAL_CAM_2 = _T_LOCAL_CAM_0

_T_CAM_OBJ_0 = [[1.0, 0.0, 0.0, 0.1], [0.0, 1.0, 0.0, 0.02], [0.0, 0.0, 1.0, 0.3], [0.0, 0.0, 0.0, 1.0]]
_T_CAM_OBJ_2_NONFINITE = [
    [1.0, 0.0, 0.0, float("nan")],
    [0.0, 1.0, 0.0, 0.02],
    [0.0, 0.0, 1.0, 0.3],
    [0.0, 0.0, 0.0, 1.0],
]

_GRID = {
    "x_min": 0.15, "x_max": 0.39, "nx": 5,
    "y_min": -0.22, "y_max": 0.02, "ny": 5,
    "repeats": 8,
}


def _write_meta(tmp_path, num_envs=3):
    meta = {
        "robot": "right",
        "num_envs": num_envs,
        "grid": _GRID,
        "head_tilt": -0.6,
        "T_local_cam": {"0": _T_LOCAL_CAM_0, "1": _T_LOCAL_CAM_1, "2": _T_LOCAL_CAM_2},
        "git_sha": "deadbeef",
        "k_source": "nominal",
    }
    p = tmp_path / "meta.json"
    p.write_text(json.dumps(meta))
    return p


def _write_poses(tmp_path, num_envs=3, nonfinite=True):
    poses = {
        "0": {"ok": True, "T_cam_obj": _T_CAM_OBJ_0},
        "1": {"ok": False, "reason": "register_failed"},
    }
    if nonfinite:
        poses["2"] = {"ok": True, "T_cam_obj": _T_CAM_OBJ_2_NONFINITE}
    else:
        poses["2"] = {"ok": True, "T_cam_obj": _T_CAM_OBJ_0}
    data = {"robot": "right", "num_envs": num_envs, "poses": poses}
    p = tmp_path / "poses.json"
    p.write_text(json.dumps(data))
    return p


class TestCameraFileProvider:
    def test_ok_env_pos_local_matches_compose(self, tmp_path):
        meta_path = _write_meta(tmp_path)
        poses_path = _write_poses(tmp_path)
        provider = CameraFileProvider(str(poses_path), str(meta_path))
        expected = compose_local_pose(np.array(_T_LOCAL_CAM_0), np.array(_T_CAM_OBJ_0))
        ov = provider.get_override(env=None)
        assert torch.allclose(ov[0], torch.tensor(expected, dtype=torch.float32), atol=1e-6)

    def test_fail_env_in_failed_envs_and_nan_row(self, tmp_path):
        meta_path = _write_meta(tmp_path)
        poses_path = _write_poses(tmp_path)
        provider = CameraFileProvider(str(poses_path), str(meta_path))
        assert 1 in provider.failed_envs
        ov = provider.get_override(env=None)
        assert torch.isnan(ov[1]).all()

    def test_nonfinite_ok_env_demoted_to_failed(self, tmp_path):
        meta_path = _write_meta(tmp_path)
        poses_path = _write_poses(tmp_path, nonfinite=True)
        provider = CameraFileProvider(str(poses_path), str(meta_path))
        assert 2 in provider.failed_envs
        assert provider.fail_reasons[2] == "nonfinite"
        ov = provider.get_override(env=None)
        assert torch.isnan(ov[2]).all()

    def test_invalid_rotation_env_demoted_not_crashing(self, tmp_path):
        # 유한하지만 회전부가 비직교(스케일-2 R) — 퇴화한 FoundationPose 출력 가능성
        bad_T_cam_obj = [
            [2.0, 0.0, 0.0, 0.1],
            [0.0, 2.0, 0.0, 0.02],
            [0.0, 0.0, 2.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ]
        meta_path = _write_meta(tmp_path)
        poses = {
            "0": {"ok": True, "T_cam_obj": _T_CAM_OBJ_0},
            "1": {"ok": False, "reason": "register_failed"},
            "2": {"ok": True, "T_cam_obj": bad_T_cam_obj},
        }
        poses_path = tmp_path / "poses.json"
        poses_path.write_text(json.dumps({"robot": "right", "num_envs": 3, "poses": poses}))

        provider = CameraFileProvider(str(poses_path), str(meta_path))  # 생성이 예외 없이 성공해야 함

        assert 2 in provider.failed_envs
        assert provider.fail_reasons[2] == "invalid_rotation"
        ov = provider.get_override(env=None)
        assert torch.isnan(ov[2]).all()
        # 다른 env(0)는 정상 계산되어 있어야 함(전체 생성 실패로 오염되지 않음)
        expected0 = compose_local_pose(np.array(_T_LOCAL_CAM_0), np.array(_T_CAM_OBJ_0))
        assert torch.allclose(ov[0], torch.tensor(expected0, dtype=torch.float32), atol=1e-6)

    def test_expected_grid_is_defensive_copy(self, tmp_path):
        meta_path = _write_meta(tmp_path)
        poses_path = _write_poses(tmp_path)
        provider = CameraFileProvider(str(poses_path), str(meta_path))
        provider.expected_grid["x_min"] = 999.0  # 반환된 dict를 in-place 수정
        # 동일 파일로 새 provider를 구성해도 오염되지 않아야 함(내부 파싱 dict를 그대로 노출하지 않음)
        provider2 = CameraFileProvider(str(poses_path), str(meta_path))
        assert provider2.expected_grid["x_min"] == _GRID["x_min"]

    def test_num_envs_mismatch_raises(self, tmp_path):
        meta_path = _write_meta(tmp_path, num_envs=3)
        poses_path = _write_poses(tmp_path, num_envs=4)
        with pytest.raises(ValueError):
            CameraFileProvider(str(poses_path), str(meta_path))

    def test_expected_grid_matches_meta_grid_verbatim(self, tmp_path):
        meta_path = _write_meta(tmp_path)
        poses_path = _write_poses(tmp_path)
        provider = CameraFileProvider(str(poses_path), str(meta_path))
        assert provider.expected_grid == _GRID

    def test_on_reset_is_noop(self, tmp_path):
        meta_path = _write_meta(tmp_path)
        poses_path = _write_poses(tmp_path)
        provider = CameraFileProvider(str(poses_path), str(meta_path))
        before = provider.get_override(env=None).clone()
        provider.on_reset(env=None, env_ids=torch.tensor([0, 1, 2]))
        after = provider.get_override(env=None)
        # NaN 행은 allclose가 False를 주므로 finite 부분만 비교
        assert torch.allclose(before[0], after[0])

    def test_get_override_returns_clone(self, tmp_path):
        meta_path = _write_meta(tmp_path)
        poses_path = _write_poses(tmp_path)
        provider = CameraFileProvider(str(poses_path), str(meta_path))
        ov = provider.get_override(env=None)
        ov[0, 0] = 999.0  # 반환된 텐서를 in-place 수정
        ov2 = provider.get_override(env=None)
        assert float(ov2[0, 0]) != pytest.approx(999.0)


class TestMakeProviderCameraFrozen:
    def test_missing_kwargs_raises_value_error(self):
        with pytest.raises(ValueError):
            make_provider("camera_frozen")

    def test_with_kwargs_returns_camera_file_provider(self, tmp_path):
        meta_path = _write_meta(tmp_path)
        poses_path = _write_poses(tmp_path)
        provider = make_provider(
            "camera_frozen", poses_path=str(poses_path), frames_meta_path=str(meta_path)
        )
        assert isinstance(provider, CameraFileProvider)

    def test_live_and_state_frozen_still_zero_arg(self):
        # 기존 계약(zero-arg) 유지 확인 — make_provider가 **kwargs를 얻어도 하위호환
        from scripts.eval_s2r.providers import LiveProvider, StateFrozenProvider
        assert isinstance(make_provider("live"), LiveProvider)
        assert isinstance(make_provider("state_frozen"), StateFrozenProvider)

"""fp_batch.py 순수 함수 테스트 (FP 미의존 — numpy만 사용, 컨테이너 없이 메인 repo 환경에서 실행).

fp_batch.py는 py3.8 호환 standalone 파일이라 상단 import가 numpy/json/argparse/pathlib뿐이고
estimater/trimesh는 run_fp() 안 지연 import다 — 그래서 이 테스트는 FP 설치 없이도 전부 통과해야 한다.
"""
import json

import numpy as np
import pytest

from scripts.eval_s2r.fp_batch import (
    collect_stats, error_cm, load_frame, mesh_path_for, pose_entry,
)


def _write_frame_npz(path, **overrides):
    """render_pass.py _capture_env 스키마 그대로(값은 tiny 4x4)."""
    frame = {
        "rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "depth": np.full((4, 4), 0.5, dtype=np.float32),
        "mask": np.zeros((4, 4), dtype=bool),
        "K": np.eye(3, dtype=np.float64),
        "T_local_cam": np.eye(4, dtype=np.float64),
        "gt_obj_pos_local": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        "gt_obj_pos_init": np.array([0.1, 0.2, 0.3], dtype=np.float64),
        "gt_obj_rot_wxyz": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "obj_idx": np.array(0),
        "cell_idx": np.array(0),
        "cell_x": np.array(0.2),
        "cell_y": np.array(-0.1),
        "mask_pixels": np.array(0),
    }
    frame.update(overrides)
    np.savez(path, **frame)
    return path


class TestLoadFrame:
    def test_happy_path_returns_all_keys(self, tmp_path):
        p = _write_frame_npz(tmp_path / "env_0.npz")
        out = load_frame(str(p))
        assert np.allclose(out["gt_obj_pos_local"], [0.1, 0.2, 0.3])
        assert out["mask"].shape == (4, 4)
        assert int(out["obj_idx"]) == 0

    def test_missing_key_raises_value_error_naming_key(self, tmp_path):
        # np.savez 자체는 임의 키 조합을 허용하므로 필수 키 하나만 빼고 저장한다
        path = tmp_path / "env_1.npz"
        np.savez(
            path,
            rgb=np.zeros((4, 4, 3), dtype=np.uint8),
            depth=np.full((4, 4), 0.5, dtype=np.float32),
            mask=np.zeros((4, 4), dtype=bool),
            K=np.eye(3),
            T_local_cam=np.eye(4),
            gt_obj_pos_local=np.array([0.1, 0.2, 0.3]),
            gt_obj_pos_init=np.array([0.1, 0.2, 0.3]),
            gt_obj_rot_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            obj_idx=np.array(0),
            cell_idx=np.array(0),
            cell_x=np.array(0.2),
            # cell_y 누락
            mask_pixels=np.array(0),
        )
        with pytest.raises(ValueError, match="cell_y"):
            load_frame(str(path))


class TestMeshPathFor:
    def _object_map(self, usd_path, obj_id="cup0", scale=(1.0, 1.0, 1.0)):
        return {"0": {"id": obj_id, "usd_path": usd_path, "scale": list(scale)}}

    def test_sibling_obj_preferred_over_mesh_dir(self, tmp_path):
        usd_dir = tmp_path / "objs"
        usd_dir.mkdir()
        usd_path = usd_dir / "cup0.usd"
        usd_path.write_text("")
        sibling_obj = usd_dir / "cup0.obj"
        sibling_obj.write_text("")

        mesh_dir = tmp_path / "export"
        mesh_dir.mkdir()
        fallback_obj = mesh_dir / "cup0.obj"
        fallback_obj.write_text("")

        object_map = self._object_map(str(usd_path))
        path, scale = mesh_path_for(0, object_map, str(mesh_dir))
        assert path == str(sibling_obj)
        assert scale == [1.0, 1.0, 1.0]

    def test_falls_back_to_mesh_dir_when_no_sibling(self, tmp_path):
        usd_dir = tmp_path / "objs"
        usd_dir.mkdir()
        usd_path = usd_dir / "cup0.usd"
        usd_path.write_text("")
        # sibling .obj 없음

        mesh_dir = tmp_path / "export"
        mesh_dir.mkdir()
        fallback_obj = mesh_dir / "cup0.obj"
        fallback_obj.write_text("")

        object_map = self._object_map(str(usd_path))
        path, scale = mesh_path_for(0, object_map, str(mesh_dir))
        assert path == str(fallback_obj)

    def test_mesh_missing_returns_none_path(self, tmp_path):
        usd_dir = tmp_path / "objs"
        usd_dir.mkdir()
        usd_path = usd_dir / "cup0.usd"
        usd_path.write_text("")

        mesh_dir = tmp_path / "export"
        mesh_dir.mkdir()

        object_map = self._object_map(str(usd_path))
        path, scale = mesh_path_for(0, object_map, str(mesh_dir))
        assert path is None
        assert scale == [1.0, 1.0, 1.0]

    def test_mesh_dir_none_only_checks_sibling(self, tmp_path):
        usd_dir = tmp_path / "objs"
        usd_dir.mkdir()
        usd_path = usd_dir / "cup0.usd"
        usd_path.write_text("")

        object_map = self._object_map(str(usd_path))
        path, scale = mesh_path_for(0, object_map, None)
        assert path is None

    def test_unknown_obj_idx_raises_value_error(self, tmp_path):
        object_map = self._object_map(str(tmp_path / "objs" / "cup0.usd"))
        with pytest.raises(ValueError, match="1"):
            mesh_path_for(1, object_map, None)


class TestPoseEntry:
    def test_ok_true_with_valid_T(self):
        T = np.eye(4)
        T[:3, 3] = [0.1, 0.2, 0.3]
        entry = pose_entry(True, T=T)
        assert entry == {"ok": True, "T_cam_obj": T.tolist()}

    def test_ok_true_without_T_raises(self):
        with pytest.raises(ValueError):
            pose_entry(True)

    def test_ok_true_with_wrong_shape_T_raises(self):
        with pytest.raises(ValueError):
            pose_entry(True, T=np.eye(3))

    def test_ok_false_with_reason(self):
        entry = pose_entry(False, reason="mesh_missing")
        assert entry == {"ok": False, "reason": "mesh_missing"}

    def test_ok_false_without_reason_raises(self):
        with pytest.raises(ValueError):
            pose_entry(False)

    def test_pose_entry_json_roundtrip(self):
        T = np.eye(4)
        entry = pose_entry(True, T=T)
        # CameraFileProvider가 읽는 poses.json 스키마와 리터럴 일치(json 직렬화 가능해야 함)
        s = json.dumps(entry)
        back = json.loads(s)
        assert back["ok"] is True
        assert np.allclose(np.array(back["T_cam_obj"]), T)


class TestErrorCm:
    def test_zero_error_hand_computed(self):
        T_local_cam = np.eye(4)
        T_local_cam[:3, 3] = [0.5, 0.0, 0.3]
        T_cam_obj = np.eye(4)
        T_cam_obj[:3, 3] = [0.1, 0.02, 0.3]
        gt = np.array([0.6, 0.02, 0.6])  # 0.5+0.1, 0+0.02, 0.3+0.3
        assert error_cm(T_cam_obj, T_local_cam, gt) == pytest.approx(0.0, abs=1e-9)

    def test_nonzero_error_hand_computed(self):
        T_local_cam = np.eye(4)
        T_local_cam[:3, 3] = [0.5, 0.0, 0.3]
        T_cam_obj = np.eye(4)
        T_cam_obj[:3, 3] = [0.1, 0.02, 0.3]
        # 예측 위치 = (0.6, 0.02, 0.6). GT를 x에서 1cm(0.01m) 어긋나게
        gt = np.array([0.61, 0.02, 0.6])
        assert error_cm(T_cam_obj, T_local_cam, gt) == pytest.approx(1.0, abs=1e-6)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            error_cm(np.eye(3), np.eye(4), np.zeros(3))
        with pytest.raises(ValueError):
            error_cm(np.eye(4), np.eye(4), np.zeros(2))


class TestCollectStats:
    def test_empty_is_safe(self):
        stats = collect_stats({}, {})
        assert stats["num_envs"] == 0
        assert stats["success_rate"] is None
        assert stats["error_cm"]["mean"] is None
        assert stats["error_cm"]["n"] == 0

    def test_mixed_entries_success_rate_and_reasons(self):
        entries = {
            0: {"ok": True, "T_cam_obj": np.eye(4).tolist()},
            1: {"ok": False, "reason": "mesh_missing"},
            2: {"ok": False, "reason": "empty_mask"},
            3: {"ok": True, "T_cam_obj": np.eye(4).tolist()},
        }
        errors = {0: 1.0, 3: 3.0}
        stats = collect_stats(entries, errors)
        assert stats["num_envs"] == 4
        assert stats["num_ok"] == 2
        assert stats["num_failed"] == 2
        assert stats["success_rate"] == pytest.approx(0.5)
        assert stats["failure_reasons"] == {"mesh_missing": 1, "empty_mask": 1}
        assert stats["error_cm"]["mean"] == pytest.approx(2.0)
        assert stats["error_cm"]["median"] == pytest.approx(2.0)
        assert stats["error_cm"]["n"] == 2

    def test_p95_computed_with_numpy(self):
        errors = {i: float(i) for i in range(20)}  # 0..19
        entries = {i: {"ok": True} for i in range(20)}
        stats = collect_stats(entries, errors)
        assert stats["error_cm"]["p95"] == pytest.approx(np.percentile(np.arange(20), 95))

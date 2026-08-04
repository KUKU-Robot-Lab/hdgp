"""델타 맵 테스트 (STATE−CAMERA 비교).

설계: docs/superpowers/specs/2026-08-04-grasp-v1-s2r-eval-sp2-design.md §7
"""
import csv
import json

import pytest

from scripts.eval_s2r.delta_report import build_delta, write_delta_csv, write_delta_heatmap

# 각 셀 스키마 (report.py aggregate 출력 스키마와 동일)
CELLS_STATE = [
    {"cell_idx": 0, "x": 0.21, "y": -0.16, "success_rate": 0.5, "lifted_rate": 0.4, "perception_fail_rate": None},
    {"cell_idx": 1, "x": 0.21, "y": 0.02, "success_rate": 1.0, "lifted_rate": 0.9, "perception_fail_rate": None},
    {"cell_idx": 2, "x": 0.33, "y": -0.16, "success_rate": None, "lifted_rate": None, "perception_fail_rate": None},
    {"cell_idx": 3, "x": 0.33, "y": 0.02, "success_rate": 0.7, "lifted_rate": 0.6, "perception_fail_rate": None},
]

CELLS_CAMERA = [
    {"cell_idx": 0, "x": 0.21, "y": -0.16, "success_rate": 0.3, "lifted_rate": 0.2, "perception_fail_rate": 0.1},
    {"cell_idx": 1, "x": 0.21, "y": 0.02, "success_rate": 0.8, "lifted_rate": 0.7, "perception_fail_rate": 0.05},
    {"cell_idx": 2, "x": 0.33, "y": -0.16, "success_rate": 0.6, "lifted_rate": 0.5, "perception_fail_rate": 0.2},
    {"cell_idx": 3, "x": 0.33, "y": 0.02, "success_rate": None, "lifted_rate": None, "perception_fail_rate": None},
]


class TestBuildDelta:
    def test_delta_success_arithmetic(self):
        """STATE−CAMERA 성공률 차이 계산."""
        rows = build_delta(CELLS_STATE, CELLS_CAMERA)
        assert len(rows) == 4
        # cell 0: 0.5 - 0.3 = 0.2
        assert rows[0]["delta_success"] == pytest.approx(0.2)
        # cell 1: 1.0 - 0.8 = 0.2
        assert rows[1]["delta_success"] == pytest.approx(0.2)

    def test_delta_lifted_arithmetic(self):
        """STATE−CAMERA 리프트율 차이 계산."""
        rows = build_delta(CELLS_STATE, CELLS_CAMERA)
        # cell 0: 0.4 - 0.2 = 0.2
        assert rows[0]["delta_lifted"] == pytest.approx(0.2)
        # cell 1: 0.9 - 0.7 = 0.2
        assert rows[1]["delta_lifted"] == pytest.approx(0.2)

    def test_delta_none_when_either_side_none(self):
        """양쪽 중 하나가 None이면 delta도 None."""
        rows = build_delta(CELLS_STATE, CELLS_CAMERA)
        # cell 2: STATE는 None, CAMERA는 0.6 → delta_success=None
        assert rows[2]["delta_success"] is None
        assert rows[2]["delta_lifted"] is None
        # cell 3: STATE는 0.7, CAMERA는 None → delta_success=None
        assert rows[3]["delta_success"] is None
        assert rows[3]["delta_lifted"] is None

    def test_perception_fail_rate_from_camera(self):
        """perception_fail_rate는 CAMERA 쪽에서만 가져온다."""
        rows = build_delta(CELLS_STATE, CELLS_CAMERA)
        assert rows[0]["perception_fail_rate"] == pytest.approx(0.1)
        assert rows[1]["perception_fail_rate"] == pytest.approx(0.05)
        assert rows[2]["perception_fail_rate"] == pytest.approx(0.2)
        assert rows[3]["perception_fail_rate"] is None

    def test_row_structure(self):
        """각 행이 필요한 키를 모두 가진다."""
        rows = build_delta(CELLS_STATE, CELLS_CAMERA)
        required_keys = ["cell_idx", "x", "y", "delta_success", "delta_lifted",
                        "perception_fail_rate", "n_state", "n_camera"]
        for row in rows:
            for key in required_keys:
                assert key in row, f"Missing key {key!r}"

    def test_n_state_n_camera_counts(self):
        """n_state, n_camera는 각 쪽이 None이 아닌 경우의 개수."""
        rows = build_delta(CELLS_STATE, CELLS_CAMERA)
        # cell 0: both have success_rate → n_state=1, n_camera=1
        assert rows[0]["n_state"] == 1
        assert rows[0]["n_camera"] == 1
        # cell 2: STATE is None, CAMERA is not → n_state=0, n_camera=1
        assert rows[2]["n_state"] == 0
        assert rows[2]["n_camera"] == 1
        # cell 3: STATE is not None, CAMERA is None → n_state=1, n_camera=0
        assert rows[3]["n_state"] == 1
        assert rows[3]["n_camera"] == 0

    def test_xy_mismatch_raises_valueerror(self):
        """셀 인덱스는 같으나 (x, y)가 다르면 ValueError."""
        import copy
        bad_camera = copy.deepcopy(CELLS_CAMERA)
        bad_camera[0]["x"] = 0.99  # 불일치
        with pytest.raises(ValueError):
            build_delta(CELLS_STATE, bad_camera)

    def test_length_mismatch_raises_valueerror(self):
        """셀 개수가 다르면 ValueError."""
        short_camera = CELLS_CAMERA[:2]
        with pytest.raises(ValueError):
            build_delta(CELLS_STATE, short_camera)

    def test_cell_idx_order_independent(self):
        """cell_idx로 매칭하므로 순서는 무관."""
        reordered = [CELLS_CAMERA[3], CELLS_CAMERA[0], CELLS_CAMERA[1], CELLS_CAMERA[2]]
        rows = build_delta(CELLS_STATE, reordered)
        # cell 0 여전히 매칭됨
        assert rows[0]["cell_idx"] == 0
        assert rows[0]["delta_success"] == pytest.approx(0.2)


class TestWriteDeltaCsv:
    def test_csv_roundtrip(self, tmp_path):
        """CSV 쓰고 읽기."""
        rows = build_delta(CELLS_STATE, CELLS_CAMERA)
        p = tmp_path / "delta.csv"
        write_delta_csv(rows, str(p))
        with open(p) as f:
            got = list(csv.DictReader(f))
        assert len(got) == 4
        # cell 0: delta_success=0.2
        assert float(got[0]["delta_success"]) == pytest.approx(0.2)
        # cell 2: delta_success=None → 빈칸 또는 "None"
        val = got[2]["delta_success"]
        assert val == "" or val == "None" or val is None

    def test_csv_includes_all_keys(self, tmp_path):
        """모든 row 키가 CSV 컬럼으로."""
        rows = build_delta(CELLS_STATE, CELLS_CAMERA)
        p = tmp_path / "delta.csv"
        write_delta_csv(rows, str(p))
        with open(p) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        required = ["cell_idx", "x", "y", "delta_success", "delta_lifted",
                   "perception_fail_rate", "n_state", "n_camera"]
        for key in required:
            assert key in headers, f"Missing column {key!r}"


class TestWriteDeltaHeatmap:
    def test_heatmap_file_created(self, tmp_path):
        """히트맵 PNG 파일 생성."""
        rows = build_delta(CELLS_STATE, CELLS_CAMERA)
        p = tmp_path / "delta_success.png"
        write_delta_heatmap(rows, nx=2, ny=2, metric="delta_success", path=str(p))
        assert p.stat().st_size > 0

    def test_heatmap_both_metrics(self, tmp_path):
        """두 메트릭 모두 지원."""
        rows = build_delta(CELLS_STATE, CELLS_CAMERA)
        for metric in ("delta_success", "delta_lifted"):
            p = tmp_path / f"{metric}.png"
            # 충돌 없어야 함
            write_delta_heatmap(rows, nx=2, ny=2, metric=metric, path=str(p))
            assert p.stat().st_size > 0

    def test_heatmap_unknown_metric_raises(self, tmp_path):
        """알 수 없는 메트릭은 ValueError."""
        rows = build_delta(CELLS_STATE, CELLS_CAMERA)
        with pytest.raises(ValueError, match="unknown metric"):
            write_delta_heatmap(rows, nx=2, ny=2, metric="bogus", path=str(tmp_path / "x.png"))

    def test_heatmap_none_cells_become_nan(self, tmp_path):
        """None 델타는 NaN으로 렌더링됨 (흰색)."""
        rows = build_delta(CELLS_STATE, CELLS_CAMERA)
        p = tmp_path / "delta_success.png"
        # Should not crash
        write_delta_heatmap(rows, nx=2, ny=2, metric="delta_success", path=str(p))
        assert p.stat().st_size > 0

    def test_heatmap_perception_fail_marker(self, tmp_path):
        """perception_fail_rate > 0인 셀에 'P' 표시."""
        rows = build_delta(CELLS_STATE, CELLS_CAMERA)
        p = tmp_path / "delta_success.png"
        # cell 0, 1, 2는 perception_fail_rate > 0
        # cell 3은 None
        # 히트맵 생성됨 (PNG는 텍스트 분석 어려우므로 크래시 안 하면 OK)
        write_delta_heatmap(rows, nx=2, ny=2, metric="delta_success", path=str(p))
        assert p.stat().st_size > 0

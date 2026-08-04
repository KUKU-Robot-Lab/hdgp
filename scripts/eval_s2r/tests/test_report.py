import csv
import json

import pytest

from scripts.eval_s2r.report import EpisodeResult, aggregate, write_csv, write_summary, write_heatmap

CELLS = [(0.21, -0.16), (0.21, 0.02), (0.33, -0.16), (0.33, 0.02)]  # nx=2, ny=2


def _ep(cell, success, lifted=None, grip=4.0, disp=0.01, obj=0, invalid=False,
        finger_contacts=(1.0, 1.0, 1.0, 1.0, 1.0)):
    return EpisodeResult(
        cell_idx=cell, success=success,
        lifted=success if lifted is None else lifted,
        grip_count=grip, displacement=disp, obj_idx=obj, invalid=invalid,
        finger_contacts=finger_contacts,
    )


class TestAggregate:
    def test_success_rate_per_cell(self):
        results = [_ep(0, True), _ep(0, False), _ep(1, True), _ep(1, True)]
        rows = aggregate(results, CELLS)
        assert len(rows) == 4
        assert rows[0]["success_rate"] == pytest.approx(0.5)
        assert rows[1]["success_rate"] == pytest.approx(1.0)
        assert rows[2]["n_episodes"] == 0 and rows[2]["success_rate"] is None

    def test_invalid_excluded_and_counted(self):
        results = [_ep(0, True), _ep(0, True, invalid=True)]
        rows = aggregate(results, CELLS)
        assert rows[0]["n_episodes"] == 1
        assert rows[0]["n_invalid"] == 1
        assert rows[0]["success_rate"] == pytest.approx(1.0)

    def test_per_object_breakdown(self):
        results = [_ep(0, True, obj=3), _ep(0, False, obj=3), _ep(0, True, obj=5)]
        rows = aggregate(results, CELLS)
        assert rows[0]["per_obj_success"][3] == pytest.approx(0.5)
        assert rows[0]["per_obj_success"][5] == pytest.approx(1.0)

    def test_cell_xy_matches_cells(self):
        rows = aggregate([], CELLS)
        assert rows[3]["x"] == pytest.approx(0.33) and rows[3]["y"] == pytest.approx(0.02)

    def test_finger_contact_rates_elementwise_mean(self):
        results = [
            _ep(0, True, finger_contacts=(1.0, 0.0, 1.0, 0.0, 1.0)),
            _ep(0, False, finger_contacts=(0.0, 0.0, 1.0, 1.0, 1.0)),
        ]
        rows = aggregate(results, CELLS)
        assert rows[0]["finger_contact_rates"] == pytest.approx([0.5, 0.0, 1.0, 0.5, 1.0])

    def test_finger_contact_rates_none_when_empty(self):
        rows = aggregate([], CELLS)
        assert rows[0]["finger_contact_rates"] is None


class TestWriters:
    def test_csv_roundtrip(self, tmp_path):
        rows = aggregate([_ep(0, True, finger_contacts=(1.0, 1.0, 0.0, 1.0, 1.0))], CELLS)
        p = tmp_path / "results.csv"
        write_csv(rows, str(p))
        with open(p) as f:
            got = list(csv.DictReader(f))
        assert len(got) == 4
        assert float(got[0]["success_rate"]) == pytest.approx(1.0)
        assert json.loads(got[0]["finger_contact_rates"]) == pytest.approx([1.0, 1.0, 0.0, 1.0, 1.0])
        assert json.loads(got[2]["finger_contact_rates"]) is None

    def test_summary_json(self, tmp_path):
        rows = aggregate([_ep(0, True), _ep(1, False)], CELLS)
        p = tmp_path / "summary.json"
        write_summary(rows, meta={"checkpoint": "ck.pth", "git_sha": "abc"}, path=str(p))
        got = json.loads(p.read_text())
        assert got["meta"]["checkpoint"] == "ck.pth"
        assert got["overall_success_rate"] == pytest.approx(0.5)
        assert got["total_episodes"] == 2

    def test_heatmap_file_created(self, tmp_path):
        rows = aggregate([_ep(i, True) for i in range(4)], CELLS)
        p = tmp_path / "hm.png"
        write_heatmap(rows, nx=2, ny=2, metric="success_rate", path=str(p))
        assert p.stat().st_size > 0

    def test_heatmap_unknown_metric_raises(self, tmp_path):
        rows = aggregate([], CELLS)
        with pytest.raises(ValueError):
            write_heatmap(rows, nx=2, ny=2, metric="bogus", path=str(tmp_path / "x.png"))

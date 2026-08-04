import math
import pytest
import torch

from scripts.eval_s2r.grid import (
    GridSpec, build_cells, env_to_cell, build_spawn_tensor, single_spawn_tensor,
)


def _spec(**kw):
    base = dict(x_min=0.21, x_max=0.33, nx=3, y_min=-0.16, y_max=0.02, ny=2, repeats=2)
    base.update(kw)
    return GridSpec(**base)


class TestGridSpec:
    def test_valid_spec(self):
        s = _spec()
        assert s.nx == 3 and s.repeats == 2

    @pytest.mark.parametrize("kw", [
        dict(x_min=0.5, x_max=0.2),          # min > max
        dict(nx=0),                          # nx < 1
        dict(ny=-1),
        dict(repeats=0),
        dict(y_min=0.1, y_max=0.1, ny=2),    # 폭 0인데 셀 2개
    ])
    def test_invalid_spec_raises(self, kw):
        with pytest.raises(ValueError):
            _spec(**kw)

    def test_single_cell_zero_width_ok(self):
        # nx=1이면 min==max 허용 (단일 x 라인)
        s = _spec(x_min=0.27, x_max=0.27, nx=1)
        assert build_cells(s)[0][0] == pytest.approx(0.27)


class TestBuildCells:
    def test_count_and_corners(self):
        s = _spec()
        cells = build_cells(s)
        assert len(cells) == 6  # 3*2
        xs = sorted({c[0] for c in cells})
        ys = sorted({c[1] for c in cells})
        assert xs[0] == pytest.approx(0.21) and xs[-1] == pytest.approx(0.33)
        assert ys[0] == pytest.approx(-0.16) and ys[-1] == pytest.approx(0.02)

    def test_x_major_order(self):
        s = _spec()
        cells = build_cells(s)
        # x-major: 같은 x에서 y가 먼저 돈다
        assert cells[0][0] == cells[1][0]
        assert cells[0][1] != cells[1][1]


class TestEnvMapping:
    def test_env_to_cell(self):
        assert env_to_cell(0, repeats=2) == 0
        assert env_to_cell(1, repeats=2) == 0
        assert env_to_cell(2, repeats=2) == 1

    def test_spawn_tensor_shape_and_values(self):
        s = _spec()
        cells = build_cells(s)
        t = build_spawn_tensor(cells, s.repeats)
        assert t.shape == (12, 3) and t.dtype == torch.float32
        assert t[0, 0] == pytest.approx(cells[0][0])
        assert t[2, 0] == pytest.approx(cells[1][0])  # env2 → cell1
        assert math.isnan(float(t[0, 2]))             # z 기본 NaN(물체별 테이블 높이 유지)

    def test_spawn_tensor_explicit_z(self):
        t = build_spawn_tensor([(0.1, 0.2)], repeats=1, z=0.3)
        assert float(t[0, 2]) == pytest.approx(0.3)

    def test_single_spawn_tensor(self):
        t = single_spawn_tensor(0.27, -0.1, float("nan"), num_envs=4)
        assert t.shape == (4, 3)
        assert torch.allclose(t[:, 0], torch.full((4,), 0.27))

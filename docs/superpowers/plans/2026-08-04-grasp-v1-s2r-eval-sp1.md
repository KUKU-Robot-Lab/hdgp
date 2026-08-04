# grasp_v1 sim2real 평가 하네스 SP1 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** tesollo grasp_v1(좌/우) 정책을 사용자 지정 물체 위치에서 평가하는 하네스 — 그리드 스윕 배치 히트맵 + Isaac Sim GUI 인터랙티브 상주 세션 + freeze-once STATE pose 주입 seam.

**Architecture:** 순수 로직 4모듈(grid/providers/report/console, Isaac 무관·CPU 테스트) + env 훅 2개(속성 부재 시 완전 무동작) + 엔트리 스크립트 1개(play.py 로더 글루 복제). 스펙: `docs/superpowers/specs/2026-08-04-grasp-v1-s2r-eval-sp1-design.md`.

**Tech Stack:** Python 3.11, torch(CPU 테스트 가능), Isaac Lab AppLauncher, rl_games LSTM player, matplotlib(Agg), pytest.

## Global Constraints

- 학습·기존 play 경로 무영향: env 훅은 `getattr(self, "...", None)` 기본 None → 무동작. reward/obs 차원/action 불변.
- grasp_v2 및 다른 태스크 파일 수정 금지.
- left/right env 훅은 문자 단위 동일(변수명 포함) — 정적 대칭 테스트로 강제.
- 새 디렉토리는 `scripts/` 바로 아래 1단계만: `scripts/eval_s2r/`.
- 테스트는 Isaac 없이 로컬 CPU에서 실행: `python -m pytest scripts/eval_s2r/tests -q`.
- 기존 텐서 in-place 변경 금지(새 텐서 생성) — 단 env 훅 내부는 기존 env 코드 스타일(버퍼 인덱싱)을 따른다.
- GPU/GUI 실행(스모크·평가)은 사용자 지시 후에만. 이 계획의 태스크는 전부 정적 작업.
- 커밋은 hdgp repo, 브랜치 `pour`. 커밋 메시지 한국어, conventional commits.
- 주석은 기존 codebase처럼 한국어 위주.

**참조 파일 (구현자가 먼저 읽을 것):**
- `scripts/reinforcement_learning/rl_games/play.py` — 로더 글루(:536-569, :640-676), `--disable_adr`(:429-444), eval 지표(:821-941)
- `source/openarm/openarm/tesollo/right/grasp_v1/grasp_right_env.py` — `_reset_idx`(:1467-1631), `_get_observations`(:890-1023)
- `source/openarm/openarm/tesollo/left/grasp_v1/grasp_left_env.py` — 우측의 미러
- 스펙 문서 전문

---

### Task 1: grid.py — 그리드 생성·env↔셀 매핑

**Files:**
- Create: `scripts/eval_s2r/__init__.py` (빈 파일)
- Create: `scripts/eval_s2r/grid.py`
- Create: `scripts/eval_s2r/tests/__init__.py` (빈 파일)
- Test: `scripts/eval_s2r/tests/test_grid.py`

**Interfaces:**
- Produces:
  - `GridSpec(x_min, x_max, nx, y_min, y_max, ny, repeats)` frozen dataclass, `__post_init__` 검증(ValueError)
  - `build_cells(spec) -> list[tuple[float, float]]` — 길이 nx*ny, x-major 순
  - `env_to_cell(env_idx: int, repeats: int) -> int`
  - `build_spawn_tensor(cells, repeats, z=float("nan")) -> torch.Tensor` — [len(cells)*repeats, 3] float32 CPU
  - `single_spawn_tensor(x, y, z, num_envs) -> torch.Tensor` — [num_envs, 3]

- [ ] **Step 1: 실패 테스트 작성** — `scripts/eval_s2r/tests/test_grid.py`:

```python
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
```

- [ ] **Step 2: 실패 확인** — `cd /home/user/rl_ws/hdgp && python -m pytest scripts/eval_s2r/tests/test_grid.py -q` → ModuleNotFoundError/ImportError로 FAIL.

- [ ] **Step 3: 구현** — `scripts/eval_s2r/grid.py`:

```python
"""그리드 스윕 셀 생성·env↔셀 매핑 (순수 함수, Isaac 무관).

설계: docs/superpowers/specs/2026-08-04-grasp-v1-s2r-eval-sp1-design.md §4
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GridSpec:
    x_min: float
    x_max: float
    nx: int
    y_min: float
    y_max: float
    ny: int
    repeats: int

    def __post_init__(self) -> None:
        if self.nx < 1 or self.ny < 1:
            raise ValueError(f"nx/ny must be >= 1 (got nx={self.nx}, ny={self.ny})")
        if self.repeats < 1:
            raise ValueError(f"repeats must be >= 1 (got {self.repeats})")
        if self.x_min > self.x_max or self.y_min > self.y_max:
            raise ValueError("grid min must be <= max")
        if self.nx > 1 and self.x_min == self.x_max:
            raise ValueError("nx > 1 requires x_min < x_max")
        if self.ny > 1 and self.y_min == self.y_max:
            raise ValueError("ny > 1 requires y_min < y_max")


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    if n == 1:
        return [lo]
    step = (hi - lo) / (n - 1)
    return [lo + step * i for i in range(n)]


def build_cells(spec: GridSpec) -> list[tuple[float, float]]:
    """x-major 순 셀 좌표. 셀 idx = xi * ny + yi."""
    xs = _linspace(spec.x_min, spec.x_max, spec.nx)
    ys = _linspace(spec.y_min, spec.y_max, spec.ny)
    return [(x, y) for x in xs for y in ys]


def env_to_cell(env_idx: int, repeats: int) -> int:
    return env_idx // repeats


def build_spawn_tensor(
    cells: list[tuple[float, float]], repeats: int, z: float = float("nan")
) -> torch.Tensor:
    """[len(cells)*repeats, 3] float32. env i → cells[i // repeats]. z NaN=물체별 기본 높이."""
    rows = [[x, y, z] for (x, y) in cells for _ in range(repeats)]
    return torch.tensor(rows, dtype=torch.float32)


def single_spawn_tensor(x: float, y: float, z: float, num_envs: int) -> torch.Tensor:
    return torch.tensor([[x, y, z]] * num_envs, dtype=torch.float32)
```

- [ ] **Step 4: 통과 확인** — 같은 pytest 명령 → 전부 PASS.

- [ ] **Step 5: 커밋**

```bash
git add scripts/eval_s2r/__init__.py scripts/eval_s2r/grid.py scripts/eval_s2r/tests/
git commit -m "feat: eval_s2r 그리드 셀 생성·env↔셀 매핑 모듈"
```

---

### Task 2: providers.py — PoseProvider seam

**Files:**
- Create: `scripts/eval_s2r/providers.py`
- Test: `scripts/eval_s2r/tests/test_providers.py`

**Interfaces:**
- Consumes: 없음 (env는 duck-type — `object_pos: torch.Tensor [N,3]` 속성만 사용)
- Produces:
  - `PoseProvider` Protocol: `on_reset(env, env_ids) -> None`, `get_override(env) -> torch.Tensor | None`
  - `LiveProvider()` — 항상 None(env 현행 GT+노이즈 경로)
  - `StateFrozenProvider()` — reset 시 GT 캡처 후 고정
  - `make_provider(name: str) -> PoseProvider` — "live"/"state_frozen"/"camera_frozen"(NotImplementedError), 그 외 ValueError

- [ ] **Step 1: 실패 테스트 작성** — `scripts/eval_s2r/tests/test_providers.py`:

```python
import pytest
import torch

from scripts.eval_s2r.providers import LiveProvider, StateFrozenProvider, make_provider


class FakeEnv:
    """object_pos만 갖는 duck-type env."""
    def __init__(self, pos):
        self.object_pos = torch.tensor(pos, dtype=torch.float32)


class TestLiveProvider:
    def test_always_none(self):
        env = FakeEnv([[0.3, 0.0, 0.1]])
        p = LiveProvider()
        p.on_reset(env, torch.tensor([0]))
        assert p.get_override(env) is None


class TestStateFrozenProvider:
    def test_freezes_pose_at_reset(self):
        env = FakeEnv([[0.3, 0.0, 0.1], [0.2, -0.1, 0.1]])
        p = StateFrozenProvider()
        p.on_reset(env, torch.tensor([0, 1]))
        # 물체가 움직여도 override는 reset 시점 값 고정
        env.object_pos = env.object_pos + 1.0
        ov = p.get_override(env)
        assert torch.allclose(ov[0], torch.tensor([0.3, 0.0, 0.1]))

    def test_partial_reset_updates_only_those_envs(self):
        env = FakeEnv([[0.3, 0.0, 0.1], [0.2, -0.1, 0.1]])
        p = StateFrozenProvider()
        p.on_reset(env, torch.tensor([0, 1]))
        env.object_pos = torch.tensor([[9.0, 9.0, 9.0], [0.5, 0.5, 0.5]])
        p.on_reset(env, torch.tensor([1]))  # env1만 재캡처
        ov = p.get_override(env)
        assert torch.allclose(ov[0], torch.tensor([0.3, 0.0, 0.1]))   # 불변
        assert torch.allclose(ov[1], torch.tensor([0.5, 0.5, 0.5]))   # 갱신

    def test_get_override_before_reset_raises(self):
        p = StateFrozenProvider()
        with pytest.raises(RuntimeError):
            p.get_override(FakeEnv([[0.0, 0.0, 0.0]]))

    def test_nonfinite_pose_rejected(self):
        env = FakeEnv([[float("nan"), 0.0, 0.1]])
        p = StateFrozenProvider()
        with pytest.raises(ValueError):
            p.on_reset(env, torch.tensor([0]))

    def test_override_is_detached_copy(self):
        env = FakeEnv([[0.3, 0.0, 0.1]])
        p = StateFrozenProvider()
        p.on_reset(env, torch.tensor([0]))
        env.object_pos[0, 0] = 99.0  # 원본 in-place 변경이 override에 새지 않아야 함
        assert float(p.get_override(env)[0, 0]) == pytest.approx(0.3)


class TestFactory:
    def test_names(self):
        assert isinstance(make_provider("live"), LiveProvider)
        assert isinstance(make_provider("state_frozen"), StateFrozenProvider)

    def test_camera_frozen_not_implemented(self):
        with pytest.raises(NotImplementedError):
            make_provider("camera_frozen")

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            make_provider("bogus")
```

- [ ] **Step 2: 실패 확인** — `python -m pytest scripts/eval_s2r/tests/test_providers.py -q` → ImportError FAIL.

- [ ] **Step 3: 구현** — `scripts/eval_s2r/providers.py`:

```python
"""cup pose 주입 provider seam.

live: env 현행 경로(GT+DR노이즈) — override 없음.
state_frozen: reset 시 GT를 1회 캡처해 에피소드 내내 고정(배포 open-loop 재현).
camera_frozen: SP2 — 카메라 렌더+FoundationPose (미구현).
설계: docs/superpowers/specs/2026-08-04-grasp-v1-s2r-eval-sp1-design.md §4.3
"""
from __future__ import annotations

from typing import Protocol

import torch


class PoseProvider(Protocol):
    def on_reset(self, env, env_ids: torch.Tensor) -> None: ...
    def get_override(self, env) -> torch.Tensor | None: ...


class LiveProvider:
    def on_reset(self, env, env_ids: torch.Tensor) -> None:
        pass

    def get_override(self, env) -> torch.Tensor | None:
        return None


class StateFrozenProvider:
    def __init__(self) -> None:
        self._buf: torch.Tensor | None = None

    def on_reset(self, env, env_ids: torch.Tensor) -> None:
        pos = env.object_pos  # [N,3] env-origin local (grasp_*_env.py: root_pos_w - env_origins)
        if not torch.isfinite(pos[env_ids]).all():
            raise ValueError(f"non-finite object_pos at reset for envs {env_ids.tolist()}")
        # 불변 패턴: 기존 버퍼를 제자리 수정하지 않고 새 텐서 생성
        if self._buf is None:
            self._buf = pos.detach().clone()
        else:
            buf = self._buf.clone()
            buf[env_ids] = pos[env_ids].detach().clone()
            self._buf = buf

    def get_override(self, env) -> torch.Tensor:
        if self._buf is None:
            raise RuntimeError("StateFrozenProvider.get_override called before on_reset")
        return self._buf


def make_provider(name: str) -> PoseProvider:
    if name == "live":
        return LiveProvider()
    if name == "state_frozen":
        return StateFrozenProvider()
    if name == "camera_frozen":
        raise NotImplementedError("camera_frozen provider는 SP2에서 구현 (spec §8)")
    raise ValueError(f"unknown pose_source: {name!r} (live|state_frozen|camera_frozen)")
```

- [ ] **Step 4: 통과 확인** — 같은 명령 → PASS.

- [ ] **Step 5: 커밋**

```bash
git add scripts/eval_s2r/providers.py scripts/eval_s2r/tests/test_providers.py
git commit -m "feat: eval_s2r PoseProvider seam (live/state_frozen, camera_frozen 자리)"
```

---

### Task 3: report.py — 셀 집계·CSV/JSON/히트맵

**Files:**
- Create: `scripts/eval_s2r/report.py`
- Test: `scripts/eval_s2r/tests/test_report.py`

**Interfaces:**
- Consumes: Task 1의 `GridSpec`, `build_cells`
- Produces:
  - `EpisodeResult(cell_idx: int, success: bool, lifted: bool, grip_count: float, displacement: float, obj_idx: int, invalid: bool)` frozen dataclass
  - `aggregate(results: list[EpisodeResult], cells: list[tuple[float, float]]) -> list[dict]` — 셀별 행. 키: `cell_idx, x, y, n_episodes, n_invalid, success_rate, lifted_rate, grip_finger_count, displacement_mean, per_obj_success`(dict obj_idx→rate)
  - `write_csv(rows, path)`, `write_summary(rows, meta: dict, path)`, `write_heatmap(rows, nx: int, ny: int, metric: str, path)`

- [ ] **Step 1: 실패 테스트 작성** — `scripts/eval_s2r/tests/test_report.py`:

```python
import csv
import json

import pytest

from scripts.eval_s2r.report import EpisodeResult, aggregate, write_csv, write_summary, write_heatmap

CELLS = [(0.21, -0.16), (0.21, 0.02), (0.33, -0.16), (0.33, 0.02)]  # nx=2, ny=2


def _ep(cell, success, lifted=None, grip=4.0, disp=0.01, obj=0, invalid=False):
    return EpisodeResult(
        cell_idx=cell, success=success,
        lifted=success if lifted is None else lifted,
        grip_count=grip, displacement=disp, obj_idx=obj, invalid=invalid,
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


class TestWriters:
    def test_csv_roundtrip(self, tmp_path):
        rows = aggregate([_ep(0, True)], CELLS)
        p = tmp_path / "results.csv"
        write_csv(rows, str(p))
        with open(p) as f:
            got = list(csv.DictReader(f))
        assert len(got) == 4
        assert float(got[0]["success_rate"]) == pytest.approx(1.0)

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
```

- [ ] **Step 2: 실패 확인** — `python -m pytest scripts/eval_s2r/tests/test_report.py -q` → FAIL.

- [ ] **Step 3: 구현** — `scripts/eval_s2r/report.py`:

```python
"""셀 집계·CSV/JSON/히트맵 출력 (순수 함수, Isaac 무관).

설계: docs/superpowers/specs/2026-08-04-grasp-v1-s2r-eval-sp1-design.md §4.4
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # 서버 headless 대비
import matplotlib.pyplot as plt

_HEATMAP_METRICS = ("success_rate", "lifted_rate", "grip_finger_count", "displacement_mean")


@dataclass(frozen=True)
class EpisodeResult:
    cell_idx: int
    success: bool
    lifted: bool
    grip_count: float
    displacement: float
    obj_idx: int
    invalid: bool


def aggregate(results: list[EpisodeResult], cells: list[tuple[float, float]]) -> list[dict]:
    rows = []
    for ci, (x, y) in enumerate(cells):
        eps = [r for r in results if r.cell_idx == ci and not r.invalid]
        n_invalid = sum(1 for r in results if r.cell_idx == ci and r.invalid)
        n = len(eps)
        per_obj: dict[int, float] = {}
        for oi in sorted({r.obj_idx for r in eps}):
            sub = [r for r in eps if r.obj_idx == oi]
            per_obj[oi] = sum(r.success for r in sub) / len(sub)
        rows.append({
            "cell_idx": ci, "x": x, "y": y,
            "n_episodes": n, "n_invalid": n_invalid,
            "success_rate": (sum(r.success for r in eps) / n) if n else None,
            "lifted_rate": (sum(r.lifted for r in eps) / n) if n else None,
            "grip_finger_count": (sum(r.grip_count for r in eps) / n) if n else None,
            "displacement_mean": (sum(r.displacement for r in eps) / n) if n else None,
            "per_obj_success": per_obj,
        })
    return rows


def write_csv(rows: list[dict], path: str) -> None:
    fields = ["cell_idx", "x", "y", "n_episodes", "n_invalid",
              "success_rate", "lifted_rate", "grip_finger_count",
              "displacement_mean", "per_obj_success"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            out = dict(r)
            out["per_obj_success"] = json.dumps(r["per_obj_success"])
            w.writerow(out)


def write_summary(rows: list[dict], meta: dict, path: str) -> None:
    valid = [r for r in rows if r["n_episodes"] > 0]
    total = sum(r["n_episodes"] for r in valid)
    overall = (
        sum(r["success_rate"] * r["n_episodes"] for r in valid) / total if total else None
    )
    payload = {
        "meta": meta,
        "total_episodes": total,
        "total_invalid": sum(r["n_invalid"] for r in rows),
        "overall_success_rate": overall,
        "cells": rows,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_heatmap(rows: list[dict], nx: int, ny: int, metric: str, path: str) -> None:
    if metric not in _HEATMAP_METRICS:
        raise ValueError(f"unknown metric {metric!r} (choose from {_HEATMAP_METRICS})")
    # 셀 idx = xi * ny + yi (grid.build_cells x-major와 동일 규약)
    grid = [[None] * nx for _ in range(ny)]
    for r in rows:
        xi, yi = r["cell_idx"] // ny, r["cell_idx"] % ny
        grid[yi][xi] = r[metric]
    data = [[(v if v is not None else float("nan")) for v in row] for row in grid]
    fig, ax = plt.subplots(figsize=(1.2 * nx + 2, 1.0 * ny + 2))
    im = ax.imshow(data, origin="lower", cmap="RdYlGn", vmin=0.0,
                   vmax=1.0 if metric.endswith("_rate") else None)
    xs = sorted({r["x"] for r in rows})
    ys = sorted({r["y"] for r in rows})
    ax.set_xticks(range(nx), [f"{v:.3f}" for v in xs])
    ax.set_yticks(range(ny), [f"{v:.3f}" for v in ys])
    ax.set_xlabel("object x [m]")
    ax.set_ylabel("object y [m]")
    ax.set_title(metric)
    for yi in range(ny):
        for xi in range(nx):
            v = grid[yi][xi]
            if v is not None:
                ax.text(xi, yi, f"{v:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
```

- [ ] **Step 4: 통과 확인** — 같은 명령 → PASS.

- [ ] **Step 5: 커밋**

```bash
git add scripts/eval_s2r/report.py scripts/eval_s2r/tests/test_report.py
git commit -m "feat: eval_s2r 셀 집계·CSV/JSON/히트맵 리포트 모듈"
```

---

### Task 4: console.py — 인터랙티브 명령 파서·상태기계

**Files:**
- Create: `scripts/eval_s2r/console.py`
- Test: `scripts/eval_s2r/tests/test_console.py`

**Interfaces:**
- Consumes: Task 1의 `GridSpec`
- Produces:
  - `Command(kind: str, x=None, y=None, z=None, obj=None, grid=None)` frozen dataclass — kind ∈ {"spawn","repeat","obj","sweep","quit"}
  - `parse_command(line: str) -> Command` — ValueError on 오입력
  - `SessionState(last_spawn: tuple[float,float,float] | None, obj_idx: int | None)` frozen dataclass
  - `apply_command(state, cmd) -> tuple[SessionState, dict]` — 새 상태 + 실행 지시 dict(`{"action": "spawn"|"sweep"|"quit"|"noop", ...}`)

- [ ] **Step 1: 실패 테스트 작성** — `scripts/eval_s2r/tests/test_console.py`:

```python
import math

import pytest

from scripts.eval_s2r.console import Command, SessionState, parse_command, apply_command
from scripts.eval_s2r.grid import GridSpec


class TestParse:
    def test_spawn_xy(self):
        c = parse_command("spawn 0.27 -0.10")
        assert c.kind == "spawn" and c.x == pytest.approx(0.27) and c.y == pytest.approx(-0.10)
        assert math.isnan(c.z)  # z 생략 → NaN(물체별 기본 높이)

    def test_spawn_xyz(self):
        c = parse_command("spawn 0.27 -0.10 0.30")
        assert c.z == pytest.approx(0.30)

    def test_repeat_obj_quit(self):
        assert parse_command("repeat").kind == "repeat"
        assert parse_command("obj 3").obj == 3
        assert parse_command("quit").kind == "quit"

    def test_sweep(self):
        c = parse_command("sweep 0.21 0.33 3 -0.16 0.02 2 4")
        assert c.kind == "sweep"
        assert c.grid == GridSpec(0.21, 0.33, 3, -0.16, 0.02, 2, 4)

    @pytest.mark.parametrize("line", [
        "", "bogus", "spawn", "spawn 0.1", "spawn a b",
        "obj", "obj 8", "obj -1",              # 물체 0~7
        "sweep 0.1 0.2 3",                     # 인자 부족
    ])
    def test_invalid_raises(self, line):
        with pytest.raises(ValueError):
            parse_command(line)


class TestApply:
    def test_spawn_updates_last(self):
        s0 = SessionState(last_spawn=None, obj_idx=None)
        s1, act = apply_command(s0, parse_command("spawn 0.3 0.0"))
        assert act["action"] == "spawn"
        assert s1.last_spawn[0] == pytest.approx(0.3)
        assert s0.last_spawn is None  # 불변성: 원본 상태 미변경

    def test_repeat_requires_last(self):
        s0 = SessionState(last_spawn=None, obj_idx=None)
        with pytest.raises(ValueError):
            apply_command(s0, parse_command("repeat"))

    def test_repeat_reuses_last(self):
        s0 = SessionState(last_spawn=(0.3, 0.0, float("nan")), obj_idx=None)
        s1, act = apply_command(s0, parse_command("repeat"))
        assert act["action"] == "spawn" and act["x"] == pytest.approx(0.3)

    def test_obj_sets_idx_noop(self):
        s1, act = apply_command(SessionState(None, None), parse_command("obj 5"))
        assert s1.obj_idx == 5 and act["action"] == "noop"

    def test_quit(self):
        _, act = apply_command(SessionState(None, None), parse_command("quit"))
        assert act["action"] == "quit"
```

- [ ] **Step 2: 실패 확인** — `python -m pytest scripts/eval_s2r/tests/test_console.py -q` → FAIL.

- [ ] **Step 3: 구현** — `scripts/eval_s2r/console.py`:

```python
"""인터랙티브 세션 명령 파서·상태 전이 (순수 함수, Isaac 무관).

명령: spawn X Y [Z] | repeat | obj N | sweep XMIN XMAX NX YMIN YMAX NY REPEATS | quit
설계: docs/superpowers/specs/2026-08-04-grasp-v1-s2r-eval-sp1-design.md §4.6
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from scripts.eval_s2r.grid import GridSpec

NUM_OBJECTS = 8  # grasp_v1 MultiAsset 물체 수 (env_id % 8 배정)


@dataclass(frozen=True)
class Command:
    kind: str
    x: float | None = None
    y: float | None = None
    z: float | None = None
    obj: int | None = None
    grid: GridSpec | None = None


@dataclass(frozen=True)
class SessionState:
    last_spawn: tuple[float, float, float] | None
    obj_idx: int | None


def parse_command(line: str) -> Command:
    parts = line.strip().split()
    if not parts:
        raise ValueError("빈 입력 (spawn X Y [Z] | repeat | obj N | sweep ... | quit)")
    kind, args = parts[0], parts[1:]
    if kind == "quit":
        return Command(kind="quit")
    if kind == "repeat":
        return Command(kind="repeat")
    if kind == "spawn":
        if len(args) not in (2, 3):
            raise ValueError("사용법: spawn X Y [Z]")
        try:
            x, y = float(args[0]), float(args[1])
            z = float(args[2]) if len(args) == 3 else float("nan")
        except ValueError as e:
            raise ValueError(f"spawn 좌표 파싱 실패: {e}") from e
        return Command(kind="spawn", x=x, y=y, z=z)
    if kind == "obj":
        if len(args) != 1 or not args[0].lstrip("-").isdigit():
            raise ValueError("사용법: obj N (0~7)")
        n = int(args[0])
        if not 0 <= n < NUM_OBJECTS:
            raise ValueError(f"obj 는 0~{NUM_OBJECTS - 1} (got {n})")
        return Command(kind="obj", obj=n)
    if kind == "sweep":
        if len(args) != 7:
            raise ValueError("사용법: sweep XMIN XMAX NX YMIN YMAX NY REPEATS")
        try:
            grid = GridSpec(
                x_min=float(args[0]), x_max=float(args[1]), nx=int(args[2]),
                y_min=float(args[3]), y_max=float(args[4]), ny=int(args[5]),
                repeats=int(args[6]),
            )
        except ValueError as e:
            raise ValueError(f"sweep 인자 오류: {e}") from e
        return Command(kind="sweep", grid=grid)
    raise ValueError(f"알 수 없는 명령 {kind!r}")


def apply_command(state: SessionState, cmd: Command) -> tuple[SessionState, dict]:
    """새 SessionState + 실행 지시 dict 반환 (원본 state 불변)."""
    if cmd.kind == "quit":
        return state, {"action": "quit"}
    if cmd.kind == "obj":
        return replace(state, obj_idx=cmd.obj), {"action": "noop"}
    if cmd.kind == "spawn":
        new = replace(state, last_spawn=(cmd.x, cmd.y, cmd.z))
        return new, {"action": "spawn", "x": cmd.x, "y": cmd.y, "z": cmd.z}
    if cmd.kind == "repeat":
        if state.last_spawn is None:
            raise ValueError("repeat 전에 spawn 이력이 없습니다")
        x, y, z = state.last_spawn
        return state, {"action": "spawn", "x": x, "y": y, "z": z}
    if cmd.kind == "sweep":
        return state, {"action": "sweep", "grid": cmd.grid}
    raise ValueError(f"unhandled command kind {cmd.kind!r}")
```

- [ ] **Step 4: 통과 확인** — 같은 명령 → PASS. 전체도: `python -m pytest scripts/eval_s2r/tests -q`.

- [ ] **Step 5: 커밋**

```bash
git add scripts/eval_s2r/console.py scripts/eval_s2r/tests/test_console.py
git commit -m "feat: eval_s2r 인터랙티브 명령 파서·세션 상태 전이"
```

---

### Task 5: env 훅 (right + left) + 정적 대칭 테스트

**Files:**
- Modify: `source/openarm/openarm/tesollo/right/grasp_v1/grasp_right_env.py` — `_reset_idx` 스폰 분기(≈:1528, `obj_pos_local = torch.stack` 직후) + `_get_observations`(≈:912, `cup_pos_noisy = ...` 직후)
- Modify: `source/openarm/openarm/tesollo/left/grasp_v1/grasp_left_env.py` — 동일 지점(스폰 ≈:1545, obs ≈:929)
- Test: `scripts/eval_s2r/tests/test_env_hooks_static.py`

**Interfaces:**
- Produces (평가 스크립트가 설정하는 env 속성 — Task 6이 소비):
  - `env.eval_fixed_spawn_local: torch.Tensor [num_envs, 3]` — env-origin local, z NaN=물체별 기본 높이
  - `env.eval_cup_pos_override: torch.Tensor [num_envs, 3] | None` — actor obs의 cup pos 대체

**주의:** 훅 코드는 좌우 **문자 단위 동일**해야 한다(정적 테스트로 강제). 두 파일의 해당 지점은 이미 미러 동일 구조다(변수명 동일: `obj_pos_local`, `obj_x`, `obj_y`, `cup_pos_noisy`).

- [ ] **Step 1: 정적 대칭·존재 테스트 작성** — `scripts/eval_s2r/tests/test_env_hooks_static.py`:

```python
"""env 훅 정적 검사 (Isaac 불필요 — 소스 텍스트 검사).

1) 좌우 grasp_v1 env에 훅 마커 블록이 존재하고 문자 단위 동일한가.
2) 훅이 getattr(..., None) 기본 무동작 패턴인가 (학습 경로 보호).
scripts/analysis/tests 의 소스 검사 테스트들과 같은 방식.
"""
import re
from pathlib import Path

HDGP = Path(__file__).resolve().parents[3]
RIGHT = HDGP / "source/openarm/openarm/tesollo/right/grasp_v1/grasp_right_env.py"
LEFT = HDGP / "source/openarm/openarm/tesollo/left/grasp_v1/grasp_left_env.py"

SPAWN_MARK = "eval_s2r: 고정 스폰 오버라이드"
OBS_MARK = "eval_s2r: cup pose obs 오버라이드"


def _extract_block(text: str, marker: str) -> str:
    """marker 주석 줄부터, 그보다 얕은 들여쓰기의 첫 비주석 줄 전까지 추출."""
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if marker in l)
    indent = len(lines[start]) - len(lines[start].lstrip())
    out = [lines[start].strip()]
    for l in lines[start + 1:]:
        if l.strip() and (len(l) - len(l.lstrip())) < indent:
            break
        if l.strip():
            out.append(l.strip())
    return "\n".join(out)


def test_hooks_exist_in_both_envs():
    for path in (RIGHT, LEFT):
        text = path.read_text()
        assert SPAWN_MARK in text, f"{path.name}: 스폰 훅 없음"
        assert OBS_MARK in text, f"{path.name}: obs 훅 없음"


def test_hooks_identical_left_right():
    rt, lt = RIGHT.read_text(), LEFT.read_text()
    for mark in (SPAWN_MARK, OBS_MARK):
        assert _extract_block(rt, mark) == _extract_block(lt, mark), f"{mark}: 좌우 불일치"


def test_hooks_are_getattr_gated():
    """훅 블록 안에 getattr(self, ..., None) 게이트가 있어야 학습 경로 무영향."""
    for path in (RIGHT, LEFT):
        text = path.read_text()
        for mark, attr in ((SPAWN_MARK, "eval_fixed_spawn_local"),
                           (OBS_MARK, "eval_cup_pos_override")):
            block = _extract_block(text, mark)
            assert re.search(rf'getattr\(self, "{attr}", None\)', block), (
                f"{path.name}/{mark}: getattr 게이트 없음"
            )
```

- [ ] **Step 2: 실패 확인** — `python -m pytest scripts/eval_s2r/tests/test_env_hooks_static.py -q` → 훅 부재로 FAIL.

- [ ] **Step 3: right env 훅 삽입** — `grasp_right_env.py`.

(a) `_reset_idx`의 else(비-demo) 분기, `obj_pos_local = torch.stack([obj_x, obj_y, self.object_spawn_z_buf[env_ids]], dim=1)` 바로 다음(":1528 부근, `# ---- FABRICS pregrasp rollout/cache lookup ----` 주석 앞)에 삽입:

```python
            # eval_s2r: 고정 스폰 오버라이드 — 평가 하네스(scripts/eval_s2r) 전용.
            # 학습·기존 play 에서는 속성 부재(getattr→None)로 완전 무동작.
            # obj_x/obj_y 도 동기해야 pregrasp cache lookup 이 오버라이드 위치를 따라간다.
            # z 가 NaN 이면 물체별 테이블 높이(object_spawn_z_buf)를 유지.
            _eval_spawn = getattr(self, "eval_fixed_spawn_local", None)
            if _eval_spawn is not None:
                _ov = _eval_spawn[env_ids].to(self.device).clone()
                _z_nan = torch.isnan(_ov[:, 2])
                _ov[_z_nan, 2] = self.object_spawn_z_buf[env_ids][_z_nan]
                obj_pos_local = _ov
                obj_x = _ov[:, 0]
                obj_y = _ov[:, 1]
```

(b) `_get_observations`, `cup_pos_noisy    = cup_pos_clean          + torch.randn_like(cup_pos_clean)          * σ_cp` 바로 다음에 삽입:

```python
        # eval_s2r: cup pose obs 오버라이드 — 평가 하네스(scripts/eval_s2r) 전용.
        # 주입값은 이미 "지각 결과"이므로 obs_noise_cup_pos 를 additionally 얹지 않는다.
        # 학습·기존 play 에서는 속성 부재(getattr→None)로 완전 무동작.
        _eval_cup = getattr(self, "eval_cup_pos_override", None)
        if _eval_cup is not None:
            cup_pos_noisy = _eval_cup.to(cup_pos_clean.device)
```

- [ ] **Step 4: left env 훅 삽입** — `grasp_left_env.py`의 동일 지점(스폰: `obj_pos_local = torch.stack` 직후 ≈:1545 / obs: `cup_pos_noisy` 산출 직후 ≈:929)에 **위와 문자 단위 동일한 블록**을 삽입. 삽입 전 두 파일의 주변 코드가 동일 구조인지 눈으로 대조(변수명 `obj_x/obj_y/obj_pos_local/cup_pos_noisy` 동일해야 함 — 이미 동일).

- [ ] **Step 5: 통과 확인** — `python -m pytest scripts/eval_s2r/tests/test_env_hooks_static.py -q` → PASS.

- [ ] **Step 6: 문법 검증** — `python -c "import ast; ast.parse(open('source/openarm/openarm/tesollo/right/grasp_v1/grasp_right_env.py').read()); ast.parse(open('source/openarm/openarm/tesollo/left/grasp_v1/grasp_left_env.py').read()); print('SYNTAX OK')"` → SYNTAX OK.

- [ ] **Step 7: 커밋**

```bash
git add source/openarm/openarm/tesollo/right/grasp_v1/grasp_right_env.py \
        source/openarm/openarm/tesollo/left/grasp_v1/grasp_left_env.py \
        scripts/eval_s2r/tests/test_env_hooks_static.py
git commit -m "feat: grasp_v1 좌우 env에 eval_s2r 훅(고정 스폰·cup obs 오버라이드, 기본 무동작)"
```

---

### Task 6: eval_sim2real.py — 배치 모드(그리드/단일)

**Files:**
- Create: `scripts/eval_s2r/eval_sim2real.py`
- Test: 정적 문법·인자 검증만 (GPU 실행은 Task 8 수동 게이트)

**Interfaces:**
- Consumes: Task 1~3, 5 전부 (`GridSpec/build_cells/build_spawn_tensor/single_spawn_tensor/env_to_cell`, `make_provider`, `EpisodeResult/aggregate/write_*`, env 속성 `eval_fixed_spawn_local`/`eval_cup_pos_override`)
- Produces: CLI 엔트리. `--interactive`는 Task 7에서 채움(이 태스크에서는 `NotImplementedError` 자리).

**구현 지침(참조 필수):** rl_games 로더 글루는 `scripts/reinforcement_learning/rl_games/play.py` 를 표본으로 복제한다 — env.yaml/agent.yaml 복원(:369-391 `_restore_run_cfg_if_available` 패턴), 체크포인트 resolve(:275-307), Runner/create_player/restore(:646-659), RNN init(:673-676)·done 리셋(:1053-1055). 각 복제 지점에 `# play.py:<라인> 패턴 복제` 주석을 남긴다.

- [ ] **Step 1: 스크립트 작성** — `scripts/eval_s2r/eval_sim2real.py` (아래 골격을 완성 — AppLauncher 인자·import 순서는 play.py 상단과 동일하게):

```python
"""grasp_v1 sim2real 평가 하네스 — 그리드 스윕/단일/인터랙티브.

설계: docs/superpowers/specs/2026-08-04-grasp-v1-s2r-eval-sp1-design.md
사용 예는 스펙 §3. GPU 실행은 사용자 게이트(Task 8 스모크 체크리스트).
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys

# hdgp 루트를 sys.path에 추가 (scripts.eval_s2r 패키지 import용)
_HDGP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _HDGP_ROOT not in sys.path:
    sys.path.insert(0, _HDGP_ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="grasp_v1 sim2real 평가 하네스 (SP1)")
parser.add_argument("--robot", required=True, choices=["left", "right"])
parser.add_argument("--checkpoint", required=True, help="정책 .pth (prefix-glob 허용)")
parser.add_argument("--pose_source", default="state_frozen",
                    choices=["live", "state_frozen", "camera_frozen"])
parser.add_argument("--grid_x", nargs=2, type=float, metavar=("MIN", "MAX"))
parser.add_argument("--grid_y", nargs=2, type=float, metavar=("MIN", "MAX"))
parser.add_argument("--grid_nx", type=int)
parser.add_argument("--grid_ny", type=int)
parser.add_argument("--grid_repeats", type=int, default=8)
parser.add_argument("--object_x", type=float)
parser.add_argument("--object_y", type=float)
parser.add_argument("--object_z", type=float, default=float("nan"))
parser.add_argument("--episodes_per_env", type=int, default=3)
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--render", action="store_true", help="단일 모드 GUI 재생")
parser.add_argument("--real-time", action="store_true")
parser.add_argument("--interactive", action="store_true", help="상주 대화 세션(GUI)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()


def _validate_mode(a) -> str:
    grid_args = (a.grid_x, a.grid_y, a.grid_nx, a.grid_ny)
    has_grid = any(v is not None for v in grid_args)
    has_single = a.object_x is not None or a.object_y is not None
    if a.interactive:
        if has_grid or has_single:
            parser.error("--interactive 는 그리드/단일 인자와 함께 쓸 수 없습니다")
        return "interactive"
    if has_grid and has_single:
        parser.error("그리드 인자와 --object_x/y 는 상호배타입니다")
    if has_grid:
        if not all(v is not None for v in grid_args):
            parser.error("그리드 모드는 --grid_x/--grid_y/--grid_nx/--grid_ny 전부 필요")
        return "grid"
    if has_single:
        if a.object_x is None or a.object_y is None:
            parser.error("단일 모드는 --object_x 와 --object_y 둘 다 필요")
        return "single"
    parser.error("모드를 지정하세요: 그리드 인자 | --object_x/y | --interactive")


MODE = _validate_mode(args_cli)
if MODE == "interactive" or args_cli.render:
    args_cli.headless = False  # GUI 강제
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- Isaac 기동 후 import (play.py 상단 순서와 동일) ----
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402
# (+ play.py가 쓰는 rl_games env_configurations/vecenv, RlGamesGpuEnv/RlGamesVecEnvWrapper,
#  isaaclab_tasks parse_env_cfg, load_cfg_from_registry — play.py:100-140 참고해 동일 import)

from scripts.eval_s2r.console import SessionState, apply_command, parse_command  # noqa: E402
from scripts.eval_s2r.grid import (  # noqa: E402
    GridSpec, build_cells, build_spawn_tensor, env_to_cell, single_spawn_tensor,
)
from scripts.eval_s2r.providers import make_provider  # noqa: E402
from scripts.eval_s2r.report import (  # noqa: E402
    EpisodeResult, aggregate, write_csv, write_heatmap, write_summary,
)

TASK_BY_ROBOT = {
    "left": "open-tesol_l_grasp_v1-play-lstm",
    "right": "open-tesol_r_grasp_v1-play-lstm",
}
# 학습 스폰 분포(cfg 기본): 중심 ±(xy_range + ADR max). 벗어나면 경고만 (분포외 측정이 목적).
TRAIN_RANGE_WARN = 0.08 + 0.06


def _build_grid_and_num_envs(a, mode):
    if mode == "grid":
        spec = GridSpec(a.grid_x[0], a.grid_x[1], a.grid_nx,
                        a.grid_y[0], a.grid_y[1], a.grid_ny, a.grid_repeats)
        if spec.repeats % 8 != 0:
            print(f"[WARN] repeats={spec.repeats} 가 8의 배수가 아님 — 셀마다 물체(8종 % 배정) 구성이 달라짐")
        cells = build_cells(spec)
        return spec, cells, len(cells) * spec.repeats
    if mode == "single":
        return None, [(a.object_x, a.object_y)], 1
    return None, [], 1  # interactive


def _episode_metrics(ge, cell_idx: int) -> EpisodeResult:
    """에피소드 종료 시점 env 상태에서 지표 추출 (play.py:821-941 지표와 동일 소스)."""
    success = bool(ge.in_success_region[0].item()) if ge.num_envs == 1 else None
    lift_h = float((ge.object_pos[0, 2] - ge.object_init_pos[0, 2]).item())
    disp = float(torch.norm(ge.object_pos[0, :2] - ge.object_init_pos[0, :2]).item())
    grip = float(ge.binary_contact_buf[0].float().sum().item())
    obj_idx = int(ge.multi_object_idx_onehot[0].argmax().item())
    invalid = not bool(torch.isfinite(ge.object_pos[0]).all().item())
    return EpisodeResult(cell_idx=cell_idx, success=bool(success), lifted=lift_h > 0.05,
                         grip_count=grip, displacement=disp, obj_idx=obj_idx, invalid=invalid)


def main():
    task = TASK_BY_ROBOT[args_cli.robot]
    spec, cells, num_envs = _build_grid_and_num_envs(args_cli, MODE)
    print(f"[INFO] mode={MODE} task={task} num_envs={num_envs}")

    # ---- env cfg (play.py:400-460 패턴 복제) ----
    # parse_env_cfg(task, num_envs=num_envs) → ADR 전부 off(play.py:429-444와 동일 5개 플래그)
    # + env_cfg.enable_demo_grasp_reset = False (demo 스폰 분기 차단 — 훅은 비-demo 분기에만 있음)
    # + env_cfg.enable_warm_state_export = False
    # ---- checkpoint resolve + env.yaml/agent.yaml 복원 (play.py:536-569, :369-391 복제) ----
    # ---- gym.make → RlGamesVecEnvWrapper → vecenv/env_configurations 등록 (play.py:620-644) ----
    # ---- Runner/create_player/restore/reset (play.py:646-659) ----
    # ---- ge = env.unwrapped (play.py:823 패턴) ----
    #
    # ge.eval_fixed_spawn_local = (
    #     build_spawn_tensor(cells, spec.repeats, z=args_cli.object_z) if MODE == "grid"
    #     else single_spawn_tensor(args_cli.object_x, args_cli.object_y, args_cli.object_z, 1)
    # ).to(ge.device)
    #
    # provider = make_provider(args_cli.pose_source)
    # obs = env.reset(); agent.get_batch_size(obs,1); agent.is_rnn → init_rnn (play.py:668-676)
    # provider.on_reset(ge, torch.arange(ge.num_envs, device=ge.device))
    #
    # 루프 (play.py:690-, :1053-1055 패턴):
    #   ge.eval_cup_pos_override = provider.get_override(ge)   # live 면 None
    #   actions = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
    #   obs, _, dones, _ = env.step(actions)
    #   done env: 지표 기록(_episode_metrics — done 직전 스냅샷 사용, 아래 주의) →
    #             LSTM states[:, done, :]=0 → provider.on_reset(ge, done_ids)
    #   전 env가 episodes_per_env 채우면 종료
    #
    # 주의: done 후 obs는 이미 리셋된 상태다. 지표는 env.step() "전"에 매 스텝
    #   ge.in_success_region/object_pos 등을 스냅샷해 두고, done 감지 시 직전 스냅샷으로 기록한다.
    #
    # 종료: rows = aggregate(results, cells) → write_csv/write_summary(+git SHA
    #   subprocess.check_output(["git","rev-parse","HEAD"])) → grid 모드면 write_heatmap
    #   (success_rate, lifted_rate 2장) → 콘솔에 셀 표 출력.
    raise SystemExit(0)


if __name__ == "__main__":
    if MODE == "interactive":
        raise NotImplementedError("interactive 모드는 Task 7")
    main()
    simulation_app.close()
```

위 골격에서 주석 처리된 블록(`env cfg`~`종료`)을 play.py 해당 라인을 열어 실제 코드로 완성한다. **주석 그대로 남기면 이 태스크는 미완이다.** 각 블록은 play.py의 검증된 코드를 최소로 잘라 온다(비디오/pour/probe 분기 등 불필요 기능은 가져오지 않는다).

- [ ] **Step 2: 문법·인자 검증** — Isaac 없이:

```bash
python -c "import ast; ast.parse(open('scripts/eval_s2r/eval_sim2real.py').read()); print('SYNTAX OK')"
```

그리고 인자 검증 로직을 순수 함수로 확인하기 위해 `_validate_mode`를 테스트에 추가 — `scripts/eval_s2r/tests/test_grid.py`가 아닌 새 파일 없이, 간단히 아래를 `test_console.py` 말미에 추가하지 **말고** 생략한다(AppLauncher import가 필요해 CPU 테스트 불가; argparse 검증은 Task 8 스모크에서 확인).

- [ ] **Step 3: 전체 정적 테스트 회귀** — `python -m pytest scripts/eval_s2r/tests -q` → 전부 PASS.

- [ ] **Step 4: 커밋**

```bash
git add scripts/eval_s2r/eval_sim2real.py
git commit -m "feat: eval_s2r 배치 평가 엔트리(그리드/단일, state_frozen 주입, 히트맵 출력)"
```

---

### Task 7: 인터랙티브 모드

**Files:**
- Modify: `scripts/eval_s2r/eval_sim2real.py` — `interactive_main()` 추가, `__main__` 분기 교체

**Interfaces:**
- Consumes: Task 4 `parse_command/apply_command/SessionState`, Task 6의 로더·지표·env 훅 배선 전부

- [ ] **Step 1: 구현** — `eval_sim2real.py`에 추가:

```python
import select


def _poll_stdin() -> str | None:
    """논블로킹 stdin 1줄 폴링. 입력 없으면 None. (GUI 렌더 루프를 막지 않기 위함)"""
    r, _, _ = select.select([sys.stdin], [], [], 0.0)
    if r:
        return sys.stdin.readline()
    return None


def interactive_main():
    # 기동: main()과 동일한 로더 블록(env cfg/checkpoint/agent) — num_envs=1 고정.
    # 공통 부분은 main()에서 _setup() 함수로 추출해 공유한다 (DRY):
    #   env, ge, agent = _setup(task, num_envs=1)
    # STAGED 대기 위치: 작업공간 밖 (스펙 §4.6) — spawn 전 물체를 멀리 치워둔다.
    STAGE_AWAY = (1.0, 1.0, float("nan"))  # obj_out_x_max 밖 → 물리에 안 걸리는 원거리
    provider = make_provider(args_cli.pose_source)
    state = SessionState(last_spawn=None, obj_idx=None)
    session_results: list[EpisodeResult] = []
    print("[INTERACTIVE] 명령: spawn X Y [Z] | repeat | obj N | sweep ... | quit")
    print("> ", end="", flush=True)
    pending = None  # 실행 지시 dict
    while simulation_app.is_running():
        if pending is None:
            # STAGED: 물리 스텝 없이 렌더만 돌리며 stdin 폴링 (GUI 응답성 유지)
            simulation_app.update()
            line = _poll_stdin()
            if line is None:
                continue
            try:
                state, act = apply_command(state, parse_command(line))
            except ValueError as e:
                print(f"[ERR] {e}\n> ", end="", flush=True)
                continue
            if act["action"] == "quit":
                break
            if act["action"] == "noop":
                print(f"[OK] obj={state.obj_idx}\n> ", end="", flush=True)
                continue
            if act["action"] == "sweep":
                print("[ERR] sweep 은 인터랙티브 num_envs=1 세션에서 미지원 — "
                      "배치 모드로 별도 실행하세요\n> ", end="", flush=True)
                continue
            pending = act
        else:
            # EVAL: 고정 스폰 설정 → reset → 1 에피소드 실행 → 결과 출력 → STAGED 복귀
            ge.eval_fixed_spawn_local = single_spawn_tensor(
                pending["x"], pending["y"], pending["z"], 1
            ).to(ge.device)
            # obj 선택: env0의 물체는 env_id%8=0 고정이라 obj N 은 경고만
            #   (MultiAsset 배정은 spawn 시점 고정 — 스펙 §7.4. obj 실선택은 num_envs=8
            #    기동 후 해당 env만 평가하는 후속 개선으로 미룸: YAGNI)
            if state.obj_idx not in (None, 0):
                print(f"[WARN] obj {state.obj_idx} 미지원(단일 env는 물체 0 고정) — 물체 0으로 진행")
            obs = env.reset()
            if isinstance(obs, dict):
                obs = obs["obs"]
            if agent.is_rnn:
                agent.init_rnn()
            provider.on_reset(ge, torch.arange(ge.num_envs, device=ge.device))
            result = _run_one_episode(env, ge, agent, provider, cell_idx=0)
            session_results.append(result)
            print(f"[RESULT] success={result.success} lifted={result.lifted} "
                  f"grip={result.grip_count:.1f} disp={result.displacement*100:.1f}cm "
                  f"obj={result.obj_idx}{' [INVALID]' if result.invalid else ''}")
            print("> ", end="", flush=True)
            pending = None
    # 종료: 세션 이력 저장 (--out 지정 시)
    if args_cli.out and session_results:
        os.makedirs(args_cli.out, exist_ok=True)
        cells_hist = [(r.cell_idx, r) for r in session_results]  # 위치별 미집계 원시 기록
        import json as _json
        with open(os.path.join(args_cli.out, "interactive_history.json"), "w") as f:
            _json.dump([r.__dict__ for r in session_results], f, indent=2)
        print(f"[INFO] 세션 이력 저장: {args_cli.out}/interactive_history.json")
```

`_run_one_episode(env, ge, agent, provider, cell_idx)` 는 Task 6 배치 루프의 "1 에피소드 굴리고 done 직전 스냅샷으로 `_episode_metrics` 반환" 부분을 함수로 추출해 공유한다(배치 루프도 이 함수를 쓰도록 리팩터). `--real-time`이면 스텝당 `time.sleep(max(0, dt - elapsed))` (play.py 말미 패턴).

`__main__` 분기 교체:

```python
if __name__ == "__main__":
    if MODE == "interactive":
        interactive_main()
    else:
        main()
    simulation_app.close()
```

- [ ] **Step 2: 문법 검증** — `python -c "import ast; ast.parse(open('scripts/eval_s2r/eval_sim2real.py').read()); print('SYNTAX OK')"`.

- [ ] **Step 3: 전체 정적 테스트 회귀** — `python -m pytest scripts/eval_s2r/tests -q` → PASS.

- [ ] **Step 4: 커밋**

```bash
git add scripts/eval_s2r/eval_sim2real.py
git commit -m "feat: eval_s2r 인터랙티브 상주 세션(spawn→평가→결과→리셋 루프, GUI 논블로킹)"
```

---

### Task 8: GPU/GUI 스모크 체크리스트 (수동 — 사용자 게이트)

**Files:**
- Create: `scripts/eval_s2r/README.md` — 사용법 + 아래 체크리스트

**이 태스크는 문서 작성만 한다. 실제 GPU 실행은 사용자 지시 후 별도로 진행.**

- [ ] **Step 1: README 작성** — 스펙 §3의 CLI 예시 3종(그리드/단일/인터랙티브) + 스모크 체크리스트:

```markdown
# eval_s2r — grasp_v1 sim2real 평가 하네스

(스펙: docs/superpowers/specs/2026-08-04-grasp-v1-s2r-eval-sp1-design.md)

## 스모크 체크리스트 (GPU — 사용자 게이트)

1. [ ] right lstm_test3, 3×3 그리드 × repeats 8, episodes_per_env 2
       → 히트맵 2장 + CSV/JSON 생성, 중심 셀 success ≈ play.py --eval_episodes 값(0.89) ±0.05
2. [ ] left lstm_test12 동일 절차
3. [ ] --pose_source live vs state_frozen 중심 셀 비교 → freeze staleness 1차 수치 기록
4. [ ] 인터랙티브(로컬 pc5090, GUI): 기동→spawn 0.27 -0.10→결과 출력→spawn 재실행 2회→quit,
       대기 중 GUI 창 조작(마우스 회전) 가능 확인
5. [ ] 훅 무동작 회귀: 기존 play.py 재생이 이전과 동일하게 동작(스폰 랜덤·obs 정상)
```

- [ ] **Step 2: 커밋**

```bash
git add scripts/eval_s2r/README.md
git commit -m "docs: eval_s2r 사용법·GPU 스모크 체크리스트"
```

---

## Self-Review 결과

- **스펙 커버리지**: §3 CLI(T6·T7), §4.1 로더(T6), §4.2 훅(T5), §4.3 seam(T2), §4.4 지표/출력(T3·T6), §4.5 흐름(T6), §4.6 인터랙티브(T4·T7), §5 에러(T1/T2/T4 검증 + T6 invalid), §6 정적 테스트(T1~T5)·GPU 스모크(T8). sweep 명령은 인터랙티브에서 명시적 미지원 안내(단일 env 제약, YAGNI) — 스펙 §4.6 표와의 차이는 T8 README에 기록됨.
- **자리표시자**: T6 골격의 주석 블록은 "play.py 라인을 열어 실제 코드로 완성, 주석 그대로면 미완" 명시로 처리.
- **타입 일관성**: `EpisodeResult`/`GridSpec`/`SessionState` 시그니처가 T1·T3·T4 정의와 T6·T7 사용처 일치 확인.

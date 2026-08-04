"""그리드 스윕 셀 생성·env↔셀 매핑 (순수 함수, Isaac 무관).

설계: docs/superpowers/specs/2026-08-04-grasp-v1-s2r-eval-sp1-design.md §4
"""
from __future__ import annotations

from dataclasses import dataclass

# torch는 텐서 생성 함수 내부에서만 지연 import한다 — Isaac Lab 관례상 AppLauncher 기동
# 전에는 torch를 import하면 안 되는데(CUDA/omni 컨텍스트 초기화 순서), GridSpec은 eval_sim2real.py가
# AppLauncher 생성 전 인자 검증(M4)에 재사용하므로 모듈 최상단에서는 torch에 의존하지 않는다.


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
):
    """[len(cells)*repeats, 3] float32 torch.Tensor. env i → cells[i // repeats]. z NaN=물체별 기본 높이."""
    import torch

    rows = [[x, y, z] for (x, y) in cells for _ in range(repeats)]
    return torch.tensor(rows, dtype=torch.float32)


def single_spawn_tensor(x: float, y: float, z: float, num_envs: int):
    import torch

    return torch.tensor([[x, y, z]] * num_envs, dtype=torch.float32)

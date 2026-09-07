# Copyright 2025 Enactic, Inc.
# Licensed under the Apache License, Version 2.0
"""Minimal ADR scheduler used for pour task.

GraspADR from 5g_pour_right_v9 is reused so we can linearly ramp
parameters (e.g., spill penalty, noise) based on episode success rate.
"""

ADR_KEYS = ("spill", "noise", "success", "outcome")


class PourADR:
    """Lightweight ADR parameter scheduler.

    Args:
        custom_cfg: {group: {param: (initial, final)}}
        num_increments: number of increments until reaching final value.
        increment_interval: how often to check (env steps).
        trigger_threshold: success-rate threshold to increment.
    """

    def __init__(
        self,
        custom_cfg: dict,
        num_increments: int = 50,
        increment_interval: int = 200,
        trigger_threshold: float = 0.1,
    ):
        self.custom_cfg = custom_cfg
        self.num_increments = max(1, num_increments)
        self.increment_interval = increment_interval
        self.trigger_threshold = trigger_threshold

        self.increment_counter: int = 0
        self._step_counter: int = 0

    def get_param(self, group: str, name: str) -> float:
        """Return linearly interpolated value for the current progress."""
        lo, hi = self.custom_cfg[group][name]
        t = min(self.increment_counter / float(self.num_increments), 1.0)
        return lo + (hi - lo) * t

    def maybe_increment(self, metric) -> bool:
        """Increment when step interval reached and metric ≥ threshold."""
        self._step_counter += 1
        if self._step_counter % self.increment_interval != 0:
            return False

        if metric >= self.trigger_threshold and self.increment_counter < self.num_increments:
            self.increment_counter += 1
            return True
        return False

    def set_increment(self, n: int) -> None:
        self.increment_counter = min(max(0, n), self.num_increments)

    @property
    def progress(self) -> float:
        return self.increment_counter / float(self.num_increments)



def collect_adr_progress_pins(cfg) -> dict:
    """cfg 의 `adr_initial_progress_<key>` 스칼라들을 {key: 진행률} 로 모은다.

    체크포인트 재개 시 ADR 초기 레벨을 고정하기 위한 것이다. 음수 = 고정 안 함이라
    "0 으로 고정"(음수 아님)과 "건드리지 않음"이 구분된다.

    범위를 벗어난 값은 조용히 자르지 않고 예외를 던진다 — 잘못 적은 레벨로
    학습이 며칠 굴러가는 쪽이 훨씬 비싸다.
    """
    out: dict = {}
    for key in ADR_KEYS:
        frac = getattr(cfg, f"adr_initial_progress_{key}", -1.0)
        if frac is None or float(frac) < 0.0:
            continue
        frac = float(frac)
        if frac > 1.0:
            raise ValueError(
                f"adr_initial_progress_{key} = {frac} — 진행률은 0~1 이다"
                f"(카운터가 아니라 비율)")
        out[key] = frac
    return out

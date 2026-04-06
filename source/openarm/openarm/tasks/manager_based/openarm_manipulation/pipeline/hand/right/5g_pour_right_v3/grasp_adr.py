# Copyright 2025 Enactic, Inc.
# Licensed under the Apache License, Version 2.0
"""Minimal ADR scheduler used for pour task.

GraspADR from 5g_grasp_right_v9 is reused so we can linearly ramp
parameters (e.g., spill penalty, noise) based on episode success rate.
"""

class GraspADR:
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


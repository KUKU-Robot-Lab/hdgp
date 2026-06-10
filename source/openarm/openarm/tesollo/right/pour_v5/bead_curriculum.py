# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Bead-count curriculum (DexPour-style difficulty progression).

물리 비드는 startup에 max개(=schedule[-1]) 고정 생성되므로, 커리큘럼은
"활성 개수 N"만 success 기반으로 1→5→8→10→20→30 으로 올린다. 비활성 비드는
env에서 hide(z=-10)되고, bead fraction은 활성 N개(앞 N 슬라이스)로 정규화된다.

근거: sparse한 bead_in(200)을 reachable하게 만들어 tilt 부트스트랩 (analysis 참조).
DexPour ablation: 커리큘럼 없는 full reward는 직립-park로 premature 수렴(=v3 fresh deadlock).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BeadCountCurriculum:
    """활성 비드 수를 success 기반으로 단조 증가시키는 커리큘럼.

    Args:
        schedule: 활성 비드 수 단계 (오름차순). 마지막 값이 물리 spawn 최대 개수.
        success_threshold: 현 단계에서 이 success_rate 이상이면 advance 후보.
        min_updates_per_stage: 단계당 최소 update() 호출 수 (성급한 advance 방지).
    """

    schedule: tuple[int, ...] = (1, 5, 8, 10, 20, 30)
    success_threshold: float = 0.5
    min_updates_per_stage: int = 5
    _stage: int = field(default=0)
    _updates_at_stage: int = field(default=0)

    def __post_init__(self) -> None:
        if not self.schedule:
            raise ValueError("schedule must be non-empty")
        if list(self.schedule) != sorted(self.schedule):
            raise ValueError("schedule must be ascending")
        if not (0.0 <= self.success_threshold <= 1.0):
            raise ValueError("success_threshold must be in [0, 1]")

    @property
    def current_count(self) -> int:
        """현재 활성 비드 수."""
        return int(self.schedule[self._stage])

    @property
    def max_count(self) -> int:
        """물리 spawn 최대 개수 (= schedule 마지막)."""
        return int(self.schedule[-1])

    @property
    def is_final(self) -> bool:
        return self._stage >= len(self.schedule) - 1

    def update(self, success_rate: float) -> bool:
        """주기적 호출. 임계+최소호출 충족 시 다음 단계로 advance.

        Returns:
            이번 호출에서 advance했으면 True.
        """
        self._updates_at_stage += 1
        if self.is_final:
            return False
        if (
            success_rate >= self.success_threshold
            and self._updates_at_stage >= self.min_updates_per_stage
        ):
            self._stage += 1
            self._updates_at_stage = 0
            return True
        return False

    def state_dict(self) -> dict:
        return {"stage": self._stage, "updates_at_stage": self._updates_at_stage}

    def load_state_dict(self, state: dict) -> None:
        self._stage = int(state.get("stage", 0))
        self._updates_at_stage = int(state.get("updates_at_stage", 0))
        self._stage = max(0, min(self._stage, len(self.schedule) - 1))

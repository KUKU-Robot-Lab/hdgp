"""TaskADR — 자동 도메인 무작위화 스케줄러.

`tesollo/*/grasp_adr.py` (DEXTRAH `DextrahADR` 이식본)를 **1벌로 통합**한 것이다.
같은 파일이 10개 폴더에 복붙돼 있었고, 그중 일부만 고쳐지는 사고가 반복됐다.

동작:
    increment_counter 가 0 → num_increments 로 오르면서 각 파라미터가
    initial → final 로 선형 보간된다. 트리거는 환경 코드가 직접 호출한다.

        adr = TaskADR(custom_cfg={"spawn": {"xy_range": (0.02, 0.10)}})
        ...
        if adr.maybe_increment(instant_success_rate):
            event_manager.reset(...); event_manager.apply(...)
        radius = adr.get_param("spawn", "xy_range")

★트리거 지표는 **순간 성공률**이다(`in_success.float().mean()`).
  `episode_success_rate`(누적 평균)를 넣으면 관성 때문에 커리큘럼이 거의 안 오른다.

★`enabled=False` 면 `get_param` 이 항상 initial 값을 반환한다 —
  커리큘럼 없는 고정 세팅과 **정확히 동치**라 스위치 하나로 끌 수 있다.
"""

from __future__ import annotations

import copy


class TaskADR:
    """Args:
        custom_cfg: {그룹: {파라미터: (initial, final)}}
        num_increments: 최대 증분 횟수(도달 시 final 고정)
        increment_interval: 증분 사이 최소 step 수 (DEXTRAH min_steps_for_dr_change)
        trigger_threshold: 트리거 지표가 이 값을 넘으면 증분
        enabled: False 면 증분하지 않고 initial 값에 고정
        event_manager: IsaacLab EventManager — physics_cfg 와 함께 주면
            증분 시 물리 DR EventTerm 범위를 확장
        physics_cfg: {term: {param: terminal_range}} (DEXTRAH adr_cfg_dict 형식)
    """

    def __init__(
        self,
        custom_cfg: dict,
        *,
        num_increments: int = 50,
        increment_interval: int = 3000,
        trigger_threshold: float = 0.4,
        enabled: bool = True,
        event_manager=None,
        physics_cfg: dict | None = None,
    ) -> None:
        self.custom_cfg = custom_cfg
        self.num_increments = max(1, int(num_increments))
        self.increment_interval = int(increment_interval)
        self.trigger_threshold = float(trigger_threshold)
        self.enabled = bool(enabled)

        self.increment_counter = 0
        self._steps_since_change = 0

        self.event_manager = event_manager
        self.physics_cfg = physics_cfg or {}
        self._physics_initial: dict = {}
        if self.event_manager is not None:
            for term_name, term_params in self.physics_cfg.items():
                term = self.event_manager.get_term_cfg(term_name)
                self._physics_initial[term_name] = {
                    p: copy.deepcopy(term.params[p]) for p in term_params
                }

    # ------------------------------------------------------------------
    @property
    def progress(self) -> float:
        """0.0(초기) → 1.0(만렙)."""
        return self.increment_counter / float(self.num_increments)

    def get_param(self, group: str, name: str) -> float:
        """현재 진행률 기준 선형 보간값. 비활성이면 initial."""
        try:
            lo, hi = self.custom_cfg[group][name]
        except KeyError as exc:                      # 오타를 조용히 넘기지 않는다
            raise KeyError(
                f"ADR 파라미터 없음: group={group!r} name={name!r}. "
                f"선언된 것: { {g: sorted(v) for g, v in self.custom_cfg.items()} }"
            ) from exc
        if not self.enabled:
            return float(lo)
        return float(lo) + (float(hi) - float(lo)) * self.progress

    # ------------------------------------------------------------------
    def maybe_increment(self, metric) -> bool:
        """DEXTRAH 케이던스: 마지막 변경 후 interval step 경과 AND metric > threshold.

        고정 주기 검사가 아니라 조건 충족 즉시 증분하고 카운터를 리셋한다.
        metric 은 float 또는 0-dim tensor 모두 허용(비교가 파이썬 레벨이라
        GPU 동기화를 추가로 유발하지 않는다).
        """
        if not self.enabled:
            return False
        if (
            self._steps_since_change >= self.increment_interval
            and metric > self.trigger_threshold
            and self.increment_counter < self.num_increments
        ):
            self._steps_since_change = 0
            self.increment_counter += 1
            self._expand_physics_ranges()
            return True
        self._steps_since_change += 1
        return False

    def set_increment(self, n: int) -> None:
        """체크포인트 복원 등에서 직접 설정."""
        self.increment_counter = min(max(0, int(n)), self.num_increments)
        self._expand_physics_ranges()

    def _expand_physics_ranges(self) -> None:
        """물리 DR EventTerm 범위를 initial → terminal 로 선형 확장.

        호출측이 event_manager.reset/apply 로 새 범위를 전 env 에 즉시 반영해야 한다.
        """
        if self.event_manager is None:
            return
        t = self.progress
        for term_name, term_params in self.physics_cfg.items():
            term = self.event_manager.get_term_cfg(term_name)
            for param_name, terminal in term_params.items():
                init = self._physics_initial[term_name][param_name]
                term.params[param_name] = (
                    init[0] + (terminal[0] - init[0]) * t,
                    init[1] + (terminal[1] - init[1]) * t,
                )

    # ------------------------------------------------------------------
    def log_dict(self) -> dict:
        """TFEvents 로 내보낼 스칼라. 판정은 trigger_metric 으로 한다."""
        return {
            "adr/increment": float(self.increment_counter),
            "adr/progress": self.progress,
        }

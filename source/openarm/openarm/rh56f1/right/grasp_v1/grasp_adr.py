# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GraspADR: DEXTRAH DextrahADR 이식.

원본: DEXTRAH/dextrah_lab/tasks/dextrah_kuka_allegro/dextrah_adr.py
이식 범위:
  - get_custom_param_value() (get_param)
  - increase_ranges(): event_manager 물리 DR 파라미터 범위를 initial→terminal
    선형 확장 (physics_cfg 제공 시)
  - 증분 케이던스: DEXTRAH 원본 = "마지막 변경 후 min_steps 경과 AND metric >
    threshold" (고정 주기 검사 아님)

동작:
  - increment_counter가 0 → num_increments 로 올라가면서
    각 파라미터가 initial → final 값으로 선형 보간됨.
  - 트리거는 환경 코드에서 직접 호출: adr.maybe_increment(metric)
"""

import copy


class GraspADR:
    """파지 태스크용 ADR 파라미터 스케줄러.

    Args:
        custom_cfg: 파라미터 그룹 딕셔너리.
            형식: {group_name: {param_name: (initial_value, final_value)}}
        num_increments: 최대 increment 횟수 (이 횟수에 도달하면 final 값 고정).
        increment_interval: 증분 사이 최소 step 수 (DEXTRAH min_steps_for_dr_change).
        trigger_threshold: 트리거 메트릭이 이 값 초과면 increment.
        event_manager: (선택) IsaacLab EventManager — physics_cfg 와 함께 주면
            increment 시 물리 DR EventTerm 파라미터 범위를 확장.
        physics_cfg: (선택) {term_name: {param_name: terminal_range}} —
            DEXTRAH adr_cfg_dict 형식 (num_increments 키 제외).
    """

    def __init__(
        self,
        custom_cfg: dict,
        num_increments: int = 50,
        increment_interval: int = 200,
        trigger_threshold: float = 0.1,
        event_manager=None,
        physics_cfg: dict | None = None,
    ):
        self.custom_cfg = custom_cfg
        self.num_increments = max(1, num_increments)
        self.increment_interval = increment_interval
        self.trigger_threshold = trigger_threshold

        self.increment_counter: int = 0
        self._steps_since_change: int = 0

        # ---- 물리 DR (DEXTRAH adr_cfg_dict + event_manager) ----
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
    # 파라미터 조회
    # ------------------------------------------------------------------

    def get_param(self, group: str, name: str) -> float:
        """현재 increment_counter 기준 선형 보간 값 반환.

        DEXTRAH get_custom_param_value()와 동일 로직.
        """
        lo, hi = self.custom_cfg[group][name]
        t = min(self.increment_counter / float(self.num_increments), 1.0)
        return lo + (hi - lo) * t

    # ------------------------------------------------------------------
    # Increment 관리
    # ------------------------------------------------------------------

    def maybe_increment(self, metric) -> bool:
        """DEXTRAH 케이던스: 마지막 변경 후 interval step 경과 AND metric > threshold.

        (고정 주기 검사가 아니라 조건 충족 즉시 증분 후 카운터 리셋 — 원본
        step_since_last_dr_change >= min_steps_for_dr_change 와 동일 의미)

        metric은 float 또는 0-dim torch.Tensor 모두 허용.
        tensor 비교는 Python 레벨에서 이루어지므로 GPU 동기화 없음.

        Returns:
            bool: increment가 발생했으면 True.
        """
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

    def _expand_physics_ranges(self) -> None:
        """물리 DR EventTerm 범위를 initial→terminal 로 선형 확장.

        DEXTRAH increase_ranges() 동일 로직 (증분마다 lower/upper 한계를
        1/num_increments 씩 종점으로 이동). 호출측이 event_manager
        reset/apply 로 새 범위를 전 env 에 즉시 반영해야 한다.
        """
        if self.event_manager is None:
            return
        for term_name, term_params in self.physics_cfg.items():
            term = self.event_manager.get_term_cfg(term_name)
            for param_name, terminal in term_params.items():
                init = self._physics_initial[term_name][param_name]
                t = self.increment_counter / float(self.num_increments)
                lower = init[0] + (terminal[0] - init[0]) * t
                upper = init[1] + (terminal[1] - init[1]) * t
                term.params[param_name] = (lower, upper)

    def set_increment(self, n: int) -> None:
        """체크포인트 복원 등에서 직접 increment 설정."""
        self.increment_counter = min(max(0, n), self.num_increments)
        self._expand_physics_ranges()

    @property
    def progress(self) -> float:
        """0.0 (초기) → 1.0 (최대 난이도) 진행률."""
        return self.increment_counter / float(self.num_increments)

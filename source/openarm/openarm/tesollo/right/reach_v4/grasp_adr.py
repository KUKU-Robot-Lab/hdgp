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

"""GraspADR: DEXTRAH DextrahADR의 custom_param_value 기능 이식.

원본: DEXTRAH/dextrah_lab/tasks/dextrah_kuka_allegro/dextrah_adr.py
이식 범위: get_custom_param_value() + (08.16) grasp_v2 의 물리 DR 확장.

동작:
  - increment_counter가 0 → num_increments 로 올라가면서
    각 파라미터가 initial → final 값으로 선형 보간됨.
  - 트리거는 환경 코드에서 직접 호출: adr.maybe_increment(metric, threshold)
  - event_manager+physics_cfg 를 주면 증분마다 물리 DR EventTerm(질량/마찰)
    범위도 initial → terminal 로 확장 (grasp_v2 와 동일 메커니즘).
"""

import copy


class GraspADR:
    """파지 태스크용 ADR 파라미터 스케줄러.

    Args:
        custom_cfg: 파라미터 그룹 딕셔너리.
            형식: {group_name: {param_name: (initial_value, final_value)}}
        num_increments: 최대 increment 횟수 (이 횟수에 도달하면 final 값 고정).
        increment_interval: increment 검사 주기 (env step 단위).
        trigger_threshold: 트리거 메트릭이 이 값 이상이면 increment.
        event_manager: (선택) IsaacLab EventManager — physics_cfg 와 함께 주면
            increment 시 물리 DR EventTerm 파라미터 범위를 확장.
        physics_cfg: (선택) {term_name: {param_name: terminal_range}} —
            DEXTRAH adr_cfg_dict 형식. EventCfg 의 현재 값이 initial 이 된다.
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
        self._step_counter: int = 0

        # ---- 물리 DR (grasp_v2 이식): EventTerm 현재 범위를 initial 로 보관 ----
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
        """마지막 증분 후 interval 경과 AND metric이 threshold 이상이면 increment.

        ★08.16 케이던스를 grasp_v2(DEXTRAH 원본)와 동일하게 교체.
        구: `_step_counter % interval == 0` 인 **고정 주기 검사** — 그 한 스텝에 메트릭이
        마침 임계 미만이면 다음 검사까지 interval 을 통째로 더 기다렸다. 순간 성공률처럼
        스텝마다 출렁이는 메트릭에서는 증분 타이밍이 운에 좌우된다.
        신: 조건을 만족하는 **즉시** 증분하고 카운터를 리셋 — interval 은 "증분 사이 최소
        간격"이라는 본래 의미가 되고, 성능이 임계 아래면 회복할 때까지 자연히 대기한다.

        metric은 float 또는 0-dim torch.Tensor 모두 허용.
        tensor 비교는 Python 레벨에서 이루어지므로 GPU 동기화 없음.

        Returns:
            bool: increment가 발생했으면 True.
        """
        # tensor인 경우 item() 없이 비교 (Python bool 변환은 단일 값 tensor에서 자동)
        if (
            self._step_counter >= self.increment_interval
            and metric >= self.trigger_threshold
            and self.increment_counter < self.num_increments
        ):
            self._step_counter = 0
            self.increment_counter += 1
            self._expand_physics_ranges()
            return True
        self._step_counter += 1
        return False

    def _expand_physics_ranges(self) -> None:
        """물리 DR EventTerm 범위를 initial→terminal 로 선형 확장 (grasp_v2 이식).

        호출측이 event_manager reset/apply 로 새 범위를 전 env 에 즉시 반영해야 한다.
        """
        if self.event_manager is None:
            return
        t = self.increment_counter / float(self.num_increments)
        for term_name, term_params in self.physics_cfg.items():
            term = self.event_manager.get_term_cfg(term_name)
            for param_name, terminal in term_params.items():
                init = self._physics_initial[term_name][param_name]
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

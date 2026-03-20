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
이식 범위: get_custom_param_value() 로직만 — event_manager 의존성 제거.
외부 프로젝트 import 없음.

동작:
  - increment_counter가 0 → num_increments 로 올라가면서
    각 파라미터가 initial → final 값으로 선형 보간됨.
  - 트리거는 환경 코드에서 직접 호출: adr.maybe_increment(metric, threshold)
"""


class GraspADR:
    """파지 태스크용 ADR 파라미터 스케줄러.

    Args:
        custom_cfg: 파라미터 그룹 딕셔너리.
            형식: {group_name: {param_name: (initial_value, final_value)}}
        num_increments: 최대 increment 횟수 (이 횟수에 도달하면 final 값 고정).
        increment_interval: increment 검사 주기 (env step 단위).
        trigger_threshold: 트리거 메트릭이 이 값 이상이면 increment.
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
        """step_counter가 interval에 도달하고 metric이 threshold 이상이면 increment.

        metric은 float 또는 0-dim torch.Tensor 모두 허용.
        tensor 비교는 Python 레벨에서 이루어지므로 GPU 동기화 없음.

        Returns:
            bool: increment가 발생했으면 True.
        """
        self._step_counter += 1
        if self._step_counter % self.increment_interval != 0:
            return False

        # tensor인 경우 item() 없이 비교 (Python bool 변환은 단일 값 tensor에서 자동)
        if metric >= self.trigger_threshold and self.increment_counter < self.num_increments:
            self.increment_counter += 1
            print(
                f"[GraspADR] Increment {self.increment_counter}/{self.num_increments} "
                f"(metric={metric:.3f} >= threshold={self.trigger_threshold:.3f})"
            )
            return True
        return False

    def set_increment(self, n: int) -> None:
        """체크포인트 복원 등에서 직접 increment 설정."""
        self.increment_counter = min(max(0, n), self.num_increments)

    @property
    def progress(self) -> float:
        """0.0 (초기) → 1.0 (최대 난이도) 진행률."""
        return self.increment_counter / float(self.num_increments)

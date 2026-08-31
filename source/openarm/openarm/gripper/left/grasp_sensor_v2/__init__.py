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

"""gripper/left/grasp_sensor_v2 — Lee et al. 계단식 보상 재설계판.

v1(`grasp_sensor`)은 **동결**한다. t79(best 167.57 · 컵–목표 90 mm)가 현재 챔피언이자
폴백이고, `both/pour_sensor` 왼팔이 최종적으로 이긴 쪽을 가리킨다.
"""

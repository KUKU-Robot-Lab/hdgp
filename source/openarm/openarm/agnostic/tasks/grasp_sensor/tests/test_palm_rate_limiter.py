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

"""palm 지령 리미터 계약 (08.26 계획서 §3).

좌팔 test_fab_contract 의 리미터 절 패턴 — "상수만 있고 코드가 없다"와
"코드가 있는데 기본값이 켜져 있다"를 둘 다 잡는다:
  ① cfg 기본값은 **0.0(비활성)** — corridor_test1 등 진행 중 런이 코드 머지로
    행동이 바뀌면 안 된다. 켜는 것은 fresh 런의 명시적 선택이어야 한다.
  ② env 소스에 리미터 배선이 실제로 존재한다(상수 참조 + 프라이밍 예외).
Isaac 없이 도는 텍스트/설정 수준 계약이다(무거운 import 금지 — 기존 테스트 규약).
"""

from __future__ import annotations

import re
from pathlib import Path

_TASK_DIR = Path(__file__).resolve().parents[1]


def _cfg_source() -> str:
    return (_TASK_DIR / "grasp_sensor_env_cfg.py").read_text(encoding="utf-8")


def _env_source() -> str:
    return (_TASK_DIR / "grasp_sensor_env.py").read_text(encoding="utf-8")


def test_rate_limit_default_disabled():
    """기본값 0.0 — 진행 중 런(corridor_test1)이 머지로 영향받지 않아야 한다."""
    m = re.search(r"palm_cmd_rate_limit_m:\s*float\s*=\s*([0-9.]+)", _cfg_source())
    assert m, "cfg 에 palm_cmd_rate_limit_m 이 없다"
    assert float(m.group(1)) == 0.0, (
        f"palm_cmd_rate_limit_m 기본값이 {m.group(1)} — 반드시 0.0(비활성)이어야 한다. "
        "켜는 것은 fresh 런의 명시적 오버라이드로만.")


def test_rate_limit_wired_in_pre_physics():
    """상수만 있고 배선이 없는 상태(좌팔 트랙에서 실제로 잡은 결함)를 차단."""
    src = _env_source()
    pre = src.split("def _pre_physics_step", 1)
    assert len(pre) == 2, "_pre_physics_step 이 없다"
    body = pre[1].split("\n    def ", 1)[0]
    assert "palm_cmd_rate_limit_m" in body, "리미터가 _pre_physics_step 에 배선되지 않았다"
    assert "_palm_cmd_primed" in body, "프라이밍 예외가 없다 — 리셋마다 팔이 끌려간다"
    assert "_palm_cmd_step_raw" in body, "클램프 전 원값 로깅이 없다(reward-clamp 규칙)"


def test_rate_limit_priming_cleared_on_reset():
    """리셋에서 프라이밍이 풀려야 첫 지령이 리미터에 안 걸린다."""
    src = _env_source()
    reset = src.split("def _reset_idx", 1)
    assert len(reset) == 2, "_reset_idx 가 없다"
    body = reset[1].split("\n    def ", 1)[0]
    assert "_palm_cmd_primed[env_ids] = False" in body, (
        "_reset_idx 가 _palm_cmd_primed 를 해제하지 않는다")

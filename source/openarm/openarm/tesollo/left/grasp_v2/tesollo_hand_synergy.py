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

"""Tesollo LEFT hand synergy basis — right basis의 부호맵 파생(수동 편집 금지).

right/grasp_v2/tesollo_hand_synergy.py 를 import 시점에 로드해
q_left = S·q_right 부호맵(S=diag ±1, grasp_left_preset._HAND_SIGN)을 열별로 적용한다.

수학적 근거 (S² = I, S 직교 대각):
  - BASIS_L  = BASIS_R · S            (행=PC, 열=관절 — 열별 부호 곱)
  - ANCHOR_L = S · ANCHOR_R           (= left HAND_APPROACH_POSE 와 자동 일치)
  - 계수 z   = B_L(q_L − a_L) = B_R·S·S·(q_R − a_R) = B_R(q_R − a_R)
    → COEFF_MINS/MAXS/GRIP 는 right 와 동일 (그대로 re-export)
  - 직교정규 보존: (B·S)(B·S)ᵀ = B·Bᵀ

right 가 retarget_allegro_pca_to_tesollo.py 재실행으로 갱신되면 여기는 자동 동기화.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve()


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"module load 실패: {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_right = _load(
    _HERE.parents[2] / "right" / "grasp_v2" / "tesollo_hand_synergy.py",
    "_grasp_v2_right_tesollo_hand_synergy",
)
_preset = _load(_HERE.with_name("grasp_left_preset.py"), "_grasp_v2_left_preset_for_synergy")

_SIGN = _preset._HAND_SIGN
assert len(_SIGN) == 20

HAND_SYNERGY_BASIS = [
    [v * s for v, s in zip(row, _SIGN)] for row in _right.HAND_SYNERGY_BASIS
]
HAND_SYNERGY_ANCHOR = [v * s for v, s in zip(_right.HAND_SYNERGY_ANCHOR, _SIGN)]

# 계수 공간은 부호맵에 불변 (모듈 docstring 증명 참조)
HAND_SYNERGY_COEFF_MINS = list(_right.HAND_SYNERGY_COEFF_MINS)
HAND_SYNERGY_COEFF_MAXS = list(_right.HAND_SYNERGY_COEFF_MAXS)
HAND_SYNERGY_COEFF_GRIP = list(_right.HAND_SYNERGY_COEFF_GRIP)

# 정합성 자기검증: anchor 는 left HAND_APPROACH_POSE 와 일치해야 함
assert all(
    abs(a - b) < 1e-6 for a, b in zip(HAND_SYNERGY_ANCHOR, _preset.HAND_APPROACH_POSE)
), "ANCHOR ≠ left HAND_APPROACH_POSE — right anchor 또는 부호맵 변경 여부 확인 필요"

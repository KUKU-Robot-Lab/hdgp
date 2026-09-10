"""벤더 rl_games 의 저장 정책이 **설정을 따르는지** 확인한다.

★09.11 저장 정책은 `tesollo/right/grasp_v1` 과 같게 둔다(save_best_after 100 /
  save_frequency 500). 그런데 SAPG 포크가 두 곳에 값을 박아 두어 설정이 안 먹었다:
    · Continuous 의 best 기준이 `epoch_num >= 10` (save_best_after 무시)
    · 복구용 `last/model` 저장이 `% 3`(Discrete) / `% 200`(Continuous)
  체크포인트가 16384 env 에서 255MB 이고 디스크가 58.8MB/s 라(실측), 설정이 실제로
  먹어야 저장 빈도를 제어할 수 있다. 하드코딩이 되살아나면 여기서 잡는다.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# 저장소 루트는 **탐색으로** 찾는다 — parents[N] 은 파일이 옮겨지면 조용히 어긋나고,
# 이 파일에서도 실제로 한 번 틀려 테스트가 통째로 skip 됐다(무증상 통과).
_REL = Path("vendor") / "rl_games_sapg" / "rl_games" / "common" / "a2c_common.py"
_VENDOR = next((q / _REL for q in Path(__file__).resolve().parents if (q / _REL).exists()),
               Path("/nonexistent"))


def test_checkpoint_save_paths_follow_save_frequency():
    """주기 저장이 **설정**을 따라야 한다 — 원본은 3/200 을 하드코딩했다.

    하드코딩이 남으면 `save_frequency` 를 올려도 저장이 줄지 않는다(09.11 실측:
    16384env 에서 벽시계의 42%가 저장 대기였다).
    """
    if not _VENDOR.exists():
        pytest.skip("벤더 rl_games 없음")
    src = _VENDOR.read_text(encoding="utf-8")
    for banned in ("epoch_num % 3 == 0", "epoch_num % 200 == 0"):
        assert banned not in src, f"주기 저장에 하드코딩이 남아 있다: {banned}"

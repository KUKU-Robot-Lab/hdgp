"""벤더 rl_games 에 우리가 넣은 패치의 **동작**을 검증한다(문자열 대조가 아니라).

★09.11 왜 이 파일이 있나. `vendor/rl_games_sapg` 는 외부 코드라 우리 테스트가 안 닿는데,
  체크포인트 저장 동작을 우리가 고쳤다(창 안 최고 기록을 메모리에 잡아뒀다가 창 끝에
  한 번만 디스크에 쓴다). 그 핵심 전제가 **스냅샷이 참조가 아니라 복사**라는 것이다 —
  참조면 저장되는 것이 '창 최고'가 아니라 '창 끝 시점' 가중치가 되고, 증상이 없다.

  a2c_common 전체 import 는 무겁고 isaac 의존이 붙으므로, 해당 메서드만 소스에서 떼어
  독립 클래스에 붙여 실행한다.
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


def _load_method(name: str):
    if not _VENDOR.exists():
        pytest.skip(f"벤더 rl_games 없음: {_VENDOR}")
    src = _VENDOR.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == name), None)
    assert fn is not None, f"벤더에 {name} 이 없다 — 패치가 사라졌다"
    body = textwrap.dedent("\n".join(src.splitlines()[fn.lineno - 1:fn.end_lineno]))
    ns: dict = {}
    exec("class _Host:\n" + textwrap.indent(body, "    "), {"torch": torch}, ns)
    return ns["_Host"]


def test_best_window_snapshot_is_a_deep_cpu_copy():
    """스냅샷 뒤 원본이 바뀌어도 스냅샷은 그대로여야 한다."""
    host = _load_method("_snapshot_full_state_cpu")()
    host.global_rank = 0
    live = torch.ones(4)
    host.get_full_state_weights = lambda: {"model": {"w": live}, "epoch": 3, "opt": [live]}

    snap = host._snapshot_full_state_cpu()
    live.mul_(99.0)

    assert torch.allclose(snap[0]["model"]["w"], torch.ones(4)), (
        "참조였다 — 창 끝에 저장되는 것이 창 최고가 아니라 창 끝 시점 가중치가 된다")
    assert torch.allclose(snap[0]["opt"][0], torch.ones(4)), "리스트 안 텐서가 참조다"
    assert snap[0]["epoch"] == 3, "텐서가 아닌 값은 그대로 실려야 한다"
    assert snap[0]["model"]["w"].device.type == "cpu"
    assert set(snap) == {0}, "save(fn, override_state) 가 받는 rank 형식이어야 한다"


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

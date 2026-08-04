"""env 훅 정적 검사 (Isaac 불필요 — 소스 텍스트 검사).

1) 좌우 grasp_v1 env에 훅 마커 블록이 존재하고 문자 단위 동일한가.
2) 훅이 getattr(..., None) 기본 무동작 패턴인가 (학습 경로 보호).
scripts/analysis/tests 의 소스 검사 테스트들과 같은 방식.
"""
import re
from pathlib import Path

HDGP = Path(__file__).resolve().parents[3]
RIGHT = HDGP / "source/openarm/openarm/tesollo/right/grasp_v1/grasp_right_env.py"
LEFT = HDGP / "source/openarm/openarm/tesollo/left/grasp_v1/grasp_left_env.py"

SPAWN_MARK = "eval_s2r: 고정 스폰 오버라이드"
OBS_MARK = "eval_s2r: cup pose obs 오버라이드"


def _extract_block(text: str, marker: str) -> str:
    """marker 주석 줄부터, 첫 빈 줄 또는 그보다 얕은 들여쓰기의 줄 전까지 추출.

    훅 삽입 지점은 모두 flat 한 들여쓰기의 분기(else 본문 등) 안에 있어
    들여쓰기만으로는 훅 블록 경계를 특정할 수 없다(같은 들여쓰기의 무관한
    후속 코드까지 휩쓸림 — 특히 좌우 비대칭 pregrasp orientation 값).
    두 훅 모두 뒤에 빈 줄이 오도록 삽입되므로, 빈 줄을 블록 종료 신호로 겸용한다.
    """
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if marker in l)
    indent = len(lines[start]) - len(lines[start].lstrip())
    out = [lines[start].strip()]
    for l in lines[start + 1:]:
        if not l.strip():
            break
        if (len(l) - len(l.lstrip())) < indent:
            break
        out.append(l.strip())
    return "\n".join(out)


def test_hooks_exist_in_both_envs():
    for path in (RIGHT, LEFT):
        text = path.read_text()
        assert SPAWN_MARK in text, f"{path.name}: 스폰 훅 없음"
        assert OBS_MARK in text, f"{path.name}: obs 훅 없음"


def test_hooks_identical_left_right():
    rt, lt = RIGHT.read_text(), LEFT.read_text()
    for mark in (SPAWN_MARK, OBS_MARK):
        assert _extract_block(rt, mark) == _extract_block(lt, mark), f"{mark}: 좌우 불일치"


def test_hooks_are_getattr_gated():
    """훅 블록 안에 getattr(self, ..., None) 게이트가 있어야 학습 경로 무영향."""
    for path in (RIGHT, LEFT):
        text = path.read_text()
        for mark, attr in ((SPAWN_MARK, "eval_fixed_spawn_local"),
                           (OBS_MARK, "eval_cup_pos_override")):
            block = _extract_block(text, mark)
            assert re.search(rf'getattr\(self, "{attr}", None\)', block), (
                f"{path.name}/{mark}: getattr 게이트 없음"
            )

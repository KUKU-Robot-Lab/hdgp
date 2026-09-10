"""소스텍스트 계약 테스트용 공용 헬퍼 — AST·정규식만 쓰는 순수 유틸리티.

★★09.10 이동. 원래 `grasp_kp/tests/test_grasp_kp_contract.py` 안에 있었고
`grasp_fj` 의 계약 테스트가 거기서 import 했다. 사용자 확정 "s2r 하고 fj 는
공유 금지" 에 따라 fj 를 다른 트랙에서 떼어내면서, 이 헬퍼들만 남아 fj → kp
의존을 만들고 있었다. 과제 의미가 전혀 없는 텍스트 파서이므로 중립 위치로
옮겨 양쪽이 여기서 읽는다(복제하지 않는다).

isaaclab 을 import 하지 않는다 — 계약 테스트는 시스템 python3 pytest 로 돈다.
"""

from __future__ import annotations

import ast
import re
import textwrap


def _code(src: str) -> str:
    """주석·docstring 을 뺀 실행 코드만 — 설명문에 적힌 이름이 계약을 통과시키면 안 된다."""
    tree = ast.parse(src)
    doc_lines: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                and isinstance(first.value.value, str):
            doc_lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    out = []
    for i, line in enumerate(src.split("\n"), start=1):
        if i in doc_lines:
            continue
        s = line.split("#", 1)[0]
        if s.strip():
            out.append(s)
    return "\n".join(out)


def _fn_block(src: str, name: str) -> str:
    """`def name(` 함수 본문(주석 제거)."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            lines = src.split("\n")[node.lineno - 1:node.end_lineno]
            return _code(textwrap.dedent("\n".join(lines)))
    raise AssertionError(f"함수 {name} 부재")


def _call_block(src: str, name: str) -> str:
    """`name = torch.cat(` 다중행 호출의 괄호 안 본문 — 괄호 균형으로 끝을 찾는다."""
    m = re.search(rf"\n\s*{re.escape(name)} = torch\.cat\(", src)
    assert m, f"{name} = torch.cat( 부재"
    i = src.index("(", m.start())
    depth, j = 0, i
    while j < len(src):
        if src[j] == "(":
            depth += 1
        elif src[j] == ")":
            depth -= 1
            if depth == 0:
                return src[i + 1:j]
        j += 1
    raise AssertionError(f"{name} 호출의 괄호가 안 닫힌다")


def _ordered(block: str, tokens: list[str]) -> None:
    idx = [block.find(t) for t in tokens]
    missing = [t for t, i in zip(tokens, idx) if i < 0]
    assert not missing, f"누락 {missing}"
    assert idx == sorted(idx), f"순서 어긋남 {list(zip(tokens, idx))}"


def _class_methods(src: str, cls: str) -> set[str]:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
    raise AssertionError(f"클래스 {cls} 부재")

"""생성된 보상 코드 로더.

파일 계약(Eureka 형 시그니처를 배치 텐서로):
    def compute_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]
        return total (N,), {"term_name": (N,), ...}

★`exec` 로 읽되 이름공간은 `torch`·`math` 뿐이다(text2reward 는 `exec` 후 메서드를
  갈아끼운다 — 같은 방식). 검증은 validator 가 따로 한다; 로더는 서명과 반환 형태만 본다.
★경로가 비어 있으면 **영 보상**(terms 없음) — Phase 1 부팅/무작위 롤아웃용.
"""

from __future__ import annotations

import math
import os
from typing import Callable

import torch

from .context import RewardContext

RewardFn = Callable[[RewardContext], tuple[torch.Tensor, dict[str, torch.Tensor]]]

ENTRY_NAME = "compute_reward"


def zero_reward(ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return torch.zeros(ctx.num_envs, device=ctx.device), {}


def load_reward_fn(path: str | None) -> tuple[RewardFn, str]:
    """(함수, 출처 설명). path 가 비면 zero_reward."""
    if not path:
        return zero_reward, "zero (reward_code_path 비어 있음)"
    if not os.path.isfile(path):
        raise FileNotFoundError(f"reward_code_path 가 없다: {path}")
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    ns: dict = {"torch": torch, "math": math, "RewardContext": RewardContext,
                "__name__": "t2r_generated_reward"}
    code = compile(src, path, "exec")
    exec(code, ns)   # noqa: S102 — 생성 코드는 validator 가 먼저 거른다
    fn = ns.get(ENTRY_NAME)
    if not callable(fn):
        raise RuntimeError(f"{path}: `{ENTRY_NAME}(ctx)` 가 정의되지 않았다")
    return fn, path


def call_reward_fn(fn: RewardFn, ctx: RewardContext) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """호출 + 반환 형태 강제. 형태가 틀리면 fail-loud(조용한 브로드캐스트 금지)."""
    out = fn(ctx)
    if not (isinstance(out, tuple) and len(out) == 2):
        raise RuntimeError(f"{ENTRY_NAME} 는 (reward, terms) 튜플을 돌려줘야 한다: {type(out)}")
    total, terms = out
    n = ctx.num_envs
    if not isinstance(total, torch.Tensor) or tuple(total.shape) != (n,):
        raise RuntimeError(
            f"{ENTRY_NAME} 의 reward 는 (N,)=({n},) 텐서여야 한다: "
            f"{getattr(total, 'shape', type(total))}")
    if not isinstance(terms, dict):
        raise RuntimeError(f"{ENTRY_NAME} 의 terms 는 dict 여야 한다: {type(terms)}")
    for k, v in terms.items():
        if not isinstance(v, torch.Tensor) or tuple(v.shape) != (n,):
            raise RuntimeError(
                f"{ENTRY_NAME} terms['{k}'] 는 (N,) 텐서여야 한다: "
                f"{getattr(v, 'shape', type(v))}")
    return total, terms

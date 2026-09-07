"""SAPG 블록 산술 — 순수 torch. rl_games·isaaclab 의존 없음.

SimToolReal 이 쓰는 SAPG(`use_others_experience: lf`)의 **핵심 산술만** 떼어낸 모듈이다.
포크된 rl_games 를 통째로 들여오면 SAPG 와 무관한 버전 드리프트 465 줄이 함께 딸려와
IsaacLab 2.3.0 래퍼와 충돌할 위험이 있다(09.07 함수 단위 실측: SAPG 892 / 드리프트 465).
여기 있는 것은 Isaac 없이 단위테스트할 수 있는 부분이고, rl_games 훅은 이것을 호출한다.

구조
----
환경 `num_actors` 개를 `block_size` 씩 잘라 `num_blocks` 개 블록으로 나눈다. 블록마다
**다른 entropy 계수**를 주되 마지막 블록은 0 — 그 블록이 "리더"(평가용 정책)다.

    계수 = linspace(0.5, 0, num_blocks)[블록] × expl_reward_coef_scale

SimToolReal 원본(24,576 env ÷ 4,096 = 6블록, scale 0.002)의 실효값:
    0.00100 · 0.00080 · 0.00060 · 0.00040 · 0.00020 · 0.00000(리더)
즉 yaml 의 `entropy_coef: 0.0` 은 **쓰이지 않는다** — expl 경로가 그 값을 건너뛴다.

★블록이 1개면 `linspace(0.5, 0, 1) = [0.5]` 라 계수가 균일해지고 리더/팔로워가 사라진다.
  그건 SAPG 가 아니라 그냥 PPO 다(우리 fj_b3 와 동일). 그래서 여기서 **거부**한다 —
  4,096 env 에 원본 block_size 4,096 을 그대로 쓰면 정확히 이 함정에 빠진다.

리더-팔로워 재라벨링
------------------
같은 롤아웃을 `roll_blocks(·, k)` 로 k 블록만큼 밀어 "다른 탐색 계수가 낸 경험"인 것처럼
다시 라벨링한다. 롤 뒤 **리더 계수를 달게 되는 블록**은 정확히 `(k-1) % num_blocks` 이고
(`leader_block_after_roll`), `filter_leader` 가 그 구간만 남긴다. 결과적으로 리더 정책이
팔로워들이 발견한 궤적으로도 학습한다 — 이게 SAPG 가 탐색을 공유하는 지점이다.
"""

from __future__ import annotations

import dataclasses

import torch

__all__ = [
    "SAPGBlocks",
    "SAPGCfg",
    "filter_leader",
    "leader_block_after_roll",
    "roll_blocks",
    "sample_repeat_idxs",
]

# 원본 상수. embedding 은 관측에 실려 정책이 "나는 어느 탐색 블록인가"를 알게 하는 스칼라다.
_EMBD_HI, _EMBD_LO = 50.0, 0.0
# 계수 램프의 양 끝. 마지막 블록이 0 = 리더(탐색 보너스 없음).
_COEF_HI, _COEF_LO = 0.5, 0.0


@dataclasses.dataclass(frozen=True)
class SAPGCfg:
    """SimToolRealSAPG.yaml 의 expl_* 블록과 1:1."""

    block_size: int
    coef_scale: float
    off_policy_ratio: float = 1.0

    def __post_init__(self) -> None:
        if int(self.block_size) < 1:
            raise ValueError(f"block_size ≥ 1, got {self.block_size}")
        if float(self.coef_scale) < 0.0:
            raise ValueError(f"coef_scale ≥ 0, got {self.coef_scale}")
        if float(self.off_policy_ratio) < 0.0:
            raise ValueError(f"off_policy_ratio ≥ 0, got {self.off_policy_ratio}")


class SAPGBlocks:
    """블록 분할 + 블록별 계수/식별자. 상태를 갖지 않는 값 묶음."""

    def __init__(self, num_actors: int, cfg: SAPGCfg, device: torch.device | str) -> None:
        bs = int(cfg.block_size)
        if int(num_actors) % bs != 0:
            raise ValueError(
                f"num_actors({num_actors}) 가 block_size({bs}) 로 나누어떨어져야 한다")
        nb = int(num_actors) // bs
        if nb < 2:
            raise ValueError(
                f"블록이 {nb}개다 — SAPG 는 블록이 2개 이상이어야 리더/팔로워가 생긴다. "
                f"num_actors({num_actors}) 에 block_size({bs}) 를 쓰면 계수가 균일해져 "
                f"평범한 PPO 와 같아진다. block_size 를 줄여라(예: {num_actors // 4}).")
        self.num_actors, self.block_size, self.num_blocks = int(num_actors), bs, nb
        self.device = device
        # 블록 id: [0]*bs + [1]*bs + ... — 환경 순서 그대로 잘린다.
        self.ids = torch.arange(nb, device=device).repeat_interleave(bs)
        # 관측에 붙는 식별자(N,1). 원본은 'learn_param'/'disjoint' 에서 스칼라 그대로 쓴다.
        self.embedding = torch.linspace(
            _EMBD_HI, _EMBD_LO, nb, device=device)[self.ids].reshape(-1, 1)
        # 블록별 entropy 계수(N,). 마지막 블록 = 0 = 리더.
        self.entropy_coef = torch.linspace(
            _COEF_HI, _COEF_LO, nb, device=device)[self.ids] * float(cfg.coef_scale)

    @property
    def embd_dim(self) -> int:
        """관측·상태에 추가되는 폭. 이만큼 observation_space 가 넓어진다."""
        return int(self.embedding.shape[-1])

    def block_coefs(self) -> torch.Tensor:
        """블록당 하나씩 (num_blocks,) — 로깅·검증용."""
        return self.entropy_coef[:: self.block_size]

    def __repr__(self) -> str:  # pragma: no cover - 진단 출력
        c = [round(float(v), 6) for v in self.block_coefs().tolist()]
        return (f"SAPGBlocks(env {self.num_actors} = {self.num_blocks}블록 × "
                f"{self.block_size} · entropy_coef {c} · 리더=마지막)")


def roll_blocks(t: torch.Tensor, block_size: int, k: int, dim: int = 0) -> torch.Tensor:
    """블록 단위 회전. `k` 블록만큼 밀어 다른 탐색 계수로 재라벨링한다."""
    return torch.roll(t, int(block_size) * int(k), dims=dim)


def leader_block_after_roll(k: int, num_blocks: int) -> int:
    """`roll_blocks(·, k)` 뒤 **리더 계수를 달게 되는** 블록 번호.

    롤은 위치 p 의 값을 p−bs·k 에서 가져온다. 따라서 블록 b 는 linspace[b−k mod nb] 를 단다.
    리더 계수는 linspace[nb−1] 이므로 b−k ≡ nb−1, 즉 b ≡ k−1 (mod nb).
    `filter_leader` 의 `(idx-1)*bsize` 슬라이스가 바로 이 항등식이다.
    """
    return (int(k) - 1) % int(num_blocks)


def sample_repeat_idxs(
    num_blocks: int, off_policy_ratio: float, generator: torch.Generator | None = None
) -> list[int]:
    """이번 업데이트에 섞을 롤 양들. 항상 0(자기 자신)을 포함한다.

    원본은 `min(num_blocks, off_policy_ratio + 1)` 개를 쓰고 1..nb−1 에서 비복원 추출한다.
    """
    nb = int(num_blocks)
    n = min(nb, int(off_policy_ratio) + 1)
    if n <= 1:
        return [0]
    perm = torch.randperm(nb - 1, generator=generator)[: n - 1] + 1
    return [0] + [int(v) for v in perm.tolist()]


def filter_leader(
    val: torch.Tensor, orig_len: int, repeat_idxs: list[int], num_blocks: int, dim: int = 0
) -> torch.Tensor:
    """재라벨링된 복사본에서 **리더로 라벨된 구간만** 남긴다.

    `val` 은 `repeat_idxs` 개의 복사본을 이어붙인 것이다(각 길이 `orig_len`).
    - roll 0(자기 자신)인 복사본은 통째로 남긴다.
    - roll k≠0 인 복사본은 `leader_block_after_roll(k)` 블록 구간만 남긴다.
      그 구간이 "팔로워가 굴렸지만 리더 라벨이 붙은" 경험이다.
    """
    nb = int(num_blocks)
    bsize = int(orig_len) // nb
    out = []
    for i, k in enumerate(repeat_idxs):
        base = i * int(orig_len)
        if int(k) == 0:
            lo, hi = base, base + int(orig_len)
        else:
            b = leader_block_after_roll(k, nb)
            lo, hi = base + b * bsize, base + (b + 1) * bsize
        out.append(val[lo:hi] if dim == 0 else val[:, lo:hi])
    return torch.cat(out, dim=dim)

"""cup pose 주입 provider seam.

live: env 현행 경로(GT+DR노이즈) — override 없음.
state_frozen: reset 시 GT를 1회 캡처해 에피소드 내내 고정(배포 open-loop 재현).
camera_frozen: SP2 — 카메라 렌더+FoundationPose (미구현).
설계: docs/superpowers/specs/2026-08-04-grasp-v1-s2r-eval-sp1-design.md §4.3
"""
from __future__ import annotations

from typing import Protocol

import torch


class PoseProvider(Protocol):
    def on_reset(self, env, env_ids: torch.Tensor) -> None: ...
    def get_override(self, env) -> torch.Tensor | None: ...


class LiveProvider:
    def on_reset(self, env, env_ids: torch.Tensor) -> None:
        pass

    def get_override(self, env) -> torch.Tensor | None:
        return None


class StateFrozenProvider:
    def __init__(self) -> None:
        self._buf: torch.Tensor | None = None

    def on_reset(self, env, env_ids: torch.Tensor) -> None:
        pos = env.object_pos  # [N,3] env-origin local (grasp_*_env.py: root_pos_w - env_origins)
        if not torch.isfinite(pos[env_ids]).all():
            raise ValueError(f"non-finite object_pos at reset for envs {env_ids.tolist()}")
        # 불변 패턴: 기존 버퍼를 제자리 수정하지 않고 새 텐서 생성
        if self._buf is None:
            self._buf = pos.detach().clone()
        else:
            buf = self._buf.clone()
            buf[env_ids] = pos[env_ids].detach().clone()
            self._buf = buf

    def get_override(self, env) -> torch.Tensor:
        if self._buf is None:
            raise RuntimeError("StateFrozenProvider.get_override called before on_reset")
        return self._buf


def make_provider(name: str) -> PoseProvider:
    if name == "live":
        return LiveProvider()
    if name == "state_frozen":
        return StateFrozenProvider()
    if name == "camera_frozen":
        raise NotImplementedError("camera_frozen provider는 SP2에서 구현 (spec §8)")
    raise ValueError(f"unknown pose_source: {name!r} (live|state_frozen|camera_frozen)")

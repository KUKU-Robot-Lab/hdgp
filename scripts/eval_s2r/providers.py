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
    """Reset 시 객체 위치를 캡처해 에피소드 내 고정 반환.

    의미론:
    - on_reset(env, env_ids): env_ids에 속한 환경들의 object_pos를 캡처. 미검증 환경은 NaN으로 표시.
    - get_override(env): 버퍼의 방어 복사 반환. 다운스트림 in-place 수정이 buffer 부패 불가.
    - 부분 first reset 후 get_override는 RuntimeError (미리셋 환경 NaN 검출).
    """
    def __init__(self) -> None:
        self._buf: torch.Tensor | None = None

    def on_reset(self, env, env_ids: torch.Tensor) -> None:
        pos = env.object_pos  # [N,3] env-origin local (grasp_*_env.py: root_pos_w - env_origins)
        if not torch.isfinite(pos[env_ids]).all():
            raise ValueError(f"non-finite object_pos at reset for envs {env_ids.tolist()}")
        # 불변 패턴: 기존 버퍼를 제자리 수정하지 않고 새 텐서 생성.
        # 부분 first reset 방지: 항상 전체 buffer를 대체, env_ids만 갱신.
        if self._buf is None:
            # 첫 호출: 전체 NaN으로 초기화, env_ids만 채우기
            buf = torch.full_like(pos, float("nan"))
            buf[env_ids] = pos[env_ids].detach().clone()
            self._buf = buf
        else:
            buf = self._buf.clone()
            buf[env_ids] = pos[env_ids].detach().clone()
            self._buf = buf

    def get_override(self, env) -> torch.Tensor:
        if self._buf is None:
            raise RuntimeError("StateFrozenProvider.get_override called before on_reset")
        # NaN 존재 확인: 부분 first reset 오용 또는 미리셋 환경 검출
        if not torch.isfinite(self._buf).all():
            raise RuntimeError(
                f"StateFrozenProvider.get_override: some envs never reset (contains NaN). "
                f"Check that on_reset() includes all env_ids."
            )
        # 방어 복사: 다운스트림 in-place 수정이 frozen buffer를 부패시키지 않도록
        return self._buf.clone()


def make_provider(name: str) -> PoseProvider:
    if name == "live":
        return LiveProvider()
    if name == "state_frozen":
        return StateFrozenProvider()
    if name == "camera_frozen":
        raise NotImplementedError("camera_frozen provider는 SP2에서 구현 (spec §8)")
    raise ValueError(f"unknown pose_source: {name!r} (live|state_frozen|camera_frozen)")

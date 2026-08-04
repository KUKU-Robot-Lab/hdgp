"""cup pose 주입 provider seam.

live: env 현행 경로(GT+DR노이즈) — override 없음.
state_frozen: reset 시 GT를 1회 캡처해 에피소드 내내 고정(배포 open-loop 재현).
camera_frozen: SP2 — 카메라 렌더+FoundationPose 결과 파일을 로드해 고정 주입.
설계: docs/superpowers/specs/2026-08-04-grasp-v1-s2r-eval-sp1-design.md §4.3
      docs/superpowers/specs/2026-08-04-grasp-v1-s2r-eval-sp2-camera-design.md §4.3
"""
from __future__ import annotations

import json
from typing import Protocol

import numpy as np
import torch

from scripts.eval_s2r.transforms import compose_local_pose


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
    - on_reset(env, env_ids): env_ids에 속한 환경들의 object_init_pos를 캡처. 미검증 환경은 NaN으로 표시.
      object_pos가 아닌 object_init_pos를 쓰는 이유: object_pos는 _get_dones()
      (→ _compute_intermediate_values) 안에서만 갱신되는데, on_reset은 env.reset() 직후
      (아직 _get_dones 미실행 → 0으로 남은 stale) 또는 env.step()의 내부 auto-reset 직후
      (직전 에피소드의 마지막 관측값이 남은 stale)에 호출된다. 두 경우 모두 object_pos는
      "지금 이 리셋"의 값이 아니다. 반면 object_init_pos는 _reset_idx()가 정확히 리셋되는
      env_ids에 대해 env-origin-local GT 스폰 위치를 그 자리에서 기록한다
      (grasp_right_env.py:1593, grasp_left_env.py:1615) — on_reset 시점에 이미 최신값.
    - get_override(env): 버퍼의 방어 복사 반환. 다운스트림 in-place 수정이 buffer 부패 불가.
    - 부분 first reset 후 get_override는 RuntimeError (미리셋 환경 NaN 검출).
    """
    def __init__(self) -> None:
        self._buf: torch.Tensor | None = None

    def on_reset(self, env, env_ids: torch.Tensor) -> None:
        pos = env.object_init_pos  # [N,3] env-origin local GT 스폰 pose (grasp_*_env.py:1593/1615, _reset_idx가 즉시 기록)
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


class CameraFileProvider:
    """카메라 렌더+FoundationPose 결과 파일(poses.json+meta.json)을 로드해 고정 주입.

    의미론:
    - 생성 시 1회 로드: env별 T_local_cam(meta) @ T_cam_obj(poses) → env-local 위치 [N,3] 버퍼 구성.
      poses.json은 Task 4 fp_batch, meta.json은 Task 3 render_pass 산출물(스키마는
      task-2-brief.md에 고정, 임의 변경 시 test_camera_provider.py가 잡는다).
    - 실패 env(ok=false, 키 누락, 또는 ok=true인데 T_cam_obj 비유한)는 버퍼 행이 NaN이고
      `failed_envs`(env id 집합)에 포함. `fail_reasons`에 사유 문자열 기록.
      계약: 실패 env를 실제로 돌리지 않는 책임은 eval 루프 쪽(Task 5) — 여기서는 예외를
      던지지 않고(개별 env 실패로 전체 로드를 막지 않기 위해) 표시만 한다.
    - on_reset: 파일 고정 데이터라 매 리셋마다 다시 캡처할 게 없음 → no-op.
    - get_override: StateFrozenProvider와 동일하게 방어적 `.clone()` 반환.
    """

    def __init__(self, poses_path: str, frames_meta_path: str) -> None:
        with open(frames_meta_path) as f:
            meta = json.load(f)
        with open(poses_path) as f:
            poses_data = json.load(f)

        num_envs = meta["num_envs"]
        if poses_data["num_envs"] != num_envs:
            raise ValueError(
                f"num_envs mismatch: meta={num_envs} poses={poses_data['num_envs']}"
            )

        self.expected_grid: dict = dict(meta["grid"])  # 방어 복사: 파싱된 JSON dict 원본을 노출하지 않음

        T_local_cam_by_env = meta["T_local_cam"]
        poses_by_env = poses_data["poses"]

        buf = np.full((num_envs, 3), np.nan, dtype=np.float32)
        failed_envs: set[int] = set()
        fail_reasons: dict[int, str] = {}

        for env_id in range(num_envs):
            key = str(env_id)
            entry = poses_by_env.get(key)
            if entry is None:
                failed_envs.add(env_id)
                fail_reasons[env_id] = "missing"
                continue
            if not entry.get("ok", False):
                failed_envs.add(env_id)
                fail_reasons[env_id] = entry.get("reason", "unknown")
                continue

            T_cam_obj = np.asarray(entry["T_cam_obj"], dtype=float)
            if not np.isfinite(T_cam_obj).all():
                # ok=true여도 비유한 값이면 예외 없이 실패로 강등 (compose_local_pose는 raise함)
                failed_envs.add(env_id)
                fail_reasons[env_id] = "nonfinite"
                continue

            T_local_cam = np.asarray(T_local_cam_by_env[key], dtype=float)
            try:
                buf[env_id] = compose_local_pose(T_local_cam, T_cam_obj).astype(np.float32)
            except ValueError:
                # 유한하지만 회전부가 비직교(퇴화한 FoundationPose 출력 등) → 개별 env만 강등,
                # 다른 env 구성을 막지 않는다 (클래스 계약: 생성자는 예외 없이 표시만 한다)
                failed_envs.add(env_id)
                fail_reasons[env_id] = "invalid_rotation"

        self.failed_envs = failed_envs
        self.fail_reasons = fail_reasons
        self._buf = torch.from_numpy(buf)

    def on_reset(self, env, env_ids: torch.Tensor) -> None:
        pass  # 파일 고정 데이터라 리셋마다 다시 캡처할 것이 없음

    def get_override(self, env) -> torch.Tensor:
        return self._buf.clone()  # 방어 복사: StateFrozenProvider와 동일 패턴


def make_provider(name: str, **kwargs) -> PoseProvider:
    if name == "live":
        return LiveProvider()
    if name == "state_frozen":
        return StateFrozenProvider()
    if name == "camera_frozen":
        try:
            poses_path = kwargs["poses_path"]
            frames_meta_path = kwargs["frames_meta_path"]
        except KeyError as e:
            raise ValueError(
                f"camera_frozen requires poses_path and frames_meta_path kwargs (missing {e})"
            ) from e
        return CameraFileProvider(poses_path, frames_meta_path)
    raise ValueError(f"unknown pose_source: {name!r} (live|state_frozen|camera_frozen)")

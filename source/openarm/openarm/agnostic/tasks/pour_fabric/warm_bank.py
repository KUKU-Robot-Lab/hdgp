"""warm 뱅크 로더 — grasp 정책 성공 종료 상태를 pour 초기 상태로.

**설계 원칙 (skill-chaining 의 s2r 정합 조건 2가지):**
  1. 분포 충실도 — 뱅크는 배포할 grasp 정책(grasp_lift_fabric)의 성공 종료 상태에서
     수집하고, 물리 플래그(robot_usd·gravity·self-collision)를 meta 에 기록해
     로드 시 **hard-fail 대조**한다(08.17 DG-5F 조용한 불일치 사고의 재발 방지).
  2. 정책 상태는 저장하지 않는다 — slew 지령·prev_action 은 실기 인계 순간
     측정량(관절+FK)에서 재구성한다. env._reset_idx 의 재구성 함수가 곧
     실기 인계 프로토콜이다. 뱅크에는 **물리 상태만** 담는다.

스키마 (HDF5 group "warm_states"):
    joint_pos    (N, J) float32 — 이 뱅크가 담당하는 팔+손 관절 **측정** 위치
    joint_target (N, J) float32 — 같은 관절의 **PD 지령 목표** (★필수)
        ★측정 위치만 저장하면 파지가 풀린다 — 파지력 = kp·(target − q) 인데
          target=q 로 복원하면 오차가 0 이라 힘이 소멸한다(실측: 복원 직후 22N →
          120스텝 뒤 2.4N, 컵이 손에서 93mm 미끄러져 낙하). 지령 목표는 실기에서도
          컨트롤러가 만드는 값이라 s2r 계약(측정가능량+지령량 재구성)에 부합한다.
    cup_pose    (N, 7)  float32 — env-local pos(3) + quat wxyz(4)
    bead_state  (N, k, 13) float32 — env-local root state (source 뱅크만, 선택)
    num_contacts (N,)   int64
  attrs: joint_names(S, J) · robot_usd · profile · checkpoint · git_hash
         · enable_gravity · enable_self_collisions

pour_v1 warm_state_bank 에서 계승한 게이트: robot_usd hard-fail, bead all-or-nothing.
계승하지 않는 것: 하드코딩 fallback 경로(/home/oem/…) · 체크포인트 롤아웃 폴백
· spawn_z 게이트(스폰 z 는 이제 origin_offset+surface_z 로 파생되어 뱅크와 무관).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PourWarmBank:
    """한쪽 팔 담당 뱅크 (source=파지+비드 / receiver=파지)."""

    path: str
    joint_names: tuple            # (J,) 뱅크 저장 순서의 관절 이름
    joint_pos: np.ndarray         # (N, J) 측정 위치
    joint_target: np.ndarray      # (N, J) PD 지령 목표 (파지력의 원천)
    cup_pose: np.ndarray          # (N, 7) env-local pos + quat wxyz
    num_contacts: np.ndarray      # (N,)
    bead_state: np.ndarray | None  # (N, k, 13) env-local | None
    meta: dict

    def __len__(self) -> int:
        return int(self.joint_pos.shape[0])

    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expect_robot_usd: str,
        expect_gravity: bool,
        expect_self_collisions: bool,
        min_states: int = 64,
    ) -> "PourWarmBank":
        import h5py

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"warm 뱅크가 없다: {path}\n"
                "  scripts/warm_states/collect_pour_fab_warm_states.py 로 수집하거나 "
                "probe 전용이면 cfg.require_warm_bank=False 로 띄워라.")

        with h5py.File(path, "r") as f:
            if "warm_states" not in f:
                raise RuntimeError(f"{path}: 'warm_states' group 이 없다 (구 스키마?)")
            g = f["warm_states"]
            if "joint_target" not in g:
                raise RuntimeError(
                    f"{path}: 'joint_target' 부재 — 측정 위치만으로는 파지력이 "
                    "복원되지 않는다(구 스키마). 재수집할 것.")
            joint_pos = np.asarray(g["joint_pos"], dtype=np.float32)
            joint_target = np.asarray(g["joint_target"], dtype=np.float32)
            cup_pose = np.asarray(g["cup_pose"], dtype=np.float32)
            num_contacts = np.asarray(g["num_contacts"], dtype=np.int64)
            bead_state = (np.asarray(g["bead_state"], dtype=np.float32)
                          if "bead_state" in g else None)
            meta = {k: g.attrs[k] for k in g.attrs}
            joint_names = tuple(
                n.decode() if isinstance(n, bytes) else str(n)
                for n in g.attrs["joint_names"])

        n = joint_pos.shape[0]
        if n < min_states:
            raise RuntimeError(f"{path}: 상태 {n}개 < 최소 {min_states}")
        if cup_pose.shape != (n, 7):
            raise RuntimeError(f"{path}: cup_pose {cup_pose.shape} != ({n},7)")
        if joint_target.shape != joint_pos.shape:
            raise RuntimeError(
                f"{path}: joint_target {joint_target.shape} != joint_pos {joint_pos.shape}")
        if joint_pos.shape[1] != len(joint_names):
            raise RuntimeError(
                f"{path}: joint_pos 폭 {joint_pos.shape[1]} != "
                f"joint_names {len(joint_names)}")

        # ---- hard-fail 게이트 -------------------------------------------------
        def _meta_str(k: str) -> str:
            v = meta.get(k, "")
            return v.decode() if isinstance(v, bytes) else str(v)

        got_usd = _meta_str("robot_usd")
        if got_usd != expect_robot_usd:
            raise RuntimeError(
                f"{path}: robot_usd 불일치 — 뱅크 '{got_usd}' vs 기대 '{expect_robot_usd}'.\n"
                "  다른 로봇으로 수집된 뱅크다. 차원이 맞아도 기하가 다르다 — 재수집할 것.")
        for key, expect in (("enable_gravity", expect_gravity),
                            ("enable_self_collisions", expect_self_collisions)):
            if key not in meta:
                raise RuntimeError(
                    f"{path}: meta '{key}' 부재 — 물리 플래그 미기록 뱅크는 쓰지 않는다.")
            if bool(meta[key]) != bool(expect):
                raise RuntimeError(
                    f"{path}: {key} 불일치 — 뱅크 {bool(meta[key])} vs env {expect}.\n"
                    "  다른 물리로 수집된 상태는 리셋 직후 정착 거동이 다르다 — 재수집할 것.")

        # ---- bead all-or-nothing (pour_v1 규약 계승) ---------------------------
        if bead_state is not None and (
                bead_state.ndim != 3 or bead_state.shape[0] != n
                or bead_state.shape[2] != 13):
            print(f"[warm_bank] ⚠ {path}: bead_state {bead_state.shape} 형식 불량 — "
                  "전부 버린다(부분 적용 금지)", flush=True)
            bead_state = None

        if (not np.isfinite(joint_pos).all() or not np.isfinite(cup_pose).all()
                or not np.isfinite(joint_target).all()):
            raise RuntimeError(f"{path}: joint_pos/joint_target/cup_pose 에 NaN/Inf")

        return cls(path=str(path), joint_names=joint_names, joint_pos=joint_pos,
                   joint_target=joint_target, cup_pose=cup_pose,
                   num_contacts=num_contacts, bead_state=bead_state, meta=meta)


def save_bank(
    path: str | Path,
    *,
    joint_names: tuple,
    joint_pos: np.ndarray,
    joint_target: np.ndarray,
    cup_pose: np.ndarray,
    num_contacts: np.ndarray,
    bead_state: np.ndarray | None,
    meta: dict,
) -> None:
    """collector 용 대칭 writer — 로더와 같은 파일에 두어 스키마 표류를 막는다."""
    import h5py

    required = ("robot_usd", "enable_gravity", "enable_self_collisions",
                "profile", "checkpoint")
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(f"meta 필수 키 누락: {missing}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group("warm_states")
        g.create_dataset("joint_pos", data=np.asarray(joint_pos, dtype=np.float32))
        g.create_dataset("joint_target", data=np.asarray(joint_target, dtype=np.float32))
        g.create_dataset("cup_pose", data=np.asarray(cup_pose, dtype=np.float32))
        g.create_dataset("num_contacts", data=np.asarray(num_contacts, dtype=np.int64))
        if bead_state is not None:
            g.create_dataset("bead_state", data=np.asarray(bead_state, dtype=np.float32))
        g.attrs["joint_names"] = [str(n) for n in joint_names]
        for k, v in meta.items():
            g.attrs[k] = v

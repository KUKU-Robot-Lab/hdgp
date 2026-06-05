"""IL Phase 환경: demo 궤적 추적 보상으로 전체 grasp→pour 모션 사전학습.

warmstart 없음. source cup만 소환. a11-a20 demo에서 팔/손 관절 목표 추출.
RL Phase(5g_pour_right-v6-diffusion)에서 이 checkpoint를 가져와 fine-tune.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import List

import numpy as np
import torch

from .il_demo_loader import ILTrajectory, load_il_trajectory
from .pour_right_env import PourRightEnv
from .pour_right_env_cfg import PourRightEnvCfg
from .pour_right_constants import NUM_ARM_DOF, NUM_HAND_DOF

_HDGP_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../../../../")
)
_DATASETS_DIR = os.path.normpath(os.path.join(_HDGP_ROOT, "..", "datasets"))
_DEFAULT_IL_DEMOS = tuple(
    os.path.join(_DATASETS_DIR, f"pour_v1_a{i}.hdf5") for i in range(11, 21)
)


class ILPourRightEnvCfg(PourRightEnvCfg):
    """IL Phase 설정 — warmstart 비활성, joint tracking reward 사용."""

    # ---- warmstart 비활성 ----
    enable_warmstart_reset: bool = False
    enable_demo_grasp_reset: bool = False  # HDF5 demo grasp reset 비사용

    # ---- IL 전용 파라미터 ----
    il_demo_paths: tuple[str, ...] = _DEFAULT_IL_DEMOS
    weight_arm_tracking: float = 5.0    # arm joint error 패널티 weight
    weight_hand_tracking: float = 2.0   # hand joint error 패널티 weight


def _pad_to_max(arrays: List[np.ndarray], pad_value: float = 0.0) -> np.ndarray:
    """(T_i, D) 리스트 → (N, T_max, D) 패딩된 배열."""
    T_max = max(a.shape[0] for a in arrays)
    D = arrays[0].shape[1]
    out = np.full((len(arrays), T_max, D), pad_value, dtype=np.float32)
    for i, a in enumerate(arrays):
        out[i, : a.shape[0]] = a
    return out


class ILPourRightEnv(PourRightEnv):
    """IL Phase 환경.

    _reset_idx: warmstart 없이 demo[0] 관절 설정에서 시작.
    _get_rewards: arm/hand joint tracking reward만 반환.
    """

    def __init__(self, cfg: ILPourRightEnvCfg, render_mode: str | None = None) -> None:
        # demo 로드 (Isaac 초기화 전에 완료)
        trajs: List[ILTrajectory] = []
        for p in cfg.il_demo_paths:
            if Path(p).exists():
                trajs.append(load_il_trajectory(Path(p)))
            else:
                print(f"[ILPourRightEnv] missing demo: {p}")
        if not trajs:
            raise RuntimeError("IL demo 파일을 하나도 찾지 못했습니다.")

        self._il_trajs = trajs

        # 패딩된 tensor (CPU, 나중에 device로 이동)
        arm_np  = _pad_to_max([t.arm_joints  for t in trajs])  # (D, T_max, 7)
        hand_np = _pad_to_max([t.hand_joints for t in trajs])  # (D, T_max, 20)
        lengths = np.array([t.arm_joints.shape[0] for t in trajs], dtype=np.int64)

        self._il_arm_np   = arm_np
        self._il_hand_np  = hand_np
        self._il_lengths_np = lengths

        super().__init__(cfg, render_mode)

        # device 이동 (super().__init__ 후 device 확정)
        self._il_arm   = torch.from_numpy(arm_np).to(self.device)    # (D, T_max, 7)
        self._il_hand  = torch.from_numpy(hand_np).to(self.device)   # (D, T_max, 20)
        self._il_lengths = torch.from_numpy(lengths).to(self.device) # (D,)
        self._il_T_max = int(arm_np.shape[1])

        # per-env 추적 상태
        self._il_demo_idx  = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._il_demo_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        """warmstart 없이 demo[0] 관절 설정에서 시작."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        if len(env_ids) == 0:
            return

        n = len(env_ids)
        env_ids_t = torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)

        # 랜덤 demo 배정
        demo_idxs = torch.randint(len(self._il_trajs), (n,), device=self.device)
        self._il_demo_idx[env_ids_t]  = demo_idxs
        self._il_demo_step[env_ids_t] = 0

        # demo[0] 관절 → 시뮬레이터에 직접 기입
        demo_arm_0  = self._il_arm[demo_idxs, 0]   # (n, 7)
        demo_hand_0 = self._il_hand[demo_idxs, 0]  # (n, 20)

        full_pos = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_vel = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_pos[:, self.arm_dof_indices]  = demo_arm_0
        full_pos[:, self.hand_dof_indices] = demo_hand_0
        self.robot.write_joint_state_to_sim(full_pos, full_vel, env_ids=env_ids_t)

        # Fabrics 상태도 동기화
        self.fabric_q[env_ids_t, :NUM_ARM_DOF]  = demo_arm_0
        self.fabric_q[env_ids_t, NUM_ARM_DOF:]  = demo_hand_0
        self.fabric_qd[env_ids_t].zero_()
        self.fabric_qdd[env_ids_t].zero_()

    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        """arm/hand joint tracking error 기반 보상."""
        arm_joint_pos  = self.robot.data.joint_pos[:, self.arm_dof_indices]   # (N, 7)
        hand_joint_pos = self.robot.data.joint_pos[:, self.hand_dof_indices]  # (N, 20)

        # 현재 demo step 인덱스 (범위 클램프)
        demo_lengths_per_env = self._il_lengths[self._il_demo_idx]  # (N,)
        step_clamped = self._il_demo_step.clamp(max=self._il_T_max - 1)
        # 에피소드 끝 도달 시 마지막 프레임 반복
        step_clamped = torch.minimum(step_clamped, demo_lengths_per_env - 1)

        # 벡터화된 인덱싱: (N,) demo + step → (N, D)
        demo_arm  = self._il_arm [self._il_demo_idx, step_clamped]   # (N, 7)
        demo_hand = self._il_hand[self._il_demo_idx, step_clamped]   # (N, 20)

        r_arm  = -self.cfg.weight_arm_tracking  * (arm_joint_pos  - demo_arm ).pow(2).sum(-1)
        r_hand = -self.cfg.weight_hand_tracking * (hand_joint_pos - demo_hand).pow(2).sum(-1)

        self._il_demo_step += 1

        return r_arm + r_hand

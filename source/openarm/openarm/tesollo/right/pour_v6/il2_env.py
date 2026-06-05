"""IL Phase 1 환경: EEF 궤적 추종 보상으로 grasp→pour 사전학습.

구조:
  - source cup만 존재 (target cup은 왼팔 FK에 부착, beads 미소환)
  - ARM_START_POSE에서 warmstart 없이 시작
  - 보상: r = -w_pos*||palm_pos - demo_eef[t]|| - w_rot*angle(palm_quat, demo_quat[t])
  - 손가락: demo right_hand_joint_pos[t] open-loop replay (RL 학습 제외)
  - 관측: 52D (Phase 2와 동일 layout, 직접 연결 가능)
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import List

import numpy as np
import torch

from .il2_demo_loader import IL2Trajectory, load_il2_trajectory
from .pour_right_env import PourRightEnv
from .pour_right_env_cfg import PourRightEnvCfg
from .pour_right_constants import NUM_ARM_DOF

_HDGP_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../../../../")
)
_DATASETS_DIR = os.path.normpath(os.path.join(_HDGP_ROOT, "..", "datasets"))
_DEFAULT_IL2_DEMOS = tuple(
    os.path.join(_DATASETS_DIR, f"pour_v1_a{i}.hdf5") for i in range(11, 21)
)


class IL2PourRightEnvCfg(PourRightEnvCfg):
    """IL Phase 1 설정."""

    # warmstart / demo grasp reset 비활성
    enable_warmstart_reset: bool = False
    enable_demo_grasp_reset: bool = False
    use_demo_left_target_pose: bool = False
    freeze_grasp_hand_during_episode: bool = False  # demo 손가락 replay 허용
    episode_hold_steps: int = 0                     # warmstart 안착 대기 불필요

    # 사용하지 않는 RL 부가 기능 비활성
    enable_demo_pose_reward: bool = False
    enable_spill_adr: bool = False
    enable_noise_adr: bool = False
    enable_success_adr: bool = False
    enable_trajectory_capture: bool = False
    enable_left_arm_domain_random: bool = False

    # 컵 스폰: 데모 위치에 맞춤 (source_cup 고정값 x=0.27, y=-0.10, z=0.20)
    object_spawn_x_center: float = 0.27
    object_spawn_y_center: float = -0.10
    object_spawn_z: float = 0.20
    object_spawn_xy_range: float = 0.0  # XY 랜덤화 없음
    obj_fallen_z: float = 0.10          # cup 안착 z=0.20 기준 10cm 여유

    # IL 데모 및 보상 가중치
    il_demo_paths: tuple[str, ...] = _DEFAULT_IL2_DEMOS
    weight_eef_pos: float = 10.0
    weight_eef_rot: float = 1.0


def _pad_to_max_last_frame(arrays: List[np.ndarray]) -> np.ndarray:
    """(T_i, D) 리스트 → (N, T_max, D). 빈 구간은 마지막 프레임으로 패딩."""
    T_max = max(a.shape[0] for a in arrays)
    D = arrays[0].shape[1]
    out = np.empty((len(arrays), T_max, D), dtype=np.float32)
    for i, a in enumerate(arrays):
        T = a.shape[0]
        out[i, :T] = a
        if T < T_max:
            out[i, T:] = a[-1]
    return out


class IL2PourRightEnv(PourRightEnv):
    """IL Phase 1 환경: EEF 추종 보상 + 손가락 demo replay."""

    def __init__(self, cfg: IL2PourRightEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        trajs: List[IL2Trajectory] = []
        for p in cfg.il_demo_paths:
            if Path(p).exists():
                trajs.append(load_il2_trajectory(Path(p)))
            else:
                print(f"[IL2PourRightEnv] missing demo: {p}")
        if not trajs:
            raise RuntimeError("IL2 demo 파일을 하나도 찾지 못했습니다.")

        eef_pos_np  = _pad_to_max_last_frame([t.eef_pos for t in trajs])        # (D, T_max, 3)
        eef_quat_np = _pad_to_max_last_frame([t.eef_quat_xyzw for t in trajs])  # (D, T_max, 4)
        hand_np     = _pad_to_max_last_frame([t.hand_joints for t in trajs])    # (D, T_max, 20)
        arm_np      = _pad_to_max_last_frame([t.arm_joints  for t in trajs])    # (D, T_max, 7)
        lengths = np.array([t.eef_pos.shape[0] for t in trajs], dtype=np.int64)

        super().__init__(cfg, render_mode, **kwargs)

        self._il_eef_pos  = torch.from_numpy(eef_pos_np).to(self.device)
        self._il_eef_quat = torch.from_numpy(eef_quat_np).to(self.device)
        self._il_hand     = torch.from_numpy(hand_np).to(self.device)
        self._il_arm      = torch.from_numpy(arm_np).to(self.device)
        self._il_lengths  = torch.from_numpy(lengths).to(self.device)
        self._il_T_max    = eef_pos_np.shape[1]

        self._il_demo_idx  = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._il_demo_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    # ------------------------------------------------------------------

    def _step_clamped(self) -> torch.Tensor:
        """demo step 인덱스 (demo 길이와 T_max 범위 내로 클램프)."""
        demo_lengths = self._il_lengths[self._il_demo_idx]  # (N,)
        return torch.minimum(
            self._il_demo_step.clamp(max=self._il_T_max - 1),
            demo_lengths - 1,
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # 팜/Fabrics 처리는 base class에 위임
        super()._pre_physics_step(actions)
        # 손가락 target을 demo 관절 위치로 덮어씀 (RL policy 출력 무시)
        step_idx = self._step_clamped()
        demo_hand = self._il_hand[self._il_demo_idx, step_idx]  # (N, 20)
        self.hand_joint_targets.copy_(demo_hand)
        self.fabric_q[:, NUM_ARM_DOF:] = demo_hand
        self.fabric_qd[:, NUM_ARM_DOF:].zero_()

    def _get_rewards(self) -> torch.Tensor:
        step_idx  = self._step_clamped()
        demo_pos  = self._il_eef_pos [self._il_demo_idx, step_idx]  # (N, 3)
        demo_quat = self._il_eef_quat[self._il_demo_idx, step_idx]  # (N, 4) xyzw

        palm_pos = self.palm_center_pos  # (N, 3) env-local
        palm_quat_wxyz = self.robot.data.body_quat_w[:, self.palm_body_index]  # (N, 4) wxyz→xyzw
        palm_quat_xyzw = torch.cat(
            [palm_quat_wxyz[:, 1:4], palm_quat_wxyz[:, 0:1]], dim=-1
        )

        pos_err = torch.norm(palm_pos - demo_pos, dim=-1)
        dot = (palm_quat_xyzw * demo_quat).sum(-1).abs().clamp(max=1.0 - 1e-7)
        rot_err = 2.0 * torch.acos(dot)

        r = -self.cfg.weight_eef_pos * pos_err - self.cfg.weight_eef_rot * rot_err

        self._il_demo_step += 1

        self.extras["log"] = {
            "eef_pos_err": pos_err.mean(),
            "eef_rot_err": rot_err.mean(),
        }
        return r

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        terminated = self.object_pos[:, 2] < self.cfg.obj_fallen_z
        truncated  = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return

        env_ids_t = torch.as_tensor(list(env_ids), dtype=torch.long, device=self.device)

        # 랜덤 demo 배정 및 step 초기화
        n = len(env_ids_t)
        demo_idxs = torch.randint(len(self._il_lengths), (n,), device=self.device)
        self._il_demo_idx[env_ids_t]  = demo_idxs
        self._il_demo_step[env_ids_t] = 0

        # 데모 t=0 arm/hand 자세로 로봇 초기화
        arm0  = self._il_arm [self._il_demo_idx[env_ids_t], 0]  # (n, 7)
        hand0 = self._il_hand[self._il_demo_idx[env_ids_t], 0]  # (n, 20)
        full_pos = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_vel = torch.zeros(n, self.robot.num_joints, device=self.device)
        full_pos[:, self.arm_dof_indices]      = arm0
        full_pos[:, self.hand_dof_indices]     = hand0
        full_pos[:, self.left_arm_dof_indices] = self.left_arm_target_pos[env_ids_t]
        self.robot.write_joint_state_to_sim(full_pos, full_vel, env_ids=env_ids)

        # Fabrics 상태 동기화
        self.fabric_q [env_ids_t, :NUM_ARM_DOF] = arm0
        self.fabric_q [env_ids_t, NUM_ARM_DOF:] = hand0
        self.fabric_qd[env_ids_t].zero_()
        self.fabric_qdd[env_ids_t].zero_()
        self.open_tesollo_fabric.default_config[env_ids, :NUM_ARM_DOF] = arm0
        self.pregrasp_arm_pos_buf[env_ids_t] = arm0
        self.hand_joint_targets[env_ids_t]   = hand0

        # palm target을 demo t=0 EEF로 동기화 (Fabrics 첫 타겟 일관성)
        eef_pos0  = self._il_eef_pos [self._il_demo_idx[env_ids_t], 0]  # (n, 3)
        eef_quat0 = self._il_eef_quat[self._il_demo_idx[env_ids_t], 0]  # (n, 4) xyzw
        palm_pose0 = torch.cat([eef_pos0, eef_quat0], dim=-1)            # (n, 7)
        self.palm_pose_targets[env_ids_t]      = palm_pose0
        self.pregrasp_palm_pose_buf[env_ids_t] = palm_pose0

        # beads 소환 방지 (숨김 후 spawned=True로 마킹)
        self._beads_spawned[env_ids_t] = True
        self._hide_beads(env_ids)

# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fabrics 팔 액션 — 절대 palm 6D pose → 관절 위치 목표.

왜 이 항인가 (08.22 사용자 지시)
--------------------------------
test17(관절공간)은 이송까지 성공했지만 목표에서 못 멈췄다(잔류 0.17 m/s). 정책의 raw
지령이 매 스텝 속도한계 포화였고(|Δa| 2.89, 반전 60.5%) 물리 평활은 rate limiter 가
억지로 만들고 있었다. Fabrics 는 attractor 를 2차 적분으로 감쇠 수렴시키므로
(hold L1 0.93 mm·계단 오버슈트 0% 실측) 정책이 진동 지령을 내도 팔이 흡수하고,
실기에도 같은 Fabrics 가 올라가 s2r 이 성립한다.

배선 규약 (전부 실측 근거, 어기면 조용히 틀린다)
------------------------------------------------
· 적분은 `process_actions` 에서 **한 번만** 한다. `apply_actions` 는 physics decimation
  횟수만큼 불리므로 거기서 적분하면 fabric 시간이 2배로 흐른다(agnostic 트랙 주석 실증).
· `fabrics_dt = env.step_dt / FABRIC_DECIMATION` 으로 fabric 시간 = 벽시계.
· 회전은 **quaternion 경로**만 쓴다. 기준 palm 자세 (0, π/2, 0) 이 euler_zyx 짐벌
  특이점 정확히 위라 euler 표현이 퇴화한다(08.21 회전 계단 오버슈트 19~32% 의 정체).
· world 는 좌팔 전용(`open_gripper_left_boxes_no_table`). 우팔용을 쓰면 좌팔이 자기
  대역물(`left_arm_body`)과 잡을 컵(`left_target_cup`)에서 밀려난다(08.21 실측 13~30 mm).
· fabric 의 cspace rest(default_config)는 **이 태스크의 홈**이다. 내장값(ABORTED 홈,
  j7=+1.356)을 쓰면 fabric 이 팔을 자기충돌 구간(j7>0.7 에서 l_al_5↔l_al_7 여유<9 mm)
  으로 끈다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_angle_axis, quat_mul

from . import grasp_left_preset as P

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class FabricPalmAction(ActionTerm):
    """절대 palm 6D pose 액션. Fabrics 가 관절 목표로 바꿔 PD 에 넘긴다."""

    cfg: "FabricPalmActionCfg"

    def __init__(self, cfg: "FabricPalmActionCfg", env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)

        # fabric 관련 import 는 여기서 한다 — 모듈 임포트 시점에 warp 초기화를 피한다.
        from fabrics_sim.fabrics.openarm_tesollo_pose_fabric import (
            OpenArmGripperLeftPoseFabric,
        )
        from fabrics_sim.integrator.integrators import DisplacementIntegrator
        from fabrics_sim.utils.utils import initialize_warp
        from fabrics_sim.worlds.world_mesh_model import WorldMeshesModel

        num_envs = env.num_envs
        device = env.device
        initialize_warp(str(device)[-1])

        self._world_model = WorldMeshesModel(
            batch_size=num_envs,
            max_objects_per_env=8,
            device=device,
            world_filename=P.FABRIC_WORLD_FILENAME,
        )
        self._object_ids, self._object_indicator = self._world_model.get_object_ids()

        # fabric 내부 dt. 시간 = 벽시계가 되도록 env step 을 decimation 으로 나눈다.
        self._fabric_dt = float(env.step_dt) / float(P.FABRIC_DECIMATION)
        home = [P.LEFT_ARM_HOME_JOINT_POS[f"l_aj_{i}"] for i in range(1, 8)]
        self._fabric = OpenArmGripperLeftPoseFabric(
            num_envs, device, self._fabric_dt,
            graph_capturable=False,
            robot_dir_name=P.FABRIC_ROBOT_DIR,
            robot_name=P.FABRIC_ROBOT_DIR,
            default_config_override=home,
        )
        if int(self._fabric.num_joints) != 7:
            raise ValueError(
                f"fabric cspace 가 {self._fabric.num_joints} 이다 — 팔 7 DOF 여야 한다. "
                "손 관절이 fixed 로 동결된 좌팔 그리퍼 URDF 를 확인할 것."
            )
        self._integrator = DisplacementIntegrator(self._fabric)

        # articulation 쪽 팔 관절 인덱스 (이름 기반 — 순서 가정 금지)
        self._arm_joint_ids = [
            self._asset.joint_names.index(n) for n in P.LEFT_ARM_JOINT_NAMES
        ]
        self._q_home = torch.tensor(home, device=device)

        # ★★중력 처짐 보상. Fabrics 는 순수 기구학이라 중력을 모르고, PD 는 중력 부하만큼
        #   뒤처진다(실측 32.9 mrad → TCP 40~48 mm). 지금까지는 정책이 그 선행량을 스스로
        #   학습했고 그만큼이 목표 정확도에서 빠져나갔다. 관절공간에서 상쇄한다.
        #   상한 = effort 한계 / 강성 — 그 이상 밀어도 토크가 포화하고 windup 만 쌓인다.
        limits = []
        for name in P.LEFT_ARM_JOINT_NAMES:
            idx = int(name.rsplit("_", 1)[1])
            limits.append(
                P.ARM_IK_MAX_TRACKING_ERROR["l_aj_[1-2]"] if idx <= 2
                else P.ARM_IK_MAX_TRACKING_ERROR["l_aj_[3-4]"] if idx <= 4
                else P.ARM_IK_MAX_TRACKING_ERROR["l_aj_[5-7]"]
            )
        self._droop_limit = torch.tensor(limits, device=device)
        self._droop = torch.zeros(num_envs, 7, device=device)

        # ★★palm 지령 변화율 상한. 실측: 목표 10 cm 안에서도 지령이 52 mm/step(2.6 m/s)
        #   튄다 — 팔의 능력(약 1 m/s)의 2.6 배다. Fabrics 가 흡수해도 컵에 0.307 m/s 가
        #   남는다. 보상(settle)은 세 층 아래인 컵 속도를 보므로 이걸 못 고친다.
        #   ⚠ 리셋 직후 첫 스텝은 클램프하지 않는다 — 이전 에피소드 지령에서 끌려오면
        #     시작이 오염된다(이 태스크에서 리셋 오염에 세 번 당했다).
        self._prev_cmd_pos = torch.zeros(num_envs, 3, device=device)
        self._prev_cmd_rot = torch.zeros(num_envs, 3, device=device)
        self._cmd_primed = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # fabric 상태
        self._fabric_q = self._q_home.unsqueeze(0).repeat(num_envs, 1).contiguous()
        self._fabric_qd = torch.zeros(num_envs, 7, device=device)
        self._fabric_qdd = torch.zeros(num_envs, 7, device=device)
        self._damping = P.FABRIC_DAMPING_GAIN * torch.ones(num_envs, 1, device=device)
        # 손 fabric 미사용이지만 set_features 시그니처가 PCA 인자를 요구한다.
        self._pca_zeros = torch.zeros(num_envs, 5, device=device)

        # 액션 버퍼
        self._raw_actions = torch.zeros(num_envs, self.action_dim, device=device)
        self._palm_target_xyz_q = torch.zeros(num_envs, 7, device=device)  # [xyz, xyzw]

        # 정규화 [-1,1] → PALM_BOX
        lo = torch.tensor(
            [P.PALM_BOX_X[0], P.PALM_BOX_Y[0], P.PALM_BOX_Z[0]], device=device
        )
        hi = torch.tensor(
            [P.PALM_BOX_X[1], P.PALM_BOX_Y[1], P.PALM_BOX_Z[1]], device=device
        )
        self._box_center = 0.5 * (lo + hi)
        self._box_half = 0.5 * (hi - lo)
        # 기준 파지 자세 (wxyz)
        self._ref_quat_wxyz = torch.tensor(
            P.PALM_REF_QUAT_WXYZ, device=device
        ).unsqueeze(0).repeat(num_envs, 1)

    # ------------------------------------------------------------------
    @property
    def action_dim(self) -> int:
        return 6  # 위치 3 + 축각 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._palm_target_xyz_q

    # ------------------------------------------------------------------
    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions

        # 위치: [-1,1] 클램프 후 박스 절대 좌표. a=0 = 박스 중심 (절대 규약).
        pos = self._box_center + actions[:, :3].clamp(-1.0, 1.0) * self._box_half
        # ★변화율 상한 — 팔이 낼 수 있는 속도를 넘는 지령은 포화만 만든다(test13 교훈).
        _fresh = ~self._cmd_primed | (not P.PALM_CMD_RATE_LIMIT_ENABLED)
        _d = pos - self._prev_cmd_pos
        _n = _d.norm(dim=-1, keepdim=True)
        _s = (P.PALM_CMD_RATE_LIMIT / _n.clamp(min=1e-9)).clamp(max=1.0)
        pos = torch.where(_fresh.unsqueeze(-1), pos, self._prev_cmd_pos + _d * _s)
        self._prev_cmd_pos = pos.detach()

        # 회전: 축각 벡터(노름 클램프) → 기준 자세에 세계프레임 합성.
        rotvec = actions[:, 3:6] * P.PALM_ROT_MAX_RAD
        _dr = rotvec - self._prev_cmd_rot  # (리미터 꺼져 있으면 _fresh 로 무효화된다)
        _nr = _dr.norm(dim=-1, keepdim=True)
        _sr = (P.PALM_ROT_RATE_LIMIT / _nr.clamp(min=1e-9)).clamp(max=1.0)
        rotvec = torch.where(_fresh.unsqueeze(-1), rotvec, self._prev_cmd_rot + _dr * _sr)
        self._prev_cmd_rot = rotvec.detach()
        self._cmd_primed |= True
        angle = rotvec.norm(dim=-1)
        scale = torch.where(
            angle > P.PALM_ROT_MAX_RAD, P.PALM_ROT_MAX_RAD / angle.clamp(min=1e-9),
            torch.ones_like(angle),
        )
        rotvec = rotvec * scale.unsqueeze(-1)
        angle = angle.clamp(max=P.PALM_ROT_MAX_RAD)
        axis = rotvec / angle.clamp(min=1e-9).unsqueeze(-1)
        # angle≈0 이면 axis 가 무의미하지만 quat_from_angle_axis(0, ·) = identity 라 안전.
        q_delta = quat_from_angle_axis(angle, axis)          # wxyz
        q_target = quat_mul(q_delta, self._ref_quat_wxyz)    # 세계프레임 회전 합성

        # set_features 의 quaternion 규약은 **xyzw** (내부에서 [6,3,4,5] 재배열).
        self._palm_target_xyz_q[:, :3] = pos
        self._palm_target_xyz_q[:, 3:6] = q_target[:, 1:4]
        self._palm_target_xyz_q[:, 6] = q_target[:, 0]

        # ★적분은 여기서 한 번만 (apply_actions 는 decimation 번 불린다).
        self._fabric.set_features(
            self._pca_zeros,
            self._palm_target_xyz_q,
            "quaternion",
            self._fabric_q.detach(),
            self._fabric_qd.detach(),
            self._object_ids,
            self._object_indicator,
            self._damping,
        )
        for _ in range(int(P.FABRIC_DECIMATION)):
            self._fabric_q, self._fabric_qd, self._fabric_qdd = self._integrator.step(
                self._fabric_q.detach(), self._fabric_qd.detach(),
                self._fabric_qdd.detach(), self._fabric_dt,
            )

        # ★처짐 추정은 **env step 당 한 번**만 갱신한다(apply_actions 는 decimation 번 불린다).
        #   순간 오차를 그대로 쓰면 가속 구간의 속도 지연까지 보상해 팔이 과격해지므로
        #   저역통과로 준정적 성분만 남긴다.
        if P.GRAVITY_COMP_ENABLED:
            # ★**적분**이다. 저역통과(순간 오차 추종)로 짰다가 처짐이 정확히 절반만 줄었다
            #   (실측 40.9 → 22.3 mm). 대수적으로 2d = τ_g/kp 에서 멈추기 때문이다.
            #   적분은 오차가 0 이 될 때까지 쌓여 완전 상쇄한다. clamp 가 anti-windup 이다.
            err = self._fabric_q - self._asset.data.joint_pos[:, self._arm_joint_ids]
            self._droop = (self._droop + P.GRAVITY_COMP_GAIN * err.detach()).clamp(
                -self._droop_limit, self._droop_limit
            )

    def apply_actions(self) -> None:
        target = self._fabric_q
        if P.GRAVITY_COMP_ENABLED:
            # 정상상태에서 τ = kp·(목표 − q) 이므로 관측된 처짐만큼 목표를 더 밀면
            # 그 토크가 중력을 상쇄해 q → fabric_q 가 된다.
            target = target + self._droop.clamp(-self._droop_limit, self._droop_limit)
        self._asset.set_joint_position_target(
            target, joint_ids=self._arm_joint_ids
        )
        self._asset.set_joint_velocity_target(
            torch.zeros_like(self._fabric_q), joint_ids=self._arm_joint_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._fabric_q[env_ids] = self._q_home
        self._fabric_qd[env_ids] = 0.0
        self._fabric_qdd[env_ids] = 0.0
        # ★리셋 시 초기화하지 않으면 직전 에피소드의 처짐이 남아 첫 스텝이 튄다
        #   (이 태스크에서 리셋 오염에 세 번 당했다).
        self._droop[env_ids] = 0.0
        # 다음 스텝을 "첫 지령"으로 표시 — 이전 에피소드 지령에서 끌려오지 않게 한다.
        self._cmd_primed[env_ids] = False


@configclass
class FabricPalmActionCfg(ActionTermCfg):
    """Fabrics 팔 액션 설정. 파라미터는 전부 프리셋에서 온다(단일 출처)."""

    class_type: type = FabricPalmAction
    asset_name: str = "robot"

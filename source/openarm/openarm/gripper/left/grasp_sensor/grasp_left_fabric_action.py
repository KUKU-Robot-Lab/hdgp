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
· `fabrics_dt = env.step_dt` × FABRIC_DECIMATION → fabric 시간 = **벽시계의 2배속**.
  ★★fab_test21 에 바뀌었다. 원본 kuka 가 그렇게 돈다(fabrics_dt 1/60 = 정책 스텝,
  decimation 2 → 1/30 s 적분). 구 배선(step_dt/decimation)은 1배속이라 정책 스텝당
  적분량이 원본의 60% 였고, 그만큼 attractor 수렴이 느렸다.
· 회전은 **euler_zyx 절대**(kuka `compute_absolute_action` 규약). ★★fab_test21 에
  바뀌었다 — 사용자 지시 "모든 기본 구성은 kuka setting".
  ⚠ 이 항목에는 **08.21 측정 기각 이력**이 있다: 당시 기준 palm 자세가 (0, π/2, 0) 으로
    euler_zyx 짐벌 특이점 **정확히 위**였고 "회전 계단 오버슈트 19~32%" 를 근거로
    quaternion 경로를 택했다. 지금 다시 여는 근거는 둘이다:
      ① 기준 자세가 바뀌었다 — 현 중심 ey = **−76.09°** 로 특이점에서 14° 떨어져 있다
         (정확히 위였던 때와 다르다). 다만 ±45° 박스는 여전히 −90° 를 통과한다.
      ② 그 19~32% 는 **결함 있는 플랜트 위에서 잰 값**이다(속도 피드포워드 0 ·
         fabric 60% 속도 · damping 하드끝). 셋을 원본으로 되돌린 지금은 재측정 대상이다.
  ⚠ 정정 — 짐벌 특이점은 **불연속이 아니다**. 전방 사상(euler→R)은 ey=±90° 에서도
    연속이고, 비용은 **조건수 저하**다(ez 와 ex 가 같은 회전을 만들어 액션 1D 가 국소 중복).
  ⚠ fabric 은 euler/quaternion 을 같은 회전행렬로 변환한다(set_features 두 분기가 동일한
    9D 를 채운다) — 제어 플랜트는 규약과 무관하게 동일하다. 바뀌는 건 액션 파라미터화뿐.
· world 는 좌팔 전용(`open_gripper_left_boxes_no_table`). 우팔용을 쓰면 좌팔이 자기
  대역물(`left_arm_body`)과 잡을 컵(`left_target_cup`)에서 밀려난다(08.21 실측 13~30 mm).
· fabric 의 cspace rest(default_config)는 **이 태스크의 홈**이다. 내장값(ABORTED 홈,
  j7=+1.356)을 쓰면 fabric 이 팔을 자기충돌 구간(j7>0.7 에서 l_al_5↔l_al_7 여유<9 mm)
  으로 끈다.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

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

        # ★2-스케일 전환 술어와 지령 마커가 씬을 읽는다. `SceneEntityCfg` 는 매니저가
        #   제자리 변경하는 가변 객체라 여기서 직접 `resolve` 해야 body_ids 가 채워진다
        #   (`GatedBinaryJointPositionAction` 이 쓰는 것과 같은 패턴).
        self._env = env
        self._jaw_cfg = SceneEntityCfg(
            "robot", body_names=list(P.GRIPPER_FINGER_BODIES)
        )
        self._jaw_cfg.resolve(env.scene)

        self._world_model = WorldMeshesModel(
            batch_size=num_envs,
            max_objects_per_env=8,
            device=device,
            world_filename=P.FABRIC_WORLD_FILENAME,
        )
        self._object_ids, self._object_indicator = self._world_model.get_object_ids()

        # fabric 내부 dt. 시간 = 벽시계가 되도록 env step 을 decimation 으로 나눈다.
        # ★★fab_test21 원본 정합: 원본 kuka 는 `fabrics_dt = 정책 스텝`(1/60)을 쓰고
        #   `fabric_decimation` 번 적분한다 → fabric 시간 = 벽시계의 **2배속**.
        #   우리는 step_dt/decimation 을 써서 1배속이었고, 정책 스텝당 적분량이 원본의
        #   60%(0.02 vs 0.0333)였다. 전문은 preset FABRIC_DECIMATION 주석.
        self._fabric_dt = float(env.step_dt)
        home = [P.LEFT_ARM_HOME_JOINT_POS[f"l_aj_{i}"] for i in range(1, 8)]
        self._fabric = OpenArmGripperLeftPoseFabric(
            num_envs, device, self._fabric_dt,
            graph_capturable=False,
            robot_dir_name=P.FABRIC_ROBOT_DIR,
            robot_name=P.FABRIC_ROBOT_DIR,
            default_config_override=home,
            # ★08.29 A/B — 기본은 원본 파일이라 이 인자만으로는 거동이 바뀌지 않는다.
            #   근거·사용법은 preset `FABRIC_PARAMS_FILENAME` 주석 참조.
            fabric_params_filename=P.FABRIC_PARAMS_FILENAME,
        )
        print(f"[fabric] params = {P.FABRIC_PARAMS_FILENAME}")
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
        self._prev_cmd_euler = torch.zeros(num_envs, 3, device=device)   # fab_test48 회전 리미터
        self._cmd_primed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._cmd_step_norm = torch.zeros(num_envs, device=device)
        # 속도 피드포워드 배율 — ADR 이 낮춘다(1.0 → 0.0). 스칼라 하나라 텐서가 아니다.
        self._vel_ff_scale = float(P.FABRIC_VEL_FF_SCALE)

        # fabric 상태
        self._fabric_q = self._q_home.unsqueeze(0).repeat(num_envs, 1).contiguous()
        self._fabric_qd = torch.zeros(num_envs, 7, device=device)
        self._fabric_qdd = torch.zeros(num_envs, 7, device=device)
        self._damping = P.FABRIC_DAMPING_GAIN * torch.ones(num_envs, 1, device=device)
        # 손 fabric 미사용이지만 set_features 시그니처가 PCA 인자를 요구한다.
        self._pca_zeros = torch.zeros(num_envs, 5, device=device)

        # 액션 버퍼
        self._raw_actions = torch.zeros(num_envs, self.action_dim, device=device)
        self._palm_pose_target = torch.zeros(num_envs, 6, device=device)  # [xyz, ez,ey,ex]
        # ★A1: 이송 국면 래치. **단조**다 — 한 번 서면 에피소드가 끝날 때까지 유지한다.
        #   깜빡이는 게이트(`grasp_ok` 등)로 걸면 리미터가 깜빡여 지령이 더 떨린다.
        self._fine_latched = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # 정규화 [-1,1] → PALM_BOX
        lo = torch.tensor(
            [P.PALM_BOX_X[0], P.PALM_BOX_Y[0], P.PALM_BOX_Z[0]], device=device
        )
        hi = torch.tensor(
            [P.PALM_BOX_X[1], P.PALM_BOX_Y[1], P.PALM_BOX_Z[1]], device=device
        )
        self._box_center = 0.5 * (lo + hi)
        self._box_half = 0.5 * (hi - lo)
        # euler_zyx 절대 규약의 박스(중심 ± MAX_POSE_ANGLE). 축별 독립이다.
        self._euler_center = torch.tensor(
            P.PALM_EULER_ZYX_CENTER, device=device, dtype=torch.float32
        ).unsqueeze(0).repeat(num_envs, 1)
        _pma = (P.PALM_MAX_POSE_ANGLE if self.cfg.palm_max_pose_angle is None
                else float(self.cfg.palm_max_pose_angle))
        self._euler_half = torch.full((num_envs, 3), _pma, device=device)
        print(f"[action] palm_max_pose_angle = {math.degrees(_pma):.1f}deg")
        # ★H1 재앵커 상태 — 단조 래치. 홈 앵커를 보존해 리셋에서 되돌린다.
        self._anchor_home = self._fabric.default_config.clone()
        self._anchor_latched = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._box_lo, self._box_hi = lo, hi

        # ★★fab_test46: 2-스케일(FINE 래치) **제거** (사용자 결정).
        #   존재 이유였던 "지터 = σ×박스반폭"은 리미터(PALM_CMD_RATE_LIMIT 0.02)가
        #   박스 크기와 무관하게 20 mm/step 으로 캡하면서 소멸했다. 남은 것은 해악뿐:
        #   앵커 = 진입 순간의 지령인데, 리미터 하에서 지령은 턱보다 ~50 mm 앞서 걸으므로
        #   앵커가 컵 −50 mm 에 잠기고, 최전방 = 앵커+57.5 ≈ 컵+7.5 mm < 필요 컵+43 mm
        #   (턱오프셋 33 + fabric 처짐 10). **삽입 지령이 구조적으로 불가능**했다.
        #   t45 실측: 지령 x 최대 347(필요 423), 턱-컵이 FINE 진입선(100) 바로 위
        #   118~121 mm 에 1218 epoch... 정정: 608 epoch 정체, contact 1e-4.

    # ------------------------------------------------------------------
    @property
    def action_dim(self) -> int:
        return 6  # 위치 3 + 축각 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._palm_pose_target

    @property
    def vel_ff_scale(self) -> float:
        """속도 피드포워드 배율 (0~1). ADR 커리큘럼이 낮춘다."""
        return self._vel_ff_scale

    @vel_ff_scale.setter
    def vel_ff_scale(self, value: float) -> None:
        self._vel_ff_scale = float(value)

    @property
    def cmd_step_norm(self) -> torch.Tensor:
        """이번 스텝에 palm 지령이 실제로 이동한 거리 (m). (num_envs,)

        정책 raw 액션이 아니라 **리미터 통과 후** 값이다 — fabric 에 들어간 것이 이것이고,
        층 분해 실측상 팔의 움직임을 결정하는 것도 이것이다(raw 액션의 진동은 ③에서 0 이 된다).
        """
        return self._cmd_step_norm

    # ------------------------------------------------------------------
    def _rate_limit_per_env(self) -> torch.Tensor:
        """지령 변화율 상한 (N,1). A1 이 꺼져 있으면 전 env 공통 상수다.

        ★래치는 여기서 **갱신**된다(process_actions 초입에 해당). 컵 높이는 단조 조건이라
          한 번 넘으면 `reset` 전까지 유지된다 — 리미터가 깜빡이지 않게 하는 것이 요점이다.
        """
        base = torch.full((self._palm_pose_target.shape[0], 1),
                          P.PALM_CMD_RATE_LIMIT, device=self._palm_pose_target.device)
        fine = self.cfg.fine_cmd_rate_limit
        latch_z = self.cfg.fine_latch_cup_z
        if fine is None or latch_z is None:
            return base                      # ← v1 경로. 여기서 끝난다.
        obj = self._env.scene[self.cfg.fine_latch_object_name]
        self._fine_latched |= obj.data.root_pos_w[:, 2] > latch_z
        return torch.where(self._fine_latched.unsqueeze(-1),
                           torch.full_like(base, float(fine)), base)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions

        # 위치: [-1,1] 클램프 후 박스 절대 좌표. a=0 = 박스 중심 (절대 규약).
        # ★★fab_test21: **rate limiter 제거 — 원본에 없다.** 원본 `compute_absolute_action`
        #   은 스케일 + clamp 만 하고, 변화율 상한은 fabric 이 정한다. 우리가 붙였던
        #   리미터는 자초한 굼뜸(fabric 60% 속도 · damping 하드끝 · vel_ff 0)의 증상
        #   억제기였다. 셋을 원본으로 되돌렸으므로 함께 뗀다.
        # ★fab_test46: 2-스케일 분기 제거 — 단일 박스 + 리미터. 폐기 근거는 __init__ 주석.
        pos = self._box_center + actions[:, :3].clamp(-1.0, 1.0) * self._box_half
        # ★FINE 지령도 COARSE 박스 안에 있어야 한다 — 앵커가 박스 가장자리면 밖으로 나간다.
        pos = torch.max(torch.min(pos, self._box_hi), self._box_lo)
        # ★★fab_test25: 지령 **변화율 상한**. 절대 규약은 그대로다 — 목표는 여전히 박스
        #   안의 절대 좌표이고, 한 스텝에 갈 수 있는 거리만 묶는다. 근거는 preset 주석.
        #   ⚠ 리셋 직후(_cmd_primed False)에는 걸지 않는다. 홈에서 첫 지령까지의 거리는
        #     "변화"가 아니라 초기화라, 여기에 상한을 걸면 리셋마다 팔이 몇 스텝 끌려간다.
        if P.PALM_CMD_RATE_LIMIT_ENABLED:
            step = pos - self._prev_cmd_pos
            dist = step.norm(dim=-1, keepdim=True)
            # ★A1: 기본은 상수. cfg 가 켜져 있을 때만 **env 별**로 갈린다.
            lim = self._rate_limit_per_env()
            scale = (lim / dist.clamp(min=1e-9)).clamp(max=1.0)
            pos = torch.where(self._cmd_primed.unsqueeze(-1),
                              self._prev_cmd_pos + step * scale, pos)
        # 지령 이동량은 계속 기록한다 — 보상 항(`palm_command_rate_at_goal`)이 읽는다.
        #   리셋 직후(_fresh)는 0. 텔레포트 차분을 벌하면 첫 스텝마다 가짜 벌금이 나간다.
        _fresh = ~self._cmd_primed
        self._cmd_step_norm = torch.where(
            _fresh, torch.zeros(pos.shape[0], device=pos.device),
            (pos - self._prev_cmd_pos).norm(dim=-1),
        ).detach()
        self._prev_cmd_pos = pos.detach()
        # ★★fab_test29 버그 수정. `_cmd_primed` 를 **어디서도 True 로 만들지 않고 있었다** —
        #   초기화 False, 리셋 False, 끝. 결과가 조용한 0 두 개였다:
        #     ① 리미터가 한 번도 안 걸렸다(`_cmd_primed` 게이트) → t27 이 리미터 없는 t24 와
        #        똑같이 돌았다(drop 0.9127 vs 0.914). "리미터를 넣었는데 안 낫는다"는
        #        판정을 할 뻔했다 — 넣은 적이 없었다.
        #     ② `cmd_step_norm` 이 항상 0 → 그걸 곱해 쓰는 `palm_cmd_rate` 보상도
        #        t20 이후 **줄곧 죽어 있었다**(TB 에서 정확히 0.0000).
        #   ⚠ 이 플래그는 "리셋 후 첫 지령을 이미 냈는가" 다. 첫 지령 **뒤에** 켠다.
        # ★fab_test48: True 세팅을 **회전 리미터 뒤로** 미뤘다 — 같은 스텝에서 회전
        #   게이트도 "리셋 직후 첫 지령"을 리미터 없이 통과시켜야 하기 때문이다.

        # ★★fab_test21: 회전 = **euler_zyx 절대**(kuka `compute_absolute_action` 규약).
        #   a ∈ [-1,1] → 중심 ± MAX_POSE_ANGLE, 축별 독립. 구 규약(축각 3D 를 기준 quat 에
        #   합성)은 폐기. 전문·특이점 주석은 preset PALM_EULER_ZYX_CENTER 참조.
        # ★설계안에 없던 항 — 회전도 같은 크기의 지렛대다. 팜→턱 140 mm 라 ±45° 는
        #   σ0.35 에서 턱을 38 mm 흔들어 FINE 위치 이득(19·12·14 mm)을 그대로 삼킨다.
        #   FINE 에서는 **회전 중심을 그 시점 지령으로 래치**하고 박스를 ±11.3° 로 좁힌다.
        euler = self._euler_center + actions[:, 3:6].clamp(-1.0, 1.0) * self._euler_half
        # ★★fab_test48: **회전 리미터** — 위치 리미터(0.02 m/step)의 회전판.
        #   위치만 묶고 회전을 안 묶은 것이 t46/t47 의 남은 지터 채널이었다:
        #   palm→턱 레버 140 mm 라 σ0.47 에서도 회전 지터가 턱을 ±52 mm/step 쓸고
        #   다녔다(위치 20 mm 의 2.6배). s/l/h 는 gripper_base 프레임 측정이라 회전이
        #   흔들리면 벌점 지형 자체가 출렁여 하강 신호가 씻겼다. 27판 중 유일하게
        #   h −44·λ 0.8 까지 내려간 t45 는 FINE 래치가 회전을 ±11.3° 로 조여준 판이었다.
        #   0.05 rad/step(≈2.9°) → 턱 스윙 기여 ≤ 7 mm/step (위치 20 mm 보다 작게).
        #   도달 범위는 그대로다(±45° 전체, 걸어서 간다 — 위치 리미터와 동일 논리).
        if P.PALM_CMD_RATE_LIMIT_ENABLED:
            d_euler = (euler - self._prev_cmd_euler).clamp(
                -P.PALM_EULER_RATE_LIMIT, P.PALM_EULER_RATE_LIMIT)
            euler = torch.where(self._cmd_primed.unsqueeze(-1),
                                self._prev_cmd_euler + d_euler, euler)
        self._prev_cmd_euler = euler.detach()
        self._cmd_primed[:] = True

        self._palm_pose_target[:, :3] = pos
        self._palm_pose_target[:, 3:6] = euler

        # ★★H1 — 컵이 리프트 임계를 넘은 **첫 스텝**에 그 env 의 현재 팔 자세로
        #   cspace rest 를 재앵커한다. 단조 래치라 에피소드 안에서 한 번만 일어난다
        #   (깜빡이면 널스페이스 목표가 흔들려 지령이 더 떤다).
        if self.cfg.anchor_relatch_cup_z is not None:
            _obj = self._env.scene[self.cfg.fine_latch_object_name]
            _new = (~self._anchor_latched) & (
                _obj.data.root_pos_w[:, 2] > float(self.cfg.anchor_relatch_cup_z))
            if bool(_new.any()):
                _q_now = self._asset.data.joint_pos[:, self._arm_joint_ids]
                self._fabric.default_config[_new, :7] = _q_now[_new]
                self._anchor_latched |= _new

        # ★적분은 여기서 한 번만 (apply_actions 는 decimation 번 불린다).
        self._fabric.set_features(
            self._pca_zeros,
            self._palm_pose_target,
            "euler_zyx",
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
        # ★★fab_test21: **속도 피드포워드**. `_fabric_qd` 가 바로 위에서 적분돼 있는데
        #   여기서 0 을 넣고 있었다 — 저장소 전역이 그랬고, DEXTRAH 원본에만 있다:
        #       self.dof_vel_targets[:, idx] = torch.clone(self.fabric_qd)
        #       self.robot.set_joint_velocity_target(vel_scale * self.dof_vel_targets[...])
        #
        #   0 을 넣으면 PD 의 감쇠항이 **움직임 자체를 반대로 밀어낸다**. 정상상태에서
        #       vel_ff 없음: kp·err = kd·v + τ_마찰  →  err ≈ (kd/kp)·v = 0.2·v [rad]
        #       vel_ff 있음: kp·err = τ_마찰만       →  err ≈ 0.0012 rad = 0.07°
        #   실측이 이 식에 들어맞았다 — 정책 구동 중 관절속도 0.855 rad/s 일 때
        #   예측 드래그 171 mrad vs 실측 |fabric_q − q| **140 mrad**.
        #   즉 "팔이 지령을 못 따라간다"의 원인은 leash 도 리미터도 아니고 이 배선이었다.
        #
        # ⚠ vel_scale 은 ADR 이 낮춘다(DEXTRAH 와 동일하게 1.0 → 0.0). 원본에서도 이건
        #   `pd_targets/velocity_target_factor` **ADR 파라미터**이고 범위가 (1.0, 0.0) 이다
        #   — 우리는 그 ADR 의 **가장 어려운 끝값을 시작 조건으로 하드코딩**하고 있었다.
        #   이 트랙에서 반복된 실패 유형 그대로다(억제/난이도를 과제 성립 전에 최대로).
        self._asset.set_joint_velocity_target(
            self._vel_ff_scale * self._fabric_qd, joint_ids=self._arm_joint_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._cmd_step_norm[env_ids] = 0.0
        self._fabric_q[env_ids] = self._q_home
        self._fabric_qd[env_ids] = 0.0
        self._fabric_qdd[env_ids] = 0.0
        # ★리셋 시 초기화하지 않으면 직전 에피소드의 처짐이 남아 첫 스텝이 튄다
        #   (이 태스크에서 리셋 오염에 세 번 당했다).
        self._droop[env_ids] = 0.0
        # 다음 스텝을 "첫 지령"으로 표시 — 이전 에피소드 지령에서 끌려오지 않게 한다.
        self._cmd_primed[env_ids] = False
        # ★★fab_test68: 지령 상태가 obs 에 들어가면서 이것도 리셋 오염원이 됐다.
        #   `_palm_pose_target` 은 다음 `process_actions` 전까지 **직전 에피소드 값**이라,
        #   리셋 직후 첫 관측이 남의 파지 자세를 가리킨다. a=0 지령(박스 중심)으로 둔다.
        # ★★H1 — 재앵커를 홈으로 되돌린다. 안 되돌리면 직전 에피소드의 파지 자세가
        #   다음 에피소드의 널스페이스 목표가 되어 **리셋 오염**이 된다
        #   (이 트랙에서 리셋 오염에 이미 세 번 당했다).
        if self.cfg.anchor_relatch_cup_z is not None:
            self._anchor_latched[env_ids] = False
            self._fabric.default_config[env_ids] = self._anchor_home[env_ids]
        self._palm_pose_target[env_ids, :3] = self._box_center
        self._palm_pose_target[env_ids, 3:6] = self._euler_center[env_ids]
        # ★A1 래치 해제 — 안 하면 다음 에피소드가 접근 국면부터 FINE 으로 시작해 느려진다.
        self._fine_latched[env_ids] = False
        # ★2-스케일 상태도 반드시 지운다. 이 트랙은 리셋 오염에 네 번 당했다 —
        #   앵커가 남으면 새 에피소드의 첫 지령이 지난 에피소드의 파지 자세 주변으로 묶인다.

    # ------------------------------------------------------------------
    # 지령 마커 (GUI 학습용) — 사용자 지시: "tcp 마커말고 액션 지령 마커를 6D 로"
    # ★TCP 마커는 `object_pose` 커맨드의 `body_pose` 다. 그쪽 `debug_vis` 를 끄고
    #   여기서 **정책이 실제로 내는 지령**을 그린다. 큰 프레임 = 팜 지령(6D),
    #   작은 프레임 = 이송 목표. 지령과 실제가 어긋나는 순간을 눈으로 잡기 위한 것이다
    #   (이 트랙은 "지령과 실제를 나란히 본 적이 없어" 진단이 늘 사후 프로브가 됐다).
    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "_cmd_marker"):
                from isaaclab.markers import VisualizationMarkers
                from isaaclab.markers.config import (
                    FRAME_MARKER_CFG,
                    POSITION_GOAL_MARKER_CFG,
                )

                cfg = FRAME_MARKER_CFG.copy()
                cfg.prim_path = "/Visuals/PalmActionCommand"
                self._cmd_marker = VisualizationMarkers(cfg)
                # ★★fab_test62(사용자 지적): "goal position 이 정확히 어디인지 안 보인다".
                #   프레임 마커(축 3개)는 작게 그리면 배경에 묻힌다. 목표는 **방향이 없는
                #   점**이므로 구로 그리는 것이 맞다. 이 프리셋은 멀면 빨강 / 가까우면
                #   초록이라 **이송이 됐는지가 색으로 바로 읽힌다**.
                gcfg = POSITION_GOAL_MARKER_CFG.copy()
                gcfg.prim_path = "/Visuals/TransferGoal"
                self._goal_marker = VisualizationMarkers(gcfg)
            self._cmd_marker.set_visibility(True)
            self._goal_marker.set_visibility(True)
        elif hasattr(self, "_cmd_marker"):
            self._cmd_marker.set_visibility(False)
            self._goal_marker.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        # ★★fab_test61(사용자 지시): **env 0 에서만** 그린다. 1024 env 를 전부 그리면
        #   화면이 프레임으로 뒤덮여 정작 보려던 "지령과 실제의 어긋남"이 안 보이고,
        #   마커 렌더 자체가 GUI 학습의 fps 를 갉아먹는다.
        _E = slice(0, 1)
        # 팜 지령은 env 로컬(로봇 base)이라 world 로 올린다.
        pos_w = self._palm_pose_target[_E, :3] + self._env.scene.env_origins[_E]
        ez = self._palm_pose_target[_E, 3]
        ey = self._palm_pose_target[_E, 4]
        ex = self._palm_pose_target[_E, 5]
        # ★euler_zyx (ez,ey,ex) = Rz·Ry·Rx = XYZ 규약의 (roll=ex, pitch=ey, yaw=ez).
        quat_w = quat_from_euler_xyz(ex, ey, ez)
        big = torch.full_like(pos_w, 0.12)
        # 팜 지령 프레임(큰 축 3개)은 이 마커가 전담한다.
        self._cmd_marker.visualize(
            translations=pos_w, orientations=quat_w, scales=big
        )
        # 이송 목표는 **구**로 따로 그린다 — 컵이 합격 반경 안에 들어오면 초록으로 바뀐다.
        try:
            goal = self._env.command_manager.get_command("object_pose")[_E, :3]
            goal_w = goal + self._env.scene.env_origins[_E]
            obj = self._env.scene["object"]
            d = (goal_w - obj.data.root_pos_w[_E]).norm(dim=-1)
            # 0 = target_far(빨강) · 1 = target_near(초록). 임계는 이 태스크의 합격 기준.
            idx = (d < P.TRANSFER_GOAL_HIT_M).long()
            self._goal_marker.visualize(
                translations=goal_w,
                marker_indices=idx,
                scales=torch.full_like(goal_w, P.TRANSFER_GOAL_MARKER_SCALE),
            )
        except (KeyError, AttributeError):
            pass


@configclass
class FabricPalmActionCfg(ActionTermCfg):
    """Fabrics 팔 액션 설정. 파라미터는 전부 프리셋에서 온다(단일 출처)."""

    class_type: type = FabricPalmAction
    asset_name: str = "robot"

    # ★★08.29 A1 — **이송 국면 전용** 지령 변화율 상한 (grasp_sensor_v2 실험용).
    #   `None` 이면 이 기능 자체가 꺼진다 ⇒ **v1 거동은 한 글자도 바뀌지 않는다.**
    #   근거(run 0 결정론 프로브, σ 배제): 기본 리미터 0.02 m/step = **1000 mm/s** 인데
    #   액션이 절대 규약(`box_half` x190·y165·z220 mm)이라 정책의 평범한 스텝 차분
    #   `|Δa| ≈ 0.1` 이 곧 지령 19 mm/step 을 만든다 → **리미터 포화 71~76%**,
    #   지령 방향 반전 45.6%, 직진 효율 0.248. 정책이 "천천히"를 표현할 해상도가 없다.
    #   ⇒ 접근·파지는 그대로 두고 **리프트 성립 후에만** 상한을 조인다.
    # ★★08.29 라운드 6 — 회전 자유도. `None` 이면 preset 값(±20°) 그대로 = v1 불변.
    #   실측 근거(baseline best, 결정론 1024 env, 리프트 전/후 분리):
    #     euler 박스(±20°) 포화율 0.954 / 0.982 — 지령이 98% 스텝에서 클램프에 붙어 있다.
    #     축별 부호: ez 는 전 국면 −1(중심 오류) · ey 는 접근 +0.594 / 이송 −0.885
    #     (부호가 뒤집힌다 = **최소 40° 스팬 필요**) · ex 는 리프트 후 −0.886.
    #   ⇒ 중심 오류와 범위 부족이 동시에 있고, 범위를 넓히면 둘 다 흡수된다.
    palm_max_pose_angle: float | None = None
    # ★★H1 — env 별 cspace 재앵커. `None` 이면 재앵커 없음 = v1 불변.
    #   앵커 FK palm = (0.171, −0.101, 0.185) 로 **액션 박스 y[0.10,0.43] 밖**(−201 mm),
    #   컵 스폰에서 374 mm · 목표에서 486 mm 이고 **모든 env 공용 하나**다.
    #   컵을 든 순간의 관절 자세로 per-env 재앵커한다(자매 `pour_fabric` 선례:
    #   "cspace rest 를 warm 자세로 재앵커 — nullspace α 채널의 대체물").
    anchor_relatch_cup_z: float | None = None
    fine_cmd_rate_limit: float | None = None      # m/step (예: 0.005 = 250 mm/s)
    fine_latch_cup_z: float | None = None         # m, world. 컵이 이 높이를 넘으면 래치
    fine_latch_object_name: str = "object"

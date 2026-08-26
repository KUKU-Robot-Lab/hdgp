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

"""관측 항 — **fabric 내부 상태**와 critic 전용 특권 관측.

★★fab_test22 원본 정합. DEXTRAH kuka 는 정책과 critic **양쪽**에 fabric 의 내부 상태
  `fabric_q · fabric_qd · fabric_qdd` 를 그대로 넣는다:

      # dextrah_kuka_allegro_env.py compute_policy_observations
      self.fabric_q_for_obs, self.fabric_qd_for_obs, self.fabric_qdd_for_obs

  우리는 이 셋이 **전부 빠져 있었다**. 그 결과가 무엇이었는지는 08.25 실측이 말한다 —
  지령과 실제 TCP 가 이송 중 90 mm 어긋나 있는데(추종 지연), 정책은 자기가 낸 지령이
  실현됐는지를 **관측할 수단이 아예 없었다**. 팔이 어디 있는지는 `joint_pos` 로 보지만
  "fabric 이 지금 어디로 가는 중인가"(qd·qdd)는 어디에도 없다.

  ⚠ fabric 은 오픈루프 plant 다(실측을 되먹이지 않는다 — E1 사고 2건). 그래서 fabric 상태는
    실측 관절과 **다른 정보**이고, 둘을 같이 줘야 정책이 그 차이(=추종 오차)를 볼 수 있다.

critic 전용 특권 관측도 원본을 따른다(`compute_critic_observations`):
  손끝 접촉력 · 실측 관절 토크 · 물체 선/각속도 · 노이즈 없는 실측.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import math

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp

from . import grasp_left_obs_noise as obs_noise

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _arm_action(env: "ManagerBasedRLEnv"):
    return env.action_manager.get_term("arm_action")


def fabric_q(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """fabric 관절 위치 목표 (num_envs, 7). 실측이 아니라 **참조 궤적**이다."""
    return _arm_action(env)._fabric_q


def fabric_qd(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """fabric 관절 속도 (num_envs, 7). "지금 어디로 가는 중인가"."""
    return _arm_action(env)._fabric_qd


def fabric_qdd(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """fabric 관절 가속도 (num_envs, 7)."""
    return _arm_action(env)._fabric_qdd


def palm_pose_target(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """정책이 낸 palm 목표 [xyz, ez, ey, ex] (num_envs, 6).

    `last_action` 과 중복처럼 보이지만 아니다 — last_action 은 [-1,1] 원값이고
    이것은 박스 스케일·clamp 를 거친 **실제 지령**이다. 박스 경계에 걸리면 둘이 갈린다.
    """
    return _arm_action(env)._palm_pose_target


def finger_contact_forces(
    env: "ManagerBasedRLEnv", sensor_names: tuple[str, ...]
) -> torch.Tensor:
    """손가락 접촉력 크기 (num_envs, F) — **critic 전용**(원본 `hand_forces`).

    ⚠ 센서가 없으면 조용히 0 을 주지 않고 터뜨린다 — 이 트랙은 "지표가 정확히 0.0000"
      인 죽은 센서에 이미 당했다(reward-clamp-kills-gradient 메모).
    """
    out = []
    for name in sensor_names:
        sensor = env.scene.sensors[name]
        fm = sensor.data.force_matrix_w
        out.append(fm.view(env.num_envs, -1, 3).sum(dim=1).norm(dim=-1, keepdim=True))
    return torch.cat(out, dim=-1)


def object_lin_ang_vel(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """물체 선속도+각속도 (num_envs, 6) — **critic 전용**(원본은 policy 에서 제외했다:
    "took this out because it's fairly privileged")."""
    obj: RigidObject = env.scene[asset_cfg.name]
    return torch.cat((obj.data.root_lin_vel_w, obj.data.root_ang_vel_w), dim=-1)


def arm_applied_torque(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """실측 관절 토크 (num_envs, J) — **critic 전용**(원본 `measured_joint_torque`)."""
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.applied_torque[:, asset_cfg.joint_ids]


def object_out_of_workspace(
    env: "ManagerBasedRLEnv",
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """물체가 작업공간 x·y 박스를 벗어났는가 (종료 항).

    ★★fab_test22 원본 정합. kuka `_get_dones` 는 물체가 스폰 박스 x·y 를 벗어나거나
      z 가 낮으면 즉시 종료한다. 우리는 **낙하(z)만** 봤고 옆으로 굴러 나가는 경우를
      놓쳐, 그 뒤 스텝이 전부 낭비 표본이 됐다(정책은 회수할 수 없다 — 팔 워크스페이스
      밖이다). z 종료는 부모의 `object_dropping` 이 이미 담당한다.
    """
    obj: RigidObject = env.scene[asset_cfg.name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    return (
        (pos[:, 0] < x_range[0]) | (pos[:, 0] > x_range[1])
        | (pos[:, 1] < y_range[0]) | (pos[:, 1] > y_range[1])
    )


# ─────────────────────────────────────────────────────────────────────────
# 손 직교 상태 · 물체 회전 — 원본 정합으로 추가 (fab_test23)
#
# ★★원본 policy obs 에는 `hand_pos_noisy`(팜+손끝 위치)와 `hand_vel_noisy`(6D),
#   그리고 `object_rot_noisy` 가 들어 있는데 우리에겐 **셋 다 없었다**.
#   정책은 관절각만 보고 턱이 어디 있는지를 스스로 FK 해야 했다. 이 태스크의 보상은
#   전부 턱–컵 기하(lateral/along/파지대역)로 정의돼 있는데, 그 기하를 만드는 입력이
#   관측에 없었다는 뜻이다.
#
# ⚠ 원본과 한 군데 다르다. 원본은 `hand_points_taskmap(robot_dof_pos_noisy)` 로
#   **노이즈 낀 관절각을 FK 해서** 손 위치를 만든다(실기에서 손 위치는 엔코더로부터
#   추정되므로 그게 옳다). 우리는 배치 FK taskmap 이 없어 실측 body 위치에 직접
#   노이즈를 건다. 등가는 아니다 — 원본에서는 관절 노이즈와 손 위치 노이즈가 **상관**
#   되는데 우리는 독립이다. 폭은 관절 노이즈를 지렛대 길이로 환산해 맞춘다
#   (`HAND_POINT_NOISE_LEVER`). 이 차이를 지우려면 FK taskmap 이 필요하다.
# ─────────────────────────────────────────────────────────────────────────


def hand_body_pos(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    noisy: bool = False,
    lever: float = 0.0,
) -> torch.Tensor:
    """팜·턱 위치 (num_envs, B*3), env 로컬 좌표 (원본 `hand_pos`)."""
    robot: Articulation = env.scene[asset_cfg.name]
    pos = robot.data.body_pos_w[:, asset_cfg.body_ids] - env.scene.env_origins.unsqueeze(1)
    pos = pos.reshape(env.num_envs, -1)
    if not noisy:
        return pos
    return _lever_corrupt(env, obs_noise.JOINT_POS, pos, lever)


def hand_body_vel(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    noisy: bool = False,
    lever: float = 0.0,
) -> torch.Tensor:
    """팜·턱 6D 속도 (num_envs, B*6) (원본 `hand_vel`)."""
    robot: Articulation = env.scene[asset_cfg.name]
    vel = robot.data.body_vel_w[:, asset_cfg.body_ids].reshape(env.num_envs, -1)
    if not noisy:
        return vel
    return _lever_corrupt(env, obs_noise.JOINT_VEL, vel, lever)


def _lever_corrupt(env, channel: str, x: torch.Tensor, lever: float) -> torch.Tensor:
    """관절 노이즈 채널을 지렛대 길이로 환산해 직교 좌표에 건다."""
    st = obs_noise.state(env)
    width = st.noise_width[channel] * lever
    bias = st.bias[channel] * lever
    return x + width * 2.0 * (torch.rand_like(x) - 0.5) + bias


def object_rotation(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    noisy: bool = False,
) -> torch.Tensor:
    """물체 자세 쿼터니언 (num_envs, 4) — 원본 `object_rot` / `object_rot_noisy`.

    ⚠ 원본은 노이즈를 쿼터니언 성분에 그대로 더하고 정규화하지 않는다. 그대로 따른다
      (정규화하면 노이즈 크기가 회전 오차로 옮겨가 원본과 분포가 달라진다).
    """
    obj: RigidObject = env.scene[asset_cfg.name]
    quat = obj.data.root_quat_w
    return obs_noise.corrupt(env, obs_noise.OBJ_ROT, quat) if noisy else quat


def joint_pos_noisy(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """관절 위치(기본자세 상대) + per-step 노이즈 + per-episode bias."""
    robot: Articulation = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    return obs_noise.corrupt(env, obs_noise.JOINT_POS, q)


def joint_vel_noisy(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """관절 속도(기본자세 상대) + per-step 노이즈 + per-episode bias."""
    robot: Articulation = env.scene[asset_cfg.name]
    qd = robot.data.joint_vel[:, asset_cfg.joint_ids] - robot.data.default_joint_vel[:, asset_cfg.joint_ids]
    return obs_noise.corrupt(env, obs_noise.JOINT_VEL, qd)


def object_position_noisy(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """로봇 루트 프레임 물체 위치 + 노이즈 + bias (원본 `object_pos_noisy`)."""
    pos = lift_mdp.object_position_in_robot_root_frame(env, robot_cfg, object_cfg)
    return obs_noise.corrupt(env, obs_noise.OBJ_POS, pos)


def object_tipped(
    env: "ManagerBasedRLEnv",
    max_tilt_deg: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """컵이 규정 각도 이상 기울었는가 (종료 항).

    ★★fab_test39 신설. **이 트랙만 전도 종료가 없었다.** 자매 트랙 넷은 전부 명시적
      각도 종료를 가진다:
          tesollo/right/grasp_v1   60° (`cup_tipping_max_deg`, 스크립트 램프 중 면제)
          agnostic/tasks/grasp_sensor 60° (`tilt_reset_deg`)
          gripper/left/grasp_v1    30° (`mdp.cup_tipped`)
          gripper/right/grasp_v1   30°
      우리는 `OBJECT_DROP_HEIGHT = 0.27` 이 높이로 전도를 대신하게 뒀는데, 원점 z 는
      기울기에 둔감해 **거의 누워야** 걸린다(60° 에서 원점 0.299 > 임계 0.27 통과).

    그 결과가 fab_test38 이다 — 결정론 실측에서 컵을 **45 mm 비스듬히 밀고 다니는 동안**
    에피소드가 끝까지 살아 있었고, `drop` 은 0.039 에 머물렀다. 자매 트랙이었으면
    끊겼을 표본이 전부 학습에 섞였다.

    ⚠ 임계는 **60°** 로 잡는다. 30° 는 파지 중 필연적인 흔들림을 끊을 수 있고, 이 트랙은
      이미 "자세 조건을 AND 게이트로 걸어 학습을 죽인" 이력이 있다(test6/test7:
      lifting 6.14 → 0.0000, 에피소드 130 → 13, 총보상 +34.9 → −0.46). 성공 파지의
      실측 컵 기울기는 4.1° 라 60° 는 충분히 넉넉하다.
    ⚠ 종료는 반드시 `terminated`(= `time_out=False`)여야 한다. `truncated` 로 내보내면
      `value_bootstrap` 이 `γ·V(s)` 를 얹어 **컵을 쓰러뜨릴 때마다 보너스**가 된다
      (agnostic 트랙 실측: 실보상 3307→103 붕괴인데 shaped 72.8→79.4 상승).
    """
    obj: RigidObject = env.scene[asset_cfg.name]
    w, x, y, _z = obj.data.root_quat_w.unbind(-1)
    cos_tilt = 1.0 - 2.0 * (x * x + y * y)          # 컵 로컬 +z 의 world z 성분
    return cos_tilt < math.cos(math.radians(max_tilt_deg))


def palm_action_scale(
    env: "ManagerBasedEnv", action_term: str = "arm_action"
) -> torch.Tensor:
    """현재 palm 액션 박스의 half-width (num_envs, 3) [m]. **policy obs 필수.**

    ★★2-스케일 액션(fab_test40)에서 같은 액션 벡터가 문맥에 따라 다른 지령이 된다.
      그 문맥이 관측에 없으면 POMDP 가 된다 — 정책이 자기 액션의 현재 스케일을 볼 수
      있어야 한다. FINE 인지 COARSE 인지가 이 값 하나로 드러난다.
    """
    term = env.action_manager.get_term(action_term)
    fine = term.fine_phase.unsqueeze(-1)
    return torch.where(fine, term._fine_half.expand_as(term.fine_anchor),
                       term._box_half.expand_as(term.fine_anchor))


def palm_action_anchor(
    env: "ManagerBasedEnv", action_term: str = "arm_action"
) -> torch.Tensor:
    """현재 액션 박스의 중심 (num_envs, 3), env 로컬 [m]. **policy obs 필수.**

    COARSE 면 박스 중심, FINE 이면 진입 시점에 래치한 지령이다. 스케일과 짝으로
    있어야 `[-1,1]` 이 어느 절대 좌표로 펼쳐지는지 정책이 알 수 있다.
    """
    term = env.action_manager.get_term(action_term)
    fine = term.fine_phase.unsqueeze(-1)
    return torch.where(fine, term.fine_anchor,
                       term._box_center.expand_as(term.fine_anchor))

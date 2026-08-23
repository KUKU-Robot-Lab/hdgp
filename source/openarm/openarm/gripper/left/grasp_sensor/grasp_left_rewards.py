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

"""리프트 판정에 **쥐고 있는가**를 더한 보상.

레퍼런스(`lift/mdp/rewards.py`)의 리프트 판정은 물체 원점 z 하나만 본다. 큐브에는 그것으로
충분하다 — 쳐도 잘 안 뜨고, 뜨더라도 곧 떨어진다. 우리 shaker 는 134 g 에 높이 175 mm 라
사정이 다르다.

★test3(1500 epoch)이 실증한 것: 정책이 컵을 **위로 힘껏 쳐 날리고**, 컵이 공중에 있는
  1.8 초 동안 리프트 보상(weight 15)과 goal-tracking(16)을 계속 받았다.
      리프트 판정 비율 85.9% / 그 동안 **TCP–컵 거리 평균 3044 mm**
      reaching_object 0.019 → 0.018 (평탄, 그리퍼는 컵에 가지 않는다)
      object_dropping 종료 99.8% (결국 떨어져서 끝난다)
  z 만 보는 판정에서 이것은 완벽하게 합리적인 전략이고, 보상을 아무리 재조정해도
  "던지기"가 "집기"보다 쉬운 한 사라지지 않는다.

그래서 판정을 `z > 임계` 에서 `z > 임계 **그리고** TCP 가 컵 곁에 있다` 로 바꾼다.
쳐서 날린 컵은 즉시 TCP 에서 멀어지므로 보상이 끊긴다.

레퍼런스 시그니처를 그대로 유지하고 `max_ee_distance` 만 더했다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, matrix_from_quat

from . import grasp_left_preset as P

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cup_upright_cos(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg) -> torch.Tensor:
    """컵 로컬 +z 의 world z 성분. 1 = 완전히 세워짐, 0 = 옆으로 누움."""
    obj: RigidObject = env.scene[object_cfg.name]
    w, x, y, z = obj.data.root_quat_w.unbind(-1)
    return 1.0 - 2.0 * (x * x + y * y)


def perpendicular_quality(
    env: "ManagerBasedRLEnv", robot_cfg: SceneEntityCfg, body_name: str,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """TCP z축(그리퍼 접근축)과 컵 z축(원통 축)이 **직교**하는 정도. 1 = 90°, 0 = 평행.

    ★원통을 2 지 그리퍼로 제대로 물려면 **옆에서** 접근해야 한다. 두 축이 평행하면
      컵 축 방향으로 내려꽂은 것이라 두 손가락이 지름을 잡지 못한다.
      test8 실측 81.2° — 방식은 맞지만 8.8° 어긋나 있었다.
    품질 = |sin(두 축 사이 각)| 이므로 90° → 1.0, 60° → 0.87, 0° → 0.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_idx = robot.body_names.index(body_name)
    w, x, y, z = robot.data.body_quat_w[:, body_idx, :].unbind(-1)
    tcp_axis = torch.stack(
        [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=-1
    )
    cw, cx, cy, cz = obj.data.root_quat_w.unbind(-1)
    cup_axis = torch.stack(
        [2 * (cx * cz + cw * cy), 2 * (cy * cz - cw * cx), 1 - 2 * (cx * cx + cy * cy)], dim=-1
    )
    cos_between = (tcp_axis * cup_axis).sum(dim=-1).abs().clamp(max=1.0)
    return (1.0 - cos_between * cos_between).clamp(min=0.0).sqrt()      # |sin|


def jaw_level_quality(
    env: "ManagerBasedRLEnv", robot_cfg: SceneEntityCfg, body_name: str
) -> torch.Tensor:
    """jaw 축이 수평인 정도. 1 = 완전 수평, 0 = 수직.

    jaw 축은 두 손가락을 잇는 방향, 즉 `gripper_base` 프레임의 **y 축**이다
    (URDF: 손가락이 base 의 ±y 로 벌어진다). 그 world z 성분의 크기가 곧 기울기의 sin 이다.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    body_idx = robot.body_names.index(body_name)
    w, x, y, z = robot.data.body_quat_w[:, body_idx, :].unbind(-1)
    jaw_axis_z = 2.0 * (y * z + w * x)              # 회전행렬 R[2,1]
    return (1.0 - jaw_axis_z.abs()).clamp(min=0.0)


def _held(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    ramp_zero_z: float,
    max_ee_distance: float,
    enclose_half_width: float,
    pad_offset: float,
    jaw_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    min_upright_cos: float = -1.0,
) -> torch.Tensor:
    """물체를 **제대로 들고 있는가**. (num_envs,) float, 0 또는 1.

    · 높이: **연속 램프**. `ramp_zero_z`(놓인 높이 +6.0 mm)에서 0, `minimal_height`
      (놓인 높이 +40 mm)에서 1. 그 사이가 끊김 없이 이어진다.
    · **enclose**: 두 손가락이 컵 축 양쪽에 있는 정도(0~1)를 램프에 곱한다 — 치거나 튀긴
      컵으로는 못 받는다.
    · TCP 가 곁에 있고(그리퍼가 아닌 부위로 떠받치는 것 차단)
    · 컵이 세워져 있다(`min_upright_cos`)

    ★★08.22 연속 램프에서 **되돌렸다**. 램프는 "IK test3 이 총보상 149 인데 컵을 3.6 mm 만
      올렸다"를 보고 넣은 것인데, 그 런의 진짜 원인은 게이트 **모양**이 아니라 **임계값**이었다:
      `minimal_height 0.27709` 가 놓인 컵 원점 0.29209 보다 **낮아** 상시 참이었다(공짜).
      같은 이진 게이트를 스폰 +4 cm 로 제대로 준 관절공간 런은 실제로 들어 올렸다 —
      **test13 lift 0.83 / test16 lift 0.84**(상한 대비). 절벽이 아니었다.
      "IK 1 차가 827 epoch 동안 lifting 0.000" 도 보상이 아니라 제어기 문제였다
      (diff-IK 씨앗 처짐 111 mm + 변화율 무제한). 검증된 구성으로 복귀한다.

    ★임계는 반드시 **놓인 컵의 원점**에서 출발한다. shaker 원점은 바닥에서 92 mm 위라
      "상면 + 4 cm" 로 계산하면 놓인 상태보다 낮아져 게이트가 상시 열린다(test1-r2 실증:
      lifting 14.63/15 인데 reaching 은 0.007 로 떨어졌다 = 가만히 있는 것이 최적).

    ★컵 자세 조건이 필요한 이유: 근접 조건만으로는 컵을 **47° 기울인 채** 손가락 끝으로
      걸어 올리는 파지가 학습된다(test4 실측: 컵 기울기 47.1°, 그리퍼 개도 5.6 mm 로
      몸통을 물지 못한 상태). 사용자 요구는 "수평으로 제대로 잡기" 다.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    obj_pos_w = obj.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    # ★★08.23 **연속 램프로 되돌렸다.** `ramp_zero_z`(놓인 높이 +6.0 mm)에서 0,
    #   `minimal_height`(놓인 높이 +40 mm)에서 1. 근접·자세는 게이트로 남는다.
    lifted = ((obj_pos_w[:, 2] - ramp_zero_z) / (minimal_height - ramp_zero_z)).clamp(0.0, 1.0)
    near = torch.norm(obj_pos_w - ee_pos_w, dim=1) < max_ee_distance
    upright = _cup_upright_cos(env, object_cfg) > min_upright_cos
    # ★★08.23 램프에 **enclose 를 곱한다.** 순수 램프(fab_test6)는 학습 초기에 컵이 튀어
    #   오르는 순간에도 부분 점수를 줬고(1~50 epoch 의 lift 0.002 가 전 구간 최고), 컵을
    #   치기에는 주먹이 유리해서 정책이 주먹으로 고착됐다 — fab_test1 의 실패 모드로 회귀했다
    #   (enclose 0.019 · 개도 4.4 mm · 거의닫힘 76% · drop 0.73).
    #   ⚠ `near` 80 mm 게이트로는 못 막는다 — 툭 치는 거리가 그 안이다.
    #   enclose 를 곱하면 주먹(0.019)도 손을 떠난 컵도 0 이고, 제대로 감싼 상태(0.78~0.85)만
    #   받는다. 부수 효과로 **자동 커리큘럼**이 된다: 감싸기를 배우기 전에는 lift 항이
    #   사실상 0(=하드 게이트와 동일, 치는 유인 없음)이고, 감싼 뒤에 램프가 켜진다.
    held = _enclose(env, enclose_half_width, pad_offset, jaw_cfg, object_cfg)
    return lifted * held * (near & upright).float()


def held_with_good_pose(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    ramp_zero_z: float,
    max_ee_distance: float,
    enclose_half_width: float,
    pad_offset: float,
    jaw_cfg: SceneEntityCfg,
    body_name: str,
    upright_zero_at_cos: float = 0.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """든 상태에서 **자세가 좋을수록** 커지는 보너스 (0~1).

    품질 = (컵이 똑바로 선 정도) × (jaw 가 수평인 정도). 둘 다 0~1 연속.

    ★TCP축⊥컵축 항은 **뺐다** — |sin| 이라 81.8° 에서 이미 0.99 여서 개선 압력이 없었고
      (test8 81.2° → test12 81.8°, 제자리), 결과적으로 필요한 것은 컵을 똑바로 드는 것이다.
    ★upright 는 cos 를 그대로 쓰면 12.8° 에서 0.975 라 압력이 없다. `upright_zero_at_cos`
      에서 0 이 되도록 재척도해 가파르게 만든다.

    ★★자세를 **게이트로 넣으면 안 된다** — 한 번 실패한 설계다. 컵 자세를 40° AND 게이트로
      걸었더니(test6/test7) 파지 중 필연적인 흔들림이 전부 차단돼 양의 보상이 **완전히 0**이
      됐고, 남은 것이 페널티뿐이라 **에피소드를 빨리 끝내는 것이 최적**이 됐다:
          lifting 6.14 → 0.0000 / 에피소드 길이 130 → 13 / 총보상 +34.9 → −0.46
      학습이 시작조차 못 한다. 자세는 반드시 연속 보너스로만 유도한다.
    """
    gate = _held(env, minimal_height, ramp_zero_z, max_ee_distance, enclose_half_width,
                 pad_offset, jaw_cfg, object_cfg, ee_frame_cfg)
    cos_tilt = _cup_upright_cos(env, object_cfg)
    upright = ((cos_tilt - upright_zero_at_cos) / (1.0 - upright_zero_at_cos)).clamp(0.0, 1.0)
    return gate * upright * jaw_level_quality(env, robot_cfg, body_name)


def object_is_held_and_lifted(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    ramp_zero_z: float,
    max_ee_distance: float,
    enclose_half_width: float,
    pad_offset: float,
    jaw_cfg: SceneEntityCfg,
    min_upright_cos: float = -1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """`mdp.object_is_lifted` 에 근접·컵 자세 조건을 더한 것."""
    return _held(env, minimal_height, ramp_zero_z, max_ee_distance, enclose_half_width,
                 pad_offset, jaw_cfg, object_cfg, ee_frame_cfg, min_upright_cos)


def object_goal_distance_when_held(
    env: "ManagerBasedRLEnv",
    std: float,
    minimal_height: float,
    ramp_zero_z: float,
    max_ee_distance: float,
    enclose_half_width: float,
    pad_offset: float,
    jaw_cfg: SceneEntityCfg,
    command_name: str,
    min_upright_cos: float = -1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """`mdp.object_goal_distance` 의 게이트를 근접 조건까지 요구하도록 바꾼 것.

    레퍼런스와 마찬가지로 목표 위치는 **로봇 베이스 기준** 명령을 world 로 변환해 쓴다.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3]
    )
    distance = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)
    gate = _held(env, minimal_height, ramp_zero_z, max_ee_distance, enclose_half_width,
                 pad_offset, jaw_cfg, object_cfg, ee_frame_cfg, min_upright_cos)
    return gate * (1 - torch.tanh(distance / std))


def object_settled_at_goal(
    env: "ManagerBasedRLEnv",
    std: float,
    lin_vel_std: float,
    ang_vel_std: float,
    minimal_height: float,
    ramp_zero_z: float,
    max_ee_distance: float,
    enclose_half_width: float,
    pad_offset: float,
    jaw_cfg: SceneEntityCfg,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """컵을 목표까지 옮겨 **가만히 정지**시켰는가 (0~1).

    ★레퍼런스 lift 의 `object_goal_distance` 는 **거리만** 본다. 목표 근처에서 컵이 계속
      흔들려도 만점이라, "옮겨서 세워 둔다"는 요구를 표현하지 못한다. 실제로 test8 은
      goal-tracking 이 상한의 68% 까지 갔는데 정밀 항(`goal_fine`)은 16% 에 머물렀다.

    품질 = (게이트) × (목표 근접) × (정지 정도). 셋 다 연속이라 gradient 가 이어진다.
      · 목표 근접을 곱하므로 "든 채 제자리에 가만히 있기"는 0 이다(목표에서 멀면 근접이 0).
      · 정지 정도는 컵의 선속도·각속도 둘 다 본다 — 각속도를 빼면 제자리에서 빙빙 도는
        상태가 만점이 된다.

    ⚠ 자세와 마찬가지로 **게이트가 아니라 보너스**다. 판정 게이트에 조건을 더 얹으면
      양의 보상이 0 이 되고 조기 종료가 최적이 된다(test6/test7 에서 실증).
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3]
    )
    distance = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)
    near_goal = 1.0 - torch.tanh(distance / std)

    lin = torch.norm(obj.data.root_lin_vel_w, dim=1)
    ang = torch.norm(obj.data.root_ang_vel_w, dim=1)
    # ★선속도·각속도를 **평균**한다(곱이 아니라). 곱하면 각각 0.19 일 때 0.036 까지 떨어져
    #   lifting(11.0)에 비해 400 배 작아지고, 조금 개선해도 증가분이 다른 항에 묻힌다
    #   (test11 실측: settle 이 상한의 0.4% 에서 정체). 평균이면 같은 상태에서 5 배 크다.
    still = 0.5 * (1.0 - torch.tanh(lin / lin_vel_std)) + 0.5 * (
        1.0 - torch.tanh(ang / ang_vel_std)
    )

    gate = _held(env, minimal_height, ramp_zero_z, max_ee_distance, enclose_half_width,
                 pad_offset, jaw_cfg, object_cfg, ee_frame_cfg)
    return gate * near_goal * still


class ActionJerkL2(ManagerTermBase):
    """액션의 **2차 차분**(jerk) 제곱합 페널티.

    ★레퍼런스 lift 에는 `action_rate_l2`(1차 차분)와 `joint_vel_l2` 만 있다. 그런데
      test12 실측은 1차보다 2차가 더 크고 방향 반전이 68.6% 였다:
          1차 |Δa| 0.943 / **2차 |Δ²a| 1.755** / 방향 반전 68.6%
      전형적인 고주파 채터링이다. `action_rate` 는 "변화량"만 벌하므로 **일정 크기로 계속
      진동하면 그 대가를 감수하고 유지**할 수 있다. jerk 는 방향을 되돌릴 때마다 커지므로
      진동을 직접 벌한다.

    구현이 클래스인 이유: 2차 차분에는 직전 **차분**이 필요한데 `ActionManager` 는
    `action` 과 `prev_action` 만 보관한다. 그래서 직전 차분을 이 term 이 들고 있는다.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._prev_delta = torch.zeros(
            env.num_envs, env.action_manager.total_action_dim, device=env.device
        )

    def reset(self, env_ids=None):
        # ★에피소드 경계에서 초기화하지 않으면 리셋 직후 스텝이 가짜 jerk 로 벌을 받는다.
        if env_ids is None:
            self._prev_delta[:] = 0.0
        else:
            self._prev_delta[env_ids] = 0.0

    def __call__(self, env) -> torch.Tensor:  # noqa: D102
        delta = env.action_manager.action - env.action_manager.prev_action
        jerk = torch.sum(torch.square(delta - self._prev_delta), dim=1)
        self._prev_delta[:] = delta
        return jerk


def _jaw_frame(
    env: "ManagerBasedRLEnv",
    pad_offset: float,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
):
    """턱 기준 프레임 — 두 손가락(패드 중앙 보정), 턱 축 u, 중점, 컵 축 최근접점.

    ★기준선은 **손가락 패드 중앙**이다. 손가락 강체 원점은 base z=+15 mm 인데 성공 파지의
      컵 축은 z=+46.9 mm 다(test17 13,058 샘플 중앙값). 원점 그대로 쓰면 보상이
      "컵을 손바닥까지 32 mm 더 밀어넣어라"를 가리킨다.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    fingers = robot.data.body_pos_w[:, robot_cfg.body_ids, :]      # (N, 2, 3)
    # 손가락은 base 의 y 로만 미끄러지므로 손가락 자세 = base 자세다(접근축을 여기서 얻는다).
    approach = matrix_from_quat(robot.data.body_quat_w[:, robot_cfg.body_ids[0], :])[:, :, 2]
    fingers = fingers + (approach * pad_offset).unsqueeze(1)
    p_l, p_r = fingers[:, 0, :], fingers[:, 1, :]
    mid = 0.5 * (p_l + p_r)
    jaw = p_r - p_l
    u = jaw / torch.norm(jaw, dim=-1, keepdim=True).clamp(min=1e-6)

    cup_z = matrix_from_quat(obj.data.root_quat_w)[:, :, 2]
    to_mid = mid - obj.data.root_pos_w
    # ★★축 위 최근접점을 **잡을 수 있는 높이 대역**으로 clamp 한다.
    #   clamp 가 없으면 컵 축이 **무한 직선**이라 컵 위 허공에서 감싸도 만점이 나온다.
    #   fab_test10 이 정확히 그 행동을 학습했다 — 턱이 컵 원점 +157.6 mm(상단 +83 mm 보다
    #   75 mm 위)에서 between 2.15/3.0 을 받으면서 컵은 0.1 mm 도 안 움직였다.
    axis_t = (to_mid * cup_z).sum(-1, keepdim=True).clamp(
        P.CUP_GRASP_BAND_AXIS[0], P.CUP_GRASP_BAND_AXIS[1]
    )
    cup_pt = obj.data.root_pos_w + cup_z * axis_t
    return p_l, p_r, u, mid, cup_pt


def _enclose(
    env: "ManagerBasedRLEnv",
    enclose_half_width: float,
    pad_offset: float,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """두 손가락이 컵 축 **양쪽**에 있는가 (0~1). 주먹도, 날아간 컵도 0 이다.

    s_l = (컵축점 − 왼손가락)·u,  s_r = (오른손가락 − 컵축점)·u — 둘 다 양수여야 사이에 있다.
    실측 판별력: 주먹 정책 **0.019~0.026** · 제대로 감싼 정책 **0.78~0.85**.
    """
    p_l, p_r, u, _mid, cup_pt = _jaw_frame(env, pad_offset, robot_cfg, object_cfg)
    s_l = ((cup_pt - p_l) * u).sum(-1)
    s_r = ((p_r - cup_pt) * u).sum(-1)
    return (torch.minimum(s_l, s_r) / enclose_half_width).clamp(0.0, 1.0)


def _jaw_geometry(
    env: "ManagerBasedRLEnv",
    along_std: float,
    lateral_std: float,
    enclose_half_width: float,
    pad_offset: float,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
):
    """턱 ↔ 컵 기하 (align, enclose). 두 보상 항이 **같은 기하**를 쓰도록 여기 모은다.

    · align   = 평균( 턱 축 방향 정렬, 턱 축 직선까지의 근접 )  — 곱하지 않는다
    · enclose = 두 손가락이 컵 축 **양쪽**에 있는가 (0~1)
    """
    p_l, p_r, u, mid, cup_pt = _jaw_frame(env, pad_offset, robot_cfg, object_cfg)
    d = cup_pt - mid
    along = (d * u).sum(-1).abs()
    lateral = (d - u * (d * u).sum(-1, keepdim=True)).norm(dim=-1)
    align = 0.5 * (1.0 - torch.tanh(along / along_std)) + 0.5 * (
        1.0 - torch.tanh(lateral / lateral_std)
    )
    s_l = ((cup_pt - p_l) * u).sum(-1)
    s_r = ((p_r - cup_pt) * u).sum(-1)
    enclose = (torch.minimum(s_l, s_r) / enclose_half_width).clamp(0.0, 1.0)
    return align, enclose


def cup_between_jaws(
    env: "ManagerBasedRLEnv",
    along_std: float,
    lateral_std: float,
    enclose_half_width: float,
    enclose_floor: float,
    pad_offset: float,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """**컵이 두 턱 사이에 들어왔는가** (0~1). 리프트와 무관한 전제조건 보상.

    ★왜 필요한가 — fab_test1 실측: 개도 3.1 mm · '열기' 지령 0.0% · enclose 0.026.
      **주먹을 쥔 채 컵 옆구리를 누르고** 있었다. 닫힌 턱에는 컵이 들어갈 자리가 없다.
      이 항을 넣은 fab_test4 는 enclose **0.845** 로 뒤집혔다 — 항은 의도대로 작동한다.

    ★enclose 는 바닥값을 남긴다. 완전 곱셈이면 주먹 상태에서 "가서 정렬하라" 신호까지
      사라진다(= 게이트와 같아진다). 바닥값이 있으면 개선 경로가 단조롭다.
      ⚠ 폐쇄 보상(`grip_closure_when_enclosed`)에는 **바닥값을 주지 않는다** — 거기서는
        "감싸지 않은 폐쇄"가 정확히 0 이어야 옛 주먹 해킹이 되살아나지 않는다.
    """
    align, enclose = _jaw_geometry(env, along_std, lateral_std, enclose_half_width,
                                   pad_offset, robot_cfg, object_cfg)
    return align * (enclose_floor + (1.0 - enclose_floor) * enclose)


def grip_closure_when_enclosed(
    env: "ManagerBasedRLEnv",
    along_std: float,
    lateral_std: float,
    enclose_half_width: float,
    pad_offset: float,
    open_pos: float,
    drive_joint: str,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """**컵을 감싼 상태에서 닫는 것**에 주는 보상 (0~1). 리프트와 무관하다.

    ★★왜 필요한가 — fab_test4 실측이 정확히 이 구멍을 보여줬다:
          enclose **0.845**(턱이 컵을 잘 감쌌다) · 개도 평균 **35.5 mm**
          '열기' 지령 **78.0%** · 거의 닫힘 스텝 **0.0%** · 컵 최대 상승 +20.5 mm
      턱은 제자리에 갔는데 **한 번도 닫지 않는다.** 닫는 것을 보상하는 항이 없고, 닫다가
      컵이 밀리면 `cup_between_jaws` 를 잃으니 **닫지 않는 것이 최적**이었다.
      (닭-달걀: 닫는 이득은 들어올린 뒤에만 생기는데, 들려면 먼저 닫아야 한다.)

    ★★왜 옛 `gripper_closure_on_cup` 이 아닌가 — 그 식은 closure 를 **약한 straddle** 에만
      곱해서 허공/주먹 폐쇄가 사실상 만점이었다. 여기서는 **enclose 를 곱한다**:
      주먹은 컵 반경에 막혀 두 손가락이 컵 축 양쪽에 설 수 없으므로 enclose≈0 이다
      (fab_test1 실측 0.026 이 그 증거).
    ⚠ 바닥값을 주지 않는다 — "감싸지 않은 폐쇄"는 정확히 0 이어야 한다.

    closure 는 컵 지름에서 포화한다(58 mm 단면이면 약 0.32). 그 이상은 리프트로만 벌 수
    있다 — 의도한 대로다.
    """
    align, enclose = _jaw_geometry(env, along_std, lateral_std, enclose_half_width,
                                   pad_offset, robot_cfg, object_cfg)
    robot: Articulation = env.scene[robot_cfg.name]
    drive = robot.data.joint_pos[:, robot.joint_names.index(drive_joint)]
    closure = (1.0 - drive / open_pos).clamp(0.0, 1.0)
    return align * enclose * closure


def ee_grasp_point_distance(
    env: "ManagerBasedRLEnv",
    std: float,
    grasp_offset: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """도달 보상. 레퍼런스와 같은 tanh 커널이되 목표가 컵 원점이 아니라 **파지점**이다.

    ★★08.22 — 레퍼런스 `object_ee_distance` 는 컵 **원점**을 겨냥한다. 큐브는 원점이
      기하 중심이라 그게 파지점과 같지만, 우리 shaker 는 다르다:
          컵 원점 = 상면 +92 mm · 그리퍼 통과 대역 = 상면 +10~85 mm
      원점 높이의 컵 지름(88 mm)이 개구(84.5 mm)보다 넓어 **턱이 물리적으로 못 들어간다.**
      즉 도달 보상이 학습 내내 **들어갈 수 없는 높이**를 가리키고 있었다.
      G3 스크립트가 같은 지점을 겨냥했을 때 실측: pregrasp 까지 TCP 오차 2.9 mm 인데
      진입에서 100.2 mm(벡터 −31,+93,+15)로 튕긴다. 파지 대역으로 내리면 70.7 mm 로 준다.
      ⚠ 관절공간 test17 이 성공한 건 lifting(15) 이 압도적이라 정책이 도달 보상이 가리키는
        높이를 **무시하도록** 배웠기 때문이다. 제어 여유가 적은 트랙은 그걸 못 이긴다.

    오프셋은 **컵의 로컬 축을 따라** 적용한다 — world z 로 내리면 컵이 기울었을 때
    파지점이 컵 밖으로 나간다.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cup_z = matrix_from_quat(obj.data.root_quat_w)[:, :, 2]
    grasp_pt = obj.data.root_pos_w + cup_z * grasp_offset
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    return 1.0 - torch.tanh(torch.norm(grasp_pt - ee_w, dim=1) / std)

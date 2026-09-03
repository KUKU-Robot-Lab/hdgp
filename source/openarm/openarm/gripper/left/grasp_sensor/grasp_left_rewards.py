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
    lat_ok: float,
    along_ok: float,
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
    # ★★08.24 게이트를 `_enclose` 에서 **`grasp_ok`** 로 바꿨다.
    #   `enclose` 는 판별력이 없다 — fab_test11 이 컵 축에서 **옆으로 85.5 mm** 떨어져
    #   개도 16.3 mm(컵 58 mm 보다 좁게) 닫은 채 enclose **0.824** 를 받으면서
    #   컵을 0.2 mm 도 못 들었다(성공 정책 test17 은 0.804 — 구분이 안 된다).
    #   턱이 벌어져 있으면 멀리 떨어져도 "축이 턱 사이를 지난다"가 성립하기 때문이다.
    #   `grasp_ok` 는 lateral 을 직접 보므로 그 구멍이 닫힌다(성공 20~22 vs 실패 79~86 mm).
    #   ⚠ 하드 게이트라 감쌈 이전의 연속 기울기(옛 "자동 커리큘럼")는 사라진다. 그 역할은
    #     이제 **액션 게이트**가 대신한다 — 접근 전에는 그리퍼가 강제로 열려 있어
    #     "닫고 서 있기" 국소최적 자체가 도달 불가능해진다.
    held = grasp_ok(env, lat_ok, along_ok, pad_offset, jaw_cfg, object_cfg).float()
    return lifted * held * (near & upright).float()


def held_with_good_pose(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    ramp_zero_z: float,
    max_ee_distance: float,
    enclose_half_width: float,
    pad_offset: float,
    lat_ok: float,
    along_ok: float,
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
                 pad_offset, lat_ok, along_ok, jaw_cfg, object_cfg, ee_frame_cfg)
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
    lat_ok: float,
    along_ok: float,
    jaw_cfg: SceneEntityCfg,
    min_upright_cos: float = -1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """`mdp.object_is_lifted` 에 근접·컵 자세 조건을 더한 것."""
    return _held(env, minimal_height, ramp_zero_z, max_ee_distance, enclose_half_width,
                 pad_offset, lat_ok, along_ok, jaw_cfg, object_cfg, ee_frame_cfg, min_upright_cos)


def object_goal_distance_when_held(
    env: "ManagerBasedRLEnv",
    std: float,
    minimal_height: float,
    ramp_zero_z: float,
    max_ee_distance: float,
    enclose_half_width: float,
    pad_offset: float,
    lat_ok: float,
    along_ok: float,
    jaw_cfg: SceneEntityCfg,
    command_name: str,
    min_upright_cos: float = -1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """`mdp.object_goal_distance` 의 게이트를 근접 조건까지 요구하도록 바꾼 것.

    레퍼런스와 마찬가지로 목표 위치는 **로봇 베이스 기준** 명령을 world 로 변환해 쓴다.

    ★★fab_test73(사용자 지시): 거리를 **TCP** 로 잰다(레퍼런스는 컵 원점).
      이유는 프레임 정합이다 — 목표 상자 `GOAL_POINT`/`GOAL_JITTER` 는
      `probe_grip_l_goal_ws_xalign.py` 의 **TCP 제약 IK**(tcp_z ∥ world +x)로
      "도달 가능한 곳만" 골라 만든 것이다. 채점을 컵 원점으로 하면 도달성을 검증한
      바디와 채점하는 바디가 달라진다. 실측 계통차 약 30 mm(턱 중점이 컵 원점보다
      30.3 mm 아래).
      ⚠ 대신 컵은 게이트의 `near` 임계(`max_ee_distance` 80 mm)만큼 목표에서 벗어날 수
        있다. **최종 합격 판정은 반드시 컵–목표로 읽는다** — `diag_cup_goal_dist`(컵)와
        `diag_tcp_goal_dist`(TCP)를 나란히 찍어 벌어지는 순간을 본다.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3]
    )
    distance = torch.norm(des_pos_w - ee_frame.data.target_pos_w[..., 0, :], dim=1)
    gate = _held(env, minimal_height, ramp_zero_z, max_ee_distance, enclose_half_width,
                 pad_offset, lat_ok, along_ok, jaw_cfg, object_cfg, ee_frame_cfg, min_upright_cos)
    return gate * (1 - torch.tanh(distance / std))


def object_goal_distance_height_gated(
    env: "ManagerBasedRLEnv",
    std: float,
    gate_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """레퍼런스 `mdp.object_goal_distance` **그대로** — 게이트도 거리도 컵 원점.

    ★★fab_test74(E1) 의 단일 변수다. IsaacLab `manager_based/manipulation/lift` 대조에서
      나온 유일한 구조적 차이가 **goal 신호가 도는 시점**이었다:
        레퍼런스  게이트 `cube.z > 0.04` 인데 스폰이 0.055 → **step 0 부터 항상 참**.
                  정책이 조건부 목표 추종을 맨 처음부터 배우고 파지가 그 위에 얹힌다.
                  (부수적으로 `lifting_object` 15 는 스폰부터 상수라 gradient 가 없다 —
                   레퍼런스에서 컵을 실제로 들게 만드는 건 goal 보상이다.)
        우리      `_held` = 램프 ∧ grasp_ok ∧ near ∧ upright → 파지·리프트를 **완성한 뒤**.
                  이미 굳은 정책에 조건부 신호가 늦게 도착한다.
      실측 증거: t73 결정론 프로브의 목표 조건부 추종 기울기가 x 0.109 · y 0.297 · z 0.053
      (1.0 이어야 한다). 정책이 목표가 어디든 **늘 같은 자리**(0.475, 0.208, 0.406)에 놓는다.

    ⚠⚠ **거리는 반드시 컵 원점이다**(사용자 지적). 게이트를 높이 하나로 낮추면서 거리를
      TCP 로 재면 **빈 그리퍼를 목표에 가져다 놓기만 해도 만점**이 된다 — 컵은 테이블에 둔 채로.
      t73 이 TCP 채점으로 안전했던 건 `_held` 가 파지를 요구해 그 구멍을 막고 있었기 때문이고,
      게이트를 여는 순간 그 전제가 사라진다. TCP 채점은 `held` 모드 전용이다.
      부수 효과로 이 모드는 **보상이 재는 값과 합격을 재는 값이 같아진다**(`diag_cup_goal_dist`).
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3]
    )
    obj_pos_w = obj.data.root_pos_w
    distance = torch.norm(des_pos_w - obj_pos_w, dim=1)
    return (obj_pos_w[:, 2] > gate_height).float() * (1 - torch.tanh(distance / std))


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
    lat_ok: float,
    along_ok: float,
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
                 pad_offset, lat_ok, along_ok, jaw_cfg, object_cfg, ee_frame_cfg)
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
    band: tuple[float, float] | None = None,
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
    #   ★대역은 **호출자가 줄 수 있다** — v2 는 판 위 80~150 mm 로 올려 쓴다.
    #     기본값(v1 값)을 쓰면 v1 거동은 그대로다. 예전엔 환경변수로 이 모듈의 상수를
    #     통째로 바꿨는데, 그러면 같은 프로세스의 v1 까지 조용히 오염된다.
    _band = band if band is not None else P.CUP_GRASP_BAND_AXIS
    axis_t_raw = (to_mid * cup_z).sum(-1, keepdim=True)
    axis_t = axis_t_raw.clamp(_band[0], _band[1])
    cup_pt = obj.data.root_pos_w + cup_z * axis_t
    # ★clamp **전** 축 좌표도 돌려준다 — "턱이 파지 대역 안인가"는 clamp 된 값으로는
    #   알 수 없다(밖에 있어도 경계값으로 접혀 들어온다). 접근 성공 판정에 필요하다.
    return p_l, p_r, u, mid, cup_pt, axis_t_raw.squeeze(-1)


def grasp_ok(
    env: "ManagerBasedRLEnv",
    lat_ok: float,
    along_ok: float,
    pad_offset: float,
    jaw_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    band: tuple[float, float] | None = None,
) -> torch.Tensor:
    """**접근 성공** — 컵이 실제로 턱 사이 파지 위치에 있는가. (num_envs,) bool

    액션 게이트(그리퍼 강제 개방 해제)와 리프트 보상 게이트가 **같은 술어**를 쓴다.
    이 트랙에서 두 함수가 서로 다른 자를 쓰다 조용히 어긋난 사고가 반복됐다
    (패드 중앙 보정 누락 · 컵 축 clamp 누락). 단일 출처로 묶는다.

    ★★기준은 `lateral` 이다. `enclose` 는 **판별력이 없다** — 실측:
          정책               along    lateral   enclose
          test17(파지 성공)   13.5 mm   21.7 mm   0.804
          fab_test8(성공)     12.2 mm   20.2 mm   0.849
          fab_test1(주먹)     27.8 mm   78.6 mm   0.022
          fab_test11(옆 대기) 12.0 mm  **85.5 mm** **0.824**
      턱이 벌어져 있으면 컵 축에서 8.5 cm 떨어져도 "축이 턱 사이를 지난다"가 성립한다.
      fab_test11 이 정확히 그 상태로 4000 epoch 을 돌며 컵을 0.2 mm 도 못 들었다.

    ★`axis_t` 는 **clamp 전** 값을 쓴다. clamp 된 값은 대역 밖이어도 경계로 접혀 들어와
      "대역 안"이 항상 참이 된다.
    """
    _band = band if band is not None else P.CUP_GRASP_BAND_AXIS
    _p_l, _p_r, u, mid, cup_pt, axis_t = _jaw_frame(env, pad_offset, jaw_cfg, object_cfg,
                                                    band=_band)
    d = cup_pt - mid
    along = (d * u).sum(-1).abs()
    lateral = (d - u * (d * u).sum(-1, keepdim=True)).norm(dim=-1)
    in_band = (axis_t > _band[0]) & (axis_t < _band[1])
    return (lateral < lat_ok) & (along < along_ok) & in_band


def jaw_lateral(
    env: "ManagerBasedRLEnv",
    pad_offset: float,
    jaw_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    band: tuple[float, float] | None = None,
) -> torch.Tensor:
    """턱 축 직선에서 컵 축까지의 수직거리 (m). 래치 **해제** 판정에 쓴다.

    ★★`band` 를 **반드시 `grasp_ok` 와 같은 값으로** 넘겨야 한다. 래치는 `grasp_ok`
      로 걸리고 이 함수로 풀리는데, 둘이 다른 대역을 보면 `cup_pt` 가 컵 축의 다른
      높이로 clamp 되어 "걸자마자 풀리는" 채터링이 된다. 09.03 정리에서 실제로
      이 인자를 빠뜨려 grasp_ok 가 0.033 → 0.455 로 바뀌었다.
    """
    _p_l, _p_r, u, mid, cup_pt, _axis = _jaw_frame(env, pad_offset, jaw_cfg, object_cfg,
                                                   band=band)
    d = cup_pt - mid
    return (d - u * (d * u).sum(-1, keepdim=True)).norm(dim=-1)


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
    p_l, p_r, u, _mid, cup_pt, _axis = _jaw_frame(env, pad_offset, robot_cfg, object_cfg)
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
    band: tuple[float, float] | None = None,
):
    """턱 ↔ 컵 기하 (align, enclose). 두 보상 항이 **같은 기하**를 쓰도록 여기 모은다.

    · align   = 평균( 턱 축 방향 정렬, 턱 축 직선까지의 근접 )  — 곱하지 않는다
    · enclose = 두 손가락이 컵 축 **양쪽**에 있는가 (0~1)
    """
    p_l, p_r, u, mid, cup_pt, _axis = _jaw_frame(env, pad_offset, robot_cfg, object_cfg,
                                                 band=band)
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


def gripper_gate_open(env: "ManagerBasedRLEnv", action_term: str = "gripper_action") -> torch.Tensor:
    """그리퍼 게이트가 열렸는가 (num_envs, 1) float. **관측 항**으로 쓴다.

    ★★하드 게이트는 정책이 볼 수 없는 **숨은 상태**다. phase 0 에서 정책의 그리퍼 지령은
      롤아웃 버퍼에 기록되지만 실행되지 않아, 그 차원의 gradient 가 환경 응답과 무관해진다.
      게이트 상태를 관측에 넣어야 정책이 "지금 내 그리퍼 지령은 무시된다"를 알 수 있다.
      obs 차원이 1 늘어난다 — 체크포인트 호환이 깨지므로 fresh 학습에서만 켤 것.
    """
    term = env.action_manager.get_term(action_term)
    return term.gate_open.float().unsqueeze(-1)


def diag_action_z_mu(env: "ManagerBasedRLEnv", action_term: str = "arm_action") -> torch.Tensor:
    """palm 액션 z 성분의 평균. **weight 0 진단 항.**

    ★★fab_test64 실측이 이 항을 만든 이유다: z 의 raw 액션 평균이 **1.336** 으로 상한 +1 을
      넘어 있었고 |a|>0.95 가 **90.3%** 였다. `FabricPalmAction` 은 액션을 clamp 하므로
      그 구간의 미분이 0 이고, 그래서 z 는 목표를 따라갈 수단 자체가 없었다
      (목표 조건부 기울기 z **0.005** vs x 1.019 · y 0.921).
    ⚠ 지금까지 이 값은 **프로브로만** 볼 수 있어 판이 끝난 뒤에야 알았다. TFEvents 에
      찍어 학습 중 판정한다 — 1.0 을 넘기 시작하는 epoch 이 곧 병목의 발생 시점이다.
    """
    return env.action_manager.get_term(action_term).raw_actions[:, 2]


def diag_action_z_sat(env: "ManagerBasedRLEnv", action_term: str = "arm_action") -> torch.Tensor:
    """palm 액션 z 가 경계(|a|>0.95)에 붙어 있는 비율. **weight 0 진단 항.**

    포화율이 높으면 그 축은 clamp 미분 0 이라 **조건부 학습이 구조적으로 불가능**하다.
    판정: 이 값이 0.3 을 넘으면 박스 상한이 다시 천장 노릇을 하는 것이다.
    """
    return (env.action_manager.get_term(action_term).raw_actions[:, 2].abs() > 0.95).float()


def diag_action_axis_mu(
    env: "ManagerBasedRLEnv", axis: int, action_term: str = "arm_action"
) -> torch.Tensor:
    """palm 액션의 축별 평균. **weight 0 진단 항.**

    ★★fab_test68 이 이 항을 만든 이유: z 만 찍어 두고 x·y 를 못 봐서, t67 의 진짜
      병목(리프트 후 y 포화 99.7% · mu 3.11)을 판이 끝난 뒤 프로브로야 알았다.
      세 축을 다 찍는다 — 어느 축이 언제 박스를 벗어나는지가 곧 병목의 발생 시점이다.
    """
    return env.action_manager.get_term(action_term).raw_actions[:, axis]


def diag_action_axis_sat(
    env: "ManagerBasedRLEnv", axis: int, action_term: str = "arm_action"
) -> torch.Tensor:
    """palm 액션의 축별 경계(|a|>0.95) 부착률. **weight 0 진단 항.**"""
    return (
        env.action_manager.get_term(action_term).raw_actions[:, axis].abs() > 0.95
    ).float()


def diag_cup_goal_dz(
    env: "ManagerBasedRLEnv",
    command_name: str = "object_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """목표 z − 컵 z (m). **weight 0 진단 항.**

    fab_test64 결정론 실측에서 남은 거리 103 mm 의 최대 성분이 **dz −64 mm** 였다.
    x·y 는 이미 목표를 따라가므로(기울기 1.02 / 0.92) 이 값 하나가 이송의 병목이다.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3]
    )
    return des_pos_w[:, 2] - obj.data.root_pos_w[:, 2]


def diag_cup_goal_dist(
    env: "ManagerBasedRLEnv",
    command_name: str = "object_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """목표 ↔ **컵 원점** 거리 (m). **weight 0 진단 항.**

    ★★보상은 TCP 로 채점하지만 **합격 판정은 컵이다**. 둘이 벌어지면 여기서 보인다
    (게이트의 `near` 80 mm 만큼 벌어질 수 있다 — reward-audit Check 2).
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3]
    )
    return torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)


def diag_tcp_goal_dist(
    env: "ManagerBasedRLEnv",
    command_name: str = "object_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """목표 ↔ **TCP** 거리 (m) = 보상이 실제로 재는 값. **weight 0 진단 항.**"""
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3]
    )
    return torch.norm(des_pos_w - ee_frame.data.target_pos_w[..., 0, :], dim=1)


def gripper_gate_rate(env: "ManagerBasedRLEnv", action_term: str = "gripper_action") -> torch.Tensor:
    """게이트가 열린 env 비율. **weight 0 진단 항** — TFEvents 에 찍혀야 조기 판정이 된다.

    이번 런의 1차 관전 지표다. epoch 200 안에 0.1 을 못 넘으면 게이트가 너무 빡빡한 것이고,
    `GRASP_GATE_LATERAL_OK` 를 0.040 으로 완화해야 한다(fab_test9~11 정체의 재발 방지).
    """
    term = env.action_manager.get_term(action_term)
    return term.gate_open.float()

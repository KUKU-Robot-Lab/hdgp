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

from . import grasp_left_observations as obs_mdp
from . import grasp_left_preset as P

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cup_upright_cos(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg) -> torch.Tensor:
    """컵 로컬 +z 의 world z 성분. 1 = 완전히 세워짐, 0 = 옆으로 누움."""
    obj: RigidObject = env.scene[object_cfg.name]
    w, x, y, z = obj.data.root_quat_w.unbind(-1)
    return 1.0 - 2.0 * (x * x + y * y)


def lift_height(
    env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """컵 **최저점**이 테이블 상면 위로 뜬 높이 [m]. 놓여 있으면 0, 기울여도 0.

    ★★원점 z 로 리프트를 재면 **기울이기가 리프트로 계산된다.** 컵을 바닥 림 모서리로
      피벗시키면 원점이 실제로 올라간다 — 최대 `CUP_TIP_RISE_MAX` = 4.61 mm @ 17.5°:

          기울기   0°     10°    17.5°   30°    45°
          원점    0.0    3.6    4.61    2.2   −6.5  mm      ← 오른다
          최저점  0.0    0.0    0.0     0.0    0.0  mm      ← 안 오른다

      fab_test38(4000 ep 완주) 실측이 정확히 이 함정이었다: 컵 최대 상승 **+2.9 mm**,
      1 cm 이상 올린 스텝 0.0%. 즉 그 판이 한 것은 리프트가 아니라 **기울이기**였다.

    ★기존 방어는 램프 0 점을 `CUP_TIP_RISE_MAX × 1.3`(= 놓인 높이 +6 mm) 위에 두는
      것이었다(`LIFT_RAMP_ZERO_Z`). 안전하지만 **첫 6 mm 가 사구간**이라 "접촉했는데
      아직 못 든" 상태에 gradient 가 전혀 없다. DexPour(IROS 2025) 의 `r_lift` 는
      접촉만 성립하면 높이에 **선형 비례**해 즉시 지급된다(Fig. 3 의 μ·r_lift).
      최저점으로 재면 기울이기가 구조적으로 0 이므로 **사구간 없이** 그 구조를 쓸 수 있다.

    기하: 바닥 원판 중심 = 원점 − 컵축 × `CUP_BOTTOM_TO_ORIGIN`,
          그 원판의 최저점 = 중심 z − `CUP_BASE_RADIUS` × sin(기울기).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    cup_z = matrix_from_quat(obj.data.root_quat_w)[:, :, 2]        # 컵 로컬 +z (world)
    bottom_c_z = obj.data.root_pos_w[:, 2] - cup_z[:, 2] * P.CUP_BOTTOM_TO_ORIGIN
    sin_tilt = (1.0 - cup_z[:, 2].square()).clamp(min=0.0).sqrt()
    lowest_z = bottom_c_z - P.CUP_BASE_RADIUS * sin_tilt
    return lowest_z - P.TABLE_SURFACE_Z


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
    # ★★fab_test39: 원점 z 램프 → **최저점 기준**(`lift_height`). 기울이기가 리프트로
    #   계산되던 구멍을 기하로 닫는다. `ramp_zero_z` 가 막던 것과 같은 해킹인데,
    #   최저점은 기울여도 정확히 0 이라 **사구간이 필요 없다** — 첫 1 mm 부터 gradient 가 산다.
    #   근거 전문은 `lift_height` docstring. 인자 `ramp_zero_z` 는 계약 호환을 위해 남기되
    #   더 이상 램프 0 점이 아니다(상단만 `minimal_height` 에서 온다).
    lifted = (lift_height(env, object_cfg) / (minimal_height - P.CUP_SPAWN_Z)).clamp(0.0, 1.0)
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
    # ★★fab_test33: 이진 `grasp_ok` → **연속 `grasp_quality`**. 근거는 그 함수 docstring.
    held = grasp_quality(env, lat_ok, along_ok, pad_offset, jaw_cfg, object_cfg)
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
    # ★★fab_test39: 게이트를 `_held` → `grasp_quality` 로 교체했다.
    #   이 항은 **올바른 물기 자세를 만드는 유일한 gradient** 인데, 자신이 만들어야 할
    #   결과(리프트) 뒤에 갇혀 있었다. t38 4000 ep 완주 실측이 그 대가다 —
    #       grasp_pose 0.00001 · TCP z↔컵 z **49.8°**(올바름 90°) · jaw 수평이탈 **37.6°**
    #       · lateral **62.4 mm**(`grasp_ok` 문턱 30 mm) · 액션 게이트 개방률 **< 0.5%**
    #   물기가 비스듬해 게이트가 안 열리고, 게이트가 안 열려 못 들고, 못 들어서 물기를
    #   고칠 신호가 안 온다. 그 순환을 여기서 끊는다.
    #
    # ⚠ 게이트를 **그냥 제거하면 안 된다.** `jaw_level_quality` 는 로봇 `body_quat_w` 만
    #   보고 `upright` 는 컵이 테이블에 서 있기만 해도 1.0 이라, 무게이트면 **아무 데서나
    #   그리퍼를 수평으로 들고 있으면 만점**이라는 새 해킹면이 생긴다(reward-audit Check 2).
    #   `grasp_quality` 는 lateral·along·파지대역을 전부 보므로 컵 곁에서만 값이 나온다.
    gate = grasp_quality(env, lat_ok, along_ok, pad_offset, jaw_cfg, object_cfg)
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
    sensor_names: tuple[str, ...] = (),
    force_threshold: float = 0.0,
    min_upright_cos: float = -1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """리프트 보상. **게이트는 접촉, 높이는 크기.** (num_envs,) ∈ [0, 1]

    ```
    r_lift = grasp_quality × 닿은_턱_비율 × clamp(최저점_상승 / 40 mm, 0, 1)
    ```

    ★★fab_test39: 게이트를 **높이 → 접촉**으로 반전했다. DexPour(IROS 2025) Fig. 3 이
      `r_lift` 를 `μ`(전 손가락 접촉)로 게이트하고 `ν`(높이)로는 게이트하지 **않는다**:
        *"Once the cup reaches a certain height threshold, the lift reward ceases to
          accumulate"* — 높이는 보상을 **여는 하한**이 아니라 **끊는 상한**이다.
      우리는 정반대로 `_held` 안의 `lifted` 가 하한이었고, 그래서 t22~t38 열일곱 판
      내내 이 항이 0 이었다(t38 최종 **0.00005**, contact_engage 는 1.774).
      논문 Table II **Config. 4**(리프트/이송 보상 제거)가 우리와 같은 지표를 낸다 —
      η_ft 0% · P_grasp 0% · *"never discovers a stable lifting motion"*.
      우리 항은 존재하되 **도달 불가**였으므로 기능적으로 제거된 것과 같았다.

    ★접촉을 게이트로 쓰면 옛 "쳐 날리기"(test3: 리프트 판정 중 TCP–컵 3044 mm)가
      **구조적으로 불가능**해진다 — 튕겨 날아간 컵은 접촉이 끊겨 `닿은_턱_비율` = 0 이다.
      옛 방어(`near` AND)보다 강하다.

    ★높이는 `lift_height`(최저점)로 잰다. 원점 z 로 재면 기울이기가 최대 4.61 mm 의
      가짜 리프트를 만든다 — t38 의 "+2.9 mm 상승"이 바로 그것이었다.
    """
    # ⚠ 접촉 센서가 없는 태스크(관절공간 `open-grip_l_grasp_sensor`, t16 계보 positive
    #   control)에서는 구 `_held` 거동으로 돌아간다. 그쪽 씬에는 `contact_*` 센서가 아예
    #   없어서 이 함수가 KeyError 로 죽었다(fab_test42 스모크에서 실제로 터졌다).
    #   fab 태스크는 이 항을 `None` 으로 끄고 `stage_lift` 를 쓰므로 영향이 없다.
    if not sensor_names:
        return _held(env, minimal_height, ramp_zero_z, max_ee_distance, enclose_half_width,
                     pad_offset, lat_ok, along_ok, jaw_cfg, object_cfg, ee_frame_cfg,
                     min_upright_cos)
    forces = obs_mdp.finger_contact_forces(env, sensor_names)
    contact_frac = (forces > force_threshold).float().mean(dim=-1)
    quality = grasp_quality(env, lat_ok, along_ok, pad_offset, jaw_cfg, object_cfg)
    rise = (lift_height(env, object_cfg) / (minimal_height - P.CUP_SPAWN_Z)).clamp(0.0, 1.0)
    return quality * contact_frac * rise


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
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3]
    )
    distance = torch.norm(des_pos_w - obj.data.root_pos_w, dim=1)
    gate = _held(env, minimal_height, ramp_zero_z, max_ee_distance, enclose_half_width,
                 pad_offset, lat_ok, along_ok, jaw_cfg, object_cfg, ee_frame_cfg, min_upright_cos)
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
    obj: RigidObject = env.scene[object_cfg.name]
    # ★게이트 × 목표근접은 억제 항과 **같은 자**를 쓴다(중복 구현 금지 — 조용히 어긋난다).
    gate_near = held_and_near_goal(
        env, std, minimal_height, ramp_zero_z, max_ee_distance, enclose_half_width,
        pad_offset, lat_ok, along_ok, jaw_cfg, command_name,
        robot_cfg, object_cfg, ee_frame_cfg,
    )

    lin = torch.norm(obj.data.root_lin_vel_w, dim=1)
    ang = torch.norm(obj.data.root_ang_vel_w, dim=1)
    # ★선속도·각속도를 **평균**한다(곱이 아니라). 곱하면 각각 0.19 일 때 0.036 까지 떨어져
    #   lifting(11.0)에 비해 400 배 작아지고, 조금 개선해도 증가분이 다른 항에 묻힌다
    #   (test11 실측: settle 이 상한의 0.4% 에서 정체). 평균이면 같은 상태에서 5 배 크다.
    still = 0.5 * (1.0 - torch.tanh(lin / lin_vel_std)) + 0.5 * (
        1.0 - torch.tanh(ang / ang_vel_std)
    )

    return gate_near * still


def held_and_near_goal(
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
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """"제대로 들고 목표 근처에 있는가" (0~1) — `object_settled_at_goal` 의 앞 두 인수.

    ★`object_settled_at_goal` 에서 **정지 정도만 뺀** 값이다. 억제 항의 게이트로 쓰려고
      분리했다. 정지 정도를 게이트에 넣으면 "이미 멈춰 있을 때만 멈추라고 벌하는" 꼴이 돼
      정작 필요한 곳(아직 배회 중)에서 약해진다.

    ⚠ 같은 판정을 두 함수가 각자 다시 짜면 조용히 어긋난다 — 이 트랙에서 네 번 당했다.
      그래서 `object_settled_at_goal` 과 **같은 자**(_held · near_goal)를 쓴다.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, command[:, :3]
    )
    near_goal = 1.0 - torch.tanh(torch.norm(des_pos_w - obj.data.root_pos_w, dim=1) / std)
    gate = _held(env, minimal_height, ramp_zero_z, max_ee_distance, enclose_half_width,
                 pad_offset, lat_ok, along_ok, jaw_cfg, object_cfg, ee_frame_cfg)
    return gate * near_goal


def palm_command_rate_at_goal(
    env: "ManagerBasedRLEnv",
    action_term_name: str,
    rate_limit: float,
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
) -> torch.Tensor:
    """목표 근처에서 **palm 지령이 계속 배회하는 것**을 벌한다 (0~1, weight 는 음수).

    ★★fab_test19 층 분해 실측이 이 항의 근거다. 진동의 층을 나눠 재 보니:
        ① 정책 raw 액션   |Δ|0.217 |Δ²|0.205  방향반전 **18.8%**
        ② 리미터 통과 지령 |Δ|11.5mm            방향반전 8.4%
        ③ fabric 관절목표  |Δ|15.5mrad           방향반전 **0.0%**
        ④ 실제 팔 관절     |Δ|17.1mrad           방향반전 **0.0%**
      즉 액션의 **2차 성분(jerk)은 fabric 이 전부 지운다** — 팔은 떨지 않는다.
      `ActionJerkL2` 가 헛일이었던 이유가 이것이고, 그 벌금은 |Δ²a| 가 큰 접근·이송에서만
      물려 fab_test19 의 이송 학습을 무너뜨렸다(전문: grasp_left_curriculums.py).

      실제로 눈에 보이는 진동은 **1차**다 — dwell 구간에서 지령이 1.16~5.2 mm/step 씩
      계속 배회한다(초당 60~260 mm). 그래서 2차가 아니라 1차를 벌한다.

    ⚠ 게이트가 핵심이다. 목표에서 멀면 0 이므로 접근·이송의 빠른 지령은 **전혀** 벌하지
      않는다. fab_test14/19 의 실패는 억제 항이 "아직 아무것도 못 하는" 구간에 걸린 것이었고,
      이 항은 구조적으로 그 구간에 존재하지 않는다.

    ⚠ 리미터 상한으로 정규화해 0~1 로 만든다. 그래야 weight 가 "최악의 경우 몇 점"이라는
      해석 가능한 수가 되고, 상금(settle 15 · dwell 10) 대비 비율을 눈으로 검산할 수 있다.

    ⚠ 잔류 **속도**를 벌하지 않는 이유: 동결 실측에서 컵의 순간속도 바닥값이 71 mm/s 인데
      5 초 순변위는 11.7 mm(2.3 mm/s)였다. 그 71 은 서브밀리미터 솔버 버즈이지 움직임이
      아니고, 중력보상·PD damping·fabric damping 어느 것에도 불변이었다. 정책이 줄일 수
      없는 양을 벌하면 그냥 상수 벌금이다.
    """
    term = env.action_manager.get_term(action_term_name)
    moving = (term.cmd_step_norm / rate_limit).clamp(max=1.0)
    gate = held_and_near_goal(
        env, std, minimal_height, ramp_zero_z, max_ee_distance, enclose_half_width,
        pad_offset, lat_ok, along_ok, jaw_cfg, command_name,
    )
    return moving * gate


class ActionJerkL2(ManagerTermBase):
    """액션의 **2차 차분**(jerk) 제곱합 페널티. ★★기각됨 — 어디에도 배선하지 말 것.

    기각 근거(fab_test19 층 분해 실측): 액션의 2차 성분은 리미터+fabric 이 전부 흡수해
    팔 관절의 방향반전이 **정확히 0.0%** 다. 벌금은 |Δ²a| 가 큰 접근·이송에서만 물리고
    dwell 잔류에는 손도 못 댄다. fab_test19 는 이 항을 켠 뒤 dwell 1.02 → 0.005 로
    무너졌다. 대체 항은 `palm_command_rate_at_goal`(1차 · 목표 근처 게이트)이다.

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


class DwellSettledAtGoal(ManagerTermBase):
    """정지 상태를 **연속 유지**한 시간에 지급하는 보너스 (fab_test13 신설).

    ★fab_test12 실증: 순간 settle 항만으로는 목표 100 mm 옆 순회(잔류 0.22 m/s)가
      국소최적으로 굳는다 — 순간 항은 스쳐 지나가도 그 스텝만큼 지급되기 때문이다.
      순간 품질 q(`object_settled_at_goal` 그대로) > `q_thresh` 가 연속 유지된 스텝을
      세어 clamp(count/hold_steps, 0, 1) 로 지급한다. 순회는 카운터가 계속 리셋된다.

    임계·상수 근거는 preset `DWELL_*` 주석에 실측과 함께 있다.
    구현이 클래스인 이유: 연속 유지 카운터는 스텝 간 상태다(`ActionJerkL2` 와 동일 패턴).
    ⚠ `reset` 누락 금지 — 이 트랙에서 리셋 오염에 네 번 당했다.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._count = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._count[:] = 0.0
        else:
            self._count[env_ids] = 0.0

    def __call__(  # noqa: D102
        self,
        env,
        q_thresh: float,
        hold_steps: int,
        # 이하 object_settled_at_goal 로 그대로 전달 — RewardManager 의 시그니처 검사가
        # **kwargs 를 허용하지 않아 명시적으로 나열한다.
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
    ) -> torch.Tensor:
        q = object_settled_at_goal(
            env, std, lin_vel_std, ang_vel_std, minimal_height, ramp_zero_z,
            max_ee_distance, enclose_half_width, pad_offset, lat_ok, along_ok,
            jaw_cfg, command_name,
        )
        above = q > q_thresh
        self._count = torch.where(above, self._count + 1.0, torch.zeros_like(self._count))
        return (self._count / float(hold_steps)).clamp(max=1.0)


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
    axis_t_raw = (to_mid * cup_z).sum(-1, keepdim=True)
    axis_t = axis_t_raw.clamp(P.CUP_GRASP_BAND_AXIS[0], P.CUP_GRASP_BAND_AXIS[1])
    cup_pt = obj.data.root_pos_w + cup_z * axis_t
    # ★clamp **전** 축 좌표도 돌려준다 — "턱이 파지 대역 안인가"는 clamp 된 값으로는
    #   알 수 없다(밖에 있어도 경계값으로 접혀 들어온다). 접근 성공 판정에 필요하다.
    return p_l, p_r, u, mid, cup_pt, axis_t_raw.squeeze(-1)


def grasp_quality(
    env: "ManagerBasedRLEnv",
    lat_ok: float,
    along_ok: float,
    pad_offset: float,
    jaw_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """`grasp_ok` 의 **연속판** (0~1). 같은 세 측정을 쓰되 절벽을 없앤다.

    ★★fab_test33: 이 트랙의 sparse 구간을 여기서 연다. `_held()` 안의 이진 `grasp_ok` 가
      **다섯 항의 공통 목**이었다 — `lifting_object` · `object_goal_tracking(+fine)` ·
      `settled_at_goal` · `dwell_at_goal` · `grasp_pose`. 그래서 파지가 성립하기 전까지
      다섯이 **정확히 0** 이었고(t30/t31 790 epoch 실측), 접근에서 파지로 넘어가는
      지점이 절벽이었다.

    설계는 agnostic 트랙의 단계 사다리를 따른다 — 하드 게이트 대신 **연속 품질을 공통
    인자로** 곱하고, 인자가 깊어질수록 값이 작아지는 것을 감안한다. 다만 우리는 가중을
    다시 잡는 대신 **성공 기하에서 1.0 이 되도록 정규화**한다:

        q = exp(−lateral/lat_ok) · exp(−along/along_ok) · band_q
        G = clamp(q / GRASP_QUALITY_REF, 0, 1)

    그러면 성공 지점의 보상 크기는 **지금과 정확히 같고**(가중 재조정 불필요) 그 아래로만
    연속 기울기가 생긴다. 실측 기하로 검산:

        성공 (lateral 20.0 · along 13.0 mm)  q = 0.333 → G = **1.00**
        주먹 (lateral 78.6 · along 27.8 mm)  q = 0.029 → G = 0.086
        컵 옆(lateral 85.5 · along 12.0 mm)  q = 0.039 → G = 0.116

    ⚠ 던지기 재발 우려는 없다. `_held` 는 G 외에 `near`(TCP 80 mm 이내)·`upright` 를
      **여전히 게이트로** 곱한다 — 쳐 날린 컵은 즉시 near 가 거짓이라 0 이다
      (test3 사고의 차단 장치는 그대로 남는다).
    ⚠ band 는 부드럽게 — 대역 밖으로 나간 거리만큼 지수 감쇠한다. 하드 판정이면
      대역 경계에서 다시 절벽이 생긴다.
    """
    _l, _r, u, mid, cup_pt, axis_t = _jaw_frame(env, pad_offset, jaw_cfg, object_cfg)
    d = cup_pt - mid
    along = (d * u).sum(-1).abs()
    lateral = (d - u * (d * u).sum(-1, keepdim=True)).norm(dim=-1)
    lo, hi = P.CUP_GRASP_BAND_AXIS
    out = (lo - axis_t).clamp(min=0.0) + (axis_t - hi).clamp(min=0.0)
    q = (torch.exp(-lateral / lat_ok) * torch.exp(-along / along_ok)
         * torch.exp(-out / P.GRASP_BAND_SOFT_TAU))
    return (q / P.GRASP_QUALITY_REF).clamp(max=1.0)


def grasp_ok(
    env: "ManagerBasedRLEnv",
    lat_ok: float,
    along_ok: float,
    pad_offset: float,
    jaw_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
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
    _p_l, _p_r, u, mid, cup_pt, axis_t = _jaw_frame(env, pad_offset, jaw_cfg, object_cfg)
    d = cup_pt - mid
    along = (d * u).sum(-1).abs()
    lateral = (d - u * (d * u).sum(-1, keepdim=True)).norm(dim=-1)
    in_band = (axis_t > P.CUP_GRASP_BAND_AXIS[0]) & (axis_t < P.CUP_GRASP_BAND_AXIS[1])
    return (lateral < lat_ok) & (along < along_ok) & in_band


def jaw_lateral(
    env: "ManagerBasedRLEnv",
    pad_offset: float,
    jaw_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """턱 축 직선에서 컵 축까지의 수직거리 (m). 래치 **해제** 판정에 쓴다."""
    _p_l, _p_r, u, mid, cup_pt, _axis = _jaw_frame(env, pad_offset, jaw_cfg, object_cfg)
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
):
    """턱 ↔ 컵 기하 (align, enclose). 두 보상 항이 **같은 기하**를 쓰도록 여기 모은다.

    · align   = 평균( 턱 축 방향 정렬, 턱 축 직선까지의 근접 )  — 곱하지 않는다
    · enclose = 두 손가락이 컵 축 **양쪽**에 있는가 (0~1)
    """
    p_l, p_r, u, mid, cup_pt, _axis = _jaw_frame(env, pad_offset, robot_cfg, object_cfg)
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


_AXIS = {"x": 0, "y": 1, "z": 2}


# ★fab_test46: `gripper_gate_open`/`gripper_gate_rate` 제거 — 그리퍼 하드 게이트 폐기
#   (근거는 env cfg 의 fab_test46 주석). 게이트 상태라는 개념 자체가 사라졌다.

def _jaw_mid_local(env, pad_offset: float, jaw_cfg: SceneEntityCfg) -> torch.Tensor:
    """턱 중점(패드 중앙 보정 포함), env 로컬. `_jaw_frame` 과 같은 자다."""
    robot: Articulation = env.scene[jaw_cfg.name]
    pos = robot.data.body_pos_w[:, jaw_cfg.body_ids, :]
    approach = matrix_from_quat(robot.data.body_quat_w[:, jaw_cfg.body_ids[0], :])[:, :, 2]
    pos = pos + (approach * pad_offset).unsqueeze(1)
    return pos.mean(dim=1) - env.scene.env_origins


def diag_palm_cmd(env, action_term_name: str, axis: str) -> torch.Tensor:
    """정책이 낸 palm **지령** 위치 성분 (m, env 로컬)."""
    return env.action_manager.get_term(action_term_name)._palm_pose_target[:, _AXIS[axis]]


def diag_jaw_pos(env, axis: str, pad_offset: float, jaw_cfg: SceneEntityCfg) -> torch.Tensor:
    """**실제** 턱 중점 위치 성분 (m, env 로컬)."""
    return _jaw_mid_local(env, pad_offset, jaw_cfg)[:, _AXIS[axis]]


def diag_cmd_jaw_gap(env, action_term_name: str, pad_offset: float,
                     jaw_cfg: SceneEntityCfg) -> torch.Tensor:
    """지령과 실제의 거리 (m) = **추종 오차**.

    ★이 값이 커지면 정책이 낸 지령을 팔이 못 따라가고 있다는 뜻이고, 그러면 정책의
      액션과 환경 응답의 대응이 끊긴다. 08.25 실측 90 mm 가 그 상태였다.
    """
    cmd = env.action_manager.get_term(action_term_name)._palm_pose_target[:, :3]
    return (cmd - _jaw_mid_local(env, pad_offset, jaw_cfg)).norm(dim=-1)


def diag_cmd_step(env, action_term_name: str) -> torch.Tensor:
    """스텝당 지령 이동량 (m) — **리미터가 실제로 무는지**를 이걸로 확인한다."""
    return env.action_manager.get_term(action_term_name).cmd_step_norm


def diag_jaw_cup_dist(env, pad_offset: float, jaw_cfg: SceneEntityCfg,
                      object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """턱 중점 ↔ 컵 원점 거리 (m) — shaping 안 거친 **날 거리**.

    `reaching_object` 는 커널을 통과한 값이라 "얼마나 가까운가"를 못 읽는다.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    cup = obj.data.root_pos_w - env.scene.env_origins
    return (_jaw_mid_local(env, pad_offset, jaw_cfg) - cup).norm(dim=-1)


def diag_duty(env) -> torch.Tensor:
    """상수 1.0 — **정규화 분모를 측정한다.**

    ★TB 의 `Episode_Reward/<name>` 은 `(Σ raw·dt)/episode_length_s` 라, 에피소드가 만기
      전에 끝나면 그 비율만큼 작게 찍힌다. 위치를 m 로 읽으려면 그 비율을 나눠야 하는데,
      `episode_lengths/iter`(rl_games)는 **다른 집합에서 평균된 값**이라 안 맞는다
      (실측: 그걸로 정규화하니 `diag_cmd_step` 이 상한 0.10 을 넘는 0.146 이 나왔다).
    이 항은 raw=1 이므로 로깅값이 정확히 `에피소드 길이 / episode_length_s` 다.
    다른 diag 값을 **이걸로 나누면** 단위가 정확히 복원된다 — 추정이 아니라 측정이다.
    """
    return torch.ones(env.num_envs, device=env.device)


def tcp_x_level_quality(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "l_hl_gripper_base",
    power: float = 4.0,
) -> torch.Tensor:
    """TCP 로컬 **+x 축이 world +z 와 수직**인 정도 (0~1). 1 = 완전 수평.

    사용자 지시(08.25): "접근할 때부터 tcp_+x 가 world +z 와 수직이 되게 접근해야 한다."

    품질 = (1 − |x축의 z성분|)^power. |cos| 이 아니라 **sin 의 거듭제곱**이라
    수직 근처에서 평평하지 않고 기울수록 빠르게 떨어진다(cos^4 규약은 자매 트랙 교훈).

    ⚠ **게이트로 쓰지 말 것.** 이 태스크에서 자세를 AND 게이트로 넣었다가 학습이
      시작조차 못 한 적이 있다(test6·test7: lifting 0.0000 · 총보상 −0.46 · 에피소드
      130→13). 양의 보상이 전부 0 이 되면 조기 종료가 최적이 된다.
      접근 보상에 **곱하는 연속 배수**로만 쓴다 — 0 이 되지 않도록 floor 를 둔다.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    idx = robot.body_names.index(body_name)
    w, x, y, z = robot.data.body_quat_w[:, idx, :].unbind(-1)
    # 회전행렬의 (2,0) 성분 = 로컬 x 축의 world z 성분
    x_axis_z = 2.0 * (x * z - w * y)
    return (1.0 - x_axis_z.abs()).clamp(min=0.0) ** power


def reach_with_tcp_level(
    env: "ManagerBasedRLEnv",
    std: float,
    grasp_offset: float,
    level_floor: float = 0.25,
    level_power: float = 4.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """접근 보상 × TCP 수평 품질. 자세를 **접근 단계부터** 유도한다.

    배수 = floor + (1 − floor) · quality  → 최악이어도 `level_floor` 는 남는다.
    floor 를 두는 이유는 위 docstring 의 test6/test7 사고 때문이다 — 접근 보상은
    초기에 **유일하게 살아 있는 신호**라 0 이 되면 학습이 시작되지 않는다.
    """
    reach = ee_grasp_point_distance(env, std=std, grasp_offset=grasp_offset)
    q = tcp_x_level_quality(env, robot_cfg=robot_cfg, power=level_power)
    return reach * (level_floor + (1.0 - level_floor) * q)


# ═══════════════════════════════════════════════════════════════════════════
# 접근 보상 — agnostic/tasks/grasp_sensor 이식 (사용자 지시 08.25)
# ═══════════════════════════════════════════════════════════════════════════


def approach_opposed(
    env: "ManagerBasedRLEnv",
    sharpness: float,
    side_radius: float,
    grasp_offset: float,
    pad_offset: float,
    jaw_cfg: SceneEntityCfg,
    palm_body: str = "l_hl_gripper_base",
    side_weight_a: float = 0.5,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """`exp(−s·(d_palm + d_side))` — agnostic 트랙 `approach_reward` 이식.

    ★★왜 옮겼나. 우리 접근 보상은 `1 − tanh(d/0.1)` 로 **거리만** 봤다. 그래서
      "컵 근처에 있기"만 하면 만점에 가까웠고, 턱이 컵을 어떻게 감싸는지는 아무 압력도
      받지 않았다. 실측(fab_test31): 정책이 목표 자리(y 0.321 = 목표 y 0.320)에 머물며
      턱–컵 거리를 166 → 234 mm 로 **벌리는데도** 총보상이 유지됐다.

    이식본은 거리와 **프리그래스프 기하**를 한 지수 안에서 함께 본다:
        d_palm  = |palm − 파지중심|
        d_side  = 턱들이 파지중심 양옆 **대향점**(중심 ± n·side_radius)에 얼마나 가까운가
                  n = 접근 방향(palm→중심)의 xy 수직 — 즉 **턱 축이 컵을 가로질러야** 작아진다
    자세를 각도 배수로 유도하던 방식(제가 넣었다가 reaching 을 1/5 로 떨어뜨린 것)을
    대체한다 — 자세가 **기하로** 강제되므로 별도 배수가 필요 없다.

    ⚠ 원본과 한 곳 다르다. 원본은 `d_side = 0.6·d_a + 0.4·d_b` 로 엄지 쪽에 가중을 준다
      (5 지 손의 엄지 1 대 4 지 4 비대칭 때문). **우리 두 턱은 대칭**이라 0.5/0.5 로 둔다 —
      한쪽에 가중을 주면 그 턱만 맞추고 반대쪽이 벌어지는 해가 생긴다.
    ⚠ n 의 부호는 **현재 A 턱이 있는 쪽**으로 동적 선택한다(원본과 동일). 고정 부호를
      쓰면 좌우 미러에서 감쌈이 뒤집힌다.
    """
    robot: Articulation = env.scene[jaw_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    origin = env.scene.env_origins

    # 파지중심 = 컵 원점 + 파지대역 오프셋 (env 로컬)
    grasp_center = obj.data.root_pos_w - origin
    grasp_center = grasp_center + torch.tensor(
        [0.0, 0.0, grasp_offset], device=grasp_center.device)

    # 턱 위치 — 패드 중앙 보정 포함(보상 전체가 같은 자를 쓴다)
    tips = robot.data.body_pos_w[:, jaw_cfg.body_ids, :]
    approach_axis = matrix_from_quat(
        robot.data.body_quat_w[:, jaw_cfg.body_ids[0], :])[:, :, 2]
    tips = tips + (approach_axis * pad_offset).unsqueeze(1) - origin.unsqueeze(1)

    palm_idx = robot.body_names.index(palm_body)
    palm_pos = robot.data.body_pos_w[:, palm_idx, :] - origin

    d_palm = torch.norm(palm_pos - grasp_center, dim=-1)

    a_xy = palm_pos[:, :2] - grasp_center[:, :2]
    a_xy = a_xy / a_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    n = torch.stack([-a_xy[:, 1], a_xy[:, 0], torch.zeros_like(a_xy[:, 0])], dim=-1)
    sign = torch.sign(((tips[:, 0] - grasp_center) * n).sum(dim=-1, keepdim=True))
    n = n * torch.where(sign == 0, torch.ones_like(sign), sign)

    target_a = grasp_center + n * side_radius
    target_b = grasp_center - n * side_radius
    d_a = torch.norm(tips[:, 0] - target_a, dim=-1)
    d_b = torch.norm(tips[:, 1] - target_b, dim=-1)
    d_side = side_weight_a * d_a + (1.0 - side_weight_a) * d_b

    return torch.exp(-sharpness * (d_palm + d_side))


# ═══════════════════════════════════════════════════════════════════════════
# 접촉 보상 — DexPour(IROS 2025) `r_contact` 이식 (fab_test38)
# ═══════════════════════════════════════════════════════════════════════════


def contact_engage(
    env: "ManagerBasedRLEnv",
    sensor_names: tuple[str, ...],
    force_threshold: float,
    all_bonus: float,
) -> torch.Tensor:
    """턱이 컵에 **닿는 것 자체**에 값을 매긴다. `f + all_bonus·[f == 1]`, f = 닿은 턱 비율.

    ★★t22~t37 열세 판이 전부 `lifting_object` 정확히 0 이었고, 공통 서명은 `drop` 이
      ep50 안에 0.000 으로 죽는 것이었다. 로그 전수(아카이브 23 런 + 오늘 10 런):
          drop(ep50-200) ≥ 0.02  →  리프트한 10 개 전부
          drop(ep50-200) < 0.02  →  9 개 전부 lifting 0
      성공한 런들은 수백 epoch 컵을 10~50% 넘어뜨리며 돌았다 — 그 실패가 곧 탐색이었다.
      오늘 판들은 컵을 만지지 않으니 파지를 찾을 **표본 자체가 없다.**

    왜 컵을 안 만지나 — 만져서 얻는 것이 **아무것도 없기 때문이다.** 낙하는 페널티가
    아니라 종료라 위험만 있고, 파지 계열 보상은 `grasp_quality` 를 지나야 하는데 거기
    도달하려면 이미 잘 잡고 있어야 한다. DexPour 의 ablation 이 같은 실패를 기록한다 —
    Config.2(전 보상·커리큘럼 없음)가 *"avoiding cup movement to minimize penalties"* 로
    조기 수렴해 파지 성공률 0% 다. 그 해법이 이 항이다(논문 III-A Stage 2 `r_contact`).

    ⚠ **`all_bonus`(양 턱 동시 접촉)는 현재 도달 불가**다. 컵 파지대역 단면이 58 mm 인데
      액션 게이트가 `grasp_ok` 전에는 그리퍼를 84.5 mm 로 강제 개방한다 — 닫히지 않으면
      두 턱이 동시에 닿을 수 없다. 순환이다. 이번 판은 **한 턱 접촉(f=0.5) 경로만** 열고,
      게이트 연속화는 다음 단계로 미룬다. 보너스 항은 그때를 위해 배선만 해 둔다.
    ⚠ 센서 자체가 08.26 까지 죽어 있었다(`force_matrix_w` 최대까지 정확히 0). 필터가
      `/Object`(프림 루트)를 가리켜 PhysX 가 GPU 접촉 필터를 못 만들었고, 시뮬레이터가
      env 마다 경고를 찍는데 로그가 길어 묻혔다. `/Object/baseLink` 로 고친 뒤에야
      이 항이 의미를 갖는다 — 그 전에 넣었으면 상시 0 인 죽은 항이었다.
    """
    forces = obs_mdp.finger_contact_forces(env, sensor_names)      # (N, F)
    touch = (forces > force_threshold).float()
    frac = touch.mean(dim=-1)
    return frac + all_bonus * (frac >= 1.0).float()


def diag_lift_height(
    env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """★진단(weight 0.001) — 컵 **최저점** 상승 [m]. 기울여도 0 이라 가짜 리프트가 안 섞인다.

    t38 은 원점 기준으로 최대 +2.9 mm 였는데 그건 기울기였다. 이 값이 0 을 벗어나야
    진짜로 든 것이다. `lifting_object` 가 0 일 때 원인이 "안 들었다"인지 "게이트가
    막았다"인지를 이 항 하나로 가른다.
    """
    return lift_height(env, object_cfg)


def diag_jaw_lateral(
    env: "ManagerBasedRLEnv", pad_offset: float, jaw_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """★진단(weight 0.001) — 턱축까지의 **수직거리** lateral [m]. `grasp_ok` 의 1차 조건.

    t38 결정론 실측 62.4 mm(최선 27.4) vs 문턱 `GRASP_GATE_LATERAL_OK` 30 mm.
    액션 게이트 개방률이 0.5% 였던 직접 원인이고, D2(`grasp_pose` 게이트 교체)가
    겨냥하는 값이다. TB 에 없어서 t38 내내 사후 프로브로만 볼 수 있었다.
    """
    return jaw_lateral(env, pad_offset, jaw_cfg, object_cfg)


# ═══════════════════════════════════════════════════════════════════════════
# DexPour 계층 보상 (fab_test41) — 5단계 사다리
#
# 논문 Fig. 3:  r_t = (1−λ)·p + μ·r_grasping + μ·r_lift + ν·r_transporting + ρ·r_pouring
# 우리 5단계(사용자 규격): approach/align → grasp → lift → transfer → stay
#
# ★단계 상태(λ/μ/ν/ρ · 진척량)는 `grasp_left_stages.compute` 가 **스텝당 한 번** 계산해
#   캐시한다. 항마다 재계산하면 자를 일곱 개 두는 것이고, 이 트랙은 그렇게 조용히
#   어긋난 사고를 이미 겪었다(패드 중앙 보정 · 컵 축 clamp).
# ★가중은 preset 에서 오고 **단조 증가**한다(2→1→3→5→7→10). 근거는 그쪽 주석.
# ★TB 태그 = 보상 슬롯 이름이므로 함수 이름·슬롯 이름·태그가 전부 일치한다.
# ═══════════════════════════════════════════════════════════════════════════


def _stage(env, jaw_cfg: SceneEntityCfg, sensor_names: tuple[str, ...]):
    from . import grasp_left_stages as stages
    return stages.compute(env, jaw_cfg, sensor_names)


def stage_approach(
    env: "ManagerBasedRLEnv", jaw_cfg: SceneEntityCfg, sensor_names: tuple[str, ...],
) -> torch.Tensor:
    """① 접근 — **원본 lift 형태의 순수 거리 양수 커널** (fab_test50, 사용자 결정).

        approach = 1 − tanh( d(턱중점 → 컵 파지점) / 0.1 )

    세 보상 체계를 실측으로 소거한 끝의 복귀다:
      · t42 양수+곱셈인자  → orient 지렛대(6.7배) 파밍. **죄는 양수가 아니라 곱셈 인자**
      · t44~48 절대 벌점   → critic 조건부 advantage 순환 + σ 조기 붕괴(반감기 ep90~200)
      · t49 PBRS 차분      → 정지가 보상 중립이라 "컵 앞 무한 대기"가 최적
    원본 커널은 **가까이 서 있는 상태 자체를 매 스텝 지급**해 대기 자세가 인력이 되고,
    인자가 거리 하나뿐이라 우회 파밍이 없다(t16/t17 관절공간판이 같은 씬에서 이 형태로
    파지·리프트까지 성공한 실증). 30 mm 호버(0.74/step)는 사다리 lift(5)~stay(10)에
    언제든 역전된다.

    ★기준점 = **컵 원점** (fab_test51, 사용자 결정). 초안은 파지점(원점 −44.6 mm =
      실측 대역 중앙)이었으나 기각 — std 0.1 커널에 44.6 mm 는 미세 조준이 아니라
      **바닥 쪽 바이어스**다(테이블 위 47 mm 를 가리켜 테이블 근접 위험만 키운다).
      미세 z 는 contact/grasp 품질(파지대역 인코딩 유지)이 사다리에서 찾는다.
    ⚠ 자세(perp)·축분해 가중은 커널에 넣지 않는다 — t42 의 죄를 되살리는 길이다.
      자세는 euler 중심(수평 재센터) + 회전 리미터가, 축 정밀도는 contact 가 맡는다.
    """
    s = _stage(env, jaw_cfg, sensor_names)
    return 1.0 - torch.tanh(s.d_jaw_cup / P.APPROACH_KERNEL_STD)


def stage_tip(
    env: "ManagerBasedRLEnv", jaw_cfg: SceneEntityCfg, sensor_names: tuple[str, ...],
) -> torch.Tensor:
    """전도 — **벌점 크기(양수)**. 구 `object_tipped` **종료를 대체**한다.

    ★★종료였을 때 무슨 일이 있었나(t42 실측): 컵 앞에 있으면 tipped 0.82 · 에피소드 83
      스텝 · 리턴 0.72. 물러서면 tipped 0.00 · 300 스텝 · 리턴 2.95. **도망이 4.1배**였고
      정책은 정확히 그렇게 했다. 종료는 "조심해서 잡아라"가 아니라 "근처에 가지 마라"를
      가르친다. 벌점으로 바꾸면 미래 보상이 안 끊기므로 도망칠 이유가 사라진다.
    ★리프트 후에는 `(1−ν)` 로 끈다 — 이송 중 기울기는 `U_tol`·`U_up` 이 맡는다.
      상시 걸면 사용자 규격("15° 이내면 됨")과 어긋난다.
    ★크기: 60° 에서 1.5 로 잡아 **물러서 있기(≈0.85)보다 나쁘게** 한다. 그래야
      "쓰러뜨리기 < 물러서기 < 제대로 잡기" 순서가 선다.
    """
    s = _stage(env, jaw_cfg, sensor_names)
    excess = (s.tilt_deg - P.STAGE_TIP_MARGIN_DEG).clamp(min=0.0)
    return (P.STAGE_TIP_PER_DEG * excess).clamp(max=P.STAGE_TIP_PENALTY_MAX) * (1.0 - s.nu)


def stage_contact(
    env: "ManagerBasedRLEnv", jaw_cfg: SceneEntityCfg, sensor_names: tuple[str, ...]
) -> torch.Tensor:
    """② 접촉 — 무게이트, **파지 기하를 곱한다**: `닿은턱비율 × grasp_quality`.

    ★★fab_test43 에서 기하 곱을 넣었다. approach 가 벌점이 되면 스텝당 값이 작아지는데
      (정체점 −0.85 · 컵 앞 −0.1), 구 무게이트 `touch_frac` 은 **손끝 하나로 컵 옆구리를
      누르기만 해도 +0.5** 라 제대로 자리잡고 안 닿은 것보다 네 배 나았다. 그리고 그
      행동은 컵을 쓰러뜨린다. 실측 기하로 검산:
          컵 옆구리(lateral 85.5 mm) grasp_quality 0.116 → 0.5 × 0.116 = **0.058**
          제대로 감쌈(lateral 20 · along 13 mm) 1.00     → 1.0 × 1.00  = **1.00**
      옆구리 찌르기가 17배 죽는다.
    ★무게이트를 유지하는 이유(λ=1·μ=0 사각지대)는 그대로다. 다만 그 사각지대는 이제
      거리 벌점이 s→46.9 mm 까지 **단조**로 이어져 이미 메워져 있다.
    """
    s = _stage(env, jaw_cfg, sensor_names)
    # ★fab_test52: × U_perp — 기울인 접촉은 지급이 죽는다(30°에서 0 ← 15°에서 1).
    #   사용자 규격 "lift 전까지 TCP z ⊥ world z". 게이트라 양수 흐름을 안 만든다.
    return s.grasp_q * s.U_perp


def stage_grasp(
    env: "ManagerBasedRLEnv", jaw_cfg: SceneEntityCfg, sensor_names: tuple[str, ...]
) -> torch.Tensor:
    """③ 파지 — `μ · (파지기하 × 닿은턱비율)`.

    ★사용자 규격 "가까이 가면 그리퍼 끝단을 **동시에** 닫기 시작". `μ` 가 그것이다 —
      두 턱이 **모두** 접촉해야 열린다.
    ★★품질에 **접촉을 곱한다.** 구 `cup_between_jaws`·`grip_closure_when_enclosed` 는
      `enclose` 가 턱축 **투영**만 봐서 접촉 없이도 만점이었고, t38 이 그걸로
      **170 mm 허공에서 closure 를 상한의 74%(1.85/2.50)** 까지 받았다. 그 구간
      `contact_engage` 는 정확히 0 이었다. 같은 맹점에 네 번 속았다
      (fab_test11 85.5 mm 에 0.824 / t32 195 mm 에 0.852 / t38 closure).
      접촉을 곱하면 그 해킹이 **구조적으로 불가능**해진다.
    """
    s = _stage(env, jaw_cfg, sensor_names)
    return s.mu * s.grasp_q


def stage_lift(
    env: "ManagerBasedRLEnv", jaw_cfg: SceneEntityCfg, sensor_names: tuple[str, ...]
) -> torch.Tensor:
    """④ 리프트 — `μ · U_tol · clamp(최저점상승 / 40 mm)`.

    ★★게이트가 **μ(접촉)** 이지 높이가 아니다. 논문 Fig. 3 이 Transporting 을
      `μ·r_lift + ν·r_transporting` 로 쪼개 놓았고, 본문이 *"the lift reward ceases to
      accumulate"* 라고 못 박는다 — 높이는 **여는 하한이 아니라 끊는 상한**이다.
      우리 구 `_held` 는 높이가 하한이라 이 항이 열아홉 판 내내 0 이었다.
      접촉만 성립하면 **1 mm 만 올려도 지급**되고 40 mm 에서 포화한다.
    ★높이는 `lift_height`(컵 **최저점**)로 잰다. 원점 z 로 재면 컵을 바닥 림으로
      피벗시켜 최대 4.61 mm 를 위조할 수 있다 — t38 의 "+2.9 mm 상승"이 그것이었다.
    ★사용자 규격 "그 자세로" = `U_tol`(기울기 25°→15° 전이). 20° 정도는 관용한다.
    """
    s = _stage(env, jaw_cfg, sensor_names)
    return s.mu * s.U_tol * s.H


def stage_transfer(
    env: "ManagerBasedRLEnv", jaw_cfg: SceneEntityCfg, sensor_names: tuple[str, ...]
) -> torch.Tensor:
    """⑤ 이송 — `ν · U_tol · exp(−d_goal / std)`.

    ★게이트가 `ν`(= μ 이고 리프트 성립)다. 논문 식 5 그대로 — 이송은 실제로 들린
      뒤에만 의미가 있다. 사용자 규격 "이때도 cup+z 가 최대한 world+z 와 같은 방향"
      = `U_tol`.
    """
    s = _stage(env, jaw_cfg, sensor_names)
    return s.nu * s.U_tol * s.T


def stage_stay(
    env: "ManagerBasedRLEnv", jaw_cfg: SceneEntityCfg, sensor_names: tuple[str, ...]
) -> torch.Tensor:
    """⑥ 정지 — `ρ · exp(−컵속도 / 0.05) · U_up`.

    ★사용자 규격 "팔을 가만히 (목표 지점 **5cm 이내**)" + "cup+z 와 world+z 15도 이내".
      `ρ` 가 목표 근접, `S` 가 정지, `U_up`(15°→5°)이 직립이다. **셋을 다 요구한다** —
      구 `settled_at_goal` 은 직립 인자가 없어 기울인 채로도 성립했다.
    ★정지를 **컵 속도**로 잰다. 액션 변화량으로 재면 "액션을 안 바꾼다"이지
      "안 움직인다"가 아니다(자매 트랙이 같은 실수를 고쳤다).
    """
    s = _stage(env, jaw_cfg, sensor_names)
    return s.rho * s.S * s.U_up


# ── 단계 진단 (weight 0.001) — TB 값 매칭용 ────────────────────────────────
# ⚠ IsaacLab 은 `weight == 0` 항을 건너뛴다(호스트 패치 전제이지만 0.001 이 안전하다).
#   총보상 대비 1e-4 수준이라 학습에 무영향이면서 단계 진입률을 TB 에서 읽을 수 있다.
def _mk_diag(attr: str):
    def _f(env, jaw_cfg: SceneEntityCfg, sensor_names: tuple[str, ...]) -> torch.Tensor:
        v = getattr(_stage(env, jaw_cfg, sensor_names), attr)
        return v.float() if v.dtype != torch.float32 else v
    _f.__name__ = f"stage_diag_{attr}"
    _f.__doc__ = f"★진단 — 단계 상태 `{attr}`. λ/μ/ν/ρ 는 각 단계 진입률이다."
    return _f


stage_diag_lam = _mk_diag("lam")
stage_diag_mu = _mk_diag("mu")
stage_diag_nu = _mk_diag("nu")
stage_diag_rho = _mk_diag("rho")
stage_diag_tilt_deg = _mk_diag("tilt_deg")
stage_diag_perp_q = _mk_diag("perp_q")
stage_diag_d_goal = _mk_diag("d_goal")
# ★사용자 규격 `BASE — CUP(xy) — TCP` 를 TB 에서 직접 본다.
#   `enter_s` 가 (0, 80) mm 창 안으로 들어와 46.9 mm 로 수렴하면 순서가 성립한 것이다.
stage_diag_enter_s = _mk_diag("enter_s")
stage_diag_jaw_l = _mk_diag("jaw_l")
stage_diag_height_h = _mk_diag("height_h")


def stage_palm_cmd_rate(
    env: "ManagerBasedRLEnv", jaw_cfg: SceneEntityCfg, sensor_names: tuple[str, ...],
    action_term_name: str, rate_limit: float,
) -> torch.Tensor:
    """목표에서 **팜 지령이 계속 배회하는 것**을 벌한다 (0~1, weight 는 음수).

    ★fab_test41: 게이트를 `held_and_near_goal`(구 높이 게이트) → **`ρ`** 로 옮겼다.
      DexPour 사다리의 ρ 가 정확히 "리프트해서 목표 5cm 안에 왔다"이고, 두 함수가 서로
      다른 자를 쓰면 조용히 어긋난다(이 트랙이 반복한 사고: 패드 중앙 보정 · 컵 축 clamp).

    ★재는 층이 `action_rate`·`joint_vel` 과 다르다. fab_test19 층 분해 실측:
        ① 정책 raw 액션   |Δ|0.217 |Δ²|0.205  방향반전 18.8%   ← action_rate 가 보는 층
        ② 리미터 통과 지령 |Δ|11.5mm            방향반전  8.4%   ← **이 항이 보는 층**
        ③ fabric 관절목표  |Δ|15.5mrad          방향반전  0.0%
        ④ 실제 팔 관절     |Δ|17.1mrad          방향반전  0.0%   ← joint_vel 이 보는 층
      눈에 보이는 배회는 ②의 **1차** 성분이다(dwell 구간 1.16~5.2 mm/step = 초당 60~260 mm).
      ①은 σ 노이즈가 지배해 "정책을 부드럽게" 대신 "σ 를 줄이기"로 최적화되고,
      ④는 fabric 이 이미 평활화한 뒤라 남는 신호가 없다.

    ⚠ 게이트가 핵심이다. 목표에서 멀면 0 이므로 접근·이송의 빠른 지령은 **전혀** 안 벌한다.
      이 트랙은 억제 항을 "아직 아무것도 못 하는" 구간에 걸어 두 번 학습을 죽였다
      (fab_test14 jerk · fab_test19). [[suppression-terms-need-task-first]]
    ⚠ 잔류 **속도**는 안 벌한다. 동결 실측에서 컵 순간속도 바닥이 71 mm/s 인데 5 초 순변위는
      11.7 mm(2.3 mm/s)였다 — 그 71 은 서브밀리미터 솔버 버즈이고 게인·damping 어느 것에도
      불변이었다. 정책이 못 줄이는 양을 벌하면 그냥 상수 벌금이다.
    """
    term = env.action_manager.get_term(action_term_name)
    moving = (term.cmd_step_norm / rate_limit).clamp(max=1.0)
    return moving * _stage(env, jaw_cfg, sensor_names).rho

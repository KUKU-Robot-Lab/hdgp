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

"""DexPour 계층 보상의 **단계 상태** — λ/μ/ν/ρ 와 각 단계의 진척량.

논문(IROS 2025) Fig. 3 의 총보상:

    r_t = (1−λ)·p_penalties + μ·r_grasping + μ·r_lift + ν·r_transporting + ρ·r_pouring
          └ Approaching ┘   └ Grasping ┘   └───── Transporting ─────┘   └ Pouring ┘

    λ = 1     if dist_hand_cup < d_approach                                (식 3)
    μ = λ     if c_contact == c_finger                                     (식 4)
    ν = λ·μ   if height_cup ≥ h_lift                                       (식 5)
    ρ = λ·μ·ν if dist_cup_target < d_pour                                  (식 6)

★★`r_lift` 의 게이트가 **μ(접촉)** 이지 ν(높이)가 아니다. Fig. 3 에서 Transporting 이
  `μ·r_lift + ν·r_transporting` 로 쪼개져 있고, 본문이 못 박는다 — *"Once the cup
  reaches a certain height threshold, the lift reward **ceases to accumulate**"*.
  즉 높이는 보상을 **여는 하한이 아니라 끊는 상한**이다. 우리 구 `_held` 는 높이가
  하한이었고, 그래서 t22~t40 열아홉 판 내내 리프트 항이 0 이었다(t40 최종 0.00002).

⚠ **논문 구조의 사각지대**: `r_contact`("접촉점당 소액")가 `μ·r_grasping` 안에 있어
  4지가 다 닿기 전에는 죽어 있다 — λ=1·μ=0 구간의 보상이 0 이다. 이 트랙은 **보상이 0 이면
  조기 종료가 최적**이 되는 실패를 이미 겪었다(test6/test7: lifting 6.14 → 0.0000,
  에피소드 130 → 13, 총보상 +34.9 → −0.46). 자매 트랙 `agnostic/tasks/grasp_sensor` 가
  같은 사각지대를 실측으로 발견해 **무게이트 shaping**(`stage_contact_weight = 1.0
  # 게이트 없음 — λ=1·μ=0 사각지대 방지`)으로 고쳤다. 논문 원본이 아니라 그 고친 쪽을
  이식한다 — `approach` 와 `contact` 는 게이트 밖에 둔다.

★단계 상태를 **스텝당 한 번만** 계산해 캐시한다. 일곱 항이 각자 재계산하면 자를 일곱 개
  두는 것이고, 이 트랙은 그렇게 조용히 어긋난 사고를 이미 겪었다(패드 중앙 보정 · 컵 축
  clamp). 캐시 키는 `env.common_step_counter` 다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat

from . import grasp_left_preset as P

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def smoothstep(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """`lo` 에서 0, `hi` 에서 1 로 부드럽게 (lo > hi 이면 감소 방향).

    자매 트랙 `agnostic/tasks/grasp_sensor/rewards_tip_cyl.smoothstep` 과 같은 식이다.
    이진 AND 게이트는 이 트랙에서 학습을 죽인 이력이 있으므로(test6/test7) 자세·정지
    조건은 전부 이 전이로 낸다.
    """
    t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class StageState:
    """한 스텝의 단계 상태. 보상 항들이 공유한다."""

    __slots__ = (
        "lam", "mu", "nu", "rho",           # 논문 식 3~6 트리거
        "d_jaw_cup", "d_goal", "tilt_deg",  # 원시 측정값
        "touch_frac", "lift_h", "cup_speed", "xy_disp",
        "U_tol", "U_up", "H", "T", "S",     # 단계 진척량
        "perp_q", "align_q", "grasp_q",
        "enter_s", "jaw_l", "height_h",     # gripper_base 프레임 3축 분해
        "d_jaw_grasp",                      # 턱중점 ↔ 컵 파지점 유클리드 거리 (fab_test50)
    )


_CACHE: dict[int, tuple[int, StageState]] = {}
_SPAWN: dict[int, torch.Tensor] = {}
_BASE_ID: dict[int, int] = {}


def _cup_axis_and_tilt(obj: RigidObject) -> tuple[torch.Tensor, torch.Tensor]:
    """컵 로컬 +z 의 world z 성분과 기울기[deg]."""
    cup_z = matrix_from_quat(obj.data.root_quat_w)[:, :, 2]
    cos_up = cup_z[:, 2].clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return cup_z, torch.rad2deg(torch.acos(cos_up))


def compute(env: "ManagerBasedRLEnv", jaw_cfg: SceneEntityCfg,
            sensor_names: tuple[str, ...]) -> StageState:
    """스텝당 한 번. 같은 스텝에 두 번째 호출부터는 캐시를 준다."""
    key = id(env)
    step = int(env.common_step_counter)
    hit = _CACHE.get(key)
    if hit is not None and hit[0] == step:
        return hit[1]

    from . import grasp_left_observations as obs_mdp
    from . import grasp_left_rewards as rewards

    robot: Articulation = env.scene[jaw_cfg.name]
    obj: RigidObject = env.scene["object"]
    origin = env.scene.env_origins
    s = StageState()

    # ── 원시 측정 ────────────────────────────────────────────────────
    s.d_jaw_cup = rewards.diag_jaw_cup_dist(env, P.JAW_PAD_OFFSET, jaw_cfg)
    forces = obs_mdp.finger_contact_forces(env, sensor_names)
    touch = (forces > P.CONTACT_FORCE_THRESHOLD).float()
    s.touch_frac = touch.mean(dim=-1)
    n_touch = touch.sum(dim=-1)
    s.lift_h = rewards.lift_height(env)          # ★컵 **최저점** — 기울여도 0
    cup_z, s.tilt_deg = _cup_axis_and_tilt(obj)
    cup_pos = obj.data.root_pos_w - origin
    goal = env.command_manager.get_command("object_pose")[:, :3]
    s.d_goal = torch.norm(cup_pos - goal, dim=-1)
    s.cup_speed = torch.norm(obj.data.root_lin_vel_w, dim=-1)
    # ★스폰 xy 를 **실측으로** 잡는다. `default_root_state` 는 랜덤화 **전** 중심이라
    #   스폰 박스(±30·±20 mm)만큼 틀리고, 그건 밀기 여유(25 mm)보다 크다.
    #   리셋 직후 스텝(`episode_length_buf <= 1`)에 그 에피소드의 실제 스폰을 기록한다.
    spawn = _SPAWN.get(key)
    if spawn is None or spawn.shape[0] != env.num_envs:
        spawn = cup_pos[:, :2].clone()
        _SPAWN[key] = spawn
    fresh = env.episode_length_buf <= 1
    if fresh.any():
        spawn[fresh] = cup_pos[fresh, :2]
    s.xy_disp = torch.norm(cup_pos[:, :2] - spawn, dim=-1)

    # ── gripper_base 프레임 3축 분해 (사용자 규격 `BASE — CUP(xy) — TCP`) ──
    # ★★등방 거리(norm)를 쓰지 않는다. 축마다 허용치가 3배 넘게 다르다:
    #     z(진입 깊이) 목표 46.9 mm · y(턱축) ±12.75 mm · x(높이) ±37.5 mm
    #   norm 으로 뭉치면 "턱축으로 60 mm 어긋난 채 깊이만 맞춤"과 "제대로 감쌈"이
    #   같은 값이 된다. t42 가 그 자로 200 mm 밖에서 자세만 다듬었다.
    base_id = _BASE_ID.get(key)
    if base_id is None:
        base_id = robot.body_names.index(P.GRIPPER_BASE_BODY)
        _BASE_ID[key] = base_id
    # 목표점은 컵 **원점이 아니라 파지 높이의 컵 축 위 점**이다. 컵 원점은 테이블 위
    # 92.1 mm 로 파지 대역(10~85 mm) **밖**이라, 원점으로 끌면 44.6 mm 높은 곳에서 멈춘다
    # (이 트랙이 이미 밟은 함정 — 절대 z 판정의 기준선은 "놓였을 때의 값"이다).
    cup_pt = obj.data.root_pos_w + cup_z * P.CUP_ORIGIN_TO_GRASP_Z
    R_base = matrix_from_quat(robot.data.body_quat_w[:, base_id, :])      # base→world
    delta = torch.einsum("nji,nj->ni", R_base, cup_pt - robot.data.body_pos_w[:, base_id, :])
    s.height_h, s.jaw_l, s.enter_s = delta[:, 0], delta[:, 1], delta[:, 2]

    # ── 트리거 (논문 식 3~6). **이진**이고 매 스텝 재평가한다 ─────────
    # ⚠ 래치가 아니다. 이 트랙이 과거에 제거한 것은 "한 번 열리면 유지"하는 래치였고,
    #   이건 순간 술어라 성질이 다르다(자매 트랙 rewards_tip_cyl 주석과 같은 판단).
    s.lam = (s.d_jaw_cup < P.STAGE_GATE_APPROACH_M).float()
    s.mu = s.lam * (n_touch >= P.STAGE_GATE_CONTACT_N).float()
    s.nu = s.mu * (s.lift_h >= P.STAGE_GATE_LIFT_M).float()
    s.rho = s.nu * (s.d_goal < P.STAGE_GATE_TRANSFER_M).float()

    # ── 자세 (사용자 규격) ───────────────────────────────────────────
    # ★"cup_+z 가 world_+z 와 15° 이내". 단계별로 요구가 다르다 —
    #   이송 중에는 관용(U_tol), 목표에서 정지할 때는 직립(U_up).
    s.U_tol = smoothstep(s.tilt_deg, *P.STAGE_TILT_TOLERANCE_DEG)
    s.U_up = smoothstep(s.tilt_deg, *P.STAGE_UPRIGHT_GATE_DEG)

    # ★"TCP_+z 가 world_+z 와 **수직**이 되게 접근". 턱 body 의 z 축이 접근축이고
    #   (`approach_opposed` 도 같은 열을 쓴다), 그 world-z 성분이 0 이어야 한다.
    #   컵이 서 있으면 이 조건은 "원통을 옆에서 문다"와 같아진다(CLAUDE.md 규약 90°).
    approach_axis = matrix_from_quat(
        robot.data.body_quat_w[:, jaw_cfg.body_ids[0], :])[:, :, 2]
    s.perp_q = (1.0 - approach_axis[:, 2].abs()).clamp(0.0, 1.0) ** P.STAGE_PERP_EXPONENT

    # 접근축이 컵을 겨누는가 (1자유도). floor 를 둬 최악에서도 approach 가 안 죽는다.
    jaw_mid = rewards._jaw_mid_local(env, P.JAW_PAD_OFFSET, jaw_cfg)
    to_cup = cup_pos - jaw_mid
    s.align_q = torch.nn.functional.cosine_similarity(
        approach_axis, to_cup, dim=-1, eps=1e-6).clamp(-1.0, 1.0)

    # ★fab_test50: 턱중점 ↔ **컵 파지점**(원점 −44.6 mm) 유클리드 거리. 원본 lift 의
    #   `object_ee_distance` 대응 — 단 기준점을 원점이 아니라 파지점으로 둬서
    #   BASE—CUP—TCP 순서(사용자 규격)가 커널 안에 그대로 들어간다.
    s.d_jaw_grasp = torch.norm(jaw_mid - (cup_pt - origin), dim=-1)

    # ── 단계 진척량 ──────────────────────────────────────────────────
    # ★파지 품질은 **접촉을 곱한다** — 기하만으로는 0 이어야 한다. t38 이 기하 투영만
    #   보는 `enclose` 로 **170 mm 허공에서 closure 를 상한의 74%** 까지 받았다.
    s.grasp_q = rewards.grasp_quality(
        env, P.GRASP_GATE_LATERAL_OK, P.GRASP_GATE_ALONG_OK,
        P.JAW_PAD_OFFSET, jaw_cfg) * s.touch_frac
    # 리프트 진척 — 목표(4 cm)에서 포화. 논문의 "ceases to accumulate".
    s.H = (s.lift_h / P.STAGE_LIFT_REF_M).clamp(0.0, 1.0)
    s.T = torch.exp(-s.d_goal / P.STAGE_TRANSFER_STD_M)
    s.S = torch.exp(-s.cup_speed / P.STAGE_STAY_SPEED_REF)

    _CACHE[key] = (step, s)
    return s

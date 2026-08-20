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

"""관절 목표가 한 스텝에 움직일 수 있는 양을 **관절 속도 한계**로 제한하는 액션.

왜 필요한가 — 실측이다. test13 결정론 정책의 액션 1차 차분은 8 차원 노름 1.713 이고,
그리퍼 축이 정지해 있다고 보면 관절당 목표 도약은 `1.713/√7 × scale(0.5) = 0.324 rad`,
한 제어 스텝은 20 ms 이므로 **16 rad/s 지령**이다. 관절 속도 한계는 2.175~2.61 rad/s —
**지령이 한계의 7 배**다. 팔은 따라갈 수가 없어 그냥 포화하고(실측 관절속도 2.02 rad/s
≈ 한계), PD 가 저역통과 필터 노릇을 하는 상태가 된다. 이것이 "부드럽지 않다"의 정체다.

왜 보상으로 못 고쳤나 — `action_rate_l2` 는 **액션공간 통계**라 탐색 노이즈에 오염된다.
학습 중 액션은 `μ + σ·ε` 이고 `Var(Δa) = Var(Δμ) + 2σ²` 인데 여기서는 노이즈 쪽이 더
컸다. 그래서 옵티마이저가 이 페널티를 줄이는 가장 싼 길은 정책을 부드럽게 만드는 게
아니라 **σ 를 줄이는 것**이었고, test13 이 정확히 그랬다(σ 1.67 → 0.93 인데 ⟨Δμ²⟩ 는
12.3 → 9.9 로 평탄). 자세한 근거는 이 태스크의 CLAUDE.md 함정 절에 있다.

그래서 보상이 아니라 **액션 자체를 제한한다.** 이건 실기에서도 하는 일이라 sim2real
방향으로도 맞다 — 실기 컨트롤러는 슬루 한계를 넘는 목표를 받으면 그냥 거부하거나 포화한다.

★잔여 채터링은 기계적으로 걸러진다. 한계까지 매 스텝 방향을 뒤집어도 목표 진폭은
관절당 ±2.5° / 25 Hz 인데, 팔의 PD 대역폭은 그보다 한참 낮다(k=80 N·m/rad).

★부분관측 문제 없음. 제한기의 상태(직전 목표)는 obs 에 없지만, 지령이 **추종 가능한
크기**로 제한되므로 직전 목표 ≈ 현재 관절 위치이고 `joint_pos_rel` 로 관측된다.
리셋 직후에도 둘 다 기본 자세라 일치한다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass
from isaaclab.utils.string import resolve_matching_names_values

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class RateLimitedJointPositionAction(JointPositionAction):
    """`JointPositionAction` 에 목표 변화율 상한을 씌운다."""

    cfg: RateLimitedJointPositionActionCfg

    def __init__(self, cfg: RateLimitedJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)

        # 관절별 상한 [rad/s] → 한 제어 스텝당 [rad]. 정규식 해석은 부모가 scale 을
        # 푸는 방식과 같게 맞춘다(관절 순서가 아니라 이름으로 매칭돼야 한다).
        rate = torch.zeros(self.action_dim, device=self.device)
        index_list, _, value_list = resolve_matching_names_values(
            cfg.rate_limit, self._joint_names
        )
        rate[index_list] = torch.tensor(value_list, device=self.device)
        if bool((rate <= 0.0).any()):
            missing = [n for i, n in enumerate(self._joint_names) if rate[i] <= 0.0]
            raise ValueError(
                f"rate_limit 이 풀리지 않은 관절이 있다: {missing}. "
                "정규식이 모든 액션 관절을 덮어야 한다."
            )
        self._max_step_delta = (rate * env.step_dt).unsqueeze(0)

        # 제한기의 상태 = 직전에 실제로 내보낸 목표. 리셋 기준은 기본 자세(= offset)다.
        self._prev_target = self._default_target().clone()

    def _default_target(self) -> torch.Tensor:
        """액션 0 이 가리키는 목표. `use_default_offset=True` 면 기본 관절 자세다."""
        if isinstance(self._offset, torch.Tensor):
            return self._offset
        return torch.full_like(self._raw_actions, float(self._offset))

    def process_actions(self, actions: torch.Tensor) -> None:
        # 부모가 scale·offset·clip 을 적용해 **절대 목표**를 만든다.
        super().process_actions(actions)
        step = torch.clamp(
            self._processed_actions - self._prev_target,
            min=-self._max_step_delta,
            max=self._max_step_delta,
        )
        self._processed_actions = self._prev_target + step
        # ★clone 이 아니라 in-place 복사여야 한다. 대입하면 다음 스텝에
        #   `_processed_actions` 와 같은 텐서를 가리켜 제한이 무효가 된다.
        self._prev_target.copy_(self._processed_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        default = self._default_target()
        if env_ids is None:
            self._prev_target.copy_(default)
        else:
            self._prev_target[env_ids] = default[env_ids]


@configclass
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
    """`rate_limit` 은 관절 정규식 → 목표 변화율 상한 [rad/s]."""

    class_type: type = RateLimitedJointPositionAction

    rate_limit: dict[str, float] = None


class JointLimitedDifferentialIKAction(DifferentialInverseKinematicsAction):
    """diff-IK 결과를 **관절 한계 안으로 클램프**한다.

    레퍼런스 `DifferentialInverseKinematicsAction.apply_actions` 는 IK 해를 그대로
    `set_joint_position_target` 에 넘긴다(`task_space_actions.py:214`) — 한계 검사가 없다.
    이 팔은 손목 j6 가 ±45° 뿐이고 손목 3 축 effort 가 7 N·m 라, 도달 불가능한 자세를
    지령하면 관절이 한계에 눌린 채 버티며 IK 가 계속 같은 방향을 밀어 고착된다
    (Fabrics 경로에서 실제로 j5 한계 고착으로 관측됐다).

    agnostic 트랙(`agnostic/tasks/grasp_lift`)도 같은 이유로 IK 해를 한계로 clamp 한다.

    ★★그리고 **관절 변화율**도 묶는다. `IK_ACTION_SCALE` 은 TCP 변위만 묶을 뿐, IK 가
      그걸 관절로 푸는 단계는 안 묶인다 — 자코비안 조건이 나쁜 자세에서는 2 cm 요청이
      큰 관절 이동이 된다. 관절공간 판에는 변화율 상한을 넣어 놓고 IK 판에는 안 넣은
      것이 실측으로 드러났다(test4, epoch 1100):

          적용된 관절 목표 변화 **2.17 rad/s** = 관절 속도 한계(2.175)에 정확히 포화
          방향 반전 **49.3%** · jaw 수평 이탈 **32.4°**
          그리퍼 개도 최소 **16.9 mm** — 컵(지름 58 mm)을 감쌌다면 30.2 mm 에서 막혀야
          하는데 그보다 더 닫혔다 = **컵이 턱 사이에 없을 때 닫는다.**

      한계에서 떨고 있는 팔은 어떤 자세도 유지할 수 없고, 그러면 58 mm 물체를 턱 사이에
      넣는 것이 운이 된다. 관절공간 판에서 같은 처방이 jaw 이탈 23.3° → 8.8° 로 줄인
      직접 증거가 있다.

    ★상한을 **두 군데** 건다. 하나만으로는 부족하다 — 실측으로 확인했다.
      ① `|목표 − 현재 관절| ≤ v·dt` : 목표가 실제 관절보다 한 스텝 이상 앞서지 못하게 한다
         (windup 방지, PD 오차와 토크를 묶는다).
      ② `|목표(t) − 목표(t−1)| ≤ v·dt` : 지령 자체의 진동을 묶는다.
      ① 만 걸었더니 `probe_action_rate_limit.py` 가 목표-대-목표 변화에서 **상한의 200%**
      를 쟀다. 현재 위치를 중심으로 ±Δ 를 오갈 수 있어서다 — 팔은 그 고주파를 못 따라가
      제자리에 서고, 정책은 "움직이려 했는데 안 움직인다"를 겪는다.
    """

    cfg: "JointLimitedDifferentialIKActionCfg"

    def __init__(self, cfg: "JointLimitedDifferentialIKActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        rate = torch.zeros(len(self._joint_ids), device=self.device)
        index_list, _, value_list = resolve_matching_names_values(
            cfg.rate_limit, self._joint_names
        )
        rate[index_list] = torch.tensor(value_list, device=self.device)
        if bool((rate <= 0.0).any()):
            missing = [n for i, n in enumerate(self._joint_names) if rate[i] <= 0.0]
            raise ValueError(
                f"rate_limit 이 풀리지 않은 관절이 있다: {missing}. "
                "정규식이 모든 액션 관절을 덮어야 한다."
            )
        # ★★`apply_actions` 는 **물리 스텝마다** 불린다(decimation 2 → env 스텝당 2 회).
        #   IK 는 매 substep 현재 자세에서 다시 풀어야 하므로 클램프도 여기 있어야 하고,
        #   따라서 상한도 **물리 스텝 기준**이어야 한다. env 스텝 기준으로 잡았더니
        #   프로브가 정확히 **상한의 200%** 를 쟀다(2 회 적용).
        #   ⚠ 관절공간 판(`RateLimitedJointPositionAction`)은 `process_actions` 에서 묶는데
        #     그건 env 스텝당 1 회라 `env.step_dt` 가 맞다. 같은 상한 표를 쓰지만 **곱하는
        #     dt 가 다르다** — 두 클래스를 나란히 읽을 때 헷갈리기 쉬운 지점이다.
        physics_dt = getattr(env, "physics_dt", None) or env.step_dt / env.cfg.decimation
        self._max_step_delta = (rate * physics_dt).unsqueeze(0)
        # ②용 상태. 리셋 기준은 기본 자세다(리셋 직후 팔이 거기 있다).
        self._prev_target = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        default = self._asset.data.default_joint_pos[:, self._joint_ids]
        if env_ids is None:
            self._prev_target.copy_(default)
        else:
            self._prev_target[env_ids] = default[env_ids]

    def apply_actions(self) -> None:
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(
                ee_pos_curr, ee_quat_curr, jacobian, joint_pos
            )
        else:
            joint_pos_des = joint_pos.clone()
        # ① 현재 관절 기준: 목표가 실제 관절보다 앞서 나가지 못하게(windup·토크 제한)
        joint_pos_des = joint_pos + torch.clamp(
            joint_pos_des - joint_pos, min=-self._max_step_delta, max=self._max_step_delta
        )
        # ② 직전 목표 기준: 지령 자체의 진동을 묶는다
        joint_pos_des = self._prev_target + torch.clamp(
            joint_pos_des - self._prev_target,
            min=-self._max_step_delta,
            max=self._max_step_delta,
        )
        # ③ 관절 한계: 한계 밖 지령은 눌린 채 고착시킨다
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, :]
        joint_pos_des = joint_pos_des.clamp(min=limits[..., 0], max=limits[..., 1])
        # ★in-place 복사. 대입하면 같은 텐서를 가리켜 ② 가 무효가 된다.
        self._prev_target.copy_(joint_pos_des)
        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)


@configclass
class JointLimitedDifferentialIKActionCfg(DifferentialInverseKinematicsActionCfg):
    """`rate_limit` 은 관절 정규식 → 목표 변화율 상한 [rad/s]."""

    class_type: type = JointLimitedDifferentialIKAction

    rate_limit: dict[str, float] = None

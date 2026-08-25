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

"""커리큘럼 항 — **과제가 성립한 뒤에** 억제 항을 켠다.

★★fab_test14 가 이 파일이 존재하는 이유다. 액션 jerk 페널티를 epoch 0 부터 **정적으로**
  켰더니 이송 학습 자체가 무너졌다(TB 전수 분석):
      σ(탐색 폭)  ep0~99 에서 t13 0.915 → t14 **0.543** (−41%)
      리프트 습득  t13 epoch 250 → t14 **epoch 600** (350 epoch 지연)
      dwell        t13 최고 3.19 → t14 **0.023 에서 평탄**(사실상 사망)
  초기에는 대부분 항이 0 이라 비교 대상이 달랐다 — ep0~99 에 살아 있던 항은 셋뿐이고
  jerk(−0.21)는 그 중 `reaching_object`(0.44)의 **48%**, `grip_closure`(0.05)의 **449%** 였다.
  "아직 아무것도 못 하는" 정책이 받은 가장 선명한 신호가 **"급하게 움직이지 마라"** 였던 것이다.
  그리고 dwell 을 벌려면 미세 보정이 필요한데 그게 곧 jerk 라, 상금(0.002~0.023)이 벌금
  (0.08~0.25)을 **한 번도 넘지 못했다**(배율 4~95배).

  ⚠ 절대값만 보고 "jerk 는 총보상 120 대비 0.2 니 무시할 만하다"고 판단하면 안 된다.
    문제는 총보상 대비 크기가 아니라 **그 시점에 살아 있는 신호 대비 크기**다.

성공 런(fab_test13)은 우연히 올바른 순서였다 — dwell 이 epoch 850 에 1.0 을 지속 돌파했고,
레퍼런스 `action_rate`/`joint_vel` 커리큘럼은 그 한참 뒤인 epoch 1501 에 발동했다.
이 파일은 그 순서를 **우연이 아니라 코드로** 강제한다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

from . import grasp_left_obs_noise as obs_noise

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class enable_penalty_after_dwell(ManagerTermBase):
    """`dwell` 지표가 임계를 넘긴 뒤에야 억제 항의 weight 를 켠다 (한 번 켜지면 래치).

    구현 메모:
      · 감시 대상은 `RewardManager._step_reward` 의 dwell 열이다. 이 값은 weight 가 곱해진
        스텝당 값(raw × weight)이라, TFEvents 의 `Episode_Reward/dwell_at_goal` 과 **같은
        척도**다(에피소드가 만기까지 가면 일치). 그래서 TB 보드에서 읽은 임계를 그대로 쓴다.
      · 커리큘럼 항은 리셋 시점에 호출된다(`CurriculumManager.compute(env_ids)`). 1024 env 가
        엇갈려 리셋되므로 EMA 표본은 충분하다.
      · ★래치 필수 — 임계 근처에서 켜졌다 꺼졌다 하면 정책이 두 보상 지형 사이를 오간다.
      · ★꺼진 상태는 weight 0.0 인데, IsaacLab 은 weight==0 인 항을 **log-only** 로 처리해
        원값을 계속 로깅한다(reward_manager.compute 의 `if term_cfg.weight == 0.0` 분기).
        즉 꺼둔 동안에도 TB 에서 jerk 원값 추이를 볼 수 있다 — 게이트 판정에 유용하다.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._term_cfg = env.reward_manager.get_term_cfg(cfg.params["term_name"])
        self._dwell_idx = env.reward_manager.active_terms.index(cfg.params["dwell_term_name"])
        self._ema = 0.0
        self._latched = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        dwell_term_name: str,
        weight: float,
        dwell_threshold: float,
        ema_alpha: float,
    ) -> float:
        if not self._latched:
            current = float(env.reward_manager._step_reward[:, self._dwell_idx].mean())
            self._ema = (1.0 - ema_alpha) * self._ema + ema_alpha * current
            if self._ema >= dwell_threshold:
                self._latched = True
                self._term_cfg.weight = weight
                env.reward_manager.set_term_cfg(term_name, self._term_cfg)
        # 반환값은 TFEvents 에 `Curriculum/<이름>` 으로 찍힌다 — 언제 켜졌는지 관측 가능.
        return self._term_cfg.weight


class adr_expand_on_dwell(ManagerTermBase):
    """성공 지표(dwell)가 임계를 넘을 때마다 **난이도를 한 단계씩** 넓힌다 (ADR).

    ★사용자 지시로 grasp_v1 의 ADR 구조를 이 트랙에 옮긴 것이다. 이 파일의 다른 항
      (`enable_penalty_after_dwell`)과 **같은 원리**를 쓴다 — 난이도를 올리는 요소는
      과제가 성립한 뒤에 켠다. fab_test14 가 그 반대(정적으로 처음부터)를 해서 초기 탐색이
      41% 죽고 이송 학습이 무너졌다.

    한 단계에서 다음 항목이 함께 넓어진다:
      · 스폰 x·y 랜덤 폭         (물체 위치 일반화)
      · 목표 jitter 배율          (이송 목표 일반화 — pouring 대비)
      · 컵 마찰·반발·질량        (물성 일반화, s2r)
      · 컵 외란 힘                (파지 강건성)

    ⚠ 확장 간 `min_epochs_between` 만큼 쉬어 간다 — 연속 확장으로 난이도가 급등하면
      dwell 이 무너지고 그때는 이미 늦다(t16 의 정점-후-붕괴와 같은 형태).
    ⚠ 레벨은 **단조 증가만** 한다(내리지 않는다). 내리면 정책이 두 난이도 사이를 오간다.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._metric_idx = env.reward_manager.active_terms.index(cfg.params["metric_term"])
        self._ema = 0.0
        self._level = 0
        self._last_step = -10**9

    # -- 내부: 레벨 → 실제 값 -------------------------------------------------
    def _frac(self) -> float:
        # ★kuka 규약: 분모는 `num_increments`, 카운터는 0..num_increments (51 단계).
        #   `param_slope = (upper - lower) / num_increments; value = slope * counter + lower`
        n = int(self.cfg.params["levels"])
        return 0.0 if n <= 0 else min(self._level / n, 1.0)

    def _lerp(self, pair):
        a, b = pair
        f = self._frac()
        if isinstance(a, (tuple, list)):
            return tuple(x + (y - x) * f for x, y in zip(a, b))
        return a + (b - a) * f

    def _apply(self, env: ManagerBasedRLEnv) -> None:
        """현재 레벨을 이벤트 term 들의 params 에 써 넣는다."""
        from . import grasp_left_preset as P

        em = env.event_manager

        def _set(term: str, **kw):
            # 없는 term 을 조용히 건너뛰면 ADR 이 아무 일도 안 하고도 통과한다 — 터뜨린다.
            cfg_ = em.get_term_cfg(term)
            cfg_.params.update(kw)
            em.set_term_cfg(term, cfg_)

        sx = self._lerp(P.ADR_SPAWN_X_RANGE)
        sy = self._lerp(P.ADR_SPAWN_Y_RANGE)
        _set("reset_object_position",
             pose_range={"x": (-sx, sx), "y": (-sy, sy), "z": (0.0, 0.0)})

        gs = self._lerp(P.ADR_GOAL_JITTER_SCALE)
        cmd = env.command_manager.get_term("object_pose")
        jx, jy, jz = (j * gs for j in P.GOAL_JITTER)
        cmd.cfg.ranges.pos_x = (P.GOAL_POINT[0] - jx, P.GOAL_POINT[0] + jx)
        cmd.cfg.ranges.pos_y = (P.GOAL_POINT[1] - jy, P.GOAL_POINT[1] + jy)
        cmd.cfg.ranges.pos_z = (P.GOAL_POINT[2] - jz, P.GOAL_POINT[2] + jz)

        _set("cup_physics_material",
             static_friction_range=self._lerp(P.ADR_CUP_STATIC_FRICTION),
             dynamic_friction_range=self._lerp(P.ADR_CUP_DYNAMIC_FRICTION),
             restitution_range=self._lerp(P.ADR_CUP_RESTITUTION))
        _set("cup_mass", mass_distribution_params=self._lerp(P.ADR_CUP_MASS_SCALE))
        # ★외란은 **가속도**로 준다(원본 `object_wrench.max_linear_accel`). 이벤트 함수가
        #   질량을 곱해 힘을 만든다 — 그래야 질량 DR 과 외란 DR 이 서로를 상쇄하지 않는다.
        env._dextrah_wrench_max_accel = self._lerp(P.ADR_CUP_MAX_LINEAR_ACCEL)

        # ★속도 피드포워드를 난이도로 걷어낸다(DEXTRAH `pd_targets/velocity_target_factor`).
        #   레벨 0 = 1.0(완전 피드포워드), 최고 레벨 = 0.0. 이벤트가 아니라 액션항이라
        #   `_set` 이 아니라 프로퍼티로 쓴다.
        env.action_manager.get_term("arm_action").vel_ff_scale = self._lerp(P.ADR_VEL_FF_SCALE)

        # ★fabric cspace damping — 원본 kuka ADR `fabric_damping.gain (10, 20)`.
        #   10 = 덜 감쇠 = 빠른 수렴(쉬움), 20 = 굼뜸(어려움).
        act = env.action_manager.get_term("arm_action")
        act._damping[:] = self._lerp(P.ADR_FABRIC_DAMPING_GAIN)

        # ★팔 PD 게인·관절 마찰 DR — 원본 kuka ADR 에 있고 우리에겐 없던 항목.
        gs = self._lerp(P.ADR_ARM_GAIN_SCALE)
        _set("arm_gains", stiffness_distribution_params=gs, damping_distribution_params=gs)
        _set("arm_friction", friction_distribution_params=self._lerp(P.ADR_ARM_FRICTION))
        # ★관측 노이즈 — 원본 `object_state_noise` · `robot_state_noise`.
        #   ⚠ `ObsTerm.noise`(Unoise) 로는 원본을 표현할 수 없다. 원본은 (a) 폭 자체를
        #     env 마다 다시 뽑고 (b) 에피소드 내내 유지되는 **bias** 를 따로 얹는다.
        #     둘 다 상태가 필요해서 grasp_left_obs_noise 모듈이 대신 든다.
        for _key, _n, _b in (
            (obs_noise.JOINT_POS, P.ADR_OBS_JOINT_POS_NOISE, P.ADR_OBS_JOINT_POS_BIAS),
            (obs_noise.JOINT_VEL, P.ADR_OBS_JOINT_VEL_NOISE, P.ADR_OBS_JOINT_VEL_BIAS),
            (obs_noise.OBJ_POS, P.ADR_OBS_OBJ_POS_NOISE, P.ADR_OBS_OBJ_POS_BIAS),
            (obs_noise.OBJ_ROT, P.ADR_OBS_OBJ_ROT_NOISE, P.ADR_OBS_OBJ_ROT_BIAS),
        ):
            obs_noise.set_level_value(env, _key, noise=self._lerp(_n), bias=self._lerp(_b))

        # ★리셋 관절 노이즈 — 원본 `robot_spawn.joint_pos_noise / joint_vel_noise`.
        _pn = self._lerp(P.ADR_ROBOT_SPAWN_POS_NOISE)
        _vn = self._lerp(P.ADR_ROBOT_SPAWN_VEL_NOISE)
        _set("arm_spawn_noise", position_range=(-_pn, _pn), velocity_range=(-_vn, _vn))

        _set("robot_physics_material",
             static_friction_range=self._lerp(P.ADR_ROBOT_STATIC_FRICTION),
             dynamic_friction_range=self._lerp(P.ADR_ROBOT_DYNAMIC_FRICTION),
             restitution_range=self._lerp(P.ADR_ROBOT_RESTITUTION))

    # -- 커리큘럼 훅 ---------------------------------------------------------
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        metric_term: str,
        trigger: float,
        levels: int,
        min_steps_between: int,
        ema_alpha: float,
    ) -> float:
        if self._level == 0 and self._last_step < 0:
            self._apply(env)          # 레벨 0(중립)을 명시적으로 써 둔다
            self._last_step = 0

        if self._level < levels:
            # ★★fab_test22 원본 정합: kuka 는 보상 **크기**가 아니라 성공 구역에 있는
            #   env **비율**로 올린다(`in_success_region.float().mean() > success_for_adr`).
            #   dwell 항의 스텝 보상이 0 보다 크면 그 env 는 성공 구역 안에 있다
            #   (DwellSettledAtGoal 은 q > q_thresh 인 스텝에만 카운터를 올린다).
            #   ⚠ 비율은 weight 에 불변이라 보상을 손봐도 ADR 속도가 안 흔들린다 —
            #     구 방식(EMA of 크기)은 weight 를 바꿀 때마다 임계 의미가 달라졌다.
            cur = float(
                (env.reward_manager._step_reward[:, self._metric_idx] > 0.0).float().mean()
            )
            self._ema = (1.0 - ema_alpha) * self._ema + ema_alpha * cur
            # ⚠ `common_step_counter` 는 **env 스텝**이다(epoch 아님). 프리셋에서
            #   epoch × horizon_length 로 환산한 값을 받는다 — 안 하면 확장이 25배 빨라진다.
            step = int(env.common_step_counter)
            if self._ema >= trigger and step - self._last_step >= min_steps_between:
                self._level += 1
                self._last_step = step
                self._ema = 0.0       # 다음 단계는 새로 벌어야 한다
                self._apply(env)
        return float(self._level)

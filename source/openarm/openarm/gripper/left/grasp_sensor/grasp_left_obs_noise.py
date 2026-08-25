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

"""관측 노이즈 — DEXTRAH kuka `object_state_noise` · `robot_state_noise` 정합.

★★원본은 관측 노이즈를 **두 층**으로 준다. 우리는 `Unoise` 한 층만 썼고, 그것도
  전 env 공통 폭이었다. 원본(`dextrah_kuka_allegro_env.py:_reset_idx`)은:

      # 에피소드마다 env 별로 **폭 자체를** 다시 뽑는다
      noise_width[env] = adr_max_noise * rand()               # U(0, adr_max)
      bias_width[env]  = adr_max_bias  * rand()               # U(0, adr_max)
      bias[env]        = bias_width[env] * (rand() - 0.5)      # 에피소드 내내 고정

      # 매 스텝
      x_noisy = x + noise_width * 2 * (rand_like(x) - 0.5) + bias

  두 층이 하는 일이 다르다:
    · **per-step 노이즈**는 센서 잡음이다. 평균이 0 이라 정책이 시간축으로 평균 내 지울 수 있다.
    · **per-episode bias**는 캘리브레이션 오차다. **평균 내도 안 지워진다** — 정책이
      "내 추정이 일정하게 틀어져 있을 수 있다"를 전제로 행동하게 만드는 건 이쪽이다.
      실기 이식에서 실제로 문제가 되는 것도 bias 쪽이다(extrinsics·엔코더 오프셋).

  우리에겐 bias 층이 통째로 없었다. `ObsTerm.noise` 는 `NoiseCfg` 만 받고 상태를 못 들기
  때문에(에피소드 내내 유지되는 per-env 값을 표현할 수 없다) 이 모듈이 필요하다.

  ⚠ 폭이 env 마다 다르다는 점도 원본의 핵심이다. 전 env 공통 폭이면 "이번 판은 센서가
    깨끗하다"는 경우가 아예 없어서, 정책이 노이즈 수준 자체를 조건으로 쓰지 못한다.

사용:
  · `resample` 을 `mode="reset"` EventTerm 으로 건다.
  · ADR 커리큘럼이 `set_level_value(env, key, value)` 로 폭 상한을 올린다.
  · 관측 함수가 `corrupt(env, key, x)` 를 호출한다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 채널 이름 — 원본 `object_state_noise` / `robot_state_noise` 의 항목과 1:1.
OBJ_POS = "object_pos"
OBJ_ROT = "object_rot"
JOINT_POS = "robot_joint_pos"
JOINT_VEL = "robot_joint_vel"
CHANNELS = (OBJ_POS, OBJ_ROT, JOINT_POS, JOINT_VEL)

_STATE_ATTR = "_grasp_left_obs_noise"


class _NoiseState:
    """env 에 붙는 per-env 노이즈/바이어스 버퍼.

    ⚠ `noise_width` · `bias` 는 (num_envs, 1) 이다. 관측 텐서가 (num_envs, D) 라
      브로드캐스트로 **한 env 안의 모든 성분에 같은 폭·같은 bias** 가 걸린다.
      원본과 같은 규약이다(원본도 스칼라 폭을 (N,1) 로 들고 다닌다).
    """

    def __init__(self, num_envs: int, device: str):
        z = lambda: torch.zeros(num_envs, 1, device=device)   # noqa: E731
        self.noise_width = {k: z() for k in CHANNELS}
        self.bias = {k: z() for k in CHANNELS}
        # ADR 이 올리는 상한. 레벨 0 에서 전부 0 → 노이즈도 bias 도 정확히 0.
        self.max_noise = {k: 0.0 for k in CHANNELS}
        self.max_bias = {k: 0.0 for k in CHANNELS}


def state(env: "ManagerBasedRLEnv") -> _NoiseState:
    st = getattr(env, _STATE_ATTR, None)
    if st is None:
        st = _NoiseState(env.num_envs, env.device)
        setattr(env, _STATE_ATTR, st)
    return st


def set_level_value(env: "ManagerBasedRLEnv", key: str, *, noise: float, bias: float) -> None:
    """ADR 커리큘럼이 호출한다 — 채널 하나의 폭 상한을 세운다."""
    if key not in CHANNELS:
        raise KeyError(f"unknown obs-noise channel: {key!r} (expected one of {CHANNELS})")
    st = state(env)
    st.max_noise[key] = float(noise)
    st.max_bias[key] = float(bias)


def resample(env: "ManagerBasedRLEnv", env_ids: torch.Tensor | None = None) -> None:
    """EventTerm(mode="reset") — 에피소드 시작마다 폭과 bias 를 다시 뽑는다."""
    st = state(env)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    if n == 0:
        return
    for key in CHANNELS:
        # 원본: width = adr_max * rand()  →  env 마다 "이번 판의 센서 품질"이 다르다.
        st.noise_width[key][env_ids, 0] = st.max_noise[key] * torch.rand(n, device=env.device)
        bias_width = st.max_bias[key] * torch.rand(n, device=env.device)
        # 원본: bias = bias_width * (rand() - 0.5)  →  부호가 양쪽으로 갈린다.
        st.bias[key][env_ids, 0] = bias_width * (torch.rand(n, device=env.device) - 0.5)


def corrupt(env: "ManagerBasedRLEnv", key: str, x: torch.Tensor) -> torch.Tensor:
    """per-step 균등 노이즈 + per-episode bias 를 더한 **새 텐서**를 돌려준다."""
    st = state(env)
    # ⚠ 절대 in-place 로 쓰지 말 것 — x 는 시뮬레이터 버퍼의 뷰일 수 있다.
    return x + st.noise_width[key] * 2.0 * (torch.rand_like(x) - 0.5) + st.bias[key]

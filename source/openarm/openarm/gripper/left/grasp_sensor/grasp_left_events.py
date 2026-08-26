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

"""이 태스크 전용 이벤트 term.

lift 레퍼런스는 로봇의 **모든** 관절이 액션 대상이라 필요 없던 것이, 비대칭 양팔 로봇에서는
필요해진다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from . import grasp_left_preset as P

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.envs import ManagerBasedEnv


def hold_joints_at_target(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    joint_targets: dict[str, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """액션이 건드리지 않는 관절의 **PD 위치 목표**를 지정 자세로 고정한다.

    ★왜 필요한가. `ArticulationCfg.init_state.joint_pos` 는 관절의 **상태**만 정한다.
      PD 목표(`joint_pos_target`)는 정하지 않으며, 그 버퍼는 0 으로 시작한다. 액션 대상
      관절은 매 스텝 ActionTerm 이 목표를 써 주지만, **액션 대상이 아닌 관절은 아무도 쓰지
      않는다** — 목표가 0 인 채로 남아 팔이 "차렷"으로 내려간다.

      이 로봇은 왼팔 7 + 그리퍼 1 만 액션 대상이고 오른팔 7 + 오른손 20 + 헤드 2 는 아니다.
      그래서 유휴 오른팔이 중력이 아니라 **0 을 향한 PD 지령** 때문에 내려가 테이블·바닥에
      닿았다(렌더 관찰). effort_limit 을 아무리 올려도 안 고쳐지는 종류의 문제다.

      실측(프로브): 목표를 명시하지 않으면 관절 오차 최대 25.4°, 명시하면 2.1°.

    리셋 때 한 번만 써 주면 된다 — 목표 버퍼는 다음 리셋까지 유지된다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids, joint_names = asset.find_joints(list(joint_targets), preserve_order=True)
    target = torch.tensor(
        [joint_targets[n] for n in joint_names], device=asset.device, dtype=torch.float32
    )
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device)
    asset.set_joint_position_target(
        target.unsqueeze(0).expand(len(env_ids), -1), joint_ids=joint_ids, env_ids=env_ids
    )


def apply_object_wrench(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    jaw_cfg: SceneEntityCfg,
    torsional_radius: float,
    hand_dist_threshold: float,
) -> None:
    """물체 외란 렌치 — DEXTRAH kuka `apply_object_wrench` 정합.

    ★★원본과 우리 것이 세 군데 갈려 있었다. 전부 IsaacLab 기본
      `mdp.apply_external_force_torque` 를 쓴 데서 온다:

      ① **방향이 등방이 아니었다.** 기본 항은 `force_range=(0.0, F)` 를 성분마다 균등
         추출한다 → 세 성분이 전부 양수, 즉 힘이 항상 **+x+y+z 한 팔분면**으로만 간다.
         정책은 "외란은 오른쪽 위 뒤로 온다"를 외워 버릴 수 있다. 원본은 정규분포
         방향벡터를 정규화해 **등방**으로 뽑는다.
      ② **크기가 질량과 무관했다.** 원본은 가속도를 뽑고(`U(0, a_max)`) 질량을 곱한다.
         그래야 ADR 이 질량을 ×3 로 키워도 외란이 만드는 **가속도**가 일정하다. 고정
         힘이면 무거운 컵일수록 외란이 약해져, 질량 DR 과 외란 DR 이 서로를 상쇄한다.
      ③ **토크가 0 이었다.** 원본은 `mass · accel · torsional_radius` 크기의 등방 토크를
         같이 건다. 병진만 흔들면 컵이 손 안에서 **돌아가는** 실패 모드를 훈련하지 못한다.

    그리고 원본은 렌치를 **손이 물체 근처일 때만** 건다
    (`hand_to_object_pos_error <= hand_to_object_dist_threshold`). 이유가 있다 — 접근
    중에 아직 잡지도 않은 컵을 밀어 버리면 그건 외란 강건성이 아니라 그냥 과제가
    무작위로 바뀌는 것이고, 스폰 랜덤화가 이미 그 역할을 한다.

    ⚠ `mode="interval"` 로 걸어야 한다(원본 `wrench_trigger_every` = 1 s 재추첨).
      힘은 다음 재추첨까지 유지된다 — 원본도 `torch.where(step % every == 0, new, old)` 로
      같은 규약이다.
    """
    asset = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[jaw_cfg.name]
    num_bodies = asset.num_bodies
    device = env.device

    max_accel = float(getattr(env, "_dextrah_wrench_max_accel", 0.0))
    if max_accel <= 0.0:
        return

    # 질량 (num_envs, 1) — ADR 이 질량을 스케일하므로 매 회 새로 읽는다.
    mass = asset.root_physx_view.get_masses().to(device).sum(dim=-1, keepdim=True)
    accel = max_accel * torch.rand(env.num_envs, 1, device=device)
    force_mag = (mass * accel).unsqueeze(-1)                      # (N, 1, 1)
    torque_mag = force_mag * torsional_radius

    def _isotropic() -> torch.Tensor:
        return torch.nn.functional.normalize(
            torch.randn(env.num_envs, num_bodies, 3, device=device), dim=-1)

    forces = force_mag * _isotropic()
    torques = torque_mag * _isotropic()

    # 손–물체 거리 게이트. 턱 body 들의 중점을 손 위치로 본다.
    jaw_pos = robot.data.body_pos_w[:, jaw_cfg.body_ids].mean(dim=1)
    dist = (jaw_pos - asset.data.root_pos_w).norm(dim=-1)
    near = (dist <= hand_dist_threshold)[:, None, None]
    forces = torch.where(near, forces, torch.zeros_like(forces))
    torques = torch.where(near, torques, torch.zeros_like(torques))

    asset.set_external_force_and_torque(forces, torques, env_ids=None)

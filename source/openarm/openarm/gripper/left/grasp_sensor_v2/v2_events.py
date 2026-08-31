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

"""v2 이벤트 — 낙하/전도 컵의 **에피소드 내 재소환** (라운드 8 후보, 08.29).

★★왜. `object_dropping` 은 bootstrap 없는 **진짜 종료**라 잔여 보상을 전액 몰수한다.
  실측 기대값: 흔들기 0.814/step(안전) vs 정착 1.0/step(낙하 위험) — 손익분기 실패율
  22.9% < 실측 전도율 51% ⇒ 정책이 정착을 회피한다(8 판 공통 '정점 후 붕괴'의 구조).
  재소환은 그 절벽을 "계단 0 층 복귀 + 재시도 시간"이라는 **회복 가능한 비용**으로
  바꾼다. 부수 이득: 지금 에피소드의 17~20% 가 drop 조기 종료로 버려지는데, 그
  데이터가 전부 재시도 경험이 된다.

★해킹면: 일부러 떨어뜨려 얻는 것이 없다 — 계단이 0 층으로 떨어질 뿐이고 컵은
  스폰 상자(출발점)로 돌아간다.

⚠ 순변위 트래커: 텔레포트가 한 번의 가짜 고속으로 보인다. 현재 라운드는
  `STILL_NET=0`(순간속도)라 무해하지만, 순변위 판에 켤 때는 트래커 리셋을 같이 배선할 것.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from ..grasp_sensor.grasp_left_rewards import _cup_upright_cos
from . import v2_preset as P

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 검증용 누적 통계 — 스모크에서 "TCP 와 겹치지 않는가"를 로그로 증명한다.
_STATS = {"n": 0, "min_tcp": float("inf"), "sum_tcp": 0.0, "next_print": 200, "defer": 0}


def resample_obs_bias(env: "ManagerBasedRLEnv", env_ids: torch.Tensor,
                      bias_range: float = 0.0) -> None:
    """에피소드 고정 obs bias 재샘플 (리셋 모드). `bias_range` 는 ADR 사다리가 넓힌다."""
    buf = getattr(env, "_v2_cup_obs_bias", None)
    if buf is None:
        buf = torch.zeros(env.num_envs, 3, device=env.device)
        env._v2_cup_obs_bias = buf
    buf[env_ids] = (torch.rand(len(env_ids), 3, device=env.device) * 2.0 - 1.0) * bias_range


def respawn_dropped_cup(env: "ManagerBasedRLEnv", env_ids: torch.Tensor,
                        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
                        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")) -> None:
    """낙하(`z < OBJECT_DROP_HEIGHT`) 또는 전도(`upright cos < RESPAWN_TIPPED_COS`)한
    컵을 스폰 상자 안, **TCP 에서 `RESPAWN_TCP_CLEARANCE` 이상** 떨어진 곳에 다시 놓는다.

    · 팔과의 겹침 방지 = 리젝션 샘플링(`RESPAWN_MAX_TRIES` 회).
      ★★전부 실패하면 **이번 스텝은 보류한다** — 스폰 상자가 40×40 mm 라 팔이 그 위에
        있으면 어떤 후보도 여유를 못 채운다. 초판의 "가장 먼 후보" 폴백은 스모크에서
        TCP 19 mm 옆에 컵을 떨어뜨렸다(그리퍼 안 텔레포트 = 물리 폭발 위험).
        보류해도 컵은 떨어진/넘어진 채 그대로라 다음 스텝에 다시 잡힌다. 데드락은
        없다 — 정책이 그 자리에 머물면 stage 0 보상뿐이라 gradient 가 밀어낸다.
    · 자세는 직립(항등 quat)·속도 0 — "내려놓기"다.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    z_local = obj.data.root_pos_w[env_ids, 2] - env.scene.env_origins[env_ids, 2]
    tipped = _cup_upright_cos(env, object_cfg)[env_ids] < P.RESPAWN_TIPPED_COS
    need = (z_local < P.OBJECT_DROP_HEIGHT) | tipped
    ids = env_ids[need]
    if ids.numel() == 0:
        return

    dev = env.device
    n = ids.numel()
    tries = int(P.RESPAWN_MAX_TRIES)
    # 후보 (n, tries, 3) — env-local
    x = P.CUP_SPAWN_X_CENTER + (torch.rand(n, tries, device=dev) * 2 - 1) * P.CUP_SPAWN_X_RANGE
    y = P.CUP_SPAWN_Y_CENTER + (torch.rand(n, tries, device=dev) * 2 - 1) * P.CUP_SPAWN_Y_RANGE
    z = torch.full((n, tries), float(P.CUP_SPAWN_Z), device=dev)
    cand = torch.stack((x, y, z), dim=-1)

    ee = env.scene[ee_frame_cfg.name]
    tcp = ee.data.target_pos_w[ids, 0, :] - env.scene.env_origins[ids]
    d = torch.norm(cand - tcp.unsqueeze(1), dim=-1)                  # (n, tries)
    ok = d >= P.RESPAWN_TCP_CLEARANCE
    # ★여유를 채운 후보가 있는 env 만 이번 스텝에 재소환 — 나머지는 보류(다음 스텝 재시도)
    has_ok = ok.any(dim=1)
    if not bool(has_ok.any()):
        _STATS["defer"] += n
        return
    _STATS["defer"] += int((~has_ok).sum())
    first = torch.argmax(ok.int(), dim=1)
    sel = torch.nonzero(has_ok).squeeze(-1)
    ids = ids[sel]
    n = ids.numel()
    pick = cand[sel, first[sel]]
    pick_d = d[sel, first[sel]]

    pose = torch.zeros(n, 7, device=dev)
    pose[:, :3] = pick + env.scene.env_origins[ids]
    pose[:, 3] = 1.0                                                  # 직립 (w,x,y,z)
    obj.write_root_pose_to_sim(pose, env_ids=ids)
    obj.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=ids)

    # 검증 통계 — 스모크가 "겹침 없음"을 로그로 판정한다.
    _STATS["n"] += n
    _STATS["min_tcp"] = min(_STATS["min_tcp"], float(pick_d.min()))
    _STATS["sum_tcp"] += float(pick_d.sum())
    if _STATS["n"] >= _STATS["next_print"]:
        print(f"[respawn] 누적 {_STATS['n']}회 · TCP거리 min {_STATS['min_tcp']*1000:.0f}mm"
              f" · 평균 {_STATS['sum_tcp']/_STATS['n']*1000:.0f}mm"
              f" · 여유기준 {P.RESPAWN_TCP_CLEARANCE*1000:.0f}mm"
              f" · 보류 {_STATS['defer']}", flush=True)
        _STATS["next_print"] = _STATS["n"] + 1000

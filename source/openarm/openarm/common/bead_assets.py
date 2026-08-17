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

"""비드(구슬) 자산 설정 — grasp 수집과 pour 소비가 **동일한 비드**를 쓰게 하는 단일 출처.

왜 공용으로 뺐는가
------------------
`both/pour_v1` 은 warm state 에 담긴 `bead_state` 를 그대로 복원해 에피소드를 시작한다.
그 상태를 만든 쪽(grasp_v1 수집)과 복원하는 쪽(pour_v1)의 비드 물성이 다르면
같은 좌표를 복원해도 **동역학이 달라진다** — 질량·마찰·반발·솔버 반복수가 전부 관여한다.
따라서 비드 정의는 한 곳에만 두고 양쪽이 같은 함수를 호출한다.

수치의 출처는 pour 계열의 튜닝 이력이다(주석에 근거를 남겨둔다). 값을 바꾸면
**수집 캐시와 소비 환경이 함께 무효**가 되므로, 바꿀 때는 warm 재수집이 필요하다.
"""

from __future__ import annotations

import math
import os as _os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

# 기본 비드 개수. pour 의 `bead_count` 기본값과 같아야 warm 복원이 성립한다.
DEFAULT_BEAD_COUNT = 20

# 비드 1개 질량[kg]. 5g→1g: deep tilt 시 쏠림 토크 감소 → grasp 슬립 완화.
BEAD_MASS = 0.001
BEAD_SCALE = (0.5, 0.5, 0.5)


def bead_offsets_in_cup(n: int = DEFAULT_BEAD_COUNT) -> list[list[float]]:
    """컵 body frame 기준 비드 소환 offset (n,3) — **검증된 단일 배치**.

    ★2026-08-17 사고: grasp 수집용으로 이 배치를 재사용하지 않고 골든앵글 나선을 새로
      만들었더니 **최소 중심간 거리가 9.1mm** 로 좁아져 소환 순간 겹쳤다. PhysX 가
      침투를 밀어내며 비드가 컵 벽을 관통해 튀어나갔고, 수집된 warm state 는
      "빈 컵 + 컵 밑에 깔린 비드"였다(컵 내부 유지율 0.042, dz −0.15~−0.09 에 47%).
      아래 배치는 pour 가 20개를 안정적으로 담아 온 값이다 — **최소 거리 15.6mm**.
      새로 만들지 말고 이 함수를 쓸 것.

    배치 규칙: 층당 5개를 72° 균등 배치하고 층마다 0.35rad 비틀어 수직 정렬을 깬다.
      반경은 층 짝/홀로 14/18mm 교대, 층 간격 14mm.
    """
    out: list[list[float]] = []
    beads_per_layer = 5
    for i in range(n):
        layer = i // beads_per_layer
        slot = i % beads_per_layer
        angle = (2.0 * math.pi * slot / beads_per_layer) + (0.35 * layer)
        radius = 0.014 + 0.004 * (layer % 2)
        z = 0.006 + 0.014 * layer
        out.append([radius * math.cos(angle), radius * math.sin(angle), z])
    return out


def make_beads_cfg(
    assets_dir: str,
    n: int = DEFAULT_BEAD_COUNT,
    *,
    prim_prefix: str = "Bead",
) -> RigidObjectCollectionCfg:
    """비드 RigidObjectCollection 설정을 만든다.

    Args:
        assets_dir: `hdgp/assets` 절대경로 (`bead/bead.usd` 를 찾는다).
        n: 비드 개수. **warm 수집과 pour 소비가 같아야 한다.**
        prim_prefix: prim 이름 접두사. 기본값을 바꾸면 기존 캐시와 무관하지만
            (상태는 이름이 아니라 순서로 저장됨) 씬 디버깅 시 혼동을 줄이려 노출한다.
    """
    rigid_objects: dict[str, RigidObjectCfg] = {}
    for i in range(n):
        bead_spawn_cfg = UsdFileCfg(
            usd_path=_os.path.join(assets_dir, "bead", "bead.usd"),
            scale=BEAD_SCALE,
            activate_contact_sensors=False,
            mass_props=sim_utils.MassPropertiesCfg(mass=BEAD_MASS),
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=8,   # 16→8: GPU contact stage 연산 부하 감소
                solver_velocity_iteration_count=2,   # 4→2: 동일 이유
                linear_damping=0.0,                  # 0.1→0.0: 인위 공기저항 제거 (컵 벽 자연 흐름)
                angular_damping=0.0,                 # 0.1→0.0: 구름 방해 제거
                max_depenetration_velocity=1.0,      # 5.0→1.0: 침투 보정 폭발 방지 (PhysX crash 주원인)
                max_linear_velocity=10.0,            # 5.0→10.0: 속도 제한 완화 (깊은 tilt 시 비드 흐름)
                max_angular_velocity=100.0,          # 10.0→100.0: 회전 제한 완화 (자연 굴림)
            ),
        )
        # 이 IsaacLab 버전의 UsdFileCfg는 physics_material 생성자 인자를 직접 받지 않는다.
        # spawn_from_usd()는 cfg.physics_material 속성이 있으면 바인딩하므로 생성 후 후첨가한다.
        # 기본 material 마찰(0.5/0.5)보다 낮춰 컵 내부에서 구슬이 더 쉽게 굴러가게 한다.
        bead_spawn_cfg.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.1,
            dynamic_friction=0.08,
            restitution=0.3,                         # 0.1→0.3: 반발력 증가 (표면 접착 완화)
            friction_combine_mode="min",
            restitution_combine_mode="max",
        )
        rigid_objects[f"bead_{i:02d}"] = RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/{prim_prefix}_{i:02d}",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.42, -0.18, 0.38],
                rot=[1.0, 0.0, 0.0, 0.0],
            ),
            spawn=bead_spawn_cfg,
        )
    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)

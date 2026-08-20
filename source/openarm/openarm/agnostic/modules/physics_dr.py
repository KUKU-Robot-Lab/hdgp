"""물리 도메인 무작위화 — s2r 강건성용 EventTerm 묶음 + ADR 종점 범위.

grasp_v2 의 `EventCfg` + `adr_physics_cfg` 를 이식했다(DEXTRAH 원본 계보).

설계 규칙:
  **초기 범위는 전부 중립**이다(스케일 1 또는 0). 그래서 ADR 없이 켜도 물리가 변하지
  않고, ADR 을 붙이면 `TaskADR._expand_physics_ranges` 가 종점까지 선형 확장한다.
  → `enable_physics_dr` 를 켜는 것만으로는 학습이 흔들리지 않는다(안전한 기본값).

  `mode="reset"` 이라 에피소드 리셋마다 env 별로 재샘플링된다.

★함정: `SceneEntityCfg` 는 **가변 객체**다. 여러 term 이 같은 인스턴스를 공유하면
  IsaacLab 이 인덱스를 채우는 과정에서 서로를 덮어쓴다. term 마다 **새 인스턴스**를
  만들어야 한다(아래 팩토리가 그렇게 한다).

★자산 이름 규약: 로봇은 "robot", 물체는 `object_asset_name`(기본 "object").
  grasp_v2 는 "cup" 을 썼지만 이 트랙은 물체가 컵이 아닐 수 있어 일반 이름을 쓴다.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

# ADR 종점 범위 — grasp_v2 실측 운용값.
#   mass 0.5~3.0배, dynamic friction 0.3~1.0, joint gain 0.5~2.0배
# `TaskADR(physics_cfg=PHYSICS_ADR_TERMINAL)` 로 넘긴다.
PHYSICS_ADR_TERMINAL: dict = {
    "robot_physics_material": {
        "static_friction_range":  (0.5, 1.2),
        "dynamic_friction_range": (0.3, 1.0),
        "restitution_range":      (0.8, 1.0),
    },
    "robot_joint_stiffness_and_damping": {
        "stiffness_distribution_params": (0.5, 2.0),
        "damping_distribution_params":   (0.5, 2.0),
    },
    "robot_joint_friction": {
        "friction_distribution_params": (0.0, 5.0),
    },
    "object_physics_material": {
        "static_friction_range":  (0.5, 1.2),
        "dynamic_friction_range": (0.3, 1.0),
        "restitution_range":      (0.8, 1.0),
    },
    "object_scale_mass": {
        "mass_distribution_params": (0.5, 3.0),
    },
}

OBJECT_ASSET_NAME = "object"


def _robot(**kw) -> SceneEntityCfg:
    return SceneEntityCfg("robot", **kw)


def _object(**kw) -> SceneEntityCfg:
    return SceneEntityCfg(OBJECT_ASSET_NAME, **kw)


@configclass
class PhysicsDREventCfg:
    """초기 범위 = 중립. ADR 이 PHYSICS_ADR_TERMINAL 까지 확장한다."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": _robot(body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": _robot(joint_names=".*"),
            "stiffness_distribution_params": (1.0, 1.0),
            "damping_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    robot_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": _robot(joint_names=".*"),
            "friction_distribution_params": (0.0, 0.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": _object(body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": _object(),
            "mass_distribution_params": (1.0, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


def terminal_cfg_for(term_names=None) -> dict:
    """`TaskADR(physics_cfg=...)` 에 넘길 종점 범위.

    일부 term 만 ADR 에 올리고 싶으면 이름을 골라 준다.
    """
    if term_names is None:
        return dict(PHYSICS_ADR_TERMINAL)
    missing = set(term_names) - set(PHYSICS_ADR_TERMINAL)
    if missing:
        raise KeyError(f"종점 범위가 없는 DR term: {sorted(missing)}")
    return {k: PHYSICS_ADR_TERMINAL[k] for k in term_names}

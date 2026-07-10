"""Articulation-only replay 씬.

grasp env를 재사용하지 않는다. 물체/reward/fabric/reset이 얽히면 순수 actuator
tracking 측정에 잡음이 되고, grasp_v11의 legacy 이름 문제까지 끌고 들어온다.

이 모듈은 AppLauncher 이후에 import해야 한다 (isaaclab import 규칙).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from r2s_autotune.config import AutotuneConfig
from r2s_autotune.excitation import ExcitationSpec, interior_neutral
from r2s_autotune.joint_contract import JointContractError, Manifest


def build_articulation_cfg(config: AutotuneConfig) -> ArticulationCfg:
    actuators = {
        name: ImplicitActuatorCfg(
            joint_names_expr=list(group.joint_names_expr),
            stiffness=group.stiffness,
            damping=group.damping,
            friction=group.joint_friction,
        )
        for name, group in config.groups.items()
    }
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(config.usd_path),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(),
        actuators=actuators,
        soft_joint_pos_limit_factor=1.0,
    )


@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0),
    )
    robot: ArticulationCfg = None  # type: ignore[assignment]


def make_scene(config: AutotuneConfig, num_envs: int, env_spacing: float = 2.5) -> InteractiveScene:
    scene_cfg = ReplaySceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    scene_cfg.robot = build_articulation_cfg(config)
    return InteractiveScene(scene_cfg)


def verify_articulation(robot: Articulation, manifest: Manifest) -> None:
    """Isaac이 실제로 띄운 DOF 이름이 manifest movable set과 같은지 확인한다.

    USD가 canonical로 변환되지 않았다면 여기서 즉시 멈춘다. 그대로 진행하면
    regex가 0개 매칭되어 actuator가 사라지고, MSE는 낮은데 정책은 망가진다.
    """
    sim_joints = tuple(robot.joint_names)
    expected = frozenset(manifest.movable_joints)
    unexpected = [j for j in sim_joints if j not in expected]
    missing = [j for j in manifest.movable_joints if j not in frozenset(sim_joints)]
    if unexpected or missing:
        raise JointContractError(
            "Isaac articulation DOF names differ from manifest movable joints "
            f"(unexpected={unexpected[:8]}, missing={missing[:8]}). "
            "USD가 canonical 이름으로 변환되지 않았을 수 있다."
        )


def group_joint_indices(
    robot: Articulation,
    config: AutotuneConfig,
) -> Mapping[str, tuple[int, ...]]:
    """Isaac 자신의 find_joints로 group을 해석한다 (regex 의미 차이 방지)."""
    resolved: dict[str, tuple[int, ...]] = {}
    for name, group in config.groups.items():
        ids, _ = robot.find_joints(list(group.joint_names_expr), preserve_order=False)
        if not ids:
            raise JointContractError(f"group '{name}' matched no DOF in articulation")
        resolved[name] = tuple(int(i) for i in ids)
    return resolved


def apply_gains(
    robot: Articulation,
    stiffness: np.ndarray,
    damping: np.ndarray,
    friction: np.ndarray,
) -> None:
    """[K, J] gain 행렬을 env별로 sim에 쓴다. 후보 K개를 한 sim에서 병렬 평가하는 핵심."""
    device = robot.device
    expected = (robot.num_instances, robot.num_joints)
    for name, matrix in (("stiffness", stiffness), ("damping", damping), ("friction", friction)):
        if matrix.shape != expected:
            raise ValueError(f"{name} matrix must be {expected}, got {matrix.shape}")

    robot.write_joint_stiffness_to_sim(torch.as_tensor(stiffness, dtype=torch.float32, device=device))
    robot.write_joint_damping_to_sim(torch.as_tensor(damping, dtype=torch.float32, device=device))
    robot.write_joint_friction_coefficient_to_sim(
        torch.as_tensor(friction, dtype=torch.float32, device=device)
    )


def rest_pose(robot: Articulation, spec: ExcitationSpec) -> np.ndarray:
    """모든 관절의 안전한 기준 자세 [J].

    default_joint_pos를 그대로 쓰면 안 된다. Tesollo curl 관절은 default가 0인데
    하한도 0이라, 그 자리에서 구동하면 관절이 한계를 뚫고 나간다 (실측 -1.3 rad).
    """
    default = robot.data.default_joint_pos[0].detach().cpu().numpy()
    limits = robot.data.soft_joint_pos_limits[0].detach().cpu().numpy()
    return interior_neutral(default, limits[:, 0], limits[:, 1], spec)


def reset_and_settle(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    joint_pos: np.ndarray,
    steps: int = 100,
) -> None:
    """관절을 기준 자세로 강제 설정하고 정착시킨다.

    sim.reset()만으로는 articulation이 default_joint_pos에 있지 않다. 정착 없이
    replay를 시작하면 첫 구간의 과도응답이 actuator 특성이 아니라 초기 자세 오차를 반영한다.
    """
    target = torch.as_tensor(joint_pos, dtype=torch.float32, device=robot.device)
    target = target.unsqueeze(0).expand(robot.num_instances, -1).contiguous()

    robot.write_joint_state_to_sim(target, torch.zeros_like(target))
    robot.reset()
    dt = sim.get_physics_dt()
    for _ in range(steps):
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)


def build_full_targets(
    robot: Articulation,
    q_cmd: np.ndarray,
    tracked_joints: Sequence[str],
    hold_pose: np.ndarray,
) -> torch.Tensor:
    """추적 대상 관절만 q_cmd로 구동하고, 나머지는 hold_pose로 유지한다.

    Returns [T, J] (env 공통). 후보마다 같은 명령을 넣어야 error 비교가 성립한다.
    """
    num_steps = q_cmd.shape[0]
    targets = np.tile(np.asarray(hold_pose, dtype=np.float64), (num_steps, 1))

    name_to_index = {name: i for i, name in enumerate(robot.joint_names)}
    missing = [j for j in tracked_joints if j not in name_to_index]
    if missing:
        raise JointContractError(f"tracked joints absent from articulation: {missing}")

    columns = [name_to_index[j] for j in tracked_joints]
    targets[:, columns] = q_cmd
    return torch.as_tensor(targets, dtype=torch.float32, device=robot.device)


def replay(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    targets: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """모든 env에 동일한 target 시퀀스를 인가하고 [K, T, J] 응답을 기록한다."""
    num_steps = int(targets.shape[0])
    num_envs, num_joints = robot.num_instances, robot.num_joints
    dt = sim.get_physics_dt()

    q_sim = np.empty((num_envs, num_steps, num_joints), dtype=np.float64)
    dq_sim = np.empty_like(q_sim)

    for step in range(num_steps):
        command = targets[step].unsqueeze(0).expand(num_envs, -1)
        robot.set_joint_position_target(command)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)

        q_sim[:, step, :] = robot.data.joint_pos.detach().cpu().numpy()
        dq_sim[:, step, :] = robot.data.joint_vel.detach().cpu().numpy()

    return q_sim, dq_sim


def select_columns(
    data: np.ndarray,
    robot: Articulation,
    joints: Sequence[str],
) -> np.ndarray:
    """[K, T, J_full] → [K, T, len(joints)]"""
    name_to_index = {name: i for i, name in enumerate(robot.joint_names)}
    columns = [name_to_index[j] for j in joints]
    return data[..., columns]

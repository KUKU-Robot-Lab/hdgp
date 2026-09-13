"""Event terms of the IKER shoe task that Isaac Lab's stock terms do not cover."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform


def randomize_rigid_object_com(
    env: ManagerBasedEnv, env_ids: torch.Tensor | None, com_range: dict[str, tuple[float, float]], asset_cfg: SceneEntityCfg
) -> None:
    """Add a uniform offset to a single-body RigidObject's centre of mass.

    ``isaaclab.envs.mdp.randomize_rigid_body_com`` indexes CoMs as (envs, bodies, 7), the articulation layout;
    a RigidObject view returns (envs, 7) and that term raises IndexError.
    """
    asset = env.scene[asset_cfg.name]
    ids = torch.arange(env.scene.num_envs, device="cpu") if env_ids is None else env_ids.cpu()
    bounds = torch.tensor([com_range.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z")], device="cpu")
    offsets = sample_uniform(bounds[:, 0], bounds[:, 1], (len(ids), 3), device="cpu")
    coms = asset.root_physx_view.get_coms().clone()
    if coms.ndim != 2 or coms.shape[1] != 7:
        raise ValueError(f"{asset_cfg.name}: expected (envs, 7) CoMs from a single-body view, got {tuple(coms.shape)}")
    coms[ids, :3] += offsets
    asset.root_physx_view.set_coms(coms, ids)

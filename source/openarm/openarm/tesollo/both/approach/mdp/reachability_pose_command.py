from __future__ import annotations

import os
from dataclasses import MISSING

import numpy as np
import torch

from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply_inverse,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_from_matrix,
    quat_mul,
    quat_unique,
)


def _erode_mask(mask: np.ndarray, steps: int) -> np.ndarray:
    """Erode a 3D boolean mask using 6-neighborhood steps."""
    if steps <= 0:
        return mask
    eroded = mask.copy()
    for _ in range(steps):
        if eroded.shape[0] < 3 or eroded.shape[1] < 3 or eroded.shape[2] < 3:
            return np.zeros_like(eroded, dtype=bool)
        inner = (
            eroded[1:-1, 1:-1, 1:-1]
            & eroded[:-2, 1:-1, 1:-1]
            & eroded[2:, 1:-1, 1:-1]
            & eroded[1:-1, :-2, 1:-1]
            & eroded[1:-1, 2:, 1:-1]
            & eroded[1:-1, 1:-1, :-2]
            & eroded[1:-1, 1:-1, 2:]
        )
        eroded[:] = False
        eroded[1:-1, 1:-1, 1:-1] = inner
    return eroded


def _angle_in_range(angle: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Return True for angles within a [min, max] range, handling wrap-around."""
    if min_val <= max_val:
        return (angle >= min_val) & (angle <= max_val)
    return (angle >= min_val) | (angle <= max_val)


def _quat_from_z_axis(z_axis: torch.Tensor, roll: torch.Tensor | None = None) -> torch.Tensor:
    """Create a quaternion with the body z-axis aligned to the given direction."""
    z_axis = z_axis / torch.clamp_min(torch.linalg.norm(z_axis, dim=-1, keepdim=True), 1.0e-8)
    ref_x = torch.tensor([1.0, 0.0, 0.0], device=z_axis.device, dtype=z_axis.dtype)
    ref_y = torch.tensor([0.0, 1.0, 0.0], device=z_axis.device, dtype=z_axis.dtype)
    ref = ref_x.expand_as(z_axis)
    use_ref_y = torch.abs(torch.sum(z_axis * ref, dim=-1, keepdim=True)) > 0.95
    ref = torch.where(use_ref_y, ref_y.expand_as(z_axis), ref)
    y_axis = torch.cross(z_axis, ref, dim=-1)
    y_axis = y_axis / torch.clamp_min(torch.linalg.norm(y_axis, dim=-1, keepdim=True), 1.0e-8)
    x_axis = torch.cross(y_axis, z_axis, dim=-1)
    rot = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    quat = quat_from_matrix(rot)
    if roll is not None:
        roll_quat = quat_from_angle_axis(roll, z_axis)
        quat = quat_mul(roll_quat, quat)
    return quat


class ReachabilityPoseCommand(UniformPoseCommand):
    """Pose command generator that samples positions only from reachable voxels."""

    cfg: ReachabilityPoseCommandCfg

    def __init__(self, cfg: ReachabilityPoseCommandCfg, env):
        super().__init__(cfg, env)

        if not os.path.isfile(cfg.voxel_npz_path):
            raise FileNotFoundError(f"Voxel file not found: {cfg.voxel_npz_path}")

        data = np.load(cfg.voxel_npz_path, allow_pickle=True)
        ee_body_names = data["ee_body_names"].tolist()
        reach_mesh = data["reach_mesh"]
        grid_min = data["grid_min"].astype(np.float32)
        grid_max = data["grid_max"].astype(np.float32)
        voxels_per_dim = int(data["voxels_per_dim"])

        if cfg.ee_name not in ee_body_names:
            raise ValueError(f"EE name '{cfg.ee_name}' not in voxel data: {ee_body_names}")

        ee_index = ee_body_names.index(cfg.ee_name)
        mask = reach_mesh[ee_index] > 0
        z_axis_bins = None
        if cfg.use_z_axis_bins:
            if "z_axis_bins" in data:
                z_axis_bins = data["z_axis_bins"][ee_index]
                yaw_edges = data["z_axis_yaw_edges"].astype(np.float32)
                pitch_edges = data["z_axis_pitch_edges"].astype(np.float32)
            else:
                print("[WARN] z_axis_bins not found in voxel data. Falling back to uniform orientations.")

        if cfg.inner_margin > 0.0 and voxels_per_dim > 1:
            spacing = (grid_max - grid_min) / (voxels_per_dim - 1)
            min_spacing = float(np.min(spacing))
            margin_steps = int(np.ceil(cfg.inner_margin / max(min_spacing, 1.0e-9)))
            mask = _erode_mask(mask, margin_steps)

        self._z_axis_bin_choices = None
        self._z_axis_pitch_bins = 0
        self._z_axis_yaw_bins = 0
        self._z_axis_yaw_edges = None
        self._z_axis_pitch_edges = None
        if z_axis_bins is not None:
            yaw_bins = int(z_axis_bins.shape[-2])
            pitch_bins = int(z_axis_bins.shape[-1])
            yaw_centers = 0.5 * (yaw_edges[:-1] + yaw_edges[1:])
            pitch_centers = 0.5 * (pitch_edges[:-1] + pitch_edges[1:])
            yaw_ok = _angle_in_range(yaw_centers, cfg.ranges.yaw[0], cfg.ranges.yaw[1])
            pitch_ok = _angle_in_range(pitch_centers, cfg.ranges.pitch[0], cfg.ranges.pitch[1])
            allowed_bins = (yaw_ok[:, None] & pitch_ok[None, :]).reshape(-1)

            flat_bins = z_axis_bins.reshape(-1, yaw_bins * pitch_bins)
            if not np.all(allowed_bins):
                flat_bins = flat_bins & allowed_bins[None, :]
            self._z_axis_bin_choices = [np.flatnonzero(flat_bins[i]) for i in range(flat_bins.shape[0])]
            bin_available = np.array([choices.size > 0 for choices in self._z_axis_bin_choices], dtype=bool)
            bin_available = bin_available.reshape(z_axis_bins.shape[:3])
            if cfg.z_axis_bins_fallback == "resample":
                mask = mask & bin_available
            self._z_axis_pitch_bins = pitch_bins
            self._z_axis_yaw_bins = yaw_bins
            self._z_axis_yaw_edges = torch.tensor(yaw_edges, device=self.device)
            self._z_axis_pitch_edges = torch.tensor(pitch_edges, device=self.device)

        occupied = np.argwhere(mask)
        if occupied.size == 0:
            raise ValueError(f"No reachable voxels for EE '{cfg.ee_name}' after applying inner_margin.")

        self._voxel_indices = torch.tensor(occupied, device=self.device, dtype=torch.int64)
        self._grid_min = torch.tensor(grid_min, device=self.device)
        self._grid_max = torch.tensor(grid_max, device=self.device)
        self._voxels_per_dim = voxels_per_dim
        if voxels_per_dim > 1:
            self._voxel_spacing = (self._grid_max - self._grid_min) / (voxels_per_dim - 1)
        else:
            self._voxel_spacing = torch.zeros_like(self._grid_min)

    def _resample_command(self, env_ids):
        # sample reachable voxel centers in configured voxel frame
        num_ids = len(env_ids)
        rand_idx = torch.randint(0, self._voxel_indices.shape[0], (num_ids,), device=self.device)
        voxel_coords = self._voxel_indices[rand_idx].to(torch.float32)
        centers = self._grid_min + voxel_coords * self._voxel_spacing
        if self.cfg.jitter:
            jitter = (torch.rand_like(centers) - 0.5) * self._voxel_spacing
            centers = centers + jitter
        if self.cfg.mirror_axis is not None:
            axis_map = {"x": 0, "y": 1, "z": 2}
            axis = axis_map.get(self.cfg.mirror_axis.lower())
            if axis is None:
                raise ValueError(f"Unsupported mirror_axis: {self.cfg.mirror_axis}")
            centers[:, axis] = 2.0 * self.cfg.mirror_center - centers[:, axis]

        if self.cfg.voxel_frame == "world":
            # convert from world to base frame for command storage
            root_pos_w = self.robot.data.root_pos_w[env_ids]
            root_quat_w = self.robot.data.root_quat_w[env_ids]
            centers = quat_apply_inverse(root_quat_w, centers - root_pos_w)

        self.pose_command_b[env_ids, 0:3] = centers

        if self._z_axis_bin_choices is None:
            # orientation sampling (same as UniformPoseCommand)
            euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
            euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
            euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
            euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
            quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
            self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat
            return

        voxel_coords_cpu = voxel_coords.to(torch.int64).cpu().numpy()
        lin_idx = (
            voxel_coords_cpu[:, 0] * (self._voxels_per_dim * self._voxels_per_dim)
            + voxel_coords_cpu[:, 1] * self._voxels_per_dim
            + voxel_coords_cpu[:, 2]
        )
        bin_ids = torch.full((num_ids,), -1, device=self.device, dtype=torch.long)
        use_uniform = torch.zeros((num_ids,), device=self.device, dtype=torch.bool)
        for i, flat_idx in enumerate(lin_idx):
            choices = self._z_axis_bin_choices[int(flat_idx)]
            if choices.size == 0:
                use_uniform[i] = True
            else:
                choice = int(choices[np.random.randint(choices.size)])
                bin_ids[i] = choice

        if torch.any(~use_uniform):
            idx = torch.nonzero(~use_uniform, as_tuple=False).squeeze(-1)
            yaw_idx = bin_ids[idx] // self._z_axis_pitch_bins
            pitch_idx = bin_ids[idx] % self._z_axis_pitch_bins
            yaw_low = self._z_axis_yaw_edges[yaw_idx]
            yaw_high = self._z_axis_yaw_edges[yaw_idx + 1]
            pitch_low = self._z_axis_pitch_edges[pitch_idx]
            pitch_high = self._z_axis_pitch_edges[pitch_idx + 1]
            yaw = yaw_low + (yaw_high - yaw_low) * torch.rand_like(yaw_low)
            pitch = pitch_low + (pitch_high - pitch_low) * torch.rand_like(pitch_low)
            z_axis = torch.stack(
                (
                    torch.cos(pitch) * torch.cos(yaw),
                    torch.cos(pitch) * torch.sin(yaw),
                    torch.sin(pitch),
                ),
                dim=-1,
            )
            roll = torch.empty_like(yaw).uniform_(*self.cfg.ranges.roll)
            quat = _quat_from_z_axis(z_axis, roll=roll)
            self.pose_command_b[env_ids[idx], 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

        if torch.any(use_uniform):
            idx = torch.nonzero(use_uniform, as_tuple=False).squeeze(-1)
            euler_angles = torch.zeros((idx.shape[0], 3), device=self.device, dtype=self.pose_command_b.dtype)
            euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
            euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
            euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
            quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
            self.pose_command_b[env_ids[idx], 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat


@configclass
class ReachabilityPoseCommandCfg(UniformPoseCommandCfg):
    """Configuration for reachability-aware pose command generator."""

    class_type: type = ReachabilityPoseCommand

    voxel_npz_path: str = MISSING
    """Path to voxel data (.npz) generated by compute_voxel_workspace.py."""

    ee_name: str = MISSING
    """End-effector name to select inside the voxel data."""

    jitter: bool = True
    """Jitter sample within voxel spacing if True."""

    inner_margin: float = 0.0
    """Shrink reachable voxels inward by this margin (in meters)."""

    mirror_axis: str | None = None
    """Mirror samples across a plane normal to this axis ("x", "y", or "z")."""

    mirror_center: float = 0.0
    """Mirror plane center along the chosen axis (in meters)."""

    voxel_frame: str = "base"
    """Frame of voxel coordinates: "base" (per-env robot base) or "world"."""

    use_z_axis_bins: bool = False
    """If True, sample orientations from z-axis bins stored in voxel data."""

    z_axis_bins_fallback: str = "uniform"
    """Fallback when a voxel has no z-axis bins: "uniform" or "resample"."""

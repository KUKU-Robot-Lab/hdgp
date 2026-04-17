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


import torch


ACTOR_OBSERVATION_GROUP_DIMS: dict[str, int] = {
    "right_joint_pos": 27,
    "right_joint_vel": 27,
    "left_arm_joint_pos": 7,
    "left_arm_joint_vel": 7,
    "fingertip_pos": 15,
    "cup_pose_vel": 13,
    "target_opening_pos": 3,
    "bead_centroid_pos": 3,
    "prev_actions": 18,
    "mouth_delta": 3,
    "mouth_xy_distance": 1,
    "mouth_z_clearance": 1,
    "source_up_dot_world": 1,
    "directional_tilt_cos": 1,
    "mouth_alignment_cos": 1,
    "bead_cross_fraction": 1,
    "bead_in_target_fraction": 1,
    "bead_in_source_fraction": 1,
    "spill_ratio": 1,
    "g_ready": 1,
    "g_pour": 1,
}
ACTOR_OBSERVATION_DIM = sum(ACTOR_OBSERVATION_GROUP_DIMS.values())


@torch.jit.script
def scale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """[-1, 1] 정규화 액션을 [lower, upper] 범위로 스케일."""
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def tensor_clamp(t: torch.Tensor, min_t: torch.Tensor, max_t: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.min(t, max_t), min_t)


def to_torch(x, dtype=torch.float, device: str = "cuda:0", requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def assemble_actor_observation(**group_tensors: torch.Tensor) -> torch.Tensor:
    """Assemble the pour_v1 actor observation in the documented group order."""
    expected_keys = tuple(ACTOR_OBSERVATION_GROUP_DIMS.keys())
    missing = [key for key in expected_keys if key not in group_tensors]
    extra = [key for key in group_tensors if key not in ACTOR_OBSERVATION_GROUP_DIMS]
    if missing or extra:
        raise ValueError(f"Observation groups mismatch: missing={missing}, extra={extra}")

    batch_size = None
    device = None
    dtype = None
    ordered_groups = []
    for key in expected_keys:
        tensor = group_tensors[key]
        expected_dim = ACTOR_OBSERVATION_GROUP_DIMS[key]
        if tensor.ndim != 2:
            raise ValueError(f"{key} must be rank-2, got shape {tuple(tensor.shape)}")
        if tensor.shape[1] != expected_dim:
            raise ValueError(f"{key} dim mismatch: {tensor.shape[1]} != {expected_dim}")
        if batch_size is None:
            batch_size = tensor.shape[0]
            device = tensor.device
            dtype = tensor.dtype
        if tensor.shape[0] != batch_size:
            raise ValueError(f"{key} batch mismatch: {tensor.shape[0]} != {batch_size}")
        if tensor.device != device:
            raise ValueError(f"{key} device mismatch: {tensor.device} != {device}")
        if tensor.dtype != dtype:
            raise ValueError(f"{key} dtype mismatch: {tensor.dtype} != {dtype}")
        ordered_groups.append(tensor)

    actor_obs = torch.cat(ordered_groups, dim=-1)
    if actor_obs.shape[1] != ACTOR_OBSERVATION_DIM:
        raise ValueError(
            f"Actor observation dim mismatch: {actor_obs.shape[1]} != {ACTOR_OBSERVATION_DIM}"
        )
    return actor_obs

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


_TASK_DIR = Path(__file__).resolve().parent.parent
_UTILS_PATH = _TASK_DIR / "pour_utils.py"
_SPEC = importlib.util.spec_from_file_location("pour_v1_pour_utils", _UTILS_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

assemble_actor_observation = _MODULE.assemble_actor_observation


def _mock_group(batch: int, dim: int, start: float) -> torch.Tensor:
    values = torch.arange(start, start + batch * dim, dtype=torch.float32)
    return values.reshape(batch, dim)


class TestObservationAssembly:
    def test_assembles_all_groups_in_design_order(self):
        batch = 2
        groups = {
            "right_joint_pos": _mock_group(batch, 27, 0),
            "right_joint_vel": _mock_group(batch, 27, 100),
            "left_arm_joint_pos": _mock_group(batch, 7, 200),
            "left_arm_joint_vel": _mock_group(batch, 7, 300),
            "fingertip_pos": _mock_group(batch, 15, 400),
            "cup_pose_vel": _mock_group(batch, 13, 500),
            "target_opening_pos": _mock_group(batch, 3, 600),
            "bead_centroid_pos": _mock_group(batch, 3, 700),
            "prev_actions": _mock_group(batch, 18, 800),
            "mouth_delta": _mock_group(batch, 3, 900),
            "mouth_xy_distance": _mock_group(batch, 1, 1000),
            "mouth_z_clearance": _mock_group(batch, 1, 1100),
            "source_up_dot_world": _mock_group(batch, 1, 1200),
            "directional_tilt_cos": _mock_group(batch, 1, 1300),
            "mouth_alignment_cos": _mock_group(batch, 1, 1400),
            "bead_cross_fraction": _mock_group(batch, 1, 1500),
            "bead_in_target_fraction": _mock_group(batch, 1, 1600),
            "bead_in_source_fraction": _mock_group(batch, 1, 1700),
            "spill_ratio": _mock_group(batch, 1, 1800),
            "g_ready": _mock_group(batch, 1, 1900),
            "g_pour": _mock_group(batch, 1, 2000),
        }

        obs = assemble_actor_observation(**groups)

        assert obs.shape == (batch, 134)
        assert torch.equal(obs[:, 0:27], groups["right_joint_pos"])
        assert torch.equal(obs[:, 27:54], groups["right_joint_vel"])
        assert torch.equal(obs[:, 54:61], groups["left_arm_joint_pos"])
        assert torch.equal(obs[:, 61:68], groups["left_arm_joint_vel"])
        assert torch.equal(obs[:, 68:83], groups["fingertip_pos"])
        assert torch.equal(obs[:, 83:96], groups["cup_pose_vel"])
        assert torch.equal(obs[:, 96:99], groups["target_opening_pos"])
        assert torch.equal(obs[:, 99:102], groups["bead_centroid_pos"])
        assert torch.equal(obs[:, 102:120], groups["prev_actions"])
        assert torch.equal(obs[:, 120:123], groups["mouth_delta"])
        assert torch.equal(obs[:, 123:124], groups["mouth_xy_distance"])
        assert torch.equal(obs[:, 124:125], groups["mouth_z_clearance"])
        assert torch.equal(obs[:, 125:126], groups["source_up_dot_world"])
        assert torch.equal(obs[:, 126:127], groups["directional_tilt_cos"])
        assert torch.equal(obs[:, 127:128], groups["mouth_alignment_cos"])
        assert torch.equal(obs[:, 128:129], groups["bead_cross_fraction"])
        assert torch.equal(obs[:, 129:130], groups["bead_in_target_fraction"])
        assert torch.equal(obs[:, 130:131], groups["bead_in_source_fraction"])
        assert torch.equal(obs[:, 131:132], groups["spill_ratio"])
        assert torch.equal(obs[:, 132:133], groups["g_ready"])
        assert torch.equal(obs[:, 133:134], groups["g_pour"])

    def test_rejects_wrong_group_dimension(self):
        batch = 1
        groups = {
            "right_joint_pos": _mock_group(batch, 26, 0),
            "right_joint_vel": _mock_group(batch, 27, 100),
            "left_arm_joint_pos": _mock_group(batch, 7, 200),
            "left_arm_joint_vel": _mock_group(batch, 7, 300),
            "fingertip_pos": _mock_group(batch, 15, 400),
            "cup_pose_vel": _mock_group(batch, 13, 500),
            "target_opening_pos": _mock_group(batch, 3, 600),
            "bead_centroid_pos": _mock_group(batch, 3, 700),
            "prev_actions": _mock_group(batch, 18, 800),
            "mouth_delta": _mock_group(batch, 3, 900),
            "mouth_xy_distance": _mock_group(batch, 1, 1000),
            "mouth_z_clearance": _mock_group(batch, 1, 1100),
            "source_up_dot_world": _mock_group(batch, 1, 1200),
            "directional_tilt_cos": _mock_group(batch, 1, 1300),
            "mouth_alignment_cos": _mock_group(batch, 1, 1400),
            "bead_cross_fraction": _mock_group(batch, 1, 1500),
            "bead_in_target_fraction": _mock_group(batch, 1, 1600),
            "bead_in_source_fraction": _mock_group(batch, 1, 1700),
            "spill_ratio": _mock_group(batch, 1, 1800),
            "g_ready": _mock_group(batch, 1, 1900),
            "g_pour": _mock_group(batch, 1, 2000),
        }

        with pytest.raises(ValueError, match="right_joint_pos"):
            assemble_actor_observation(**groups)

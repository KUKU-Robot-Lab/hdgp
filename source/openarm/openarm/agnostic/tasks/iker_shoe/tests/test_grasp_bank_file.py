"""The committed grasp bank holds no finger pinned at its opposite or beyond limits (no Isaac)."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import torch

from openarm.agnostic.modules import robot_profiles
from openarm.agnostic.modules.iker import run_files
from openarm.agnostic.tasks.iker_shoe import grasp_bank as gb
from openarm.agnostic.tasks.iker_shoe import layout


def _hand_joint_limits(urdf_path: Path, names: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
    root = ET.parse(urdf_path).getroot()
    limits = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in names:
            continue
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"{urdf_path}: joint {name!r} has no <limit>")
        limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
    missing = [n for n in names if n not in limits]
    if missing:
        raise ValueError(f"{urdf_path}: URDF lacks joints {missing}")
    lower = torch.tensor([limits[n][0] for n in names])
    upper = torch.tensor([limits[n][1] for n in names])
    return lower, upper


def test_bank_entries_have_no_finger_at_an_opposite_or_beyond_limit():
    doc = run_files.read_json(layout.RUNS_DIR / "config_00" / "grasp_bank.json")
    profile = robot_profiles.PROFILES[layout.PROFILE_NAME]

    urdf_path = layout.HDGP_ROOT / "assets" / Path(profile.usd_relpath).with_suffix(".urdf")
    assert urdf_path.is_file(), f"missing URDF for profile {profile.name!r}: {urdf_path}"

    thumb3_idx = profile.hand_joint_names.index("r_hj_thumb_3")
    grip_thumb3 = profile.hand_grip_pose[thumb3_idx]
    lo, hi = gb.THUMB3_RANGE
    assert not (lo <= grip_thumb3 <= hi), (
        f"NEEDS_CONTEXT: THUMB3_RANGE {gb.THUMB3_RANGE} straddles the grip value {grip_thumb3} "
        "for r_hj_thumb_3 -- the start pose used here would not have an unambiguous closing direction"
    )

    lower, upper = _hand_joint_limits(urdf_path, profile.hand_joint_names)
    grip_pose = torch.tensor(profile.hand_grip_pose)

    hand_columns = [doc["joint_names"].index(n) for n in profile.hand_joint_names]
    joint_pos = torch.tensor(doc["entries"]["joint_pos"], dtype=torch.float32)[:, hand_columns]

    start_row = list(profile.hand_open_pose)
    start_row[thumb3_idx] = (lo + hi) / 2.0
    start_pose = torch.tensor(start_row, dtype=torch.float32).unsqueeze(0).expand(joint_pos.shape[0], -1)

    valid = gb.hand_state_valid(joint_pos, start_pose, grip_pose, lower, upper)
    assert valid.all(), f"{int((~valid).sum())} of {valid.numel()} bank entries have a finger at an opposite or beyond limit"


def test_bank_was_built_for_the_selected_robot():
    doc = run_files.read_json(layout.RUNS_DIR / "config_00" / "grasp_bank.json")
    profile = robot_profiles.PROFILES[layout.PROFILE_NAME]
    assert doc["metadata"]["robot_usd"] == str(profile.usd_relpath)

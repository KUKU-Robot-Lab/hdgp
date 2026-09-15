"""The committed grasp bank holds no finger pinned at its opposite or beyond limits (no Isaac)."""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import pytest
import torch

from openarm.agnostic.modules import robot_profiles
from openarm.agnostic.modules.iker import run_files
from openarm.agnostic.tasks.iker_shoe import grasp_bank as gb
from openarm.agnostic.tasks.iker_shoe import layout

BANK_PATH = layout.RUNS_DIR / "config_00" / "grasp_bank.json"


def _read_bank() -> dict:
    if not BANK_PATH.is_file():
        pytest.skip(f"{BANK_PATH} does not exist yet — the auto loop harvests it (auto-loop spec §13)")
    return run_files.read_json(BANK_PATH)


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


def _skip_if_bank_is_for_the_other_arm(doc, profile) -> None:
    # Banks record the grasping arm's side since 2026-09-14; older banks were all built with the right arm.
    bank_side = float(doc["metadata"].get("side_sign", 1.0))
    if bank_side != gb.side_sign(profile.hand_joint_names):
        pytest.skip(
            f"grasp_bank.json was built for side_sign {bank_side} but {profile.name} grasps with the other arm -- "
            "the right-arm K1 bank stays committed until plan 2c replaces it with the learned left-arm bank"
        )


def test_bank_entries_have_no_finger_at_an_opposite_or_beyond_limit():
    doc = _read_bank()
    profile = robot_profiles.PROFILES[layout.PROFILE_NAME]
    _skip_if_bank_is_for_the_other_arm(doc, profile)

    urdf_path = layout.HDGP_ROOT / "assets" / Path(profile.usd_relpath).with_suffix(".urdf")
    assert urdf_path.is_file(), f"missing URDF for profile {profile.name!r}: {urdf_path}"

    thumb3_name = gb.hand_joint_name(profile.hand_joint_names, "thumb", 3)
    thumb3_idx = profile.hand_joint_names.index(thumb3_name)
    grip_thumb3 = profile.hand_grip_pose[thumb3_idx]
    # The left hand is the right one mirrored: its thumb_3 pre-curl range carries the side sign.
    sign = gb.side_sign(profile.hand_joint_names)
    lo, hi = sorted((sign * gb.THUMB3_RANGE[0], sign * gb.THUMB3_RANGE[1]))
    assert not (lo <= grip_thumb3 <= hi), (
        f"NEEDS_CONTEXT: THUMB3_RANGE {(lo, hi)} straddles the grip value {grip_thumb3} "
        f"for {thumb3_name} -- the start pose used here would not have an unambiguous closing direction"
    )

    lower, upper = _hand_joint_limits(urdf_path, profile.hand_joint_names)
    grip_pose = torch.tensor(profile.hand_grip_pose)

    hand_columns = [doc["joint_names"].index(n) for n in profile.hand_joint_names]
    joint_pos = torch.tensor(doc["entries"]["joint_pos"], dtype=torch.float32)[:, hand_columns]

    start_row = list(profile.hand_open_pose)
    start_row[thumb3_idx] = (lo + hi) / 2.0
    start_pose = torch.tensor(start_row, dtype=torch.float32).unsqueeze(0).expand(joint_pos.shape[0], -1)

    # The bank's real r_hj_thumb_3 start varies within THUMB3_RANGE, so the midpoint start used above is off
    # by up to half the range -- widen the back-bend allowance for that joint only.
    thumb3_slack = (hi - lo) / 2.0
    others = [i for i in range(len(profile.hand_joint_names)) if i != thumb3_idx]
    worst_other = gb.worst_backbend(joint_pos[:, others], start_pose[:, others], grip_pose[others])
    worst_thumb3 = gb.worst_backbend(joint_pos[:, [thumb3_idx]], start_pose[:, [thumb3_idx]], grip_pose[[thumb3_idx]])
    hand_ok = gb.hand_state_valid(joint_pos, start_pose, grip_pose, lower, upper)
    valid = hand_ok & (worst_other <= gb.MAX_BACKBEND_RAD) & (worst_thumb3 <= gb.MAX_BACKBEND_RAD + thumb3_slack)
    assert valid.all(), f"{int((~valid).sum())} of {valid.numel()} bank entries have a finger at an opposite/beyond limit or excess back-bend"


def test_bank_was_built_for_the_selected_robot():
    doc = _read_bank()
    profile = robot_profiles.PROFILES[layout.PROFILE_NAME]
    assert doc["metadata"]["robot_usd"] == str(profile.usd_relpath)

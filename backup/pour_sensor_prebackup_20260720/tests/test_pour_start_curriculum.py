"""Unit tests for the pour-start reset curriculum geometry.

These tests exercise only pure-torch geometry helpers (no Isaac Sim), so they
run under plain ``python3`` with torch installed.

Goal of the curriculum (axis 1 of the redesign): a fraction of envs reset into a
configuration where the source cup is already tilted *over the target opening*,
so the policy experiences ``bead_in`` reward from step one. The geometry below
must:

  1. place the source pour-point directly above the target opening (zband);
  2. produce a cup up-axis whose dot with world-up equals cos(tilt);
  3. tip the cup *toward* the target (not away);
  4. preserve the rigid grasp transform (palm<->cup relative pose unchanged);
  5. expose a curriculum ratio that anneals monotonically from start to end.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "pour_start_curriculum.py"
SPEC = importlib.util.spec_from_file_location("pour_start_curriculum", MODULE_PATH)
psc = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = psc
SPEC.loader.exec_module(psc)

compute_pour_start_pose = psc.compute_pour_start_pose
pour_start_ratio = psc.pour_start_ratio


def _quat_rotate_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q (w,x,y,z). Local reimplementation."""
    w = q[:, 0:1]
    xyz = q[:, 1:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _make_inputs(n: int = 4, target_dir: str = "+x"):
    cup_pos = torch.zeros(n, 3)
    cup_pos[:, 0] = 0.30
    cup_pos[:, 1] = 0.00
    cup_pos[:, 2] = 0.45
    # palm sits a fixed offset from the cup (the grasp).
    palm_pos = cup_pos + torch.tensor([0.0, -0.08, 0.05]).expand(n, -1)
    # palm orientation: identity for simplicity (xyzw).
    palm_quat_xyzw = torch.zeros(n, 4)
    palm_quat_xyzw[:, 3] = 1.0
    rim_b = torch.tensor([0.0, 0.0, 0.06])  # pour point above cup center in body frame
    target_opening = torch.zeros(n, 3)
    if target_dir == "+x":
        target_opening[:, 0] = 0.10
    else:
        target_opening[:, 1] = 0.10
    target_opening[:, 2] = 0.42
    return cup_pos, palm_pos, palm_quat_xyzw, rim_b, target_opening


def test_pour_point_lands_over_target_with_zband():
    cup_pos, palm_pos, palm_quat, rim_b, target_opening = _make_inputs()
    zband = 0.05
    out = compute_pour_start_pose(
        cup_pos, palm_pos, palm_quat, rim_b, target_opening,
        tilt_deg=100.0, z_clearance=zband,
    )
    new_cup_pos = out["cup_pos"]
    new_cup_quat = out["cup_quat_wxyz"]
    pour_point = new_cup_pos + _quat_rotate_wxyz(new_cup_quat, rim_b.expand_as(new_cup_pos))
    expected = target_opening.clone()
    expected[:, 2] += zband
    assert torch.allclose(pour_point, expected, atol=1e-5), (pour_point, expected)


def test_up_dot_matches_tilt_angle():
    cup_pos, palm_pos, palm_quat, rim_b, target_opening = _make_inputs()
    tilt = 100.0
    out = compute_pour_start_pose(
        cup_pos, palm_pos, palm_quat, rim_b, target_opening,
        tilt_deg=tilt, z_clearance=0.05,
    )
    world_up = torch.tensor([0.0, 0.0, 1.0]).expand_as(cup_pos)
    cup_up = _quat_rotate_wxyz(out["cup_quat_wxyz"], world_up)
    up_dot = (cup_up * world_up).sum(dim=-1)
    assert torch.allclose(up_dot, torch.full_like(up_dot, math.cos(math.radians(tilt))), atol=1e-5)


def test_tilt_is_toward_target():
    cup_pos, palm_pos, palm_quat, rim_b, target_opening = _make_inputs(target_dir="+x")
    out = compute_pour_start_pose(
        cup_pos, palm_pos, palm_quat, rim_b, target_opening,
        tilt_deg=90.0, z_clearance=0.05,
    )
    world_up = torch.tensor([0.0, 0.0, 1.0]).expand_as(cup_pos)
    cup_up = _quat_rotate_wxyz(out["cup_quat_wxyz"], world_up)
    # At 90deg the cup up-axis lies horizontal; it must point toward the target
    # (horizontal direction from the original pour point to the target opening).
    pour_point0 = cup_pos + rim_b.expand_as(cup_pos)
    horiz = (target_opening - pour_point0).clone()
    horiz[:, 2] = 0.0
    dir_xy = horiz / horiz.norm(dim=-1, keepdim=True)
    align = (cup_up[:, :2] * dir_xy[:, :2]).sum(dim=-1)
    assert (align > 0.9).all(), (cup_up, dir_xy, align)


def test_rigid_grasp_transform_preserved():
    cup_pos, palm_pos, palm_quat, rim_b, target_opening = _make_inputs()
    out = compute_pour_start_pose(
        cup_pos, palm_pos, palm_quat, rim_b, target_opening,
        tilt_deg=110.0, z_clearance=0.04,
    )
    before = torch.norm(palm_pos - cup_pos, dim=-1)
    after = torch.norm(out["palm_pos"] - out["cup_pos"], dim=-1)
    assert torch.allclose(before, after, atol=1e-5), (before, after)


def test_ratio_anneals_monotonically():
    start, end = 0.6, 0.0
    a0 = pour_start_ratio(0, start, end, anneal_start_step=0, anneal_steps=1000)
    a_mid = pour_start_ratio(500, start, end, anneal_start_step=0, anneal_steps=1000)
    a_end = pour_start_ratio(1000, start, end, anneal_start_step=0, anneal_steps=1000)
    a_past = pour_start_ratio(5000, start, end, anneal_start_step=0, anneal_steps=1000)
    assert abs(a0 - start) < 1e-6
    assert end < a_mid < start
    assert abs(a_end - end) < 1e-6
    assert abs(a_past - end) < 1e-6


def test_ratio_holds_before_anneal_start():
    a = pour_start_ratio(200, 0.5, 0.0, anneal_start_step=500, anneal_steps=1000)
    assert abs(a - 0.5) < 1e-6


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if failures else 0)

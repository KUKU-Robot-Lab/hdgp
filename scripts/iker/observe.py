"""Head-camera snapshot of an executed IKER scene (design 2026-09-14-iker-auto-loop §4 observe_requery, §9).

``--states <final_states.json> --env K`` places the robot joints, their targets and both shoes at env K's final state of a
noise-free rollout (eval_iker.py --final-states). ``--human-target`` places the moving shoe at the human baseline target
pose with the robot at home instead (the observation smoke). The head holds the snapshot angles while the scene settles;
the render is annotated as in snapshot.py and written, with the placed state (state.json), to --out-dir. Every snapshot
check applies except settling, which is reported only: a closed hand may still be adjusting its hold.

Usage:
    cd ~/rl_ws/hdgp-iker && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm ../IsaacLab/_isaac_sim/python.sh \
        scripts/iker/observe.py --states <final_states.json> --env <k> --out-dir <dir> --headless
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Head-camera snapshot of an executed IKER scene.")
parser.add_argument("--config-index", type=int, default=0)
parser.add_argument("--states", default="", help="final_states.json written by eval_iker.py --final-states")
parser.add_argument("--env", type=int, default=-1, help="the env of --states to place")
parser.add_argument("--human-target", action="store_true", help="place the moving shoe at the human baseline target instead")
parser.add_argument("--out-dir", required=True)
parser.add_argument("--settle-steps", type=int, default=120)
parser.add_argument("--render-steps", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    print("OBSERVE FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from openarm.agnostic.modules.iker import run_files  # noqa: E402
from openarm.agnostic.modules.iker.gate import kabsch  # noqa: E402
from openarm.agnostic.modules.iker.rotations import matrix_to_quat_wxyz  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import head_snapshot as hs  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import layout  # noqa: E402

REPORTED_CHECKS = ("settle",)
SHOE_KEYS = ((layout.MOVING_SHOE, "shoe_move"), (layout.OTHER_SHOE, "shoe_other"))


def rollout_state(path: Path, env: int) -> dict:
    doc = run_files.read_json(path)
    row = next((row for row in doc["rows"] if row["env"] == env), None)
    if row is None:
        raise ValueError(f"{path} holds no env {env}")
    return {
        "schema": run_files.SCHEMA_VERSION, "source": "noise_free_rollout", "checkpoint": doc["checkpoint"], "env": env,
        "joint_names": doc["joint_names"],
        **{key: row[key] for key in ("success", "end_dist", "joint_pos", "joint_target", "shoe_move", "shoe_other")},
    }


def human_target_state(config_dir: Path, meta: dict) -> dict:
    """The moving shoe at the rigid fit of its keypoints onto the human baseline targets; the other shoe as snapshotted."""
    interaction = run_files.read_json(config_dir / "interaction_human.json")
    snapshot = run_files.read_json(config_dir / "keypoints.json")
    offsets = np.asarray(meta["objects"][layout.MOVING_SHOE]["keypoint_offsets"], dtype=float)
    rotation, center = kabsch(offsets, np.asarray(interaction["target_keypoints"], dtype=float))
    center = center + np.array([0.0, 0.0, layout.SPAWN_CLEARANCE_M])
    other = snapshot["objects"][layout.OTHER_SHOE]
    return {
        "schema": run_files.SCHEMA_VERSION, "source": "human_target", "checkpoint": None, "env": None, "joint_names": None,
        "success": None, "end_dist": None, "joint_pos": None, "joint_target": None,
        "shoe_move": [*center.tolist(), *matrix_to_quat_wxyz(rotation).tolist()],
        "shoe_other": [*other["position"], *other["quat_wxyz"]],
    }


def main() -> int:
    if args.human_target == bool(args.states):
        raise ValueError("give exactly one of --states (with --env) and --human-target")
    config = layout.sample_configs(args.config_index + 1)[args.config_index]
    meta = run_files.load_shoe_meta(layout.SHOE_META_PATH)
    config_dir = layout.RUNS_DIR / f"config_{config.index:02d}"
    state = human_target_state(config_dir, meta) if args.human_target else rollout_state(Path(args.states), args.env)
    poses = {name: (np.asarray(state[key][:3]), np.asarray(state[key][3:])) for name, key in SHOE_KEYS}
    scene = hs.build_scene(poses, args.device)
    joint_pos = scene.arm.data.default_joint_pos.clone()
    joint_target = joint_pos.clone()
    if state["joint_names"] is not None:
        order = [state["joint_names"].index(name) for name in scene.arm.data.joint_names]
        joint_pos[0] = torch.tensor(state["joint_pos"], device=joint_pos.device)[order]
        joint_target[0] = torch.tensor(state["joint_target"], device=joint_pos.device)[order]
    target, aim = hs.head_target(scene.arm, joint_target)
    joint_pos[:, aim.pan_id], joint_pos[:, aim.tilt_id] = target[:, aim.pan_id], target[:, aim.tilt_id]
    scene.arm.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    traces = hs.settle(scene, target, args.settle_steps, args.render_steps)
    snapshot = hs.capture(scene, traces, meta, aim)
    out_dir = Path(args.out_dir)
    passed = hs.write(snapshot, out_dir, config, ignore=REPORTED_CHECKS)
    run_files.write_json(out_dir / "state.json", state)
    print(f"OBSERVE config {config.index:02d} source {state['source']} env {state['env']} margin "
          f"{snapshot.annotated.min_margin_px:+.1f} px settle {snapshot.checks['settle_motion_m'] * 1000:.2f} mm passed {passed} -> {out_dir}",
          flush=True)
    for name, failure in snapshot.failures.items():
        print(f"OBSERVE CHECK {'REPORTED' if name in REPORTED_CHECKS else 'FAILED'}: {failure}", flush=True)
    return 0 if passed else 1


os._exit(main())

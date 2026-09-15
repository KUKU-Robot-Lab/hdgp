"""Harvest the learned grasp bank from a stage-1 checkpoint (design 2026-09-14-iker-auto-loop §5).

For each seed every env runs its first episode with deterministic actions, no noise, no disturbance and no startup
randomisation; the environment captures each success moment before the same step's reset overwrites it. The captured
states are restored into their envs and lifted: the hand action is the normalized captured EMA target (a fixed point of
the hand law). One zero arm action refreshes the restored state (the first step after a write still reads the old palm
pose and Jacobian); then the arm servoes the palm toward a goal 10 cm above that pose with its orientation kept, each
action component clamped at full scale: +z at 0.02 m per step for 5 steps, then 10 steps (1 s) holding the goal (a zero
delta would let the arm sag). A state is kept when ``grasp_bank.lift_held`` holds, with the shoe's position measured in the
palm frame, and its env did not reset.
grasp_bank.json is written only with ``--min-entries`` verified grasps.
The harvest runs the parent stage-1 task — observations, actions, terminations and the success capture equal the t2r env's (§14).

Usage:
    cd ~/rl_ws/hdgp-iker && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm ../IsaacLab/_isaac_sim/python.sh \
        scripts/iker/harvest_grasp_bank.py --checkpoint <stage-1 .pth> --headless
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Harvest the learned grasp bank from a stage-1 checkpoint.")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--config-index", type=int, default=0)
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
parser.add_argument("--min-entries", type=int, default=64)
parser.add_argument("--out", default="", help="bank file (default: iker_runs/shoe_place/config_XX/grasp_bank.json)")
parser.add_argument("--t2r-iter", type=int, default=-1, help="t2r round of the checkpoint (with --reward-code)")
parser.add_argument("--reward-code", default="", help="generated reward the checkpoint trained with; recorded as stage1_reward")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    print("HARVEST FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import axis_angle_from_quat, quat_apply_inverse, quat_inv, quat_mul, subtract_frame_transforms  # noqa: E402

import openarm.agnostic.tasks.iker_shoe.config  # noqa: E402,F401  (registers the gym ids)
from openarm.agnostic.modules.iker import run_files  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import grasp_bank as gb  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import grasp_stage as gs  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import layout, robot  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.iker_shoe_grasp_env_cfg import ARM_ACTION_DIM, IkerShoeGraspPlayEnvCfg  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.policy_player import load_player, reset_player, step_player  # noqa: E402

TASK = "open-sens_l_iker_shoe_grasp"
SETTLE_STEPS, LIFT_STEPS, HOLD_STEPS = 1, 5, 10  # 0.02 m per policy step at full action: 10 cm, then 1 s at 10 Hz
LIFT_HEIGHT_M = 0.10
BANK_FIELDS = ("joint_pos", "joint_target", "shoe_pose", "palm_pose")


def make_env():
    cfg = IkerShoeGraspPlayEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.config_index = args.config_index
    cfg.seed = args.seeds[0]
    cfg.wrench_prob_range = (1e-9, 1e-9)
    cfg.capture_success_states = True
    cfg.events.shoe_mass = None  # nominal physics, as make_grasp_bank.py
    cfg.events.shoe_material = None
    cfg.events.shoe_com = None
    return gym.make(TASK, cfg=cfg)


def first_episode_successes(u, wrapped, agent) -> torch.Tensor:
    """(N,) bool: envs whose first episode ended in success; their captures hold that moment."""
    done_first = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
    succeeded = torch.zeros_like(done_first)
    obs = reset_player(wrapped, agent)
    for _ in range(u.max_episode_length + 5):
        live = ~done_first
        obs, dones = step_player(wrapped, agent, obs)
        succeeded |= u._last.success & live
        done_first |= dones.bool()
        if bool(done_first.all()):
            break
    return succeeded & u._capture.valid


def quantiles(values: torch.Tensor) -> list[float]:
    if values.numel() == 0:
        return []
    return [round(float(v), 3) for v in torch.quantile(values.float(), torch.tensor([0.1, 0.5, 0.9], device=values.device))]


def palm_in_base(u) -> tuple[torch.Tensor, torch.Tensor]:
    """(N, 3) position and (N, 4) quaternion of the palm in the robot base frame, the frame of the arm action."""
    root, palm = u._robot.data.root_pose_w, u._robot.data.body_pose_w[:, u._palm]
    return subtract_frame_transforms(root[:, :3], root[:, 3:7], palm[:, :3], palm[:, 3:7])


def servo_arm_actions(u, goal_pos: torch.Tensor, goal_quat: torch.Tensor) -> torch.Tensor:
    """(N, 6) arm actions toward a base-frame palm goal, each component at most the full-scale step: +z at full rate while
    far below the goal, and holding the goal (against the gravity sag a zero delta would accumulate) once there."""
    pos, quat = palm_in_base(u)
    move = ((goal_pos - pos) / u.cfg.action_pos_scale).clamp(-1.0, 1.0)
    turn = (axis_angle_from_quat(quat_mul(goal_quat, quat_inv(quat))) / u.cfg.action_rot_scale).clamp(-1.0, 1.0)
    return torch.cat([move, turn], dim=-1)


def palm_state(u, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(palm z env-local, palm quaternion, shoe position in the palm frame) of ``rows``."""
    palm = u._robot.data.body_pose_w[rows, u._palm]
    shoe_in_palm = quat_apply_inverse(palm[:, 3:7], u._shoe.data.root_pos_w[rows] - palm[:, :3])
    return palm[:, 2] - u.scene.env_origins[rows, 2], palm[:, 3:7].clone(), shoe_in_palm


def verify(u, rows: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per restored capture (len(rows),): ``held`` and the lift diagnostics (shoe and palm rise, slip, palm tilt, reset)."""
    capture = u._capture
    u.restore_success_captures(rows)
    actions = torch.zeros(u.num_envs, u.cfg.action_space, device=u.device)
    actions[:, ARM_ACTION_DIM:] = gs.normalized_targets(capture.joint_target[:, u._hand_ids], u._hand_lo, u._hand_hi)
    reset = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)

    def step(arm: torch.Tensor | None) -> None:
        nonlocal reset
        actions[:, :ARM_ACTION_DIM] = 0.0 if arm is None else arm
        _, _, terminated, truncated, _ = u.step(actions)
        reset = reset | terminated | truncated

    for _ in range(SETTLE_STEPS):
        step(None)
    goal_pos, goal_quat = palm_in_base(u)
    goal_pos = goal_pos + torch.tensor([0.0, 0.0, LIFT_HEIGHT_M], device=u.device)
    origins_z = u.scene.env_origins[rows, 2]
    shoe_z_start = u._shoe.data.root_pos_w[rows, 2] - origins_z
    palm_z_start, palm_quat_start, _ = palm_state(u, rows)
    for _ in range(LIFT_STEPS):
        step(servo_arm_actions(u, goal_pos, goal_quat))
    palm_z_lift, _, rel_after_lift = palm_state(u, rows)
    for _ in range(HOLD_STEPS):
        step(servo_arm_actions(u, goal_pos, goal_quat))
    palm_z_end, palm_quat_end, rel_after_hold = palm_state(u, rows)
    shoe_z_end = u._shoe.data.root_pos_w[rows, 2] - origins_z
    tilt = torch.rad2deg(2.0 * torch.acos((palm_quat_start * palm_quat_end).sum(dim=-1).abs().clamp(max=1.0)))
    return {
        "held": gb.lift_held(shoe_z_start, shoe_z_end, rel_after_lift, rel_after_hold) & ~reset[rows],
        "rise": shoe_z_end - shoe_z_start,
        "palm_lift": palm_z_lift - palm_z_start,
        "palm_rise": palm_z_end - palm_z_start,
        "slip": (rel_after_hold - rel_after_lift).norm(dim=-1),
        "tilt_deg": tilt,
        "reset": reset[rows],
    }


def stage1_reward_record() -> dict:
    """The reward the checkpoint trained with: the generated t2r reward (spec 2026-09-15-iker-stage1-t2r §10) or the hand-written config."""
    if not args.reward_code:
        return asdict(IkerShoeGraspPlayEnvCfg().grasp_reward)
    code = Path(args.reward_code).resolve()
    return {"source": "t2r", "iter": args.t2r_iter, "reward_code_path": str(code), "reward_code_sha256": hashlib.sha256(code.read_bytes()).hexdigest()}


def main() -> int:
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    env = make_env()
    u = env.unwrapped
    wrapped, agent = load_player(env, TASK, checkpoint)
    kept = {field: [] for field in BANK_FIELDS}
    captured = 0
    with torch.inference_mode():
        for seed in args.seeds:
            u.seed(seed)
            u.clear_success_captures()
            rows = torch.nonzero(first_episode_successes(u, wrapped, agent)).squeeze(-1)
            if len(rows) == 0:
                print(f"HARVEST seed {seed} envs {u.num_envs} first-episode successes 0 verified 0", flush=True)
                continue
            result = verify(u, rows)
            for field in BANK_FIELDS:
                kept[field].append(getattr(u._capture, field)[rows[result["held"]]].clone())
            captured += len(rows)
            print(f"HARVEST seed {seed} envs {u.num_envs} first-episode successes {len(rows)} verified {int(result['held'].sum())} "
                  f"q10/50/90: shoe rise m {quantiles(result['rise'])} palm rise after lift {quantiles(result['palm_lift'])} "
                  f"after hold {quantiles(result['palm_rise'])} slip in palm m {quantiles(result['slip'])} "
                  f"palm tilt deg {quantiles(result['tilt_deg'])} resets {int(result['reset'].sum())}", flush=True)
    entries = {field: torch.cat(parts) if parts else torch.zeros(0) for field, parts in kept.items()}
    verified = int(entries["joint_pos"].shape[0])
    out = Path(args.out) if args.out else layout.RUNS_DIR / f"config_{args.config_index:02d}" / "grasp_bank.json"
    passed = verified >= args.min_entries
    if passed:
        columns, names = gb.sort_joint_columns(entries, list(u._robot.data.joint_names))
        metadata = gb.learned_bank_metadata(
            u._boot_metadata, side_sign=gb.side_sign(robot.profile().hand_joint_names), checkpoint=str(checkpoint),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), stage1_reward=stage1_reward_record(),
            seeds=args.seeds, captured=captured, verified=verified,
        )
        run_files.write_json(out, gb.bank_document(columns, names, metadata))
    print(f"HARVEST config {args.config_index:02d} checkpoint {checkpoint.name} captured {captured} verified {verified} "
          f"(min {args.min_entries}) passed {passed} -> {out if passed else 'not written'}", flush=True)
    return 0 if passed else 1


os._exit(main())

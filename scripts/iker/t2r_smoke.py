"""Smoke-check the IKER stage-1 t2r environment (spec 2026-09-15-iker-stage1-t2r §8).

--mode wire  (once, seeded for reproducibility, 140 envs by default — 20 per finger group): the prompt's hand action table
             equals the boot limits; with the shoe written against the palm every step while one finger group closes, that
             finger's own contact row must be non-zero and strictly the largest (argmax) among the non-thumb fingers'
             rows for that group (adjacent fingers may press the shoe onto a neighbour, like the thumb, whose row stays
             exempt) — a fixed fraction threshold is not used because GPU contact results vary run to run, so the gate
             compares rows within a run instead; the all-finger group has an env where the thumb and another finger touch;
             a shoe written 0.5 m away reads exactly 0 on every link and palm row.
--mode round (every t2r round, with that round's reward): zero actions for 30 steps then random actions; the reward and every
             logged term are finite, `t2r_reward/total` is in every step's log with one key set and no `grasp_reward/` key,
             previous actions are 0 right after a reset, the hold predicate is evaluated; the zero-action reward mean is reported.

Prints `T2R SMOKE CHECK FAILED: <check>` per failed check and `T2R SMOKE passed True|False`; `T2R SMOKE FAILED` on an exception.

Usage:
    cd ~/rl_ws/hdgp-iker && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm ../IsaacLab/_isaac_sim/python.sh \
        scripts/iker/t2r_smoke.py --mode wire --reward-code source/openarm/openarm/agnostic/tasks/iker_shoe/tests/fixtures/t2r_reward_fixture.py --headless
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke-check the IKER stage-1 t2r environment.")
parser.add_argument("--mode", choices=("wire", "round"), required=True)
parser.add_argument("--reward-code", required=True)
parser.add_argument("--out", default="")
parser.add_argument("--num-envs", type=int, default=0, help="0: 140 for wire, 64 for round")
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--config-index", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    print("T2R SMOKE FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

import hashlib  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

import openarm.agnostic.tasks.iker_shoe.config  # noqa: E402,F401  (registers the gym ids)
from openarm.agnostic.modules.iker import run_files  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import grasp_bank as gb  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import grasp_stage as gs  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import robot  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.iker_shoe_grasp_env_cfg import ARM_ACTION_DIM  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.iker_shoe_grasp_t2r_env_cfg import IkerShoeGraspT2rEnvCfg  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.t2r import prompts as P  # noqa: E402

TASK = "open-sens_l_iker_shoe_grasp_t2r"
TOUCH_N = 0.1
RANGE_TOL_RAD = 2e-3
IDLE_STEPS = 30
CLOSE_STEPS = 20
HOLD_OFFSET_M = 0.06  # shoe reference point along the palm normal while a finger group closes
FAR_OFFSET_M = 0.5
GROUPS = ("thumb", "index", "middle", "ring", "pinky", "all", "far")
WIRE_ENVS = 140  # 20 per group (len(GROUPS) == 7); a fixed-count gate needs enough samples that an argmax is stable
WIRE_SEED = 0  # fixes the pregrasp-bank reset draws; GPU contact results still vary between runs (PhysX contact solving
# is not run-to-run reproducible even with the same seed), which is why the gate below compares rows within a run
# (own finger's row is the largest) instead of a fixed touch-fraction threshold.


def make_env(num_envs: int, noise: bool):
    cfg = IkerShoeGraspT2rEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.config_index = args.config_index
    cfg.reward_code_path = str(Path(args.reward_code).resolve())
    cfg.add_noise = noise
    cfg.wrench_prob_range = (1e-9, 1e-9)
    if args.mode == "wire":
        cfg.seed = WIRE_SEED  # round mode stays unseeded: each t2r round should sample a fresh distribution
    return gym.make(TASK, cfg=cfg).unwrapped


def joint_table_failures(env) -> list[str]:
    prof = robot.profile()
    names = [env._robot.data.joint_names[int(i)] for i in env._hand_ids]
    measured = list(zip(env._hand_lo.tolist(), env._hand_hi.tolist()))
    error = max(max(abs(a - c), abs(b - d)) for (a, b), (c, d) in zip(measured, P.HAND_ACTION_RANGES, strict=True))
    print(f"T2R SMOKE joint table: order matches profile {names == list(prof.hand_joint_names)}, max range error {error:.4f} rad", flush=True)
    return [] if names == list(prof.hand_joint_names) and error <= RANGE_TOL_RAD else ["prompt hand action table differs from the boot limits"]


def write_shoe_at_palm(env, far: torch.Tensor) -> None:
    palm = env._robot.data.body_pose_w[:, env._palm]
    position = palm[:, :3] + quat_apply(palm[:, 3:7], env._palmar_axis) * HOLD_OFFSET_M
    position[far, 0] += FAR_OFFSET_M
    pose = torch.cat([position, env._shoe.data.root_quat_w], dim=-1)
    env._shoe.write_root_pose_to_sim(pose)
    env._shoe.write_root_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device))


def wire(env) -> tuple[list[str], dict]:
    n, dev, prof = env.num_envs, env.device, robot.profile()
    failures = joint_table_failures(env)
    finger_of_joint = [next(finger for finger in gb.FINGERS if f"_{finger}_" in name) for name in prof.hand_joint_names]
    open_pose = torch.tensor(prof.hand_open_pose, dtype=torch.float32, device=dev)
    grip_pose = torch.tensor(prof.hand_grip_pose, dtype=torch.float32, device=dev)
    group_of = [GROUPS[e % len(GROUPS)] for e in range(n)]
    targets = open_pose.expand(n, -1).clone()
    for e, group in enumerate(group_of):
        closing = [j for j, finger in enumerate(finger_of_joint) if group in ("all", "far") or finger == group]
        targets[e, closing] = grip_pose[closing]
    env.reset()
    actions = torch.zeros(n, env.cfg.action_space, device=dev)
    actions[:, ARM_ACTION_DIM:] = gs.normalized_targets(torch.max(torch.min(targets, env._hand_hi), env._hand_lo), env._hand_lo, env._hand_hi)
    far = torch.tensor([group == "far" for group in group_of], device=dev)
    for _ in range(CLOSE_STEPS):
        write_shoe_at_palm(env, far)
        env.step(actions)
    links, palm = env._link_shoe_forces()
    touching = (links > TOUCH_N).any(dim=2)  # (N, 5) finger touches with any of its links
    report = {}
    for index, group in enumerate(GROUPS):
        rows = torch.tensor([e for e in range(n) if e % len(GROUPS) == index], device=dev)
        hits = touching[rows]
        report[group] = [round(v, 2) for v in hits.float().mean(dim=0).tolist()]
        if group in gb.FINGERS[1:]:
            k = gb.FINGERS.index(group)
            others = [i for i in range(1, len(gb.FINGERS)) if i != k]
            own = float(hits[:, k].float().mean())
            if own == 0.0:
                failures.append(f"group {group}: its own contact rows never lit in {rows.numel()} envs")
            for i in others:
                other = float(hits[:, i].float().mean())
                if other >= own:
                    failures.append(f"group {group}: {gb.FINGERS[i]} row {other:.2f} >= own row {own:.2f}")
        elif group == "all" and not bool((hits[:, 0] & hits[:, 1:].any(dim=1)).any()):
            failures.append("all-finger group: no env with the thumb and another finger touching")
        elif group == "far" and (bool((links[rows] > 0.0).any()) or bool((palm[rows] > 0.0).any())):
            failures.append("far group: a contact row is non-zero with the shoe 0.5 m away")
    print(f"T2R SMOKE wire touching fraction per group, fingers thumb..pinky: {report}", flush=True)
    return failures, {"touching": report}


def round_check(env) -> tuple[list[str], dict]:
    n, dev = env.num_envs, env.device
    env.reset()
    rewards, idle, key_sets = [], [], set()
    terms_finite = prev_zero = predicate = True
    for index in range(args.steps):
        action = torch.zeros(n, env.cfg.action_space, device=dev) if index < IDLE_STEPS else 2.0 * torch.rand(n, env.cfg.action_space, device=dev) - 1.0
        _, reward, terminated, truncated, extras = env.step(action)
        log = extras.get("log", {})
        rewards.append(reward)
        if index < IDLE_STEPS:
            idle.append(reward)
        key_sets.add(frozenset(log))
        terms_finite &= all(math.isfinite(float(v)) for k, v in log.items() if k.startswith("t2r_reward/"))
        reset = (terminated | truncated).bool()
        if bool(reset.any()):
            prev_zero &= bool((env._t2r_prev_actions[reset] == 0.0).all())
        last = env._last
        predicate &= all(t.shape == (n,) for t in (last.held, last.state.latched, last.success, last.lost))
    stacked = torch.stack(rewards)
    keys = next(iter(key_sets)) if len(key_sets) == 1 else frozenset().union(*key_sets)
    checks = {
        "reward_finite": bool(torch.isfinite(stacked).all()), "terms_finite": terms_finite, "total_logged": "t2r_reward/total" in keys,
        "one_log_key_set": len(key_sets) == 1, "no_hand_reward_keys": not any(k.startswith("grasp_reward/") for k in keys),
        "prev_actions_zero_after_reset": prev_zero, "hold_predicate_evaluated": predicate,
    }
    details = {"idle_reward_mean": float(torch.stack(idle).mean()), "random_reward_mean": float(stacked[IDLE_STEPS:].mean()),
               "log_keys": sorted(keys)}
    print(f"T2R SMOKE round: checks {checks}, zero-action reward mean {details['idle_reward_mean']:.4f}, "
          f"random reward mean {details['random_reward_mean']:.4f}", flush=True)
    return [name for name, ok in checks.items() if not ok], {"checks": checks, **details}


def main() -> int:
    code = Path(args.reward_code).resolve()
    if not code.is_file():
        raise FileNotFoundError(code)
    if args.mode == "wire":
        env = make_env(args.num_envs or WIRE_ENVS, noise=False)
        failures, details = wire(env)
    else:
        env = make_env(args.num_envs or 64, noise=True)
        failures, details = round_check(env)
    for failure in failures:
        print(f"T2R SMOKE CHECK FAILED: {failure}", flush=True)
    passed = not failures
    if args.out:
        run_files.write_json(Path(args.out), {"schema": run_files.SCHEMA_VERSION, "mode": args.mode, "passed": passed, "reward_code": str(code),
                                              "reward_sha256": hashlib.sha256(code.read_bytes()).hexdigest(), "failures": failures, **details})
    print(f"T2R SMOKE passed {passed}", flush=True)
    return 0 if passed else 1


os._exit(main())

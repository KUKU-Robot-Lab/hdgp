"""Smoke-check the IKER stage-1 grasp environment before training (design 2026-09-14 §10).

1. boots with the pre-grasp bank; observations (N, 78) are finite; the palm starts 5-15 cm (surface gap) from the shoe;
2. zero actions for 3 s pay no lift or bonus term and latch nothing; the palm-progress ratchet may pay the arm's small
   zero-action sag once (bounded, not a per-step income);
3. a shoe pushed up onto the rack is not a free lift (dz_free 0, never held);
4. the reward wiring with the physics bypassed: a shoe written 6 cm above its start, clear of the hand, with zero velocity
   is held; the lift bonus latches on the third ``_get_dones`` call and the success bonus on the twentieth, each once;
5. random actions for 12 s keep rewards finite, write the episode-end log, and publish the same log keys on every step
   (rl_games reads each step's log with the keys of the epoch's first step);
6. with ``capture_success_states`` the success call of a forced hold is captured, and restoring it writes the captured joints
   and hand targets back as a fresh episode (the learned grasp harvest, auto-loop design §5);
7. the thumb backstop (learned-grasp spec §16): with its target 0.8 rad past the open pose and the action path bypassed,
   thumb_3 stays at its open-pose limit, and the simulator limit equals the boot metadata;
8. with the default thumb condition, the forced hold of an open hand is never held (checks 4 and 6 run with the xy radius
   and the thumb condition switched off — their bounds are pure-test territory).

Usage:
    cd ~/rl_ws/hdgp && PYTHONPATH=source/openarm ../IsaacLab/isaaclab.sh -p scripts/iker/grasp_smoke.py --headless
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import replace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke-check the IKER stage-1 grasp environment.")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--config-index", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    print("GRASP SMOKE FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.agnostic.tasks.iker_shoe.config  # noqa: E402,F401  (registers the gym ids)
from openarm.agnostic.tasks.iker_shoe import grasp_stage as gs  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import layout  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import robot  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.iker_shoe_grasp_env_cfg import IkerShoeGraspEnvCfg  # noqa: E402

START_GAP_RANGE_M = (0.05, 0.18)  # palm 8-12 cm above the shoe top and offset to its side: measured 105-142 mm
MAX_IDLE_PALM_PROGRESS = 2.0  # 50 x 4 cm of zero-action sag; the ratchet cannot pay more than the start gap
FORCED_LIFT_M = 0.06
FORCED_SHIFT_X_M = 0.15  # clear of the open hand and of the rack footprint
FORCED_HOLD_RADIUS_M = 0.5  # the palm-distance condition is covered by the pure tests; this check targets the wiring
LATCH_CALL, SUCCESS_CALL = 2, 19  # zero-based: the third and the twentieth held call
FORCED_THUMB_CURL_MIN_RAD = -1.0  # switches the thumb condition off for the wiring checks
BACKSTOP_PUSH_RAD = 0.8
BACKSTOP_STEPS = 20
BACKSTOP_SLACK_RAD = 0.02  # probe 2026-09-15 measured 0.0003 rad past the limit


def forced_hold(env, calls: int):
    """Write every env's shoe ``FORCED_LIFT_M`` above its start with zero velocity and evaluate ``_get_dones`` without
    stepping physics: a written pose falls ~5 cm during the 12 physics substeps of a policy step, so this check isolates
    the env's dz_free -> held -> latch -> success wiring from the simulator."""
    n, dev, origins = env.num_envs, env.device, env.scene.env_origins
    pose = env._shoe.data.root_state_w[:, :7].clone()
    pose[:, 0] += FORCED_SHIFT_X_M
    pose[:, 2] += FORCED_LIFT_M
    latch_calls, success_calls, paid = [], [], torch.zeros(n, device=dev)
    for index in range(calls):
        env._shoe.write_root_pose_to_sim(pose)
        env._shoe.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
        env.scene.update(env.physics_dt)
        env._get_dones()
        step = env._last
        if index < 3:
            dz = gs.free_lift_height(env._shoe_surface(), env._start_bottom_z, layout.RACK_X_RANGE, layout.RACK_Y_RANGE)
            shoe = env._shoe.data.root_pos_w - origins
            print(f"SMOKE forced hold call {index} env0: dz_free {float(dz[0]) * 1000:.1f} mm, shoe z {float(shoe[0, 2]):.3f} "
                  f"(target {float(pose[0, 2] - origins[0, 2]):.3f}), held {bool(step.held[0])}, hold count {int(step.state.hold_count[0])}",
                  flush=True)
        if bool(step.just_latched.any()):
            latch_calls.append(index)
        if bool(step.success.any()):
            success_calls.append(index)
        paid += step.terms["lift_bonus"] + step.terms["success_bonus"]
    return latch_calls, success_calls, paid


def backstop_check(env) -> tuple[str, float, list[float], list[float]]:
    """(joint, worst travel past the open pose in the opening direction, simulator limit, boot metadata limit) after
    ``BACKSTOP_STEPS`` policy steps with the joint's target ``BACKSTOP_PUSH_RAD`` past the open pose."""
    prof = robot.profile()
    backstop = env._boot_metadata["hand_backstop"]
    name = next(iter(backstop))
    k = prof.hand_joint_names.index(name)
    j = list(env._robot.data.joint_names).index(name)
    open_q, grip_q = float(prof.hand_open_pose[k]), float(prof.hand_grip_pose[k])
    opening = -1.0 if grip_q > open_q else 1.0
    env.reset()
    env._pre_physics_step = lambda actions: None  # hold the written targets; the hand law would pull them back into range
    try:
        env._joint_targets[:] = env._robot.data.joint_pos_target
        env._joint_targets[:, j] = open_q + opening * BACKSTOP_PUSH_RAD
        for _ in range(BACKSTOP_STEPS):
            env.step(torch.zeros(env.num_envs, env.cfg.action_space, device=env.device))
    finally:
        del env._pre_physics_step
    past = float(((env._robot.data.joint_pos[:, j] - open_q) * opening).max())
    return name, past, env._robot.data.joint_pos_limits[0, j].tolist(), backstop[name]


def main() -> int:
    cfg = IkerShoeGraspEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.config_index = args.config_index
    cfg.add_noise = False
    cfg.wrench_prob_range = (1e-9, 1e-9)
    cfg.grasp_reward.hold_radius_m = FORCED_HOLD_RADIUS_M
    cfg.grasp_reward.hold_xy_radius_m = FORCED_HOLD_RADIUS_M  # the forced hold shifts the shoe 15 cm (clear of the hand)
    cfg.grasp_reward.thumb_curl_min_rad = FORCED_THUMB_CURL_MIN_RAD
    cfg.capture_success_states = True
    env = gym.make("open-sens_l_iker_shoe_grasp", cfg=cfg).unwrapped
    n, dev = env.num_envs, env.device
    failures = []

    obs, _ = env.reset()
    policy = obs["policy"]
    origins = env.scene.env_origins
    palm = env._robot.data.body_pos_w[:, env._palm] - origins
    gap = gs.nearest_distance(palm[:, None, :], env._shoe_surface())[:, 0]
    print(f"SMOKE obs {tuple(policy.shape)} finite {bool(torch.isfinite(policy).all())} pregrasp bank {env._bank.size} "
          f"start palm gap mm min {float(gap.min()) * 1000:.1f} max {float(gap.max()) * 1000:.1f}", flush=True)
    if policy.shape != (n, 78) or not torch.isfinite(policy).all():
        failures.append("observation shape or finiteness")
    if not (START_GAP_RANGE_M[0] <= float(gap.min()) and float(gap.max()) <= START_GAP_RANGE_M[1]):
        failures.append(f"start palm gap outside {START_GAP_RANGE_M}")

    term_sums = {name: torch.zeros(n, device=dev) for name in gs.REWARD_TERMS}
    for _ in range(30):
        env.step(torch.zeros(n, env.cfg.action_space, device=dev))
        for name in gs.REWARD_TERMS:
            term_sums[name] += env._last.terms[name]
    latched = float(env._stage.latched.float().mean())
    sums = {k: round(float(v.abs().max()), 4) for k, v in term_sums.items()}
    print(f"SMOKE zero action 3 s: latched {latched:.2f}, per-term max |sum| {sums}", flush=True)
    if latched > 0.0 or any(sums[k] > 0.0 for k in ("lift_progress", "lift_bonus", "success_bonus")) or sums["palm_progress"] > MAX_IDLE_PALM_PROGRESS:
        failures.append("zero-action policy earned a lift or bonus term, latched, or more palm progress than the sag bound")

    env.reset()
    rack_pose = env._shoe.data.root_state_w[:, :7].clone()
    rack_pose[:, 0] = origins[:, 0] + sum(layout.RACK_X_RANGE) / 2
    rack_pose[:, 1] = origins[:, 1] + layout.RACK_Y_RANGE[1] - 0.08
    rack_pose[:, 2] = origins[:, 2] + layout.RACK_TOP_Z + (env._shoe.data.root_pos_w[:, 2] - origins[:, 2]) - layout.TABLE_TOP_Z + 0.002
    held_any = False
    for _ in range(10):
        env._shoe.write_root_pose_to_sim(rack_pose)
        env._shoe.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
        env.step(torch.zeros(n, env.cfg.action_space, device=dev))
        held_any |= bool(env._last.held.any())
    dz_on_rack = gs.free_lift_height(env._shoe_surface(), env._start_bottom_z, layout.RACK_X_RANGE, layout.RACK_Y_RANGE)
    print(f"SMOKE shoe on rack: dz_free max {float(dz_on_rack.max()) * 1000:.1f} mm, held {held_any}", flush=True)
    if float(dz_on_rack.max()) > 0.0 or held_any:
        failures.append("a shoe on the rack counted as a free lift")

    env.reset()
    latch_calls, success_calls, paid = forced_hold(env, 25)
    print(f"SMOKE forced hold: latch calls {latch_calls}, success calls {success_calls}, bonus paid min {float(paid.min()):.1f} max {float(paid.max()):.1f}", flush=True)
    if latch_calls != [LATCH_CALL] or success_calls != [SUCCESS_CALL] or float(paid.min()) != float(paid.max()):
        failures.append(f"forced hold latched at {latch_calls} (want [{LATCH_CALL}]) and succeeded at {success_calls} (want [{SUCCESS_CALL}])")

    env.reset()
    env._episode_log = dict.fromkeys(env._episode_log, -1.0)  # sentinel: an episode end must overwrite it
    rewards, key_sets = [], set()
    for _ in range(120):
        _, reward, _, _, extras = env.step(2.0 * torch.rand(n, env.cfg.action_space, device=dev) - 1.0)
        rewards.append(reward)
        key_sets.add(frozenset(extras.get("log", {})))
    stacked = torch.stack(rewards)
    episode_logged = all(value != -1.0 for value in env._episode_log.values())
    print(f"SMOKE random 12 s: reward finite {bool(torch.isfinite(stacked).all())} mean {float(stacked.mean()):.3f} "
          f"max {float(stacked.max()):.1f} episode-end log written {episode_logged} log key sets {len(key_sets)}", flush=True)
    if not torch.isfinite(stacked).all() or not episode_logged or len(key_sets) != 1:
        failures.append("random-action rewards, the episode-end log, or log keys that differ between steps")

    env.reset()
    env.clear_success_captures()
    forced_hold(env, SUCCESS_CALL + 1)  # the success call is captured before anything could overwrite it
    capture = env._capture
    pose_error = float((capture.shoe_pose[:, :3] - (env._shoe.data.root_pos_w - origins)).abs().max())
    home = env._robot.data.default_joint_pos.clone()
    env._robot.write_joint_state_to_sim(home, torch.zeros_like(home))
    env.episode_length_buf[:] = 7
    env.restore_success_captures(torch.arange(n, device=dev))
    joint_error = float((env._robot.data.joint_pos - capture.joint_pos).abs().max())
    hand_error = float((env._hand_targets - capture.joint_target[:, env._hand_ids]).abs().max())
    length = int(env.episode_length_buf.max())
    print(f"SMOKE success capture: valid {bool(capture.valid.all())}, shoe pose error {pose_error:.2e}, restored joint error "
          f"{joint_error:.2e}, hand target error {hand_error:.2e}, episode length {length}", flush=True)
    if not bool(capture.valid.all()) or pose_error > 1e-4 or joint_error > 1e-5 or hand_error > 1e-6 or length != 0:
        failures.append("the success capture or its restore does not reproduce the captured state")

    name, past, sim_limit, meta_limit = backstop_check(env)
    print(f"SMOKE thumb backstop: {name} worst travel past the open pose {past:+.4f} rad, simulator limit "
          f"{[round(v, 4) for v in sim_limit]}, boot metadata {meta_limit}", flush=True)
    if past > BACKSTOP_SLACK_RAD or [round(v, 4) for v in sim_limit] != meta_limit:
        failures.append(f"thumb backstop: {past:+.4f} rad past the open pose or simulator limit {sim_limit} != {meta_limit}")

    env._reward_cfg = replace(env._reward_cfg, thumb_curl_min_rad=gs.Stage1RewardCfg().thumb_curl_min_rad)
    env.reset()
    latch_calls, success_calls, _ = forced_hold(env, 5)
    print(f"SMOKE open thumb under the default thumb condition ({env._reward_cfg.thumb_curl_min_rad} rad): "
          f"latch calls {latch_calls}, success calls {success_calls}", flush=True)
    if latch_calls or success_calls:
        failures.append("a forced hold with the open thumb latched under the default thumb condition")

    for failure in failures:
        print(f"SMOKE CHECK FAILED: {failure}", flush=True)
    print(f"GRASP SMOKE passed {not failures}", flush=True)
    return 0 if not failures else 1


os._exit(main())

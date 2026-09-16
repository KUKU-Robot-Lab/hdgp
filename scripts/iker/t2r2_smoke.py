"""IKER 2단계 t2r 스모크. --mode settle 은 구현 착수 전 게이트(스펙 §9.2).

--mode settle: 신발을 VLM 목표 자세로 두고 로봇을 홈으로 치운 뒤 100 스텝(10 s) 물리를 돌려
  키포인트가 5 cm 안에 남는지 본다. 기존 env_smoke 는 env.step 을 한 번(0.1 s)만 돌렸다.
"""
from __future__ import annotations

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("settle",), required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=args.headless).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import openarm.agnostic.tasks.iker_shoe.config  # noqa: F401,E402
from openarm.agnostic.modules.iker.gate import kabsch  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.iker_shoe_env_cfg import IkerShoeEnvCfg  # noqa: E402
from isaaclab.utils.math import quat_from_matrix  # noqa: E402

SETTLE_TOLERANCE_M = 0.05


def settle(env, steps: int) -> tuple[list[str], dict]:
    n = env.num_envs
    dev = env.device
    env.reset()
    home = env._robot.data.default_joint_pos.clone()
    env._robot.write_joint_state_to_sim(home, torch.zeros_like(home))
    env._robot.set_joint_position_target(home)
    env._joint_targets[:] = home

    rot, centre = kabsch(env._offsets.cpu().numpy(), env._targets.cpu().numpy())
    pose = torch.zeros(n, 7, device=dev)
    pose[:, :3] = torch.tensor(centre, dtype=torch.float32, device=dev) + env.scene.env_origins
    pose[:, 3:] = quat_from_matrix(torch.tensor(rot, dtype=torch.float32, device=dev)).expand(n, 4)
    env._shoe.write_root_pose_to_sim(pose)
    env._shoe.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))

    zero = torch.zeros(n, env.cfg.action_space, device=dev)
    start = None
    for _ in range(steps):
        env.step(zero)
        if start is None:
            start = env._keypoint_distance.clone()
    end = env._keypoint_distance
    median = float(end.median())
    drift = float((end - start).median())
    failures = []
    if not median <= SETTLE_TOLERANCE_M:
        failures.append(f"settle: median keypoint distance {median:.4f} m > {SETTLE_TOLERANCE_M}")
    print(f"T2R2 SMOKE settle: median {median*1000:.1f} mm, drift {drift*1000:+.1f} mm over {steps} steps", flush=True)
    return failures, {"median_m": median, "drift_m": drift, "steps": steps}


def main() -> int:
    cfg = IkerShoeEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = gym.make("open-sens_l_iker_shoe", cfg=cfg).unwrapped
    failures, details = settle(env, args.steps)
    for failure in failures:
        print(f"T2R2 SMOKE CHECK FAILED: {failure}", flush=True)
    passed = not failures
    print(f"T2R2 SMOKE passed {passed}", flush=True)
    env.close()
    return 0 if passed else 1


code = main()
app.close()
sys.exit(code)

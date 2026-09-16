"""IKER 2단계 t2r 스모크. --mode settle 은 구현 착수 전 게이트(스펙 §9.2).

--mode settle: 신발을 VLM 목표 자세로 두고 로봇을 홈으로 치운 뒤 10 s(steps*decimation 물리 스텝)를 돌려
  키포인트가 5 cm 안에 남는지 본다. 기존 env_smoke 는 env.step 을 한 번(0.1 s)만 돌렸다.

fix round 1 (2026-09-16): env.step(zero) 는 RL 종료/리셋 경로(_get_dones → success_count 증가 → 21 스텝
  근처에서 SUCCESS 종료 → _reset_idx 가 신발을 뱅크 자세로 되돌림)를 함께 돌려 정착을 측정하지 못했다.
  이제 env.scene.write_data_to_sim()/env.sim.step()/env.scene.update() 로 물리만 직접 진행하고(env_smoke.py
  reach_slots 와 동일 패턴), 종료가 없으니 env._keypoint_distance(= `_get_dones` 안에서만 갱신됨)도 못 쓴다 —
  거리는 transform_keypoints 로 직접 계산한다.
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
from openarm.agnostic.modules.iker.reward import transform_keypoints  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.iker_shoe_env_cfg import IkerShoeEnvCfg  # noqa: E402
from isaaclab.utils.math import quat_from_matrix  # noqa: E402

SETTLE_TOLERANCE_M = 0.05


def _keypoint_distance(env) -> torch.Tensor:
    """(N,) mean distance between the shoe's current keypoints and the interaction targets.

    Same definition as `IkerShoeEnv._get_dones` (`self._keypoint_distance`), computed directly since that buffer is
    only refreshed inside `_get_dones`, which does not run once the RL step/termination path is bypassed.
    """
    current = transform_keypoints(env._shoe.data.root_pos_w, env._shoe.data.root_quat_w, env._offsets)
    current = current - env.scene.env_origins[:, None, :]
    targets = env._targets.expand(env.num_envs, -1, -1)
    return (current - targets).norm(dim=-1).mean(dim=-1)


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

    # Advance physics directly (env_smoke.py reach_slots, lines 102-104) — env.step() would run the RL
    # termination/reset path, which teleports the shoe back to a grasp-bank pose once success_count sustains.
    physics_steps = steps * env.cfg.decimation
    window_s = physics_steps * env.physics_dt
    start_distance, start_z = None, None
    for i in range(physics_steps):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)
        if i == 0:  # a pose written before the loop is only read back after one physics step (env_smoke.py)
            start_distance = _keypoint_distance(env)
            start_z = env._shoe.data.root_pos_w[:, 2].clone()
    end_distance = _keypoint_distance(env)
    end_z = env._shoe.data.root_pos_w[:, 2]

    median = float(end_distance.median())
    drift = float((end_distance - start_distance).median())
    z_drop = float((start_z - end_z).median())
    failures = []
    if not median <= SETTLE_TOLERANCE_M:
        failures.append(f"settle: median keypoint distance {median:.4f} m > {SETTLE_TOLERANCE_M}")
    print(
        f"T2R2 SMOKE settle: median {median*1000:.1f} mm, drift {drift*1000:+.1f} mm, "
        f"z-drop {z_drop*1000:+.1f} mm over {physics_steps} physics steps ({window_s:.2f} s)",
        flush=True,
    )
    return failures, {
        "median_m": median,
        "drift_m": drift,
        "z_drop_m": z_drop,
        "physics_steps": physics_steps,
        "window_s": window_s,
    }


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

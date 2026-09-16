"""IKER 2단계 t2r 스모크. --mode settle 은 구현 착수 전 게이트(스펙 §9.2).

--mode settle: 신발을 VLM 목표 자세로 두고 로봇을 홈으로 치운 뒤 10 s(steps*decimation 물리 스텝)를 돌려
  키포인트가 5 cm 안에 남는지 본다. 기존 env_smoke 는 env.step 을 한 번(0.1 s)만 돌렸다.

fix round 1 (2026-09-16): env.step(zero) 는 RL 종료/리셋 경로(_get_dones → success_count 증가 → 21 스텝
  근처에서 SUCCESS 종료 → _reset_idx 가 신발을 뱅크 자세로 되돌림)를 함께 돌려 정착을 측정하지 못했다.
  이제 env.scene.write_data_to_sim()/env.sim.step()/env.scene.update() 로 물리만 직접 진행하고(env_smoke.py
  reach_slots 와 동일 패턴), 종료가 없으니 env._keypoint_distance(= `_get_dones` 안에서만 갱신됨)도 못 쓴다 —
  거리는 transform_keypoints 로 직접 계산한다.

fix round 2 (2026-09-16): env.step() 을 우회하면서 _apply_action() 도 함께 우회됐다 — 팔 중력보상 effort
  target 은 그 안에서만 설정되므로, 물리 루프 내내 무보상 상태였다(이 저장소에 12.76° 처짐 전례 있음). 매
  물리 서브스텝마다 _apply_action() 을 그대로 재현(position target + gravity comp effort)한다. 손이 신발을
  누르고 있었을 가능성을 배제하기 위해 palm 변위·palm-신발 최소거리를 측정하고, RACK_TOP_Z 대비 신발 바닥
  높이도 비교한다(둘 다 게이트가 아니라 판정을 검증 가능하게 만드는 보고용 수치).
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
import torch  # noqa: E402

import openarm.agnostic.tasks.iker_shoe.config  # noqa: F401,E402
from openarm.agnostic.modules.iker import run_files  # noqa: E402
from openarm.agnostic.modules.iker.gate import kabsch  # noqa: E402
from openarm.agnostic.modules.iker.reward import transform_keypoints  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import layout  # noqa: E402
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

    meta = run_files.load_shoe_meta(layout.SHOE_META_PATH)
    hull_local = torch.tensor(meta["objects"][layout.MOVING_SHOE]["hull_local"], dtype=torch.float32, device=dev)

    # Advance physics directly (env_smoke.py reach_slots, lines 102-104) — env.step() would run the RL
    # termination/reset path, which teleports the shoe back to a grasp-bank pose once success_count sustains.
    # _apply_action() (IkerShoeEnv) is reachable only through env.step(); reissue its two targets by hand each
    # physics substep — otherwise the arm's gravity-compensation effort target is never set and it sags
    # uncompensated for the whole window (this repo has a recorded 12.76 deg case of exactly this).
    physics_steps = steps * env.cfg.decimation
    window_s = physics_steps * env.physics_dt
    start_distance = start_z = palm_start = None
    palm_shoe_min = None
    for i in range(physics_steps):
        env._robot.set_joint_position_target(env._joint_targets)
        tau = env._robot.root_physx_view.get_gravity_compensation_forces()
        env._robot.set_joint_effort_target(tau[:, env._gravity_ids], joint_ids=env._gravity_ids)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)

        palm_pos = env._robot.data.body_pos_w[:, env._palm]
        gap = (palm_pos - env._shoe.data.root_pos_w).norm(dim=-1)
        palm_shoe_min = gap if palm_shoe_min is None else torch.minimum(palm_shoe_min, gap)
        if i == 0:  # a pose written before the loop is only read back after one physics step (env_smoke.py)
            start_distance = _keypoint_distance(env)
            start_z = env._shoe.data.root_pos_w[:, 2].clone()
            palm_start = palm_pos.clone()
    end_distance = _keypoint_distance(env)
    end_z = env._shoe.data.root_pos_w[:, 2]
    palm_displacement = (env._robot.data.body_pos_w[:, env._palm] - palm_start).norm(dim=-1)

    hull_world = transform_keypoints(env._shoe.data.root_pos_w, env._shoe.data.root_quat_w, hull_local)
    bottom_z = hull_world[..., 2].min(dim=1).values - env.scene.env_origins[:, 2]
    rack_gap = bottom_z - layout.RACK_TOP_Z

    median = float(end_distance.median())
    drift = float((end_distance - start_distance).median())
    z_drop = float((start_z - end_z).median())
    palm_displacement_m = float(palm_displacement.median())
    palm_shoe_min_m = float(palm_shoe_min.median())
    rack_gap_m = float(rack_gap.median())
    failures = []
    if not median <= SETTLE_TOLERANCE_M:
        failures.append(f"settle: median keypoint distance {median:.4f} m > {SETTLE_TOLERANCE_M}")
    print(
        f"T2R2 SMOKE settle: median {median*1000:.1f} mm, drift {drift*1000:+.1f} mm, "
        f"z-drop {z_drop*1000:+.1f} mm over {physics_steps} physics steps ({window_s:.2f} s), "
        f"palm displacement {palm_displacement_m*1000:.1f} mm, palm-shoe min {palm_shoe_min_m*1000:.1f} mm, "
        f"shoe bottom - rack_top {rack_gap_m*1000:+.1f} mm",
        flush=True,
    )
    return failures, {
        "median_m": median,
        "drift_m": drift,
        "z_drop_m": z_drop,
        "palm_displacement_m": palm_displacement_m,
        "palm_shoe_min_m": palm_shoe_min_m,
        "shoe_bottom_minus_rack_top_m": rack_gap_m,
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

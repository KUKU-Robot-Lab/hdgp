#!/usr/bin/env python3
"""논문 Algorithm 1의 population search.

후보 K개를 단일 sim의 K개 env에 각기 다른 actuator gain으로 배치하고,
같은 q_cmd를 replay해 real tracking과 가장 가까운 후보를 고른다.

RL 학습을 실행하지 않는다. USD를 복사하거나 수정하지 않는다.

실행 (server):
  ./isaaclab.sh -p ../hdgp/scripts/r2s_autotune/run_parallel_replay.py \
      --config ../hdgp/scripts/r2s_autotune/configs/bi_rh56f1.yaml \
      --output ../hdgp/log/logs/r2s_autotune/results/bi_rh56f1_best_calibration.json --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Parallel replay autotune (paper Algorithm 1).")
parser.add_argument("--config", required=True, help="asset autotune config yaml")
parser.add_argument("--seed-calibration", default=None, help="seed JSON (기본: config 기본값)")
parser.add_argument("--output", required=True, help="best calibration JSON 경로")
parser.add_argument("--population-size", type=int, default=None, help="config 값 override")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import dataclasses  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import isaaclab.sim as sim_utils  # noqa: E402

from r2s_autotune.calibration_io import load_calibration  # noqa: E402
from r2s_autotune.compute_tracking_error import compute_tracking_error  # noqa: E402
from r2s_autotune.config import load_config  # noqa: E402
from r2s_autotune.export_best_calibration import export_best_calibration  # noqa: E402
from r2s_autotune.gain_matrix import build_gain_matrices  # noqa: E402
from r2s_autotune.load_real_track import load_real_track  # noqa: E402
from r2s_autotune.excitation import ExcitationSpec  # noqa: E402
from r2s_autotune.replay_env import (  # noqa: E402
    apply_gains,
    build_full_targets,
    group_joint_indices,
    make_scene,
    replay,
    reset_and_settle,
    rest_pose,
    select_columns,
    verify_articulation,
)
from r2s_autotune.sample_candidates import sample_candidates  # noqa: E402
from r2s_autotune.seed_from_config import seed_from_config  # noqa: E402

# 후보 간 total 오차 산포가 이보다 작으면 excitation이 약하거나 command==measured다 (가이드 §11.2).
MIN_ERROR_SPREAD = 1e-3


def main() -> None:
    config = load_config(args_cli.config)
    if args_cli.population_size is not None:
        config = dataclasses.replace(config, population_size=args_cli.population_size)
    if config.real_track is None:
        raise SystemExit("config has no real_track; run make_synthetic_track.py first")

    track = load_real_track(config.real_track, config.manifest)
    seed = (
        load_calibration(args_cli.seed_calibration)
        if args_cli.seed_calibration
        else seed_from_config(config, source_dataset=str(config.real_track.hdf5))
    )
    if seed.robot_asset and seed.robot_asset != config.asset:
        raise SystemExit(f"seed robot_asset '{seed.robot_asset}' != config asset '{config.asset}'")

    candidates = sample_candidates(config, seed)
    print(f"[r2s] asset={config.asset}  candidates={len(candidates)}  tune={config.tune_groups}")
    print(f"[r2s] real track: {track.num_steps} steps, {len(track.joint_names)} joints")

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=track.dt, device=args_cli.device)
    )
    scene = make_scene(config, num_envs=len(candidates))
    sim.reset()

    robot = scene["robot"]
    verify_articulation(robot, config.manifest)
    group_indices = group_joint_indices(robot, config)

    matrices = build_gain_matrices(config, candidates, group_indices, robot.num_joints)
    apply_gains(robot, matrices["stiffness"], matrices["damping"], matrices["joint_friction"])

    # 추적하지 않는 관절은 한계에서 떨어진 기준 자세로 붙든다. default_joint_pos를 쓰면
    # Tesollo curl 관절이 한계 위에 얹혀 손 전체가 구속과 싸운다.
    hold_pose = rest_pose(robot, ExcitationSpec(dt=track.dt))
    reset_and_settle(sim, scene, robot, hold_pose)

    targets = build_full_targets(robot, track.q_cmd, track.joint_names, hold_pose)
    q_sim_full, dq_sim_full = replay(sim, scene, robot, targets)

    q_sim = select_columns(q_sim_full, robot, track.joint_names)
    dq_sim = select_columns(dq_sim_full, robot, track.joint_names)

    errors = compute_tracking_error(
        q_cmd=track.q_cmd,
        q_real=track.q_real,
        dq_real=track.dq_real,
        q_sim=q_sim,
        dq_sim=dq_sim,
        weights=config.error_weights,
        dt=track.dt,
    )

    spread = errors.spread()
    if len(candidates) > 1 and spread < MIN_ERROR_SPREAD:
        print(
            f"[r2s] WARNING: error spread {spread:.2e} < {MIN_ERROR_SPREAD:.0e}. "
            "excitation이 약하거나 command와 measured가 같은 값을 보고 있다."
        )

    export_best_calibration(
        output_path=args_cli.output,
        asset=config.asset,
        source_dataset=str(config.real_track.hdf5),
        candidates=candidates,
        errors=errors,
        tune_groups=config.tune_groups,
    )

    seed_total = float(errors.total[0])
    best_total = float(errors.total[errors.best_index])
    print(f"[r2s] seed error={seed_total:.6e}  best error={best_total:.6e} "
          f"(candidate {errors.best_index}, spread={spread:.3f})")
    print(f"[r2s] mse_q={errors.mse_q[errors.best_index]:.6e}  "
          f"mse_dq={errors.mse_dq[errors.best_index]:.6e}  "
          f"delay={errors.delay_penalty[errors.best_index]:.4f}s")
    print(f"[r2s] best calibration -> {args_cli.output}")
    print(f"[r2s] apply with: export OPENARM_REAL2SIM_ACTUATOR_CALIBRATION={args_cli.output}")

    if best_total > seed_total:
        print("[r2s] WARNING: seed가 모든 후보보다 낫다. scale range가 잘못됐을 수 있다.")

    # simulation_app.close()가 종종 행된다. 산출물은 이미 디스크에 있으므로 즉시 빠져나온다.
    sys.stdout.flush()
    os._exit(0)  # simulation_app.close()가 행되므로 호출하지 않는다


if __name__ == "__main__":
    main()

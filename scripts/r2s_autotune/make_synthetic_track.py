#!/usr/bin/env python3
"""알려진 gain으로 sim 궤적을 만들어 "real"인 척하는 HDF5로 저장한다.

real 데이터가 없는 동안 autotune 파이프라인의 정확성을 검증할 유일한 방법이다:
  ground truth A로 궤적 생성 → autotune이 A를 복원하는가?
복원 오차가 크면 파이프라인 버그이며, real 데이터를 넣어도 의미가 없다.

실행 (server):
  ./isaaclab.sh -p ../hdgp/scripts/r2s_autotune/make_synthetic_track.py \
      --config ../hdgp/scripts/r2s_autotune/configs/bi_rh56f1.yaml --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate a synthetic real-track for autotune tests.")
parser.add_argument("--config", required=True, help="asset autotune config yaml")
parser.add_argument("--output", default=None, help="HDF5 경로 (기본: config의 real_track.hdf5)")
parser.add_argument("--stiffness-scale", type=float, default=1.30, help="ground-truth stiffness 배율")
parser.add_argument("--damping-scale", type=float, default=0.80, help="ground-truth damping 배율")
parser.add_argument("--friction-scale", type=float, default=1.00, help="ground-truth friction 배율")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import isaaclab.sim as sim_utils  # noqa: E402

from r2s_autotune.calibration_io import Calibration, write_calibration  # noqa: E402
from r2s_autotune.config import load_config  # noqa: E402
from r2s_autotune.excitation import ExcitationSpec, build_excitation, is_saturated  # noqa: E402
from r2s_autotune.gain_matrix import build_gain_matrices  # noqa: E402
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
from r2s_autotune.sample_candidates import Candidate  # noqa: E402
from r2s_autotune.seed_from_config import seed_from_config  # noqa: E402


def _ground_truth(config, args) -> Calibration:
    seed = seed_from_config(config, source_dataset="<synthetic>")
    groups = {
        name: (
            calibration.scaled(args.stiffness_scale, args.damping_scale, args.friction_scale)
            if name in config.tune_groups
            else calibration
        )
        for name, calibration in seed.groups.items()
    }
    return seed.with_groups(groups)


def _tracked_joints(config, group_indices, joint_names) -> tuple[str, ...]:
    """tune group에 속한 관절만 여기(excite)하고 기록한다."""
    columns: list[int] = []
    for name in config.tune_groups:
        columns.extend(group_indices[name])
    return tuple(joint_names[i] for i in sorted(set(columns)))


def _write_hdf5(path: Path, time, q_cmd, q_real, dq_real, joint_names) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names_json = json.dumps(list(joint_names))
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.attrs["num_samples"] = int(q_cmd.shape[0])
        demo.create_dataset("timestamps_ns", data=(time * 1e9).astype(np.int64))
        obs = demo.create_group("obs")
        for key, array in (("q_cmd", q_cmd), ("q_real", q_real), ("dq_real", dq_real)):
            dataset = obs.create_dataset(key, data=array.astype(np.float32), compression="gzip")
            dataset.attrs["joint_names"] = names_json


def main() -> None:
    config = load_config(args_cli.config)
    if args_cli.output is None and config.real_track is None:
        raise SystemExit("config has no real_track; pass --output explicitly")
    truth = _ground_truth(config, args_cli)

    output = Path(args_cli.output) if args_cli.output else config.real_track.hdf5
    spec = ExcitationSpec(dt=config.real_track.dt if config.real_track else 0.01)

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=spec.dt, device=args_cli.device)
    )
    scene = make_scene(config, num_envs=1)
    sim.reset()

    robot = scene["robot"]
    verify_articulation(robot, config.manifest)
    group_indices = group_joint_indices(robot, config)

    tracked = _tracked_joints(config, group_indices, robot.joint_names)
    print(f"[r2s] tracked joints ({len(tracked)}): {tracked}")

    matrices = build_gain_matrices(
        config,
        [Candidate(index=0, groups=truth.groups)],
        group_indices,
        robot.num_joints,
    )
    apply_gains(robot, matrices["stiffness"], matrices["damping"], matrices["joint_friction"])

    columns = [robot.joint_names.index(j) for j in tracked]
    hold_pose = rest_pose(robot, spec)
    limits = robot.data.soft_joint_pos_limits[0].detach().cpu().numpy()[columns]
    neutral = hold_pose[columns]

    saturated = is_saturated(neutral, limits[:, 0], limits[:, 1], spec)
    if saturated.any():
        clipped = [tracked[i] for i in range(len(tracked)) if saturated[i]]
        print(f"[r2s] WARNING: excitation이 관절 한계에 잘린다 → 식별성 저하: {clipped}")

    time, q_cmd = build_excitation(neutral, limits[:, 0], limits[:, 1], spec)

    reset_and_settle(sim, scene, robot, hold_pose)
    targets = build_full_targets(robot, q_cmd, tracked, hold_pose)
    q_sim, dq_sim = replay(sim, scene, robot, targets)

    q_real = select_columns(q_sim, robot, tracked)[0]
    dq_real = select_columns(dq_sim, robot, tracked)[0]

    _write_hdf5(Path(output), time, q_cmd, q_real, dq_real, tracked)

    truth_path = Path(output).with_name(Path(output).stem + "_ground_truth.json")
    write_calibration(truth_path, truth)

    print(f"[r2s] synthetic track  -> {output}  ({q_cmd.shape[0]} steps)")
    print(f"[r2s] ground truth     -> {truth_path}")
    # simulation_app.close()가 종종 행된다. 산출물은 이미 디스크에 있으므로 즉시 빠져나온다.
    sys.stdout.flush()
    os._exit(0)  # simulation_app.close()가 행되므로 호출하지 않는다


if __name__ == "__main__":
    main()

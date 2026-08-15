#!/usr/bin/env python3
"""warm bank 전수 pour-물리 안정성 검증 → 안정 상태만 남긴 hdf5 생성.

각 warm 상태를 pour env(zero-action, freeze hand)에 순차 리셋해 bead 유지 여부를
판정한다. 원인 불문 "pour 물리에서 유지되는 파지"만 남기는 실용 필터 —
수집(grasp_v1)과 소비(pour)의 물리/파지스타일 차이로 인한 리셋 유실을 차단한다.

사용 (DR 설정과 동일 scale_set로 검증해야 spec 매칭 경로가 동일):
  isaaclab.sh -p scripts/warm_states/validate_warm_states_in_pour.py \
      --num_envs 512 --rounds 4 --steps 300 --out data/grasp_warm_tesollo_valid.hdf5 \
      "env.source_cup_scale_set=[0.85,1.0,1.15,1.3]"
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--rounds", type=int, default=4, help="전수 커버를 위한 리셋 라운드 수")
parser.add_argument("--steps", type=int, default=300, help="라운드당 관찰 스텝 (hold 120 + 안정 관찰)")
parser.add_argument("--keep_thresh", type=float, default=0.9, help="bead_in_source PASS 임계")
parser.add_argument("--src", type=str, default="data/grasp_warm_tesollo.hdf5")
parser.add_argument("--out", type=str, default="data/grasp_warm_tesollo_valid.hdf5")
parser.add_argument("--task", type=str, default="open-tesol_b_pour_sensor-lstm")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys

sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym
import h5py
import numpy as np
import torch

import openarm.tesollo  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.warm_validation_sequential = True
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    zero = torch.zeros(args_cli.num_envs, uenv.cfg.action_space, device=uenv.device)

    n_bank = int(uenv._warmstart_cache_count)
    passed = np.zeros(n_bank, dtype=bool)
    failed = np.zeros(n_bank, dtype=bool)

    for rnd in range(args_cli.rounds):
        env.reset()
        picks0 = uenv._last_warm_pick.clone()          # 라운드 시작 pick 스냅샷
        early_fail = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=uenv.device)
        for t in range(args_cli.steps):
            _, _, terminated, truncated, _ = env.step(zero)
            done = (terminated | truncated).to(uenv.device)
            early_fail |= done.bool()                  # 관찰 중 조기종료 = FAIL (유실/파국)
        keep = uenv._bead_in_source_fraction >= args_cli.keep_thresh
        ok = keep & (~early_fail)
        p = picks0.cpu().numpy()
        okn = ok.cpu().numpy()
        for i in range(args_cli.num_envs):
            if p[i] < 0:
                continue
            if okn[i]:
                passed[p[i]] = True
            else:
                failed[p[i]] = True
        print(f"[validate] round {rnd+1}/{args_cli.rounds}: pass_now={int(okn.sum())}/{args_cli.num_envs} "
              f"| bank pass={int(passed.sum())} fail={int(failed.sum())} "
              f"seen={int((passed | failed).sum())}/{n_bank}", flush=True)

    keep_mask = passed & (~failed)                     # 한 번이라도 FAIL이면 제외 (보수)
    unseen = ~(passed | failed)
    print(f"[validate] 최종: keep={int(keep_mask.sum())} fail={int(failed.sum())} "
          f"unseen={int(unseen.sum())} / {n_bank}", flush=True)

    # 필터 hdf5 저장 — 원본에서 keep 인덱스만. (spec 필터 이전의 '원본 파일' 기준으로
    # 저장해야 하므로, bank 로더 인덱스 → 원본 인덱스 매핑이 필요 없다는 전제:
    # 검증 env 의 로더는 학습과 동일 설정(spec 매칭 모드, 필터 미적용)이라 bank == 원본 순서.
    src = h5py.File(args_cli.src, "r")
    g = src["warm_states"]
    n_src = g["arm_joint_pos"].shape[0]
    if n_src != n_bank:
        raise RuntimeError(f"bank({n_bank}) != src({n_src}) — 로더 필터가 켜져 있음. "
                           "검증은 학습과 동일한 무필터(spec 매칭) 설정으로 실행해야 한다.")
    idx = np.nonzero(keep_mask)[0]
    with h5py.File(args_cli.out, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        dst.attrs["count"] = len(idx)
        dst.attrs["meta/pour_validated"] = 1.0
        gout = dst.create_group("warm_states")
        for key in g.keys():
            gout.create_dataset(key, data=np.asarray(g[key])[idx])
    print(f"[validate] saved {len(idx)} states → {args_cli.out}", flush=True)


main()
simulation_app.close()

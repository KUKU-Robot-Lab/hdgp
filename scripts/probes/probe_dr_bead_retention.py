#!/usr/bin/env python3
"""DR(scale_set) 리셋·파지·bead 물리 안정성 probe — zero-action.

정책 없이 action=0으로 N스텝 굴려 bead_in_source_fraction / spill 을 env(스케일 조합)별로
측정한다. spill 이 zero-action 에서도 높으면 학습이 아니라 리셋/물리 구조 문제.

사용:
  isaaclab.sh -p scripts/probes/probe_dr_bead_retention.py --num_envs 20 --steps 400 \
      "env.left_target_cup_scale_set=[0.8,0.9,1.0,1.1,1.2]" \
      "env.source_cup_scale_set=[0.85,1.0,1.15,1.3]"
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=20)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--task", type=str, default="open-tesol_b_pour_sensor-lstm")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys

sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym
import torch

import openarm.tesollo  # noqa: F401  (gym 등록)
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    env.reset()
    zero = torch.zeros(uenv.num_envs, uenv.cfg.action_space, device=uenv.device)

    src_scale = getattr(uenv, "_src_scale_env", torch.ones(uenv.num_envs))
    tgt_scale = getattr(uenv, "_tgt_scale_env", torch.ones(uenv.num_envs))

    # 실제 스폰 검증: USD 프림 스케일 + 물리 질량 (가정 env_id%K 와 대조)
    from pxr import UsdGeom
    import isaacsim.core.utils.stage as stage_utils
    stage = stage_utils.get_current_stage()
    masses = uenv.cup.root_physx_view.get_masses().squeeze(-1) if hasattr(uenv.cup, "root_physx_view") else None
    print("[probe] env별 (가정 src_scale | 실제 USD scale | mass kg):", flush=True)
    for i in range(uenv.num_envs):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Cup")
        usd_scale = None
        for p in [prim] + list(prim.GetChildren()):
            if p and p.IsValid():
                attr = p.GetAttribute("xformOp:scale")
                if attr and attr.Get() is not None:
                    usd_scale = tuple(round(float(v), 3) for v in attr.Get())
                    break
        m = float(masses[i]) if masses is not None else -1.0
        print(f"  env{i:02d} assumed={float(src_scale[i]):.2f} usd={usd_scale} mass={m:.4f}", flush=True)

    for t in range(args_cli.steps):
        env.step(zero)
        if (t + 1) % 100 == 0:
            bis = uenv._bead_in_source_fraction
            spill = uenv._spill_ratio
            print(f"[probe] step {t+1}: bead_in_source mean={bis.mean():.3f} "
                  f"min={bis.min():.3f} | spill mean={spill.mean():.3f} max={spill.max():.3f}",
                  flush=True)

    bis = uenv._bead_in_source_fraction
    spill = uenv._spill_ratio
    print("\n[probe] per-env 최종 (src_scale, tgt_scale, bead_in_source, spill):", flush=True)
    for i in range(uenv.num_envs):
        flag = "  <-- LOSS" if bis[i] < 0.9 else ""
        print(f"  env{i:02d} src={float(src_scale[i]):.2f} tgt={float(tgt_scale[i]):.2f} "
              f"in_src={float(bis[i]):.2f} spill={float(spill[i]):.2f}{flag}", flush=True)
    by = {}
    for i in range(uenv.num_envs):
        by.setdefault(round(float(src_scale[i]), 2), []).append(float(bis[i]))
    print("[probe] src_scale별 bead_in_source 평균:",
          {k: round(sum(v) / len(v), 3) for k, v in sorted(by.items())}, flush=True)


main()
simulation_app.close()

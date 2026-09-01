#!/usr/bin/env bash
cd /home/user/rl_ws/hdgp
export CUDA_VISIBLE_DEVICES=0
../IsaacLab/isaaclab.sh -p scripts/probes/dump_pour_cfg.py --out /tmp/claude-1000/pour_env.yaml

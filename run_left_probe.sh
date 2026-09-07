#!/usr/bin/env bash
cd /home/user/rl_ws/hdgp
export CUDA_VISIBLE_DEVICES=0
mkdir -p /tmp/claude-1000/leftarm
../IsaacLab/isaaclab.sh -p scripts/probes/probe_pour_left_arm_clear.py \
  --num_envs 4 --steps 60 --out /tmp/claude-1000/leftarm/L --headless "$@"

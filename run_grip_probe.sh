#!/usr/bin/env bash
cd /home/user/rl_ws/hdgp
export CUDA_VISIBLE_DEVICES=0
mkdir -p /tmp/claude-1000/griphold
../IsaacLab/isaaclab.sh -p scripts/probes/probe_pour_grip_hold.py \
  --num_envs 8 --steps 240 --out /tmp/claude-1000/griphold/g --headless

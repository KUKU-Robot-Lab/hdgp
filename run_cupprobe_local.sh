#!/usr/bin/env bash
cd /home/user/rl_ws/hdgp
export CUDA_VISIBLE_DEVICES=0
../IsaacLab/isaaclab.sh -p scripts/probes/probe_pour_outcome_by_cup.py \
  --checkpoint /tmp/claude-1000/ck/e1_pour1_snap2.pth \
  --adr "success=1.0,outcome=0.625,noise=0.15" \
  --num_envs 64 --episodes 6 --max_steps 6000 --headless

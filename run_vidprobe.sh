#!/usr/bin/env bash
cd /home/user/rl_ws/hdgp
export CUDA_VISIBLE_DEVICES=0
mkdir -p /tmp/claude-1000/vid
../IsaacLab/isaaclab.sh -p scripts/probes/probe_pour_video_by_cup.py \
  --checkpoint /tmp/claude-1000/ck/e1_pour1_snap2.pth \
  --cups shaker_closed,cup_big_s130 \
  --adr "success=1.0,outcome=0.625,noise=0.15" \
  --out /tmp/claude-1000/vid/e1 --max_steps 900 --headless

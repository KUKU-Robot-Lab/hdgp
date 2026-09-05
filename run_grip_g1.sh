#!/usr/bin/env bash
cd /home/user/rl_ws/hdgp
export CUDA_VISIBLE_DEVICES=0
../IsaacLab/isaaclab.sh -p scripts/probes/probe_pour_grip_hold.py \
  --num_envs 8 --steps 240 --headless \
  --bank data/grasp_warm_s2r_g1_n2048_maxgrip.hdf5 \
  --arm_gains "${1:-r2s}"

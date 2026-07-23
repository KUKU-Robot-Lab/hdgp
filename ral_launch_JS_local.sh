#!/usr/bin/env bash
# JS (#1 joint-space) LOCAL RTX5090 — 서버 JS와 동일 config. isaaclab.sh가 python 처리(conda 불필요).
cd /home/user/rl_ws/hdgp
CUDA_VISIBLE_DEVICES=0 NOTE="RA-L JS joint-space (boot OFF) seed42 LOCAL" ./train.sh open-tesol_b_pour_sensor-lstm JS_s42 --num_envs 2048 --seed 42 \
  env.right_arm_jointspace=true \
  env.receiver_control_mode=learned \
  env.pour_orient_release=false env.pour_bfull_nullspace=false env.nullspace_baseline=robot_start \
  env.enable_demo_critic_obs=true env.enable_demo_pose_reward=false \
  env.enable_deep_tilt_boot=false

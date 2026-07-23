#!/usr/bin/env bash
# reward ablation R_nointrot — NS_demo base(boot off) + env.weight_introt=0. cap ep4000. 로컬.
cd /home/user/rl_ws/hdgp
CUDA_VISIBLE_DEVICES=0 NOTE="RA-L R_nointrot (NS_demo base, weight_introt=0) seed42" ./train.sh open-tesol_b_pour_sensor-lstm R_nointrot_s42 --num_envs 2048 --seed 42 --max_iterations 4000 \
  env.receiver_control_mode=learned env.pour_orient_release=true env.pour_bfull_nullspace=true env.nullspace_baseline=demo env.enable_demo_critic_obs=true env.enable_demo_pose_reward=false env.enable_deep_tilt_boot=false env.weight_introt=0.0

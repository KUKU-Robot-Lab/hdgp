#!/usr/bin/env bash
# RA-L introt 방어실험 Run B: demo 앵커 3경로 전부 OFF + introt OFF(=0).
#   목적: 앵커도 introt도 없을 때 외회전/부호반전(j5 양수 표류)이 돌아오는지 검증 = Run A의 대조.
#   앵커 OFF = pour_orient_release=false + pour_bfull_nullspace=false + nullspace_baseline=robot_start.
cd /home/user/rl_ws/hdgp
CUDA_VISIBLE_DEVICES=0 NOTE="RA-L introtDef B: anchorOFF introt=0 seed42 solo cap3000 (posture 대조)" ./train.sh open-tesol_b_pour_sensor-lstm introtDef_B_off --num_envs 2048 --seed 42 --max_iterations 1300 \
  env.receiver_control_mode=learned env.pour_orient_release=false env.pour_bfull_nullspace=false env.nullspace_baseline=robot_start env.enable_demo_critic_obs=true env.enable_demo_pose_reward=false env.enable_deep_tilt_boot=false env.weight_introt=0.0

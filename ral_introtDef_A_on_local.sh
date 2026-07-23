#!/usr/bin/env bash
# RA-L introt 방어실험 Run A: demo 앵커 3경로 전부 OFF + introt ON(=5).
#   목적: 앵커 없을 때 introt이 내회전 자세를 복원하는지(j5 음수 / internal_rot_gate↑) 검증.
#   앵커 OFF = pour_orient_release=false + pour_bfull_nullspace=false + nullspace_baseline=robot_start.
cd /home/user/rl_ws/hdgp
CUDA_VISIBLE_DEVICES=0 NOTE="RA-L introtDef A: anchorOFF introt=5 seed42 solo cap3000 (posture 검증)" ./train.sh open-tesol_b_pour_sensor-lstm introtDef_A_on --num_envs 2048 --seed 42 --max_iterations 3000 \
  env.receiver_control_mode=learned env.pour_orient_release=false env.pour_bfull_nullspace=false env.nullspace_baseline=robot_start env.enable_demo_critic_obs=true env.enable_demo_pose_reward=false env.enable_deep_tilt_boot=false env.weight_introt=5.0

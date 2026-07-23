#!/usr/bin/env bash
# P1 eval — JS(로컬 checkpoint). joint-space config로 eval, boot off. num_envs 1024 seed100.
cd /home/user/rl_ws/hdgp
D=log/rl_games/open-tesol/both/pour-sensor
EVAL=scripts/reinforcement_learning/rl_games/eval_pour_envs.py
ISAAC=/home/user/rl_ws/IsaacLab/isaaclab.sh
ckpt=$(ls -t $D/JS_s42/nn/*.pth 2>/dev/null | head -1)
if [ -z "$ckpt" ]; then echo "!! JS checkpoint 없음(JS_s42) — rename 후 실행"; exit 1; fi
echo "########## EVAL JS ckpt=$(basename $ckpt) ##########"
CUDA_VISIBLE_DEVICES=0 "$ISAAC" -p "$EVAL" --task open-tesol_b_pour_sensor-lstm --checkpoint "$ckpt" \
  --num_envs 1024 --seed 100 --eval_steps 1200 --headless \
  --eval_out /home/user/rl_ws/hdgp/eval_JS_matrix.md \
  env.receiver_control_mode=learned env.enable_deep_tilt_boot=false \
  env.right_arm_jointspace=true \
  env.pour_orient_release=false env.pour_bfull_nullspace=false env.nullspace_baseline=robot_start \
  env.enable_demo_critic_obs=true env.enable_demo_pose_reward=false \
  2>&1 | grep -vE "omni.usd|_ReportErrors|Unresolved|recomposing" | tail -22
echo "########## JS DONE ##########"

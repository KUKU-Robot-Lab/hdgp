#!/usr/bin/env bash
cd /home/user/rl_ws/hdgp
export CUDA_VISIBLE_DEVICES=0
../IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
  --task open-tesol_r_pour_sensor-play-lstm \
  --checkpoint log/rl_games/open-tesol/right/pour-sensor/b1_multicup1-r3/nn/last_open-tesol_r_pour_sensor-lstm_ep_4500_rew_5585.962.pth \
  --num_envs 8 --video --video_length 600 --headless

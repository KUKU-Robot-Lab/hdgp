#!/usr/bin/env bash
cd /home/user/rl_ws/hdgp
export CUDA_VISIBLE_DEVICES=0
RUN_LABEL=b1_multicup1 \
NOTE="b1 warm bank(2049상태·8컵종) 기반 fresh 학습. 09.01 변경: ①받는컵 shaker_closed(바닥 콜라이더) ②붓는컵 8종 기하를 뱅크 스펙에서 env별 파생(림 0.0829~0.1304·내벽 0.0348~0.0532·내부하한 -0.0932~-0.0584) ③target_* 상수를 받는컵 스펙에서 파생(구 cup_big 고정값은 shaker 바닥(-0.0921)보다 위여서 가라앉은 bead를 spill로 오집계) ④warmstart_palm_z_boost 0.12→0.0 (b1 상태가 이미 lifted, 이중 리프트로 palm 목표 12cm 클램프)" \
./train.sh open-tesol_r_pour_sensor-lstm b1_multicup1 --num_envs 1024

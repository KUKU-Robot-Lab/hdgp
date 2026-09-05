#!/bin/bash
# Step 2 반복 실행 래퍼 — 재캡처(비전 실측) → 러너 재기동을 한 방에.
# 사용: ./step2_rerun.sh  (hdgp 루트 기준 상대경로 무관)
set -e
SIMLOG=/home/user/rl_ws/sim2real/logs
SCRATCH=${SCRATCH:-/tmp}
ssh vision-3090 'source /opt/ros/humble/setup.bash; export ROS_DOMAIN_ID=126; cd /tmp
  timeout 60 python3 cup_pose_capture.py --side right --out /tmp/cup_right_step2.json 2>&1 | grep -E "컵|✅|밖" | head -2
  timeout 60 python3 cup_pose_capture.py --side left --topic /shaker_pose --out /tmp/shaker_left_step2.json 2>&1 | grep -E "컵|✅|밖" | head -2; exit 0'
scp -q vision-3090:/tmp/cup_right_step2.json vision-3090:/tmp/shaker_left_step2.json $SIMLOG/
pkill -f "[p]robe_bimanual" 2>/dev/null && sleep 5 || true
cd /home/user/rl_ws/hdgp
HDGP_S2R_REAL_GAINS=1 exec /home/user/rl_ws/IsaacLab/isaaclab.sh -p scripts/probes/probe_bimanual_closedloop.py \
  --gui --auto --skip-pour --force-spawn \
  --right-cup-json $SIMLOG/cup_right_step2.json \
  --left-cup-json $SIMLOG/shaker_left_step2.json "$@"

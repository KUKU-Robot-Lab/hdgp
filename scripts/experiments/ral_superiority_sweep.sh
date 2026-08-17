#!/usr/bin/env bash
# RA-L — "우리 구조(explicit null-space deep tilt)가 어디서 우월한가" 확정 스윕.
#
# 배경(2026-08-17): 신 USD(실기 정합, 팔 -8mm)에서 nominal 성공률(≥50% 이송)로는
# 조건이 갈리지 않는다 — demo prior가 전혀 없는 NS_naive도 98.6%가 나온다.
# 그러나 **깊은 tilt 능력**은 명확히 갈린다 (연구일지 8단계 레퍼토리의 병목 지표):
#
#   런            tilt      j5 깊이   j6 포화   complete(20/20)
#   Full          0.532     -1.038    0.085     30.5%
#   NS_demo       0.486     -0.980    0.081      6.6%
#   P2true(보상)   0.331     -0.448    0.302        —
#   NS_naive      0.306     -0.096    0.548      5.8%   ← 얕은 tilt + 손목 포화
#
# NS_naive의 j5≈0 / j6_sat 0.55는 연구일지가 기록한 **실패 시그니처 그대로**다.
# 즉 얕은 전략은 nominal에서만 통한다. 이 스윕은 깊은 tilt가 **필요한** 조건에서
# 격차가 드러나는지를 학습 없이(eval만) 확정한다.
#
# 사용: GPU=0 ./ral_superiority_sweep.sh <조건...>
#   조건 = NS_demo | NS_naive | Full | P2true
set -euo pipefail

HDGP=/home/oem/rl_ws/hdgp
D=$HDGP/log/rl_games/open-tesol/both/pour-sensor
EVAL=$HDGP/scripts/reinforcement_learning/rl_games/eval_pour_envs.py
ISAAC=/home/oem/rl_ws/IsaacLab/isaaclab.sh
OUT=$HDGP/docs/eval/superiority
mkdir -p "$OUT"

MECH_ON="env.pour_orient_release=true  env.pour_bfull_nullspace=true  env.nullspace_baseline=demo"
MECH_OFF="env.pour_orient_release=false env.pour_bfull_nullspace=false env.nullspace_baseline=robot_start"

cond_dir()  { case "$1" in
  NS_demo)  echo NS_demo_s42_newUSD ;;  NS_naive) echo NS_naive_s42_newUSD ;;
  Full)     echo Full_M4_s42_newUSD ;;  P2true)   echo P2true_s42_newUSD ;;  esac; }
cond_env()  { case "$1" in
  NS_demo)  echo "$MECH_ON  env.enable_demo_pose_reward=false" ;;
  NS_naive) echo "$MECH_OFF env.enable_demo_pose_reward=false" ;;
  Full)     echo "$MECH_ON  env.enable_demo_pose_reward=false" ;;
  P2true)   echo "$MECH_OFF env.enable_demo_pose_reward=true"  ;;  esac; }

# 깊은 tilt가 필요한 조건들 (label:추가 CLI)
SETTINGS=(
  "bead30:--bead_fixed 30"          # E1 완전 배출 — 얕은 tilt는 바닥 잔량
  "xy0.8:--cup_scale_xy 0.8"        # E2 좁은 입구(높이 고정) — 전략역전 축
  "xy0.9:--cup_scale_xy 0.9"
  "uni0.8:--cup_scale 0.8"          # E3 기하 일반화
  "uni1.2:--cup_scale 1.2"
)

for c in "$@"; do
  dir=$(cond_dir "$c"); envov=$(cond_env "$c")
  ckpt=$(ls -t "$D/$dir"/nn/*ep_*.pth 2>/dev/null | head -1)
  [[ -n "$ckpt" ]] || { echo "!! $c: ckpt 없음 ($dir)"; continue; }
  for s in "${SETTINGS[@]}"; do
    label="${s%%:*}"; extra="${s#*:}"
    echo "########## $c / $label  ($(date '+%H:%M:%S')) ##########"
    CUDA_VISIBLE_DEVICES="${GPU:-0}" "$ISAAC" -p "$EVAL" \
      --task open-tesol_b_pour_sensor-lstm --checkpoint "$ckpt" \
      --num_envs 1024 --seed 100 --eval_steps 1200 --headless \
      --eval_out "$OUT/eval_${c}_${label}.md" \
      $extra \
      env.receiver_control_mode=learned env.enable_deep_tilt_boot=false \
      env.enable_demo_critic_obs=true $envov \
      2>&1 | grep -E "^\[EVAL\] 총|!!|Error" | tail -3
  done
done
echo "===== 스윕 완료 ====="

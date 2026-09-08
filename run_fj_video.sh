#!/usr/bin/env bash
# grasp_fj 체크포인트 재생 → 영상. ★set -u 금지(isaacsim setup 함정).
#   사용법: ./run_fj_video.sh <라벨> <체크포인트경로> [추가 hydra...]
# 게이트: 벤더 SAPG 포크가 안 잡히면 agent cfg 의 SAPG 키에서 죽거나 조용히 일반 PPO 로 된다.
LABEL="${1:?사용법: run_fj_video.sh <라벨> <ckpt> [hydra...]}"
CKPT="${2:?체크포인트 경로가 필요하다}"
shift 2

HDGP=/home/user/rl_ws/hdgp
VENDOR="$HDGP/vendor/rl_games_sapg"
cd "$HDGP" || exit 1
[ -f "$CKPT" ] || { echo "★중단: 체크포인트 없음 — $CKPT"; exit 1; }
[ -d "$VENDOR/rl_games" ] || { echo "★중단: 벤더 SAPG 포크 없음 — $VENDOR"; exit 1; }

export PYTHONPATH="$VENDOR:$HDGP/source/openarm:${PYTHONPATH:-}"   # ★앞붙이기
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

echo "=== fj 재생: $LABEL · $(basename "$CKPT") ==="
../IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
  --task open-sens_r_grasp_fj-play-lstm-sapg \
  --checkpoint "$CKPT" \
  --num_envs "${ENVS:-8}" --headless \
  --view_env_index "${VIEW_ENV:-7}" \
  --video --video_length "${VLEN:-600}" \
  ${DUMP_HAND_Q:+--dump_hand_q "$DUMP_HAND_Q"} \
  ${OBJECT_USD_DIR:+--object_usd_dir "$OBJECT_USD_DIR"} \
  ${PHYS:+--phys "$PHYS"} \
  --cam_eye "${CAM_EYE:-0.95,-0.62,0.62}" --cam_lookat "${CAM_LOOKAT:-0.36,-0.16,0.32}" \
  "$@"

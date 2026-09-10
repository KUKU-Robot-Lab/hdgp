#!/usr/bin/env bash
# grasp_fj 체크포인트 재생 → 영상. ★set -u 금지(isaacsim setup 함정).
#
#   사용법: TASK=open-short_r_grasp_fj-play-lstm-sapg LABEL=fj_w2 \
#           CKPT=<경로> GPU=0 ./run_fj_video.sh
#
# ★TASK 를 **필수**로 둔다. 09.10 에 옛 학습 런처가 `open-sens` 를 박아둔 채 남아
#   있어 24576 env × 45 epoch 를 **옛 로봇**으로 돌렸다. 재생 스크립트도 같은 결함을
#   갖고 있었다 — 용접 자산(19관절) 체크포인트를 20관절 태스크로 재생하게 된다.
#
# ★★SAPG 블록 규약. player 는 `num_envs % expl_coef_block_size == 0` 을 강제하고,
#   블록 수가 학습과 달라지면 네트워크의 sigma 행 대조가 전부 거짓이 되어 재생이
#   **블록 0(가장 탐색적)** 으로 돌아간다(vendor player.py:92 주석).
#   그래서 여기서는 **블록 수를 학습과 같게** 맞추고 블록 크기만 줄인다:
#       BLOCKS(기본 12, 학습 24576/2048) · ENVS = BLOCKS × BLOCK_SIZE
#   마지막 블록이 탐색계수 0 = **리더(그리디)** 라, 기본 시점은 거기를 본다.
set -o pipefail
HDGP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HDGP" || exit 1

: "${TASK:?TASK 필요 — 예: open-short_r_grasp_fj-play-lstm-sapg (open-sens 는 옛 dg5f-m)}"
: "${LABEL:?LABEL 필요 — 영상 이름}"
: "${CKPT:?CKPT 필요 — 체크포인트 경로}"
[ -f "$CKPT" ] || { echo "★중단: 체크포인트 없음 — $CKPT"; exit 1; }

if [ -d "$HDGP/../IsaacLab" ] && [ -z "$SERVER" ]; then
  LAUNCH=(../IsaacLab/isaaclab.sh -p)
else
  source ~/miniforge3/etc/profile.d/conda.sh
  conda activate proj-hdgp-py311
  source /home/oem/isaacsim/5.1.0/setup_conda_env.sh
  LAUNCH=(python)
fi
export PYTHONPATH="$HDGP/vendor/rl_games_sapg:$HDGP/source/openarm:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

BLOCKS="${BLOCKS:-12}"; BSZ="${BLOCK_SIZE:-2}"
E=$((BLOCKS * BSZ))
VIEW="${VIEW_ENV:-$((E - 1))}"          # 마지막 블록 = 탐색계수 0 = 리더
echo "[영상] task=$TASK · ckpt=$(basename "$CKPT")"
echo "[영상] git HEAD=$(git rev-parse --short HEAD) dirty=$(git status --short | grep -c '^ M')개"
echo "[영상] env=$E = ${BLOCKS}블록 × ${BSZ} · 시점 env=$VIEW(마지막 블록=그리디)"

exec "${LAUNCH[@]}" scripts/reinforcement_learning/rl_games/play.py \
  --task "$TASK" --checkpoint "$CKPT" \
  --num_envs "$E" --headless --view_env_index "$VIEW" \
  --video --video_length "${VLEN:-700}" \
  agent.params.config.expl_coef_block_size=$BSZ \
  --cam_eye "${CAM_EYE:-0.95,-0.62,0.62}" --cam_lookat "${CAM_LOOKAT:-0.36,-0.16,0.32}" \
  ${EXTRA}

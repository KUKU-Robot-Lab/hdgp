#!/usr/bin/env bash
# grasp_s2r 체크포인트 재생 → 영상. ★set -u 금지(isaacsim setup 함정).
#
#   사용법: LABEL=e1_ep20000 ./run_s2r_video.sh
#           CKPT=<경로> LABEL=<이름> ./run_s2r_video.sh      (체크포인트 직접 지정)
#
# ★TASK 를 기본값으로 두되 **short 자산**을 박는다. 09.10 에 옛 학습 런처가
#   `open-sens` 를 박아둔 채 남아 있어 24576 env × 45 epoch 를 **옛 로봇**으로 돌린
#   사고가 있었다. 재생도 같은 사고를 낼 수 있다 — e1_fresh 는 short 자산 런이다.
#
# ★★영상은 `~/rl_ws/our_source/` 로 옮겨 이름을 붙인다. IsaacLab 이 쓰는 파일명은
#   전부 `rl-video-step-0.mp4` 라 log 밑에 그대로 두면 서로 구분이 안 된다.
#
# ★★드라이버 전제. 09.11 에 학습 도중 nvidia-driver-580-open 이 580.173.02 →
#   580.178.04 로 갱신됐고, 커널 모듈은 GPU 를 잡은 학습 때문에 교체되지 못했다.
#   그 상태에서 새 CUDA 프로세스는 error 804 로 죽는다. 아래 가드가 그것을 먼저 잡는다.
set -o pipefail
HDGP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HDGP" || exit 1

TASK="${TASK:-open-short_r_grasp_s2r-play-lstm}"
RUN="${RUN:-e1_fresh}"
: "${LABEL:?LABEL 필요 — 영상 이름 (예: e1_ep20000)}"

# ---- 체크포인트 (미지정이면 런에서 가장 최근) --------------------------------------
if [ -z "${CKPT}" ]; then
  CKPT=$(ls -t "log/rl_games/open-short/right/grasp-s2r/${RUN}/nn/"*.pth 2>/dev/null | head -1)
fi
[ -f "$CKPT" ] || { echo "★중단: 체크포인트 없음 — ${CKPT:-<없음>}"; exit 1; }

# ---- 드라이버 정합 가드 ------------------------------------------------------------
KMOD=$(sed -n 's/.*NVRM version:.*x86_64  \([0-9.]*\).*/\1/p' /proc/driver/nvidia/version 2>/dev/null)
ULIB=$(ls -1 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.[0-9]* 2>/dev/null \
        | sed 's/.*so\.//' | sort -V | tail -1)
if [ -n "$KMOD" ] && [ -n "$ULIB" ] && [ "$KMOD" != "$ULIB" ]; then
  echo "★중단: NVIDIA 커널 모듈($KMOD) ≠ userspace 라이브러리($ULIB)"
  echo "  새 CUDA 프로세스가 error 804 로 죽는다. GPU 를 쓰는 프로세스를 전부 끄고"
  echo "  재부팅하거나 모듈을 재적재할 것:"
  echo "    sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia && sudo modprobe nvidia"
  exit 1
fi
if ! nvidia-smi >/dev/null 2>&1; then
  echo "★중단: nvidia-smi 실패 — 드라이버 상태를 먼저 확인할 것"; exit 1
fi

if [ -d "$HDGP/../IsaacLab" ] && [ -z "$SERVER" ]; then
  LAUNCH=(../IsaacLab/isaaclab.sh -p)
else
  source ~/miniforge3/etc/profile.d/conda.sh
  conda activate proj-hdgp-py311
  source ~/isaacsim/5.1.0/setup_conda_env.sh
  LAUNCH=(python)
fi
export PYTHONPATH="$HDGP/source/openarm:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

E="${ENVS:-16}"
VIEW="${VIEW_ENV:-0}"
OUT="${OUT_DIR:-$HOME/rl_ws/our_source}"
mkdir -p "$OUT"

echo "[영상] task=$TASK · ckpt=$(basename "$CKPT")"
echo "[영상] git HEAD=$(git rev-parse --short HEAD) dirty=$(git status --short | grep -c '^ M')개"
echo "[영상] env=$E · 시점 env=$VIEW · 길이 ${VLEN:-700} 스텝 · 드라이버 $KMOD"

"${LAUNCH[@]}" scripts/reinforcement_learning/rl_games/play.py \
  --task "$TASK" --checkpoint "$CKPT" \
  --num_envs "$E" --headless --view_env_index "$VIEW" \
  --video --video_length "${VLEN:-700}" \
  --cam_eye "${CAM_EYE:-0.95,-0.62,0.62}" --cam_lookat "${CAM_LOOKAT:-0.36,-0.16,0.32}" \
  ${EXTRA}
RC=$?

# ---- 산출물 이름 붙여 옮기기 --------------------------------------------------------
SRC=$(ls -t log/rl_games/open-short/right/grasp-s2r/*/videos/play/*.mp4 2>/dev/null | head -1)
if [ -n "$SRC" ] && [ -f "$SRC" ]; then
  DST="$OUT/s2r_${LABEL}_$(basename "$CKPT" .pth | sed 's/.*_ep_/ep/;s/_rew_.*//')_$(date +%m%d_%H%M).mp4"
  cp "$SRC" "$DST" && echo "[영상] 저장: $DST"
else
  echo "[영상] ⚠mp4 를 못 찾았다 — play 로그를 확인할 것 (rc=$RC)"
fi
exit $RC

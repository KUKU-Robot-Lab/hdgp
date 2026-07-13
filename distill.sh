#!/usr/bin/env bash
# 증류 시작 래퍼: teacher 체크포인트 연결 + GPU 지정 + torchrun 포트 격리
#
# 사용법:
#   ./distill.sh <task_id> <label> <teacher_ckpt> [추가 인자...]
#
# 예시:
#   GPU=0 ./distill.sh open-tesol_r_grasp_v2-distill test1 \
#       log/rl_games/open-tesol/right/grasp-v2/lstm_test12/nn/last_....pth
#
#   GPU=1 ./distill.sh open-tesol_l_grasp_v2-distill test1 \
#       log/rl_games/open-tesol/left/grasp-v2/lstm_test6/nn/last_....pth
#
# 환경변수:
#   GPU=N          - 사용할 GPU (기본 0). CUDA_VISIBLE_DEVICES 로 넘어간다.
#   NPROC=N        - GPU N장 DDP (기본 1). GPU 를 여러 장 쓸 때만.
#   ISAACLAB_ROOT  - IsaacLab 루트 (기본: hdgp 의 형제 ../IsaacLab)
#
# 추가 인자 예: --student <ckpt>  (중단된 student 학습 재개)
#              --num_envs 32     (스모크)
#              --play_policy     (학습 없이 rollout)
#
# 왜 --standalone 을 안 쓰나:
#   같은 호스트에서 --standalone 잡을 둘 이상 띄우면 rendezvous 포트가 겹쳐
#   포트 충돌이 나거나, 더 나쁘게는 **두 잡이 하나의 잡으로 병합**된다(torch 공식 경고).
#   right/left 를 GPU0/GPU1 에서 동시에 돌리면 정확히 이 상황이다.
#   --rdzv-endpoint=localhost:0 은 잡마다 빈 포트를 새로 잡아 이 사고를 막는다.

set -euo pipefail

TASK="${1:?'Usage: ./distill.sh <task_id> <label> <teacher_ckpt> [args...]'}"
LABEL="${2:?'Usage: ./distill.sh <task_id> <label> <teacher_ckpt> [args...]'}"
TEACHER="${3:?'Usage: ./distill.sh <task_id> <label> <teacher_ckpt> [args...]'}"
shift 3

HDGP_ROOT="$(cd "$(dirname "$0")" && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$(cd "${HDGP_ROOT}/.." && pwd)/IsaacLab}"
GPU="${GPU:-0}"
NPROC="${NPROC:-1}"

if [[ "$TASK" != *-distill ]]; then
    echo "ERROR: '$TASK' 는 증류 태스크가 아니다 (…-distill 이어야 한다)." >&2
    exit 1
fi

if [[ ! -f "$TEACHER" ]]; then
    echo "ERROR: teacher 체크포인트가 없다: $TEACHER" >&2
    exit 1
fi

echo "============================================"
echo " Distillation 시작"
echo "  태스크 : $TASK"
echo "  라벨   : $LABEL"
echo "  teacher: $TEACHER"
echo "  GPU    : $GPU  (nproc=$NPROC)"
echo "  로그   : log/distillation/${TASK}/${LABEL}/"
echo "============================================"

CUDA_VISIBLE_DEVICES="$GPU" \
"${ISAACLAB_ROOT}/isaaclab.sh" -p -m torch.distributed.run \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --nnodes=1 \
    --nproc_per_node="$NPROC" \
    "${HDGP_ROOT}/scripts/distillation/run_distillation.py" \
    --task "$TASK" \
    --teacher "$TEACHER" \
    --label "$LABEL" \
    --headless \
    "$@"

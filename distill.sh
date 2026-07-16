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

# 자동 실패물체 제외(옵션, 기본 비활성). teacher 로그에서 성공률<임계 물체를 추출해
# 이 arm 의 DISTILL cfg(DISTILL_EXCLUDED_OBJECT_NAMES)에 주입한다 — config 에 기록되어
# 커밋·재현 가능. onehot(153)은 유지되고 스폰만 빠져 teacher 체크포인트와 호환된다.
#   AUTO_EXCLUDE=1               활성화
#   AUTO_EXCLUDE_THRESHOLD=0.3   제외 임계 (기본 0.3)
# 좌우 합집합을 원하면 이 옵션 대신 extract_failing_objects.py --right --left --write 를
# 먼저 수동 실행하라(이 옵션은 실행 중인 arm 것만 주입한다).
if [[ -n "${AUTO_EXCLUDE:-}" ]]; then
    RUN_DIR="$(cd "$(dirname "$TEACHER")/.." && pwd)"   # <run>/nn/x.pth → <run>
    if [[ "$TASK" == *_r_* ]]; then SIDE_FLAG="--right"
    elif [[ "$TASK" == *_l_* ]]; then SIDE_FLAG="--left"
    else echo "ERROR: TASK 에서 arm(_r_/_l_) 판별 불가: $TASK" >&2; exit 1; fi
    THRESH="${AUTO_EXCLUDE_THRESHOLD:-0.3}"
    echo ">> AUTO_EXCLUDE: teacher 실패물체(<${THRESH}) 추출→주입 (${RUN_DIR})"
    python3 "${HDGP_ROOT}/scripts/distillation/extract_failing_objects.py" \
        "$SIDE_FLAG" "$RUN_DIR" --threshold "$THRESH" --write
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

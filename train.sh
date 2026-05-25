#!/usr/bin/env bash
# 학습 시작 래퍼: 스냅샷 자동 기록 후 rl_games 학습 실행
#
# 사용법:
#   ./train.sh <task_name> <test_name> [추가 인자...]
#
# 예시:
#   ./train.sh 5g_pour_right_v5 test7
#   ./train.sh 5g_pour_right_v5 test7 --num_envs 2048
#
# 환경변수:
#   NOTE="설명"  - test_history.md에 기록할 메모

set -euo pipefail

TASK="${1:?'Usage: ./train.sh <task_name> <test_name> [args...]'}"
TEST="${2:?'Usage: ./train.sh <task_name> <test_name> [args...]'}"
shift 2

HDGP_ROOT="$(cd "$(dirname "$0")" && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/user/rl_ws/IsaacLab}"
TRAIN_SCRIPT="${ISAACLAB_ROOT}/scripts/reinforcement_learning/rl_games/train.py"

# 태스크 이름 → IsaacLab gym 등록 이름 변환
# 예: 5g_pour_right_v5 → OpenArm-PourRight-v5
gym_name() {
    local raw="$1"
    # 5g_pour_right_v5 → PourRight-v5 → OpenArm-PourRight-v5
    local body
    body=$(echo "$raw" | sed 's/^5g_//' | sed 's/_v\([0-9]\+\)$/-v\1/' | sed 's/_\([a-z]\)/\u\1/g')
    echo "OpenArm-${body^}"
}

GYM_TASK="$(gym_name "$TASK")"

echo "============================================"
echo " RL 학습 시작"
echo "  태스크:  $TASK  →  $GYM_TASK"
echo "  테스트:  $TEST"
echo "============================================"

# 1. test_history.md 스냅샷 기록
echo "[1/2] 코드 스냅샷 기록 중..."
python3 "${HDGP_ROOT}/scripts/tools/record_test_snapshot.py" \
    --task "$TASK" \
    --test "$TEST" \
    ${NOTE:+--note "$NOTE"}

echo ""
echo "[2/2] 학습 시작: $GYM_TASK"
echo ""

# 2. 학습 실행 (나머지 인자 전달)
python3 "${TRAIN_SCRIPT}" \
    --task "$GYM_TASK" \
    --headless \
    "$@"

#!/usr/bin/env bash
# 학습 시작 래퍼: 스냅샷 자동 기록 후 rl_games 학습 실행
#
# 사용법:
#   ./train.sh <task_id> <test_name> [추가 인자...]
#
# 예시:
#   ./train.sh open-tesol_r_pour_v3 test8
#   ./train.sh open-tesol_r_pour_v3 test8 --num_envs 2048
#   ./train.sh open-tesol_r_grasp_v11 test3 --num_envs 512 --checkpoint log/rl_games/...
#
# 환경변수:
#   ISAACLAB_ROOT  - IsaacLab 루트 (기본값: /home/user/rl_ws/IsaacLab)
#   NOTE="설명"    - test_history.md에 기록할 메모

set -euo pipefail

TASK="${1:?'Usage: ./train.sh <task_id> <test_name> [args...]'}"
TEST="${2:?'Usage: ./train.sh <task_id> <test_name> [args...]'}"
shift 2

HDGP_ROOT="$(cd "$(dirname "$0")" && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/user/rl_ws/IsaacLab}"

echo "============================================"
echo " RL 학습 시작"
echo "  태스크: $TASK"
echo "  테스트: $TEST"
echo "============================================"

# 1. test_history.md 스냅샷 기록
echo "[1/2] 코드 스냅샷 기록 중..."
python3 "${HDGP_ROOT}/scripts/tools/record_test_snapshot.py" \
    --task "$TASK" \
    --test "$TEST" \
    ${NOTE:+--note "$NOTE"}

echo ""
echo "[2/2] 학습 시작: $TASK"
echo ""

# 2. 학습 실행 (나머지 인자 전달)
"${ISAACLAB_ROOT}/isaaclab.sh" -p \
    "${HDGP_ROOT}/scripts/reinforcement_learning/rl_games/train.py" \
    --task "$TASK" \
    --headless \
    "$@"

#!/usr/bin/env bash
# 학습 시작 래퍼: 라벨/노트를 env로 넘기면 train.py가 run 폴더 안에 test_history.md를 기록
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
#   ISAACLAB_ROOT  - IsaacLab 루트 (기본값: hdgp의 형제 디렉터리 ../IsaacLab)
#   NOTE="설명"    - test_history.md에 기록할 메모

set -euo pipefail

TASK="${1:?'Usage: ./train.sh <task_id> <test_name> [args...]'}"
TEST="${2:?'Usage: ./train.sh <task_id> <test_name> [args...]'}"
shift 2

HDGP_ROOT="$(cd "$(dirname "$0")" && pwd)"
if [ -z "${ISAACLAB_ROOT:-}" ]; then
    if [ -d "/home/user/rl_ws/IsaacLab" ]; then
        ISAACLAB_ROOT="/home/user/rl_ws/IsaacLab"
    else
        ISAACLAB_ROOT="$(cd "${HDGP_ROOT}/.." && pwd)/IsaacLab"
    fi
fi

echo "============================================"
echo " RL 학습 시작"
echo "  태스크: $TASK"
echo "  테스트: $TEST"
echo "============================================"

# 학습 실행. 스냅샷은 train.py가 run 폴더 확정 직후 직접 기록(RUN_LABEL/NOTE env 사용).
# 폴더명을 아는 곳이 train.py뿐이라(auto-increment) 사전 기록은 폴더 불일치/race 유발 → 통합.
echo "학습 시작: $TASK"
echo ""
RUN_LABEL="$TEST" NOTE="${NOTE:-}" \
"${ISAACLAB_ROOT}/isaaclab.sh" -p \
    "${HDGP_ROOT}/scripts/reinforcement_learning/rl_games/train.py" \
    --task "$TASK" \
    --headless \
    --track \
    --wandb-entity "pumky-konkuk-university" \
    --wandb-project-name "KUKU-Robot-reach" \
    --wandb-name "${TASK}_${TEST}" \
    "$@"

#!/usr/bin/env bash
# RA-L Active-Receiver ablation — 순차 학습 러너.
#
# EXP-1(Table I)의 M0/M2/M4를 동일 env·동일 15D action에서 receiver 제어만 바꿔
# 순서대로(foreground = 한 GPU 큐잉) 학습한다. 각 run은 train.sh가 test_history를 기록.
#
# 사용 (server 학습 env에서, conda proj-hdgp-py311 활성 상태):
#   CUDA_VISIBLE_DEVICES=0 NUM_ENVS=2048 ./scripts/experiments/ral_receiver_ablation.sh 42
#   두 GPU 병렬: seed별로 GPU 나눠 별도 호출
#     CUDA_VISIBLE_DEVICES=0 ... ral_receiver_ablation.sh 42   (한 셸)
#     CUDA_VISIBLE_DEVICES=1 ... ral_receiver_ablation.sh 43   (다른 셸)
#
# 환경변수:
#   NUM_ENVS   병렬 env 수 (기본 2048)
#   MAX_ITER   max_iterations override (선택)
#   METHODS    돌릴 method 부분집합 (기본 "M4 M0 M2"; 예: METHODS="M0 M2")
#
# 결과 수집: ./scripts/experiments/ral_collect.py 참조.
set -euo pipefail

HDGP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TASK="open-tesol_b_pour_sensor-lstm"
SEED="${1:-42}"
NUM_ENVS="${NUM_ENVS:-2048}"
METHODS="${METHODS:-M4 M0 M2}"

# method → receiver_control_mode 매핑 (M4=learned, M0=frozen, M2=scripted)
declare -A MODE=( [M4]=learned [M0]=frozen [M2]=scripted )

COMMON=(--num_envs "$NUM_ENVS" --seed "$SEED")
[[ -n "${MAX_ITER:-}" ]] && COMMON+=(--max_iterations "$MAX_ITER")

echo "============================================================"
echo " RA-L receiver ablation  |  seed=$SEED  num_envs=$NUM_ENVS"
echo " GPU=${CUDA_VISIBLE_DEVICES:-?}  methods=[$METHODS]"
echo "============================================================"

for m in $METHODS; do
  mode="${MODE[$m]:-}"
  [[ -z "$mode" ]] && { echo "!! 알 수 없는 method: $m (M4/M0/M2)"; exit 1; }
  label="${m}_C0_s${SEED}"
  echo
  echo ">>> [$label] receiver_control_mode=$mode  ($(date '+%H:%M:%S'))"
  NOTE="RA-L EXP-1 $m (receiver=$mode) seed=$SEED" \
    "$HDGP_ROOT/train.sh" "$TASK" "$label" \
      "${COMMON[@]}" \
      env.receiver_control_mode="$mode"
  echo "<<< [$label] 완료 ($(date '+%H:%M:%S'))"
done

echo
echo "전체 완료. 결과 수집: python3 scripts/experiments/ral_collect.py"

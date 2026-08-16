#!/usr/bin/env bash
# RA-L 논문 매트릭스 — 신 USD 기준 전면 재생성 (Table I + Table II).
#
# 배경(2026-08-16): 로봇 USD가 66b4843(07-29)에서 head 카메라 부착본으로 교체되며
# **팔이 8mm 내려갔다**(커밋: "s2r base 정합"). 논문 런은 전부 07-21 이전 = 구 로봇이라,
# 실기(s2r)로 논문을 확정하려면 sim도 같은 로봇이어야 한다 → 8조건 전면 재생성.
#
# ★ NS_naive 정의 주의 —— `nullspace_baseline`은 단독으로는 죽은 코드다.
#   pour_right_env.py:1550의 `if pour_orient_release:` 분기가 baseline과 무관하게
#   demo 자세를 강제하므로, orient_release가 켜져 있으면 baseline 플래그는 무시된다.
#   따라서 "여자유도 해소 메커니즘 off"는 아래 **3플래그를 한 단위로** 끄는 것으로 정의한다:
#     nullspace_baseline=robot_start + pour_orient_release=False + pour_bfull_nullspace=False
#   (논문 본문도 이 단위로 서술할 것 — 단일 플래그 분리는 구조상 불가능하다.)
#
# 사용 (server, conda proj-hdgp-py311 활성):
#   CUDA_VISIBLE_DEVICES=0 ./scripts/experiments/ral_paper_matrix.sh Full
#   CUDA_VISIBLE_DEVICES=1 ./scripts/experiments/ral_paper_matrix.sh NS_demo
#   여러 개 순차: ./scripts/experiments/ral_paper_matrix.sh R_noaim R_noalign
#
# 환경변수: NUM_ENVS(2048) SEED(42) MAX_ITER(6500) SUFFIX(_newUSD)
set -euo pipefail

HDGP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TASK="open-tesol_b_pour_sensor-lstm"
NUM_ENVS="${NUM_ENVS:-2048}"
SEED="${SEED:-42}"
MAX_ITER="${MAX_ITER:-6500}"
SUFFIX="${SUFFIX:-_newUSD}"

# 메커니즘 ON = 논문 제안 방식 (demo baseline + null-space 투영 + orientation release)
MECH_ON=(
  env.nullspace_baseline=demo
  env.pour_orient_release=True
  env.pour_bfull_nullspace=True
)
# 메커니즘 OFF = 3플래그 한 단위 (naive task-space)
MECH_OFF=(
  env.nullspace_baseline=robot_start
  env.pour_orient_release=False
  env.pour_bfull_nullspace=False
)

cond_args() {
  case "$1" in
    # ---- Table I: 핵심 4조건 ----
    NS_demo)   printf '%s\n' "${MECH_ON[@]}"  env.enable_deep_tilt_boot=False ;;
    Full)      printf '%s\n' "${MECH_ON[@]}"  env.enable_deep_tilt_boot=True  ;;
    NS_naive)  printf '%s\n' "${MECH_OFF[@]}" env.enable_deep_tilt_boot=False ;;
    JS)        printf '%s\n' "${MECH_OFF[@]}" env.enable_deep_tilt_boot=False \
                             env.right_arm_jointspace=True ;;
    # ---- Table II: reward ablation (NS_demo base, boot off) ----
    R_noaim)       printf '%s\n' "${MECH_ON[@]}" env.enable_deep_tilt_boot=False \
                                 env.weight_aim_precision=0.0 ;;
    R_noalign)     printf '%s\n' "${MECH_ON[@]}" env.enable_deep_tilt_boot=False \
                                 env.weight_align=0.0 ;;
    R_nointrot)    printf '%s\n' "${MECH_ON[@]}" env.enable_deep_tilt_boot=False \
                                 env.weight_introt=0.0 ;;
    R_notiltdelta) printf '%s\n' "${MECH_ON[@]}" env.enable_deep_tilt_boot=False \
                                 env.weight_tilt_delta=0.0 ;;
    *) echo "!! 알 수 없는 조건: $1" >&2; return 1 ;;
  esac
  # 전 조건 공통 (ablation 불변 통제)
  printf '%s\n' env.enable_demo_pose_reward=False env.receiver_control_mode=learned
}

[[ $# -ge 1 ]] || { echo "Usage: $0 <조건> [조건...]"; exit 1; }

echo "============================================================"
echo " RA-L 논문 매트릭스 (신 USD)  seed=$SEED  envs=$NUM_ENVS  iter=$MAX_ITER"
echo " GPU=${CUDA_VISIBLE_DEVICES:-?}  조건=[$*]"
echo "============================================================"

for c in "$@"; do
  mapfile -t OV < <(cond_args "$c")
  label="${c}_s${SEED}${SUFFIX}"
  echo
  echo ">>> [$label]  ($(date '+%m-%d %H:%M:%S'))"
  printf '     %s\n' "${OV[@]}"
  NOTE="RA-L 신USD 재생성: $c seed=$SEED (메커니즘 3플래그 단위 통제)" \
    "$HDGP_ROOT/train.sh" "$TASK" "$label" \
      --num_envs "$NUM_ENVS" --seed "$SEED" --max_iterations "$MAX_ITER" \
      "${OV[@]}"
  echo "<<< [$label] 완료 ($(date '+%m-%d %H:%M:%S'))"
done

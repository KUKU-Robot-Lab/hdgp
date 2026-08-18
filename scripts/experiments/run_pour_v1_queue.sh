#!/usr/bin/env bash
# 논문 A 실험 큐 — both/pour_v1 (양손 DG-5FS 물리 파지, 자산 a2)
#
# 정본은 docs/experiments/registry.md 다. 이 스크립트는 그 목록을 실행만 한다.
# 조건 정의를 여기와 레지스트리 두 곳에 두지 않으려면, 값을 바꿀 때 **양쪽을 같이** 고칠 것.
#
# 사용:
#   ./scripts/experiments/run_pour_v1_queue.sh E1                 # 성립 게이트 (frozen 1런)
#   CUDA_VISIBLE_DEVICES=0 ./...run_pour_v1_queue.sh E2 Full NSdemo
#   CUDA_VISIBLE_DEVICES=1 ./...run_pour_v1_queue.sh E2 NSnaive JS
#   ./...run_pour_v1_queue.sh E3                                  # reward ablation 4종 순차
#   NO_EVAL=1 ./...run_pour_v1_queue.sh E2 Full                   # eval 생략
#
# 환경변수: SEED(42) NUM_ENVS(2048) MAX_ITER(6500) ASSET(a2) PAPER(A) NO_EVAL RESUME_FROM
#
# 각 런은 학습 → 결정론 eval → STATUS 기록 순으로 진행한다.
# 런 이름은 <PAPER>-<EXP>-<COND>-<ASSET>-s<SEED> 이고, train.py 가 이 라벨의 자산 태그와
# 실제 로봇 USD 를 대조해 불일치면 학습을 거부한다(2026-08-17 사고 재발 방지).
set -euo pipefail

HDGP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TASK="open-tesol_b_pour_v1-lstm"
LOG_ROOT="$HDGP_ROOT/log/rl_games/open-tesol/both/pour-v1"
EVAL_PY="$HDGP_ROOT/scripts/reinforcement_learning/rl_games/eval_pour_envs.py"
EVAL_OUT="$HDGP_ROOT/docs/eval/pour_v1"
ISAAC="${ISAACLAB_ROOT:-$(cd "$HDGP_ROOT/.." && pwd)/IsaacLab}/isaaclab.sh"

SEED="${SEED:-42}"
NUM_ENVS="${NUM_ENVS:-2048}"
MAX_ITER="${MAX_ITER:-6500}"
ASSET="${ASSET:-a2}"
PAPER="${PAPER:-A}"

# 여자유도 해소 메커니즘 — 3플래그를 **한 단위로** 켜고 끈다.
#   `nullspace_baseline` 단독은 죽은 코드다: pour_orient_release=True 분기가 baseline 과
#   무관하게 demo 자세를 강제한다. 논문 본문도 이 단위로 서술한다.
MECH_ON=(env.nullspace_baseline=demo  env.pour_orient_release=True  env.pour_bfull_nullspace=True)
MECH_OFF=(env.nullspace_baseline=robot_start env.pour_orient_release=False env.pour_bfull_nullspace=False)

# 조건 → hydra override. 레지스트리의 표와 1:1 대응.
cond_args() {
  case "$1" in
    # ---- E1: 성립 게이트 ----
    # ★2026-08-18 E1 을 frozen → learned 로 바꿨다. pour_v1 은 왼손이 컵을 **들고** 있어
    #   receiver 가 pour_sensor 대비 7.4cm 높다(z 0.291→0.365). source 도 0.367 이라 두 컵이
    #   같은 높이에서 시작하는데, 왼팔을 고정하면 그 격차를 해소할 수단이 없다.
    #   실측: A-E1-frozen 은 2442 epoch 동안 bead_in_target 0.000 평탄 → aborted.
    #   (구 pour_sensor 는 receiver 가 테이블 위라 frozen 으로도 성립했다.)
    learned)      printf '%s\n' "${MECH_ON[@]}" env.enable_deep_tilt_boot=False ;;
    frozen)       printf '%s\n' "${MECH_ON[@]}" env.receiver_control_mode=frozen \
                                env.enable_deep_tilt_boot=False env.enable_demo_pose_reward=False ;;
    # ---- E2: 핵심 조건 (왼팔 제어 해제) ----
    Full)         printf '%s\n' "${MECH_ON[@]}"  env.enable_deep_tilt_boot=True ;;
    NSdemo)       printf '%s\n' "${MECH_ON[@]}"  env.enable_deep_tilt_boot=False ;;
    NSnaive)      printf '%s\n' "${MECH_OFF[@]}" env.enable_deep_tilt_boot=False ;;
    JS)           printf '%s\n' "${MECH_OFF[@]}" env.enable_deep_tilt_boot=False \
                                env.right_arm_jointspace=True ;;
    # ---- E3: reward ablation (NSdemo base, boot OFF) ----
    Rnoaim)       printf '%s\n' "${MECH_ON[@]}" env.enable_deep_tilt_boot=False \
                                env.weight_aim_precision=0.0 ;;
    Rnoalign)     printf '%s\n' "${MECH_ON[@]}" env.enable_deep_tilt_boot=False \
                                env.weight_align=0.0 ;;
    Rnointrot)    printf '%s\n' "${MECH_ON[@]}" env.enable_deep_tilt_boot=False \
                                env.weight_introt=0.0 ;;
    Rnotiltdelta) printf '%s\n' "${MECH_ON[@]}" env.enable_deep_tilt_boot=False \
                                env.weight_tilt_delta=0.0 ;;
    *) echo "!! 알 수 없는 조건: $1" >&2; return 1 ;;
  esac
  # E1(frozen) 외 전 조건은 왼팔 제어를 학습한다 + demo 보상 off (제안 방식 기본)
  case "$1" in
    frozen) : ;;
    *) printf '%s\n' env.receiver_control_mode=learned env.enable_demo_pose_reward=False ;;
  esac
}

exp_conds() {
  case "$1" in
    E1) echo "learned" ;;
    E2) echo "Full NSdemo NSnaive JS" ;;
    E3) echo "Rnoaim Rnoalign Rnointrot Rnotiltdelta" ;;
    *) echo "!! 알 수 없는 실험: $1 (E1|E2|E3)" >&2; return 1 ;;
  esac
}

EXP="${1:?'Usage: run_pour_v1_queue.sh <E1|E2|E3> [조건...]'}"
shift || true
CONDS=("$@")
if [[ ${#CONDS[@]} -eq 0 ]]; then read -r -a CONDS <<< "$(exp_conds "$EXP")"; fi

mkdir -p "$EVAL_OUT"
echo "============================================================"
echo " 논문 $PAPER / 실험 $EXP  (자산 $ASSET · seed $SEED · envs $NUM_ENVS · iter $MAX_ITER)"
echo " GPU=${CUDA_VISIBLE_DEVICES:-미지정}  조건=[${CONDS[*]}]"
echo " 정본: docs/experiments/registry.md"
echo "============================================================"

for cond in "${CONDS[@]}"; do
  LABEL="${PAPER}-${EXP}-${cond}-${ASSET}-s${SEED}"
  RUN_DIR="$LOG_ROOT/$LABEL"
  if [[ -n "${RESUME_FROM:-}" && "$cond" != "$RESUME_FROM" ]]; then
    # RESUME_FROM 이후부터 실행 (앞선 조건은 건너뛴다)
    if [[ -z "${_resumed:-}" ]]; then echo ">>> 건너뜀: $cond"; continue; fi
  fi
  _resumed=1

  mapfile -t OV < <(cond_args "$cond")
  echo
  echo ">>> [$LABEL]  시작 $(date '+%m-%d %H:%M:%S')"
  printf '     %s\n' "${OV[@]}"

  NOTE="registry $EXP/$cond — 양손 물리 파지 pour (자산 $ASSET)" \
    "$HDGP_ROOT/train.sh" "$TASK" "$LABEL" \
      --num_envs "$NUM_ENVS" --seed "$SEED" --max_iterations "$MAX_ITER" \
      "${OV[@]}"

  # STATUS: 러너가 정상 종료를 기록한다. 붕괴·중단 판정은 사람이 고친다(레지스트리 규칙).
  [[ -d "$RUN_DIR" ]] && echo "done" > "$RUN_DIR/STATUS"
  echo "<<< [$LABEL] 학습 종료 $(date '+%m-%d %H:%M:%S')"

  if [[ -n "${NO_EVAL:-}" ]]; then echo "    (NO_EVAL → eval 생략)"; continue; fi

  CKPT="$(ls -t "$RUN_DIR"/nn/*ep_*.pth 2>/dev/null | head -1 || true)"
  if [[ -z "$CKPT" ]]; then echo "    !! 체크포인트 없음 → eval 생략"; continue; fi
  echo "    eval: $(basename "$CKPT")"
  # eval 은 학습과 **같은 override** 로 돌려야 조건이 보존된다(빠뜨리면 다른 환경을 재는 셈).
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$ISAAC" -p "$EVAL_PY" \
    --task "$TASK" --checkpoint "$CKPT" \
    --num_envs 1024 --seed 100 --eval_steps 1200 --headless \
    --eval_out "$EVAL_OUT/eval_${LABEL}.md" \
    "${OV[@]}" \
    2>&1 | grep -E "^\[EVAL\] 총|!!|Error" | tail -3 || true
done

echo
echo "===== $EXP 완료 ($(date '+%m-%d %H:%M:%S')) ====="
echo "  런:   $LOG_ROOT/${PAPER}-${EXP}-*"
echo "  eval: $EVAL_OUT/eval_${PAPER}-${EXP}-*"
echo "  → 레지스트리 상태를 갱신할 것: docs/experiments/registry.md"

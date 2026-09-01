#!/usr/bin/env bash
# grasp_s2r 학습 기동 — **s2r 정합 세팅**을 한 곳에 모은다.
#
# 배경 (2026-09-01)
# -----------------
# D3(`s2r_d3_liftonly_fresh_v2`)가 FRESH 20k 로 8종 전수 성공(0.774~0.949)해
# 그 세팅이 소스 기본값이 됐다(커밋 544c88b). 하지만 D3 는 **가장 쉬운 조건**에서
# 잰 수치다 — 공칭 물리 · 정확한 물체 pose · ADR OFF.
#
# 이 스크립트는 그 위에 **실기 전이에 필요한 것들**을 얹는다. 프리셋 하나로 묶은
# 이유는, 이것들이 낱개로 켜면 의미가 없거나(예: floor 없는 enclosure) 서로를
# 전제하기 때문이다(예: r2s 게인 ↔ 중력보상).
#
#   D3      기본값 그대로 = 재현·대조군
#   LEAK    goal_rel 누수만 차단          ← D3 가 누수에 의존한 정도를 잰다
#   PERC    누수 차단 + 파지 후 강체부착   ← 실기 지각 조건
#   S2R     PERC + r2s 정합 게인 + ADR 물리/노이즈 랜덤화  ← ★본 목표
#
# 사용
# ----
#   ./scripts/experiments/grasp_s2r_train.sh S2R e2_s2r_full
#   NUM_ENVS=1024 SEED=43 ./scripts/experiments/grasp_s2r_train.sh PERC e1_perc
#   DRY=1 ./scripts/experiments/grasp_s2r_train.sh S2R x    # 명령만 찍고 종료
#
# 환경변수: NUM_ENVS(2048) SEED(42) MAX_ITER(20000) DRY(0)
#
# ★NUM_ENVS 는 **1024 의 배수**여야 한다. rl_games 가
#   `batch_size(= num_envs × horizon_length 16) % minibatch_size(16384) == 0` 을
#   assert 하기 때문이다(`rl_games_ppo_lstm_cfg.yaml:95-96`). 512 로 주면 부팅 뒤
#   `AssertionError` 로 죽는다 — 09.01 스모크에서 밟았다.
#
# ★★FRESH 로만 돌린다 — warmstart 금지.
#   LEAK/PERC/S2R 은 **obs 의 의미**를 바꾼다(차원은 155 그대로). D3 정책은
#   ①clean `goal_rel` 로 물체 참값을 복원하고 ②래치 뒤에도 실시간 물체 위치를
#   보는 두 채널에 의존해 학습됐다. 그 둘을 끊으면 인계가 독이 된다
#   (메모리 `fresh-vs-warmstart-lstm-rule`: LSTM 은 인계받은 행동을 답습한다 —
#   D1 이 D3 대비 success 0.185 vs 0.604 로 그걸 실증했다).
set -euo pipefail

PRESET="${1:?'Usage: grasp_s2r_train.sh <D3|LEAK|PERC|S2R> <run_label> [extra args...]'}"
LABEL="${2:?'Usage: grasp_s2r_train.sh <preset> <run_label> [extra args...]'}"
shift 2

HDGP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TASK="open-sens_r_grasp_s2r-lstm"
NUM_ENVS="${NUM_ENVS:-2048}"
SEED="${SEED:-42}"
MAX_ITER="${MAX_ITER:-20000}"

# ---- 공통: D3 기본값은 **소스에 있다**. 여기서 다시 적지 않는다 -------------------
# ★중복해서 적으면 소스 기본값이 바뀔 때 조용히 갈린다. D3 세팅 전체는
#   `grasp_s2r_env_cfg.py` 의 `test_d3_default_set_is_intact` 가 잠근다.
# hydra 오버라이드(`env.*`)만 여기 모은다 — CLI 플래그는 train.sh 에 따로 넘긴다.
ARGS=()

# ---- 프리셋 ----------------------------------------------------------------------
USE_REAL_GAINS=0
case "$PRESET" in
  D3)
    : ;;                                  # 기본값 그대로 — 대조군
  LEAK)
    ARGS+=(env.obs_object_noise_coherent=true) ;;
  PERC)
    ARGS+=(env.obs_object_noise_coherent=true
           env.obs_object_rigid_after_latch=true) ;;
  S2R)
    USE_REAL_GAINS=1
    ARGS+=(env.obs_object_noise_coherent=true
           env.obs_object_rigid_after_latch=true
           # --- ADR: 과제 난이도가 아니라 **실기 불확실성**을 키운다 ---------------
           env.enable_adr=true
           # 관절 상태 노이즈 — 실기 bag 실측(6 bag · /joint_states · CDR 직접 파싱).
           #   운동구간 σ 중앙값: pos **9e-4 rad** · vel **4.5e-2 rad/s**
           #   (양자화: 팔 pos 3.815e-4 rad = 100/2^18 · 손 pos 1.745e-3 rad = 0.1°)
           # ★★base 도 함께 내린다. 소스 기본 `obs_noise_qpos=0.01` 은 **실측의 10배**다
           #   (0.01 rad = 0.57° = 26 LSB). 정밀 파지 트랙에서 매 관절에 실기의 10배
           #   잡음을 넣고 있었다는 뜻이고, 그대로 두면 ADR 종점(0.003)이 base 보다
           #   작아져 `_assert_adr_monotonic` 가드에 걸린다(= 승급할수록 쉬워지는 축).
           #   ⚠D3 기본값은 건드리지 않는다 — 여기 프리셋에서만 내린다. 노이즈를
           #     **줄이는** 방향이라 성능은 오르면 올랐지 떨어지지 않는다.
           env.obs_noise_qpos=0.001
           env.adr_obs_noise_qpos_max=0.003
           # qvel 은 base 0.05 가 실측(4.5e-2)과 거의 일치해 그대로 둔다.
           env.adr_obs_noise_qvel_max=0.12
           # 컵 질량 — 붓기로 가면 내용물이 질량으로 온다(빈 컵 ↔ 물 찬 컵)
           env.adr_mass_scale_max=[0.5,2.0]
           # 관절 PD 게인 — 벤더 고정값의 불확실성
           #   ⚠하한 0.8 은 손 파지력도 20% 깎는다(손 PD 는 이미 토크 포화 21%).
           #     파지가 무너지면 0.9 로 좁힌다.
           env.adr_joint_gain_scale_max=[0.8,1.2]
           # 마찰 — ★ADR 축이 **아니다**. 재질 버킷이 1회 샘플링이라 런타임 확장이
           #   무증상 no-op 다. cfg 고정 범위로만 연다(절대값, 배율 아님).
           env.object_friction_range=[0.5,1.5]
           # 스폰 — m1 실측상 ±2→±5cm 는 −4.4%p 뿐이라 가장 안전한 축
           env.adr_spawn_range_max=0.05) ;;
  *)
    echo "알 수 없는 프리셋: $PRESET (D3|LEAK|PERC|S2R)" >&2; exit 2 ;;
esac

# ---- r2s 정합 게인 ---------------------------------------------------------------
# ★★환경변수로만 켜진다 — `robot_profiles.py` 가 **import 시점**에 읽기 때문이다.
#   cfg 의 `use_real_gains` 는 그 의도를 `params/env.yaml` dump 에 남기고,
#   `finalize_after_overrides` 가 실제 조립 결과와 대조해 어긋나면 부팅에서 죽인다.
#   (환경변수는 dump 에 안 남아서, 이게 없으면 재생 시 조용히 KUKA 게인으로 돈다 —
#    m1_final 이 죽은 것과 같은 계열의 실패다.)
# ★게인 실측: sim2real/docs/R2S_FRAMEWORK.md — 오버슈트 재현오차 0.429 → 0.084.
#   kp 70/70/70/60/10/10/10 · kd 7.053/4.182/7.804/6.531/2.236/0.580/0.242
# ⚠이 게인은 **중력보상 전제**다. sim 로봇 중력은 계속 꺼둔다(`disable_gravity=True`) —
#   실기가 중력보상을 켠 채 돌므로 켜면 중력을 두 번 세는 것이 된다.
#   09.01 probe 실측으로 확인: 중력 ON 이면 손목 처짐이 실기 무보상(12.76°) 쪽으로 간다.
if [ "$USE_REAL_GAINS" = "1" ]; then
  export HDGP_S2R_REAL_GAINS=1
else
  unset HDGP_S2R_REAL_GAINS || true
fi

echo "============================================================"
echo " grasp_s2r 학습 — 프리셋 $PRESET"
echo "  라벨      : $LABEL"
echo "  env/seed  : $NUM_ENVS / $SEED   (max_iter $MAX_ITER)"
echo "  r2s 게인  : $([ "$USE_REAL_GAINS" = 1 ] && echo 'ON (실기 정합)' || echo 'OFF (KUKA 기본)')"
echo "  ★FRESH 전용 — --checkpoint 를 붙이지 말 것"
echo "============================================================"
if [ ${#ARGS[@]} -gt 0 ]; then printf '  %s\n' "${ARGS[@]}"; else echo "  (오버라이드 없음 — D3 기본값)"; fi
echo "------------------------------------------------------------"

if [ "${DRY:-0}" = "1" ]; then
  echo "DRY=1 — 실행하지 않는다."
  exit 0
fi

# ---- 부팅 게이트 (기동 뒤 로그에서 반드시 확인할 것) -----------------------------
#   1) `[grasp_s2r] profile=... action=21 obs=155 state=193`
#   2) S2R 이면 `[grasp_s2r][ADR]` 승급 로그에 mass/gain 범위가 함께 찍힌다
#   3) 게인 분기 불일치면 `[grasp_s2r] 게인 분기 불일치` 로 **즉시 죽는다**
#   4) `done/abnormal` 0 · `contact/force_max_postlatch` < 150 N
cd "$HDGP_ROOT"
NOTE="${NOTE:-grasp_s2r $PRESET 프리셋 (FRESH · r2s게인=$USE_REAL_GAINS)}" \
  exec ./train.sh "$TASK" "$LABEL" \
       --num_envs "$NUM_ENVS" --seed "$SEED" --max_iterations "$MAX_ITER" \
       "$@" ${ARGS[@]+"${ARGS[@]}"}

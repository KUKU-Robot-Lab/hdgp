#!/usr/bin/env bash
# [both/pour_v1] 좌/우 warm 뱅크 재수집 파이프라인 (grasp_v1 재학습 완료 후 실행).
#
# 왜 스크립트인가 (2026-08-18)
# ---------------------------
# 수동으로 돌리면 매번 같은 곳에서 넘어진다:
#   · 수집기는 **목적지 파일 존재**를 완료 신호로 쓴다 → 구 뱅크를 안 지우면
#     1스텝 만에 "저장 완료"를 찍고 끝난다(구 캐시를 그대로 잰 셈).
#   · 프리셋 체크포인트는 학습 런이 늘면 낡는다(lstm_test1 → lstm_test2).
#   · source(우)는 **비드를 채운 채** 파지해야 하고 receiver(좌)는 빈 컵이다.
# 이 순서를 고정하고, 끝나면 겹침을 바로 실측한다.
#
# 사용:
#   ./scripts/warm_states/refresh_bimanual_warm_banks.sh                 # 최신 ckpt 자동
#   R_CKPT=/abs/r.pth L_CKPT=/abs/l.pth ./scripts/warm_states/refresh_bimanual_warm_banks.sh
#   PULL_FROM_SERVER=1 ./scripts/warm_states/refresh_bimanual_warm_banks.sh
#
# 환경변수: NUM_ENVS(256) TARGET(2048) PULL_FROM_SERVER R_CKPT L_CKPT
set -euo pipefail

HDGP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$HDGP_ROOT"

NUM_ENVS="${NUM_ENVS:-256}"
TARGET="${TARGET:-2048}"
DATA="$HDGP_ROOT/data"
BAK="$DATA/warm_bak_$(date +%Y%m%d_%H%M%S)"
COLLECT="scripts/warm_states/collect_grasp_v1_warm_states.py"

echo "============================================================"
echo " both/pour_v1 warm 뱅크 재수집   envs=$NUM_ENVS target=$TARGET"
echo "============================================================"

# ---- 0. (선택) 서버에서 최신 체크포인트 받아오기 -------------------------
if [[ -n "${PULL_FROM_SERVER:-}" ]]; then
  echo ">>> 서버에서 최신 grasp_v1 체크포인트를 받는다"
  for side in right left; do
    remote="$(ssh server "ls -t ~/rl_ws/hdgp/log/rl_games/open-tesol/$side/grasp-v1/*/nn/*ep_*.pth 2>/dev/null | grep -v '__' | head -1")"
    if [[ -z "$remote" ]]; then echo "    !! $side 체크포인트 없음"; continue; fi
    run_dir="$(basename "$(dirname "$(dirname "$remote")")")"
    dest="$HDGP_ROOT/log/rl_games/open-tesol/$side/grasp-v1/$run_dir/nn"
    mkdir -p "$dest"
    echo "    $side: $run_dir/$(basename "$remote")"
    scp -q "server:$remote" "$dest/"
  done
fi

# ---- 1. 기존 뱅크 백업 후 제거 ------------------------------------------
# 수집기는 목적지 존재를 완료 신호로 쓰므로 **반드시 비워야** 한다.
mkdir -p "$BAK"
for f in grasp_warm_tesollo_right.hdf5 grasp_warm_tesollo_left.hdf5; do
  if [[ -f "$DATA/$f" ]]; then
    mv "$DATA/$f" "$BAK/$f"
    echo ">>> 백업: $f → $BAK/"
  fi
done

# ---- 2. 수집 (우: 비드 채움 / 좌: 빈 컵) ---------------------------------
r_args=(--robot tesollo_right --with_beads --num_envs "$NUM_ENVS" --target_count "$TARGET")
l_args=(--robot tesollo_left               --num_envs "$NUM_ENVS" --target_count "$TARGET")
if [[ -n "${R_CKPT:-}" ]]; then r_args+=(--checkpoint "$R_CKPT"); else r_args+=(--latest); fi
if [[ -n "${L_CKPT:-}" ]]; then l_args+=(--checkpoint "$L_CKPT"); else l_args+=(--latest); fi

echo; echo ">>> [1/2] source(우팔) 수집 — 비드 채운 상태"
python3 "$COLLECT" "${r_args[@]}"
echo; echo ">>> [2/2] receiver(좌팔) 수집 — 빈 컵"
python3 "$COLLECT" "${l_args[@]}"

# ---- 3. 겹침 실측 --------------------------------------------------------
echo; echo ">>> [검증 1/2] 좌/우 페어 겹침"
python3 scripts/probes/verify_bimanual_cup_overlap.py

# ---- 4. warm → pour 제어 인계 검사 (Isaac 불필요, 즉시) -------------------
# 뱅크의 palm pose 가 pour workspace 밖이면 리셋이 클램프해 palm 목표가 실제 팔
# 자세와 분리된다. 그 상태로 학습을 걸면 제어가 어긋난 채 시작한다.
echo; echo ">>> [검증 2/2] warm → pour 제어 인계"
if ! python3 scripts/probes/verify_warm_to_pour_handoff.py; then
  echo
  echo "!! 인계 검사 실패 — 학습으로 넘어가지 말 것. 위 항목을 먼저 해결한다."
  exit 1
fi

cat <<'EOT'

------------------------------------------------------------
정적 검증 통과. 다음 단계
  1) 겹침이 크면 grasp cfg `object_spawn_y_center` 를 더 벌리고 재수집한다
     (정책 재학습은 불필요 — 스폰만 옮겨도 평균이 따라 이동함을 실측했다).
  2) E0-3 물리 게이트 (Isaac 필요, 수 분):
       ../IsaacLab/isaaclab.sh -p scripts/probes/probe_bimanual_warm_coexist.py \
         --num_envs 128 --steps 300 --headless
  3) 통과하면 학습:
       ./scripts/experiments/run_pour_v1_queue.sh E1
------------------------------------------------------------
EOT

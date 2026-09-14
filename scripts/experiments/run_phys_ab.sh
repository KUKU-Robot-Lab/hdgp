#!/usr/bin/env bash
# 09.10 솔버 설정 A/B — **같은 체크포인트**를 물리 설정만 바꿔 재생한다.
#   정책·액션이 동일하므로 관절 한계 이탈의 차이는 **물리 설정만의 몫**이다.
#   학습을 4~6시간 기다리지 않고 판정하기 위한 것. ★set -u 금지(isaacsim setup 함정).
#
#   조건당 시드 2개를 강제한다 — 09.08 에 조건당 1런으로 편차를 효과로 오판한 전례가 있고,
#   `fj_joint_limit_viol.py` 가 baseline 2개 이상을 요구한다.
HDGP="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # scripts/experiments/ → hdgp 루트
cd "$HDGP" || exit 1
CKPT="${CKPT:?CKPT 필요}"
OUT="${OUT:-/tmp/claude-1000/phys_ab}"
mkdir -p "$OUT"
export PYTHONPATH="$HDGP/vendor/rl_games_sapg:$HDGP/source/openarm:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

# ★IsaacLab 권고는 vel_iters 상향과 이 플래그를 **함께** 켜라는 것이라 B 조건에 묶는다.
#   `--phys` 에는 이 키가 없어 hydra 로 넘긴다(sim cfg 는 finalize 가 재조립하지 않는다).
EXTF_ON="env.sim.physx.enable_external_forces_every_iteration=true"

run () {                       # run <이름> <--phys 문자열> <시드> [hydra...]
  local name="$1" phys="$2" seed="$3"; shift 3
  local npz="$OUT/${name}_s${seed}.npz"
  [ -f "$npz" ] && { echo "[건너뜀] $npz 이미 있음"; return 0; }
  echo "=== $name seed=$seed · phys=$phys ==="
  ../IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
    --task open-sens_r_grasp_fj-play-lstm-sapg --checkpoint "$CKPT" \
    --num_envs 12 --headless --seed "$seed" \
    --video --video_length 400 \
    --phys "$phys" --dump_hand_q "$npz" "$@" \
    > "$OUT/${name}_s${seed}.log" 2>&1
  if [ -f "$npz" ]; then echo "  ✓ $(basename "$npz")"; else
    echo "  ★실패 — 로그 꼬리:"; tail -5 "$OUT/${name}_s${seed}.log"; fi
}

for s in 1 2; do
  run baseline "pos_iters=8,vel_iters=0,max_depen=1000"  "$s"
  run ab       "pos_iters=8,vel_iters=2,max_depen=10"    "$s" "$EXTF_ON"
  run abc      "pos_iters=32,vel_iters=2,max_depen=10"   "$s" "$EXTF_ON"
done
echo "=== 완료 ==="; ls -la "$OUT"/*.npz 2>/dev/null

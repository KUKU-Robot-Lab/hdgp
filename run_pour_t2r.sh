#!/usr/bin/env bash
# pour_fabric(text2reward) 학습 기동 — 서버/로컬 공용.
#   ./run_pour_t2r.sh <label> <iter_dir> [--num_envs N] [추가 train 인자...]
#   예) CUDA_VISIBLE_DEVICES=1 ./run_pour_t2r.sh t2r_i00 reward_gen/pour_bi/iter_00 --num_envs 256
# ★reward_code_path 는 절대경로로 넘긴다(hydra 는 env.* 필드를 덮고 env 가 런타임에 읽는다).
set -euo pipefail
LABEL="${1:?label}"; ITER="${2:?iter_dir}"; shift 2
HDGP_ROOT="$(cd "$(dirname "$0")" && pwd)"
CODE="$(cd "$HDGP_ROOT" && realpath "$ITER/compute_reward.py")"
[ -f "$CODE" ] || { echo "no $CODE"; exit 1; }
python3 - "$ITER" <<'EOF'
import json, sys, pathlib
v = json.load(open(pathlib.Path(sys.argv[1]) / "validation.json"))
assert v["ok"], f"validation FAIL: {v['errors']}"
EOF
echo "reward: $CODE"
# minibatch = num_envs×horizon(64)/8 (최소 8192) — actor·central_value 둘 다 같은 값이어야 한다.
#   09.13 사용자 지시: 4096 env 로 확대(분산 축소). 8192 로 두면 32 minibatch 가 되어 느리다.
NE=""; prev=""; for a in "$@"; do [ "$prev" = "--num_envs" ] && NE="$a"; prev="$a"; done
MB=8192; if [ -n "$NE" ]; then MB=$(( NE * 64 / 8 )); [ "$MB" -lt 8192 ] && MB=8192; fi
echo "num_envs=${NE:-cfg} minibatch=$MB"
NOTE="${NOTE:-t2r $ITER}" exec "$HDGP_ROOT/train.sh" open-short_b_pour_fab "$LABEL" \
    --headless "env.reward_code_path=$CODE" \
    "agent.params.config.minibatch_size=$MB" \
    "agent.params.config.central_value_config.minibatch_size=$MB" "$@"

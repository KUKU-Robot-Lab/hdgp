#!/usr/bin/env bash
# pour_fabric_mimic(RH56F1, text2reward) 학습 기동 — vision-3090/로컬 공용. 대응 트랙 tasks/pour_fabric_mimic.
#   ./scripts/experiments/pour_t2r_rh_train.sh <label> <iter_dir> [--num_envs N] [추가 train 인자...]
#   필수: iter_dir/validation.json ok · reward_code_path 는 절대경로로 넘긴다. 로그는 호출측이 log/pour_fabric_mimic/ 에.
set -euo pipefail
LABEL="${1:?label}"; ITER="${2:?iter_dir}"; shift 2
HDGP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CODE="$(cd "$HDGP_ROOT" && realpath "$ITER/compute_reward.py")"
[ -f "$CODE" ] || { echo "no $CODE"; exit 1; }
python3 - "$HDGP_ROOT/$ITER" <<'PY'
import json, sys, pathlib
v = json.load(open(pathlib.Path(sys.argv[1]) / "validation.json"))
assert v["ok"], f"validation FAIL: {v['errors']}"
PY
NE=""; prev=""; for a in "$@"; do [ "$prev" = "--num_envs" ] && NE="$a"; prev="$a"; done
MB=8192; if [ -n "$NE" ]; then MB=$(( NE * 64 / 8 )); [ "$MB" -lt 8192 ] && MB=8192; fi
echo "reward: $CODE  num_envs=${NE:-cfg} minibatch=$MB"
NOTE="${NOTE:-t2r_rh $ITER}" exec "$HDGP_ROOT/train.sh" open-rh_b_pour_fab_mimic "$LABEL" \
    --headless "env.reward_code_path=$CODE" \
    "agent.params.config.minibatch_size=$MB" \
    "agent.params.config.central_value_config.minibatch_size=$MB" "$@"

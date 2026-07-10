#!/usr/bin/env bash
set -euo pipefail

# Phase D KPI matrix runner (4 representative cases).
# Uses report_phase32_success_metrics.py to extract success/contact/penalty KPIs.

ISAACLAB_ROOT="${ISAACLAB_ROOT:-/home/user/rl_ws/IsaacLab}"
TASK="${TASK:-5g_grasp_right-v2}"
NUM_ENVS="${NUM_ENVS:-128}"
STEPS="${STEPS:-512}"
HEADLESS="${HEADLESS:-1}"
ENABLE_CUP="${ENABLE_CUP:-true}"
ENABLE_PRIMITIVES="${ENABLE_PRIMITIVES:-false}"
OUT_DIR="${OUT_DIR:-/tmp/phaseD_kpi_reports}"

mkdir -p "${OUT_DIR}"

if [[ -z "${TERM:-}" || "${TERM}" == "dumb" ]]; then
  export TERM=xterm
fi

if [[ "${HEADLESS}" == "1" ]]; then
  HEADLESS_ARG="--headless"
else
  HEADLESS_ARG=""
fi

echo "[PhaseD-KPI] task=${TASK} num_envs=${NUM_ENVS} steps=${STEPS} out_dir=${OUT_DIR}"
echo "[PhaseD-KPI] cup=${ENABLE_CUP} primitives=${ENABLE_PRIMITIVES}"

# 4 representative integration cases
cases=(
  "A0B0C0 false false false"
  "A1B1C1 true  true  true"
  "A1B0C1 true  false true"
  "A0B1C1 false true  true"
)

cd "${ISAACLAB_ROOT}"

for row in "${cases[@]}"; do
  read -r NAME A B C <<< "${row}"
  echo "==== [PhaseD-KPI:${NAME}] start ===="
  ./isaaclab.sh -p ../hdgp/scripts/reports/report_phase32_success_metrics.py \
    --task "${TASK}" \
    ${HEADLESS_ARG} \
    --num_envs "${NUM_ENVS}" \
    --steps "${STEPS}" \
    --output "${OUT_DIR}/${NAME}.json" \
    env.enable_cup="${ENABLE_CUP}" \
    env.enable_primitives="${ENABLE_PRIMITIVES}" \
    env.use_reference_replay="${A}" \
    env.use_tip_contact_gate="${B}" \
    env.use_object_pc_feat="${C}"
  echo "==== [PhaseD-KPI:${NAME}] done ===="
done

python3 ../hdgp/scripts/reports/summarize_phase_d_kpi.py --input_dir "${OUT_DIR}" --output "${OUT_DIR}/summary.json"
echo "[PhaseD-KPI] summary: ${OUT_DIR}/summary.json"

#!/usr/bin/env bash
# grasp_s2r(g1) 궤적을 컵 소환 Y 3지점에서 뽑는다: 중심 −5cm / 중심 / 중심 +5cm.
#
# 위치마다 Isaac 을 새로 띄운다. 소환 중심은 프로필 상수라 프로세스 안에서 바꾸면
# 이미 조립된 cfg·부팅 가드와 어긋나기 때문이다 — 기동 비용(위치당 ~2분)을 내고
# "한 파일 = 한 위치, 라벨이 참" 을 산다.
#
# 사용:
#   ./run_grasp_s2r_traj_grid.sh [출력디렉터리] [체크포인트]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HDGP="$(cd "${HERE}/../.." && pwd)"
WS="$(cd "${HDGP}/.." && pwd)"
# ★`python` 이 아니라 isaaclab.sh -p 로 띄운다 — Isaac Sim 의 kit python 이 아니면
#   omni/isaaclab 이 없다(맨 python 은 `command not found` 로 조용히 아무것도 안 남긴다).
ISAACLAB_ROOT="${ISAACLAB_ROOT:-${WS}/IsaacLab}"

OUT_DIR="${1:-${HDGP}/log/grasp_traj/g1}"
CKPT="${2:-${WS}/sim2real/logs/policy/right_g1/nn/g1_ep17000.pth}"

# 프로필 `tesollo_right.object_spawn_center` = (0.362, −0.160). X 는 고정.
SPAWN_X=0.362
CENTER_Y=-0.160
OFFSETS=(-0.05 0.00 0.05)
LABELS=(ym05 y00 yp05)

if [[ ! -f "${CKPT}" ]]; then
  echo "체크포인트가 없다: ${CKPT}" >&2
  exit 1
fi
mkdir -p "${OUT_DIR}"

for i in "${!OFFSETS[@]}"; do
  Y=$(python3 -c "print(f'{${CENTER_Y} + (${OFFSETS[$i]}):.4f}')")
  TAG="${LABELS[$i]}"
  echo "=============================================================="
  echo "[GRID] ${TAG}: 컵 소환 (${SPAWN_X}, ${Y})"
  echo "=============================================================="
  "${ISAACLAB_ROOT}/isaaclab.sh" -p "${HERE}/record_grasp_s2r_traj.py" \
    --checkpoint "${CKPT}" \
    --spawn_x "${SPAWN_X}" --spawn_y "${Y}" \
    --object_species cup_big_s100 \
    --record_out "${OUT_DIR}/g1_${TAG}.hdf5" \
    --num_envs 16 --video --headless
done

echo
echo "[GRID] 완료 → ${OUT_DIR}"
ls -la "${OUT_DIR}" "${OUT_DIR}/videos" 2>/dev/null || true

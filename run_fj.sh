#!/usr/bin/env bash
# grasp_fj 학습 런처 (서버·로컬 공용). ★set -u 금지(isaacsim setup 함정).
#
#   사용법: TASK=open-short_r_grasp_fj-lstm-sapg GPU=0 RUN=fj_x1 ./run_fj.sh
#
# ★TASK 를 **필수**로 둔다. 09.10 에 옛 런처가 `--task open-sens_...` 를 박아둔 채로
#   남아 있어, dg5f-m-short 로 바꾼 뒤에도 두 런이 **옛 로봇(open-sens)** 으로 24576 env
#   × 45 epoch 를 돌았다. 기본값이 있으면 자산을 바꿔도 런처가 조용히 옛 것을 고른다.
# ★학습 전 커밋 해시를 찍는다 — 서버는 git pull 로만 코드를 받는다(사용자 확정 09.10).
set -o pipefail
HDGP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HDGP" || exit 1

: "${TASK:?TASK 필요 — 예: open-short_r_grasp_fj-lstm-sapg (open-sens 는 옛 dg5f-m)}"
: "${RUN:?RUN 필요 — 런 폴더 이름(train.py 가 RUN_LABEL 로 읽는다)}"

if [ -d "$HDGP/../IsaacLab" ] && [ -z "$SERVER" ]; then
  LAUNCH=(../IsaacLab/isaaclab.sh -p)                       # 로컬
else
  source ~/miniforge3/etc/profile.d/conda.sh
  conda activate proj-hdgp-py311
  source /home/oem/isaacsim/5.1.0/setup_conda_env.sh
  LAUNCH=(python)                                           # 서버
fi
export PYTHONPATH="$HDGP/vendor/rl_games_sapg:$HDGP/source/openarm:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export RUN_LABEL="$RUN"

E="${ENVS:-24576}"
echo "[런처] task=$TASK · run=$RUN · env=$E · gpu=${GPU:-0} · seed=${SEED:-42}"
echo "[런처] git HEAD=$(git rev-parse --short HEAD) dirty=$(git status --short | grep -c '^ M')개"
echo "[런처] EXTRA=${EXTRA:-(없음)}"
# SAPG 블록 — yaml 기본 expl_coef_block_size 는 num_envs 8192 기준(2048=4블록)이라
# env 수를 바꾸면 블록 수가 조용히 달라진다. 무엇으로 도는지 로그에 남긴다.
BLK="${BLK:-4096}"   # 상류 고정: 24,576 ÷ 4,096 = 6블록
echo "[런처] SAPG expl_coef_block_size=$BLK -> $((E / BLK))블록 (나머지 $((E % BLK)))"
[ $((E % BLK)) -ne 0 ] && echo "[런처] 경고: env 수가 블록크기의 배수가 아니다 — SAPG 분할이 어긋난다"

exec "${LAUNCH[@]}" scripts/reinforcement_learning/rl_games/train.py \
  --task "$TASK" --headless \
  --num_envs "$E" --seed "${SEED:-42}" \
  ${EXTRA}

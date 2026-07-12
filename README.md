# hdgp — Hand Dexterous Grasping Pipeline

OpenArm(7 DOF) 양팔 기반 dexterous manipulation RL 학습 저장소.
Isaac Lab + Isaac Sim 5.1.0 위에서 동작하며, 팔 제어에 Geometric Fabrics(DEXTRAH 방식)를 쓴다.

지원하는 손(엔드이펙터) 3종:

| 코드 | 손 | DOF |
|---|---|---|
| `tesol` | Tesollo | 20 |
| `rh56f1` | Inspire RH56F1 | 6 |
| `grip` | 2지 그리퍼 | 1 |

태스크: grasp(파지→goal 운반) / pour(따르기) / lift / reach / approach,
그리고 vision student **distillation**.

> 실험 진행 규칙·분석 방법론은 [CLAUDE.md](CLAUDE.md) 를 따른다. **로그 수치 근거 없이 코드를 고치지 않는다.**

---

## 전제 조건

```
~/rl_ws/
├── IsaacLab/          Isaac Lab (Isaac Sim 5.1.0 연결)
├── hdgp/              본 저장소
│   └── source/FABRICS/    Geometric Fabrics (fabrics_sim) — 벤더링됨
└── repo/DEXTRAH/      DEXTRAH 원본 (설계 참고용, 직접 의존 없음)
```

FABRICS는 `hdgp/source/FABRICS/` 안에 들어있고 env 코드가 스스로 `sys.path`에 넣으므로
따로 설치·경로 설정할 것이 없다. (예전 README가 안내하던 `~/rl_ws/FABRICS/` 는 더 이상 없다.)

---

## 설치

### 1. openarm 패키지 경로

`IsaacLab/isaaclab.sh` 에 아래 블록이 있으면 된다(이미 적용되어 있다).

```bash
HDGP_CANDIDATE="${ISAACLAB_PATH}/../hdgp"
if [[ -d "${HDGP_CANDIDATE}/source/openarm" ]]; then
    export PYTHONPATH="${HDGP_CANDIDATE}/source/openarm:${PYTHONPATH}"
fi
```

`train.py` / `play.py` 는 이와 별개로 로컬 `source/openarm` 을 강제 주입하므로,
학습·재생만 한다면 이 설정 없이도 동작한다. 일반 python에서 `import openarm` 을 쓰려면:

```bash
cd ~/rl_ws/hdgp/source/openarm && pip install -e .
```

### 2. Isaac Sim python 의존 패키지

```bash
~/rl_ws/IsaacLab/_isaac_sim/python.sh -m pip install lxml urdfpy
```

**warp 다운그레이드 (필수).** Isaac Sim 5.1.0 기본 `warp 1.10` 은 `warp.sim` 이 제거되어
FABRICS와 호환되지 않는다.

```bash
~/rl_ws/IsaacLab/_isaac_sim/python.sh -m pip install "warp-lang==1.8.1"
```

**networkx 2.2 / urdfpy 의 Python 3.11 호환 패치.** `urdfpy` 가 요구하는 `networkx 2.2` 는
`from collections import Mapping` 등 구식 import를 써서 그대로는 뜨지 않는다.

```bash
PYTHON=~/rl_ws/IsaacLab/_isaac_sim/kit/python/bin/python3
NX_DIR=$($PYTHON -m pip show networkx | grep "Location:" | awk '{print $2}')
BASE="$NX_DIR/networkx"

sed -i "s|from collections import Mapping|from collections.abc import Mapping|g" "$BASE/classes/graph.py"
sed -i "s|from collections import Mapping|from collections.abc import Mapping|g" "$BASE/classes/coreviews.py"
sed -i "s|from collections import Mapping, Set, Iterable|from collections.abc import Mapping, Set, Iterable|g" "$BASE/classes/reportviews.py"
sed -i "s|from fractions import gcd|from math import gcd|g" "$BASE/algorithms/dag.py"
sed -i "/from collections import defaultdict, Mapping, Set/c\\from collections.abc import Mapping, Set\nfrom collections import defaultdict" "$BASE/algorithms/lowest_common_ancestors.py"
sed -i "s|(np.int, \"int\"), (np.int8, \"int\"),|(int, \"int\"), (np.int8, \"int\"),|g" "$BASE/readwrite/graphml.py"

URDFPY_DIR=$($PYTHON -m pip show urdfpy | grep "Location:" | awk '{print $2}')
sed -i "s|value = np.asanyarray(value).astype(np.float)|value = np.asanyarray(value).astype(float)|g" "$URDFPY_DIR/urdfpy/urdf.py"
```

---

## 태스크 이름

```
open-<손>_<팔>_<태스크>_<버전>[-접미사]
      tesol   r        grasp    v2      -lstm | -play | -play-lstm | -distill
      rh56f1  l        pour
      grip    b        lift / reach / approach
```

예: `open-tesol_r_grasp_v2-lstm`, `open-rh56f1_r_pour_v1`, `open-tesol_b_pour_sensor`

- `-lstm` : recurrent 정책 (현재 주력)
- `-play` : 재생용 소규모 env
- `-distill` : vision student 증류 (아래 참조)

전체 목록:

```bash
../IsaacLab/isaaclab.sh -p scripts/tools/list_envs.py
```

---

## 학습

`train.sh` 를 쓴다. 학습 시작 시 git diff·파라미터 스냅샷을 run 폴더에 자동 기록한다
(`test_history.md`) — 나중에 "이 실험이 뭘 바꾼 거였지"를 복원하는 유일한 수단이므로
`train.py` 를 직접 부르지 말 것.

```bash
cd ~/rl_ws/hdgp

./train.sh <task_id> <test_name> [추가 인자...]

# 예시
./train.sh open-tesol_r_grasp_v2-lstm lstm_test7 --num_envs 4096 --headless
NOTE="lift weight 5→0" ./train.sh open-tesol_r_pour_v5-lstm test9 --num_envs 2048 --headless
```

- `NOTE="설명"` : `test_history.md` 에 남길 메모
- `--checkpoint <path>` : 재개
- server(GPU 2장)에서는 `CUDA_VISIBLE_DEVICES=N` 지정 후 setsid 백그라운드로 띄운다

### 재생

```bash
../IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py \
    --task open-tesol_r_grasp_v2-play-lstm \
    --checkpoint log/rl_games/open-tesol/right/grasp-v2/lstm_test6/nn/<ckpt>.pth \
    --num_envs 16
```

### Distillation (teacher → vision student)

state 기반 teacher를 RealSense D435i RGB-D만 보는 student로 증류한다.
**torchrun 필수** — 자세한 사용법은 [scripts/distillation/README.md](scripts/distillation/README.md).

```bash
../IsaacLab/isaaclab.sh -p -m torch.distributed.run --standalone --nproc_per_node=1 \
    scripts/distillation/run_distillation.py \
    --task open-tesol_r_grasp_v2-distill --teacher <teacher.pth> \
    --label test1 --num_envs 256 --headless
```

---

## 로그

```
log/rl_games/<손>/<팔>/<태스크-버전>/<test_name>/
├── nn/            체크포인트
├── summaries/     TFEvents
└── test_history.md  학습 시작 시점의 git diff + 파라미터 스냅샷

log/distillation/<task>/<label>/     증류는 별도 트리
```

예: `log/rl_games/open-tesol/right/grasp-v2/lstm_test6/`

TFEvents는 tensorflow 없이 파싱한다:

```bash
python3 scripts/tools/parse_tfevents.py log/rl_games/open-tesol/right/grasp-v2/lstm_test6/summaries
```

---

## 디렉토리 구조

```
hdgp/
├── CLAUDE.md            실험 진행 규칙 (로그 우선, 증거 우선순위, 수정 규칙)
├── train.sh             학습 진입 래퍼 (스냅샷 자동 기록)
├── assets/              USD 에셋 (로봇, 물체 뱅크, 씬)
├── docs/                설계 문서·평가 리포트
├── log/                 학습 산출물
├── scripts/             → scripts/README.md 참조 (디렉토리는 1단계 고정)
│   ├── tools/               parse_tfevents, list_envs, record_test_snapshot, notion_log …
│   ├── reinforcement_learning/rl_games/   train.py / play.py
│   ├── distillation/        teacher→student 증류 (자체 README)
│   ├── warm_states/         grasp 성공 상태 → pour warm start
│   ├── r2s_autotune/        Real2Sim actuator autotune (자체 README)
│   └── analysis/ pca/ assets_tools/ datasets/ reports/ probes/ …
└── source/
    ├── FABRICS/         Geometric Fabrics (벤더링)
    └── openarm/openarm/
        ├── common/          태스크 공유 코어 (reward/adaptive/logging/contract)
        ├── distillation/    DAgger + vision student 네트워크 (태스크 무관, 공유)
        ├── tesollo/{right,left,both}/<task>/    Tesollo 20 DOF
        ├── rh56f1/right/<task>/                 Inspire RH56F1 6 DOF
        └── gripper/{right,left,both}/<task>/    2지 그리퍼
```

태스크 폴더 하나의 구성 (예: `tesollo/right/grasp_v2/`):

```
grasp_right_env.py         env 본체
grasp_right_env_cfg.py     설정 (obs/action/reward weight/씬)
grasp_right_constants.py   차원·상수
grasp_right_preset.py      로봇·손 메타데이터, 카메라 상수
grasp_adr.py               ADR 커리큘럼
config/__init__.py         gym 등록
config/agents/*.yaml       rl_games 네트워크·PPO 설정
tests/                     정적 테스트 (isaaclab 없이 스텁으로 실행)
CLAUDE.md                  (일부 태스크) 도메인 진단 방법론
```

---

## 테스트

isaaclab 없이 도는 정적 테스트다. 각 태스크 폴더 안에서 돌린다.

```bash
python3 -m pytest source/openarm/openarm/tesollo/right/grasp_v2/tests -q
python3 -m pytest source/openarm/openarm/distillation/tests -q
```

`scripts/` 전체에 pytest를 걸지 말 것 — `probes/test_finger_joints.py` 는 테스트가 아니라
Isaac Sim 스크립트라 수집되면 import 에러를 낸다.

---

## 문서 지도

| 문서 | 내용 |
|---|---|
| [CLAUDE.md](CLAUDE.md) | 실험 규칙: 로그 우선, 증거 우선순위, 코드 수정·기록 규칙 |
| [scripts/README.md](scripts/README.md) | scripts 디렉토리 지도 (1단계 규칙의 이유 포함) |
| [scripts/tools/README.MD](scripts/tools/README.MD) | 도구별 사용법 |
| [scripts/distillation/README.md](scripts/distillation/README.md) | teacher→student 증류 |
| [scripts/r2s_autotune/](scripts/r2s_autotune/) | Real2Sim actuator autotune |
| `source/openarm/openarm/*/*/<task>/CLAUDE.md` | 태스크별 도메인 진단 방법론 (pour_v5 등) |
| `docs/` | 설계 문서·평가 리포트 |

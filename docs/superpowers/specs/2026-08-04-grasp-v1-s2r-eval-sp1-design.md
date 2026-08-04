# grasp_v1 sim2real 평가 하네스 — SP1 (그리드 스윕 + STATE provider) 설계

날짜: 2026-08-04
대상: tesollo grasp_v1 (left + right), rl_games LSTM 정책
상태: 설계 승인됨 (SP1 범위)

## 1. 목표

sim에서 물체를 **사용자가 지정한 그리드 위치들**에 고정 스폰하고, 학습된
grasp_v1 정책을 그 위치를 기점으로 실행해 **작업공간 성공률 히트맵**을
산출한다. sim2real 배포 전에 "이 물체 위치에서 정책이 성공하는가"를
정량적으로 답하는 것이 목적이다.

obs의 cup pose를 외부에서 주입하는 **provider seam**을 함께 만든다:

- **SP1 (이 문서)**: `state_frozen` provider — reset 시 ground-truth cup
  pose를 1회 캡처해 에피소드 내내 고정 주입. 배포 open-loop(freeze-once)와
  동일한 staleness를 갖는 **지각 상한선**.
- **SP2 (별도 스펙)**: `camera_frozen` provider — sim 카메라 렌더 →
  FoundationPose 6D 추정 → freeze-once 주입. seam은 SP1에서 완성되므로
  SP2는 provider 구현만 추가한다.

STATE(SP1) 히트맵 − CAMERA(SP2) 히트맵 = 지각 유발 열화 지도.

## 2. 범위

### 포함 (SP1)

0. **인터랙티브 모드 (`--interactive`, 주 사용 방식)**: Isaac Sim GUI
   상주 세션 — 씬(책상·로봇·뷰포트 카메라)·정책 로드 후 대기, 터미널
   명령으로 물체 소환→평가→결과 출력→리셋→대기 반복. §4.6 참조.
1. 신규 평가 스크립트 `scripts/eval_s2r/eval_sim2real.py` (+ 순수 로직
   모듈 + 테스트).
2. 그리드 스윕(무인 배치): `--grid_x/--grid_y/--grid_nx/--grid_ny/
   --grid_repeats` → env를 셀에 1:1 배정, 고정 스폰. 서버 headless 가능.
3. grasp_v1 env(left/right)에 **최소 훅 2개**:
   - 고정 스폰 오버라이드 (`_reset_idx`)
   - cup-pose obs 오버라이드 (`_get_observations`)
4. pose source 3종: `live`(현행 학습과 동일, sanity) / `state_frozen`
   (기본) / `camera_frozen`(SP2 자리, SP1에서는 미구현 에러).
5. 셀별 지표 집계 → CSV + JSON + 히트맵 PNG.
6. `--render` 단일-env 시각 재생 모드.
7. `--robot left|right` 양팔 공용.

### 제외 (SP1 아님)

- 카메라 렌더·FoundationPose·extrinsics·py3.8 브리지 (전부 SP2).
- 실기(ROS) 입력 연동 — 이 하네스는 순수 sim 평가.
- grasp_v2 및 다른 태스크 — grasp_v1 전용.
- 학습 코드·reward·기존 랜덤 스폰 경로 변경 — 훅은 평가 시에만 활성.

## 3. 사용자 인터페이스 (CLI)

```bash
# 배치 히트맵 (기본)
python scripts/eval_s2r/eval_sim2real.py \
  --robot left \
  --checkpoint <path/to/ep_20000.pth> \
  --pose_source state_frozen \
  --grid_x 0.21 0.33 --grid_nx 5 \
  --grid_y -0.16 0.02 --grid_ny 5 \
  --grid_repeats 8 \
  --episodes_per_env 3 \
  --out log/eval_s2r/left_lstm_test12

# 단일 위치 시각 확인
python scripts/eval_s2r/eval_sim2real.py \
  --robot right --checkpoint <...> \
  --object_x 0.27 --object_y -0.10 --render --real-time

# 인터랙티브 상주 세션 (GUI, 주 사용 방식)
python scripts/eval_s2r/eval_sim2real.py \
  --robot right --checkpoint <...> --interactive
```

인자 규칙:

- `--robot {left,right}` (필수) → task id `open-tesol_l_grasp_v1-play-lstm`
  / `open-tesol_r_grasp_v1-play-lstm` 자동 매핑.
- `--checkpoint` (필수). play.py와 동일한 prefix-glob 해석 지원.
- 그리드 모드: `--grid_x MIN MAX --grid_nx N --grid_y MIN MAX --grid_ny N`
  전부 지정. `num_envs = nx*ny*grid_repeats` 자동 산출.
- 단일 모드: `--object_x X --object_y Y` (그리드 인자와 상호배타).
  `--render` 시 단일 모드 강제.
- `--interactive`: 상주 대화 세션(§4.6). GUI 강제(headless 불가),
  num_envs=1 고정, 그리드/단일 인자와 상호배타.
- `--object_z`: 생략 시 env의 `object_spawn_z_buf`(물체별 테이블 높이)
  사용. 지정 시 전 env 공통 오버라이드.
- `--pose_source {live,state_frozen,camera_frozen}` 기본 `state_frozen`.
- `--episodes_per_env N` 기본 3: env마다 N 에피소드 순차 실행 →
  셀당 표본 = `grid_repeats × N`.
- ADR은 항상 비활성(내부에서 play.py의 `--disable_adr` 동작 고정).
  스폰 랜덤화가 그리드 고정과 충돌하기 때문.
- 검증: 그리드 min>max, nx/ny<1, 그리드+단일 동시 지정, robot 오타,
  checkpoint 부재는 즉시 명확한 에러로 종료 (fail fast).

## 4. 아키텍처

```
scripts/eval_s2r/
├── eval_sim2real.py   # 엔트리: AppLauncher, env 생성, agent 로드, 루프
├── grid.py            # 그리드 생성·env↔셀 매핑 (순수 함수)
├── providers.py       # PoseProvider 인터페이스 + Live/StateFrozen
├── report.py          # 셀 집계, CSV/JSON/heatmap PNG (순수 함수)
├── console.py         # 인터랙티브 명령 파서·상태기계 (순수 함수)
└── tests/
    ├── test_grid.py
    ├── test_providers.py
    ├── test_report.py
    └── test_console.py
```

- `scripts/` 바로 아래 1단계 신규 디렉토리 — hdgp 레이아웃 규칙 준수.
- `grid.py`/`report.py`는 Isaac 의존 없는 순수 함수 → CPU에서 단위 테스트.
- `providers.py`는 env를 duck-type으로만 사용 → mock으로 테스트.

### 4.1 체크포인트 로딩 (play.py 패턴 복제)

play.py의 검증된 시퀀스를 그대로 따른다: `params/env.yaml·agent.yaml`
복원 → `load_checkpoint/load_path` 설정 → `Runner` → `create_player()` →
`agent.restore()` → `agent.reset()` → `is_rnn`이면 `init_rnn()`, 루프에서
done env의 LSTM states를 0으로 리셋.

**의도적 트레이드오프**: play.py는 모듈 로드 시 argparse를 실행하므로
import 재사용이 불가능하다. play.py를 리팩터링하지 않고(현행 배포 경로
보호, 외과적 변경 원칙) 로더 글루 ~50줄을 eval_sim2real.py에 복제한다.
복제 지점은 주석으로 play.py 라인 참조를 남긴다.

### 4.2 env 훅 (grasp_left_env.py / grasp_right_env.py — 좌우 미러 동일)

훅은 **속성 부재 시 완전 무동작**(getattr 기본 None) — 학습·기존 play
경로에 영향 0. 두 파일에 동일하게 넣고 좌우 diff 대칭을 검증한다.

**훅 A — 고정 스폰** (`_reset_idx`, 비-demo 분기의 랜덤 스폰 직후):

```python
# eval_s2r: 평가 하네스가 설정하는 고정 스폰 오버라이드 (학습 시 None)
override = getattr(self, "eval_fixed_spawn_local", None)  # [num_envs,3] or None
if override is not None:
    obj_pos_local = override[env_ids].clone()
```

`eval_fixed_spawn_local`은 env-origin 기준 local 좌표. z에 NaN이 들어오면
해당 env는 `object_spawn_z_buf` 값을 유지(물체별 테이블 높이 존중).

**훅 B — cup pose obs 오버라이드** (`_get_observations`, cup_pos_noisy
산출 직후):

```python
# eval_s2r: 외부 주입 cup pose가 있으면 GT+noise 대신 사용 (학습 시 None)
pose_override = getattr(self, "eval_cup_pos_override", None)  # [num_envs,3] or None
if pose_override is not None:
    cup_pos_noisy = pose_override
```

- 주입값은 이미 "지각 결과"이므로 `obs_noise_cup_pos`를 **추가로 얹지
  않는다**(이중 노이즈 방지). live 모드에서는 속성이 None → 현행 노이즈
  경로 그대로.
- 이 텐서는 `palm_to_cup`·`cup_to_fingertip` 두 슬롯에만 전파된다
  (actor obs에서 cup이 쓰이는 곳 전부). critic obs는 건드리지 않는다 —
  평가 시 critic은 행동에 영향 없음.

### 4.3 PoseProvider seam (`providers.py`)

```python
class PoseProvider(Protocol):
    def on_reset(self, env, env_ids) -> None: ...
    def get_override(self, env) -> torch.Tensor | None: ...  # [num_envs,3] local
```

- `LiveProvider`: 항상 None 반환 → env 현행 경로(GT+DR노이즈). 학습 분포
  재현 sanity 기준선.
- `StateFrozenProvider`: `on_reset`에서 해당 env의 GT cup pos(local)를
  캡처해 버퍼에 저장(새 텐서 생성, 기존 버퍼 변경 없음), `get_override`는
  그 고정 버퍼 반환. **에피소드 중 갱신하지 않음** — 파지·리프트 후 obs가
  stale해지는 배포 open-loop 조건을 그대로 재현.
- `CameraFrozenProvider`(SP2): SP1에서는 선택 시
  `NotImplementedError("SP2")`로 즉시 종료.

평가 루프가 매 스텝 `env.eval_cup_pos_override = provider.get_override(env)`
를 갱신하고, done env에 대해 `provider.on_reset` 재호출.

### 4.4 평가 루프·지표 (`eval_sim2real.py` + `report.py`)

에피소드 종료 판정과 지표는 play.py `--eval_episodes` 로직(:821-941)을
따른다. env i → 셀 (i // repeats) 고정 매핑(`grid.py`).

셀별 집계(에피소드 단위):

| 지표 | 산출 |
|---|---|
| success_rate | 에피소드 말 `in_success_region` 래치 비율 |
| lifted_rate | lift 래치 비율 |
| grip_finger_count | 파지 중 평균 접촉 손가락 수 |
| object_displacement | `object_pos` − `object_init_pos` 수평 변위(넘어뜨림/밀림 탐지) |
| finger_contact_hist | 손가락별 접촉 비율 |

출력(`--out` 디렉토리):

- `results.csv` — 셀×지표 (x, y, n_episodes, 지표들)
- `summary.json` — 전역 평균, 인자 스냅샷(재현성), checkpoint 경로,
  git SHA
- `heatmap_success.png`, `heatmap_lifted.png` — x·y 축 그리드, 값 주석
  포함 (matplotlib, 서버 headless 대비 Agg 백엔드)

### 4.5 데이터 흐름

```
CLI 인자 → grid.py: 셀 목록 + env→셀 매핑 + spawn 텐서[num_envs,3]
 → env 생성(ADR off) → env.eval_fixed_spawn_local = spawn 텐서
 → agent 로드(LSTM)
 → reset → provider.on_reset(전체)
 → 루프: provider.get_override → env 속성 갱신 → obs → action → step
          done env: LSTM state 0 리셋 + provider.on_reset(done ids)
          에피소드 기록 → 셀 버킷 적재
 → 전 셀 episodes_per_env 충족 시 종료 → report.py → CSV/JSON/PNG
```

### 4.6 인터랙티브 모드 (`--interactive`)

Isaac Sim **GUI 상주 세션**. 기동 시 씬(책상·로봇·뷰포트 카메라,
`--cam_eye/--cam_lookat` 재사용)과 정책을 로드하고 대기 모드로 진입.

상태기계 (`console.py`, 순수 함수로 구현·테스트):

```
STAGED(대기) --spawn X Y [Z]--> EVAL(에피소드 실행)
EVAL --에피소드 종료--> REPORT(결과 출력) --자동--> RESET --> STAGED
STAGED --quit--> 종료
```

터미널 명령:

| 명령 | 동작 |
|---|---|
| `spawn X Y [Z]` | 해당 위치에 물체 스폰(훅 A) 후 즉시 정책 시작 |
| `repeat` | 직전 위치로 재실행 |
| `obj N` | 다음 스폰에 사용할 물체(0~7) 선택 |
| `sweep ...` | 그리드 인자를 받아 배치 스윕을 세션 내 실행 |
| `quit` | 세션 종료 |

- **대기 중 물체 처리**: STAGED에서는 물체를 작업공간 밖 대기 위치
  (테이블 아래 z<0 또는 원거리)에 두어 "물체 없음" 상태를 연출. spawn
  명령이 목표 위치로 이동시킨다.
- **GUI 정지 방지**: stdin 블로킹 읽기는 렌더 루프를 멈춰 창이 "응답
  없음"이 된다. 대기 중에도 매 프레임 `app.update()`(렌더)를 돌리면서
  `select()`로 stdin을 논블로킹 폴링한다. STAGED 중 물리 스텝은 하지
  않는다(로봇은 리셋 자세 유지).
- **에피소드 결과 즉시 출력**: success / lifted / grip_finger_count /
  변위 / 손가락별 접촉을 종료 직후 터미널에 표. 세션 전체 이력은 종료
  시 `--out`에 CSV로 저장(있으면).
- num_envs=1, `--real-time` 기본 on. 로컬 pc5090 실행 전제(서버는
  headless라 GUI 불가).

## 5. 에러 처리

- 인자 검증 실패·checkpoint/params 부재 → argparse error 또는 명확한
  메시지로 즉시 종료.
- 그리드가 학습 스폰 범위(중심 ±0.06 + ADR 최대 ±0.08)를 벗어나면
  **경고 출력 후 진행** — 분포외 위치 성능 측정 자체가 이 도구의
  용도이므로 차단하지 않는다.
- 물리 폭주(비유한 obs) 감지 시 해당 에피소드를 `invalid`로 표기하고
  집계에서 제외, summary에 invalid 건수 보고 (조용히 삼키지 않음).
- provider가 비유한 pose를 반환하면 즉시 에러 (시스템 경계 검증).

## 6. 테스트

정적(CPU, Isaac 불필요 — pytest):

1. `test_grid.py` — 셀 좌표 생성(min/max/개수), env→셀 매핑, repeats,
   단일 모드, 잘못된 인자 거부.
2. `test_providers.py` — mock env로 StateFrozen: reset 캡처 후 물체를
   움직여도 override 불변(freeze 검증), done 부분 리셋 시 해당 env만
   재캡처, Live는 항상 None.
3. `test_report.py` — 셀 집계 산술, invalid 제외, CSV/JSON 스키마,
   heatmap 파일 생성.
3b. `test_console.py` — 명령 파싱(spawn/repeat/obj/sweep/quit, 잘못된
   입력 거부), 상태 전이(STAGED→EVAL→REPORT→RESET→STAGED).
4. env 훅 회귀 — 속성 None일 때 `_reset_idx`/`_get_observations` 출력이
   현행과 동일함(기존 grasp_v1 테스트 스위트 통과로 확인).

GPU 스모크(사용자 게이트 — 학습 자원 사용은 지시 후):

- right lstm_test3, 3×3 그리드 × repeats 2 → 히트맵 산출, 중심 셀
  success가 play.py `--eval_episodes` 결과(≈0.89)와 ±0.05 일치.
- left lstm_test12 동일 절차.
- `live` vs `state_frozen` 중심 셀 비교 → freeze staleness 영향 1차 수치.
- 인터랙티브 스모크(로컬 pc5090, GUI): 기동→spawn→평가→결과 출력→리셋
  →재-spawn 2회→quit. 대기 중 GUI 응답성(창 조작 가능) 확인.

## 7. 위험·미해결 (point 4 "문제 서칭"의 SP1 몫)

1. **freeze staleness**: 학습은 live cup_pos, 평가는 frozen — 리프트 후
   stale obs를 정책이 견디는지가 첫 측정 대상. live 모드가 대조군.
2. **in_success_region 정의 의존**: 성공 판정을 학습 코드와 공유하므로
   판정 기준 변경 시 히트맵 수치 의미가 달라짐 — summary에 git SHA 기록
   으로 추적.
3. **셀당 env 수 × episodes 조합의 GPU 메모리**: nx·ny·repeats가 크면
   num_envs 폭증 — 스크립트가 num_envs를 출력하고 진행 확인 문구 표시.
4. **다물체(8종 % 배정)**: env→셀 매핑과 env→물체 배정(env_id % 8)이
   얽혀 셀마다 물체 구성이 달라질 수 있음 — repeats를 8의 배수로 권장
   (경고 출력), 물체별 분해 컬럼을 CSV에 포함.

## 8. SP2 인터페이스 계약 (참고)

SP2는 `CameraFrozenProvider` 하나만 구현하면 된다: `on_reset`에서
카메라 렌더 → FoundationPose → base-local 변환 → 버퍼 저장. seam·훅·
지표·히트맵은 SP1 산출물을 그대로 사용한다.

# eval_s2r — grasp_v1 sim2real 평가 하네스

물체를 사용자가 지정한 그리드/단일 위치에 스폰하여 학습된 grasp_v1 정책을 평가하고, 작업공간 성공률 히트맵·CSV·JSON을 산출한다.

**설계 및 아키텍처**: [docs/superpowers/specs/2026-08-04-grasp-v1-s2r-eval-sp1-design.md](../../docs/superpowers/specs/2026-08-04-grasp-v1-s2r-eval-sp1-design.md)

---

## 사용법

### 1. 배치 히트맵 (무인 그리드 스윕)

```bash
python scripts/eval_s2r/eval_sim2real.py \
  --robot left \
  --checkpoint <path/to/ep_20000.pth> \
  --pose_source state_frozen \
  --grid_x 0.21 0.33 --grid_nx 5 \
  --grid_y -0.16 0.02 --grid_ny 5 \
  --grid_repeats 8 \
  --episodes_per_env 3 \
  --out log/eval_s2r/left_lstm_test12
```

- `--grid_x MIN MAX` / `--grid_y MIN MAX`: 스폰 범위
- `--grid_nx N` / `--grid_ny N`: 격자점 개수 (nx × ny × grid_repeats 개 env 자동 생성)
- `--grid_repeats`: 각 셀에서 독립적으로 반복 실행 횟수 (기본값 8)
- `--episodes_per_env`: 각 env마다 순차 에피소드 개수 (기본값 3)
- `--pose_source {state_frozen,live,camera_frozen}`: 지각 모드 (기본값 state_frozen)
- `--out`: 결과 CSV/JSON/PNG 저장 경로

### 2. 단일 위치 시각 확인

```bash
python scripts/eval_s2r/eval_sim2real.py \
  --robot right \
  --checkpoint <path/to/ep_20000.pth> \
  --object_x 0.27 --object_y -0.10 \
  --render --real-time
```

- `--object_x X --object_y Y`: 스폰 좌표 (단일)
- `--object_z Z`: Z 좌표 지정 (생략 시 물체별 기본값)
- `--render`: GUI 단일-env 시각 재생 활성화
- `--real-time`: 실시간 페이싱 (기본 고속 재생)

### 3. 인터랙티브 상주 세션 (주 사용 방식)

```bash
python scripts/eval_s2r/eval_sim2real.py \
  --robot right --checkpoint <path/to/ep_20000.pth> \
  --interactive
```

GUI 상주 세션을 시작한다. 터미널에서 다음 명령어로 물체 소환·평가·리셋:

| 명령 | 인자 | 설명 |
|------|------|------|
| `spawn` | `X Y [Z]` | 물체를 (X, Y, Z)에 스폰 후 평가. Z 생략 시 기본값 사용. 완료 후 결과 출력. |
| `repeat` | 없음 | 마지막 `spawn` 위치에서 재실행 (객체 위치 유지). |
| `reset` | 없음 | 환경 초기화 후 물체를 대기 위치로 파킹, STAGED 복귀. |
| `back2ini` | 없음 | 직전 에피소드 관절 궤적을 **역재생** — 잡기→놓기 연속동작(내려놓고 초기 자세 복귀). 에피소드가 타임아웃 직전 조기 정지해 장면이 보존된 경우에만 가능(물체 낙하 등으로 자연 done되면 불가). 역재생은 정책 무관 kinematic 추종이며 물체는 물리로 자연 안착. |
| `obj` | `N` | 물체 인덱스를 N으로 변경 시도. **단, 단일 env 세션에서는 무시되며 물체 0 고정** (다음 spawn 시 경고 메시지). |
| `sweep` | `XMIN XMAX NX YMIN YMAX NY REPEATS` | 문법상 7개 인자 필수이나 인터랙티브 모드에서는 미지원: `[ERR] sweep 은 인터랙티브 num_envs=1 세션에서 미지원` 메시지 출력. 배치 모드(--grid_x/y 등)로 별도 실행하세요. |
| `quit` | 없음 | 세션 종료. |

---

## GPU 스모크 체크리스트 (사용자 게이트)

### Phase 1: 기본 모드 검증

1. [ ] **right lstm_test3, 3×3 그리드 × repeats 8, episodes_per_env 2**
   - 히트맵 2장 생성 확인 (전·중심 셀)
   - 중심 셀 `episode_success_buf` ≈ play.py --eval_episodes 값(0.89) ±0.05 범위 확인
   - CSV/JSON 로그 생성 확인

2. [ ] **left lstm_test12 동일 절차**
   - 좌팔 정책 로드 및 평가 성공
   - 히트맵·CSV 생성

3. [ ] **--pose_source live vs state_frozen 중심 셀 비교**
   - live (현행 학습 동등) vs state_frozen (고정 추적) 성공률 차이 기록
   - freeze staleness 1차 수치화

4. [ ] **인터랙티브 (로컬 pc5090, GUI) 기본 흐름**
   - 기동 → `spawn 0.27 -0.10` → 결과 출력 → `spawn` 재실행 2회 → `quit` 정상 종료
   - GUI 마우스 상호작용(뷰포트 회전) 대기 중 가능 확인

5. [ ] **훅 무동작 회귀 (기존 play.py 검증)**
   - 기존 play.py 재생이 이전과 동일 동작 (랜덤 스폰, obs 정상)
   - 평가 세션 종료 후 play.py 실행 무영향 확인

### Phase 2: GPU 심화 검증

6. [ ] **episode_success_buf 프리스텝 스냅샷 타이밍 vs play.py --eval_episodes 교차검증**
   - 동일 체크포인트에서 eval_s2r과 play.py 성공률 ±0.05 범위 일치 확인

7. [ ] **체크포인트 params/env.yaml·agent.yaml 복원 실동작**
   - 로깅된 환경 설정 복원 검증
   - BC-pretrained 체크포인트인 경우 optimizer restore 이슈 주의 (현재 _patch_optimizer_restore 미이식)

8. [ ] **vars(args_cli) JSON 직렬화 가능 여부**
   - AppLauncher 주입 필드가 로그·전달에서 JSON 형식으로 손실 없이 처리되는지 확인

9. [ ] **rl_games get_action(is_deterministic) API 호환**
   - 결정적 action 추출 정상 동작 확인

10. [ ] **num_envs=1 truncation이 dones로 실제 전달 (무한루프 방지)**
    - 단일 env 에피소드 자동 종료 확인 (종료 조건 미전달 = 무한루프)

11. [ ] **--real-time 페이싱 정상**
    - 실시간 렌더링 시 wall-clock과 sim 시간 일치 확인

12. [ ] **GUI 창 닫기 mid-episode → invalid 처리·정상 종료**
    - 에피소드 중 창 닫기 → 에러 없이 graceful 종료

13. [ ] **파이프 stdin EOF → 정상 quit**
    - 인터랙티브 모드에서 입력 종료(EOF) 시 hang 없이 정상 종료

---

## 출력 형식

그리드/단일 모드는 둘 다 `main()`을 거치므로 `--out`을 지정했는지 여부와 무관하게 항상
`results.csv`+`summary.json`을 남긴다(`--out` 미지정 시 `log/eval_s2r/{robot}_{timestamp}/`에 자동 생성).
히트맵 PNG만 그리드 모드 전용.

- **그리드 모드**: `{out}/heatmap_success.png`, `{out}/heatmap_lifted.png` (히트맵 이미지 2장)
                 `{out}/results.csv` (셀별 성공률·통계, `per_obj_success`/`finger_contact_rates`는 JSON 인코딩된 문자열 컬럼)
                 `{out}/summary.json` (전체 메타데이터 + 셀별 집계)
                 터미널에 셀별 요약 표 출력
- **단일 모드**: 위와 동일하게 `{out}/results.csv`+`{out}/summary.json` 기록(셀 1개, 히트맵 없음) + 터미널 출력
- **인터랙티브**: 각 `spawn` 이후 터미널 출력(`fingers=[...]` 포함). `--out` 지정 시 세션 종료 시
                `{out}/interactive_history.json` 에 세션 전체 에피소드 이력을 **JSON**으로 저장
                (다른 모드의 CSV/JSON 이중 출력과 달리 인터랙티브는 JSON 단일 파일만 — 스펙 문서의
                CSV 언급과는 의도적으로 다르다: 세션 이력은 셀 집계가 없는 원시 `EpisodeResult` 리스트라
                CSV 스키마가 맞지 않는다)

---

## SP2 — camera_frozen (D435i 렌더 + FoundationPose)

**오프라인 3-pass 워크플로우**: Pass 1에서 D435i 헤드카메라로 렌더링한 프레임 집합을 고정 카메라 pose로 처리하고, Pass 2에서 FoundationPose로 컵 위치를 추정한 뒤, Pass 3에서 고정 pose를 eval_sim2real에 주입하여 정책 평가를 수행한다. 각 env마다 1프레임만 캡처되므로(freeze-once 정책) 전체 워크플로우는 **render(Pass 1) → mesh export(Task 6) → FoundationPose 배치(Pass 2, vision-3090 컨테이너) → eval camera_frozen(Pass 3, 로컬) → delta 비교(사용자)**의 5단계로 진행된다.

### 실행 절차

#### (0) 프리뷰: 렌더 헤드 자세 확인

```bash
isaaclab.sh -p scripts/eval_s2r/render_pass.py \
  --robot right \
  --grid_x 0.21 0.33 --grid_nx 3 \
  --grid_y -0.16 0.02 --grid_ny 3 \
  --grid_repeats 8 \
  --head_tilt <TILT_RAD> \
  --head_pan 0.0 \
  --out frames/right \
  --preview
```

- `--head_tilt` / `--head_pan`: 헤드 카메라 관절 목표각[rad] (기본값 0.0)
- `--preview`: 중심 셀 1환경만 렌더 → `preview_env0.png` 저장 후 종료 (NPZ/메타 미생성)
- **출력**: 프리뷰 이미지로 시야 확인 후 실제 `--head_tilt` 값 결정

#### (1) 메시 추출

```bash
isaaclab.sh -p scripts/eval_s2r/export_meshes.py \
  --object_map frames/right/object_map.json \
  --out meshes/
```

- `--object_map`: Pass 1 렌더에서 생성된 object_map.json 경로
- `--out`: 메시 출력 디렉토리 (각 물체가 `<id>.obj`로 저장됨)
- `--kit`: USD pxr import 실패 시만 추가 (일반적으로 불필요)

#### (2) 렌더 (전체 프레임)

```bash
isaaclab.sh -p scripts/eval_s2r/render_pass.py \
  --robot right \
  --grid_x 0.21 0.33 --grid_nx 3 \
  --grid_y -0.16 0.02 --grid_ny 3 \
  --grid_repeats 8 \
  --head_tilt <TILT_RAD> \
  --head_pan 0.0 \
  --out frames/right
```

- (0)에서 확정한 `--head_tilt` 값 사용
- `--preview` 제거 (전체 렌더 실행)
- **출력**: `frames/right/` 디렉토리
  - `env_0.npz`, `env_1.npz`, ... (RGB/depth/mask/K/T_local_cam/GT 위치 포함)
  - `object_map.json` (물체 ID → USD 경로 매핑)
  - `meta.json` (그리드 사양·메타데이터)

#### (3) FoundationPose 배치 (vision-3090 컨테이너)

**로컬에서 먼저 rsync**:
```bash
rsync -av frames/right/ vision-3090:hdgp_perception/frames/right/
rsync -av meshes/ vision-3090:hdgp_perception/meshes/
```

**vision-3090에서 컨테이너 내 실행** (perception repo 런북 참조 — 컨테이너 기동 방법):
```bash
python3 scripts/eval_s2r/fp_batch.py \
  --robot right \
  --frames frames/right \
  --mesh_dir meshes \
  --out poses/right.json \
  --iteration 5
```

- `--frames`: Pass 1 렌더 출력 디렉토리 경로
- `--robot`: `left` 또는 `right`
- `--mesh_dir`: (1)에서 추출한 메시 디렉토리
- `--out`: poses 출력 JSON 경로 (기본값 `poses/<robot>.json`)
- `--iteration`: FoundationPose register() 정제 iteration 수 (기본값 5)
- **출력**: `poses/right.json`
  ```json
  {
    "env_0": {"ok": true, "T_cam_obj": [[4x4 행렬]]},
    "env_1": {"ok": false, "reason": "mesh_missing"},
    ...
  }
  ```

#### (4) eval camera_frozen (로컬)

**rsync 회수**:
```bash
rsync -av vision-3090:hdgp_perception/poses/ log/eval_s2r/poses/
```

**로컬에서 평가**:
```bash
isaaclab.sh -p scripts/eval_s2r/eval_sim2real.py \
  --robot right \
  --checkpoint <path/to/ep_20000.pth> \
  --pose_source camera_frozen \
  --poses log/eval_s2r/poses/right.json \
  --frames_meta frames/right/meta.json \
  --grid_x 0.21 0.33 --grid_nx 3 \
  --grid_y -0.16 0.02 --grid_ny 3 \
  --grid_repeats 8 \
  --episodes_per_env 3 \
  --out log/eval_s2r/right_camera \
  --headless
```

- `--pose_source camera_frozen`: 고정 pose 모드 활성화
- `--poses`: (3) FoundationPose 결과 JSON
- `--frames_meta`: Pass 1 메타 JSON (env 배치 검증용)
- `--grid_x/y/nx/ny/repeats`: **Pass 1 렌더와 동일해야 함** (meta.json으로 자동 검증)
- `--checkpoint`: 정책 .pth 경로 (prefix-glob 지원)
- `--episodes_per_env`: 각 env마다 순차 에피소드 개수 (기본값 3)
- `--headless`: GUI 비활성화 (배치 모드)
- **출력**: `log/eval_s2r/right_camera/`
  - `heatmap_success.png`, `heatmap_lifted.png`
  - `results.csv`, `summary.json`

#### (5) 델타 맵 생성

```bash
python3 scripts/eval_s2r/delta_report.py \
  --state log/eval_s2r/right_lstm_test3_grid/summary.json \
  --camera log/eval_s2r/right_camera/summary.json \
  --out log/eval_s2r/right_delta
```

- `--state`: state_frozen 평가 summary.json (Pass 1 baseline)
- `--camera`: camera_frozen 평가 summary.json (Pass 3 결과)
- `--out`: 델타 결과 디렉토리
- **출력**: `log/eval_s2r/right_delta/`
  - `delta_success.csv` (셀별 성공률 차이)
  - `heatmap_delta_success.png`, `heatmap_delta_lifted.png`

---

### 4-게이트 체크리스트

| 게이트 | 확인 항목 | 검증 방법 |
|--------|----------|---------|
| **G1: 렌더 프리뷰** | 헤드 카메라 시야·tilt 각도 적절 | (0) --preview 실행 → preview_env0.png 확인(사용자 시각) |
| **G2: NPZ 무결성** | Pass 1 렌더 200장 생성 완료 | `ls frames/right/env_*.npz \| wc -l` → 예상 env 개수(grid_nx × grid_ny × grid_repeats) 확인 |
| **G3: FP 성공률/오차** | FoundationPose 추정 품질 | poses.json의 ok=true 비율 > 90% + GT 대비 위치오차 < 5cm(메시 bbox 단위) |
| **G4: 평가 + 델타** | camera_frozen 평가 완료 및 베이스라인과 비교 | heatmap_success.png 생성 + delta_success.csv 셀별 비교 |

**주의**: 게이트를 통과하지 못한 경우 다음 단계로 진행하지 마세요.

---

### poses.json 및 meta.json 스키마

#### poses.json (FoundationPose 출력)
```json
{
  "env_0": {
    "ok": true,
    "T_cam_obj": [
      [0.9, 0.0, 0.1, 0.05],
      [0.0, 0.95, 0.0, -0.02],
      [-0.1, 0.0, 0.9, 0.35],
      [0.0, 0.0, 0.0, 1.0]
    ]
  },
  "env_1": {
    "ok": false,
    "reason": "mesh_missing"
  }
}
```

- `ok`: 추정 성공 여부
- `T_cam_obj` (ok=true일 때): 4×4 카메라→물체 변환 행렬
- `reason` (ok=false일 때): 실패 사유 (`mesh_missing`, `inference_fail`, 등)

#### meta.json (Pass 1 렌더 메타)
```json
{
  "grid": {
    "x_min": 0.21, "x_max": 0.33, "nx": 3,
    "y_min": -0.16, "y_max": 0.02, "ny": 3,
    "repeats": 8
  },
  "robot": "right",
  "head_tilt": -0.3,
  "head_pan": 0.0,
  "camera_K": [[...], [...], [...]],
  "timestamp": "2026-08-04T21:30:00"
}
```

---

### 신규 산출물 표

| 경로 | 출처 | 용도 |
|------|------|------|
| `frames/right/env_*.npz` | Pass 1 render_pass.py | RGB/depth/mask/K/intrinsics/GT 위치 |
| `frames/right/object_map.json` | Pass 1 render_pass.py | 물체 ID → USD/scale 매핑 |
| `frames/right/meta.json` | Pass 1 render_pass.py | 그리드 사양·카메라 내부 파라미터 |
| `meshes/<id>.obj` | export_meshes.py | FoundationPose용 메시 |
| `poses/right.json` | fp_batch.py (Pass 2) | T_cam_obj 추정치 |
| `log/eval_s2r/right_camera/heatmap_success.png` | eval_sim2real.py (Pass 3) | 성공률 히트맵 |
| `log/eval_s2r/right_camera/summary.json` | eval_sim2real.py (Pass 3) | 셀별 집계 (perception_fail_rate 포함) |
| `log/eval_s2r/right_delta/delta_success.csv` | delta_report.py | 베이스라인 대비 성공률 차이 |

---

### 주의사항 (SP2 특정)

- **camera_frozen은 그리드 모드 전용**: Pass 1 렌더 그리드와 eval env 배치를 1:1로 정렬하는 계약이므로, 단일/인터랙티브 모드에서 `--pose_source camera_frozen`을 지정하면 부팅 전 검증 오류로 거부됨.
- **메타 검증**: eval_sim2real이 부팅 시 --frames_meta의 그리드 사양과 CLI 인자(--grid_x/y/nx/ny)를 비교하여 불일치 시 종료.
- **FP 실패 처리**: poses.json에서 ok=false인 env는 평가 시 건너뜀(해당 셀의 perception_fail_rate 컬럼에 집계).
- **메시 누락**: mesh_dir/object_map 탐색 순서: (1) object_map 동일 디렉토리의 sibling .obj, (2) mesh_dir/<id>.obj. 둘 다 없으면 해당 물체를 reason="mesh_missing"으로 처리.

---

## 주의사항

- **ADR 비활성화**: 그리드 고정 스폰과 충돌하므로 평가 중 자동 비활성화
- **pose_source**: SP1은 `state_frozen`, `live` 만 지원. `camera_frozen`은 SP2에서 구현.
- **finger_contact_rates**: `results.csv`/`summary.json`에 손가락별(엄지·검지·중지·약지·소지, T/I/M/R/P 순)
  접촉률이 셀당 원소별 평균으로 포함된다(표본 0이면 `null`).
- **체크포인트 경로**: play.py와 동일한 prefix-glob 지원 (`*_ep_*.pth` 자동 해석)

# grasp_v1 sim2real 평가 SP2 — camera_frozen (D435i 렌더 + FoundationPose++) 설계

날짜: 2026-08-04
전제: SP1 완료(`2026-08-04-grasp-v1-s2r-eval-sp1-design.md`) — seam·훅·그리드 하네스 재사용
상태: 설계 승인 대기

## 1. 목표

STATE(GT freeze-once) 히트맵과 동일 조건에서 **지각 파이프라인(FoundationPose++)이
추정한 cup pose**를 주입해 CAMERA 히트맵을 산출한다.

**STATE − CAMERA = 지각 유발 열화 지도** — sim2real 배포 전에 "카메라 지각 때문에
워크스페이스가 어디서 얼마나 줄어드는가"를 정량화한다.

SP1의 두 결과가 아키텍처를 결정한다:
- freeze-once 비용 ≈ 0 (검증됨) → 에피소드 시작 pose 1회만 필요
- 그리드 스폰은 env마다 매 에피소드 동일 → **env당 프레임 1장이면 충분**

→ 라이브 Isaac↔FP 브리지가 불필요한 **오프라인 3-pass**로 구성한다.
py3.8(FP 컨테이너)↔py3.11(Isaac), Blackwell에서 FP 미빌드 문제가 전부 소멸.

## 2. 아키텍처 (오프라인 3-pass)

```
Pass 1  [server 또는 pc5090, Isaac]  render_pass
  그리드 스폰(기존 훅) → head tilt 명령 → D435i 카메라 렌더 1프레임/env
  → frames/<robot>/env_<i>.npz (RGB-D, K, T_local_cam, GT pose, seg 마스크, 메타)
        │ rsync (Tailscale)
Pass 2  [vision-3090, perception-plus-plus:humble 컨테이너]  fp_batch
  NPZ 배치 → FoundationPose register(mesh, K, rgb, depth, mask) → T_cam_obj
  → poses/<robot>.json (env_idx → pose | FAIL)
        │ rsync
Pass 3  [server, Isaac]  기존 eval_sim2real.py + CameraFileProvider
  pose 파일 → base-local 변환 → seam 주입 → 그리드 평가
  → CAMERA 히트맵 + STATE−CAMERA 델타 맵
```

## 3. 범위

### 포함

1. **Pass 1 렌더 스크립트** `scripts/eval_s2r/render_pass.py` (Isaac 엔트리)
2. **Pass 2 FP 배치 스크립트** `scripts/eval_s2r/fp_batch.py` (컨테이너 내부 실행,
   Isaac·torch 프로젝트 의존 없음 — numpy/imageio 수준)
3. **CameraFileProvider** (`providers.py` 확장 — seam 규격 그대로)
4. **물체 mesh 준비**: visdex 8종 `visual_model.obj` + cup 계열 → FP 입력 목록/매핑
5. **델타 맵 리포트**: STATE·CAMERA summary.json 2개 → 셀별 Δ CSV + 델타 히트맵 PNG
   (`scripts/eval_s2r/report.py` 소폭 확장 또는 독립 스크립트)
6. 양팔(left/right) 지원

### 제외

- 실기 카메라/로봇 — 전 과정 sim (실기 extrinsics 캘리브 오차는 별도 문제)
- 라이브(매 스텝) 지각 — freeze-once만
- 학습 env 수정 — 카메라는 render_pass 전용 씬에서만 생성. grasp_v1/v2 학습 경로 무변경
- YOLO/bbox 마스크 — sim instance seg가 완벽 마스크 제공(FP의 마스크 의존성만 검증)

## 4. 컴포넌트 설계

### 4.1 Pass 1 — render_pass.py

- **씬**: eval_sim2real.py의 `_setup`과 동일한 env 생성(같은 task, 같은 그리드 훅,
  ADR off) + **TiledCamera 추가**. 정책 실행 없음 — reset 후 카메라 안정화 몇 스텝
  렌더 → 캡처 → 종료. (에피소드를 돌리지 않으므로 checkpoint 불필요)
- **카메라 부착**: 로봇 head 카메라 링크에 부착
  (`prim_path="/World/envs/env_.*/Robot/<head_cam_link>/Camera"` — 링크명은 구현 시
  USD에서 확인). 좌우 USD(sensor_rl/bi_rl) 각각 확인.
- **각도**: head는 pan/tilt revolute DOF(`head_j_pan`, `head_j_tilt`, actuator group
  "head_camera" 기존재). reset 후 tilt 목표각을 joint target으로 명령해 테이블을
  내려다보게 한다. **각도 값은 프리뷰 렌더로 확정**: `--preview` 모드가 중심 셀
  1 env의 RGB PNG를 저장 → 사람이 확인 → `--head_tilt` 인자로 고정.
  같은 각도를 실기에도 명령하면 마운트 동일로 extrinsics가 구조적으로 일치.
- **Intrinsics**: 기본값 = grasp_v2의 D435i 상수(CAMERA_* — 공칭). `--camera_info
  <yaml>` 지정 시 실기 camera_info의 K/해상도로 오버라이드(fx,fy→focal/aperture 환산).
  실기 yaml은 vision-3090에서 1회 추출해 repo에 커밋.
- **캡처 데이터** (env당 `env_<i>.npz`):

| 키 | 내용 |
|---|---|
| `rgb` | (H,W,3) uint8 |
| `depth` | (H,W) float32 [m] |
| `mask` | (H,W) bool — Cup prim instance seg |
| `K` | (3,3) float64 |
| `T_local_cam` | (4,4) — env-origin local 프레임 기준 카메라 pose (sim GT) |
| `gt_obj_pos_local` | (3,) — 검증·오차 산출용 GT |
| `gt_obj_rot_wxyz` | (4,) |
| `obj_idx`, `cell_idx`, `cell_x`, `cell_y` | 메타 |

- `meta.json` 1개: 그리드 인자 전체, num_envs, robot, git SHA, head_tilt, K 출처.
  **Pass 3이 이 메타와 자기 그리드 인자를 대조해 불일치 시 즉시 에러** (env 인덱스
  정렬이 전제이므로).
- 프레임 수: 5×5×8 = 200장/팔. 에피소드 반복은 동일 스폰이라 재사용.

### 4.2 Pass 2 — fp_batch.py (vision-3090 컨테이너)

- 입력: `frames/<robot>/` NPZ 디렉토리 + mesh 디렉토리.
- 물체 mesh 매핑: `obj_idx → assets/visdex_objects/urdf/<name>/visual_model.obj`
  (+ cup 계열). 매핑 테이블은 env의 MultiAsset 순서(`_GRASP_OBJECT_SPAWN` 목록,
  env_id % 8)에서 추출해 json으로 Pass 1이 함께 저장 — fp_batch는 그것만 읽는다.
  **mesh 스케일/단위(m)** 확인 필수(FP는 m 단위 mesh 가정).
- env당: estimater(mesh) → `register(K, rgb, depth, ob_mask)` → `T_cam_obj` (4,4).
  물체 8종이므로 estimater를 물체별로 1회 생성해 재사용(200회 등록).
- 실패 판정: register 예외/NaN/depth 결측 과다 → `"FAIL"` 기록(사유 문자열 포함).
- 출력: `poses/<robot>.json` — `{env_idx: {"T_cam_obj": [[...]], "ok": true} | {"ok":
  false, "reason": ...}}` + 처리 통계. GT와의 오차(위치 [cm])도 함께 산출해 기록
  (지각 자체 품질 리포트 — 열화 지도와 별개의 1차 진단).
- 실행: 컨테이너 안 python3.8. 프로젝트 import 없음(standalone). rsync로 파일 왕복.

### 4.3 Pass 3 — CameraFileProvider (`providers.py`)

```python
class CameraFileProvider:
    """Pass 2 pose 파일을 읽어 freeze-once 주입. seam 규격 동일."""
    def __init__(self, poses_path, frames_meta_path): ...
    # 로드 시: T_local_obj = T_local_cam @ T_cam_obj → pos_local (3,)
    # meta 그리드 인자 vs 현재 실행 인자 대조(불일치 → ValueError)
    def on_reset(self, env, env_ids): ...   # 파일 기반이라 캡처 없음(고정 버퍼)
    def get_override(self, env): ...        # [N,3] 반환 (StateFrozen과 동일 계약)
```

- **FP 실패 env 처리(결정)**: 배포 현실 반영 — pose 없으면 파지 시도 불가 = 실패.
  실패 env는 평가를 돌리지 않고 해당 셀에 `success=False, perception_fail=True`
  에피소드로 계상한다. `report.py`에 `perception_fail_rate` 컬럼 추가 —
  "지각 때문에 죽은 셀"과 "정책 때문에 죽은 셀"이 히트맵에서 구분되게.
  (구현: provider가 실패 env 목록을 노출 → eval_sim2real이 그 env의 에피소드를
  스텝 없이 실패로 기록. 전 에피소드 동일 스폰이므로 결정론적.)
- `--pose_source camera_frozen` + `--poses <json> --frames_meta <json>` 인자 추가.
  make_provider의 NotImplementedError 자리를 교체.

### 4.4 델타 맵

- 입력: STATE summary.json + CAMERA summary.json (동일 그리드 검증).
- 출력: `delta_success.csv` + `heatmap_delta_success.png` (Δ=STATE−CAMERA,
  발산 컬러맵, perception_fail 셀 별도 표기) + FP 위치오차 통계(Pass 2 기록 활용).

## 5. 에러 처리

- 전 pass 공통: 그리드 메타 불일치·NPZ 키 누락·mesh 부재 → 즉시 명확한 에러.
- Pass 2: env별 실패는 기록하고 계속(전체 중단 아님), 통계에 사유별 집계.
- Pass 3: poses 파일의 env 수 ≠ num_envs → 에러. 비유한 pose → 해당 env FAIL 강등.

## 6. 검증·테스트

정적(CPU): 좌표변환 수학(합성 T로 왕복 검증), CameraFileProvider(가짜 poses/meta
파일), perception_fail 집계, 델타 맵 산술, meta 대조 fail-fast. 기존 58 테스트 회귀.

게이트(순서대로, 각 단계 사용자 확인):
1. **프리뷰 게이트**: render_pass `--preview` PNG로 tilt 각·시야 확인 (사용자 승인)
2. Pass 1 full (200장/팔) → NPZ 무결성 스크립트 검증
3. Pass 2 (vision-3090) → FP 성공률·GT 오차 분포 1차 보고 (여기서 오차가 크면
   Pass 3 전에 원인 규명 — mesh 단위/depth 스케일이 1순위 용의자)
4. Pass 3 + 델타 맵 → 최종 보고·Notion 기록

## 7. 위험

1. **mesh 단위/스케일 불일치** (visdex USD는 metersPerUnit 이슈 전력) — Pass 2에서
   mesh bbox 크기 로그로 즉시 탐지.
2. **depth 스케일**: Isaac depth[m] vs FP 기대 단위 — smoke 1장으로 선검증.
3. **TiledCamera 200 env 메모리**: 200×(H,W) RGB-D — grasp_v2 distillation에서 검증된
   규모. 문제 시 렌더를 배치 분할.
4. **seg 마스크 prim 매칭**: MultiAsset이라 Cup prim 경로가 env마다 동일 패턴인지
   확인 필요.
5. **head 카메라 시야 한계**: 그리드 극단 셀이 FOV 밖일 수 있음 — 그것 자체가 측정
   결과(perception_fail로 드러남). tilt 각은 중심이 아닌 **그리드 전체**가 최대한
   들어오게 선정.
6. **좌우 USD 차이**: bi_rl(left)에도 head pan/tilt가 동일한지 구현 시 확인.

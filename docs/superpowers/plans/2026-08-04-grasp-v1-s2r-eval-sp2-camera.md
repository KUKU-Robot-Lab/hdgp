# grasp_v1 s2r 평가 SP2 (camera_frozen) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** D435i 헤드 카메라 렌더 → FoundationPose++ 6D 추정 → freeze-once 주입으로 CAMERA 그리드 히트맵 + STATE−CAMERA 지각 열화 지도 산출 (오프라인 3-pass).

**Architecture:** 스펙 `docs/superpowers/specs/2026-08-04-grasp-v1-s2r-eval-sp2-camera-design.md`. Pass1 렌더(Isaac)→NPZ, Pass2 FP배치(vision-3090 컨테이너, standalone)→poses.json, Pass3 CameraFileProvider(기존 seam)→평가. SP1 하네스(`scripts/eval_s2r/`) 확장.

**Tech Stack:** Isaac Lab TiledCamera + pxr(USD mesh 추출), numpy/imageio(fp_batch, py3.8 호환), 기존 pytest 스위트.

## Global Constraints

- 학습 env/기존 play 무영향: source/openarm/** 수정 금지(SP1 훅 그대로 사용). 카메라·semantic tag는 render_pass 런타임에서 env_cfg 객체를 프로그램적으로 변경(파일 무수정).
- `fp_batch.py`는 **py3.8 호환 + 프로젝트/torch import 금지**(numpy, json, argparse, pathlib 수준; FP import는 함수 내부 지연 import).
- 기존 정적 테스트 58개 회귀 금지. 새 순수 로직은 CPU 테스트 필수.
- GPU/컨테이너 실행은 전부 사용자 게이트(스펙 §6의 4게이트) — 이 계획의 태스크는 정적 작업만.
- 커밋: hdgp repo, 브랜치 pour, 한국어 conventional commits, --no-verify 금지.
- env 인덱스 정렬 계약: Pass1/3은 동일 그리드 인자 필수 — meta 대조 fail-fast.

**확정 사실 (구현자가 그대로 사용):**
- D435i 파라미터: `source/openarm/openarm/tesollo/right/grasp_v2/grasp_right_preset.py:352-359` — `CAMERA_IMG_WIDTH=320, HEIGHT=180, FOCAL_LENGTH(파일서 확인), HORIZONTAL_APERTURE=37.9586(HFOV 87°), CLIPPING_RANGE=(0.3,3.0)`. CAMERA_POS/ROT(월드 고정)는 사용하지 않음 — SP2는 헤드 장착.
- 헤드 카메라 체인(sensor_rl URDF :918-1330): `head_base→head_mid(head_j_pan)→head_camera(head_j_tilt)→head_cam_view(fixed)`. 카메라 부착 링크 = `head_cam_view`. actuator group "head_camera"(`head_j_(pan|tilt)`)는 grasp_v1 cfg에 기존재(:419-420). left(bi_rl USD)의 대응 링크명은 구현 시 확인.
- 물체 8종(`grasp_right_env_cfg.py:66-73`, env_id%8 순서): cup_big×4(scale 0.85/1.00/1.15/1.30) + shaker_body + large_5_cyl + large_8_cyl_h12(1,1,1.5) + large_12_cyl_h12(1,1,2.4). **obj mesh 없음 → USD에서 추출 필요, 비균일 스케일 베이크 필수.**
- Cup prim: `/World/envs/env_.*/Cup` (`grasp_right_env_cfg.py:527`).

---

### Task 1: transforms.py — 좌표변환·intrinsics 순수 모듈

**Files:**
- Create: `scripts/eval_s2r/transforms.py`
- Test: `scripts/eval_s2r/tests/test_transforms.py`

**Interfaces:**
- Produces:
  - `compose_local_pose(T_local_cam: np.ndarray(4,4), T_cam_obj: np.ndarray(4,4)) -> np.ndarray(3,)` — env-local 물체 위치
  - `k_from_pinhole(focal_length, horizontal_aperture, width, height) -> np.ndarray(3,3)` — Isaac 핀홀→K (fx=fy=focal/h_aperture*width, cx=w/2, cy=h/2)
  - `pinhole_from_k(K, width, height) -> tuple[focal_length, horizontal_aperture]` — 실기 camera_info K→Isaac 파라미터 (왕복 일관)
  - `validate_T(T) -> None` — (4,4)·유한·회전부 직교(±1e-3) 아니면 ValueError

- [ ] **Step 1: 실패 테스트 작성** — `test_transforms.py`:

```python
import numpy as np
import pytest

from scripts.eval_s2r.transforms import (
    compose_local_pose, k_from_pinhole, pinhole_from_k, validate_T,
)


def _rt(rot, trans):
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T


class TestCompose:
    def test_identity_camera(self):
        # 카메라가 local 원점·무회전이면 T_cam_obj 평행이동이 그대로 local 위치
        p = compose_local_pose(np.eye(4), _rt(np.eye(3), [0.1, 0.2, 0.3]))
        assert np.allclose(p, [0.1, 0.2, 0.3])

    def test_rotated_camera_roundtrip(self):
        # 합성 검증: 알려진 local 물체 위치 → cam frame으로 보낸 뒤 복원
        Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        T_local_cam = _rt(Rz, [0.5, 0.0, 0.7])
        p_local_gt = np.array([0.3, -0.1, 0.05])
        T_cam_local = np.linalg.inv(T_local_cam)
        p_cam = (T_cam_local @ np.append(p_local_gt, 1.0))[:3]
        p = compose_local_pose(T_local_cam, _rt(np.eye(3), p_cam))
        assert np.allclose(p, p_local_gt, atol=1e-9)

    def test_invalid_T_raises(self):
        bad = np.eye(4); bad[0, 0] = 2.0  # 비직교 회전부
        with pytest.raises(ValueError):
            compose_local_pose(bad, np.eye(4))
        with pytest.raises(ValueError):
            compose_local_pose(np.eye(4), np.full((4, 4), np.nan))


class TestIntrinsics:
    def test_k_from_pinhole_center(self):
        K = k_from_pinhole(18.14756, 37.9586, 320, 180)
        assert K[0, 2] == pytest.approx(160.0) and K[1, 2] == pytest.approx(90.0)
        assert K[0, 0] == pytest.approx(18.14756 / 37.9586 * 320)
        assert K[1, 1] == pytest.approx(K[0, 0])  # 정방 픽셀

    def test_roundtrip(self):
        K = k_from_pinhole(20.0, 30.0, 640, 360)
        f, ap = pinhole_from_k(K, 640, 360)
        K2 = k_from_pinhole(f, ap, 640, 360)
        assert np.allclose(K, K2)
```

- [ ] **Step 2: 실패 확인** — `python3 -m pytest scripts/eval_s2r/tests/test_transforms.py -q` → ImportError FAIL.
- [ ] **Step 3: 구현** — `transforms.py`:

```python
"""SP2 좌표변환·intrinsics 순수 함수 (numpy only, Isaac 무관).

설계: docs/superpowers/specs/2026-08-04-grasp-v1-s2r-eval-sp2-camera-design.md §4.3
"""
from __future__ import annotations

import numpy as np


def validate_T(T: np.ndarray) -> None:
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f"invalid transform: shape={T.shape}, finite={np.isfinite(T).all()}")
    R = T[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-3):
        raise ValueError("rotation block not orthonormal")


def compose_local_pose(T_local_cam: np.ndarray, T_cam_obj: np.ndarray) -> np.ndarray:
    """env-local 물체 위치 = (T_local_cam @ T_cam_obj) 평행이동부."""
    validate_T(T_local_cam)
    validate_T(T_cam_obj)
    return (np.asarray(T_local_cam, dtype=float) @ np.asarray(T_cam_obj, dtype=float))[:3, 3].copy()


def k_from_pinhole(focal_length: float, horizontal_aperture: float,
                   width: int, height: int) -> np.ndarray:
    """Isaac PinholeCameraCfg → K. Isaac은 정방픽셀(수직 aperture는 종횡비 유도)."""
    fx = float(focal_length) / float(horizontal_aperture) * float(width)
    return np.array([[fx, 0.0, width / 2.0],
                     [0.0, fx, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=float)


def pinhole_from_k(K: np.ndarray, width: int, height: int) -> tuple[float, float]:
    """실기 camera_info K → (focal_length, horizontal_aperture). fx 기준(정방픽셀 근사)."""
    K = np.asarray(K, dtype=float)
    if K.shape != (3, 3) or K[0, 0] <= 0:
        raise ValueError(f"invalid K: {K}")
    # focal/aperture 는 비율만 의미 → aperture 를 관례값 20.955(mm)로 두고 focal 역산
    aperture = 20.955
    focal = K[0, 0] / float(width) * aperture
    return focal, aperture
```

- [ ] **Step 4: 통과 확인** → PASS. **Step 5: 커밋** — `feat: eval_s2r SP2 좌표변환·intrinsics 모듈`

---

### Task 2: CameraFileProvider (providers.py 확장)

**Files:**
- Modify: `scripts/eval_s2r/providers.py`
- Test: `scripts/eval_s2r/tests/test_camera_provider.py`

**Interfaces:**
- Consumes: Task 1 `compose_local_pose`
- Produces:
  - `CameraFileProvider(poses_path: str, frames_meta_path: str)` — 로드 시 poses.json+meta.json 파싱, env별 `pos_local[N,3]` 텐서 구성, 실패 env는 `failed_envs: set[int]` 노출. `expected_grid: dict` 노출(meta의 그리드 인자 — Pass 3 대조용)
  - `on_reset(env, env_ids)` → no-op (파일 고정), `get_override(env)` → [N,3] 텐서(clone) — **실패 env 행은 NaN**(주입되면 env 훅 이전에 걸러져야 하므로 eval 쪽에서 해당 env를 아예 돌리지 않는 것이 계약, Task 5)
  - `make_provider("camera_frozen", poses_path=..., frames_meta_path=...)` — NotImplementedError 제거, 인자 없으면 ValueError

**poses.json 스키마(Task 4 fp_batch가 생산, 여기서 소비 — 계약 고정):**
```json
{"robot": "right", "num_envs": 200,
 "poses": {"0": {"ok": true, "T_cam_obj": [[...4x4...]]},
            "7": {"ok": false, "reason": "register_failed"}}}
```
**meta.json 스키마(Task 3 render_pass가 생산 — 계약 고정):**
```json
{"robot": "right", "num_envs": 200, "grid": {"x_min":0.15,"x_max":0.39,"nx":5,
  "y_min":-0.22,"y_max":0.02,"ny":5,"repeats":8}, "head_tilt": -0.6,
 "T_local_cam": {"0": [[...4x4...]], ...}, "git_sha": "...", "k_source": "nominal|<yaml>"}
```

- [ ] **Step 1: 실패 테스트 작성** — tmp_path에 위 스키마 그대로 가짜 poses/meta json 생성 → 검증:
  - ok env의 pos_local이 `compose_local_pose(T_local_cam[i], T_cam_obj[i])`와 일치
  - fail env → `failed_envs`에 포함 + override 행 NaN
  - meta num_envs ≠ poses num_envs → ValueError
  - `expected_grid`가 meta grid dict 그대로
  - `make_provider("camera_frozen")` 인자 누락 → ValueError, 정상 인자 → CameraFileProvider
  - 비유한 T_cam_obj(ok=true인데 NaN) → 해당 env FAIL 강등 + 사유 "nonfinite"
  (테스트 코드는 위 스키마를 리터럴로 작성 — 구현자가 스키마를 임의 변경하면 테스트가 잡는다)
- [ ] **Step 2: 실패 확인** → FAIL. **Step 3: 구현** — json 로드→numpy 변환→`torch.tensor` [N,3] 버퍼(실패행 NaN)·failed_envs 구성. get_override는 `.clone()` 반환(StateFrozen과 동일 방어). 주석 한국어.
- [ ] **Step 4: 전체 스위트** — 기존 58 + 신규 전부 PASS. **Step 5: 커밋** — `feat: eval_s2r CameraFileProvider(FP pose 파일 주입, 실패 env 추적)`

---

### Task 3: render_pass.py — Pass 1 렌더 (Isaac 엔트리)

**Files:**
- Create: `scripts/eval_s2r/render_pass.py`
- 검증: 문법 + 기존 스위트 회귀 (GPU 실행은 게이트)

**Interfaces:**
- Consumes: SP1 `grid.py`(GridSpec/build_cells/build_spawn_tensor), env 훅 `eval_fixed_spawn_local`, Task 1 `k_from_pinhole/pinhole_from_k`
- Produces: `frames/<robot>/env_<i>.npz`(스펙 §4.1 표 그대로: rgb/depth/mask/K/T_local_cam/gt_obj_pos_local/gt_obj_rot_wxyz/obj_idx/cell_idx/cell_x/cell_y) + `frames/<robot>/meta.json`(Task 2 스키마) + `frames/<robot>/object_map.json`(obj_idx→{usd_path, scale, id} — env_cfg의 `_ACTIVE_OBJECT_SPECS`를 import해 그대로 덤프) + `--preview` 시 `preview_env<i>.png`

**구현 지침:**
- CLI: `--robot {left,right} --grid_x/.../--grid_repeats`(eval_sim2real과 동일 규약+검증 재사용) `--head_tilt RAD`(기본 0.0) `--head_pan RAD`(기본 0.0) `--camera_info YAML`(선택) `--out frames/<robot>` `--preview`(중심셀 1 env만·PNG 저장 후 종료) `--headless`.
- env 생성: eval_sim2real의 `_setup` 중 env-cfg 부분만 필요(체크포인트/agent 불필요) — `parse_env_cfg` → ADR off·demo off(eval_sim2real.py의 해당 블록과 동일 코드, `# eval_sim2real.py:<라인> 패턴 복제` 주석) → **추가로**:
  - `env_cfg.cup_cfg.spawn.assets_cfg[*].semantic_tags = [("class", "graspobj")]` — MultiAsset 각 서브 cfg에 프로그램적 주입(파일 무수정). instance seg 대신 semantic seg "graspobj" 클래스로 마스크 추출.
  - TiledCamera cfg 구성: `prim_path="/World/envs/env_.*/Robot/head_cam_view/Camera"`(left는 USD 확인 후 동일 패턴), `data_types=["rgb", "distance_to_image_plane", "semantic_segmentation"]`, `colorize_semantic_segmentation=False`, 핀홀 파라미터=grasp_v2 preset D435i 상수(`--camera_info` 시 `pinhole_from_k` 환산으로 대체), width/height 동일 소스. offset=identity(링크 프레임 그대로; ros convention).
  - TiledCamera를 env_cfg에 넣는 방식: grasp_v1 cfg에는 카메라 필드가 없으므로 **scene 생성 후 별도 센서로 추가**가 불가한 경우 env_cfg에 동적 속성 추가가 아니라 `TiledCamera(cfg)` 직접 생성 + `env.scene.sensors` 등록 패턴(grasp_v2 `grasp_right_env.py:566-571`의 생성 코드를 표본으로 복제) — 구현자는 grasp_v2 코드를 열어 동일 API로 작성.
- 실행 흐름: env.reset()(그리드 훅으로 스폰) → head pan/tilt 목표를 `robot.set_joint_position_target`으로 명령(+`write_data_to_sim`) → N_settle(기본 30) 물리+렌더 스텝(카메라 노출 안정)→ 카메라 `update()` → env별 데이터 취득:
  - `T_local_cam`: 카메라 prim의 world pose(`camera.data.pos_w/quat_w_ros`) − `env_origins` → 4×4 조립.
  - mask: semantic_segmentation 출력에서 "graspobj" ID 채널 == True.
  - depth: `distance_to_image_plane` [m] 그대로.
- 저장: env당 npz(uint8/float32 캐스팅 명시) + meta.json + object_map.json. `--preview`는 rgb에 mask 외곽선 오버레이 PNG 1장(+ tilt 각 출력) 저장 후 종료.
- 검증(정적): `python3 -c "import ast; ast.parse(open('scripts/eval_s2r/render_pass.py').read()); print('SYNTAX OK')"` + 스위트 회귀.
- [ ] Step 1 스크립트 작성 → Step 2 문법 OK → Step 3 스위트 회귀 → Step 4 커밋 — `feat: eval_s2r SP2 렌더 pass(D435i 헤드캠 NPZ 캡처·preview)`

---

### Task 4: fp_batch.py — Pass 2 FP 배치 (standalone)

**Files:**
- Create: `scripts/eval_s2r/fp_batch.py`
- Test: `scripts/eval_s2r/tests/test_fp_batch_prep.py` (FP 미의존 부분만)

**Interfaces:**
- Consumes: Task 3 산출(frames dir 스키마), object_map.json
- Produces: `poses/<robot>.json`(Task 2 스키마 그대로) + stderr 진행 로그 + `poses/<robot>_stats.json`(성공률·GT 위치오차 mean/median/p95[cm]·사유별 실패 수·mesh bbox 로그)

**구현 지침:**
- **py3.8 호환·프로젝트 import 금지**(Global Constraints). 상단 import는 numpy/json/argparse/pathlib/glob만. FP(`estimater` 등)와 trimesh는 `main()` 안 지연 import — 이러면 테스트가 FP 없이 순수 함수를 import 가능.
- 순수 함수(테스트 대상):
  - `load_frame(npz_path) -> dict` — 필수 키 검증(누락 → ValueError)
  - `mesh_path_for(obj_idx, object_map) -> tuple[path, scale]` — 매핑·존재 검증
  - `pose_entry(ok, T=None, reason=None) -> dict` — 스키마 생성(Task 2와 리터럴 일치)
  - `error_cm(T_cam_obj, T_local_cam, gt_pos_local) -> float` — Task 1 수학의 numpy 복제(단일 파일 자립을 위해 compose 로직 내장, `# transforms.py 와 동일 수학` 주석)
- mesh 준비: object_map의 usd_path에서 **같은 디렉토리의 .obj를 우선 탐색**, 없으면 `--mesh_dir`에서 `<id>.obj` 탐색(Task 6 export 산출물), 그것도 없으면 해당 물체 전 env FAIL(reason="mesh_missing"). scale은 trimesh로 정점에 곱해 적용(비균일 포함), **mesh bbox를 stats에 기록**(단위 오류 즉시 탐지 — 스펙 위험 1).
- FP 호출부(지연 import): 물체별 estimater 1회 생성 → env 순회 `register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=...)` → 4×4. 예외/NaN → FAIL(reason=예외명). **vision-3090 컨테이너의 기존 FP 사용 코드(perception repo fpplusplus_smoke)를 표본으로 호출 시그니처를 맞출 것** — 정확한 인자명은 컨테이너 쪽 코드가 진실.
- [ ] Step 1 순수 함수 실패 테스트(스키마 검증·매핑·pose_entry·error_cm 왕복) → Step 2 FAIL 확인 → Step 3 구현 → Step 4 PASS+문법 OK(py3.8 호환은 `python3 -c "import ast; ast.parse(...)"` + walrus/match 미사용 육안 확인) → Step 5 커밋 — `feat: eval_s2r SP2 FP 배치 스크립트(standalone, GT 오차 리포트)`

---

### Task 5: eval_sim2real 배선 — camera_frozen 실행 경로

**Files:**
- Modify: `scripts/eval_s2r/eval_sim2real.py`
- Modify: `scripts/eval_s2r/report.py` (perception_fail 컬럼)
- Test: `scripts/eval_s2r/tests/test_report.py` 확장

**Interfaces:**
- Consumes: Task 2 CameraFileProvider(failed_envs/expected_grid), Task 1
- Produces: `--pose_source camera_frozen --poses P --frames_meta M` 실행 경로; results.csv에 `perception_fail_rate` 컬럼

**구현 지침:**
- argparse: `--poses`, `--frames_meta` 추가. camera_frozen인데 누락 → parser.error(부팅 전).
- `_setup` 내 provider 생성부: camera_frozen이면 `make_provider("camera_frozen", poses_path=..., frames_meta_path=...)`; 생성 직후 `provider.expected_grid`와 현재 그리드 인자 대조 → 불일치 ValueError(fail-fast).
- **perception-fail env 처리(스펙 §4.3 결정)**: 배치 루프에서 `provider.failed_envs`에 속한 env는 스텝을 돌리지 않고, 그 env가 채웠어야 할 `episodes_per_env`개 에피소드를 `EpisodeResult(success=False, lifted=False, grip=0, disp=0, obj_idx=env%8, invalid=False, finger_contacts=(0,)*5)` + perception_fail 표시로 즉시 기록. 구현: `EpisodeResult`에 `perception_fail: bool = False` 필드 추가(**기본값 있는 신규 필드 — 기존 생성부 무수정으로 호환**), `aggregate()`에 `perception_fail_rate` 추가, CSV 필드 추가.
  - 주의: failed env도 물리 씬에는 존재(스폰됨) — 정책 스텝은 전 env 일괄이므로 "돌리지 않는다" = **기록만 안 하고 결과를 합성**한다(action은 어차피 전 env 계산됨, 해당 env 결과는 버림). 코드 주석으로 명시.
- live/state_frozen 경로 무변경(회귀: 기존 인자 조합으로 argparse 검증 로직 통과 확인).
- [ ] Step 1 report 확장 실패 테스트(perception_fail 에피소드가 success 집계에 0으로 반영+rate 컬럼) → Step 2 FAIL → Step 3 구현(report→eval 순) → Step 4 전체 스위트 PASS+문법 OK → Step 5 커밋 — `feat: eval_s2r camera_frozen 실행 경로(+perception_fail 집계)`

---

### Task 6: export_meshes.py — USD→obj 추출 도구

**Files:**
- Create: `scripts/eval_s2r/export_meshes.py`
- 검증: 문법만 (실행은 isaaclab.sh 필요 — 게이트)

**구현 지침:**
- pxr(`Usd`, `UsdGeom`) 기반: `--object_map frames/<robot>/object_map.json --out meshes/` — 각 물체 USD를 열어 Mesh prim들의 points/faceVertexIndices를 수집, `metersPerUnit` 반영, **spec scale(비균일 포함) 정점에 베이크**, 단일 obj로 병합 저장(`<id>.obj`). trimesh 없이 obj 수기 기록(v/f 라인) — 의존성 최소.
- 같은 USD 다른 스케일(cup_big×4)은 각각 별도 obj.
- bbox를 stdout에 출력(단위 검증용 — cup_big s100 예상 지름 ~7cm).
- 실행은 `isaaclab.sh -p`(pxr 포함 env). GPU 불필요하나 kit 초기화 필요 시 `--headless` AppLauncher 사용 — AppLauncher 없이 pxr 단독 import가 되면 그 편이 낫다(구현자가 확인, 안 되면 AppLauncher 추가).
- [ ] Step 1 작성 → Step 2 문법 OK → Step 3 커밋 — `feat: eval_s2r SP2 물체 USD→obj 추출 도구(스케일 베이크)`

---

### Task 7: 델타 맵 — STATE−CAMERA 리포트

**Files:**
- Create: `scripts/eval_s2r/delta_report.py`
- Test: `scripts/eval_s2r/tests/test_delta_report.py`

**Interfaces:**
- Consumes: 두 summary.json(cells 리스트 — SP1 스키마), Task 5의 perception_fail_rate
- Produces: `build_delta(cells_a, cells_b) -> list[dict]`(셀 xy 대조 후 Δsuccess/Δlifted, perception_fail_rate 병기; xy 불일치 → ValueError), `write_delta_csv`, `write_delta_heatmap`(발산 컬러맵 RdBu, vmin/vmax=±1, perception_fail>0 셀에 "P" 주석); CLI: `python3 delta_report.py --state <summary> --camera <summary> --out <dir>`

- [ ] Step 1 실패 테스트(대조 산술·xy 불일치 에러·CSV 스키마·PNG 생성) → Step 2 FAIL → Step 3 구현(report.py의 heatmap 코드 표본, Agg 백엔드) → Step 4 PASS → Step 5 커밋 — `feat: eval_s2r STATE−CAMERA 델타 맵 리포트`

---

### Task 8: README·게이트 절차 갱신

**Files:**
- Modify: `scripts/eval_s2r/README.md`

- [ ] SP2 섹션 추가: 3-pass 실행 순서(각 pass의 실제 명령 — render_pass/rsync/fp_batch(컨테이너 docker exec 형식은 "vision-3090의 perception repo 런북 참조"로 포인터만)/eval camera_frozen/delta_report), 4게이트 체크리스트(스펙 §6: ①프리뷰 tilt 확인 ②NPZ 무결성 ③FP 성공률·GT오차 ④델타 맵), 신규 플래그 표, poses/meta 스키마 예시. CLI 예시의 플래그는 실제 argparse와 대조(“존재하지 않는 플래그 문서화 금지” — SP1 Task 8 교훈).
- [ ] 커밋 — `docs: eval_s2r SP2 3-pass 사용법·게이트 체크리스트`

---

## Self-Review 결과

- **스펙 커버리지**: §4.1 렌더(T3)+intrinsics(T1·T3), §4.2 FP배치(T4)+mesh(T6), §4.3 provider·실패처리(T2·T5), §4.4 델타(T7), §5 에러(각 태스크 fail-fast), §6 정적 테스트(T1/2/4/5/7)·게이트(T8 문서화). mesh 부재 발견(계획 수립 중 확정) → T6 신설로 해소.
- **자리표시자 없음**: Isaac/FP 종속 코드는 SP1 Task 6 방식(표본 파일 지정+"주석 그대로면 미완" 규칙)으로 처리, 순수 모듈은 전체 코드 제공.
- **타입 일관성**: poses/meta 스키마를 T2에 리터럴로 고정하고 T3(생산)·T4(생산)·T5(소비)가 동일 참조. EpisodeResult 신규 필드는 기본값으로 하위호환.

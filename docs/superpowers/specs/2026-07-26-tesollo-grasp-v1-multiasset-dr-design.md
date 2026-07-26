# tesollo/right/grasp_v1 — MultiAsset + Domain Randomization 설계

- 날짜: 2026-07-26
- 대상: `source/openarm/openarm/tesollo/right/grasp_v1`
- 브랜치: `pour`

## 배경 / 목표

현재 tesollo/right/grasp_v1은 단일 컵(`assets/cup/cup_big_sdf.usd`) 하나만 스폰해
학습한다 (`self.cup = RigidObject(self.cfg.cup_cfg)`, prim `/World/envs/env_.*/Cup`).
단일 컵 시너지 그립은 이미 ~98% 성공으로 검증됨(메모리 `grasp-v1-synergy-success`).

목표: **다양한 컵 크기·마찰·무게 + 단순 세로원통 형상**을 side-to-side 접근으로
일반화 학습. reward/성공판정 구조는 유지(최소 변경). rh56f1/right/grasp_v1이 이미
보유한 MultiAsset 인프라를 tesollo에 이식하되, tesollo 고유 요구(세로원통, 위치 ADR)를 반영.

## 핵심 실측 근거 (grasp-v2 tesol/left/lstm_test1, ADR50 완주)

| 물체 | 직경×높이 | 성공률 | 판정 |
|---|---|---|---|
| large_8_cyl | 8×8cm | 0.774 | 최안정(정입방체형) |
| large_12_cyl | 12×5cm | 0.686 | 하위(납작·광폭) |
| large_5_cyl | 5×12cm | 0.658 | 하위(세장) |
| cup_big | 9×17.8cm | 0.259 | grasp-v2 최악권 |
| small_5/8/12_cyl | — | 데이터 없음 | grasp-v2에서 스폰 제외(STAGE2 보류) — 실패가 아니라 미검증 |

- cup_big은 grasp-v2(DEXTRAH lift-to-goal)에선 최악이나 grasp_v1(시너지+latch lift)에선
  98% — 태스크 메커니즘 차이. **grasp_v1은 세로 긴 컵에 강함.**
- small 계열은 실패 데이터가 아니라 미학습. tesollo 큰 손엔 물리적으로 미달.
- **사용자 결정: cyl 높이는 최소 9cm 이상(세로원통)이어야 함.** 현재 cyl 중 높이≥9cm는
  large_5_cyl(12cm)뿐 → large_8/12는 z-scale로 높이 12cm화. small 전부 탈락.

## 물체 구성 — 총 8종 (onehot 8차원)

| # | 물체 | 크기 | 구현 |
|---|---|---|---|
| 1–4 | cup_big × 4 scale (0.85 / 1.0 / 1.15 / 1.30) | 직경 7.7~11.7cm × 높이 15~23cm | 동일 USD, `UsdFileCfg(scale=)` 등방 스케일 |
| 5 | cocktail shaker_body | 직경 8.8 × 높이 17.5cm | `assets/cocktail/usd/shaker_body.usda` (metersPerUnit=1 정상) |
| 6 | cyl 직경 5cm | 높이 12cm | large_5_cyl 그대로 |
| 7 | cyl 직경 8cm | 높이 12cm | large_8_cyl `scale=(1,1,1.5)` (z 8→12cm) |
| 8 | cyl 직경 12cm | 높이 12cm | large_12_cyl `scale=(1,1,2.4)` (z 5→12cm) |

- 자산 root: cup_big·cyl은 `assets/visdex_objects/USD/{name}/{name}.usd`,
  shaker_body는 cocktail 경로 직접 지정.
- 배정: `env_id % 8` (rh56f1과 동일, 결정론적). scale은 각 asset entry에 고정
  (MultiAsset은 스폰 시점 scale 고정 제약).
- 크기 상한 검증: cup_big×1.30(직경 11.7cm)·직경12 cyl은 tesollo aperture 상한 근처
  → 설계 시 육안 검증, 필요 시 scale 하향.

## Domain Randomization

### 마찰 / 무게 (신규 EventCfg)
- static / dynamic friction ∈ [0.5, 1.2] — `randomize_rigid_body_material`
- mass 배율 ∈ [0.7, 1.3] — `randomize_rigid_body_mass` (scale 모드)
- 매 reset per-env 연속 랜덤. MultiAsset의 각 물체(`SceneEntityCfg("cup")`)에 적용.
- 현재 tesollo grasp_v1엔 EventCfg 없음 → 신규. rh56f1/grasp_v2의 DEXTRAH physics DR
  EventCfg 패턴 참조.

### 위치 xy (ADR 신규 스케줄)
- 현재: `object_spawn_x_center=0.27, y_center=-0.10, xy_range=0.06`(±6cm 고정 랜덤).
  ADR(`enable_adr=True, num_increments=50`)은 있으나 `adr_custom_cfg`가 빈 dict이고
  `get_param()` 호출처 0건 → **실질적으로 아무 파라미터도 스케줄하지 않음**.
- 변경: `object_spawn_xy_range`를 ADR 스케줄 대상으로.
  - `adr_custom_cfg`에 `{"spawn": {"xy_range": (0.02, 0.08)}}` 등록 (initial→final).
  - reset에서 `grasp_adr.get_param("spawn", "xy_range")`로 현재 범위 읽어 스폰 xy 샘플.
  - **±2cm → ±8cm 점진 확대**. approach reward가 좁은 위치서 안정 학습 후 넓힘.
- pregrasp arm 캐시(현재 13×13 grid, ±6cm 커플링)는 **최대 ±8cm 기준 grid로 생성**
  → ADR가 range를 좁혀도 캐시가 항상 커버. grid 해상도(간격 1cm) 유지 시 17×17.
- yaw(z회전) 랜덤: 물체군 전부 축대칭(원통·컵·shaker) → **미적용**.

## 유지 (최소 변경 원칙)

- reward / 성공판정 구조 전면 유지 (cup tipping / lin·ang vel threshold / upright).
- side-to-side approach 로직 유지: `approach_dir`, `fingertip_side_dist`는 물체 pos
  기준 계산 → 물체 다양화에도 자동 대응.

### 불가피한 예외 2건 (MultiAsset 이식상 필연, 구현 중 확정)

- `scene.replicate_physics: True → False` — MultiAsset(env별 다른 물체)은 physics
  복제 불가. rh56f1 동일.
- `enable_demo_grasp_reset: True → False` — demo pose는 단일 컵 전용 고정 자세라
  8종(높이·위치 ADR)에 부적합. off 시 기존 FABRICS pregrasp cache 경로 사용
  (rh56f1/grasp_v2 "demo-free reset" 동일 결정). 재학습이 필연이므로 warmstart 손실 무관.

## per-object 처리 (reward 아님, 물리 안착)

크기 의존 스칼라 상수를 per-object 텐서로 전환:
- `object_spawn_z`(0.297) — 물체 높이에 맞춰 테이블 위 안착 z (per-object bbox 반높이 기반).
- `cup_radius_approx`(0.045) — 접근/그립 반경 (per-object bbox 반경 기반).
- `cup_grasp_z_offset`(0.06) — 그립 높이 오프셋.

bbox 소스: `assets/object_bbox.json`. **shaker_body와 z-scaled cyl(6/7/8번)은
bbox가 없거나 scale 반영 필요** → 계산해 등록(스케일 적용 후 실효 bbox).
rh56f1/grasp_v1의 bbox 로딩·per-object 텐서 패턴 이식.

## Observation / 재학습 영향

- obs: `NUM_OBSERVATIONS`(106) → 106 + 8(onehot) = **114**. critic obs도 +8.
- 단일컵 체크포인트 warmstart 불가 → **scratch 재학습**(사용자 확인).

## 구현 단계 (예정)

1. 자산: shaker_body bbox 계산·등록, z-scaled cyl 실효 bbox 산출.
2. env_cfg: `_ACTIVE_OBJECT_NAMES`(8종) + `MultiAssetSpawnerCfg`(per-entry scale) +
   신규 `EventCfg`(friction/mass) + `adr_custom_cfg`(spawn xy_range) + obs 차원 +8.
3. env.py: 단일 `cup_cfg` → MultiAsset 스폰(clone→spawn 순서), `env_id % 8` 배정,
   onehot obs, per-object bbox 텐서, reset xy를 `adr.get_param` 범위로 샘플,
   pregrasp 캐시 grid ±8cm화.
4. preset/constants: side_approach 목록·onehot 차원 상수 정합.

## 검증

1. 정적: import·cfg 로드·obs 차원 일치 (로컬).
2. 서버 GPU: `env.yaml` dump로 8종 배정·friction/mass randomize·ADR xy_range 확인
   (메모리 `verify-code-before-training` 3단계).
3. play: 각 물체 스폰·테이블 안착·side approach·ADR 범위 육안 확인.
4. 재학습: server(oem), 메모리 `server-training-launch-env`.

## 참고

- 이식 원본: `source/openarm/openarm/rh56f1/right/grasp_v1` (MultiAsset·bbox·side_approach 완성본).
- 관련 메모리: `grasp-v1-synergy-success`, `grasp-v2-multiobject-plan`,
  `grasp-v2-cup-asset-fix`, `verify-code-before-training`, `server-training-launch-env`.

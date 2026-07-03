# rh56f1 fabric palm 프레임 정합 + 양팔 통합 설계

- 날짜: 2026-07-03
- 대상: rh56f1 오른팔/오른손 fabric → 실제 `r_hl_palm_sensor` 정합 + 양팔 통합 fabric 인프라
- 상태: 설계 승인됨 (구현 계획 대기)

---

## 1. 배경 · 문제

rh56f1 fabric IK(`OpenArmRh56f1PoseFabric`)가 attract하는 palm 프레임이 실제 손바닥 센서 링크
`r_hl_palm_sensor`(USD/참고 URDF, "+z가 손바닥 중심")와 **위치·자세 모두 어긋나** 있다.

원인: `generate_openarm_rh56f1_urdf.py`가 **Tesollo 손의 palm 가상프레임**(`palm_link` + `palm_x/y/z(±)`)을
그대로 재사용하고, z 위치만 "≈0.133으로 비슷"하다고 보고 xy·자세는 검증하지 않았다.

### 정량 근거 (link7 == r_al_7 가정, FK 계산)

| 비교 | 위치 차 | 자세 차 |
|------|--------|--------|
| 실제 `r_hl_palm_sensor` ↔ fabric IK 원점 `palm_link` | **3.4 cm** | **+z축 90°, +x축 90°** |
| 실제 `r_hl_palm_sensor` ↔ `palm_center` | 3.6 cm | — |

- 실제 `r_hl_palm_sensor` 로컬 +z = `[0,-1,0]` (palm_link 로컬 기준)
- fabric `palm_link` 로컬 +z = `[0,0,1]`
- → 두 프레임 축이 완전히 **90° 틀어짐**.

### 영향

`_fabric_palm_pose_from_sensor_target`는 위치만 offset(`_PALM_SENSOR_OFFSET_IN_FABRIC_PALM=(0,0.03,0.04)`,
사실 Tesollo palm_link→palm_center offset)하고 **orientation euler는 palm_sensor 값을 그대로 palm_link에
적용**한다. 결과적으로 정책이 지정한 손바닥 자세와 실제 손 자세가 90° 다르다.

정책이 "손바닥(+z)을 컵으로" 지정해도 fabric은 90° 돌아간 자세로 IK를 푼다 → envelope 그립이 안 되고
fingertip pinch로 수렴하던 **더 근본적인 원인**일 수 있다. (관측·reward는 올바른 palm_sensor 프레임을
쓰지만, 그 목표를 실현하는 fabric IK가 어긋난 프레임이라 상충)

또한 generate 스크립트 소스는 **구 네이밍**(`rh56f1_right_right_*`, `hdgp/assets/openarm_bi_rh56f1/`)인데
USD/참고본은 `_rl` 네이밍(`r_hl_*`, `hdgp/assets/robot/openarm_bi_rh56f1_rl/`)이다. 손 기하는 순수
리네임이라 동일함을 확인(예: thumb_1 origin 일치).

---

## 2. 목표

1. 정책이 지정하는 palm pose가 **실제 `r_hl_palm_sensor`(+z=손바닥 중심)를 정확히 제어**하도록 정합.
   정책·관측·IK가 전부 동일 palm_sensor 프레임 → offset 변환 소멸.
2. **왼팔+왼손도 동일하게 fabric IK로 제어 가능**하도록 양팔 통합 fabric 인프라 구축
   (env action 확장·양손 학습은 **이번 범위 밖** — 인프라만).

---

## 3. 접근 (승인: A — palm_sensor 직접 IK, 양팔 통합)

### 3.1 데이터 흐름

```
정책 6D pose (palm_sensor 기준, offset 없음)
  → convert_transform_to_points → palm_sensor 원점 + 6축점(±x/y/z @0.25m) target
  → RobotFrameOriginsTaskMap[r_hl_palm_sensor, ps_x, ps_x_neg, ...] attract
  → IK → 실제 r_hl_palm_sensor 가 target pose 도달
관측/reward: Isaac USD body_pos_w[r_hl_palm_sensor]  ← 이미 동일 프레임
```

fabric의 `convert_transform_to_points`는 `_palm_pose_target`(12D: 3 pos + 9 rot)을 원점 + 6축점(0.25m)으로
펼쳐 위치·자세를 IK로 실현한다. 접근 A는 **이 7-프레임 세트를 `r_hl_palm_sensor`에 부착**하는 것.

### 3.2 양팔 통합

단일 fabric이 cspace **26 DOF**(`r_arm7 + r_hand6 + l_arm7 + l_hand6`)를 풀고, **양쪽 palm attractor**
(`r_hl_palm_sensor` + `l_hl_palm_sensor` 각각 축점 세트)로 두 손바닥을 독립 target에 정합. 양팔이 한 cspace라
상호 충돌회피 IK가 자연히 가능.

참고 URDF는 이미 bi라 양쪽 palm_sensor 실기하 확보:
- `r_hl_palm_sensor`: parent `r_hl_palm_2`, origin `(0.01594, -0.00135, 0.07375)` rpy(90°,0,90°)
- `l_hl_palm_sensor`: parent `l_hl_palm_2`, origin `(0.01594, -0.00135, 0.07375)` rpy(90°,0,90°) (대칭 동일)

### 3.3 "인프라만"의 의미

env의 **action 차원 불변**. 오른팔은 지금처럼 정책이 palm pose를 정하고, **왼팔은 고정 중립 palm target을
상수로 fabric에 공급**해 IK로 그 자세를 유지. 나중에 왼팔 target을 action에 연결하면 곧바로 양손 태스크.

---

## 4. 컴포넌트별 변경

### (A) fabric URDF 재생성 — `generate_openarm_rh56f1_urdf.py`
- 소스: `openarm_bi_rh56f1_rl.urdf`(USD 정렬, r_hl_* / l_hl_* 실체) — 구본 대체.
- 팔: 양팔 arm 체인 포함(오른팔은 기존 Tesollo 공유 체인과 동일 기하 유지).
- 손: `link7 → r_hl_base`(참고본 마운트 origin) → r_hl 손 체인(실기하) → **`r_hl_palm_sensor`** 실체 링크.
  왼쪽도 대칭(`l_hl_*`).
- Tesollo palm 가상프레임(`palm_link`/`palm_x`…) **제거**, 대신 각 `*_hl_palm_sensor` 자식으로 축점 6개
  (`ps_x/y/z(±)` @0.25m 로컬축) 신규 — 좌우 각각.
- fingertip FK 5프레임·충돌구는 r_hl/l_hl 말단 링크 기준으로 (좌우).

### (B) fabric 코드 — `openarm_rh56f1_pose_fabric.py`
- cspace 13 → **26 DOF**. joint-limit·FK 프레임 양팔.
- `add_palm_points_attractor`를 좌우 2개(`palm_r`/`palm_l` taskmap), `control_point_frames`를
  `r_hl_palm_sensor`+축점 / `l_hl_palm_sensor`+축점.
- `_palm_pose_target` 좌우 2세트, `convert_transform_to_points`/`get_palm_pose` 좌우.
- `default_palm_euler`를 palm_sensor 로컬 기준으로 재계산(좌우).

### (C) fabric params — `openarm_rh56f1_pose_params.yaml`
- `collision_sphere_frames`/`collision_link_prefix_pairs`의 `palm_link` → palm_sensor 대응(좌우).
- 양팔 간 충돌쌍 추가(팔끼리 부딪힘 방지). attractor gain은 FK 스케일 동일이라 유지.

### (D) env 재보정 — `grasp_v1/grasp_right_env.py`, `pour_v1/pour_right_env.py`
- `_fabric_palm_pose_from_sensor_target` → **항등화 또는 제거**, `_PALM_SENSOR_OFFSET_IN_FABRIC_PALM` 삭제.
- `pregrasp_offset`·`palm_pose_mins/maxs`·`_apply_upright_palm_orientation_correction`의 euler 규약을
  **palm_sensor 기준으로 재보정**(90° 규약 변경분 반영). ← 정밀 FK 산출 필요.
- `fabric_q` 인덱싱 26 DOF로 확장. 왼팔/왼손을 fabric 출력으로 로봇에 반영(현재 `left_arm_zero_pos` 직접
  고정 → fabric IK 유지로 대체). **action 차원은 불변**(왼팔 target=고정 중립 상수).

---

## 5. 검증 전략 (3단계 게이트, 좌우 both)

1. **정적 FK 정합**: 임의 cspace q에서 `fabric FK(palm_sensor pose)` ↔ `Isaac USD body pose(palm_sensor)`를
   위치 <2mm·자세 <1° 로 일치 확인(좌우 각각). 팔 동일·손 리네임 전제 검증.
2. **IK 왕복**: target palm_sensor pose N개 → fabric step 수렴 → 실제 palm_sensor가 target 도달
   (위치/자세 오차 임계 이내, 좌우).
3. **reset/pregrasp 육안**: play.py로 초기 자세 렌더 → 손바닥(+z)이 컵을 향하는지.

---

## 6. 영향 · 리스크 · 롤백

- **재학습 필수**: grasp_v1·pour_v1 기존 정책·warmstart의 palm orientation 규약 무효화(감수 승인).
- **tesollo 무영향**: 별도 fabric(`openarm_tesollo_pose_fabric.py`)·별도 env.
- **리스크 1 (euler 재보정 오류)**: pregrasp/upright euler 재보정이 틀리면 IK가 손을 엉뚱한 자세로 →
  검증 게이트 1·3에서 차단.
- **리스크 2 (속도)**: 통합 fabric cspace 2배(13→26)라 `num_envs=2048` 학습 스텝 다소 느려질 수 있음.
  왼팔 고정이라 IK 부하는 제한적이나 실측 확인 필요.
- **롤백**: 구 URDF/generate 스크립트/offset 로직을 git 보존, 한 커밋으로 되돌림.

---

## 7. 범위 밖 (명시)

- env action 차원 확장(양팔 palm pose + 양손 finger) 및 실제 양손 조작 학습.
- 오른손 palm-first envelope reward 튜닝(별도 작업 — 이 fabric 정합이 선결과제).
- tesollo fabric/태스크.

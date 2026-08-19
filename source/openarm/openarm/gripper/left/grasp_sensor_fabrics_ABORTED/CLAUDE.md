# gripper/left/grasp_sensor — 왼팔 2지 그리퍼 단일 컵 파지

목표: `openarm_tesollo_sensor_rl` 의 **왼팔 2지 그리퍼**가 shaker 하나를 측면에서
파지·리프트·유지한다. 산출물(정책 + warm 상태)은 양팔 물붓기의 **receiver 팔**로 이어진다.

hdgp 공통 규칙(로그 먼저·증거 우선순위·코드 수정 규칙)을 그대로 따른다.
아래는 이 태스크 고유 사항만.

---

## 이 태스크가 우측(tesollo/right/grasp_sensor)과 다른 이유

같은 로봇의 **반대 손이 전혀 다른 물건**이다. 왼손은 DG-5F 20관절이 아니라
스트로크 0~0.044 m 프리즈매틱 2지 그리퍼(`l_hj_gripper_1`, `_2`는 USD PhysX mimic)다.
여기서 파생되는 차이가 전부다:

| 항목 | 우측 | 여기 |
|---|---|---|
| Action | 21D (palm 6 + 손가락 15) | **7D** (TCP 6 + 그리퍼 1) |
| obs / critic | 154 / 191 | **48 / 62** |
| Fabrics cspace | 27 (팔 7 + 손 20) | **7** (손을 fixed 로 굳힌 전용 URDF) |
| 물체 | MultiAsset 8종 + onehot | **shaker 단일**, onehot 없음 |
| grasp 품질 | 5지 감쌈(envelope/wrap) | **대향(opposition) + 압착(squeeze)** |
| 래치 게이트 | 4지 + 엄지 AND | **양 핑거 접촉 AND + hold** |

---

## 기하 제약 — 이 태스크의 모든 수치가 여기서 나온다

실측 근거는 `scripts/probes/probe_gripper_opening.py`,
`scripts/probes/probe_left_gripper_reach.py`. **추정으로 바꾸지 말 것.**

1. **그리퍼 최대 개구 = 84.5 mm** (이론치 100 mm 아님).
   충돌 근사가 convexHull 이고 통과폭은 가장 안쪽 점인 **핑거 팁**이 지배한다.
2. **shaker 는 계단형 원뿔이다.** bbox 지름 88 mm 는 상단 최대치이고 몸통은
   58 / 68 / 78 / 88 mm 로 단계적으로 굵어진다.
   → 스케일 축소 불필요, 대신 **테이블 위 10~85 mm 에서만** 파지 가능
   (채택 h=65 mm 에서 통과지름 68 mm, 편측 여유 8.2 mm).
   `shaker_body` 가 아니라 **`shaker_closed`** 를 쓴다 — 원본은 양쪽 뚫린 관이라
   내용물이 그대로 빠지고, 양팔 pour 의 receiver 로 이어지려면 받을 수 있어야 한다.
3. **jaw 수평 + 접근축 수평을 동시에 고정하면 팔이 자세를 못 낸다**(손목 j6 가 ±45°).
   파지에 필요한 건 jaw 축 수평뿐(두 접촉점이 컵 단면 지름 양끝)이므로 접근각은 풀었다.
   → 기준자세 **jaw 방위 −15° / 접근 기울기 35°**.
   파지 높이는 그리퍼 여유와 팔 도달성이 **반대 방향**이라 대역 안에서 스윕해 정했다:
   h=55 → 관절여유 0.101 / **h=65 → 0.238** / h=75 → 0.005 / h=85 → 공통해 없음.
4. **컵 스폰 x = 0.25** (우측 0.30 의 미러가 아니다). 파지점이 낮아 x=0.30 에서는
   팔이 못 미친다(실측 잔차 11~20 mm).

---

## 코드 수정 시 주의

- **래치·성공 게이트를 완화하지 말 것.** 1지 접촉 래치를 허용하면 부실 파지 국소최적이
  생긴다(좌측 grasp_v1 에서 게이트 완화로 lifted 0.72→0.002 붕괴한 이력).
  문제가 보이면 게이트가 아니라 제어(기준자세·delta·게인)를 고친다.
- **그리퍼 목표는 `l_hj_gripper_1` 에만 준다.** `_2` 는 USD mimic 이라 둘 다 지령하면
  mimic 제약과 드라이브가 싸운다.
- **`l_hl_gripper_tcp` 는 physics USD 에 강체로 없다.** ContactSensor 대상도, body 조회
  대상도 될 수 없다. TCP 는 `l_hl_gripper_base` + z 0.08 로 계산한다.
- reward/게이트/weight 변경 전 `reward-audit` 스킬 통과 필수(hdgp 공통 규칙 5).
- fabric URDF 를 다시 만들 때는 **링크·조인트 이름을 우측과 동일하게 유지**해야 한다.
  `openarm_tesollo_pose_params.yaml` 의 충돌구 프레임 리스트가 이름 하드코딩이다.

---

## 진단 순서

1. TFEvents: `metric/contact_frac` → `gate/both_contact` → `metric/latched_rate`
   → `metric/lifted_rate` → `metric/success_rate` 순으로 막힌 지점을 찾는다.
2. `metric/cup_xy_disp` 가 0.025 를 넘으면 접근이 컵을 밀고 있다 — reward 가 아니라
   기준자세·접근 경로 문제다.
3. **지표가 평탄하면 epoch 를 더 태우지 말고 zero-action probe 부터.**
   RL 지표 평탄이 보상 문제가 아니라 제어 문제였던 이력이 있다(pour_v1: 2442 epoch 를
   1분 probe 가 대체).
4. 기하가 의심되면 위 두 프로브를 다시 돌린다 — 둘 다 Isaac 없이 수 초에 끝난다.

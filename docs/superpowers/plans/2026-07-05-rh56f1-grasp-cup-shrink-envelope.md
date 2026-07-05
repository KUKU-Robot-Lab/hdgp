# rh56f1 grasp_v1 컵 축소 + envelope 파지 Implementation Plan

**Goal:** 컵 반지름을 30% 축소(scale 0.7, r 3.5→2.45cm)하여 palm floor·envelope의 kinematic 병목을 물리적으로 해소하고, 접근 pose를 정정(thumb_2 0.15→0)하여 정책이 모든 손가락 envelope + 컵 수직 리프트를 학습하도록 한다.

**근거 (test5 재분석, 5개 test 일관):**
- `full_envelope_rate` 5개 test 전부 **0.0 flat** → 컵 지름이 envelope 물리 병목.
- `palm_to_cup_dist=0.066`, `contact/palm=0.0 flat` → palm이 kinematic floor(7.4cm)에 붙어 컵에 절대 안 닿음. palm 밀착 파지 불가 확정.
- `envelope_finger_count≈1.0`, `lift_tilt_deg=20°`, `upright_quality 0.44→0.16 붕괴` → 한쪽 부분파지 → 리프트서 컵 20° 회전.
- `q1(thumb_2)=0.10 flat` → 엄지 능동 굽힘 없음, abduction 원위 tip만 수동 접촉.

## Global Constraints
- **obs/action 차원 불변** (사용자 명시 요청 없이는 금지).
- **게이트/weight 변경은 reward-audit 스킬 통과 후에만** (Task 4).
- **Tesollo 태스크 무영향**: 공유 코어(`grasp_reward_core.py`) 기본값 유지, env-level 게이팅만.
- **log-first**: 게이트 재보정 수치는 probe 실측(Task 3) 후 확정.
- 외과적 변경: 각 변경 라인은 이 목표에 직접 추적 가능해야 함.

---

## Task 1: 컵 기하 30% 축소 + 물리 질량 고정

**Files:**
- Modify: `source/openarm/openarm/rh56f1/right/grasp_v1/grasp_right_env_cfg.py`

**변경:**
1. `cup_cfg.spawn` scale `(1.0,1.0,1.0)` → `(0.7,0.7,1.0)` (xy만 축소, 높이 유지).
2. `cup_cfg.spawn`에 `mass_props=sim_utils.MassPropertiesCfg(mass=0.170)` 추가 — **scale 무관 물리 질량 고정**. (없으면 density 기반 USD가 scale 0.7²=0.49배로 질량을 몰래 절반 → force-ratio DR 붕괴, 파지 난이도 왜곡.)
3. `cup_radius_approx` `0.035` → `0.0245` (enclosure 게이트 타깃 정합).

**검증:**
- `pytest source/openarm/openarm/rh56f1/right/grasp_v1/tests/test_phase4_env_static.py -q` PASS
- `pytest source/openarm/openarm/rh56f1/right/grasp_v1/tests/test_v7_2_reward_contract.py -q` PASS
- probe(Task 3)에서 컵 실측 반지름 ≈0.0245 확인.

---

## Task 2: 접근 pose 정정 (엄지 원위 완전 폄)

**Files:**
- Modify: `source/openarm/openarm/rh56f1/right/grasp_v1/grasp_right_preset.py`

**변경:**
- `HAND_APPROACH_POSE[1]` (thumb_2) `0.15` → `0.0`. thumb_1(1.57 opposition)은 유지. index~pinky 이미 0.0.
- 주석 갱신: 렌더 관찰(q1=0.10 flat, 엄지 능동 굽힘 없음) → 접근은 엄지 원위 완전 폄에서 시작, 정책이 컵을 gap에 넣고 wrap 학습.

**검증:**
- probe(Task 3)에서 pregrasp 시 엄지(thumb_3/4) 컵 관통 없음(clearance ≥ 0) + 5 손가락 폄 확인.

---

## Task 3: probe 재보정 — 새 기하 palm floor·clearance 실측 (log-first)

**Files:** (코드 변경 없음, 측정 전용)
- Run: `scripts/tools/probe_palm_orientation.py`

**절차:**
1. Task 1·2 적용 상태에서 로컬 GPU probe 실행:
   `cd /home/user/rl_ws/IsaacLab && ./isaaclab.sh -p ../hdgp/scripts/tools/probe_palm_orientation.py --headless --num_envs 4 --grip_steps 30`
2. 측정: 새 `palm_to_cup_dist` floor(예상 7.4→6.4cm 부근), 엄지/손가락 clearance, pregrasp offset 적정성, grip 30step 후 wrap 분류(감쌈/관통/벌어짐).
3. 산출: `approach_palm_radial_max`, `thumb_freeze_release_dist`, 필요시 `pregrasp_offset_x/y/z` 재조정 후보값.

**검증:** probe 출력에서 palm floor·clearance 수치 확보 → Task 4 입력.

---

## Task 4: 게이트 재보정 + envelope-force 완화 (reward-audit 필수)

**Files:**
- Modify: `source/openarm/openarm/rh56f1/right/grasp_v1/grasp_right_env_cfg.py`

**변경 (Task 3 실측 기반, reward-audit 통과 후):**
1. `approach_palm_radial_max` — 새 floor(6.4cm)에 맞춰 재조정 (palm이 여전히 접근해야 손가락 닫히도록). Task 3 실측값.
2. `thumb_freeze_release_dist` — 새 floor에 맞춰 재조정 (enclosure 유도 게이트가 palm 근접 후 켜지도록).
3. `lift_envelope_mix` `0.65` → `0.58`, `grasp_envelope_credit` `0.55` → `0.47` — test5에서 0.65/0.55는 success를 죽이고 upright 이점도 소멸(과함 확인). 얇은 컵이 envelope의 물리적 enabling을 담당하므로 force는 완화.

**reward-audit 체크리스트** (Check 1~5 전부 통과 시에만 적용):
- Check 1/2: 게이트 임계값만 조정, 새 보상 경로·hacking 없음.
- Check 3: 얇은 컵으로 envelope-grasp 상충 완화.
- Check 4: envelope-force 완화는 test5 실패값 되돌림 → 기존 파괴 아님.
- Check 5: `full_envelope_rate`, `envelope_finger_count`, `lift_tilt_deg` TFEvents 로깅 존재.

**검증:** reward-audit 판정 ACCEPT + 정적 테스트 재PASS.

---

## Task 5: GPU 학습 test6 + 모니터링

**절차:**
1. 서버(oem@10.102.101.240) GPU0에서 `open-rh56f1_r_grasp_v1` 학습 (train.sh 자동 넘버링 → test6).
2. 지속 모니터링(parse_tfevents): `full_envelope_rate`, `envelope_finger_count`, `lift_tilt_deg`, `cup/tilt_deg`, `upright_quality`, `success_held_rate`, rewards.

**성공 기준 (5개 test 병목 탈출 확인):**
- `full_envelope_rate > 0` (5 test flat 0.0 탈출) — **1차 지표**.
- `envelope_finger_count > 3` (실제 감싸는 손가락 ≥3, 현재 ~1).
- `lift_tilt_deg < 12°` (현재 20°) + `success_held_rate > 0` (현재 flat 0).

**실패 시 분기:** full_envelope 여전히 0 → 축소율 40%(scale 0.6) 재시도 검토. envelope는 생기나 upright 미달 → success_upright_max_deg 완화 별도 논의.

---

## 리스크
- **비균등 scale + collision**: cup_middle.usd collision이 convex/SDF면 (0.7,0.7,1.0) 안전. probe/play.py 렌더로 컵 형상 붕괴 여부 육안 확인(Task 3).
- **질량 고정 누락 시**: force-ratio DR 왜곡 → Task 1의 MassPropertiesCfg가 방어.
- **게이트 미보정 시**: 새 floor에서 approach 게이트가 너무 느슨 → 즉시 손가락 닫힘(test5 pinch 재현). Task 3·4가 방어.

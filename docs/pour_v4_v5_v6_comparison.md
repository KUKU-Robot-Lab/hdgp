# pour-v4 / v5 / v6 비교 (3-way 대조군)

> 작성 2026-06-24. reward 동일성 검증 후 확정. 변수는 **제어 패러다임 단 하나**.

## 한 줄 요약

세 태스크는 **reward·관측·종료조건·자산이 100% 동일**하고, **arm 제어 방식만 다르다.**
deep tilt(110°+) 미달의 원인이 "제어 방식"인지 분리하기 위한 통제 실험.

| | 제어 패러다임 | tilt pivot | 어깨(j1-3) 정책 제어 | action[:7] 의미 |
|---|---|---|---|---|
| **v5** | Fabrics IK | **rim-pivot** | ✗ (IK null-space) | palm 6D + α nullspace |
| **v6** | Fabrics IK | **palm-pivot** | ✗ (IK null-space) | palm 6D + α nullspace |
| **v4** | **joint-PD** | (관절 직접) | **✓ 직접** | 7 arm joint 축 |

---

## 1. 공통 (동일성 검증됨)

| 항목 | 값 | 검증 |
|---|---|---|
| reward 함수 `_get_rewards` | 동일 | v4↔v5 diff=0, v5↔v6 diff=0(주석만) |
| reward cfg weight | 동일 | v4↔v5 diff=0 |
| action_space | 7 | 동일 |
| observation_space (actor) | 55 | 동일 |
| state_space (critic) | 144 | 동일 |
| class 명 | PourRightEnv / PourRightEnvCfg | 동일 |
| Fabrics world yaml | open_tesollo_boxes_pour_v5 (공유) | 동일 |
| 자산·컵·비드·warmstart | 동일 | 동일 |

**reward 핵심 구조 (Phase3 재설계, 3개 공통):**
- `r_tilt = weight_tilt·tilt_progress` (+ `weight_tilt_delta=100`로 deep tilt 증분 보상)
- `r_pour = weight_pour_bead·corridor_score·bead_in_target_fraction` (outcome ADR)
- `r_align = weight_align·(1+directional_tilt_cos)/2` (보조)
- `tilt_amount = (1−rim_antiparallel)/2` — target 상대각 기준
- `pose_success = corridor≥thresh ∧ tilt_amount≥pose_tilt_thresh(100°+)`

---

## 2. 제어 방식 차이 (유일한 변수)

### v5 — rim-pivot Fabrics IK
action xyz가 **pour-point(주둥이) target**을 이동, 팔이 rim 둘레로 회전(tilt 중 rim 고정).
```
rim_env          = source_pour_point_w − env_origins
rim_rel          = rim_env − palm_center_pos          # palm→rim 레버
pour_point_target= rim_env + delta[:, :3]
palm_ee_target   = pour_point_target − quat_apply(Δq, rim_rel)   # 역산
→ Fabrics IK(palm pose 6D) → 7 arm joint
```
- 장점: 주둥이를 직접 명령(붓기 지점 정밀). 단점: 역모델이 컵 자세에 종속(non-stationary).

### v6 — palm-pivot Fabrics IK
action xyz가 **palm을 직접** 이동, palm 둘레로 회전.
```
palm_pose[:, :3] = palm_center_pos + delta[:, :3]     # palm 직접
→ Fabrics IK(palm pose 6D) → 7 arm joint
```
- 장점: 정상(stationary) 매핑(정책이 역모델 학습 쉬움). 단점: 주둥이는 간접 제어.

### v4 — joint-position PD (Fabrics IK 우회)
action 7채널 = **7 arm joint 직접**. START→DEMO 축 파라미터화.
```
target = ARM_START_POSE + 1.3·action·(DEMO_POUR_ARM_POSE − ARM_START_POSE)
target = clamp(target, joint_limits)
→ set_joint_position_target(target)   # articulation PD drive
```
- action=0 → START(파지자세, hold 안전), action≈0.77 → DEMO(pour), 1.0 → j5=-1.52(최심 tilt)
- **핵심**: j1-3(어깨)를 정책이 **직접** 제어 → v5/v6에서 IK null-space로 막혀있던
  "deep tilt 시 어깨 협응"을 구조적으로 학습 가능.

---

## 3. 가설과 대조 논리

**진단(사용자):** "stage C에서 j5,6,7로 tilt가 안 되면 j1,2,3(어깨)을 써서 deep tilt해야 하는데,
정책이 그걸 못 배워 ~90° 정체."

- **v5/v6 (IK)**: 정책은 palm pose 6D만 제어 → 어깨는 IK 부산물. 어깨 협응을 의도적으로 못 만듦.
  - 대조: v5(rim) vs v6(palm) = 같은 IK에서 **pivot 기준**이 deep tilt 도달에 미치는 영향.
- **v4 (joint-PD)**: 7관절 직접 제어 → 어깨 협응을 정책이 직접 탐색.
  - 대조: v4 vs v5/v6 = **제어 패러다임**(직접 관절 vs IK)이 deep tilt 도달에 미치는 영향.

**측정 지표(공통):** `log/tilt_frac_110`(deep tilt 도달률), `tilt_frac_90`, `cmd_minus_actual_tilt_deg`,
`pose_success`, `bead_in_target`.

---

## 4. 현재 상태 (2026-06-24)

- **v5/v6**: B-trajectory PIECE 1(β→R(β) 전신협응 + j5 하드구동) 학습 중.
  ep~470: v6 frac_90=0.39↑·frac_110≈0, v5 frac_90=0.09. deep tilt(110°)는 아직 미돌파.
- **v4**: 코드 준비 완료(commit 968c434), 미학습. GPU 여유 시 런치.

=== Reward Audit === (2026-09-06, grasp_kp/grasp_fj 신설 보상 — reward-audit 스킬)

변경 대상: grasp_s2r 15항 접촉 기반 보상 → progress-only 8항
  fingertip_progress 50 · lift 20·clamp(0.05+dz,0,0.5) · lift_bonus 300(1회, dz>0.10) · keypoint_progress 200(lifted 후)
  · goal_bonus 100/step(near_goal) · arm_vel −0.03·Σ|q̇| · hand_vel −0.003·Σ|q̇| · hand_floor −min(10·relu(0.215−z_min), 5)
근거: 사용자 지시(접촉센서 미사용·SimToolReal 식 전면 개편) + reports/simtoolreal_적용성_평가.md §5.2

Check 1 (Local Min): ✓(조건부) — 리프트 전 상수 수입 1.0/step(20×0.05)이 "가만히 있기" 기저선을 만든다. 그러나 접근 진행분
  (50×Σ(d0−d_min) ≈ 75) > 75 step 대기, 리프트 보너스 300 + 목표 보너스 100/step 이 압도한다. 2,048~4,096 env 에서 절벽(래치 0.10)을
  넘는 탐색이 부족할 위험은 보고서 §10 에 이미 적힌 대로 `stage/lifted`·`task/lifted_frac` 로 감시한다(500 epoch 내 상승 없으면 재검토).
Check 2 (Hacking):   ✗→수정 — goal_bonus 는 lifted 게이트가 없다(SimToolReal 동일). 첫 목표 z 하한 0.12 와 시작 허용오차 0.06 이면
  dz=0.06 에서 near_goal 이 성립해 **래치(0.10)를 안 넘고도 100/step** 을 받는 구멍이 있다. 또 이후 목표 박스 z 하한 0.08 도 tol 0.06 과
  겹치면 테이블 근처 밀기가 근접 판정을 받을 수 있다.
  → 첫 목표 z 범위 (0.12,0.20) → **(0.16, 0.24)**: near_goal(tol 0.06) ⇒ dz ≥ 0.10 = 래치. 목표 박스 z 하한 0.08 → **0.10**.
  종료를 유도하는 벌점은 없다(hand_floor 는 상한 5, 리프트 전 순수입은 양수) → 조기 종료 최적화 없음.
Check 3 (Grasp):     ✓ — 접촉 항이 없어 파지와 상충하는 gradient 가 없다. 파지 품질은 보상이 아니라 리프트 후 질량정규화 외란(≈2 g)과
  낙하 종료(수입 상실)가 강제한다. hand_vel −0.003 은 SimToolReal 값 그대로(폐쇄 속도 0.005/step 리미터가 이미 더 강한 제약).
Check 4 (기존 파괴): ✓ — 신설 폴더(grasp_kp/grasp_fj). grasp_s2r 은 무변경(기준선 보존).
Check 5 (측정):      ✓ — reward/<8항>·reward/total, task/kp_dist, task/near_goal, task/lifted_frac, task/just_lifted, task/successes_mean,
  task/tol, task/hand_floor_depth_max, stage/{lifted,goal1,goal2,goal3}, done/*. 항별 로깅이 전부 TFEvents 에 남는다.

판정: REVISE → 위 2 개 기본값 수정 후 ACCEPT (DESIGN.md §2 반영, cfg 기본값 동일 적용)

예상 지표 이동:
  → stage/lifted: 0 → 0.3+ (≤500 epoch; 접근 진행분+리프트 보너스가 유일한 초기 경사)
  → task/successes_mean: 0 → ≥1 (lifted 직후; 첫 목표가 래치 바로 위)
  → task/tol: 0.06 → 0.015 계단식(3000 프레임·mean successes ≥ 2 조건)
  → reward/goal_bonus 비중이 후반 총보상의 대부분 — 정상. 리프트 전 총보상 ≈ 1.0/step 이 장기 유지되면 Check 1 의 국소최적 발동.


=== Reward Audit === (2026-09-07, A-v/B-v 리프트 후 안정 파지 — reward-audit 스킬)

변경 대상:
  ① 신설 항 cmd_rate = −rw_cmd_rate_scale · 정규화 지령변화율 · lifted   (A 0.1 · B 1.0, 상한 없음, 리프트 전 0)
     A 측도: 리미터 **전** 원지령 변화 / 리미터 상한 (위치·회전 평균, 비유계, 작동점 ≈10~15)
     B 측도: 팔 액션 1차 차분 RMS/2 ∈ [0,1] (전속 이송 0, 매 스텝 ±1 반전 1)
  ② goal_force_consecutive False → True (성공 = 공차 안 **연속** 10 스텝)
  ③ (보상 아님) SAPG yaml 9키 b1 복귀 · 사다리 상단 0.002
근거: 사용자 지시 "LIFT 이후 안전한 파지(ARM 떨림 없음) · 리미터를 믿고 아무렇게 액션 내는 건 이상" + 결정론 프로브
  act_sat 0.835 / rate_sat 0.939(μ 가 벽에 붙은 bang-bang) + kp_a6 e617 lifted 0.79 인데 rate_sat 0.976.

Check 1 (Local Min): ✓ — do-nothing 가치(리프트 전 1.0/step, γ 0.99) = 100. 리프트 후 벌점 A −0.1×12.5 = −1.25/step →
  할인 합 −125, 리프트 가치 300 − 125 = 175 > 100 (여유 1.75×; scale 0.2 면 50 < 100 로 **뒤집힌다** → 0.1 확정).
  B 최악 −1.0/step → −100, 300 − 100 = 200 > 100. "리프트 후 가만히"는 벌점 0·진행 0 이라 이송(+4/step, 세금 −0.1)보다 못하다.
Check 2 (Hacking):   ✓ — 벌점을 0 으로 만드는 길은 지령을 안 바꾸는 것(A)/액션을 안 뒤집는 것(B)뿐인데, 그 상태로 목표를
  못 맞추면 goal_bonus 100/step 을 잃는다. 연속 판정이라 흔들리며 세는 우회도 막힌다. 자살경로(penalty-reward-regime-triad ③):
  리프트 후 스텝당 순수입은 목표 근처 +100, 이송 중 +4 − 1.25 > 0 → 종료가 이득인 구간이 없다.
Check 3 (Grasp):     ✓ — 손 액션은 벌하지 않는다(A 팔 6D / B 팔 7D). 폐쇄 리미터 0.005/step·hand_vel 은 그대로. 연속 판정은
  파지가 흔들리면 성공을 미룰 뿐 손을 열게 하는 gradient 가 아니다.
Check 4 (기존 파괴): △ — 성공이 엄격해져 successes 는 a6(7.3@617) 보다 낮게 간다(의도). 리프트·접근 항 무변경, 리프트 전 벌점 0.
  ★σ 부작용: fixed_sigma 라 σ 가 전 env 공유 → lifted env 의 벌점이 σ 를 줄여 리프트 전 탐색도 준다. A 는 entropy 0.0 이라
  대항력이 없다 — 감시 diag/action_sat · losses/entropy. a6 는 e305 에 lifted 0.60 이라 벌점은 리프트 학습 **뒤에** 켜진다.
Check 5 (측정):      ✓ — reward/cmd_rate(항별 자동) · task/cmd_rate_lifted(정책 몫) · diag/arm_qd_p99_lifted · diag/obj_speed_lifted
  (외란 포함) · fabric/palm_cmd_rate_sat(A: 0.976 → 내려가야) · ctrl/arm_action_rate_lifted(B) · task/successes_mean(연속 판정).
Check 6 (표본 상태·작동점 — task-observer #12/#6/#10): ✓ — 액션 매핑 무변경(절대 유지, 탐색 분포 동일). 작동점은 상수에서 유도:
  A step_raw 0.20 m / 0.02 = 10×, 회전 벽↔벽 80°/2.9° = 27.6×, σ=1 노이즈만으로 √2σ·(0.1,0.1,0.35)/0.02 ≈ 26× → 평균 10~15.
  상한 없음(작동점 clamp 금지). score_to_win 1e6 vs 이론 상한 50 목표×1000 + 300 = 50,300 → 여유 20×.

판정: ACCEPT (Check 4 의 σ 부작용은 감시 조건부)

예상 지표 이동:
  → A fabric/palm_cmd_rate_sat 0.976 → < 0.5 (lifted 비중 커질수록) · task/cmd_rate_lifted ≈10 → < 2
  → A diag/arm_qd_p99_lifted 2.5 → < 1.0 rad/s · B ctrl/arm_action_rate_lifted ≈0.7(노이즈 바닥) → < 0.3
  → task/successes_mean: 연속 판정이라 a6 보다 낮게 시작, tol 0.015 도달 지연 가능(tol_success_threshold 2.0)
  → diag/obj_speed_lifted 는 외란 킥(Δv 0.33 m/s · p 0.001~0.1)이 바닥 — 0 이 되지 않는다. 정책 진동은 cmd_rate_lifted 로 읽는다.

--- 09.07 kp_a7 실측 후 REVISE (A-vi) ---
kp_a7 은 e25 까지 a2/a6 와 동일(close 0.37·ft 0.18·step_raw 0.168)했다가 e25→e50 에 접근이 죽었다(close 0.007·ft 0.39,
reward 0.84 = do-nothing). 그 구간 reward/cmd_rate −0.0215(lifted 0.016 → lifted env 당 **−1.35/step**, 위 산수 −1.25 와 일치).
★크기는 맞았고 **표본**이 틀렸다(Check 6 미완): e25 의 lifted 는 파지가 아니라 우연히 튕겨 올라간 컵이고, 래치가 sticky 인 데다
done/fell 이 죽어 있어(0.15 < 상판 0.205) 상판에 다시 놓인 컵도 에피소드 끝까지 lifted 다 → 진행 항이 꺼진 채 −1.35/step 이
500 스텝 붙는다. 그 env 들의 공통 행동이 "컵을 세게 건드림"이라 정책이 **회피**를 배웠다(penalty-reward-regime-triad ②③).
B(fj_b7)는 같은 게이트에 lifted env 당 −0.29/step 라 견뎠으나 b1 보다 3배 느렸다(e200 0.17 vs 0.54) — 같은 원인 의심.
수정: ① 전역 sticky 래치(lifted_frac EMA α0.002 ≥ 0.30, a6 e130) 뒤에만 벌점 — suppression-terms-need-task-first 의
enable_penalty_after_dwell 규약을 per-env 플래그로 대체했던 것을 되돌림. ② hold 게이트 lifted ∧ dz > 0.03(drop_frac 판정선):
떨어뜨린 컵을 다시 쥐러 가는 이동은 벌하지 않는다. 크기 0.1/1.0 유지. → kp_a8 / fj_b8.
Check 1 재계산(arm 뒤): 리프트 후 들고 있는 env 만 −1.25/step, 나머지 0 → 300 − 125 = 175 > 100 그대로.

--- 09.07 kp_a8 실측 후 REVERT (성공 술어) ---
kp_a8 은 cmd_rate 벌점이 **한 번도 armed 되지 않은 채**(task/cmd_rate_armed 0.000, reward/cmd_rate 0.000 전 구간) 실패했다.
a6→a8 의 hydra dump env 차이는 `goal_force_consecutive: false→true` 한 줄뿐(cmd_rate 4필드는 비활성). 따라서 a7 붕괴를
"벌점이 우연 리프트 env 에 붙었다"로만 본 진단은 **부분적으로 틀렸다** — 벌점은 가속 요인이었고 근본 원인은 이 술어다.
기전(Check 1 재발): 연속 판정 → 성공 ≈0 → 목표 미전진 → closest_kp 가 에피소드 최소에 고정 → keypoint_progress 고갈.
리프트는 `lift`(20×0.05 = 1.0/step)와 `fingertip_progress` 를 끄므로, 목표로 수렴 못 하는 리프트 env 수입 ≈ 0/step
< 안 든 env 1.0/step → **리프트가 순손실**. 실측 a8: kp_dist_min 0.201 고정(a6 0.109), succ 0.000(a6 4.25),
r_goal 0.00(a6 18.90), r_lift 1.0019(= 전 env 미리프트), lifted e50 0.108 → e200 0.0000.
B 도 같은 길: fj_b8 e215 succ 0.0044 vs fj_b1 0.1475(33배), lifted e150 이후 0.31 에서 정체(b1 은 0.18→0.75 가속).
★내 Check 4 는 이 변경을 "successes 가 낮게 간다(의도)"로 적었다 — **리프트 결정의 부호가 뒤집히는 것**을 못 봤다.
★SimToolReal 이 forceConsecutive 를 쓰는 전제는 시작 공차 0.075×keypointScale 1.5 = 0.1125 m 로 우리(0.06)의 2배라는 것.
  술어만 이식하고 공차를 안 맞춘 것이 오류. 되살리려면 tol_start 를 함께 올려야 한다.
조치: goal_force_consecutive → False 복귀(A·B). 안정 파지는 cmd_rate 벌점이 단독으로 담당한다(완료 판정과 분리).
→ kp_a9 = kp_a6 + cmd_rate(전역 arm 래치 + hold 게이트) 단일 변수 시험. fj_b9 = fj_b8 − 연속 판정.

--- 09.07 kp_a9 실측 후 정정 (진단 3회차 — 앞의 두 진단이 과잉 귀인이었다) ---
kp_a9 는 a6 과 **설정이 같다**(hydra dump env 차이 = 비활성 필드 4개, 부팅 라인의 프로필·앵커·박스·게인·정착고 전부 동일,
서버 트리 clean, 끼어든 커밋 없음). 그런데 e80→e100 에 절벽으로 무너졌다:
  close 0.304 → 0.0066 · ft 0.232 → 0.460 · syn_close 0.146 → 0.003(e150) · reward 1.41 → 0.83 · lifted 0.083 → 0.000
즉 **이 붕괴는 트랙의 성질이지 내 변경이 만든 것이 아니다.** 6런 대조로 다시 보면:
  a1 붕괴 없음(e300 lifted 0.77) · a6 붕괴 없음(0.63) · a2 e50 에 얕은 붕괴 후 150 epoch 정체하다 회복(0.45)
  a9 e90 에 완전 붕괴(0.00) · a7(벌점+연속) e50 붕괴 · a8(연속) e75 붕괴
붕괴는 **x 액션 포화와 함께 온다**: act_sat_arm0 e150 기준 a6 0.45 / a1 – / a2 0.78 / a9 0.81 → 살아남은 런은 안 포화됐다.
따라서 앞선 두 진단(①벌점이 우연 리프트 env 에 붙었다 ②연속 판정이 리프트를 순손실로 만들었다)은 **효과는 실재하나
붕괴의 필요조건이 아니다**. ②는 여전히 유효하다 — a8 은 목표가 안 전진해 **회복 경로가 없었다**(a2 는 회복했다).
★내 절차 오류: 분산이 큰 트랙에서 단일 런 대조로 인과를 세 번 주장했다. a2 하나만 봤어도 "붕괴 후 회복"이 이미 있었다.
★cmd_rate 는 아직 한 번도 발동한 적이 없다(a7 은 잘못된 표본, a8·a9 는 armed 0). arm 임계 lift_ema ≥ 0.30 은
  **붕괴하는 런에서는 영원히 도달 불가**다(a9 최고 lifted 0.083) — 게이트 시점 설계가 틀렸다.
다음 설계 후보(미실행): 벌점 대상을 `cmd_rate` 전체가 아니라 **리미터 초과분 relu(cmd_rate − 1)** 로 바꾼다.
  리미터 안의 지령은 벌점 0 이라 도달 가능한 어떤 운동도 억제하지 않는다(= 움직임 억제 항이 아니므로
  suppression-terms-need-task-first 의 게이트 요구가 적용되지 않는다) — 벌하는 것은 리미터가 버리는 몫뿐이다.
  작동점: cmd_rate ≈ 13.5(a7 e25 실측 −1.35/step ÷ 0.1) → relu(13.5−1) = 12.5. scale 0.01 이면 −0.125/step
  = 그 시점 살아있는 신호(fingertip_progress ≈ 0.5, lift 1.0)의 8%. fab_test14 의 jerk 는 48% 였다.

--- 09.07 fj_b9 실측 (cmd_rate 첫 발동 — 과제는 안 죽었고, 떨림도 안 줄었다) ---
fj_b9 는 e227 에 cmd_rate 가 arm 됐고(lift_ema 0.30 통과) **그 뒤로도 lifted 가 0.30 → 0.82 로 계속 올랐다.**
벌점이 과제를 죽이지 않는다는 첫 증거다(a7 의 붕괴는 표본 문제였음이 이로써 다시 확인된다).
그리고 b9 는 매칭 에포크에서 b1 을 앞선다: e465 succ 9.61 / 6.78 · reward 39.5 / 33.5 · lifted 0.821 / 0.763
(같은 tol 0.054). b1 하이퍼 복귀 + 누적 판정 + SAPG 4블록 조합이 옳았다.

★그러나 **벌점이 목표한 일을 하지 못하고 있다.** 벌하는 양(ctrl/arm_action_rate_lifted)이 arm 전후로 안 움직인다:
  e200(arm 전) 0.315 → e250 0.332 → e465 0.320.  목표는 < 0.30 이었다.
  qd99_lifted 1.55 → 1.41(−9%)는 미미하고, obj_speed_lifted 0.553 → 0.276 은 파지가 좋아진 효과와 뒤섞여 있다.
원인은 **크기 산정의 기준을 잘못 잡은 것**이다. Check 1 은 벌점을 do-nothing 상수(1.0/step)와 견줬는데,
과제가 풀리면 goal_bonus(100/step)가 들어와 총보상이 40/step 이 된다. 지금 벌점은 −0.26/step = **총보상의 0.65%** 로
사실상 잡음이다. 즉 초기(탐색)에는 적절했고 후기(작동점)에는 40배 작다.
→ 규칙 추가: 억제 항의 크기는 **두 작동점 모두**에서 검사한다 — (a) 켜지는 시점에 살아있는 신호 대비(탐색을 죽이지 않을 것),
  (b) 과제가 풀린 뒤 총보상 대비(무시되지 않을 것). 한쪽만 보면 "탐색은 살렸는데 아무 일도 안 하는 항"이 된다.
다음 실험 후보(미실행): B 의 rw_cmd_rate_scale 1.0 → 2.5~6. b9 를 끝까지 돌린 뒤 대조군으로.

--- 09.07 kp_a10 실측 후 **정정** (커밋 d6cdbff8 의 근거가 틀렸다) ---
d6cdbff8 은 "정렬 세트 5런에서 깨끗한 성공 0"을 근거로 하이퍼를 되돌렸다. **거짓이다.** 생애 최고치 실측:
  kp_a1 (구 세트)   e7548  lifted 최대 0.860 / 최종50 0.839 / succ 9.53 / reward 40.89
  kp_a2 (정렬)      e2279  0.867 / 0.843 / 9.22 / 39.94
  kp_a6 (정렬)      e723   0.839 / 0.775 / 9.40 / 40.35   ← **내가 죽인 것**이지 실패가 아니다
  kp_a7/a8/a9/a10          0.03~0.11 / 0.000 / ~0 / ~1.6  (전부 do-nothing)
두 하이퍼 세트는 결과가 구분되지 않는다. 나는 a6 을 "0.628 @e300"으로 인용했는데 그건 중간값이었고,
죽인 시점 e723 에 0.775 로 상승 중이었다. **중간 스냅샷을 최종 성적으로 인용한 오류다.**
또 모든 런의 seed 가 42 다 — a6 대 a9 는 처리군이 아니라 **복제쌍**이고, 차이는 PhysX GPU 비결정성이다.
진짜 경계선: **cmd_rate 코드 도입(c1bdecb3) 이전 3/3 성공 vs 이후 0/4**. 항 값은 a8/a9/a10 에서 증명된 0 이고
(task/cmd_rate_armed 0.000 전 구간) 보상은 비트 단위로 같아야 한다. 그런데도 갈렸다.
  · 가설 A: 스텝 루프에 추가된 연산(_cmd_rate·nanquantile·_lifted_mean)이 PhysX 비결정성의 실현을 바꾼다(인과 아님).
  · 가설 B: 우연. p=0.5 가정 시 이 배열이 나올 확률 1/32~1/128.
  a7·a8 은 실제 유해 변경(잘못된 표본 벌점 · 연속 판정)이 따로 있었으므로 설명되지 않는 실패는 a9·a10 둘이다.
★런 하나씩으로는 두 가설을 구분할 수 없다. 통제 실험으로 간다(GPU0 에 3개 병렬).
★부수로 확인된 구조적 결함 2건(워크플로 포렌식):
  ① **주차 상태는 진짜 흡수 최적점이다.** 리프트 전 수입 `lift = 20·(0.05+dz)·¬lifted` 는 **위치 무관 상수 1.0/step**
     이고, 유일한 위치 의존 항 `fingertip_progress` 는 **소진되는 경로적분**이다(주행 최소값 갱신분만 지급, 후퇴는 0).
     실측: 파지 중인 a6(ft 0.076)과 주차된 a9(ft 0.211)가 **에피소드당 같은 예산**을 가져간다(54.2 vs 46.1).
     소진 뒤에는 컵 쪽으로 향하는 기울기가 **아예 없고**, arm_vel(−0.04)이 정지를 보상하므로 **주차가 호버링보다 낫다**.
     a9 는 붕괴 후 보상의 55%를 "덜 움직여서" 회복했다(+0.096/step). REWARD_AUDIT 첫 감사의 Check 1 경고가 그대로 발동한 것.
  ② **목표 박스 xy 가 델타 박스에 정확히 내접**한다: 필요 |delta_xy| = spawn 0.02 + halfwidth 0.08 = 0.100
     = palm_delta_xyz[0:2] → 여유 0.000 m. 극단 목표는 |a|=1.000 에서만 지령 가능하고 거기서 ∂target/∂a = 0.
     z 는 액션 범위의 38.6%가 한 점으로 클램프된다. `_assert_goal_box_in_arm_reach` 는 절대 매핑에서 델타 비교를
     건너뛰므로 이걸 못 보고 ✓ 를 찍는다(task-observer #5 의 재발).

--- 09.08 kp_a12 실측 — cmd_rate 벌점이 A 에서 처음 발동했고, **나쁜 거래**로 판명 ---
kp_a12(HEAD, seed 7)가 e227 부근에 arm 되어 e4000+ 까지 살아 있다 — **억제 항이 과제를 죽이지 않는다**는 A 쪽 확증.
그러나 같은 tol(0.0150) 매칭 대조에서 대가가 크고 산 것이 없다:
                        kp_a1(벌점X)   kp_a12(벌점O −1.30/step)
  successes_mean            4.05            2.18      ← **−46%**
  lifted_frac               0.845           0.869     ← 차이 없음
  palm_cmd_step_raw         0.208           0.185     ← −11% 뿐
  reward/arm_vel           −0.127          −0.149     ← 팔은 **더** 움직인다
  rate_sat                    --            0.934     ← 여전히 리미터에 붙어 있다
  arm_qd_p99                  --            3.31      (a13 통제군 3.34 — 정지 효과 0)
즉 총보상의 16%를 물고 산 것은 원지령 11% 감소뿐이고, 팔 속도·리미터 포화는 그대로다.
★기전: 과지령은 **정책의 나쁜 습관이 아니라 인터페이스의 구조**다. `palm_delta_xyz` z 0.35 m 대 리미터
  0.02 m/step = **17.5배**. 액션은 박스 안의 **위치**를 이름 붙이고 리미터가 그걸 **방향**으로 바꾼다 —
  전속으로 움직이려면 cmd_rate ≈ 15 를 낼 수밖에 없다. 벌점은 나쁜 습관이 아니라 **과제 수행 자체에 매기는 세금**이다.
★내 이전 제안 `relu(cmd_rate − 1)`(초과분만 벌하기)을 **철회**한다: 작동점이 14.9 라 relu(14.9−1)=13.9 로
  현행 벌점의 93% 다. 아무것도 바뀌지 않는다. 작동점을 먼저 재지 않고 제안한 것이 오류.
→ 떨림을 줄이려면 벌점이 아니라 **액션 스케일을 한 스텝이 실제로 낼 수 있는 양에 맞춰야** 한다
  (palm_delta 축소 또는 증분 매핑). 벌점은 그 위에서 미세 조정용으로만 의미가 있다.

---

## 09.08 · fj_c 시리즈 (B: SimToolReal 과제 의미 정합 + 손 폐쇄속도 스윕)

변경 대상(전부 **B leaf 한정**, A 는 산술 무변화):
1. `goal_bonus` 지급 방식: `(1000/10)·near_goal` → `1000·is_success` (모듈 **포크** `fj_reward.py`)
2. 성공 술어: 누적 10회 → **연속** 10회 (`goal_force_consecutive: True`)
3. `tol_start` 0.06 → **0.1125**, 커리큘럼 게이트 2.0 → **3.0**
4. `goal_first_z_range` (0.16,0.24) → **(0.2125,0.28)** ← 3 의 짝
5. 스텝 예산: 에피소드당 600 → **목표당** 600 (`goal_clock_restart_step: 2`)
6. 외란 20/2.0 → **2.7/0.27**
7. 손: 20관절 독립(`hand_direct`) + 폐쇄속도 3팔 스윕 · `hand_velocity_ff_scale` 1.0 → **0.0**

근거: 사용자 지시(09.08) "팔매핑 simtooreal과 동일 / 팔실효 slew도 / 손속도 제한도 /
성공목표에피소드 동일성유지 / **목표델타는 제외** / 외란은 줄일 필요가 있음".

**Check 1 (Local Min): ✓** — 목표당 총액이 **불변**이다(1000 = 10 × 100). 분포만 뾰족해진다.
  기존 원장이 그대로다: 리프트 전 `lift` 가 위치 무관 1.0/step × 600 = 600 으로 여전히 최대
  단일 항이고(주차 흡수점은 이 판에서 **해결되지 않았다** — R1 참조), `lift_bonus` 300,
  목표당 1000. 1회성 전액이 새 고보상 경로를 만들지 않는다.
  ⚠ **단, 목표당 예산이 목표 수를 최대 50 까지 열어준다** — 총 목표 보상 상한이
  에피소드당 실질 ~19 goal → 최대 50 goal 로 늘 수 있다. `lift` 대비 비율이 커지므로
  **주차 최적점을 오히려 약화시키는 방향**이다(유리한 쪽).

**Check 2 (Hacking): ✓ — 단, 4 를 같이 올려야만 성립한다.**
  `goal_bonus` 는 lifted 게이트가 **없다**(`progress_reward.py` / `fj_reward.py` 동일).
  현행 A 의 불변식은 `goal_first_z_range[0] − tol_start = 0.16 − 0.06 = 0.10 = 리프트 래치` 로,
  "near_goal 이면 반드시 래치 위"를 기하로 보장한다. tol 만 0.1125 로 올리면 그 하한이
  0.16 − 0.1125 = **0.0475 m < 0.10** 이 되어 **물체를 4.75cm 만 띄워도 목표 보너스가 나간다**.
  → 4 로 차단(0.2125 − 0.1125 = 0.10, 불변식 복원). cfg 검증기가 부팅에서 강제하고
  (`goal_first_z_range[0] − tol_start ≥ rw_lift_latch_height`), 계약 테스트가 3중 잠금으로 고정한다.
  **이 짝을 빼면 판정은 REJECT 다.**

**Check 3 (Grasp): ✓ (2 항목은 ⚠ 관측 필요)**
  · 외란 1/7.5 은 파지 유지를 **쉽게** 한다 — 역방향 gradient 없음.
  · 팔 slew 6.7배 감속도 파지 유지에 유리(물체가 손바닥을 타고 다니던 문제의 직접 대책).
  · ⚠ 손 폐쇄가 빨라지면(c3) 자유공간에서 손 PD 가 목표를 못 따라가 `blocked` 임계(1.0 rad)를
    **오발**할 수 있다 → 폐쇄가 중간에 얼어붙는다. `_hand_blocked` 의 `_free` 조건이
    부분적으로만 막는다(한계에서 떨어져 있으면 여전히 성립). 탐지기 신설:
    `ctrl/hand_blocked_frac`(물체가 멀 때 > 0 이면 발화) · `ctrl/hand_joint_err_max`.
  · ⚠ 20관절 독립은 08.25 스윕(결합 시너지 기준)의 적용 범위 밖이다 → **그래서 스윕을 다시 뜬다**.

**Check 4 (기존 파괴): ✓ 구조적으로 보장**
  · A(`grasp_kp`): 공유 모듈 `modules/progress_reward.py` **무변경**(git diff 0). A 가 얻은 것은
    이음매 메서드 하나 + 로깅 이름표뿐이고, 이음매는 여분 인자를 버리고 같은 함수를 부른다.
    `test_only_goal_bonus_diverged` 가 나머지 8항 + 되먹임의 수치 동일성을 고정한다.
  · `grasp_fj_rh`(진행 중, 미추적): 과제 의미 값이 전부 leaf 라 상속되지 않는다.
    `test_task_semantics_live_on_the_leaf_...` 가 필드별로 base/leaf 소속을 기계 검사한다.
  · fj_b9 체크포인트는 차원이 바뀌어(22/131/155 → 27/136/160) **호환 안 됨** — FRESH 로 돈다.
    이건 파괴가 아니라 의도된 계약 변경이고, 계약 테스트가 두 차원을 모두 명시한다.

**Check 5 (측정): ✓** — 변경 7종 전부 스칼라가 있다.
  1·2 `reward/goal_bonus`·`task/successes_mean` / 3 `task/tol` / 4 `task/near_goal`+`task/lifted_frac`
  (Check 2 잔여 탐지: near_goal > 0 인데 lifted ≈ 0) / 5 `episode_lengths`+`task/goal_clock_restart`
  / 6 `diag/obj_speed_lifted` / 7 `ctrl/hand_target_step`·`ctrl/hand_blocked_frac`·`ctrl/hand_joint_err_max`.

**판정: ACCEPT** (Check 2 가 4 의 동반 변경을 **조건**으로 한다 — 검증기·테스트로 잠갔다)

예상 지표 이동:
  → `ctrl/arm_target_step × 60`: 0.83 → **0.15 rad/s** (기계적, 스모크에서 즉시 확인)
  → `diag/obj_speed_lifted`: 하락 (팔이 6.7배 느려지고 외란이 1/7.5 — 물체가 손바닥을 타고
     다니던 성분이 줄어야 한다. 이것이 "리프트 후 흔들림" 의 직접 판정 지표다)
  → `episode_lengths`: ~600 → **> 700** (목표당 시계가 일한다는 증거)
  → `task/tol` 도달 속도: **느려진다** (0.1125 출발 + 게이트 3.0). 비교는 반드시 matched-tol.

★**이 감사가 다루지 않는 것**: 주차 흡수점(리프트 전 `lift` 1.0/step 이 위치 무관)은 그대로다.
  오히려 `k_arm` 6.7배 감속이 탐색 반경을 σ 0.41 → 0.061 rad 로 줄여 **악화 위험**이 있다
  (과제 거리 ≈0.28 rad 가 σ 의 4.6배 — 이전엔 1.5배 *작았다*). 이건 보상 문제가 아니라
  액션 스케일 문제이므로 여기서 고치지 않고, 3팔 스윕의 사전 등록 분기로 판정한다:
  **세 팔이 모두** e400 에 `stage/lifted` < 0.02 이면 손이 아니라 팔 속도가 원인이다.

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

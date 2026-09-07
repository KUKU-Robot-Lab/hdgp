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

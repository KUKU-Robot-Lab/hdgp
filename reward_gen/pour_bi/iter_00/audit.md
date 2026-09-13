=== Reward Audit — pour_bi iter_00 (생성 코드 compute_reward.py, 2026-09-13) ===

변경 대상: 신규 보상 전체(15항). 학습 전 최초 audit.
근거: t2r 프롬프트 생성 결과 + 부팅 프로브 실측(시작 palm↔컵 0.160 m · +7 cm 접근 후 ≈0.103 m
      · 파지 게이트 발화 확인). rl_games `reward_shaper.scale_value=0.01` 은 전 항 공통 곱이라
      아래 비율에 영향 없음.

운영점별 스텝 수입(가만히 있을 때 = do-nothing income):
  · 리셋 자세: approach 2×exp(−5·0.160)=0.90/step, 나머지 0 → 무행동 수입 0.90/step (γ=0.998, 지평 500 → ≈450)
  · 양손 파지+10 cm 리프트: approach ≈1.2 + grasp 2.0(+closure ≤1.0) + lift 4.0 = 7.2~8.2/step
  · 정렬(both_lifted ∧ 림 xy<6 cm ∧ dz>0)+tilt 2 rad: +3 +3 → 최대 ≈14/step
  · 붓기: pour_delta 2.5/비드 **일회**(전량 50) + success 10/step(fill≥0.5 유지 동안)

Check 1 (Local Min): ✓(조건부) — "정렬+tilt 유지, 안 붓기" 국소최적 후보. 그 상태 14/step vs
  붓고 나면 14 + 10(success 레벨) + 일회 50. 붓기가 추가 비용 없이 더 크다. 단 tilt 항은 비드
  흐름과 무관하게 tilt 로 지급되므로 **컵이 비어도 만점** — 붓기 뒤 유지되는 것은 의도된 동작.
  비율 판정: 셰이핑 상한 14 vs 목표 수입 10/step+50 → 3배 미만. ✓
Check 2 (Hacking):   ✓(약점 1) — approach 가 게이트 없이 리셋 자세에서 0.90/step 을 준다(전 스텝
  일정 바닥 수입). 컵을 밀어 넘어뜨리면 tilt_premature −1.07/step(누운 컵 1.57 rad) 이 붙어
  "컵 쫓아다니기" 는 손해. 파지 플래그(엄지∧타 손가락 접촉)는 프로브에서 실제로 켜졌다.
  closure 보상(0.5)은 NEAR_PALM 0.10 m 가 실측 파지 거리 0.103 m 와 경계라 **거의 안 켜질**
  가능성 — 치명적이지 않음(상한 0.5), 피드백 라운드에서 지표(reward/grasp_*)로 드러난다.
Check 3 (Grasp):     ✓ — tilt 는 aligned(both_lifted 포함) 게이트 뒤에만 지급. 손은 접촉 동결
  시너지라 tilt 지령이 손가락을 열지 않는다. upright_rcv 는 리시버만.
Check 4 (기존 파괴): ✓ — 신규 트랙, 기존 학습 행동 없음.
Check 5 (측정):      ✓ — 15항 전부 `reward/<항>` 로 TFEvents 로깅(env 가 dict 키 그대로 씀).
  대리 지표 task/src_grasped·rcv_grasped·src_cup_lift·aim_dist·src_tilt_deg·bead/in_target·success_now.

판정: ACCEPT (iter_00 학습 기동 가능). REVISE 후보(피드백 라운드에서 지표로 판단):
  → approach 를 리셋 거리 기준 상대값으로(바닥 수입 제거) — 지표 reward/approach_* 가 학습 내내
    ≈0.9 상수면 이것.
  → NEAR_PALM 0.10 → 0.13 — reward/grasp_* 가 파지 플래그값(≤2.0)에만 머물면 이것.

예상 지표 이동(성공 시나리오): task/src_grasped·rcv_grasped 0 → >0.5 (수천 epoch) ·
  task/src_cup_lift 0 → 0.05~0.10 · task/aim_dist 0.3 → <0.06 · bead/in_target 0 → >0.5.
실패 서명: reward/tilt 상승 + bead/in_target 0 유지 → 빈 tilt 국소최적(Check 1 재감사).

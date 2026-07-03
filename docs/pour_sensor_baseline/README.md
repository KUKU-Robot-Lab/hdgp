# both/pour_sensor — 정직한 양팔 Pour Baseline (test6, ep4250)

**작성일**: 2026-07-03
**태스크**: `open-tesol_b_pour_sensor` (OpenArm 양팔 + Tesollo 5지 손, DirectRLEnv, rl_games PPO, MLP)
**baseline 체크포인트**: `log/rl_games/open-tesol/both/pour-sensor/test6/nn/last_open-tesol_b_pour_sensor_ep_4250_rew_78800.305.pth`

---

## 1. 요약

오른팔이 source cup을 기울여, 왼팔이 든 target cup에 **실제로 붓는** 정직한 양팔 pour 정책. 이전 실패(test2)의 "두 컵을 겹쳐 같은 bead를 이중카운트하는" reward hacking을 물리 충돌 복원으로 제거하고, 학습 가속 후 재개 수렴하여 확보한 baseline.

| 최종 eval (표본 크기별) | 성공률 | 평균 이송 | spill |
|---|---|---|---|
| **128-env, 197 에피소드 (통계 대표값)** | **79.7%** | **15.66/20 (78.3%)** | **11.7%** |
| 16-env, 21 에피소드 | 91.3% | 17.83/20 (89.1%) | 0.2% |
| 4-env, 4 에피소드 | 100% | 19.75/20 (98.7%) | 1.3% |

> **표본 크기 주의**: 소표본(4·16-env)은 쉬운 초기상태만 뽑혀 낙관적으로 나온다. **128-env(197 에피소드)가 다양한 초기상태를 포괄한 신뢰 가능한 실제 성능**이다: 이송 ~78%, 성공률 ~80%, spill ~12%. 개선 여지(특히 spill 감소)는 남아있음.

---

## 2. 문제 → 원인 → 해결

- **문제**: 렌더에서 source cup이 target cup과 물리적으로 겹치는 상태. TFEvents상 `bead_in_source 0.85 + bead_in_target 0.56 = 1.40 > 1.0` (같은 bead 이중카운트), 실제 붓기 0회 = reward hacking.
- **근본 원인**: (a) target cup이 정책 이동가능(왼팔 follow) (b) pour reward가 움직이는 target 프레임 기준 (c) 컵-컵 겹침 무페널티 (d) target cup의 음수 collision offset(-0.1)이 충돌 형상을 소멸시켜 유령화. → "붓기"보다 "target을 bead 밑으로 가져가 겹치기"가 최고 보상.
- **수정 1 (물리 충돌 복원)**: target cup collision offset `-0.1 → rest 0.0 / contact 0.02`. SDF 충돌 활성 → 관통·겹침 물리 봉쇄. reward 항 불변. **weight 50 최대에서도 SUM≤1.0** 검증.
- **수정 2 (학습 가속 3-fix)**: ① pose_success trigger 누적→EMA(윈도우) ② outcome_adr `num_increments 8→4`·`interval 20000→8000` ③ `entropy_coef 0.0008→0.0003`.
- **재개 수렴**: test5(ep1200, 외부 kill) 정책에서 재개(test6). dip 없이 매끄럽게 수렴, ep4250에서 정상 종료(bead_at_done 플래토 + entropy 과학습 구간 접근).

---

## 3. 최종 학습 지표 (test6 ep4250 부근)

| 지표 | 값 | 비고 |
|---|---|---|
| bead_at_done | ~0.93 | 에피소드 완료 시점 이송률 |
| spill | ~0.013 | 흘림 1.3% |
| bead_in_target | ~0.41 | 전 구간 평균(초반 source 포함) |
| bead_in_source | ~0.46 | source 잔량 |
| SUM (src+tgt) | ~0.87 | **≤1.0 = 착취 없음** (test2는 1.40) |
| source_up_dot | ~0.10 | 기울임 유지 |
| mouth_z_clearance | +0.017 | source가 target 위 (양수) |
| grasp_broken | 0 | 전 구간 파지 무붕괴 |
| entropy | ~12.8 | 정밀 수렴 |
| weight_pour_bead | 50 | outcome_adr 완주 |

### 체크포인트 선정 (중요)
`rewards/step`은 ep~3600에 정점(85k) 후 하락했으나 `bead_at_done`은 그 뒤로도 상승 → **rl_games 자동 best(reward 기준)는 최적 pour이 아님.** 후반 체크포인트를 eval로 실측 비교하여 선정:

| 체크포인트 | 성공률 | 평균 이송 | spill |
|---|---|---|---|
| 자동 best (≈ep3600, reward최고) | 90.5% | 17.33/20 (86.7%) | 0.2% |
| ep4200 | 90.5% | 17.00/20 (85.0%) | 0.2% |
| **ep4250 (선정)** | **91.3%** | **17.83/20 (89.1%)** | 0.2% |

---

## 4. 파일

| 파일 | 설명 |
|---|---|
| `video_128env.mp4` | 128개 환경 동시 실행 원거리(다중 뷰) 영상 |
| `video_1env_closeup.mp4` | 단일 환경 근접 영상 |
| `test6_metrics.csv` | 전체 학습 곡선 (ep1~ep4259, per-epoch). 플롯용 |

### CSV 컬럼
`epoch, bead_at_done, spill, bead_in_target, bead_in_source, source_up_dot, mouth_z_clearance, weight_pour_bead, reward_pour, grasp_broken, entropy, rewards_step, SUM_src_tgt`

**플롯 권장 조합**:
- 성능 곡선: `bead_at_done`, `spill` vs epoch
- 착취 감시: `SUM_src_tgt` (≤1.0 유지 확인), `bead_in_source` vs `bead_in_target`
- 수렴: `entropy`, `rewards_step` vs epoch
- 커리큘럼: `weight_pour_bead`, `reward_pour` vs epoch
- 자세: `source_up_dot`, `mouth_z_clearance` vs epoch

---

## 5. 재현

```bash
# 평가 (리포트)
CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p ../hdgp/scripts/reinforcement_learning/rl_games/eval_pour_envs.py \
  --task open-tesol_b_pour_sensor-play --num_envs 16 \
  --checkpoint <ep4250 체크포인트> --headless

# 영상
... 위 명령에 --video --video_length 500 추가 (num_envs 128=원거리, 1=근접)
```

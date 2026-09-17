# IKER 2단계 놓기 — t2r 보상 생성 + 놓음 판정 설계

> 기준: `2026-09-14-iker-auto-loop-design.md`(루프 스펙), `2026-09-13-iker-vlm-keypoint-reward-design.md`(IKER 프레임워크),
> `2026-09-15-iker-stage1-t2r-design.md`(1단계 t2r, 이 문서가 그대로 따르는 형식).
> 이 문서는 2단계(신발을 선반에 놓기)의 **보상 출처·성공 판정·액션/관측**을 개정한다. 1단계 파지·뱅크·VLM 목표·수확은 그대로다.
> 상태: 사용자 결정(2026-09-16) — 설계 1~4절 승인. **이후 같은 날 범위 축소(아래 §0).**

## 0. 범위 축소 (2026-09-16, 사용자 결정 — §2·§8·§12·§13 에 우선한다)

사용자 판단: "IKER 구조가 지금 보면 도움이 안 됨. t2r 로 계속 개선해 나가려고 함. 그래서 현재 변경은 too much 함."

IKER 가 실제로 기여하는 것은 **VLM 이 정한 목표 키포인트 좌표 하나**다. 고정 5항 보상은 1·2단계 모두에서 폐기됐고,
자동 루프는 최근 수십 틱이 `wait` 과 평가 기록이었으며, 키포인트 성공 판정은 놓음을 보지 않아 교체 대상이다.

**그래서 남기는 것:** VLM 목표 좌표(`interaction_vlm.json`, 이미 생성·게이트 통과), §3 성공 판정, §4 액션·관측,
§5 RewardContext, §6 env 훅, §7 프롬프트, §9.1 검증, §9.2 의 `settle` 게이트.

**빼는 것:**
- §2 의 루프 단계 도식과 `stage2_train` 라운드 기계 → **반복은 사람이 손으로 돈다**(프롬프트 → 생성 → 학습 기동 →
  결과 판독 → 재생성). 자동 틱·라운드 게이트·조기 종료·정책 키(`t2r2_*`)를 만들지 않는다.
- §8 라운드 판정과 정책 키 표 전체.
- §12 의 루프 파일(`loop_state.py`·`loop_probe.py`·`loop.py`·`LOOP_PROMPT.md`) 수정.
- §13 전환 절차(트랙 archive·`init`·크론) — 새 트랙을 만들지 않는다. 기존 트랙은 이미 종료·`pause` 상태로 둔다.
- §9.2 의 `wire` 모드는 별도 과제로 두지 않고, 학습 기동 전 1회 점검으로 축소한다.

**유지하는 것 중 하나 추가:** §12 의 `eval_iker.py` `placed`·`retreated` 보고 지표는 남긴다 — 한 줄짜리이고,
없으면 "실제로 놓고 떠났는가"를 숫자로 볼 방법이 없다.

구현 계획은 이 축소를 반영해 **3과제**로 다시 썼다(`docs/superpowers/plans/2026-09-16-iker-stage2-t2r.md`).

## 1. 왜 바꾸나

고정 IKER 5항 보상으로 2단계를 학습한 결과(트랙 `iker_shoe_c00_s2r8w`, 뱅크 429개, 4096 env, ep750 이어받기):

| 결정론 평가(512 env) | ep250 | ep500 | ep750 | ep1000 |
|---|---|---|---|---|
| 성공(종료 시점 키포인트 ≤ 5 cm) | 8.98 % | 16.21 % | 23.63 % | 26.76 % |
| 증가폭 | — | +7.2 | +7.4 | **+3.1** |
| 5 cm 한 번이라도 진입 | 43.4 % | 53.7 % | 67.6 % | 69.9 % |
| 중심오차 중앙값 | 5.74 cm | 5.24 | 5.47 | **5.20** |
| **회전오차 중앙값** | 51.7° | 52.8° | 56.9° | 53.4° |
| 쥔 채 시간초과(실패 중) | 67.6 % | 63.2 % | 67.0 % | **68.0 %** |
| 유지 성공(`sustained`) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

세 가지가 동시에 확인됐다.

1. **자세를 올릴 기울기가 보상에 없다.** 위치는 계속 좋아지는데(중심오차 5.74 → 5.20 cm) 회전은 네 번의 평가에서
   한 번도 시작값 아래로 내려가지 않았다(51.7 → 53.4°). `align`·`dir` 은 키포인트 4개를 평균으로 뭉개고 `dir` 은
   방향만 맞으면 포화하며, `align` 은 이동거리(≈0.25 m)로 정규화돼 10° 개선이 스텝당 0.24 에 그친다 — 같은
   에피소드에 낙하 벌점 750 의 분산이 얹히므로 신호가 묻힌다.
2. **성공 보너스가 지급된 적이 없다.** env 성공(정규화 0.10 ≈ 2.5 cm 를 누적 21스텝)은 결정론 평가 4회 모두
   `sustained 0.0000`. 정렬을 가르칠 유일한 큰 항을 정책이 경험한 적이 없다.
3. **놓을 수단이 애초에 없었다.** 2단계 액션은 팔 6축뿐이고 손가락은 에피소드 내내 뱅크 지령에 고정이다
   (`_pre_physics_step` 은 `_joint_targets[:, _arm_ids]` 만 쓰고, 손 칼럼은 `_reset_idx` 에서 한 번 박힌다).
   평가의 "놓고 종료" 23~26 % 는 의도적 놓음이 아니라 미끄러지거나 밀려난 사고다. 영상
   (`our_source/iker_stage2_s2r8w_ep1000_place_2026-09-16.mp4`)에서 육안 확인 — 제자리까지 가져다 놓고도 놓지 못해
   들고 선 채 끝난다.

또한 성공 판정(`eval_success_distance_m` 0.05, 종료 시점 거리)은 **놓았는지를 보지 않아** 호버링도 성공으로 센다.

사용자 결정(2026-09-16):
1. 2단계 보상을 **t2r 로 통째 생성**한다(IKER 고정 5항을 2단계에서 쓰지 않는다).
2. 성공 판정에 **내려놓고 손 떼기**를 넣는다. 과제는 4단계 — 쥐고 시작 → 정렬·접근 → 내려놓기 → 손 떼고 물러나기.
3. 놓기 수단으로 **그립 축 1칸**을 액션에 더한다(6 → 7). 관측은 그 상태를 보도록 38 → 39.
4. 물러나기는 판정에서 빼고 보상·계측으로 다룬다.
5. 기존 고정보상 런은 종료하고 **하위 지표 비교용 기준선**으로만 쓴다.

t2r 공통 규칙(기존 결정): reward-audit 을 쓰지 않는다 · 생성기 notes 는 확인된 사실만 · 트랙끼리 코드를 공유하지 않는다
(복사 포크) · 보상 루프는 PPO 4096 env · obs/actor 는 sim2real 출처가 있는 입력만(보상 전용 특권 정보는 허용).

## 2. 구조

```
stage2_train(개편) ─▶ observe_requery ─▶ completion_review ─▶ done
   │  라운드 iter_NN (최대 4)
   └─ write_t2r_prompt → t2r_generate → ingest_reward → run_t2r_smoke → launch_stage2
      → run_eval/record_eval(250 간격) → 게이트(평가 성공 ≥ success_target) → advance / end_round
```

- `PHASES` 는 6개 그대로 두고 `stage2_train` 안에 라운드 기계를 넣는다(1단계가 `stage1_t2r` 안에서 수확까지 한 것과 같은 구조).
- `PHASE_RUNS["stage2_train"]` 에 `t2r_smoke` 를 더한다. 하류(`observe_requery` 이후)는 바꾸지 않는다.
- env 원칙: 생성 코드는 **2단계 보상 합만** 바꾼다. 성공·종료 판정은 env 가 소유한다.

라운드 산출물: `loop_<track>/stage2_t2r/iter_NN/{prompt.md, response.md, response_attempt_K.md, compute_reward.py,
validation.json, generator.json, smoke.json, feedback.md}`.

## 3. 성공 판정

아래 다섯 조건을 **최근 W 스텝 중 N 스텝** 만족하면 성공이고 그 스텝에 에피소드가 끝난다.

| 항 | 기준 | 출처 |
|---|---|---|
| 놓인 자리 | 키포인트 4개 평균 거리 ≤ **0.03 m** | `PlaceRewardCfg.place_tolerance` — 2026-09-17 에 0.05 에서 조였다(아래) |
| 손을 뗌 | `palm_shoe_dist` > **0.15 m** | 평가의 `HOLD_RADIUS_M` 과 같은 값 — 두 지표가 같은 기준을 쓴다 |
| 받침에 얹힘 | 신발 hull 최저점이 `RACK_TOP_Z`(0.325 m) ± `resting_tol` | `shoe_meta.hull_local` + `gs.surface_subsample`(256점) |
| 정지 | `shoe_lin_vel.norm()` < **0.05 m/s** | 1단계 `success_speed` 와 동일 |
| 복귀 | 손바닥이 **홈 자세의 손바닥 위치** 0.05 m 이내 | 프로필 `init_joint_pos` 를 env 가 부팅 때 1 물리스텝으로 측정 |

**N = `stable_steps` = 20 스텝(2초)**, **W = `window_steps` = 30 스텝(3초)**.

판정 방식은 세 번 바뀌었다. ①기존 IKER 2단계의 `success_count` 는 조건이 깨져도 0 으로 돌아가지 않는 **누적**
카운터였고(코드 확인), 흩어진 21스텝으로도 성공이 되어 "놓고 버틴다"를 보증하지 못했다. ②그래서 **연속**
카운터로 바꿨다(깨지면 0). ③2026-09-17, 연속 방식이 노이즈에 부서지는 것을 실측했다 — 같은 ep250 정책을
노이즈 off/on 으로 재면 조건 충족률(`ok` 점유율 0.184 vs 0.172, 도달 env 0.520 vs 0.559)은 거의 같은데
**20스텝 완주율만 0.391 → 0.0078 로 50배** 무너졌다(한 스텝만 흔들려도 0 이 되므로). 학습 지표가 실제
실력(39 %)이 아니라 취약성(0.8 %)을 찍고 있었다. 그래서 **창 방식**으로 바꾼다: 최근 W 스텝의 충족 수를
세고, 한 스텝 깨져도 1 만 줄어든다. 흐트러짐은 여전히 걸러내면서 노이즈에는 견딘다.

같은 결정에서 **`place_tolerance` 를 0.05 → 0.03 으로 조인다.** 설계상 두 신발 사이 틈이 2.8 cm 인데
허용오차가 5 cm 여서, 판정상 성공해도 눈으로는 옆 신발에 붙지 않은 것으로 보였다(실측 틈 중앙값 6.1 cm,
키포인트 오차 중앙값 4.6 cm). 창 방식으로 정책이 **놓은 뒤 신발을 밀어 고칠 여지**가 생기므로 도달 가능하다고 본다.

**복귀를 판정에 넣는다(2026-09-17, 앞선 "물러나기는 판정에서 뺀다" 결정을 뒤집는다).** iter_02 ep350 영상에서 신발을
놓은 뒤 팔을 들어 올려 "만세" 자세가 됐다(사용자 관찰). 원인은 놓은 뒤 손을 어디 둘지 알려주는 신호가 없었던 것이다 —
보상 컨텍스트에 복귀 위치 필드가 없었고, `released` 는 방향 무관 거리라 위로 드는 것이 가장 빠른 길이었다
(`place/retreated` 0.79 → 0.05). 사용자 결정: 복귀 위치는 **로봇 기본 자세**, 강제 방식은 **성공 조건**.

관절값이 아니라 **홈 자세일 때의 손바닥 위치**로 잰다. 게이트 프로브(32 env): 홈 손바닥 위치 env-local
(0.290, 0.380, 0.418), 신발 목표 중심에서 0.447 m(손 뗌 0.15 m 와 양립). 손바닥 IK 명령으로 bank 시작 자세에서
홈 손바닥 위치로 가면 9스텝 만에 5 cm 안(최종 오차 중앙값 1.0 cm, p90 3.2 cm)이지만, 그때도 팔 관절 최대 오차가
중앙값 1.01 rad · p90 1.52 rad 였다. 액션이 손바닥 6자유도라 7축 팔의 여유 자유도를 정책이 제어할 수 없어,
관절값 조건은 학습으로 도달할 수 없다. 허용 반경 0.05 m 는 위 프로브의 p90(3.2 cm)에 여유를 둔 값이다.

`retreated`(종료 시 손이 에피소드 시작 위치 근처)는 기존 비교를 위해 보고용 지표로 남긴다.

**종료**는 성공 · 낙하(`fall_height` 아래) · 200스텝 시간초과 셋이고, 생성 보상은 종료를 소유하지 않는다.

**대가:** 이 판정으로 기준선(고정보상 ep1000)을 재면 거의 0 % 다. 대표 지표 비교는 포기하고 중심오차·회전·놓음 비율·
`placed` 같은 하위 지표로 비교한다.

## 4. 액션 6 → 7, 관측 38 → 39

**액션** `action[6] ∈ [−1, 1]` — **−1 = 뱅크 grip 자세 유지, +1 = 프로필 open 자세**. 손 목표는 1단계 법칙을 재사용한다:

```
raw    = grip + (a+1)/2 · (open − grip)                    # 20관절 스칼라 보간
target = clamp(α·raw + (1−α)·previous, 백스톱 한계)          # α = gs.HAND_EMA_ALPHA = 0.468559
```

프로필상 open↔grip 에서 실제로 움직이는 관절은 엄지 `_3`(0 ↔ −1.8)·`_4` 와 4지 `_2`(0 ↔ +1.9)·`_3`·`_4` 뿐이라
스칼라 하나로 충분하다. 리셋 시 `previous` = 뱅크 손 목표이고 **−1 상태(쥔 채)에서 출발**한다.

백스톱은 여는 방향을 막지 않는다: `grasp_stage.backstop_limits` 는 `q_grip < q_open` 일 때 `(hard_lo, q_open)` 을 써넣으므로
엄지 `_3` 는 `[hard_lo, 0.0]` 이 되고, −1.8 → 0.0 은 자유이며 open 을 지나친 역굴곡만 차단된다.

**관측** 38 → 39. 현재 38 = 팔바닥 3+4 · 신발 3+4 · 키포인트 12 · 목표 12. 여기에 **EMA 후 그립 상태 1칸**
(`gs.normalized_targets` 규약, [−1,1])을 더한다. 2단계 정책망은 `mlp [256,128,64]`·`rnn: None` 으로 순환이 없어,
관측하지 않으면 "이미 놓았는지"를 모른 채 놓기 타이밍을 결정하게 된다.

## 5. RewardContext

frozen dataclass, 모든 텐서는 배치 `(N,…)`·env-local·복사본(`clone`), 거리 m · 각 rad · 힘 N.

| 묶음 | 필드 |
|---|---|
| 상수(python) | `table_top_z`, `rack_x_min/max`, `rack_y_min/max`, `rack_top_z`, `episode_steps`(200), `control_dt`(0.1), `place_tolerance`(0.03), `release_radius`(0.15), `resting_tol`, `still_speed`(0.05), `stable_steps`(20), `window_steps`(30), `home_radius`(0.05) |
| 팔바닥 | `palm_pos (N,3)`, `palm_quat (N,4 wxyz)`, `palm_normal (N,3)` |
| 팔 | `arm_q`, `arm_qd (N,7)` |
| 그립 | `grip_norm (N,)` — EMA 후 그립 상태(−1 쥠 … +1 폄) |
| 신발 | `shoe_pos`, `shoe_quat`, `shoe_lin_vel`, `shoe_ang_vel`, `shoe_surface (N,256,3)`, `shoe_bottom_z (N,)`, `palm_gap (N,)`, `palm_shoe_dist (N,)` |
| 목표 | `target_keypoints (N,4,3)`, `keypoints (N,4,3)`, `init_keypoints (N,4,3)`, `keypoint_err (N,4)`, `keypoint_dist (N,)` |
| 상태(env 고정 판정) | `placed`, `released`, `resting`, `still`, `stable_count (N,)`, `success`, `episode_progress` |
| 행동 | `actions (N,7)`, `prev_actions (N,7)`(리셋 직후 0) |

넣지 않는 것: IKER 5항의 항별 값, 접촉 센서(2단계에는 없다), 손가락 20관절 개별 상태(그립 축으로 요약).

## 6. env 훅 (`IkerShoeT2rEnv(IkerShoeEnv)`)

- `__init__`: `super().__init__` 전에 `load_reward_fn(cfg.reward_code_path)`(빈 경로 = 영 보상, 없는 파일 = 부팅 오류),
  `hull_local → surface_subsample`, 그립 버퍼 초기화. 부팅 줄에 보상 코드 경로·sha256 을 찍는다.
- `_pre_physics_step`: 팔 6축은 부모와 동일(DLS IK), `action[6]` 으로 손 칼럼을 4절 법칙으로 갱신.
- `_get_observations`: 부모 38칸 + 그립 상태 1칸.
- `_get_dones`: 3절 판정(창 버퍼 `_place_window`) · 종료 3종.
- `_get_rewards`: `_get_dones` 가 계산한 **같은 스텝 상태**로 `_build_context` → `call_reward_fn` → `nan_to_num`.
- `_build_context`: 위 필드, 모든 텐서 `clone`.
- `_reset_idx`: 부모 리셋 뒤 `prev_actions` 0, 그립 목표를 뱅크 값으로.
- 바뀌지 않음: 뱅크 로드·검증, IK, 중력보상, 장면·이벤트·도메인 랜덤화, 관측 잡음.
- 로그: `t2r_reward/<항>`·`t2r_reward/total`·`t2r_reward/nonfinite_frac`, `place/placed`·`released`·`resting`·`retreated`,
  기존 `iker/*` 는 **유지**(기준선과 하위 지표 비교용).

## 7. 프롬프트

1단계와 같은 절 순서: 역할 → 로봇·행동 → 보상 구조 일반 안내 → `RewardContext` 원문 → 추가 사실 → 과제문·출력 규칙 →
(iter ≥ 1) 이전 코드 + 피드백 표.

- **로봇·행동**: 왼팔 7관절 + DG-5F, 팔바닥 6D 델타(위치 0.02 m/스텝, 회전 0.05 rad/스텝) **+ 그립 축 1칸**
  (−1 뱅크 grip, +1 프로필 open, EMA 0.4686), 10 Hz, 200 스텝. 1단계의 20관절 표는 넣지 않는다.
- **추가 사실**(코드로 확인한 env 사실만): 에피소드는 신발을 **이미 쥔 채** 시작(뱅크 자세) · 목표 키포인트 4개는 VLM 이
  정한 고정 좌표 · 받침 기하(x 0.11~0.43, y −0.33~−0.02, 상면 0.325) · 3절 성공 판정 식(상수는 cfg 값으로 렌더) ·
  종료 3종 · 관측 잡음 0.02(자세 0.2 rad)·액션 잡음 0.05 · ctx 읽기 전용 · 항 이름이 `t2r_reward/<항>` 로 기록됨.
- **과제문**: "Place the shoe on the rack next to the other shoe: align it with the target keypoints, set it down, let go,
  and withdraw the hand."
- **피드백 표 태그**: `t2r_reward/`, `iker/success_5cm`, `iker/keypoint_distance_m`, `iker/dropped`, `place/placed`,
  `place/released`, `place/resting`, `place/retreated`, `episode_lengths`, `rewards`.

## 8. 라운드 판정 (`_stage2_train(state, probe)`)

1. 현재 iter 의 `prompt.md` 없음 → `write_t2r_prompt`(iter 0 render / iter > 0 reflect).
2. `response.md` 없음 → `t2r_generate`(요청 < `t2r2_max_requests`), 넘으면 `pause`.
3. `validation.json` 없음 → `ingest_reward`. `ok=false` 면 `*_attempt_K` 로 옮기고 재생성, 한도 소진 시 `pause`.
4. `smoke.json` 없음 → `run_t2r_smoke`(부수 런, GPU 한도 규칙). 실패 → `pause`.
5. 학습 기록 없음 → `launch_stage2`(새 학습, `env.reward_code_path=<절대경로>`, `--max_iterations t2r2_round_epochs`,
   `--num_envs stage2_num_envs`).
6. 평가: `eval_every`(250) 간격으로 `run_eval` → `record_eval`.
7. 게이트: 평가 성공 ≥ `success_target` → `advance`(`observe_requery`, 학습 PID 종료). 미달이고 라운드 예산 소진 →
   `end_round`(RoundRecord 기록) 후 `iter += 1`. 끝난 라운드 수 = `t2r2_max_rounds` 면 `pause`.
8. 조기 종료: `t2r2_early_epoch`(250) 평가에서 성공 0 **그리고** `placed` 0 이면 신호 없음으로 라운드를 끝낸다.
9. 크래시·조기 사망·판정 태그 없음 → `pause`(로그 끝 줄).

**판정 지표는 결정론 평가다.** 학습 시점 `Episode/iker/success_5cm` 은 이 트랙에서 실제를 3배 과소평가하기도
(2.86 % vs 8.98 %) 과대평가하기도 해 게이트로 쓰지 않는다.

### 정책 키

| 키 | 값 | 뜻 |
|---|---|---|
| `t2r2_round_epochs` | 750 | 라운드 학습 최대 epoch. 4096 env·약 18 s/iter ≈ 3.7 h |
| `t2r2_early_epoch` | 250 | 조기 종료 판정 epoch |
| `t2r2_max_rounds` | 4 | 넘김 없이 채우면 `pause`(750×4 ≈ 15 h) |
| `t2r2_max_requests` | 3 | 한 iter 의 생성 요청(첫 응답 + 재생성 2) |

기존 `stage2_num_envs`(4096) · `eval_every`(250) · `success_target`(0.5) · `bin_epochs` · `side_gpu_limit_mib` 는 유지한다.
`stage2_env_smoke` 는 유지하되 라운드 스모크와 별개다.

## 9. 검증·스모크

### 9.1 `ingest_reward`(Isaac python)

그대로 재사용(수정 0): `static_check`(최상위 `compute_reward(ctx)` 하나 · import 는 torch/math/typing · `exec/eval/open/
getattr` 류 금지 · **ctx 제자리 연산** 금지 · 없는 필드 접근 거부), `dry_run`(3 seed, n=37 — 라운드 스모크의 64와 다른 수라
배치 크기 하드코딩이 양쪽에서 걸린다), `validate`(cpu + cuda — 장치 없는 상수가 cpu 만 통과하고 학습 첫 스텝에서 죽는 것을
잡는다), `cuda_usable`, `ValidationReport`.

2단계용 변경은 둘뿐: 형상 상수(`NUM_ACTIONS 26 → 7`, 손가락·링크 상수 제거, `NUM_KEYPOINTS 4` 추가)와 `make_fake_context`
(5절 필드 집합). 산출물은 동일한 `validation.json {ok, errors, warnings, term_stats, total_stats, fields_used}`.

### 9.2 `scripts/iker/t2r2_smoke.py`

**`--mode wire`**(구현 수용 검사, 1회·시드 고정):

| 검사 | 내용 | 게이트 |
|---|---|---|
| 그립 축이 연다 | 뱅크 자세에서 `action[6]=+1` → 엄지 `_3` 가 −1.8 → 0 부근까지 이동 | 이동량 ≥ 90 % |
| 놓으면 손을 떠난다 | 팔 정지 + 그립 열기 → `palm_shoe_dist` 가 0.15 m 초과 | 비율 ≥ 0.9 |
| 쥐면 유지된다 | `action[6]=−1` 로 200 스텝 | 낙하 ≤ 5 % |
| **놓은 뒤 서 있는다** | 신발을 목표 자세로 두고 손을 치운 뒤 **100 스텝(10 s)** 물리 진행 | 키포인트 중앙값 거리 ≤ 5 cm 유지 |

마지막 항목은 지금까지 **0.1 초까지만 검증된** 가정이다(기존 `env_smoke.py` 는 목표 자세로 옮긴 뒤 `env.step` 을 한 번만
돌린다). 여기서 실패하면 3절 판정이 달성 불가이므로 **구현 착수 전 게이트**로 둔다.

**`--mode round`**(라운드마다, 그 iter 의 보상): 64 env · 150 스텝(무행동 30 + 무작위) · NaN 0 · `t2r_reward/total` 기록 ·
리셋 뒤 `prev_actions` 0 · env 판정 플래그(`placed`·`released`·`resting`·`still`) 계산됨 · 무행동 보상 평균과 조기 개방률은
**보고만**(게이트 아님) → `smoke.json {passed, checks, idle_reward_mean, early_release_frac}`.

출력 규약은 1단계와 같다: `T2R SMOKE CHECK FAILED: <check>` · `T2R SMOKE passed True|False` · 예외 시 `T2R SMOKE FAILED`.

## 10. 실패 처리

| 상황 | 동작 |
|---|---|
| 생성 요청 3번에도 `response.md` 없음 | `pause` |
| ingest 실패 3번 | `pause`(마지막 오류 목록) |
| 라운드 스모크 실패 | `pause`(실패 검사 이름) |
| 학습 크래시·조기 사망 | `pause`(로그 끝) |
| 평가 태그 없음 | `pause` |
| 4 라운드 넘김 없음 | `pause` |
| 부수 런 중 GPU > `side_gpu_limit_mib` | `wait` |

## 11. 테스트

**순수(테스트 먼저):**
- t2r 포크: 컨텍스트 원문 = 프롬프트 env 설명 · 가짜 ctx 형상 = 5절 · 좋은 코드 통과 · 없는 필드/금지 import/ctx 대입/
  제자리 연산/cuda 장치 불일치 거부 · 로컬 텐서 제자리 연산 허용 · 로더 반환 형상 · 빈 경로 = 영 보상 ·
  프롬프트 판정 수치가 cfg 에서 오는지(리터럴 금지) · 피드백 표 렌더.
- **성공 판정**: 한 스텝 깨져도 0 으로 리셋되지 않고 1 만 준다 · 창 밖으로 밀려난 충족은 빠진다 · 흩어진 충족도
  창 안에서 N 을 채우면 성공(누적·연속 두 옛 동작의 회귀 테스트) · 허용오차 0.03 m · 놓음 0.15 m · 얹힘 허용치 ·
  정지 0.05 m/s · 네 조건 AND · 종료 3종.
- **액션·관측**: 액션 7 · 관측 39 · 그립 축이 손 칼럼을 1단계 `hand_targets` 법칙으로 갱신 · 백스톱 한계 안에 머문다 ·
  리셋 직후 그립 = 뱅크 값.
- env 계약(AST): 오버라이드가 `__init__`·`_pre_physics_step`·`_get_observations`·`_get_dones`·`_get_rewards`·
  `_build_context`·`_reset_idx` 뿐 · ctx 는 같은 스텝 · 텐서 `clone` · `reward_code_path` 기본 `""`.
- `loop_state.decide`(가짜 Probe): 8절 전이 전부(프롬프트 → 생성 → ingest → 재생성 한도 → 스모크 → 기동 → 평가 →
  게이트 통과 → advance / 미달 → end_round → 4 라운드 pause · 조기 종료 · 크래시 · 태그 없음).
- `loop.py` 실행기: 기동 인자(보상 경로·max_iterations·num_envs) · reflect 가 `feedback.md` 와 다음 `prompt.md` 를 씀 ·
  `end_round` 기록.

**Isaac:** `t2r2_smoke.py --mode wire` GATE PASS · `--mode round` 를 고정 보상 fixture 로 PASS · 기존 `env_smoke.py` 통과.

## 12. 파일 배치

| 파일 | 역할 |
|---|---|
| `tasks/iker_shoe/t2r2/context.py` | 2단계 `RewardContext`(5절), `context_stub_source()` |
| `tasks/iker_shoe/t2r2/loader.py` | 1단계에서 복사(수정 0) |
| `tasks/iker_shoe/t2r2/validator.py` | 1단계 복사 + 형상 상수·`make_fake_context` 교체(9.1) |
| `tasks/iker_shoe/t2r2/prompts.py` | 2단계 프롬프트(7절) |
| `tasks/iker_shoe/iker_shoe_t2r_env.py`(+cfg·등록) | 6절 훅, gym id `open-sens_l_iker_shoe_t2r`(+`-play`) |
| `scripts/iker/t2r2_smoke.py` | `--mode wire` / `--mode round`(9.2) |
| `scripts/iker/t2r2_reward.py` | `render` / `ingest` / `reflect` CLI |
| `.claude/agents/iker-t2r-generator.md` | 기존 격리 생성기 재사용 |
| `modules/iker/loop_state.py` · `loop_probe.py` · `scripts/iker/loop.py` · `LOOP_PROMPT.md` | 8절 단계·실행기·틱 절차 |
| `scripts/iker/eval_iker.py` | `placed`·`retreated` 보고 지표 추가 |

바꾸지 않는 것: 1단계 `grasp_stage.py`·`iker_shoe_grasp_env.py`·`t2r/`(1단계 포크), 뱅크·수확, VLM 목표·게이트,
붓기 `modules/t2r`, `tasks/grasp_fj_t2r/`. IKER 고정 5항(`modules/iker/reward.py`)은 2단계에서 쓰지 않을 뿐 삭제하지 않는다.

## 13. 전환 절차

1. 이 스펙·구현 계획 승인 → 구현·리뷰(SDD).
2. 기존 트랙을 `git mv` 로 `config_00/archive/2026-09-16_s2_fixed_reward/` 에 옮긴다(`loop/` = 옛 s2r8, `loop_s2r8w/` = 기준선).
3. 정책 키를 추가한다 — **2번을 먼저 하는 이유**가 이것이다. `load_state → validate_state` 가 키 집합을 검사해
   기존 상태 파일이 로드 불가가 된다.
4. `loop.py init --track iker_shoe_c00_s2t2r --state-dir …/loop_s2t2r --phase stage2_train --policy '{…}'`
   (adopt 없음 — 첫 틱이 `write_t2r_prompt`).
5. LOOP_PROMPT 크론 트랙 이름 교체.

## 14. 위험·범위 밖

- **9.2 의 정착 검사 실패** — 목표 자세가 10 초를 못 버티면 3절 판정이 달성 불가다. 구현 착수 전 게이트로 두고,
  실패하면 목표 자세(받침 칸) 자체를 다시 정해야 한다.
- 그립 축이 이동 중 조기 개방을 유발해 파지가 무너질 수 있다 — 라운드 스모크가 `early_release_frac` 을 보고한다
  (설계상 게이트로 막지 않는다).
- zero-shot·audit 없음이라 첫 보상이 호버링에 수입을 줄 수 있다 — 판정에 놓음·얹힘이 들어가 있어 호버링은 성공으로
  집계되지 않지만, 보상 자체는 그쪽으로 샐 수 있다. 피드백 표의 `place/*` 가 드러낸다.
- 기준선과 대표 지표 비교가 끊긴다(3절) — 하위 지표로만 비교한다.
- 라운드당 ≈ 3.7 h, 4 라운드 최대 ≈ 15 h.
- **범위 밖:** 1단계 재학습·뱅크 재수확, VLM 목표 변경, 서버 학습, SAPG, IKER 고정 5항 코드 삭제.

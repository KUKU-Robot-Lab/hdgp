# Vision Student RL Fine-tune (distill→RL) 설계계획

> 2026-07-25. 사용자 목표: **FP 없이 배포 가능한 vision(RGB) 정책**.
> 배경: imitation 레버 4종(seq16·손가락 손실가중·action EMA·RGB crop) 전부 무효.
> student는 action 모방(imit 2.4)·지각(aux 0.017) 완벽하나 task 성공 ~0.05(teacher 0.417의 12%).
> 진단: **imitation은 action을 맞출 뿐 task를 최적화하지 않음** → task reward로 직접 최적화(PPO) 필요.

## 1. 목표·성공 기준

- distill 체크포인트(student_test4 grasp_student_15000, imit 2.4 수렴본)로 init → grasp_v2 task reward로 PPO fine-tune.
- **성공 기준**: deterministic in_success가 baseline 0.05 → **0.20+** (1차), 0.3~0.5(목표). obj_drift 0.10 → 0.05 이하.
- **실패 판정**: 30k iter 내 in_success가 0.12(imitation 천장)를 못 넘으면 중단 → vision 천장 수용 논의.

## 2. 왜 기존 인프라로 안 되나 (조사 완료)

- 표준 rl_games PPO(train.py)의 `RlGamesVecEnvWrapper`가 distill env의 멀티모달 obs dict(벡터+rgb)를 처리 못함(clamp TypeError — DAgger가 wrapper를 우회하는 이유).
- 반면 **dagger.py는 이미 갖춤**: vision obs 처리(rgb/img/aux_info), env.step 롤아웃+reward 수집(`self.current_rewards += rew`), student 모델(value head 존재 — a2c_mono_transformer `_build_value_layer`), done 처리·hidden 관리.
- → **dagger.py 롤아웃 인프라 재사용 + PPO 목적함수 추가**가 최소 경로.

## 3. 아키텍처

### 3.1 파일 구조 (신규 ~400줄, 기존 무손상)
```
source/openarm/openarm/distillation/
  ppo_finetune.py        # PpoFinetune(Dagger 상속): 롤아웃버퍼 + GAE + PPO 업데이트
scripts/distillation/
  run_rl_finetune.py     # 런처(run_distillation.py 복제·수정, torchrun)
```
- dagger.py는 수정 최소화(공유 유틸 노출 정도). rh56f1 등 기존 distill 무영향.

### 3.2 롤아웃 버퍼
- horizon T=16 × num_envs 256. 저장: vec obs(205), **rgb(3×180×320, bf16)**, action(16), logprob, value, reward, done, lstm hidden(스텝별).
- 메모리: rgb가 지배 — 16×256×3×180×320×2B ≈ **1.4GB(bf16)** → 여유(97GB). CNN 특징 재계산 방식 불필요.
- LSTM: distill seq1 레짐과 동일하게 **hidden carry + per-step**(BPTT 없음, 저장 hidden으로 재계산 정합). BPTT는 이번 범위 밖(inplace 이슈 회피).

### 3.3 PPO 목적함수
```
L = L_clip(π, A^GAE) + c_v·L_value + c_ent·H(π) + λ_bc·L_imit + c_aux·L_aux
```
- GAE: γ=0.998, λ=0.95 (teacher lstm cfg 정합).
- clip ε=0.2, c_v=1.0, c_ent=1e-3(작게 — 이미 수렴 정책 보호), grad clip 1.0.
- **λ_bc·L_imit (BC 정규화, 핵심 안정장치)**: teacher 모방 loss를 유지하되 λ_bc 1.0→0.0 anneal(~15k iter). DAPG식 — PPO gradient가 수렴된 파지 거동을 파괴하는 것 방지. teacher forward는 이미 dagger에 있음.
- **c_aux·L_aux (지각 유지)**: aux(object_pos) 회귀 유지(c_aux=1.0) — PPO가 CNN을 지각 망각 방향으로 밀지 않게.

### 3.4 단계별 스케줄 (안정성 설계)
| Phase | iter | 내용 | 이유 |
|---|---|---|---|
| **P0 value warmup** | 0~2k | policy/CNN **동결**, value head만 lr 5e-4 학습 | distill은 value를 안 배움(random) — 나쁜 advantage로 정책 파괴 방지 |
| **P1 PPO+BC** | 2k~17k | 전체 학습 lr 3e-5, λ_bc 1→0 anneal | 정책 근방에서 task 최적화 시작 |
| **P2 순수 PPO** | 17k~ | λ_bc=0, in_success 추세 보며 지속 | task 성공 극대화 |
- KL guard: 스텝 KL(π‖π_init) > 0.05면 해당 업데이트 스킵(발산 방지).

### 3.5 환경 설정
- task=open-tesol_r_grasp_v2-distill 그대로(카메라·145종·**ADR 28 고정**=배포 작동점).
- **RGB crop 유지**(cedf702): 무해했고 aux 미세 개선 — 지각 최적 상태로 시작.
- num_envs 256, seq1 레짐 → GPU ~35GB 예상.

## 4. 검증 계획
1. **정적**: buffer shape/GAE 단위테스트(감쇠 합 검증), logprob 재계산 정합(저장 hidden으로 같은 π 재현).
2. **P0 검증**: value loss 하락, explained_variance > 0.3 도달 확인 후 P1 진입.
3. **P1 초기**: in_success가 0.05 아래로 붕괴하면 λ_bc/lr 조정(정책 파괴 신호).
4. **주기 판정**: 5k iter마다 deterministic play(--play_policy) — in_success·obj_drift·PER-OBJECT.
5. TB 로깅: ppo/clip_frac·kl·value_loss·adv_std + 기존 지표.

## 5. 리스크와 대응
| 리스크 | 대응 |
|---|---|
| PPO가 수렴 정책 파괴 | P0 동결 warmup + λ_bc + KL guard + 낮은 lr |
| CNN 지각 망각 | aux loss 유지 + P0에서 CNN 동결 |
| vision RL 샘플 비효율(256 env) | distill init이라 from-scratch 아님 — 근방 탐색만. 실패시 512 env 검토 |
| sparse-ish success 신호 | grasp_v2 reward는 dense(approach/grasp/lift/goal) — env가 이미 제공 |
| value bf16 불안정 | value head fp32 유지(모델 기존 캐스팅 관례 따름) |

## 6. 공수 추정
- 구현 ~400줄 + 정적테스트: 1세션.
- P0~P1 초기 검증(GPU): 반나절. 수렴 판정까지: 1~2일 GPU.

## 7. 미결정(구현 전 확인)
- init 체크포인트: student_test4(crop, imit 2.4) vs test1(baseline). **test4 권장**(더 낮은 imit·같은 성능·crop 유지).
- P2에서 ADR을 28→풀리게 할지(성공 오르면 게이트 재개) — 1차는 고정.

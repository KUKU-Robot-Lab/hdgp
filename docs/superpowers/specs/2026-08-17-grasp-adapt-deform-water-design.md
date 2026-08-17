# grasp_adapt Phase 4 — 종이컵 + 물 적응 파지 (deform_water)

**작성일**: 2026-08-17
**대상**: `openarm/tesollo/right/grasp_adapt`
**선행**: Phase 2 deformable(lstm_test25, success 0.88) · Phase 3 dynamic mass(massshift2, success 0.96)

---

## 1. 목표

종이컵(변형체)을 **부수지 않고** 잡고, 다음 두 상황 모두에서 **떨어뜨리지 않는다**:

- **(a) 정적** — 리셋 시점에 컵에 담긴 물의 양이 매번 다름 (무게를 모른 채 잡기)
- **(b) 동적** — 들어올린 뒤 물이 차오르며 무게가 증가 (잡은 채로 무게 변화에 대응)

정책은 무게를 **관측하지 않는다**(actor obs 133D tactile-only). 촉각으로 추론해야 하며, 이것이 실기 이식 가능성의 전제다.

## 2. 현 코드의 공백

| # | 공백 | 근거 |
|---|---|---|
| 1 | 종이컵 + 물 조합 config 부재 | `GraspRightEnvCfgDeformable`와 `GraspRightEnvCfgMassShift`가 각각 `NoActorMass`를 상속한 **형제 클래스** → 상호 배타 |
| 2 | 질량이 ADR 커리큘럼이 아님 | `adr_custom_cfg`에 spawn/noise/finger만 존재. bead는 정적 이산 {0,10,20,30} 균등샘플 |
| 3 | 질량 범위가 실물 종이컵과 불일치 | bead 10g × 30 = 300g |

## 3. 선행 실험에서 가져올 교훈

### 3.1 Phase 3 massshift2 (07.30) — 동적 질량 자체는 성립

success **0.96**, 물 부어 무게 2~3배 급증해도 damage 0. 결론: *"세게 쥐어서가 아니라 기술(마찰·자세·tactile 타이밍)로 무게를 견딤"*.

**단, 조건이 다르다** — 이 성적은 아래 두 조건에서 나왔다:

| | massshift2 | 본 설계 |
|---|---|---|
| 컵 | rigid | **deformable 종이컵** |
| actor obs | 134 (**oracle mass 관측**, 당시 configclass 버그) | **133 tactile-only** |

즉 **tactile-only + 변형 종이컵 조합은 미검증**이며 난이도가 더 높다. massshift2를 성공 보증으로 취급하지 않는다.

### 3.2 LR 교훈 — fresh가 아니라 저LR fine-tune

- massshift1(actor LR 3e-4): ep~273 **완전 붕괴** (수렴 정책에 full LR = catastrophic collapse)
- LR **1e-4**로 낮추자 해결 → massshift2가 0.96 달성
- 명시적 결론: **"fresh 대신 fine-tune이 옳았음"**

이 교훈은 **warmstart를 택할 경우에만** 적용된다 — 수렴한 정책에 full LR을 쓰지 말라는 뜻이다.

**08.17 fresh 시도 → 즉시 폭발 → warmstart로 재정정.** 처음엔 07.30 교훈 해석을 "USD 무죄=head 카메라 한정"으로 보고 fresh로 갔으나, 실제로 돌려보니 **ep1부터 reward −1e14로 즉시 붕괴**했다. 격리 실험으로 근본원인을 확인했다:

| 실험 | 조건 | 결과 |
|---|---|---|
| 기존 `deform_ft`(무변경) + 랜덤 액션 | articulated cup | step 0에서 reward −2.77e14 |
| 신규 `deform_water` + 랜덤 액션 | articulated cup | step 0에서 reward −9.98e13 |
| 기존 rigid `grasp_adapt` + 동일 랜덤 액션 | non-articulated | 40스텝 정상(reward −10~3.6) |

세 실험이 같은 결론을 가리킨다: **12패널 spring-articulated 종이컵이 랜덤(미학습) 액션에 물리적으로 못 버틴다.** 손 USD·매니페스트는 무죄(rigid 태스크가 동일 자산으로 안정), 본 설계의 mass 변경도 무죄(기존 코드도 동일하게 폭발). 순수 콜드스타트 취약성이다.

지금까지의 모든 deform 성공 사례(test24/25, massshift2)는 전부 **이미 gentle하게 잡는 법을 아는 정책에서 warmstart**했다 — 즉 이 콜드스타트 상황 자체를 겪은 적이 없다.

→ **warmstart로 되돌린다.** test25 ckpt에서 이어받아 `rl_games_ppo_lstm_deform_ft_cfg.yaml`(actor LR 1e-4)로 fine-tune한다. 위 LR 교훈(fresh 아닌 fine-tune)이 이제 이유가 하나 더 늘었다 — 수렴 정책 붕괴 방지뿐 아니라 **변형 컵 자체가 콜드스타트를 견디지 못한다.**

### 3.3 급격 도입 금지 (grasp_v1)

grasp_v1은 외란을 0에서 선형 램프한다 — 사유: *"급격 도입 시 grip 붕괴 방지"*. 질량 도입에도 같은 원칙을 적용한다.

## 4. USD 교체(DG5FS) 영향 — 재튜닝 불필요

`openarm_tesollo_bi_rl`(dg5f) → `openarm_tesollo_bi_s_rl`(dg5fs) URDF 정량 비교:

| 항목 | bi_rl | bi_s_rl | 판정 |
|---|---|---|---|
| 팔(`_aj_`) joint origin | — | — | **변경 0건** |
| palm→손끝 (손 내부 기하) | 128~200mm | 125~198mm | −1.8~−6.7mm (**사실상 불변**) |
| 손목→palm (마운트 깊이) | 69.8mm | 15.0mm | **−54.8mm** |

**같은 손이 손목에 55mm 더 깊이 부착**된 것이다.

- **재튜닝 불필요**: obs·reward가 전부 palm 상대(`fingertip_pos_rel_palm`, `palm_to_cup_pos`)이고 `PREGRASP_OFFSET`도 palm_link 기준. palm→손끝이 3mm 내로 유지되므로 이 값들은 유효하다.
- **주의**: 같은 palm 목표에 팔이 55mm 더 뻗어야 하므로 손목·전완이 테이블에 55mm 더 근접한다. 근거리+저자세에서 충돌 여유가 준다.

## 5. cup spawn workspace — 값 유지 + 학습 전 probe

| | grasp_adapt | grasp_v1 |
|---|---|---|
| 중심 | x 0.27, y −0.10, z 0.297 | **동일** ("demo 데이터와 일치" = 실기 시연 좌표) |
| ADR 범위 | ±0.01→0.06 | ±0.02→**0.08** |
| 실효 범위 | x 0.21~0.33 | x 0.19~0.35 |

grasp_v1이 더 넓은 범위로도 학습에 성공했으므로 현 설정은 근거가 있고 보수적이다. **값은 유지한다.**

단 `palm_pose_mins` x_min=**0.20** vs spawn 하한 **0.21** → 여유 1cm. USD 교체로 팔 자세 매핑이 달라졌으므로 **Gate 0에서 실측**하고, 경계에 걸리면 palm 박스 x_min만 0.20→0.18로 완화한다(spawn은 불변).

## 6. 설계

### 6.1 새 config 클래스

```
GraspRightEnvCfgDeformable            (기존, 종이컵)
    └── GraspRightEnvCfgDeformableWater   (신규)
            mass_shift_enabled = True
```

태스크: `open-tesol_r_grasp_adapt_deform_water-lstm`
agent yaml: `rl_games_ppo_lstm_deform_ft_cfg.yaml` (warmstart, actor LR 1e-4 · minibatch 65536) — §3.2 참조

기존 `deform_ft`는 **무변경** — test25 재현 경로를 보존한다.

### 6.2 질량 파라미터

| | 현재 | 변경 |
|---|---|---|
| 컵 본체 | 170g | **170g 유지** |
| bead 1개 | 10g | **8g** |
| bead 개수 이산단계 | {0,10,20,30} | **유지** |
| 물 무게 | 0~300g | **0~240g** |

컵 본체를 실물(≈10g)로 낮추지 않는 이유: 12패널 deformable 컵은 `armature=1e-3`으로 겨우 NaN을 막고 있어(Gate A 실증), 질량을 17배 낮추면 물리 폭주 위험이 크다. 정책이 배우는 것은 **무게 변화에 대한 적응**이지 절대 질량이 아니므로 목적은 달성된다.

bead 질량 한 줄만 바꾸는 이유: 개수 이산단계가 `_bead_lvl * 10`으로 코드에 박혀 있어 개수를 손대면 해당 로직까지 고쳐야 한다.

### 6.3 ADR 2단계 게이팅

```python
adr_custom_cfg["mass"] = {
    "bead_count_max":     (0, 30),   # 정적 수위
    "shift_target_count": (0, 30),   # 동적 추가 목표
}
mass_shift_adr_start: int = 25       # 신규
```

| 구간 | 정적 수위 | 동적 물 추가 |
|---|---|---|
| increment 0~25 | 0 → 가득 (램프) | **off** |
| increment 25~50 | 리셋 시 ≤10개로 클램프 | 0 → 가득 (램프) |

**사유**: 정적 수위조차 못 잡는 단계에서 동적 추가가 겹치면 "무게 추론"과 "무게 변화 대응" 신호가 섞인다. 전반부에 다양한 수위를 잡는 법을 익히고, 후반부에 가벼운 컵으로 시작해 물이 차오르는 시나리오로 넘어간다.

기존 knob 3종(spawn·noise·finger), threshold 0.8, 50단계, window 500은 그대로 공유한다.

### 6.4 검증 게이트

| Gate | 내용 | 통과 기준 |
|---|---|---|
| **0** | spawn 도달성 probe (5090, 소수 env) | palm 도달 OK, 테이블 충돌 없음 |
| **1** | 정적만 (shift off) | success ≥ 0.8, buckle 0, damage ≈ 0 |
| **2** | 동적 shift 활성 | shift 후 grip_force 상승, drop 미증가 |
| **3** | ADR 완주 | 전 mass_bin success 균등 |

Gate 1에서 test25 수준(0.88)에 크게 못 미치면 **손 USD 변경이 원인**이므로 중단하고 진단한다.

### 6.5 학습 환경

- 로컬 **RTX 5090 · num_envs 4096** (32GB 제약; test25는 8192에서 31.8GB 사용)
- **시작점: test25 ckpt warmstart** (§3.2). 로그는 `log/`에 새 실험 폴더로 남기고, 아카이브
  (`log_archive/2026-08-17_pre_dg5fs/`)는 보존용으로만 둔다.
- fresh 시도(lstm_test1, 08.17)는 ep1 즉시 붕괴로 폐기 — tfevents만 진단 증거로 남김.

## 7. 변경 파일

| 파일 | 변경 |
|---|---|
| `grasp_right_env_cfg.py` | `GraspRightEnvCfgDeformableWater`, `mass` ADR 그룹, `mass_shift_adr_start`, bead 질량 0.01→0.008 |
| `grasp_right_env.py` | reset/shift 지점 ADR 조회, 게이팅 조건 |
| `config/__init__.py` | 태스크 등록 |

## 8. 비목표 (YAGNI)

- 마찰 ADR 커리큘럼(`adr_physics_cfg` 이식) — grasp_adapt에 EventCfg 구조가 없어 비용이 크고, 현 단계에 불필요
- 컵 본체 질량 실물화 — §6.2 사유
- `--eval_episodes` 정량 모드 지원 — 다물체 grasp_v2 전용이며 본 태스크와 무관

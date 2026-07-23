# pour-v1 Reward Design — 발표용 설명 자료

> **대상 정책**: `open-tesol_r_pour_v1-lstm` (OpenArm 7-DoF + Tesollo, PALM 틸팅, 순수 DRL)
> **태스크**: source cup → target cup 으로 bead(APA proxy 20개) 붓기. **파지는 grasp 정책 warmstart로 이미 완료**된 상태에서 시작 → pour 단계에 집중.
> **제어**: palm 6D pose → **Fabrics IK** → arm 7-DoF (손가락 freeze). 액션이 관절이 아니라 **손바닥 자세(pose)** 를 명령.
> **근거**: `pour_right_env.py` reward 계산부(L1980–2223), `pour_right_env_cfg.py` weight, `CLAUDE.md`.

이 문서는 **전체 학습 흐름을 3-stage(A→B→C)로 재구성**하고, 각 stage의 목표·핵심 좌표·reward 수식·설계 의도를 정리한다. (코드 내부는 stage가 물리적으로 분리돼 있지 않고 **모든 항이 always-on 덧셈** — 아래 stage는 "무엇을 유도하는가"의 개념적 묶음이다.)

---

## 0. 한 장 요약 (전체 흐름)

```
                 [ warmstart: 컵을 이미 파지한 상태 ]
                              │
   ┌──────────────────────────┼──────────────────────────┐
   │ Stage A                  │ Stage B                   │ Stage C
   │ Available Workspace      │ Pre-pour Position         │ Tilting
   │                          │                           │
   │ palm 6D → Fabrics IK     │ pour_point_xyz 를         │ 컵을 깊게 기울여
   │ → 안정적 arm 자세 확보   │ target 입구로 정밀 이송·  │ (deep tilt 135°)
   │ + 파지 유지              │ 유지 + 내회전 접근        │ 실제 bead 배출
   │                          │ (rim 충돌 회피)           │
   │ r_hold, r_grasp          │ r_approach, r_introt,     │ r_tilt, r_align,
   │                          │ r_aim (+corridor gate)    │ r_pour, r_success
   └──────────────────────────┴──────────────────────────┘

total = r_hold + r_grasp + r_approach + r_introt        (A + B)
      + r_tilt + r_aim + r_align + r_pour                (B + C)
      + w_success · r_success                            (outcome anchor)
      − g_ready · w_spill · √spill        (w_spill = 0, OFF)
```

**설계 철학**: DexPour의 4-stage hierarchical reward(approach/grasp/transport/pour)를 pour 태스크에 맞게 재편.
grasp·transport는 **warmstart로 흡수**하고, 붓기 난제인 **정조준(B)** 과 **deep tilt(C)** 에 보상을 집중한다.
모든 항은 **곱셈 게이트 대신 덧셈**으로 합쳐 chicken-and-egg(회전해야 보상↔보상 있어야 회전)를 회피한다.

---

## Stage A — Available Workspace (안정 자세 확보 + 파지 유지)

### 목표
액션이 관절을 직접 흔들지 않고 **palm pose를 명령 → Fabrics IK가 충돌 없는 arm 7-DoF 해**를 찾게 한다.
탐색공간을 손바닥 6D + 잉여 1-DoF로 축소해 **붕괴 없는 안정 학습**의 토대를 만든다.

### 액션 → IK 매핑
$$
a = [\underbrace{a_{0:6}}_{\text{palm pose}},\ \underbrace{a_6}_{\text{nullspace }\alpha},\ \underbrace{a_{7:12}}_{\text{finger (freeze)}}]
$$

- $a_{0:6}$ = palm pose $(x,y,z,e_z,e_y,e_x)$, 정규화 $[-1,1]$ → **EMA smoothing** → **Fabrics IK** → arm 7-DoF.
- $a_6$ = nullspace $\alpha$ (`nullspace_action_scale=1.0`) → 잉여 1-DoF(elbow-swivel) 조절, palm pose는 보존.
- $a_{7:12}$ = 손가락 → **grasp_hold로 freeze**(inert). 파지 자세는 warmstart에서 형성된 것을 고정.

### Reward: 파지 유지 (grasp가 풀리지 않게)
$$
r_{\text{hold}} = w_{gm}\, r_{\text{maintain}} + w_{cm}\, \mathbb{1}_{\text{full}}\cdot g_{\text{contact}} + r_{\text{force\_bal}}\cdot g_{\text{upright}} + w_{\text{curl}}
$$
$$
r_{\text{grasp}} = w_g\,\big(\rho_{\text{contact}} + b_{\text{full}}\cdot \mathbb{1}_{\text{full}}\big),\qquad
\rho_{\text{contact}} = \frac{\#\text{contacts}}{5}
$$

- tilt 인지 게이트: $g_{\text{contact}} = 1 - 0.7\,\tau$, $g_{\text{upright}} = 1-\tau$ — **직립일수록 full grip을 강하게 요구, 깊은 tilt에선 접촉 완화**.
- weight: $w_{gm}=w_{cm}=w_{\text{curl}}=0.5$, $w_g=3$, $b_{\text{full}}=0.5$ (5지 중 4지 접촉 시 완전파지).

> **느낌**: "손은 이미 컵을 잡고 있다. Stage A는 그 파지를 놓지 않으면서, IK가 다룰 수 있는 좋은 팔 자세를 유지시키는 안정화 계층."

---

## Stage B — Pre-pour Position (배출점 정밀 이송 + 내회전 접근)

### 목표
컵의 **배출점 `pour_point`** 를 받는컵 입구($target\_opening$) 위 corridor로 **정밀 이송·유지**한다.
동시에 손바닥을 **내회전(internal rotation)** 시켜, 컵이 target에 접근할 때 **두 컵 rim이 충돌하지 않는** 자세로 진입한다.

### 핵심 좌표 ①: pour_point (배출점) — 정적↔동적 blend
$$
\text{pour\_point}_{xy} = \text{rim\_center}_{xy} + r_{\text{out}}\,|\hat g_\perp^{xy}|\,\hat d_{\text{pour}},\qquad
\hat d_{\text{pour}} = (1-w)\,\hat d_{\text{static}} + w\,\hat d_{\text{dynamic}}
$$

- $\hat d_{\text{static}}$ = 두 컵 중심 방향(자세 무관) — **이송 구간(얕은 tilt)에서 wobble 회피**.
- $\hat d_{\text{dynamic}} = \hat g_\perp^{xy}$ = 중력 최하단 rim 방향(실제 배출구) — **deep tilt에서 정밀 배출점**.
- blend 가중 $w$ = tilt 깊이 **smoothstep** ($\tau: 0.15\,(\approx45°) \to 0.30\,(\approx67°)$). 임계점 점프 없음.
- $r_{\text{out}}=0.045$ (컵 외경).

### 핵심 좌표 ②: corridor score (입구 위 통로 게이트)
$$
s_{\text{corr}} = \text{corridor}(\text{pour\_point},\ target\_opening;\ R,\ z_{\min},\ z_{\max},\ \text{scale})
$$
$$
g_{\text{ready}} = \max\big(s_{\text{corr}},\ \text{latched}\cdot \text{floor}\big) \quad\text{— 한 번 corridor 진입하면 latch}
$$
→ $g_{\text{ready}}$ 는 outcome 보상(r_pour, spill)의 **phase context**로 쓰인다.

### Reward B-1: 이송 (거리 당김)
$$
r_{\text{approach}} = w_d \cdot \exp\!\big(-\lambda\,(d_{\text{approach}} - d_0)_+\big)\cdot\big(f + (1-f)\,\tau_{\text{anti}}\big)
$$

- $w_d=8$, $\lambda=5$(exp 민감도), $d_0=0.03$(rim 반경 안쪽까지 견인), $f=0.4$(anti-floor).
- **positive exp 당김** — 0.3 m 밖에서도 gradient 생존 → 먼 거리 park 방지(먼거리 grad 소실이 과거 실패 원인).
- $\tau_{\text{anti}} = (1-\text{rim\_antiparallel})/2$ = 입구를 마주볼수록(anti-parallel) 보너스.

### Reward B-2: 내회전 접근 (rim 충돌 회피)
$$
g_{\text{introt}} = \sigma\!\left(\frac{\theta_{\text{thresh}} - \cos_{\text{rim\_facing}}}{T}\right),\qquad
r_{\text{introt}} = w_i\, g_{\text{introt}}
$$

- $\cos_{\text{rim\_facing}} = (\text{palm}\,\hat y)\cdot(\text{world}\,\hat x)$ — 손바닥 roll축과 world +x의 정렬.
- **내회전 = $\cos<0$** → $g_{\text{introt}}\to1$. $\theta_{\text{thresh}}=0$(경계 90°), $T=0.4$(완만한 sigmoid).
- $w_i=5$. **always-on 덧셈** — tilt 이전(접근 단계)부터 "내회전이 옳다"를 직접 학습시켜, r_tilt(곱셈)의 chicken-and-egg를 우회.

### Reward B-3: 주둥이 정조준 (aim)
$$
r_{\text{aim}} = w_a \cdot s_{\text{aim}},\qquad
s_{\text{aim}} = \text{corridor}(\text{pour\_point},\ target\_opening;\ R{=}0,\ z_{\min},\ z_{\max},\ \text{scale}_{\text{ADR}})
$$

- $w_a=18$ (파지 $w_g=3$보다 **우위** — 조준을 최우선). $R=0$ → flat-top 없는 **smooth unimodal peak**(gradient-everywhere, 절벽 없음).
- $\text{scale}_{\text{ADR}}$: **ADR로 10→15 점진 상승** → 주둥이를 입구 중심으로 점점 정밀하게 당김.

> **느낌**: "Stage B는 '어디에 부을지'를 맞추는 조준 계층. 배출점을 입구 위 통로에 정확히 세우고(aim), 두 컵이 부딪히지 않게 손바닥을 미리 안쪽으로 돌려(introt) 붓기 자세를 준비한다."

---

## Stage C — Tilting (deep tilt + 실제 배출)

### 목표
컵을 **수평(90°) 너머 dump(135°)까지 깊게 기울여**(deep tilt) 실제로 bead가 source→target으로 넘어가게 한다.
1차 목표는 "못 넣더라도 deep tilt를 지속", 2차 목표는 "실제 bead 진입".

### 핵심 좌표: tilt 정의 (target 상대각)
$$
\text{rim\_antiparallel} = \hat u_{\text{source}}\cdot \hat u_{\text{target}} \quad(\text{두 컵 +z축 내적})
$$
$$
\tau = \frac{1 - \text{rim\_antiparallel}}{2}\ \in[0,1]\qquad (\text{직립}=0,\ 90°=0.5,\ \text{뒤집힘}=1)
$$
$$
\tau_{\text{target}} = \frac{1-\cos(135°)}{2}\approx 0.854,\qquad
\text{tilt\_progress} = \mathrm{clamp}\!\Big(\frac{\tau}{\tau_{\text{target}}},0,1\Big)
$$

- **world-z가 아니라 두 컵의 상대각** 기준 → target 컵이 기울어도 정확한 pour 각도 측정.

### Reward C-1: deep tilt (핵심 구동)
$$
r_{\text{tilt}} = w_t \cdot \text{tilt\_progress}
$$

- $w_t=20$. **latched_ready(corridor)에서 분리** → 정조준 실패해도 tilt 보상 유지(deep_tilt_boot1의 핵심 결정).
- 0→135° **단일 연속 ramp**(과거 2단 A/B 방식의 85° dead-spot 제거).

### Reward C-2: 방향 정렬 (align)
$$
r_{\text{align}} = w_{\text{al}} \cdot \frac{1 + \cos_c}{2},\qquad
\cos_c = (\text{cup\,up}_{xy})\cdot(\text{cup\_center}\to target)_{xy}
$$

- $w_{\text{al}}=5$. DexPour $r_{align}=(1+\cos\theta)/2$ 형태. 방향 앵커를 **cup-center→target**으로 잡아 깊은 전달 자세에서도 부호가 뒤집히지 않고 안정적으로 +.

### Reward C-3: 실제 배출 outcome (r_pour)
$$
r_{\text{pour}} = w_{\text{pb}}^{\text{ADR}} \cdot s_{\text{corr}} \cdot \big(\underbrace{\phi_{\text{bead}}}_{\text{target 내 bead 비율}} + \kappa \cdot \underbrace{\Delta\phi_+}_{\text{신규 진입 증분}}\big)
$$

- $w_{\text{pb}}^{\text{ADR}}$: **outcome ADR로 0→50** — 자세 성공률 80%+ 도달 후 bead 보상 활성(1단계 자세만→2단계 실제 붓기).
- $s_{\text{corr}}$(corridor) 곱 → 입구 위에서만 유효. $\kappa=30$(신규 진입 증분 가중) → "유지 farming" 차단, plateau 해소.
- **z-only 대리 폐기** → 실제 bead outcome만 보상(거짓 조준 farming 불가).

### Reward C-4: 성공 앵커 (r_success)
$$
r_{\text{success}} = \mathbb{1}\big[\phi_{\text{bead}} \ge \phi_{\text{fill}}^{\text{ADR}}\ \wedge\ \text{spill}\le 0.40\ \wedge\ d_{\text{cup}}<0.20\big],\quad
\text{항 가중 } w_{\text{success}}=50
$$

- $\phi_{\text{fill}}^{\text{ADR}}$: 성공 문턱 **ADR 0.20(2개)→0.50(10개)** 자동 상향. shaping이 outcome 앵커 없이 farming하던 문제 해소.

### spill penalty (OFF)
$$
-\,g_{\text{ready}}\cdot w_{\text{spill}}\cdot\sqrt{\text{spill}},\qquad w_{\text{spill}}=0
$$
> spill을 **직접 처벌하지 않아도** 정조준(r_aim)+성공보상이 간접적으로 spill을 억제 (eval spill 10.8%). 직접 처벌은 "pour 회피 local min"을 유발해 OFF.

> **느낌**: "Stage C는 실제로 붓는 계층. 컵을 뒤집을수록(tilt) 보상하되, 방향을 맞추고(align), 진짜로 bead가 넘어갈 때만 outcome 보상(pour)을 준다. 성공은 마지막에 큰 앵커(success)로 못 박는다."

---

## 부록 A. Reward 항 / weight 요약표

| Stage | 항 | 식 (핵심) | weight | 유도 |
|---|---|---|---|---|
| A | `r_hold` | maintain + full·contact_gate + force_bal·upright_gate + curl | 0.5×n | 파지 유지 |
| A | `r_grasp` | $w_g(\rho_{\text{contact}} + b_{\text{full}}\mathbb{1}_{\text{full}})$ | 3 | 접촉·완전파지 |
| B | `r_approach` | $w_d e^{-\lambda(d-d_0)_+}(f+(1-f)\tau_{\text{anti}})$ | 8 | 배출점→입구 이송 |
| B | `r_introt` | $w_i\,\sigma((\theta_{th}-\cos_{rf})/T)$ | 5 | 내회전 접근(충돌회피) |
| B | `r_aim` | $w_a\,s_{\text{aim}}$ (smooth peak, ADR scale) | 18 | 주둥이 정조준 |
| C | `r_tilt` | $w_t\,\text{tilt\_progress}$ (0→135°) | 20 | deep tilt |
| C | `r_align` | $w_{al}(1+\cos_c)/2$ | 5 | 방향 정렬 |
| C | `r_pour` | $w_{pb}^{ADR}\,s_{\text{corr}}(\phi_{\text{bead}}+\kappa\Delta\phi_+)$ | 0→50 | 실제 bead 진입 |
| C | `r_success` | $\mathbb{1}[\phi\ge\phi_{\text{fill}} \wedge \text{spill}\le0.4 \wedge d_{\text{cup}}<0.2]$ | 50 | outcome anchor |
| — | spill | $-g_{\text{ready}}w_{\text{spill}}\sqrt{\text{spill}}$ | 0 (OFF) | 간접 억제 |

## 부록 B. ADR (자동 커리큘럼) — "쉬운 기준→어려운 기준"

| 파라미터 | 램프 | 효과 |
|---|---|---|
| aim scale (r_aim 경사) | 10 → 15 | mouth_xy 0.056→0.026 m |
| fill ratio (성공 문턱) | 0.20(2개) → 0.50(10개) | 문턱 상향에도 성공률 0.88 유지 |
| pour_bead weight (r_pour) | 0 → 50 | 자세 성공 80%+ 후 bead 보상 활성 → bead_at_done 0→0.88 |

## 부록 C. 설계 원칙 3가지 (발표 강조점)

1. **덧셈 always-on 구조** — 곱셈 게이트의 chicken-and-egg(회전↔보상)를 피하려, introt/tilt/aim을 모두 덧셈으로 합침. tilt를 corridor에서 분리(deep_tilt_boot1)한 것이 110° 벽 돌파의 결정타.
2. **outcome 보상 = farming 불가** — r_pour를 z-only 대리에서 **실제 bead 진입**으로 재설계. 거짓 조준으로 보상을 훔칠 수 없음 → TB `bead_at_done`(0.88)과 eval 이송률(88.4%)이 1%p 내 일치.
3. **간접 억제 > 직접 처벌** — spill을 직접 벌하면 pour 회피 local min. 대신 정조준(aim)+성공앵커(success)로 "정확히 부으면 spill이 준다"는 인과를 학습시킴.

---

*근거: `pour_right_env.py` L1480–1590(좌표) · L1980–2223(reward) · `pour_right_env_cfg.py`(weight/ADR) · `pour_v1/CLAUDE.md` · `pour_v5/CLAUDE.md`. 결과 검증은 `docs/eval/pour_v1_report.md` 참조.*

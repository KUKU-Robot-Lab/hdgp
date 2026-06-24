# β(pour-point 보존 deep tilt) 실현 방식 검토

> 작성 2026-06-24. "approach로 pour-point 맞춘 뒤, pour-point 고정한 채 잉여 조인트를 풀어
> deep tilt" — 이 β의 본래 의미를 어떻게 구현하나. 구현(A) vs 대안(B) 비교.

## 폐기된 방식 (이번 수정으로 제거)
**R(β) 절대 cspace bias + post-IK j5 override.**
- R(β): demo 절대 관절각을 cspace로 부과 → palm task(pour-point)와 경쟁해 주둥이를 끌어내림.
- j5 override: IK가 푼 7관절 해에서 j5만 덮어씀 → end-effector 포즈 파괴 → 주둥이 이탈.
- 검증: v6 ready=0.89(조준됨)인데 β억제로 frac_110=0, corridor 0.55→0.146 진동, bead_in=0.
- **공통 결함: pour-point를 보존하지 않음** (절대 관절각 강제 = demo 기하로 끌림).

---

## 실현 A — rim-pivot tilt setpoint (이번에 구현)

**기전:** β → 목표 tilt_amount. `delta[:,4]`(tilt_toward)를 피드백 구동(Kp·(목표−현재)).
회전 R을 rim 기준 pivot으로 적용: `palm_ee_target = pour_point_target − R·rim_rel`.
→ Fabrics IK가 palm pose(주둥이 고정) 추종하도록 7관절 해를 품.

**pour-point 보존:** 기하적·정확. rim_rel 레버를 R로 회전시켜 palm_ee를 역산하므로
임의 회전에도 주둥이가 명령 pour_point에 유지(1-step 정확, IK 추종오차만큼만).

| 장점 | 단점 |
|---|---|
| 기존 rim-pivot 재사용(최소 변경) | IK가 "rim 유지+deep tilt" 포즈에 **도달 가능**해야 함 |
| 주둥이 위치 **정확 보존** | 도달 불가 시 palm_clamp_viol↑(기계적 막힘) |
| Jacobian 불필요·수치 안정 | tilt 축 고정(target 방향). 잉여자유도는 IK nullspace(demo prior)가 암묵 처리 |
| β 해석 명확(tilt 깊이) | "남은 조인트 풀기"가 명시적이지 않음(IK에 위임) |

---

## 실현 B — 3D pour-point 위치 nullspace (대안)

**기전:** 주둥이 3D 위치의 Jacobian `J_p`(3×7) 계산 → nullspace `N(J_p)`는 **4차원**(7−3).
"cup tilt를 깊게 하는 관절속도"를 N(J_p)에 투영한 방향 `n_tilt`를 구해, β만큼 그 방향으로
관절 이동: `Δq = β_step · n_tilt`, `J_p·Δq ≈ 0`(주둥이 위치 1차 불변).

**pour-point 보존:** 구성상(J_p·Δq=0). 단 **위치(3D)만** 보존, **방향은 자유** → cup이 기울 수 있음.
(주의: 6D palm nullspace n_demo는 방향까지 보존 → tilt 불가. 그래서 3D 위치 nullspace여야 함.)

| 장점 | 단점 |
|---|---|
| 사용자 의도 **문자 그대로**(주둥이 고정+잉여 4-DOF 풀기) | 매 스텝 `J_p` 추출 필요(Fabrics FK Jacobian) |
| 4-DOF 잉여 전부 활용 → 어깨·팔꿈치 최적 recruit | 1차 보존 → 유한스텝 drift, 보정항 필요 |
| 특정 palm target IK 가능성에 비의존 | `n_tilt` 선택(tilt 최대화 투영)·특이점 처리 수치 복잡 |
| 더 깊은 tilt 가능역 탐색(A가 막히는 곳) | β 해석이 간접(nullspace 방향 크기) |

---

## 비교·권고

| 기준 | A (rim-pivot setpoint) | B (3D nullspace) |
|---|---|---|
| 주둥이 보존 | 위치+방향(palm target) 정확 | 위치만(1차), 방향 자유 |
| 잉여 활용 | IK nullspace 암묵(demo prior) | 명시적 4-DOF |
| 구현 복잡도 | 낮음(기존 재사용) | 높음(Jacobian·투영·보정) |
| 도달 가능역 | IK 추종 한계에 종속 | 더 넓음(redundancy 직접 탐색) |
| 위험 | 낮음 | 중(수치·drift) |

**권고: A 먼저(구현됨).** 정확·저위험·기존 자산 재사용. β가 pour-point 보존하며 tilt를
강제하므로 부트스트랩 데드락(bead_in=0→β억제)을 깰 수 있는지 먼저 검증.

**A가 막히면(palm_clamp_viol↑·IK가 rim 유지+deep tilt 도달 실패) B로.** B는 4-DOF redundancy를
직접 탐색해 A가 못 가는 deep-tilt 포즈를 찾을 여지가 큼.

**v4(joint-PD)와의 관계:** v4는 7관절을 정책에 직접 줌 → 정책이 reward 신호로 **B의 해를
스스로 발견**하게 함(구조 부과 없이). A/B=구조 부과, v4=학습으로 발견. 3-way가 이 축을 가름:
- v5(A): pour-point 기하 보존 + IK
- v4: 직접 관절(구조 무부과)
- (B는 미구현 — A 결과 보고 결정)

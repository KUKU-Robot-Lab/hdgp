# 여자유도 baseline이 deep-tilt 조준 정밀도에 미치는 영향 (A2)

**목적**: 논문 Table I의 `NS_naive 0.0%`에 대한 예상 반론 — *"학습이 덜 된 것 아니냐"* — 에
운동학적 근거로 답한다.

**방법**: 순수 FK + DLS IK (Isaac/GPU 불필요). `scripts/probes/probe_nullspace_tilt_reachability.py`
산출물 `nullspace_tilt_reachability.json`, 그림 `fig2b_redundancy_aim_error.png`.

---

## 1. 먼저 기각된 가설 — "robot_start는 deep tilt에 도달할 수 없다"

`pour_right_constants.py:113`의 설계 주석은 *"robot_start(j4=0.60)로 nullspace를 풀면 j6가
포화되어 tilt 막힘"* 이라고 서술한다. 이를 검증하기 위해 baseline이 j1–j4를 앵커한 상태에서
손목(j5–j7)을 관절 한계 전 구간 격자 스윕(1,395,009 configs/baseline)했다.

| baseline | 자유 손목 최대 tilt |
|---|---|
| robot_start | 173.9° |
| demo | 165.4° |

**→ 가설 기각.** 손목이 자유로우면 두 baseline 모두 deep-tilt 임계(100°)를 훌쩍 넘는다.
따라서 **"도달 불가"로 NS_naive 실패를 설명할 수 없다.** 논문에도 그렇게 쓰면 안 된다.

## 2. 실제 구조 차이 — 조준점을 고정하면 드러난다

붓기 중에는 손목이 자유롭지 않다. palm 6-DoF task(주둥이를 receiver 입구 위에 유지 + 필요한
기울기)가 손목을 구속하고, **남는 자유도만** cspace 인력이 baseline 쪽으로 끈다
(`pour_right_env.py:1546~1566`). 이 구조를 IK에 그대로 반영했다 —
`dq = J⁺e + (I − J⁺J)·k(q_baseline − q)`.

조준점 p\* = demo pour 자세의 palm 위치로 고정하고 tilt θ를 60→130° 스윕:

| θ | robot_start | demo (ours) |
|---|---|---|
| 60° | 5.87 mm | 2.14 mm |
| 80° | 6.98 mm | 3.05 mm |
| 100° | 8.17 mm | **0.55 mm** |
| 110° | 8.69 mm | 0.75 mm |
| 130° | 9.58 mm | 3.64 mm |
| **평균** | **7.80 mm** | **2.29 mm** |

회전오차는 양쪽 모두 ≈0.1°로 동일 — **차이는 전적으로 조준(위치) 오차에서 난다.**

**두 가지 관찰**:
1. robot_start의 조준오차는 tilt가 깊어질수록 **단조 증가**(5.9→9.6 mm)한다.
2. demo의 오차는 **deep-tilt 구간에서 최소**(100~110°에서 0.55~0.75 mm)다 — 시연이 기록된
   바로 그 자세에서 여자유도가 정확히 해소된다.

## 3. 튜닝 인공물이 아님

null-space 인력 이득을 0.1~0.8로 바꿔도 결론이 유지된다(비율 단조·항상 2배 이상).

| gain | robot_start | demo | 비율 |
|---|---|---|---|
| 0.10 | 1.62 mm | 0.83 mm | 2.0× |
| 0.20 | 3.22 mm | 1.27 mm | 2.5× |
| 0.35 | 5.54 mm | 1.77 mm | 3.1× |
| 0.50 | 7.80 mm | 2.29 mm | 3.4× |
| 0.80 | 12.14 mm | 3.75 mm | 3.2× |

⚠️ 임계값(“몇 도까지 해가 존재하는가”)으로 보고하지 않는다 — 허용오차 5 mm 기준에서는
robot_start 0° / demo 130°로 **극단적으로 갈라져** 실제 격차(3.4배)를 과장한다. 등급형
지표(잔여 조준오차)로만 인용한다.

---

## 논문 활용

- **Fig.2(b)**로 삽입 — 개념도만 있던 Fig.2에 정량 근거가 붙는다.
- **본문 표현(권장)**: "The redundancy baseline does not change whether a deep tilt is
  kinematically reachable — both baselines can reach beyond 160° when the wrist is
  unconstrained. What it changes is the **aiming precision retained while tilting**: under the
  spout-position task, the robot-start prior leaves a residual spout offset that grows
  monotonically with tilt (5.9→9.6 mm), whereas the demonstration prior is minimal exactly in
  the deep-tilt regime (0.55 mm at 100°) — a 3.4× difference that is stable across null-space
  gains."
- **주의**: 이 결과는 NS_naive의 0%를 *단독으로* 설명하지 않는다. 조준오차 8 mm는 receiver
  opening 반경(~41 mm) 대비 치명적이지 않다. 정직한 서술은 **"운동학적 불가능이 아니라,
  정밀 조준과 deep tilt를 동시에 만족시키는 configuration이 robot-start prior 주변에는
  없어 탐색이 그쪽으로 가지 않는다"** 이며, 학습곡선(Fig.4a, 6,500 iter 내내 평탄)이 보조 근거다.

# 학습 효율 정량화 (B3)

**동기**: Introduction이 task-space 정식화의 이점으로 *learning efficiency*를 주장하는데
논문에 수치가 없다. 리뷰어가 "근거는?"이라고 물으면 답할 것이 없었다.

**방법**: 각 조건의 학습 TFEvents에서 `log/adr_ep_success_rate`가 임계에 **처음 도달한
iteration**을 추출 (GPU 불필요, 후처리만). 스크립트는 `scripts/tools/parse_tfevents.py`.

---

## 결과 — 임계 도달 iteration

| 조건 | 총 iter | 최종 | 최대 | ≥0.10 | ≥0.30 | ≥0.50 | ≥0.70 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **NS_demo (ours)** | 6,630 | 0.822 | 0.822 | 679 | 1,083 | 1,624 | **3,396** |
| **NS_naive** | 6,792 | 0.063 | **0.164** | 200 | — | — | — |
| **JS (joint-space)** | 6,602 | 0.717 | 0.717 | 205 | 1,594 | 2,544 | **6,069** |
| **Full (M4, +boot)** | 6,791 | 0.849 | 0.849 | 154 | 211 | 617 | 1,597 |

## 해석

1. **NS_demo는 JS보다 1.8배 빠르다** — 0.70 도달에 3,396 vs 6,069 iteration.
   Introduction의 learning-efficiency 주장을 뒷받침하는 인용 가능한 수치다.

2. ⚠️ **효율 우위는 초반이 아니라 높은 임계에서 나타난다.** 오히려 ≥0.10 도달은
   JS(205)·NS_naive(200)가 NS_demo(679)보다 **빠르다**. 노션 마스터 페이지의
   *"NS_demo 빠른학습(ep0 상승) / JS 느림(ep1100 돌파)"* 서술은 이 지표와 맞지 않으므로
   **정정이 필요**하다. 정확한 표현: "얕은 성공은 모든 정식화가 빠르게 얻지만,
   실제 이송에 필요한 높은 성공률까지 가는 데는 JS가 1.8배 더 걸린다."

3. **NS_naive는 미학습이 아니라 천장에 막힌 것** — 6,792 iteration을 돌고도 최대 0.164,
   최종 0.063으로 오히려 하락. 초반 0.10은 200 iteration에 도달했으므로 "학습이 시작조차
   못 했다"가 아니라 **"초기 진전 후 구조적으로 정체"**다. 이것이 Table I의 0.0%가
   under-training 아티팩트가 아니라는 직접 근거다 ([nullspace_aim_error_analysis.md](nullspace_aim_error_analysis.md)의
   기하 분석과 상보적).

4. **boot(M4)의 효과는 전 구간 가속** — 모든 임계에서 최소 2배 빠르다. Table I의
   complete-transfer 우위(90.2 vs 28.3)와 정합.

## 논문 반영

- **Table I에 열 추가**: `iters to 0.70 ADR success` — NS_demo 3,396 / JS 6,069 / Full 1,597 / NS_naive n.a.
- **본문(Experiments)**: "NS_naive를 6,792 iteration까지 학습시켰으나 ADR success는 0.164를
  넘지 못했다"를 명시해 under-training 반론을 선제 차단.
- **⚠️ 주의**: `adr_ep_success_rate`는 관대한 학습 프록시다(NS_naive 0.164 ↔ 결정론 eval 0.0%).
  효율 비교에만 쓰고 성능 주장에는 쓰지 않는다.

원자료: `scratchpad/sample_efficiency.json`

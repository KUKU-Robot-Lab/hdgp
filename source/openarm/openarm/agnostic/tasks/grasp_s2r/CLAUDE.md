# grasp_s2r — 태스크 규칙

> 상위 규칙: [hdgp/CLAUDE.md](../../../../../../CLAUDE.md) (로그 먼저 · 증거 우선순위 · 분석 원칙)

## 태스크 정체성

**제자리 파지 → 리프트 → 목표 지점 이송 → 정지.** 팔은 처음부터 끝까지 정책이
Fabrics 를 통해 제어한다. 손은 관절공간 시너지(접촉 동결로 감쌈 생성).

계보:
- **제어 스택** = `agnostic/tasks/grasp_sensor` (Fabrics 팔 + 시너지 손 + 부팅 정합 게이트)
- **액션 규약·보상 8항** = `tesollo/right/grasp_v1` (grasp→lift→stabilize 98% 이력)
- **이송 2항(transfer·stay)·성공 재정의** = 이 트랙 신설

## 이 트랙을 만든 이유 (되돌리기 전에 읽을 것)

`grasp_sensor` 는 palm 액션이 **절대 매핑**(`palm = scale(a, 박스전체)`)이다. 저장소
공통 설정인 σ=1.0(`sigma_init const_initializer val:0` + `fixed_sigma: True` — 전 트랙
동일)과 곱해지면 매 스텝 목표가 작업공간 전역에서 재추첨된다. 08.27 실측: 클램프 전
지령 요청량이 **0.33~0.36 m/step 상시 포화**, 리미터는 속도만 자르고 방향의 무작위성은
못 막아 접근이 제자리 랜덤워크가 됐다.

grasp_v1 규약은 `palm = 홈 + delta(a)` 라 **a=0 이 홈**이고 탐색이 유계 오프셋으로
묶인다. 같은 σ 에서 이 문제가 없다. → **팔 액션을 절대 매핑으로 되돌리지 말 것.**

## 핵심 계약 (계약 테스트가 잠근다)

| 계약 | 왜 |
|---|---|
| palm = **홈 기준 델타**, 홈은 박스에 안 잘림 | 위 랜덤워크 |
| 래치는 **보상 단계 표시 전용** | grasp_v1 의 z 램프 스크립트를 걷어낸 것이 이 트랙의 목적 |
| fabric 손 상태를 **실제 자세로 동기화** | 끊으면 fabric 이 없는 자기충돌을 피하려 팔을 민다(실측 palm_err 475mm·5kN) |
| 적분은 `_step_fabric` 한 곳 | `_apply_action` 에서 돌리면 fabric 시간이 2배 |
| body **하나당** 접촉 센서 하나 | 다중 body 단일 센서는 `force_matrix_w` 무증상 0 |
| goal 은 **정착고** 기준 | 스폰 패드가 리프트 기준에 실리는 이중 패딩 사고 |
| 밀림 감쇠는 **래치 시점 스냅샷** | 실시간 변위를 쓰면 과제인 수평 이송이 처벌된다 |
| obs 에 물체 정체성 없음 | 배포 시 알 수 없는 정보 |
| 손 제어 분기는 시너지 하나 | 죽은 분기는 고칠 때 오해만 만든다 |

## 형상 의존값

**`object_grasp_z_offset = 0.03`(물체 원점↔파지 높이) 하나뿐이다.**

08.27 에 `object_grasp_radius`·`enclosure_thumb_weight` 를 제거했다. 접근 항이
컵 반경 기반 "대향 목표점"이 아니라 **손 자신의 대향 중점**(엄지 팁과 4지 팁 평균의
중점)과 컵 사이 거리(`cage_dist`)를 쓰기 때문이다. 구 수식은 대향축을 접근방향의
90° 회전으로 잡아 **좌/우 부호가 임의**였고, 그래서 엄지 목표가 실제 엄지의 반대편에
놓여 손목을 뒤집어야 도달 가능한 자세를 요구했다 — 사용자 GUI 관찰 "4지는 붙는데
엄지가 걸린다"의 원인이고, 실측으로 `grip_frac 0.20` 인데 `wrap_frac` 이 2,228 iter
내내 0.000 이었다.

## 핵심 지표 (TFEvents)

| 키 | 봐야 할 것 |
|---|---|
| `stage/{grasp,lift,transfer,stay}` | 사다리 도달률 — **단조**여야 한다 |
| `task/wrap_frac` | 감쌈 깊이(per-finger mid AND dist). 리프트 후 침식하면 `wrap_retention` 가 작동하는지 |
| `task/goal_dist` · `task/stay_run` | 이송·정지 |
| `fabric/palm_cmd_step_raw` | 클램프 **전** 요청량 — 리미터 포화율 |
| `fabric/joint_err_mean` · `palm_err_mean` | 팔 추종. 정상 0.06 rad / 0.16 m 대역 |
| `fabric/joint_err_max` | ★오픈루프 적분 격차의 **최대**. 평균은 막힘 구간을 묻는다 |
| `task/close_gate` · `task/cage_ctr_dist` | 닫기 게이트. ★중심은 palm 강체·거리는 3D — 팔이 정지한 구간에서 `syn_close` 와 상관이 붙으면 되먹임 재발이다 |
| `gate/contact_persistence` | ★`grip_frac` 과 **동행**해야 한다. 혼자 1.0 으로 가면 소수 손가락 파밍이다 |
| `gate/disp_factor` | 래치 시점 밀림 감쇠 |
| `task/abnormal_rate` · `contact/force_max` | 물리 건강 |

## 수정 규칙

1. reward/gate/weight 변경 전 **reward-audit 통과** (`~/.claude/skills/reward-audit/`).
2. obs/action 차원 변경 금지(명시 요청 없이). 변경 시 warmstart 전면 무효.
3. 한 번에 하나의 가설. 변경 후 **예상 지표 이동 방향**을 명시할 것.
4. 판정은 **학습 스모크의 건강 지표**로 한다 — zero-action probe 는 판별력이 없다
   (08.27 에 v4/v5 를 구분 못 해 오진을 유발했다).
5. 학습 스모크 게이트: `ep_len` 상승 · `abnormal ≈ 0` · `joint_err < 0.1` ·
   `palm_err < 0.20` · `force_max < 50N`.

## 커리큘럼

ADR 은 **끄고 시작**(`enable_adr=False`). 과제 성립 확인 후 스폰 범위부터 켠다.

## 정비 이력

- 2026-08-27 트랙 신설. 단일 컵 · ADR OFF · GUI 지령 마커.

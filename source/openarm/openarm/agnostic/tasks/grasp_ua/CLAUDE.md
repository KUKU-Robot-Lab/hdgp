# grasp_ua — 태스크 규칙

> 상위 규칙: [hdgp/CLAUDE.md](../../../../../../CLAUDE.md) (로그 먼저 · 증거 우선순위 · 분석 원칙)

## 이 트랙이 grasp_s2r 에서 갈라진 이유 (2026-09-02 신설)

`grasp_ua` = **UnderActuated**. `grasp_s2r` 를 그대로 복사한 뒤, **제어 자유도 <
물리 자유도**인 손(Inspire RH56F1: 구동 6 / 물리 12)을 일반화한 것이다. 자유도가
낮아서가 아니다 — 자유도만 낮은 손(2지 그리퍼)은 원래 프로필 하나로 됐다.

`grasp_s2r` 는 **손대지 않는다**(우팔 tesollo 배포 라인 보존). 두 트랙은 프로필
`tesollo_right` 를 공유값으로 갖고, 이 트랙에서도 `open-sens_r_grasp_ua*` 로 같은
로봇을 돌릴 수 있다 — **이식이 아무것도 안 깨뜨렸는지 재는 대조군**이다.

갈라지며 생긴 일반화 5가지(전부 기본값 = 현행 거동):

| 축 | 필드 / 스위치 | 왜 |
|---|---|---|
| 언더액추 | `hand_mimic` · `hand_mimic_mode` | USD 에 PhysxMimicJoint 가 없으면 종속관절이 init 각도에 굳어 **실기에 없는 손**을 학습한다 |
| 마디 접미사 | `hand_freeze_mid_suffixes` | 코드에 박혀 있던 `"3"` 축출. 손가락당 구동관절이 하나면 중간마디 동결이 아무 관절에도 안 걸린 채 조용히 꺼진다 |
| palm 프레임 | `palm_frame_remap` | `palm_body` 는 fabric 제어점과 같아야 부팅 FK 게이트를 통과한다. 프레임 축 배치 차이는 여기서만 흡수한다(열0 = 손바닥 법선) |
| fabric cspace | `fabric_arm_slot` · `fabric_hand_slot` | RH56F1 fabric 은 **양팔 26 DOF** 다(tesollo 는 우측 27 단독) |
| 생성자 차이 | `fabric_kwargs` + `_filter_fabric_kwargs` | fabric 클래스마다 시그니처가 다르다. 미지원 인자는 중립값일 때만 버리고 아니면 죽인다 |

감쌈 판정은 **새 모드를 만들지 않았다** — `envelope_metric` 스위치가 이미 있고,
프로필의 `(중간, 원위, 팁)` 링크 삼중이 마디 수 차이를 흡수한다. 로봇별 기본값은
EnvCfg 서브클래스가 정한다.

## 학습 기동 (RH56F1)

★기동 전 확인 3가지 — 하나라도 빠지면 조용히 다른 걸 학습한다.

```bash
cd ~/rl_ws/hdgp
# ① 계약 테스트 (Isaac 불필요, 2초)
python3 -m pytest source/openarm/openarm/agnostic/tasks/grasp_ua/tests -q
# ② 자산 shape 회계 — 둘 다 OK 여야 events 를 켤 수 있다
~/rl_ws/IsaacLab/isaaclab.sh -p scripts/probes/probe_usd_shape_audit.py
# ③ 기동
RUN_LABEL=<라벨> PYTHONPATH=$PWD/source/openarm \
  ~/rl_ws/IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task open-rh_r_grasp_ua-lstm --headless --num_envs 2048
```

- `num_envs` 는 **1024 의 배수**여야 한다(rl_games `batch_size % minibatch_size == 0`).
- `HDGP_S2R_REAL_GAINS=1` 을 붙이면 팔이 벤더 게인으로 바뀐다. **kd 는 벤더값이지
  r2s 정합값이 아니다**(RH56F1 손으로 재식별 전) — 붙일지는 의도적으로 정할 것.
- 로그는 `log/rl_games/open-rh/right/grasp-ua/<RUN_LABEL>/` 로 떨어진다.

### 기동을 막는 것 (2026-09-02 현재)

| | 상태 |
|---|---|
| 홈/박스/앵커/스폰 | ✅ 캘리브 완료 — 완화 없이 부팅된다 |
| mimic 결합 | ✅ physx, 오차 0.07 mrad |
| 물체 뱅크 | ✅ `shaker_family` 지름 70~97mm |
| 이벤트(마찰·질량·게인 DR) | ✅ `robot_material` 을 body-무관으로 돌려 자산 결함을 우회 |

★자산 shape 회계(460 vs 459 · `l_al_3`=14 vs `r_al_3`=15)는 **아직 미해결**이다.
지금은 `robot_material` 이 `body_names` 없이 전 shape 에 균일 적용하므로 우회되고
물리도 동일하다(마찰 범위가 단일값이라). ⚠**링크별로 다른 마찰**을 주려면 그때는
자산부터 고쳐야 한다 — `scripts/probes/probe_usd_shape_audit.py` 가 판정한다.

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

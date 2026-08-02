# 자산별 액추에이터 calibration

각 로봇 자산 폴더의 `calibration/` 안에 **부위별** 실측 게인이 들어 있다.

```
assets/robot/<asset>/calibration/
    right_arm.json      오른팔 7관절
    right_hand.json     오른손 Tesollo 20관절
    left_arm.json       왼팔 7관절
    left_hand.json      왼손 Tesollo 20관절   (openarm_tesollo_bi_rl에만 해당)
```

## 규칙 하나: 실측한 group만 담는다

autotune 결과 JSON(`log/logs/r2s_autotune/results/*.json`)은 튜닝 대상이 **아니었던**
group까지 config 기본값 그대로 담는다. 그걸 학습에 그대로 물리면 측정한 적 없는 값이
env_cfg를 덮어쓴다 — 에러 없이, 조용히.

실제 사례: 07.29 우팔 결과에는 손 group이 30/5(autotune config placeholder)로 들어 있다.
그 파일을 `right/grasp_v1`에 적용하면 손 강성이 400 → 30, abduction이 600 → 30이 된다.

그래서 결과를 그대로 쓰지 않고 **부위 파일로 뽑아서** 여기 둔다. 부위 파일에 없는
group은 env_cfg 기본값이 그대로 살아 있다.

## 만들기

```bash
cd /home/user/rl_ws/hdgp

# 1) 부위별 autotune 실행 (scripts/r2s_autotune/configs/<자산>_<부위>.yaml)
#    → Notion: DRL / Real2Sim & HardWare / Robot_Control / Autotune 토글

# 2) 결과에서 실측한 group만 뽑아 자산 옆에 저장
PYTHONPATH=scripts python3 -m r2s_autotune.calibration_parts extract \
    --result log/logs/r2s_autotune/results/<결과>.json \
    --part right_hand --measured-on 2026-08-01 \
    --output assets/robot/openarm_tesollo_sensor_rl/calibration/right_hand.json
```

`--asset`은 같은 하드웨어가 여러 자산에 들어 있을 때만 쓴다(예: 오른팔은 `_sensor_rl`과
`_bi_rl`에 동일). 측정이 실제로 이뤄진 자산은 `provenance.measured_with_asset`에 남는다.

## 쓰기

부위 파일은 그 자체로 학습이 읽을 수 있는 형식(`schema_version: 1`, `groups`)이다.
여러 부위를 함께 쓰려면 합친다. 같은 group이 두 부위에 겹치면 병합이 거부한다 —
나중 파일이 조용히 이기는 것이 위에서 말한 사고 유형이기 때문이다.

```bash
PYTHONPATH=scripts python3 -m r2s_autotune.calibration_parts merge \
    --output log/logs/r2s_autotune/results/sensor_rl_train.json \
    assets/robot/openarm_tesollo_sensor_rl/calibration/*.json
```

> **⚠ 학습 쪽 배선은 아직 없다.** `tesollo/right,left/grasp_v1,v2`와
> `tesollo/both/pour_sensor`의 env_cfg는 팔을 `openarm_right_arm` 한 블록
> (`stiffness=400.0`)으로 하드코딩하고 있고, calibration을 읽는 코드가 없다.
> 적용하려면 그 블록을 부위 파일의 group 이름
> (`right_arm_proximal` / `right_arm_elbow` / `right_arm_wrist`)으로 쪼개고
> 로더를 붙여야 한다. 이름이 하나라도 어긋나면 `get_actuator_params`가 에러 없이
> 기본값으로 fallback한다.

## group 이름

| group | 관절 | 비고 |
|---|---|---|
| `right_arm_proximal` / `_elbow` / `_wrist` | `r_aj_[1-3]` / `r_aj_4` / `r_aj_[5-7]` | 실물 MIT PD 게인 구조(70/60/10)를 따른 경계 |
| `left_arm_proximal` / `_elbow` / `_wrist` | `l_aj_*` | 좌우 대칭 |
| `tesollo_hand_abduction` / `_curl` / `_pip` / `_dip` | `r_hj_*_[1-4]` | 오른손 |
| `tesollo_left_hand_abduction` / … | `l_hj_*_[1-4]` | 왼손 (bi 자산) |
| `head_camera`, `openarm_left_gripper` | | 튜닝 대상 아님. group 커버리지용 |

07.29 우팔 실측 당시 이름은 `arm_proximal` / `arm_elbow` / `arm_wrist`였다. 왼팔 부위가
생기면서 좌우 구분이 필요해져 `right_arm_*`로 바꿨고, 이관은 각 부위 파일의
`provenance.renamed_groups`에 남아 있다.

## 현재 상태

| 자산 | right_arm | right_hand | left_arm | left_hand |
|---|---|---|---|---|
| `openarm_tesollo_sensor_rl` | ✅ 07.29 실측 | ⬜ | ⬜ | — (왼손은 2관절 그리퍼) |
| `openarm_tesollo_bi_rl` | ✅ 07.29 실측(이관) | ⬜ | ⬜ | ⬜ |
| `openarm_bi_rh56f1_rl` | ⬜ | ⬜ | ⬜ | ⬜ |

`right_arm`은 계단 실측으로 kp·Fc를, Isaac Lab autotune으로 damping을 얻었다
(오차 5.69e-2 → 2.86e-2). **벤더 kd가 실제 응답 대비 2.6~3.6배 부족했다**는 것이
이 측정의 핵심 발견이다.

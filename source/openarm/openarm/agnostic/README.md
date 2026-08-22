# agnostic — 로봇 교체 가능한 조작 태스크

목표: **보상과 환경 설계만으로 로봇을 갈아끼워도 성공**한다.
로봇 종속 정보는 `modules/robots.py` 의 프로필에만 있고, 태스크 코드는 조인트/바디
**이름을 하드코딩하지 않는다**(계약 테스트가 소스 grep 으로 강제).

```
agnostic/
├── modules/                 트랙 공용 부품 (tasks → modules 단방향 의존)
│   ├── robots.py            RobotAsset(USD) / RobotProfile(팔) 레지스트리
│   ├── object_bank.py       물체군 + MultiAsset 스폰 + 원점 오프셋
│   ├── adr.py               TaskADR 스케줄러
│   ├── physics_dr.py        물리 DR EventTerm + ADR 종점 범위
│   └── agents.py            rl_games yaml 선택
└── tasks/
    ├── grasp_sensor/        팔 = Fabrics (구 grasp_lift, 08.22 개명. 다른 세션이 작업 중 — 수정 금지)
    └── grasp_lift_fabric/   팔 = Fabrics (이 문서의 대상)
```

---

## 1. 스위치 한 장 요약

| 스위치 | 기본 | 의미 | obs 차원 |
|---|---|---|---|
| `profile_name` | `bis_right` | 어느 로봇의 어느 팔인가 | 로봇별로 다름 |
| `object_bank` | `single_cup` | 무엇을 잡는가 | 불변 |
| `enable_object_onehot` | `False` | 물체 종류를 obs 에 넣는가 | **+N (재학습)** |
| `enable_physics_dr` | `False` | 물리 무작위화 | 불변 |
| `enable_adr` | `False` | 커리큘럼(스폰 반경) | 불변 |

**기본값은 전부 "가장 단순한 쪽"이다.** 스위치를 켜는 것만으로 학습이 흔들리지 않도록
물리 DR 의 초기 범위는 중립(스케일 1 또는 0)이고, ADR 은 꺼져 있으면
`get_param` 이 initial 값을 반환해 **커리큘럼 없는 고정 세팅과 정확히 동치**다.

---

## 2. 단계별 조합

```bash
# Phase A — 파지-리프트가 서는지. 원인 분리 최대.
./train.sh open-bis_r_grasp_lift_fab fab_A1 --num_envs 2048

# Phase B — s2r 강건성. obs 불변이라 A 의 체크포인트를 이어서 쓸 수 있다.
./train.sh open-bis_r_grasp_lift_fab fab_B1 --num_envs 2048 \
  env.enable_physics_dr=true env.enable_adr=true \
  --checkpoint log/rl_games/open-bis/right/grasp-lift-fab/fab_A1/nn/last_*.pth

# Phase C — 다물체 일반화. ★obs 차원이 바뀐다 → 재학습 필수.
./train.sh open-bis_r_grasp_lift_fab fab_C1 --num_envs 2048 \
  env.object_bank=cup_family env.enable_object_onehot=true \
  env.enable_physics_dr=true env.enable_adr=true

# Phase D — 로봇 교체. **코드 수정 0**, task id 만 바꾼다.
./train.sh open-rh56_r_grasp_lift_fab rh56_A1 --num_envs 2048
```

### ★hydra 오버라이드가 먹는 이유
`hydra_task_config` 는 `env_cfg.from_dict(...)` 로 **이미 만들어진 cfg 의 필드만**
덮어쓰고 `__post_init__` 을 다시 돌리지 않는다. 그대로 두면
`env.object_bank=cup_family` 가 문자열만 바꾸고 스포너·차원·`replicate_physics` 는
옛 값으로 남아 **조용히 틀린 조합**이 된다.

그래서 파생 로직을 `resolve_cfg(cfg)` 로 빼고 env 가 `super().__init__()` **전에**
다시 부른다(멱등). 검증: `probe_cfg_switches.py`.

---

## 3. 등록된 로봇

| task id | 자산 | 팔 | 손 | action/obs |
|---|---|---|---|---|
| `open-bis_r_grasp_lift_fab` | bi_s_rl (a2) | 우 | DG-5F-S 20 | 19 / 121 |
| `open-bis_l_grasp_lift_fab` | bi_s_rl (a2) | 좌 | DG-5F-S 20 | 19 / 121 |
| `open-bi_{r,l}_grasp_lift_fab` | bi_rl (a3) | 좌우 | DG-5F 20 | 19 / 121 |
| `open-sens_r_grasp_lift_fab` | sensor_rl (a1) | 우 | DG-5F 20 | 19 / 121 |
| `open-sens_l_grasp_lift_fab` | sensor_rl (a1) | 좌 | 2지 그리퍼 1 | 7 / 49 |
| `open-rh56_r_grasp_lift_fab` | bi_rh56f1_rl (a0) | 우 | RH56F1 12 | 17 / 95 |

각 id 에 `-play` / `-lstm` / `-play-lstm` 접미사가 있다.
`rh56_left` 는 **fabric URDF 부재로 미등록** — 조용히 우팔 기구학으로 폴백하지 않고 fail-loud 한다.

**로그는 로봇 USD 별로 자동 분리된다**: `log/rl_games/open-<자산short>/<side>/grasp-lift-fab/`.
gym id 의 로봇 슬롯에 자산 short 를 넣어 train.py 수정 없이 얻었다.

---

## 4. 새 로봇 추가

`modules/robots.py` 에 프로필 1개 추가가 전부여야 한다. 필요한 것:

1. `RobotAsset` — USD 이름 · 자산태그(`run_naming.ASSET_TAGS` 와 동일 어휘) · short
2. `RobotProfile` — 관절 regex · palm body · **`fabric_joint_order`** · 접촉 body
   (tip/wrap 분리) · 대향 그룹 · init 자세 · actuator(전 DOF 커버) · `surface_z`

계약 테스트(`modules/tests/`)가 URDF 원본과 대조해 다음을 막는다:
regex 해석 수 불일치 · 없는 링크 참조 · **init 값이 관절한계 밖**(좌우 부호 반전 함정) ·
actuator 커버리지 누락(= 조용한 free-spin) · fabric 관절 순서 오류 · 자산태그 미등록.

---

## 5. 새 물체군 추가

`modules/object_bank.py` 의 `ObjectSpec` 에 **`base_origin_offset_z`(USD 원점이 바닥에서
뜬 높이)를 반드시 채운다.** 미측정이면 fail-loud 한다.

★**작업면 높이와 원점 오프셋을 한 상수에 합치지 말 것.**
스폰 z = `surface_z`(환경 소유) + `origin_offset_z × scale`(자산 소유) + 패딩.
합쳐두면 물체가 바뀔 때 조용히 틀리고, 그 값이 곧 **lift 보상의 기준선**이라
보상이 통째로 오염된다(실측: 24.7mm 어긋나 `height_delta` 가 음수에서 시작했다).

---

## 6. 학습 전 필수 절차

```bash
# 1) 계약 테스트 (Isaac 불필요, 즉시)
PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/ -q

# 2) probe — 물리로 확인한다. 학습 곡선으로 확인하지 말 것.
P=scripts/reinforcement_learning/probes
PYTHONUNBUFFERED=1 ./isaaclab.sh -p $P/probe_grasp_lift_fabric_smoke.py --num_envs 64 --steps 300
PYTHONUNBUFFERED=1 ./isaaclab.sh -p $P/probe_fabric_tracking.py --num_envs 4 --steps 90
PYTHONUNBUFFERED=1 ./isaaclab.sh -p $P/probe_cfg_switches.py

# 3) 기동 후 params/env.yaml 로 **실제 반영값** 확인
```

### Fabrics 정상 기준 (bi_s 우팔)
```
no-op 목표 유지    palm 오차 0.8mm · qd 0.97 · 관절한계 위반 0
0.29m 이동(1.5s)   14.3mm · qd 1.53
zero-action        추종 7.8mm · 컵 xy 0.000 · 낙하 2mm · 44k fps(2048env)
```
이보다 크게 나쁘면 배선을 의심한다 — 흔한 원인 4종은 `modules/robots.py` 상단과
메모리 `fabrics-wiring-traps` 참조.

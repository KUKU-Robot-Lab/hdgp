# gripper/left/grasp_sensor — 왼팔 2지 그리퍼 파지·이동

> 상위 규칙은 `hdgp/CLAUDE.md`. 여기는 이 태스크 고유의 판단 근거만 둔다.

`openarm_tesollo_sensor_rl` 의 **왼팔 2지 그리퍼**가 shaker 를 집어 목표 위치로 옮긴다.
최종 목적은 양팔 물붓기(`tesollo/both/pour_sensor`)에서 왼팔이 receiver 컵을 실제로 쥐는 것이다
(지금 그쪽 왼팔은 컵을 kinematic-follow 로 붙여 둔 상태다).

- gym id: `open-grip_l_grasp_sensor` (+ `-play`)
- 로그: `log/rl_games/open-grip/left/grasp-sensor/`
- env: **manager-based** (`isaaclab.envs:ManagerBasedRLEnv`) — 커스텀 env 클래스가 없다

---

## 왜 IsaacLab lift 레시피인가 (되돌리지 말 것)

처음에는 `tesollo/right/grasp_sensor`(Direct RL + Fabrics + 정확 6D TCP 포즈 attractor)를
이식했다. **아키텍처 선택이 틀렸다.** 그 구현은
`grasp_sensor_fabrics_ABORTED/` 에 보존돼 있고, 다시 살리지 않는다.

무엇이 벽이었나:

- 5지 손은 팔이 대충 가져다 대도 손가락 20개가 형상을 맞춘다. **2지 평행 그리퍼는 jaw 가
  수평이어야만** 두 접촉점이 컵 지름 양끝에 놓이므로 팔에 특정 6-DOF 자세를 강제한다.
- 이 팔은 손목 j6 가 ±45° 뿐이고 손목 3축 effort 가 **7 N·m** 이라 낼 수 있는 자세가 얇은
  곡선이다. 거기에 "정확한 포즈를 내라"는 가장 빡빡한 제어를 얹었다.
- 실측 결과: 자세 오차 28°, jaw 가 수평이 안 되고 j5 가 관절 한계에 고착. 홈·자세·스폰 박스를
  여러 번 재도출했지만 매번 다른 벽에 부딪혔다. Fabrics 는 IK 솔버가 아니라 홈에서 출발하는
  gradient flow 라, 홈이 목표 자세에서 멀면 애초에 도달하지 못한다.

`Isaac-Lift-Cube-OpenArm-v0` 는 **같은 OpenArm + 같은 2지 그리퍼(스트로크 0.044)** 로 같은 일을
하면서 정확한 자세를 요구하는 지점이 한 곳도 없다. 그래서 위 문제가 통째로 사라진다.

| | 폐기한 구현 | 지금 (lift) |
|---|---|---|
| 팔 액션 | 6D TCP 포즈 → Fabrics | `JointPositionAction` (관절 델타) |
| 그리퍼 | 연속 1D + 접촉 게이트 | `BinaryJointPositionAction` (스칼라 1개) |
| 액션 차원 | 7 | **8** |
| 관측 | 48D (접촉력·핑거 상대위치) | **36D** |
| 보상 | 8-term + latch + ADR | **6-term** |
| 자세 요구 | 정확 6D 포즈 | **없음** (보상에 회전 항이 없다) |

---

## 바꾸지 말 것

lift 가 단순한 제어로도 학습되는 이유는 아래 성질들이다. 하나라도 깨면 그 장점이 사라진다.

1. `scale=0.5` + `use_default_offset=True` → **액션 0 = 초기 자세**, ±0.5 rad 국소 탐색
2. **초기 자세가 해답 근처** (컵을 향해 뻗은 자세) — 정책 초기화 시점부터 해답 주변을 탐색한다
3. 보상에 **회전 항이 전혀 없다** (거리 norm 뿐)
4. 그리퍼가 **이진 스칼라** — 파지력·개도를 정책이 학습하지 않는다
5. `object_goal_distance` 에 lift 게이트가 곱해져 "먼저 들어라 → 옮겨라" 순서가 내장돼 있고,
   정규화 페널티는 커리큘럼으로 10000 step 후에야 1000배 강화된다
6. early termination 이 사실상 없다 (타임아웃 + 물체 낙하) — 학습 신호가 끊기지 않는다
7. `decimation=2`, `episode_length_s=5.0`, reward weight 조합

보상 term 을 **재정의하지 않는다.** `LiftEnvCfg` 를 상속해 `params` 만 덮어쓴다.
weight 를 만지고 싶어지면 먼저 `reward-audit` 를 통과할 것.

---

## 이 씬 고유의 함정 (전부 실측으로 확인)

### 리프트 임계의 기준선은 상면이 아니라 **놓인 컵의 원점** — 여기서 한 번 태웠다

`mdp.object_is_lifted` 는 물체 **root 원점**의 절대 z 를 본다. 레퍼런스가 "테이블 상면 + 0.04"
로 맞아떨어지는 건 큐브의 원점이 기하 중심이라서일 뿐이다. **shaker 는 원점이 바닥에서
92 mm 위**라, 상면(0.215)만 더한 `0.255` 는 놓인 컵의 원점 `0.30709` 보다 **낮다**.

test1-r2 가 정확히 그 상태로 돌았고, TFEvents 가 바로 드러냈다:

| 지표 | 값 | 해석 |
|---|---|---|
| `Episode_Reward/lifting_object` | **14.63 / 상한 15.0** | 컵이 놓인 채로 lifting 이 상시 1 |
| `Episode_Reward/object_goal_tracking` | 10.34 | 게이트가 늘 열려 있다 |
| `Episode_Reward/reaching_object` | 0.024 → **0.007** | 그리퍼가 컵에서 **멀어진다** |

컵을 건드리면 떨어뜨려 15를 잃을 위험만 있고 가만히 있으면 공짜로 받으며, `action_rate`·
`joint_vel` 페널티까지 있으니 **가만히 있는 것이 최적**이다. 정책은 정확히 그렇게 했다.

올바른 식은 `MINIMAL_LIFT_HEIGHT = CUP_SPAWN_Z + 0.04`. 목표 커맨드 z 하한도 같은 이유로
리프트 임계 위여야 한다(아래가 목표면 "먼저 들어라 → 옮겨라" 순서가 무너진다).
둘 다 테스트로 고정했다. `P.MINIMAL_LIFT_HEIGHT` 를 쓰고 직접 숫자를 적지 않는다.

★교훈: 이 함정을 CLAUDE.md 에 "가장 위험"이라고 적어두고도 **상면만 더하고 컵 원점
오프셋을 빼먹어** 그대로 밟았다. 절대 z 를 쓰는 판정은 "그 물체가 **놓여 있을 때의 값**"을
먼저 재고 거기서 출발할 것.

### 테이블 상면 0.215 — 두 번 틀렸던 값

- `0.2082` — `right/grasp_sensor` 가 컵 반높이로 역산한 중간값. 상면이 아니다.
- `0.2004` — USD **BBoxCache** 로 읽은 값. `Cube` 의 authored extent 가 이미 `xformOp:scale`
  반영값인데 BBoxCache 가 scale 을 또 곱한다.
- **0.215** — `Cube size=1.0 × scale.z=0.03 → 반높이 0.015`, `+ init pos z 0.2`.
  낙하 정착 실측으로 확인(정착 z 가 정확히 `0.215 + 0.09209` = 0.30709, 표준편차 0).

### 컵 스폰 z 는 메시 bottom 에서 역산

shaker 원점은 기하 중심이 아니다(`z ∈ [-0.09209, +0.08291]`). bbox 반높이(0.0875)로 역산하면
컵이 판에 파묻혀 PhysX 가 밀어내며 넘어진다. 메시 점을 직접 변환해 잰 `0.09209` 를 쓴다.

### `SceneEntityCfg` 는 가변 객체다

매니저가 `resolve()` 로 **제자리 변경**(joint_ids 를 채워 넣음)한다. 한 인스턴스를 여러 term 에
공유하면 첫 term 이 ids 를 채우고 두 번째 term 에서 *"joint_names 와 joint_ids 가 불일치"* 로
env 생성이 죽는다. `params["asset_cfg"]` 우변은 항상 **새 인스턴스**여야 한다(테스트로 고정).

### 홈 자세의 팔이 컵 자리를 점유한다 — 스폰 x 하한이 있다

홈은 컵을 감싼 **파지 자세**라, 컵을 그 공간에 스폰하면 팔·손가락 메시가 컵을 관통해 PhysX 가
컵을 수백 mm 날려버린다(zero-action 실측 최대 **886 mm**, tilt 85°). 처음 잡았던 스폰 중심
(x 0.30, `tesollo/left/grasp_v1` 을 따른 값)이 정확히 그 자리였다.

`probe_lift_left_gripper_smoke.py --sweep_beyond` 로 잰 경계:

| 컵 x | 결과 |
|---|---|
| ≥ 0.31 | y ∈ [0.17, 0.23] 전 구간 조용 (이동 0.00 mm) |
| = 0.30 | y ≤ 0.18 에서만 조용 |
| < 0.30 | **전 구간 관통** — 팔이 그 공간에 있다 |

그래서 `SPAWN_X_SAFE_MIN = 0.31`, 스폰 박스는 x ∈ [0.32, 0.40] 으로 10 mm 여유를 뒀다.
**중심이 아니라 박스 전체**가 경계 밖이어야 한다(테스트로 고정). 목표 커맨드 x 하한도 같다.

⚠ "컵을 앞에 둔다"와 "홈을 뒤로 물린다"는 로봇 기준 상대 배치가 같아 물리적으로 동등하다.
결과적으로 홈이 pre-grasp 자세가 되며, 이것이 lift 레시피가 원하는 초기 조건이다.
초기 TCP–컵 거리는 약 149 mm (레퍼런스 Franka lift 는 약 450 mm 이므로 훨씬 가깝다).

### 그리퍼 두 조는 완전 대칭이 아니다

`l_hj_gripper_2` 는 PhysX mimic 으로 따라가지만 약 3.5 mm 오프셋이 남는다
(닫힘 j1=0.00 / j2=−3.50 mm, 열림 j1=44.00 / j2=40.26 mm — 일정한 오프셋).
mimic 이 관절 하한을 밀어붙이는 것으로, 두 조가 같은 방향으로 움직이는 것 자체는 정상이다
(조의 축이 서로 반대를 향한다). 파지 실패를 진단할 때 이 비대칭을 원인으로 오해하지 말 것.

### 유휴 오른팔은 왼팔 홈의 미러가 아니다

왼팔 홈이 그리퍼 전용 파지 자세라 그 부호 미러는 오른팔에 아무 의미가 없고, 렌더에서 기괴한
자세로 드러났다. `right/grasp_sensor` 의 실측 우팔 q_home 을 그대로 쓴다(테스트로 고정).

### 등록은 조용히 실패한다

`openarm/tasks/__init__.py` 의 glob 임포트가 `except ImportError: pass` 로 감싸여 있다.
config 를 건드린 뒤에는 반드시 명시 확인:

```bash
python3 -c "import openarm.tasks, gymnasium as gym; print(gym.spec('open-grip_l_grasp_sensor'))"
```

---

## 검증 절차

```bash
# 1. 정적 계약 (Isaac 불필요, ~0.2초)
PYTHONPATH=source/openarm python3 -m pytest \
  source/openarm/openarm/gripper/left/grasp_sensor/tests/ -q

# 2. zero-action 스모크 — 학습 전 반드시. 여기서 이상하면 epoch 를 태워도 의미가 없다
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/probes/probe_lift_left_gripper_smoke.py

# 3. 학습
NOTE="..." ./train.sh open-grip_l_grasp_sensor test1 --num_envs 1024 --headless \
  --video --video_length 300 --video_interval 2000
```

스모크가 보는 것: 컵이 zero-action 에서 제자리에 있는가 / 초기 TCP-컵 거리 / 그리퍼 이진 지령이
관절을 움직이는가 / 조기 종료 / 테이블 상면 실측 / 관통 구역 스윕.
★`PYTHONUNBUFFERED=1` 없이는 Isaac 프로브의 print 가 안 보인다.

학습 초기 판정: `reaching_object` 가 먼저 오르고 → `lifting_object` 가 0 을 벗어나면 파이프라인이
건강하다. 지표가 평탄하면 epoch 를 더 태우지 말고 **zero-action 스모크부터 다시 본다**
(pour_v1 에서 2442 epoch 를 1 분 프로브가 대체한 이력이 있다).

---

## 남은 것 / 격하된 것

- `scripts/probes/probe_gripper_opening.py` — **유효**. 그리퍼 최대 개구 84.5 mm(계산값 100 mm 가
  아니다 — 충돌 근사가 convexHull 이라 통과폭을 핑거 팁이 지배한다), shaker 파지 대역 10~85 mm.
- `probe_left_gripper_reach.py` / `probe_left_gripper_home.py` — **참고 자료로 격하**.
  lift 방식은 정확 자세를 요구하지 않으므로 게이트가 아니다.
- FABRICS 쪽 좌팔 그리퍼 자산(`openarm_gripper_left_pose_params.yaml`,
  `OpenArmGripperLeftPoseFabric`, 좌팔 그리퍼 URDF)은 **이 태스크에서 더 이상 쓰지 않는다.**
  다른 소비자가 없으므로 남겨 두지만 여기서 참조하지 말 것.

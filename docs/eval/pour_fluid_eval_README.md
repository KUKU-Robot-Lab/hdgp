# Pour 정책 실제 유체(PBD) 평가 프레임워크 (Phase 2)

bead(강체 대리지표)가 아니라 **실제 PBD 물**로 pour 정책의 이송 성공률을 측정한다.
정책은 isaaclab 에서, 유체는 SimulationContext 없는 raw-app 씬에서 돌리는
**record-and-replay** 구조.

> 왜 분리하나: isaacsim/isaaclab `SimulationContext` 는 PBD 파티클 시뮬을 조용히 꺼버린다.
> 그래서 정책 재생(isaaclab)과 유체(raw omni.timeline+app.update)를 한 프로세스에서 못 섞는다.
> 정책 궤적(컵 포즈·관절각)을 기록 → 순수 Isaac Sim 유체 씬에서 재생하며 유체를 붓는다.

---

## 구성 파일

| 파일 | 역할 | GPU |
|---|---|---|
| `scripts/assets_tools/make_cm_cup.py` | 컵 USD(미터)를 cm 스케일(정점 ×100)로 재저작 → `assets/cup/cup_big_sdf_cm.usd`. **한 번만 실행.** | O |
| `scripts/reinforcement_learning/probes/record_pour_traj.py` | 정책 재생 중 두 컵 포즈 + 로봇 관절각을 에피소드별 hdf5 저장(bead frac 상위 선별) | O |
| `scripts/reinforcement_learning/probes/replay_pour_fluid.py` | hdf5 궤적으로 컵 구동 + PBD 물 붓기 + **target 내부 유체 비율(실제 이송률)** 측정 | O |
| `scripts/reinforcement_learning/probes/verify_fluid_pour.py` | Phase 1 유체 소환/붓기 단독 검증(정책 없이) | O |

---

## 실행 순서 (GPU 필요 — 사용자 요청 시)

### 0. cm 컵 자산 생성 (최초 1회)
```bash
./IsaacLab/isaaclab.sh -p hdgp/scripts/assets_tools/make_cm_cup.py
# → assets/cup/cup_big_sdf_cm.usd 생성
```

### 1. 정책 궤적 기록
```bash
./IsaacLab/isaaclab.sh -p hdgp/scripts/reinforcement_learning/probes/record_pour_traj.py \
    --task open-tesol_b_pour_sensor-play-lstm \
    --checkpoint <path/to/last_....pth> \
    --num_envs 1 --eval_steps 8000 \
    --record_episodes 16 --record_out <log_dir>/pour_traj.hdf5
```
- **반드시 `--num_envs 1`** (에피소드 궤적 단일 추적).
- `--eval_steps` 는 원하는 에피소드 수를 모을 만큼 크게. `record_collect` 만큼 모아 상위 `record_episodes` 저장.
- 저장: `ep_###/{source_pose[T,7], target_pose[T,7], joint_pos[T,J]}`, 포즈=env-rel 미터, quat wxyz.
  attrs: dt, env_origin, robot_root, robot_usd, joint_names, num_beads, bead_frac 등.

### 2. 유체 재생 + 이송 성공률 측정
```bash
./IsaacLab/isaaclab.sh -p hdgp/scripts/reinforcement_learning/probes/replay_pour_fluid.py \
    --traj <log_dir>/pour_traj.hdf5 --episodes all --headless \
    --report_out <log_dir>/pour_fluid_eval.md
```
- 각 에피소드: 컵을 궤적대로 kinematic 구동, source 컵에 물 점진 스폰(초기 프레임), 붓기.
- 궤적 종료 후 `--tail_settle` 안착 → **target 컵 로컬 프레임 기준 내부 파티클 비율** = 유체 이송률.
- 리포트: 에피소드별 유체 이송률 + 성공률(≥`--success_frac`) + bead frac 대비표.
- GUI 로 보려면 `--headless` 빼기. 프레임 저장은 `--capture_dir`.

---

## 핵심 파라미터 (replay)

| 인자 | 기본 | 의미 |
|---|---|---|
| `--particle_contact` | 0.008 | 물방울 크기(m). 작을수록 곱지만 SDF 짜임↑ |
| `--fill_height` | 0.045 | 컵 유체 깊이(m). 깊을수록 수압↑ |
| `--spawn_batches`/`--spawn_interval` | 20 / 18 | 점진 스폰(초기 폭발 방지) |
| `--fill_frames` | 380 | 유체 채우는 초기 프레임(컵이 대체로 정지인 hold 구간) |
| `--tail_settle` | 120 | 측정 전 추가 안착 |
| `--success_frac` | 0.5 | 유체 이송 성공 임계 |
| `--cup_segments` | 28 | 원통 벽 세그먼트(매끄러움) |

---

## 컵 모델 (verify_fluid_pour.py / replay 공통)

- **비주얼 컵 USD**(cm) + **내부 얇은 원통 벽**(볼록 프리미티브 충돌). 컵 SDF 는 미세 틈으로 물이 새므로 끄고, 원통 벽(바닥 디스크 + 박스 링)이 담당 → 데모 박스 캐치와 동일 원리로 안 샘.
- `--sdf_cup`: 예전 SDF 컵으로 전환(점성/CCD 로 새는 것 억제).

---

## 알려진 한계 / TODO

- **로봇 재생(`--with_robot`)**: 현재 로봇 USD 를 **base 포즈에 정적 배치**만 한다(비주얼 placeholder).
  관절각(`joint_pos`)까지 kinematic 재생하려면 articulation 관절 구동 추가 필요(raw-app 에서 비자명).
  **유체 이송 측정은 컵 포즈만으로 동작**하므로 로봇 없이도 성공률은 나온다.
- 좌표 매핑은 env-rel 미터 → cm(×100), Z-up 동일 가정. 태스크가 다른 up-axis/원점이면 조정.
- 유체 이송률 vs bead frac 상관은 리포트에서 대비 확인(둘 다 저장).

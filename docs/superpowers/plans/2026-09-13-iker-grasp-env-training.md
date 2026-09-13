# IKER 파지 뱅크·학습 환경·사람 기준선 학습 착수 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 구성 0 에 대해 DG-5F 파지 뱅크를 만들고, IKER 고정 보상으로 학습하는 DirectRLEnv 를 hdgp 에 등록한다. 그리고 사람 기준선 목표로 PPO 학습(250 에포크)을 로컬 GPU 에서 띄운다.

**Architecture:** 세 층으로 나뉜다.
- `tasks/iker_shoe/grasp_bank.py`(순수 torch): 파지 전 자세 샘플, 막히면 멈추는 시너지 닫기, 들기 판정, 뱅크 파일.
- `scripts/iker/make_grasp_bank.py`(Isaac, 64 병렬 환경): 뱅크 생성.
- `tasks/iker_shoe/iker_shoe_env*.py`: 뱅크 상태로 리셋하고, 손바닥 자세 변화량을 차분 IK 로 풀어 레거시 보상으로 학습한다. gym id 는 `open-sens_r_iker_shoe`.

학습은 hdgp `train.py` 로 띄운다(워크트리 체크아웃의 openarm 을 강제로 쓴다).

**Tech Stack:** Isaac Sim 5.1 / Isaac Lab 0.50.5, rl_games PPO, torch, pytest(시스템 python3 3.10).

**Spec:** `docs/superpowers/specs/2026-09-13-iker-vlm-keypoint-reward-design.md`(§7 학습 환경, §8 파지 시작 상태, §10 시뮬레이터 검증, 마일스톤 1·4)

**선행 조건:** 계획 1(`docs/superpowers/plans/2026-09-13-iker-front-end.md`)이 같은 브랜치에 완료돼 있어야 한다. 이 계획이 쓰는 것은 다음과 같다.
- `modules/iker/{reward,run_files,gate,rotations}.py`, `tasks/iker_shoe/{layout,robot,scene_cfg}.py`
- `assets/iker_shoe/`, `iker_runs/shoe_place/config_00/{keypoints.json,interaction_human.json}`

**실행 위치:** 워크트리 `~/rl_ws/hdgp-iker`(브랜치 `iker-front-end`). 공유 체크아웃 `~/rl_ws/hdgp` 는 건드리지 않는다(학습·다른 세션 보호, 사용자 요청 2026-09-13).

**계획 분할(3개 중 2번):** 3번은 Claude Code 생성 루프, Qwen 배선, 구성 10개 평가, 보고서다.

**검증 상태(2026-09-13):** 이 계획의 코드 블록은 스크래치패드 복제 트리에서 실제로 돌린 파일을 스크립트로 옮겨 넣은 것이다.
- 순수 테스트: 파지 뱅크 10개(태스크 테스트 합계 21개) 통과.
- 파지 뱅크: 64 환경 12 라운드 중 7 라운드(약 200 s)에서 69개로 통과. 라운드당 들기 성공 10~20, 재현 성공 13~21, 채택 5~13.
- 환경 스모크(16 환경) 통과:
  - 관측 (16, 38) 유한, 행동 0 5 s 동안 떨어짐 0.
  - 무작위 25 s 동안 보상 유한, 에피소드 로그 기록.
  - 목표 자세 배치 시 성공 판정 100 %(4.7 mm).
  - 목표 자리 IK 도달 1.5 mm. 반대쪽 자리는 114 mm 로 닿지 않는다(계획 3 인계).
- 학습 스모크: 4096 환경 3 에포크, 초당 6,975→8,161 프레임, PhysX 넘침 없음, 체크포인트·TFEvents 생성.

프로토타입에서 계획에 반영한 발견은 네 가지다.
- 손바닥을 바로 아래로 향한 낮은 파지 자세는 오른팔 IK 로 닿지 않는다. 손바닥을 world x 축 기준 −15~−40° 기울여야 도달한다.
- 한 가지 파지 자세로는 결과가 실행마다 달라진다. 그래서 자세를 흔들어 두 번 들어 올려 통과한 상태만 모은다.
- Isaac Lab `randomize_rigid_body_com` 은 관절체 전용이라 단일 강체에서 IndexError 가 난다. 그래서 `events.randomize_rigid_object_com` 을 따로 둔다.
- 뱅크 상태의 쥔 손이 목표 자리를 침범한다. 그래서 목표 배치 검사는 로봇을 홈으로 치운 뒤 한다.

## Global Constraints

- t2r(`modules/t2r`, `scripts/reward_gen`)과 grasp_s2r 을 import 하지 않는다. IKER 단독 성능 측정이 목적이다.
- 로봇 값은 `modules/robot_profiles.py`·`modules/vendor_gains.py` 에서만 읽는다(숫자 복사 금지).
- GPU 는 로컬만 쓴다(서버 미사용).
- 순수 테스트: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest <경로> -q -p no:cacheprovider` (scipy·requests 경고 줄은 무시).
- Isaac 실행: `cd ~/rl_ws/hdgp-iker && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm ../IsaacLab/_isaac_sim/python.sh <스크립트> --headless`.
  - `isaaclab.sh` 는 쓰지 않는다. 그 런처가 공유 체크아웃의 openarm 을 PYTHONPATH 맨 앞에 넣는다(`isaaclab.sh:32-36`).
  - 스크립트 docstring 의 `isaaclab.sh -p` 사용법은 병합 후 기준이라 그대로 둔다.
- Isaac 스크립트의 종료 규칙:
  - `sys.excepthook` 에서 `os._exit(1)`, 끝에서 `os._exit(...)` 로 나간다.
  - 성공 판정은 출력 표식(`BANK ... passed True`, `SMOKE passed True`, `MAX EPOCHS NUM!`)과 산출 파일로 한다.
- 공유 체크아웃 `~/rl_ws/hdgp` 에서는 어떤 명령도 실행하지 않는다.
- `git add -A`·`git add .` 금지, 각 커밋 단계의 경로만 add 하고 push·병합하지 않는다.
- 커밋 메시지 끝에는 다음 두 줄을 붙인다(각 커밋 단계에 이미 들어 있다).
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` / `Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4`
- PreToolUse 훅이 `git commit` 과 ` -n`(예: `grep -n`)이 한 Bash 호출에 같이 있으면 막는다. 커밋은 따로 호출한다.
- 학습 프로세스 종료는 PID 로만 한다(`pkill -f` 패턴 금지). 다른 세션의 학습을 죽인 사고가 있었다.
- 설정값은 스펙에서 그대로 가져온다.

| 항목 | 값 |
|---|---|
| 제어 | dt 1/120, decimation 12(10 Hz), 스텝당 최대 0.02 m·0.05 rad, DLS 차분 IK |
| 관측·보상 | 관측 38, 에피소드 200 스텝, 높이 임계 낙하 0.168·테이블 아래 0.268 |
| 무작위화 | 신발 질량 ×0.3~2.0·마찰 0.3~1.8·반발 0~1·무게중심 ±0.025 m, 관측 ±0.02·행동 ±0.05·쿼터니언 ±0.2 rad + 전체 부호 |
| 다른 신발 리셋 잡음 | ±0.01 m·±0.05 rad |
| PhysX | collision stack 2^29, patch 2^23, contact 2^24 |
| PPO | MLP 256·128·64 ELU, γ 0.99, τ 0.95, lr 5e-4 adaptive(kl 0.008), horizon 32, minibatch 16384, mini-epoch 5, critic 4, bounds 1e-4, 4096 env·250 에포크 |
| 파지 뱅크 | 64개 이상, 들어 올린 뒤 1 s 미끄러짐 < 1 cm·상승 ≥ 5 cm, 게인·dt·솔버·마찰 불일치 시 부팅 거부 |

## 파일 구조

`T = source/openarm/openarm/agnostic/tasks/iker_shoe`

| 파일 | 책임 | 태스크 |
|---|---|---|
| `T/grasp_bank.py`, `T/tests/test_grasp_bank.py` | 파지 전 자세·시너지 닫기·들기 판정·뱅크 파일(순수 torch) | 1 |
| `scripts/iker/make_grasp_bank.py` → `iker_runs/shoe_place/config_00/grasp_bank.json` | 64 병렬 환경 뱅크 생성·재현 검증 | 2 |
| `T/events.py` | 단일 강체 무게중심 무작위화 | 3 |
| `T/iker_shoe_env_cfg.py`, `T/iker_shoe_env.py` | 학습 환경 설정과 DirectRLEnv | 3 |
| `T/config/__init__.py`, `T/config/agents/__init__.py`, `T/config/agents/rl_games_ppo_cfg.yaml` | gym 등록(`open-sens_r_iker_shoe[-play]`)과 PPO | 3 |
| `scripts/iker/env_smoke.py` | 학습 전 시뮬레이터 검사 5종 | 4 |

---

### Task 1: 파지 뱅크 순수 모듈

**Files:**
- Create: `T/grasp_bank.py`, `T/tests/test_grasp_bank.py`

**Interfaces:**
- Consumes: 없음(torch 만)
- Produces:
  - `grasp_bank`
    - 상수 `BANK_SCHEMA=1`, `BLOCKED_ERR_RAD`, `LIMIT_MARGIN_RAD`, `CLOSE_RATE_PER_STEP`, `MIN_RISE_M`, `MAX_HOLD_SLIP_M`, 범위 `TILT_DEG_RANGE` 등
    - `PreGrasp(tilt_deg, height_above_top, near_side_clearance, along_length, yaw_deg, thumb3)`
    - `sample_pregrasp(count, generator, device) -> PreGrasp`
    - `palm_rotations(tilt_deg, yaw_deg) -> (N,3,3)`
    - `palm_goal_positions(pregrasp, shoe_xy, shoe_yaw_deg, shoe_half_width, shoe_top_z) -> (N,3)`
    - `synergy_step(close, target, joint_pos, start_pose, grip_pose, lower, upper) -> (close, target)`
    - `lift_held(shoe_z_start, shoe_z_end, rel_after_lift, rel_after_hold) -> (N,) bool`
    - `bank_document(entries, joint_names, metadata) -> dict`
    - `GraspBank(joint_pos, joint_target, shoe_pose, palm_pose, metadata).size`
    - `load_bank(doc, articulation_joint_names, expected_metadata, device) -> GraspBank`
    - `gains_metadata(joint_names, stiffness, damping) -> dict`

- [ ] **Step 1: 실패하는 테스트를 작성한다**

Create `source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_bank.py`:

````python
"""grasp_bank — synergy closing, pre-grasp geometry, lift test, bank file (no Isaac)."""

import math

import pytest
import torch

from openarm.agnostic.tasks.iker_shoe import grasp_bank as gb

JOINTS = ["a", "b", "c"]
START = torch.tensor([[0.0, 0.0, 0.5]])
GRIP = torch.tensor([1.0, 1.0, 0.5])  # joint c does not move between start and grip
LOWER = torch.tensor([-1.0, -1.0, -1.0])
UPPER = torch.tensor([2.0, 0.6, 2.0])


def test_synergy_closes_free_joints_and_skips_fixed_ones():
    close, target = gb.synergy_step(torch.zeros(1, 3), START.clone(), START.clone(), START, GRIP, LOWER, UPPER)
    assert torch.allclose(close, torch.tensor([[gb.CLOSE_RATE_PER_STEP, gb.CLOSE_RATE_PER_STEP, 0.0]]))
    assert torch.allclose(target[0, :2], torch.full((2,), gb.CLOSE_RATE_PER_STEP))
    assert target[0, 2] == pytest.approx(0.5)


def test_blocked_joint_freezes_but_joint_at_its_limit_does_not():
    close = torch.tensor([[0.5, 0.7, 0.5]])
    target = torch.tensor([[0.5, 0.6, 0.5]])
    joint_pos = torch.tensor([[0.1, 0.6, 0.5]])  # a lags its target by 0.4 rad; b sits on its upper limit
    new_close, new_target = gb.synergy_step(close, target, joint_pos, START, GRIP, LOWER, UPPER)
    assert new_close[0, 0] == pytest.approx(0.5)
    assert new_close[0, 1] == pytest.approx(0.7 + gb.CLOSE_RATE_PER_STEP)
    assert new_target[0, 1] == pytest.approx(0.6)  # clamped to the joint limit


def test_zero_tilt_palm_faces_down_with_fingers_toward_plus_y():
    rot = gb.palm_rotations(torch.tensor([0.0]), torch.tensor([0.0]))[0]
    assert torch.allclose(rot[:, 0], torch.tensor([0.0, 0.0, -1.0]))  # palmar side
    assert torch.allclose(rot[:, 2], torch.tensor([0.0, 1.0, 0.0]))  # fingers
    tilted = gb.palm_rotations(torch.tensor([-20.0]), torch.tensor([0.0]))[0]
    assert torch.allclose(tilted[:, 0], torch.tensor([0.0, -math.sin(math.radians(20)), -math.cos(math.radians(20))]), atol=1e-6)


def test_palm_goal_sits_beside_the_near_side_above_the_top():
    pre = gb.PreGrasp(*(torch.tensor([v]) for v in (-20.0, 0.03, 0.02, 0.01, 0.0, 0.4)))
    goal = gb.palm_goal_positions(pre, (0.25, 0.14), 90.0, 0.048, 0.309)[0]
    # yawed 90 deg: along-length is +y and the near side is +x
    assert torch.allclose(goal, torch.tensor([0.25 + 0.068, 0.14 + 0.01, 0.339]), atol=1e-6)


def test_samples_stay_in_their_ranges():
    pre = gb.sample_pregrasp(500, torch.Generator().manual_seed(0))
    for value, bounds in ((pre.tilt_deg, gb.TILT_DEG_RANGE), (pre.thumb3, gb.THUMB3_RANGE), (pre.along_length, gb.ALONG_LENGTH_RANGE)):
        assert bounds[0] <= float(value.min()) and float(value.max()) <= bounds[1]


def test_lift_held_needs_rise_and_little_slip():
    z0 = torch.tensor([0.25, 0.25, 0.25])
    z1 = torch.tensor([0.34, 0.26, 0.34])
    rel_lift = torch.zeros(3, 3)
    rel_hold = torch.tensor([[0.0, 0.0, 0.005], [0.0, 0.0, 0.0], [0.0, 0.02, 0.0]])
    assert gb.lift_held(z0, z1, rel_lift, rel_hold).tolist() == [True, False, False]


def _bank_doc(**metadata):
    entries = {
        "joint_pos": torch.tensor([[1.0, 2.0, 3.0]]),
        "joint_target": torch.tensor([[1.5, 2.5, 3.5]]),
        "shoe_pose": torch.tensor([[0.3, 0.1, 0.26, 1.0, 0.0, 0.0, 0.0]]),
        "palm_pose": torch.tensor([[0.3, 0.05, 0.34, 1.0, 0.0, 0.0, 0.0]]),
    }
    return gb.bank_document(entries, JOINTS, {"physics_dt": 1 / 120, **metadata})


def test_bank_round_trip_reorders_joints_by_name():
    bank = gb.load_bank(_bank_doc(), ["c", "a", "b"], {"physics_dt": 1 / 120}, "cpu")
    assert bank.size == 1
    assert bank.joint_pos.tolist() == [[3.0, 1.0, 2.0]] and bank.joint_target.tolist() == [[3.5, 1.5, 2.5]]


def test_bank_refuses_different_settings_or_missing_joints():
    with pytest.raises(ValueError, match="different settings"):
        gb.load_bank(_bank_doc(), JOINTS, {"physics_dt": 1 / 60}, "cpu")
    with pytest.raises(ValueError, match="lacks joints"):
        gb.load_bank(_bank_doc(), JOINTS + ["d"], {}, "cpu")


def test_bank_document_rejects_non_finite_rows():
    with pytest.raises(ValueError, match="joint_pos"):
        gb.bank_document(
            {"joint_pos": torch.tensor([[float("nan")]]), "joint_target": torch.zeros(1, 1), "shoe_pose": torch.zeros(1, 7), "palm_pose": torch.zeros(1, 7)},
            ["a"],
            {},
        )


def test_gains_metadata_rounds_for_exact_comparison():
    meta = gb.gains_metadata(["a", "b"], torch.tensor([400.00001, 10.0]), torch.tensor([80.0, 0.123456]))
    assert meta == {"stiffness": {"a": 400.0, "b": 10.0}, "damping": {"a": 80.0, "b": 0.1235}}
````

- [ ] **Step 2: 테스트가 실패하는지 확인한다**

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_bank.py -q -p no:cacheprovider`
Expected: FAIL. `ImportError`(`grasp_bank` 없음).

- [ ] **Step 3: 구현한다**

Create `source/openarm/openarm/agnostic/tasks/iker_shoe/grasp_bank.py`:

````python
"""Grasp bank for the IKER shoe task (design spec §8). Pure torch: importable without Isaac Sim.

A bank entry is the state right after the synergy hand closed on the shoe resting at the configuration's
start pose. It is kept only if lifting from that state and, separately, from the state restored into a
fresh episode both hold the shoe. Resets restore an entry: joint positions, the **commanded** joint
targets (commanding the measured hand pose would remove the squeeze), and the shoe pose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

BANK_SCHEMA = 1
BLOCKED_ERR_RAD = 0.2  # a joint this far from its target (and not at a limit) is pressed against the shoe
LIMIT_MARGIN_RAD = 0.02
CLOSE_RATE_PER_STEP = 1.0 / 150.0  # full open -> grip in 1.25 s at 120 Hz
MIN_RISE_M = 0.05  # spec §8: the palm lifts 0.10 m; the shoe must follow at least half of it
MAX_HOLD_SLIP_M = 0.01  # spec §8: shoe-to-palm drift over the hold after lifting

# Pre-grasp distribution measured with probes on configuration 0 (scratchpad notes 2026-09-13).
TILT_DEG_RANGE = (-40.0, -15.0)
HEIGHT_ABOVE_TOP_RANGE = (0.02, 0.035)
NEAR_SIDE_CLEARANCE_RANGE = (0.02, 0.035)
ALONG_LENGTH_RANGE = (-0.04, 0.04)
YAW_DEG_RANGE = (-10.0, 10.0)
THUMB3_RANGE = (0.3, 0.5)


@dataclass(frozen=True)
class PreGrasp:
    tilt_deg: torch.Tensor
    height_above_top: torch.Tensor
    near_side_clearance: torch.Tensor
    along_length: torch.Tensor
    yaw_deg: torch.Tensor
    thumb3: torch.Tensor


def sample_pregrasp(count: int, generator: torch.Generator, device: str | torch.device = "cpu") -> PreGrasp:
    def uniform(bounds):
        lo, hi = bounds
        return lo + (hi - lo) * torch.rand(count, generator=generator).to(device)

    return PreGrasp(
        tilt_deg=uniform(TILT_DEG_RANGE),
        height_above_top=uniform(HEIGHT_ABOVE_TOP_RANGE),
        near_side_clearance=uniform(NEAR_SIDE_CLEARANCE_RANGE),
        along_length=uniform(ALONG_LENGTH_RANGE),
        yaw_deg=uniform(YAW_DEG_RANGE),
        thumb3=uniform(THUMB3_RANGE),
    )


def palm_rotations(tilt_deg: torch.Tensor, yaw_deg: torch.Tensor) -> torch.Tensor:
    """(N, 3, 3) palm orientations: palmar side down, fingers across the shoe toward +y, tilted about world x
    (negative tilt turns the palmar side toward -y) and yawed about world z."""
    down = torch.tensor([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]], dtype=tilt_deg.dtype, device=tilt_deg.device)
    t, y = torch.deg2rad(tilt_deg), torch.deg2rad(yaw_deg)
    zeros, ones = torch.zeros_like(t), torch.ones_like(t)
    rx = torch.stack([ones, zeros, zeros, zeros, t.cos(), -t.sin(), zeros, t.sin(), t.cos()], dim=-1).view(-1, 3, 3)
    rz = torch.stack([y.cos(), -y.sin(), zeros, y.sin(), y.cos(), zeros, zeros, zeros, ones], dim=-1).view(-1, 3, 3)
    return rz @ rx @ down


def palm_goal_positions(
    pregrasp: PreGrasp, shoe_xy: Sequence[float], shoe_yaw_deg: float, shoe_half_width: float, shoe_top_z: float
) -> torch.Tensor:
    """(N, 3) palm origin goals: above the shoe top, offset toward the shoe's near (-y) side."""
    yaw = math.radians(shoe_yaw_deg)
    along = pregrasp.along_length
    lateral = -(shoe_half_width + pregrasp.near_side_clearance)
    x = shoe_xy[0] + along * math.cos(yaw) - lateral * math.sin(yaw)
    y = shoe_xy[1] + along * math.sin(yaw) + lateral * math.cos(yaw)
    return torch.stack([x, y, shoe_top_z + pregrasp.height_above_top], dim=-1)


def synergy_step(
    close: torch.Tensor,
    target: torch.Tensor,
    joint_pos: torch.Tensor,
    start_pose: torch.Tensor,
    grip_pose: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One closing step of the open -> grip synergy with per-joint freeze.

    ``close``/``target``/``joint_pos``/``start_pose`` are (N, J); ``grip_pose``/``lower``/``upper`` are (J,).
    A joint stops closing while it is blocked: farther than ``BLOCKED_ERR_RAD`` from its target and not
    at a joint limit. Joints whose start and grip poses coincide never move. Returns new (close, target).
    """
    movable = (grip_pose.unsqueeze(0) - start_pose).abs() > 1e-4
    free = (joint_pos > lower + LIMIT_MARGIN_RAD) & (joint_pos < upper - LIMIT_MARGIN_RAD)
    blocked = ((target - joint_pos).abs() > BLOCKED_ERR_RAD) & free
    step = torch.where(movable & ~blocked, torch.full_like(close, CLOSE_RATE_PER_STEP), torch.zeros_like(close))
    new_close = (close + step).clamp(0.0, 1.0)
    new_target = torch.max(torch.min(torch.lerp(start_pose, grip_pose.unsqueeze(0), new_close), upper), lower)
    return new_close, new_target


def lift_held(shoe_z_start: torch.Tensor, shoe_z_end: torch.Tensor, rel_after_lift: torch.Tensor, rel_after_hold: torch.Tensor) -> torch.Tensor:
    """(N,) bool: the shoe rose with the palm and stayed put in the hand during the hold."""
    rise = shoe_z_end - shoe_z_start
    slip = (rel_after_hold - rel_after_lift).norm(dim=-1)
    return (rise >= MIN_RISE_M) & (slip < MAX_HOLD_SLIP_M)


def bank_document(entries: Mapping[str, torch.Tensor], joint_names: Sequence[str], metadata: Mapping) -> dict:
    """JSON-ready bank. ``entries`` holds (K, ...) tensors: joint_pos, joint_target, shoe_pose, palm_pose."""
    required = ("joint_pos", "joint_target", "shoe_pose", "palm_pose")
    missing = [key for key in required if key not in entries]
    if missing:
        raise ValueError(f"bank entries lack {missing}")
    count = entries["joint_pos"].shape[0]
    for key in required:
        if entries[key].shape[0] != count or not torch.isfinite(entries[key]).all():
            raise ValueError(f"bank field {key!r} must be finite with {count} rows")
    return {
        "schema": BANK_SCHEMA,
        "joint_names": list(joint_names),
        "metadata": dict(metadata),
        "entries": {key: entries[key].detach().cpu().tolist() for key in required},
    }


@dataclass(frozen=True)
class GraspBank:
    joint_pos: torch.Tensor  # (K, J) in the articulation's joint order
    joint_target: torch.Tensor  # (K, J)
    shoe_pose: torch.Tensor  # (K, 7) env-local position + wxyz
    palm_pose: torch.Tensor  # (K, 7)
    metadata: Mapping

    @property
    def size(self) -> int:
        return int(self.joint_pos.shape[0])


def load_bank(doc: Mapping, articulation_joint_names: Sequence[str], expected_metadata: Mapping, device) -> GraspBank:
    """Validate a bank document against the running robot and reorder joints by name."""
    if doc.get("schema") != BANK_SCHEMA:
        raise ValueError(f"grasp bank schema {doc.get('schema')} != {BANK_SCHEMA}")
    mismatched = {k: (doc["metadata"].get(k), v) for k, v in expected_metadata.items() if doc["metadata"].get(k) != v}
    if mismatched:
        raise ValueError(f"grasp bank was made with different settings (bank, running): {mismatched}")
    names = list(doc["joint_names"])
    missing = [n for n in articulation_joint_names if n not in names]
    if missing:
        raise ValueError(f"grasp bank lacks joints {missing}")
    order = torch.tensor([names.index(n) for n in articulation_joint_names], dtype=torch.long)
    entries = doc["entries"]

    def tensor(key: str) -> torch.Tensor:
        return torch.tensor(entries[key], dtype=torch.float32)

    bank = GraspBank(
        joint_pos=tensor("joint_pos")[:, order].to(device),
        joint_target=tensor("joint_target")[:, order].to(device),
        shoe_pose=tensor("shoe_pose").to(device),
        palm_pose=tensor("palm_pose").to(device),
        metadata=dict(doc["metadata"]),
    )
    if bank.size == 0:
        raise ValueError("grasp bank is empty")
    return bank


def gains_metadata(joint_names: Sequence[str], stiffness: torch.Tensor, damping: torch.Tensor) -> dict:
    """Per-joint PD gains rounded for exact comparison between bank generation and training."""
    return {
        "stiffness": {n: round(float(k), 4) for n, k in zip(joint_names, stiffness.tolist())},
        "damping": {n: round(float(d), 4) for n, d in zip(joint_names, damping.tolist())},
    }
````

- [ ] **Step 4: 통과를 확인한다**

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests -q -p no:cacheprovider`
Expected: `21 passed`(layout 7 + shoe_meta 4 + grasp_bank 10)

- [ ] **Step 5: 커밋한다**

```bash
cd ~/rl_ws/hdgp-iker && T=source/openarm/openarm/agnostic/tasks/iker_shoe
git add $T/grasp_bank.py $T/tests/test_grasp_bank.py
git commit -F - <<'MSG'
feat(iker): 파지 뱅크 순수 모듈 — 파지 전 자세·막힘 동결 시너지·들기 판정·뱅크 파일

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4
MSG
```

---

### Task 2: 파지 뱅크 생성과 구성 0 뱅크(Isaac)

**Files:**
- Create: `scripts/iker/make_grasp_bank.py`
- Create(스크립트 산출): `iker_runs/shoe_place/config_00/grasp_bank.json`

**Interfaces:**
- Consumes:
  - `grasp_bank.*`(Task 1)
  - `layout.sample_configs`, `layout.shoe_start_poses`, `layout.RUNS_DIR`, `layout.SHOE_META_PATH`, `layout.TABLE_TOP_Z`, `layout.MOVING_SHOE`
  - `robot.robot_cfg`, `robot.profile`, `robot.GRAVITY_COMPENSATION_JOINTS`, `robot.SOLVER_*`
  - `scene_cfg.table_cfg/rack_cfg/light_cfg/shoe_cfg`, `run_files.load_shoe_meta`, `run_files.write_json`
- Produces: `grasp_bank.json`
  - 키: `schema`, `joint_names`(articulation 순서), `entries{joint_pos, joint_target, shoe_pose, palm_pose}`(env-local, wxyz)
  - `metadata{config_index, scene_config, robot_usd, physics_dt, friction, solver_position_iterations, solver_velocity_iterations, gains{stiffness,damping}, shoe_meta_sha256, seed, rounds}`

한 라운드는 다음 순서로 진행한다.
1. 64 환경이 신발을 치워 둔 채 파지 전 자세로 내려간다.
2. 신발을 구성 0 시작 자세에 놓고, 손을 닫고, 그 상태를 기록한다.
3. 0.10 m 들어 1 s 유지한다.
4. 기록한 상태를 리셋처럼 되살려 한 번 더 들어 올린다.

두 번 모두 통과한 상태만 뱅크에 넣는다. 64개가 모이거나 12 라운드가 되면 멈춘다.

- [ ] **Step 1: 스크립트를 만든다**

Create `scripts/iker/make_grasp_bank.py`:

````python
"""Build the grasp bank of one IKER configuration (design spec §8).

Every round, all envs approach a sampled pre-grasp above the shoe (the shoe parked out of the way), the shoe
is placed at the configuration's start pose, the synergy hand closes, the state is recorded, and the palm
lifts 0.10 m and holds 1 s. Recorded states that held are replayed from a fresh reset and lifted again;
only states that hold twice enter the bank.

Usage:
    cd ~/rl_ws/hdgp && PYTHONPATH=source/openarm ../IsaacLab/isaaclab.sh -p scripts/iker/make_grasp_bank.py \
        --config-index 0 --headless
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Build the IKER grasp bank of one configuration.")
parser.add_argument("--config-index", type=int, required=True)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--min-entries", type=int, default=64)
parser.add_argument("--max-rounds", type=int, default=12)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    print("BANK FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import quat_from_matrix  # noqa: E402

from openarm.agnostic.modules.iker import run_files  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import grasp_bank as gb  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import layout, robot, scene_cfg  # noqa: E402

PHYSICS_DT = 1.0 / 120.0
APPROACH_STEPS, DESCEND_STEPS, SETTLE_STEPS, CLOSE_STEPS = 360, 240, 60, 240
LIFT_STEPS, HOLD_STEPS, LIFT_HEIGHT_M = 120, 120, 0.10
PARK_POS = (0.42, 0.40)  # table corner away from the grasp area
FRICTION = 1.0  # same contact material as the training environment


def build(config, meta):
    poses = layout.shoe_start_poses(config, meta)
    move_pos, move_quat = poses[layout.MOVING_SHOE]

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        table = scene_cfg.table_cfg()
        rack = scene_cfg.rack_cfg()
        light = scene_cfg.light_cfg()
        robot = robot.robot_cfg()
        shoe = scene_cfg.shoe_cfg(layout.MOVING_SHOE, (PARK_POS[0], PARK_POS[1], move_pos[2]), move_quat)

    sim = SimulationContext(
        SimulationCfg(
            dt=PHYSICS_DT,
            device=args.device,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=FRICTION, dynamic_friction=FRICTION, restitution=0.0),
        )
    )
    scene = InteractiveScene(SceneCfg(num_envs=args.num_envs, env_spacing=2.0, replicate_physics=True))
    sim.reset()
    return sim, scene, poses


def main() -> int:
    config = layout.sample_configs(args.config_index + 1)[args.config_index]
    meta = run_files.load_shoe_meta(layout.SHOE_META_PATH)
    sim, scene, poses = build(config, meta)
    arm, shoe, dev, n = scene["robot"], scene["shoe"], sim.device, args.num_envs
    prof = robot.profile()
    origins = scene.env_origins
    arm_ids, _ = arm.find_joints(prof.arm_joint_regex)
    gravity_ids, _ = arm.find_joints(robot.GRAVITY_COMPENSATION_JOINTS)
    hand_ids = [arm.data.joint_names.index(name) for name in prof.hand_joint_names]
    palm = arm.find_bodies(prof.palm_body)[0][0]
    lower = arm.data.soft_joint_pos_limits[0, hand_ids, 0]
    upper = arm.data.soft_joint_pos_limits[0, hand_ids, 1]
    open_pose = torch.tensor(prof.hand_open_pose, device=dev)
    grip_pose = torch.tensor(prof.hand_grip_pose, device=dev)
    thumb3 = list(prof.hand_joint_names).index("r_hj_thumb_3")
    ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), num_envs=n, device=dev)
    home = arm.data.default_joint_pos.clone()
    shoe_obj = meta["objects"][layout.MOVING_SHOE]
    top_z = layout.TABLE_TOP_Z + 2.0 * float(shoe_obj["rest_height"])
    half_width = abs(float(shoe_obj["keypoint_offsets"][2][0]))
    start = torch.tensor([*poses[layout.MOVING_SHOE][0], *poses[layout.MOVING_SHOE][1]], dtype=torch.float32, device=dev)
    park = start.clone()
    park[:2] = torch.tensor(PARK_POS, device=dev)
    generator = torch.Generator().manual_seed(args.seed)
    zero_vel = torch.zeros(n, 6, device=dev)

    def write_shoe(local_pose):
        pose = local_pose.clone()
        pose[:, :3] += origins
        shoe.write_root_pose_to_sim(pose)
        shoe.write_root_velocity_to_sim(zero_vel)

    def run(steps, palm_goal, palm_quat, hand_target, parked=False):
        for _ in range(steps):
            if parked:
                write_shoe(park.expand(n, 7))
            ee_pos, ee_quat = arm.data.body_pos_w[:, palm] - origins, arm.data.body_quat_w[:, palm]
            ik.set_command(torch.cat([palm_goal, palm_quat], dim=-1), ee_pos=ee_pos, ee_quat=ee_quat)
            jac = arm.root_physx_view.get_jacobians()[:, palm - 1, :, arm_ids]
            target = arm.data.joint_pos_target.clone()
            target[:, arm_ids] = ik.compute(ee_pos, ee_quat, jac, arm.data.joint_pos[:, arm_ids])
            target[:, hand_ids] = hand_target() if callable(hand_target) else hand_target
            arm.set_joint_position_target(target)
            tau = arm.root_physx_view.get_gravity_compensation_forces()
            arm.set_joint_effort_target(tau[:, gravity_ids], joint_ids=gravity_ids)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(PHYSICS_DT)

    def lift_test(palm_goal, palm_quat, hand_target):
        z0 = shoe.data.root_pos_w[:, 2].clone()
        for k in range(1, LIFT_STEPS + 1):
            run(1, palm_goal + torch.tensor([0.0, 0.0, LIFT_HEIGHT_M * k / LIFT_STEPS], device=dev), palm_quat, hand_target)
        rel_lift = shoe.data.root_pos_w - arm.data.body_pos_w[:, palm]
        run(HOLD_STEPS, palm_goal + torch.tensor([0.0, 0.0, LIFT_HEIGHT_M], device=dev), palm_quat, hand_target)
        rel_hold = shoe.data.root_pos_w - arm.data.body_pos_w[:, palm]
        return gb.lift_held(z0, shoe.data.root_pos_w[:, 2], rel_lift, rel_hold)

    kept = {key: [] for key in ("joint_pos", "joint_target", "shoe_pose", "palm_pose")}
    stats = []
    t0 = time.time()
    for round_index in range(args.max_rounds):
        arm.write_joint_state_to_sim(home, torch.zeros_like(home))
        arm.set_joint_position_target(home)
        ik.reset()
        pre = gb.sample_pregrasp(n, generator, dev)
        palm_quat = quat_from_matrix(gb.palm_rotations(pre.tilt_deg, pre.yaw_deg))
        goal = gb.palm_goal_positions(pre, (config.move_x, config.move_y), config.move_yaw_deg, half_width, top_z)
        start_hand = open_pose.expand(n, -1).clone()
        start_hand[:, thumb3] = pre.thumb3
        start_hand = torch.max(torch.min(start_hand, upper), lower)
        above = goal + torch.tensor([0.0, 0.0, 0.12], device=dev)
        run(APPROACH_STEPS, above, palm_quat, start_hand, parked=True)
        run(DESCEND_STEPS, goal, palm_quat, start_hand, parked=True)
        write_shoe(start.expand(n, 7))
        run(SETTLE_STEPS, goal, palm_quat, start_hand)
        state = {"close": torch.zeros_like(start_hand), "target": start_hand.clone()}

        def closing():
            state["close"], state["target"] = gb.synergy_step(
                state["close"], state["target"], arm.data.joint_pos[:, hand_ids], start_hand, grip_pose, lower, upper
            )
            return state["target"]

        run(CLOSE_STEPS, goal, palm_quat, closing)
        record = {
            "joint_pos": arm.data.joint_pos.clone(),
            "joint_target": arm.data.joint_pos_target.clone(),
            "shoe_pose": torch.cat([shoe.data.root_pos_w - origins, shoe.data.root_quat_w], dim=-1),
            "palm_pose": torch.cat([arm.data.body_pos_w[:, palm] - origins, arm.data.body_quat_w[:, palm]], dim=-1),
        }
        held = lift_test(goal, palm_quat, state["target"])

        # Replay: restore the recorded state as an environment reset would, then lift again.
        arm.write_joint_state_to_sim(record["joint_pos"], torch.zeros_like(record["joint_pos"]))
        arm.set_joint_position_target(record["joint_target"])
        write_shoe(record["shoe_pose"])
        ik.reset()
        run(SETTLE_STEPS, record["palm_pose"][:, :3], record["palm_pose"][:, 3:], record["joint_target"][:, hand_ids])
        replay_held = lift_test(record["palm_pose"][:, :3], record["palm_pose"][:, 3:], record["joint_target"][:, hand_ids])
        ok = held & replay_held
        for key in kept:
            kept[key].append(record[key][ok])
        total = sum(int(t.shape[0]) for t in kept["joint_pos"])
        stats.append({"round": round_index, "held": int(held.sum()), "replay_held": int(replay_held.sum()), "kept": int(ok.sum()), "total": total})
        print(f"BANK round {round_index} held {int(held.sum())}/{n} replay {int(replay_held.sum())}/{n} kept {int(ok.sum())} total {total} ({time.time() - t0:.0f} s)", flush=True)
        if total >= args.min_entries:
            break

    entries = {key: torch.cat(parts) for key, parts in kept.items()}
    total = int(entries["joint_pos"].shape[0])
    metadata = {
        "config_index": config.index,
        "scene_config": vars(config),
        "robot_usd": str(robot.profile().usd_relpath),
        "physics_dt": PHYSICS_DT,
        "friction": FRICTION,
        "solver_position_iterations": robot.SOLVER_POSITION_ITERATIONS,
        "solver_velocity_iterations": robot.SOLVER_VELOCITY_ITERATIONS,
        "gains": gb.gains_metadata(arm.data.joint_names, arm.data.joint_stiffness[0], arm.data.joint_damping[0]),
        "shoe_meta_sha256": hashlib.sha256(layout.SHOE_META_PATH.read_bytes()).hexdigest(),
        "seed": args.seed,
        "rounds": stats,
    }
    out = layout.RUNS_DIR / f"config_{config.index:02d}" / "grasp_bank.json"
    run_files.write_json(out, gb.bank_document(entries, arm.data.joint_names, metadata))
    passed = total >= args.min_entries
    print(f"BANK config {config.index:02d} entries {total} (min {args.min_entries}) passed {passed} -> {out}", flush=True)
    return 0 if passed else 1


os._exit(main())
````

- [ ] **Step 2: 뱅크를 만든다**

Run: `cd ~/rl_ws/hdgp-iker && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm ../IsaacLab/_isaac_sim/python.sh scripts/iker/make_grasp_bank.py --config-index 0 --headless 2>&1 | grep -E "BANK|Traceback|Error"`
Expected(라운드당 약 28 s): 라운드별 `BANK round k held h/64 replay r/64 kept c total t` 줄, 마지막에 `BANK config 00 entries N (min 64) passed True`.
- 프로토타입에서는 라운드당 채택 5~13, 7 라운드에 69개였다.
- `passed False` 면 뱅크를 커밋하지 않고 멈춘 뒤 보고한다.

- [ ] **Step 3: 커밋한다**

```bash
cd ~/rl_ws/hdgp-iker
git add scripts/iker/make_grasp_bank.py iker_runs/shoe_place/config_00/grasp_bank.json
git commit -F - <<'MSG'
feat(iker): 파지 뱅크 생성 스크립트와 구성 0 파지 뱅크

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4
MSG
```

---

### Task 3: 학습 환경·gym 등록·PPO 설정

**Files:**
- Create: `T/events.py`, `T/iker_shoe_env_cfg.py`, `T/iker_shoe_env.py`, `T/config/__init__.py`, `T/config/agents/__init__.py`(빈 파일), `T/config/agents/rl_games_ppo_cfg.yaml`

**Interfaces:**
- Consumes:
  - `grasp_bank.load_bank`, `grasp_bank.gains_metadata`
  - `reward.compute_reward_and_termination`, `reward.transform_keypoints`, `reward.reward_cfg_for_start_support`, `reward.NUM_KEYPOINTS`, `reward.RewardOutput`, `reward.IkerRewardCfg`
  - `run_files.read_json`, `run_files.load_shoe_meta`
  - `layout.*`, `robot.*`, `scene_cfg.*`
- Produces:
  - gym id `open-sens_r_iker_shoe`(4096 env, 잡음 켬), `open-sens_r_iker_shoe-play`(50 env, 잡음 끔)
  - `IkerShoeEnvCfg` 필드 `config_index`, `interaction_source`(`human`|`vlm`), `interaction_path`, `keypoints_path`, `grasp_bank_path`, `add_noise` — 비어 있으면 `iker_runs/shoe_place/config_XX/` 기본 경로
  - TFEvents 로그 `Episode/iker/{keypoint_distance_m, success_5cm, sustained_success, dropped}`
  - 계획 3 의 평가가 이 필드와 로그를 쓴다.

필드는 `__post_init__` 이 아니라 `__init__` 에서 읽는다(hydra 재정의가 `__post_init__` 뒤에 적용된다). 이 파일들은 isaaclab 을 import 하므로 유닛 테스트가 없다. 검증은 Task 4 에서 한다.

- [ ] **Step 1: 무작위화·설정·환경 파일을 만든다**

Create `source/openarm/openarm/agnostic/tasks/iker_shoe/events.py`:

````python
"""Event terms of the IKER shoe task that Isaac Lab's stock terms do not cover."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform


def randomize_rigid_object_com(
    env: ManagerBasedEnv, env_ids: torch.Tensor | None, com_range: dict[str, tuple[float, float]], asset_cfg: SceneEntityCfg
) -> None:
    """Add a uniform offset to a single-body RigidObject's centre of mass.

    ``isaaclab.envs.mdp.randomize_rigid_body_com`` indexes CoMs as (envs, bodies, 7), the articulation layout;
    a RigidObject view returns (envs, 7) and that term raises IndexError.
    """
    asset = env.scene[asset_cfg.name]
    ids = torch.arange(env.scene.num_envs, device="cpu") if env_ids is None else env_ids.cpu()
    bounds = torch.tensor([com_range.get(axis, (0.0, 0.0)) for axis in ("x", "y", "z")], device="cpu")
    offsets = sample_uniform(bounds[:, 0], bounds[:, 1], (len(ids), 3), device="cpu")
    coms = asset.root_physx_view.get_coms().clone()
    if coms.ndim != 2 or coms.shape[1] != 7:
        raise ValueError(f"{asset_cfg.name}: expected (envs, 7) CoMs from a single-body view, got {tuple(coms.shape)}")
    coms[ids, :3] += offsets
    asset.root_physx_view.set_coms(coms, ids)
````

Create `source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_env_cfg.py`:

````python
"""Isaac Lab configuration of the IKER shoe-placement training environment (design spec §7)."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from openarm.agnostic.modules.iker.reward import IkerRewardCfg, reward_cfg_for_start_support

from . import events, layout, robot, scene_cfg

PHYSICS_DT = 1.0 / 120.0
DECIMATION = 12  # 10 Hz policy, as the paper
EPISODE_STEPS = 200
CONTROL_DT = PHYSICS_DT * DECIMATION
FRICTION = 1.0  # contact material shared with make_grasp_bank.py (checked against the bank at boot)
SHOE_REST_QUAT_WXYZ = (0.5, -0.5, 0.5, -0.5)  # spawn only; every reset overwrites the shoe poses


@configclass
class IkerShoeEventCfg:
    """Legacy IKER object randomization (spec §7), sampled once per environment at startup."""

    shoe_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(layout.MOVING_SHOE),
            "mass_distribution_params": (0.3, 2.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    shoe_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(layout.MOVING_SHOE, body_names=".*"),
            # absolute values; with the nominal friction 1.0 this is the legacy x0.3-1.8 scale
            "static_friction_range": (0.3 * FRICTION, 1.8 * FRICTION),
            "dynamic_friction_range": (0.3 * FRICTION, 1.8 * FRICTION),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 250,
        },
    )
    shoe_com = EventTermCfg(
        func=events.randomize_rigid_object_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(layout.MOVING_SHOE),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (-0.025, 0.025)},
        },
    )


@configclass
class IkerShoeEnvCfg(DirectRLEnvCfg):
    decimation = DECIMATION
    episode_length_s = EPISODE_STEPS * CONTROL_DT
    action_space = 6
    observation_space = 38
    state_space = 0

    # action: normalized delta palm pose per policy step
    action_pos_scale = 0.02
    action_rot_scale = 0.05

    # noise (legacy IKER, training only)
    add_noise = True
    observation_noise = 0.02
    action_noise = 0.05
    quat_noise_rad = 0.2
    other_shoe_pos_noise = 0.01
    other_shoe_yaw_noise = 0.05

    # run artifacts; empty paths resolve to iker_runs/shoe_place/config_XX/
    config_index = 0
    interaction_source = "human"
    interaction_path = ""
    keypoints_path = ""
    grasp_bank_path = ""
    eval_success_distance_m = 0.05

    reward: IkerRewardCfg = reward_cfg_for_start_support(layout.TABLE_TOP_Z, max_episode_length=EPISODE_STEPS)
    events: IkerShoeEventCfg = IkerShoeEventCfg()

    viewer: ViewerCfg = ViewerCfg(eye=(1.1, 0.9, 0.85), lookat=(0.25, -0.02, 0.3), origin_type="env", env_index=0)
    sim: SimulationCfg = SimulationCfg(
        dt=PHYSICS_DT,
        render_interval=DECIMATION,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=FRICTION, dynamic_friction=FRICTION, restitution=0.0),
        # 4096 envs overflow the default PhysX buffers and drop contacts (measured 2026-09-13)
        physx=PhysxCfg(gpu_collision_stack_size=2**29, gpu_max_rigid_patch_count=2**23, gpu_max_rigid_contact_count=2**24),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)

    robot_cfg: ArticulationCfg = robot.robot_cfg()
    shoe_move_cfg: RigidObjectCfg = scene_cfg.shoe_cfg(layout.MOVING_SHOE, (0.27, 0.14, 0.26), SHOE_REST_QUAT_WXYZ)
    shoe_other_cfg: RigidObjectCfg = scene_cfg.shoe_cfg(layout.OTHER_SHOE, (0.27, -0.15, 0.38), SHOE_REST_QUAT_WXYZ)
    table_cfg: AssetBaseCfg = scene_cfg.table_cfg()
    rack_cfg: AssetBaseCfg = scene_cfg.rack_cfg()
    light_cfg: AssetBaseCfg = scene_cfg.light_cfg()


@configclass
class IkerShoePlayEnvCfg(IkerShoeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.add_noise = False
````

Create `source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_env.py`:

````python
"""IKER shoe-placement environment (design spec §7).

Episodes start from a grasp-bank state (shoe already in the closed hand on the table). The policy moves the
palm with a 6-D delta pose through damped least-squares IK at 10 Hz; the hand keeps the bank's commanded
targets. The reward is the fixed IKER reward toward the target keypoints of one interaction file.
"""

from __future__ import annotations

from pathlib import Path

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_from_angle_axis, quat_mul, sample_uniform, subtract_frame_transforms

from openarm.agnostic.modules.iker import run_files
from openarm.agnostic.modules.iker.reward import NUM_KEYPOINTS, RewardOutput, compute_reward_and_termination, transform_keypoints

from . import grasp_bank as gb
from . import layout, robot
from .iker_shoe_env_cfg import FRICTION, PHYSICS_DT, IkerShoeEnvCfg


def _artifact(path_text: str, default: Path) -> Path:
    return Path(path_text) if path_text else default


class IkerShoeEnv(DirectRLEnv):
    cfg: IkerShoeEnvCfg

    def __init__(self, cfg: IkerShoeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        if self.max_episode_length != cfg.reward.max_episode_length:
            raise ValueError(
                f"episode is {self.max_episode_length} steps but the reward expects {cfg.reward.max_episode_length}"
            )
        run_dir = layout.RUNS_DIR / f"config_{cfg.config_index:02d}"
        meta = run_files.load_shoe_meta(layout.SHOE_META_PATH)
        interaction = run_files.read_json(_artifact(cfg.interaction_path, run_dir / f"interaction_{cfg.interaction_source}.json"))
        if not interaction["gate"]["passed"] or interaction["object"] != layout.MOVING_SHOE:
            raise ValueError(f"interaction for {interaction.get('object')} did not pass the gate: {interaction['gate']['failures']}")
        snapshot = run_files.read_json(_artifact(cfg.keypoints_path, run_dir / "keypoints.json"))

        self._offsets = torch.tensor(meta["objects"][layout.MOVING_SHOE]["keypoint_offsets"], dtype=torch.float32, device=dev)
        self._targets = torch.tensor(interaction["target_keypoints"], dtype=torch.float32, device=dev)
        if self._targets.shape != (NUM_KEYPOINTS, 3) or self._offsets.shape != (NUM_KEYPOINTS, 3):
            raise ValueError("interaction targets and shoe keypoint offsets must both be (4, 3)")
        other = snapshot["objects"][layout.OTHER_SHOE]
        self._other_pos = torch.tensor(other["position"], dtype=torch.float32, device=dev)
        self._other_quat = torch.tensor(other["quat_wxyz"], dtype=torch.float32, device=dev)

        prof = robot.profile()
        self._arm_ids, _ = self._robot.find_joints(prof.arm_joint_regex)
        self._gravity_ids, _ = self._robot.find_joints(robot.GRAVITY_COMPENSATION_JOINTS)
        self._palm = self._robot.find_bodies(prof.palm_body)[0][0]
        self._palm_jacobian = self._palm - 1  # fixed-base Jacobians omit the root body
        expected = {
            "config_index": cfg.config_index,
            "physics_dt": PHYSICS_DT,
            "friction": FRICTION,
            "solver_position_iterations": robot.SOLVER_POSITION_ITERATIONS,
            "solver_velocity_iterations": robot.SOLVER_VELOCITY_ITERATIONS,
            "gains": gb.gains_metadata(self._robot.data.joint_names, self._robot.data.joint_stiffness[0], self._robot.data.joint_damping[0]),
        }
        bank_doc = run_files.read_json(_artifact(cfg.grasp_bank_path, run_dir / "grasp_bank.json"))
        self._bank = gb.load_bank(bank_doc, self._robot.data.joint_names, expected, dev)

        self._ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"), num_envs=n, device=dev
        )
        self._action_scale = torch.tensor([cfg.action_pos_scale] * 3 + [cfg.action_rot_scale] * 3, device=dev)
        self._joint_targets = self._robot.data.default_joint_pos.clone()
        self._init_keypoints = torch.zeros(n, NUM_KEYPOINTS, 3, device=dev)
        self._keypoint_distance = torch.zeros(n, device=dev)
        self._success_count = torch.zeros(n, device=dev)
        self._failure_count = torch.zeros(n, device=dev)
        self._last: RewardOutput | None = None
        self.actions = torch.zeros(n, cfg.action_space, device=dev)

    # ---------------------------------------------------------------- scene

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self._shoe = RigidObject(self.cfg.shoe_move_cfg)
        self._other = RigidObject(self.cfg.shoe_other_cfg)
        for asset in (self.cfg.table_cfg, self.cfg.rack_cfg):
            asset.spawn.func(asset.prim_path, asset.spawn, translation=asset.init_state.pos)
        self.cfg.light_cfg.spawn.func(self.cfg.light_cfg.prim_path, self.cfg.light_cfg.spawn)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects[layout.MOVING_SHOE] = self._shoe
        self.scene.rigid_objects[layout.OTHER_SHOE] = self._other

    # --------------------------------------------------------------- action

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        command = self.actions
        if self.cfg.add_noise:
            command = (command + sample_uniform(-self.cfg.action_noise, self.cfg.action_noise, command.shape, self.device)).clamp(-1.0, 1.0)
        root = self._robot.data.root_pose_w
        palm = self._robot.data.body_pose_w[:, self._palm]
        palm_pos_b, palm_quat_b = subtract_frame_transforms(root[:, :3], root[:, 3:7], palm[:, :3], palm[:, 3:7])
        self._ik.set_command(command * self._action_scale, ee_pos=palm_pos_b, ee_quat=palm_quat_b)
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self._palm_jacobian, :, self._arm_ids]
        arm_targets = self._ik.compute(palm_pos_b, palm_quat_b, jacobian, self._robot.data.joint_pos[:, self._arm_ids])
        limits = self._robot.data.soft_joint_pos_limits[:, self._arm_ids]
        self._joint_targets[:, self._arm_ids] = torch.clamp(arm_targets, limits[..., 0], limits[..., 1])

    def _apply_action(self):
        self._robot.set_joint_position_target(self._joint_targets)
        tau = self._robot.root_physx_view.get_gravity_compensation_forces()
        self._robot.set_joint_effort_target(tau[:, self._gravity_ids], joint_ids=self._gravity_ids)

    # ---------------------------------------------------------- observation

    def _keypoints_local(self) -> torch.Tensor:
        world = transform_keypoints(self._shoe.data.root_pos_w, self._shoe.data.root_quat_w, self._offsets)
        return world - self.scene.env_origins[:, None, :]

    def _noisy_quat(self, quat: torch.Tensor) -> torch.Tensor:
        n = quat.shape[0]
        noisy = quat
        for axis in torch.eye(3, device=self.device):
            angle = sample_uniform(-self.cfg.quat_noise_rad, self.cfg.quat_noise_rad, (n,), self.device)
            noisy = quat_mul(noisy, quat_from_angle_axis(angle, axis.expand(n, 3)))
        # q and -q are the same rotation: flip the whole quaternion (the legacy task flipped components one by one)
        sign = torch.where(torch.rand(n, 1, device=self.device) < 0.5, -1.0, 1.0)
        return noisy * sign

    def _get_observations(self) -> dict:
        origins = self.scene.env_origins
        palm_pos = self._robot.data.body_pos_w[:, self._palm] - origins
        palm_quat = self._robot.data.body_quat_w[:, self._palm]
        shoe_pos = self._shoe.data.root_pos_w - origins
        shoe_quat = self._shoe.data.root_quat_w
        if self.cfg.add_noise:
            palm_quat, shoe_quat = self._noisy_quat(palm_quat), self._noisy_quat(shoe_quat)
        n = self.num_envs
        obs = torch.cat(
            [
                palm_pos,
                palm_quat[:, [1, 2, 3, 0]],
                shoe_pos,
                shoe_quat[:, [1, 2, 3, 0]],
                self._keypoints_local().reshape(n, -1),
                self._targets.expand(n, NUM_KEYPOINTS, 3).reshape(n, -1),
            ],
            dim=-1,
        )
        if self.cfg.add_noise:
            obs = obs + sample_uniform(-self.cfg.observation_noise, self.cfg.observation_noise, obs.shape, self.device)
        return {"policy": obs}

    # ------------------------------------------------------- reward / dones

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        origins = self.scene.env_origins
        current = self._keypoints_local()
        targets = self._targets.expand(self.num_envs, NUM_KEYPOINTS, 3)
        out = compute_reward_and_termination(
            actions=self.actions,
            eef_pos=self._robot.data.body_pos_w[:, self._palm] - origins,
            object_pos=self._shoe.data.root_pos_w - origins,
            current_keypoints=current,
            init_keypoints=self._init_keypoints,
            target_keypoints=targets,
            progress=self.episode_length_buf,
            success_count=self._success_count,
            failure_count=self._failure_count,
            cfg=self.cfg.reward,
        )
        self._success_count, self._failure_count, self._last = out.success_count, out.failure_count, out
        self._keypoint_distance = (current - targets).norm(dim=-1).mean(dim=-1)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return out.terminated & ~truncated, truncated

    def _get_rewards(self) -> torch.Tensor:
        if self._last is None:
            raise RuntimeError("_get_rewards called before _get_dones")
        return self._last.reward

    # ----------------------------------------------------------------- reset

    def _log_episode_end(self, env_ids: torch.Tensor) -> None:
        finished = env_ids[self.episode_length_buf[env_ids] > 0]
        if len(finished) == 0:
            return
        distance = self._keypoint_distance[finished]
        sustain = self.cfg.reward.sustain_steps
        self.extras["log"] = {
            "iker/keypoint_distance_m": distance.mean().item(),
            "iker/success_5cm": (distance <= self.cfg.eval_success_distance_m).float().mean().item(),
            "iker/sustained_success": (self._success_count[finished] > sustain).float().mean().item(),
            "iker/dropped": (self._failure_count[finished] > sustain).float().mean().item(),
        }

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        self._log_episode_end(env_ids)
        super()._reset_idx(env_ids)
        count, dev = len(env_ids), self.device
        pick = torch.randint(0, self._bank.size, (count,), device=dev)
        joint_pos = self._bank.joint_pos[pick]
        joint_target = self._bank.joint_target[pick]
        self._robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        self._robot.set_joint_position_target(joint_target, env_ids=env_ids)
        self._joint_targets[env_ids] = joint_target

        origins = self.scene.env_origins[env_ids]
        zero_velocity = torch.zeros(count, 6, device=dev)
        shoe_pose = self._bank.shoe_pose[pick].clone()
        shoe_pose[:, :3] += origins
        self._shoe.write_root_pose_to_sim(shoe_pose, env_ids=env_ids)
        self._shoe.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

        other_pose = torch.zeros(count, 7, device=dev)
        xy_noise = sample_uniform(-self.cfg.other_shoe_pos_noise, self.cfg.other_shoe_pos_noise, (count, 2), dev)
        other_pose[:, :3] = self._other_pos + origins
        other_pose[:, :2] += xy_noise
        yaw = sample_uniform(-self.cfg.other_shoe_yaw_noise, self.cfg.other_shoe_yaw_noise, (count,), dev)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(count, 3)
        other_pose[:, 3:] = quat_mul(quat_from_angle_axis(yaw, z_axis), self._other_quat.expand(count, 4))
        self._other.write_root_pose_to_sim(other_pose, env_ids=env_ids)
        self._other.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

        self._init_keypoints[env_ids] = (
            transform_keypoints(shoe_pose[:, :3], shoe_pose[:, 3:], self._offsets) - origins[:, None, :]
        )
        self._keypoint_distance[env_ids] = 0.0
        self._success_count[env_ids] = 0.0
        self._failure_count[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self._ik.reset(env_ids)
````

- [ ] **Step 2: gym 등록과 PPO 설정을 만든다**

```bash
cd ~/rl_ws/hdgp-iker && mkdir -p source/openarm/openarm/agnostic/tasks/iker_shoe/config/agents
: > source/openarm/openarm/agnostic/tasks/iker_shoe/config/agents/__init__.py
```

Create `source/openarm/openarm/agnostic/tasks/iker_shoe/config/__init__.py`:

````python
"""gym registration — IKER shoe placement (design spec §3, §7).

★gym id 규약: train.py 의 run_naming 정규식 `^(open-[A-Za-z0-9]+)_([rbl])_(.+?)...` 에 걸려야 로그가
  `log/rl_games/<robot>/<side>/<task>/` 로 분리된다. `-play` id 도 같이 등록한다.
"""

import gymnasium as gym

from . import agents

_ENTRY = "openarm.agnostic.tasks.iker_shoe.iker_shoe_env:IkerShoeEnv"
_CFG_MODULE = "openarm.agnostic.tasks.iker_shoe.iker_shoe_env_cfg"

for _suffix, _cfg_name in (("", "IkerShoeEnvCfg"), ("-play", "IkerShoePlayEnvCfg")):
    gym.register(
        id=f"open-sens_r_iker_shoe{_suffix}",
        entry_point=_ENTRY,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{_CFG_MODULE}:{_cfg_name}",
            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        },
    )
````

Create `source/openarm/openarm/agnostic/tasks/iker_shoe/config/agents/rl_games_ppo_cfg.yaml`:

````yaml
# Legacy IKER PPO (repo/skill_gen/IKER/isaacgymenvs/cfg/train/ShoePlacePPO.yaml), design spec §7.
# Differences: max_epochs 250 at 4096 envs keeps the legacy budget (10240 x 32 x 100 = 32.8 M frames);
# checkpoints every 50 epochs so a run of 250 keeps several.
params:
  seed: 42
  env:
    clip_observations: 5.0
    clip_actions: 1.0
  algo:
    name: a2c_continuous
  model:
    name: continuous_a2c_logstd
  network:
    name: actor_critic
    separate: False
    space:
      continuous:
        mu_activation: None
        sigma_activation: None
        mu_init:
          name: default
        sigma_init:
          name: const_initializer
          val: 0
        fixed_sigma: True
    mlp:
      units: [256, 128, 64]
      activation: elu
      d2rl: False
      initializer:
        name: default
      regularizer:
        name: None
  load_checkpoint: False
  load_path: ''
  config:
    name: iker_shoe
    env_name: rlgpu
    device: 'cuda:0'
    device_name: 'cuda:0'
    multi_gpu: False
    ppo: True
    mixed_precision: False
    normalize_input: False
    normalize_value: True
    value_bootstrap: True
    num_actors: -1
    reward_shaper:
      scale_value: 1.0
    normalize_advantage: True
    gamma: 0.99
    tau: 0.95
    learning_rate: 5e-4
    lr_schedule: adaptive
    schedule_type: standard
    kl_threshold: 0.008
    score_to_win: 100000
    max_epochs: 250
    save_best_after: 50
    save_frequency: 50
    print_stats: True
    grad_norm: 1.0
    entropy_coef: 0.0
    truncate_grads: True
    e_clip: 0.2
    horizon_length: 32
    minibatch_size: 16384
    mini_epochs: 5
    critic_coef: 4
    clip_value: True
    seq_length: 4
    bounds_loss_coef: 0.0001
````

- [ ] **Step 3: 순수 테스트가 여전히 통과하고 등록 경로가 발견되는지 확인한다**

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/modules/iker/tests source/openarm/openarm/agnostic/tasks/iker_shoe/tests -q -p no:cacheprovider`
Expected: `116 passed`(`modules/iker` 95 + `tasks/iker_shoe` 21)

Run: `cd ~/rl_ws/hdgp-iker && python3 -c "from pathlib import Path; root = Path('source/openarm/openarm'); print(sorted(str(p) for p in root.glob('*/*/*/config/__init__.py') if 'iker_shoe' in str(p)))"`
Expected: `['source/openarm/openarm/agnostic/tasks/iker_shoe/config/__init__.py']`
(`openarm/tasks/__init__.py` 의 `*/*/*/config/__init__.py` glob 이 이 경로를 import 한다.)

- [ ] **Step 4: 커밋한다**

```bash
cd ~/rl_ws/hdgp-iker && T=source/openarm/openarm/agnostic/tasks/iker_shoe
git add $T/events.py $T/iker_shoe_env_cfg.py $T/iker_shoe_env.py $T/config/__init__.py $T/config/agents/__init__.py $T/config/agents/rl_games_ppo_cfg.yaml
git commit -F - <<'MSG'
feat(iker): 신발 놓기 학습 환경 — 파지 뱅크 리셋·차분 IK 액션·IKER 고정 보상·레거시 무작위화·gym 등록

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4
MSG
```

---

### Task 4: 학습 전 시뮬레이터 검사(Isaac)

**Files:**
- Create: `scripts/iker/env_smoke.py`

**Interfaces:**
- Consumes: gym id `open-sens_r_iker_shoe`, `IkerShoeEnvCfg`, 환경 내부 버퍼 `_bank`, `_shoe`, `_robot`, `_palm`, `_palm_jacobian`, `_arm_ids`, `_gravity_ids`, `_offsets`, `_targets`, `_other_pos`, `_success_count`, `_keypoint_distance`, `_joint_targets`, `gate.kabsch`
- Produces: `SMOKE passed True` 판정(학습 착수 조건)

검사 5종:
1. 부팅과 관측 (N, 38) 유한.
2. 행동 0 5 s 동안 떨어짐 비율 ≤ 0.25.
3. 무작위 행동 25 s 동안 보상 유한과 에피소드 로그.
4. 로봇을 홈으로 치운 뒤 목표 자세에 놓은 신발의 성공 판정 100 %.
5. 뱅크 파지 자세로 상호작용 목표 자리 IK 도달 오차 ≤ 2 cm(반대쪽 자리는 측정·보고만).

- [ ] **Step 1: 스크립트를 만든다**

Create `scripts/iker/env_smoke.py`:

````python
"""Smoke-check the IKER shoe environment before training (design spec §10, simulator checks).

1. boots with the grasp bank and the gated interaction, observations (N, 38) are finite;
2. with zero actions the shoe stays in the hand for 5 s;
3. random actions for 25 s keep rewards finite and log episode-end metrics;
4. a shoe placed at the interaction's target keypoints (robot moved home, out of the way) counts as a success;
5. the right arm reaches the bank's grasp palm pose above the interaction's target slot (the mirrored slot on the
   other side of the other shoe is measured and reported, not required).

Usage:
    cd ~/rl_ws/hdgp && PYTHONPATH=source/openarm ../IsaacLab/isaaclab.sh -p scripts/iker/env_smoke.py --headless
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke-check the IKER shoe environment.")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--config-index", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    print("SMOKE FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.agnostic.tasks.iker_shoe.config  # noqa: E402,F401  (registers the gym ids)
from openarm.agnostic.modules.iker.gate import kabsch  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.iker_shoe_env_cfg import IkerShoeEnvCfg  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.utils.math import quat_from_matrix  # noqa: E402

MAX_DROP_FRACTION_ZERO_ACTION = 0.25  # startup DR (friction down to 0.3, mass up to x2) loosens some grasps
MAX_SLOT_REACH_ERROR_M = 0.02
PARKED_SHOE = (0.42, 0.40, 0.26)
IK_REACH_STEPS = 360


def reach_slots(env) -> dict[str, float]:
    """Worst palm position error (m) when IK drives every env's bank grasp pose above the target slot and its mirror."""
    n, dev = env.num_envs, env.device
    robot, origins, palm = env._robot, env.scene.env_origins, env._palm
    ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), num_envs=n, device=dev)
    entry = torch.arange(n, device=dev) % env._bank.size
    grasp_offset = env._bank.palm_pose[entry, :3] - env._bank.shoe_pose[entry, :3]
    palm_quat = env._bank.palm_pose[entry, 3:]
    parked = torch.zeros(n, 7, device=dev)
    parked[:, :3] = torch.tensor(PARKED_SHOE, device=dev) + origins
    parked[:, 3] = 1.0
    _, centre = kabsch(env._offsets.cpu().numpy(), env._targets.cpu().numpy())
    target_slot = torch.tensor(centre, dtype=torch.float32, device=dev)
    mirrored_slot = target_slot.clone()
    mirrored_slot[1] = 2.0 * env._other_pos[1] - target_slot[1]
    errors = {}
    for name, slot in (("target_slot", target_slot), ("mirrored_slot", mirrored_slot)):
        env.reset()
        ik.reset()
        goal = slot + grasp_offset
        for _ in range(IK_REACH_STEPS):
            env._shoe.write_root_pose_to_sim(parked)
            env._shoe.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
            ee_pos, ee_quat = robot.data.body_pos_w[:, palm] - origins, robot.data.body_quat_w[:, palm]
            ik.set_command(torch.cat([goal, palm_quat], dim=-1), ee_pos=ee_pos, ee_quat=ee_quat)
            jacobian = robot.root_physx_view.get_jacobians()[:, env._palm_jacobian, :, env._arm_ids]
            target = robot.data.joint_pos_target.clone()
            target[:, env._arm_ids] = ik.compute(ee_pos, ee_quat, jacobian, robot.data.joint_pos[:, env._arm_ids])
            robot.set_joint_position_target(target)
            tau = robot.root_physx_view.get_gravity_compensation_forces()
            robot.set_joint_effort_target(tau[:, env._gravity_ids], joint_ids=env._gravity_ids)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
        errors[name] = float((robot.data.body_pos_w[:, palm] - origins - goal).norm(dim=-1).max())
    return errors


def main() -> int:
    cfg = IkerShoeEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.config_index = args.config_index
    env = gym.make("open-sens_r_iker_shoe", cfg=cfg).unwrapped
    n, dev = env.num_envs, env.device
    failures = []

    obs, _ = env.reset()
    policy = obs["policy"]
    print(f"SMOKE obs {tuple(policy.shape)} finite {bool(torch.isfinite(policy).all())} bank {env._bank.size}", flush=True)
    if policy.shape != (n, 38) or not torch.isfinite(policy).all():
        failures.append("observation shape or finiteness")

    z0 = env._shoe.data.root_pos_w[:, 2].clone()
    for _ in range(50):
        env.step(torch.zeros(n, 6, device=dev))
    dz = env._shoe.data.root_pos_w[:, 2] - z0
    dropped = float((dz < -0.03).float().mean())
    print(f"SMOKE zero action 5 s: shoe dz mean {float(dz.mean()) * 1000:+.1f} mm, dropped fraction {dropped:.2f}", flush=True)
    if not dropped <= MAX_DROP_FRACTION_ZERO_ACTION:
        failures.append(f"zero-action drop fraction {dropped:.2f}")

    rewards, logs = [], []
    for _ in range(250):
        _, reward, terminated, truncated, extras = env.step(2.0 * torch.rand(n, 6, device=dev) - 1.0)
        rewards.append(reward)
        if "log" in extras:
            logs.append(dict(extras["log"]))
    stacked = torch.stack(rewards)
    print(f"SMOKE random 25 s: reward finite {bool(torch.isfinite(stacked).all())} mean {float(stacked.mean()):.3f} min {float(stacked.min()):.1f} max {float(stacked.max()):.1f} logs {len(logs)} last {logs[-1] if logs else None}", flush=True)
    if not torch.isfinite(stacked).all() or not logs:
        failures.append("random-action rewards or episode logs")

    # Move the robot home first: at a bank state the closed hand sits where the thumb would push a shoe placed
    # at the target. Then place every shoe at the target pose (rigid fit of its keypoints).
    env.reset()
    home = env._robot.data.default_joint_pos.clone()
    env._robot.write_joint_state_to_sim(home, torch.zeros_like(home))
    env._robot.set_joint_position_target(home)
    env._joint_targets[:] = home
    rot, centre = kabsch(env._offsets.cpu().numpy(), env._targets.cpu().numpy())
    pose = torch.zeros(n, 7, device=dev)
    pose[:, :3] = torch.tensor(centre, dtype=torch.float32, device=dev) + env.scene.env_origins
    pose[:, 3:] = quat_from_matrix(torch.tensor(rot, dtype=torch.float32, device=dev)).expand(n, 4)
    env._shoe.write_root_pose_to_sim(pose)
    env._shoe.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
    env.step(torch.zeros(n, 6, device=dev))
    counted = float((env._success_count > 0).float().mean())
    distance = float(env._keypoint_distance.mean())
    print(f"SMOKE target placement: success counted {counted:.2f}, mean keypoint distance {distance * 1000:.1f} mm", flush=True)
    if not counted == 1.0:
        failures.append(f"target placement counted as success in {counted:.2f} of envs")

    errors = reach_slots(env)
    print(f"SMOKE rack slot reach: worst palm error mm {({k: round(v * 1000, 1) for k, v in errors.items()})}", flush=True)
    if not errors["target_slot"] <= MAX_SLOT_REACH_ERROR_M:
        failures.append(f"target slot: palm error {errors['target_slot'] * 1000:.1f} mm")

    for failure in failures:
        print(f"SMOKE CHECK FAILED: {failure}", flush=True)
    print(f"SMOKE passed {not failures}", flush=True)
    return 0 if not failures else 1


os._exit(main())
````

- [ ] **Step 2: 16 환경으로 검사한다**

Run: `cd ~/rl_ws/hdgp-iker && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm ../IsaacLab/_isaac_sim/python.sh scripts/iker/env_smoke.py --num-envs 16 --headless 2>&1 | grep -E "SMOKE|Traceback|Error"`
Expected(약 90 s):
```text
SMOKE obs (16, 38) finite True bank <N>
SMOKE zero action 5 s: shoe dz mean ... mm, dropped fraction 0.00
SMOKE random 25 s: reward finite True ... logs <양수> last {...}
SMOKE target placement: success counted 1.00, mean keypoint distance ... mm
SMOKE rack slot reach: worst palm error mm {'target_slot': ..., 'mirrored_slot': ...}
SMOKE passed True
```
프로토타입 값은 `target_slot` 1.5 mm, `mirrored_slot` 114.4 mm 였다. 반대쪽 자리는 뱅크 파지 자세로 오른팔이 닿지 않는다(판정 제외, 보고만 한다).
이것은 계획 3 에서 VLM 이 그 자리를 고를 때 G6(손바닥 상자)이 걸러내지 못하는 경우라 인계 항목에 넣는다.

- [ ] **Step 3: 4096 환경 학습 스모크(3 에포크)**

Run: `cd ~/rl_ws/hdgp-iker && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm ../IsaacLab/_isaac_sim/python.sh scripts/reinforcement_learning/rl_games/train.py --task open-sens_r_iker_shoe --num_envs 4096 --headless --max_iterations 3 2>&1 | grep -E "Logging experiment|fps total|MAX EPOCHS|overflow|Traceback|Error:"`
Expected(약 90 s):
- `Logging experiment in directory: /home/user/rl_ws/hdgp-iker/log/rl_games/open-sens/right/iker-shoe`
- epoch 1~3 의 `fps total` 이 약 7,000~8,200(프로토타입 6,975→8,161)
- `MAX EPOCHS NUM!`
- `overflow` 줄 없음

Run: `cd ~/rl_ws/hdgp-iker && ls log/rl_games/open-sens/right/iker-shoe/*/nn/*.pth log/rl_games/open-sens/right/iker-shoe/*/summaries/events.out.tfevents.*`
Expected: 방금 만든 런 폴더(`test<N>`)의 체크포인트 1개 이상과 TFEvents 파일 1개. `log/` 는 git 이 무시한다.

- [ ] **Step 4: 커밋한다**

```bash
cd ~/rl_ws/hdgp-iker
git add scripts/iker/env_smoke.py
git commit -F - <<'MSG'
feat(iker): 학습 전 시뮬레이터 검사 — 부팅·파지 유지·보상·목표 성공 판정·받침 두 자리 도달

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4
MSG
```

---

### Task 5: 사람 기준선 학습 착수와 인계

이 태스크는 코드를 바꾸지 않는다. 학습을 띄우고, 돌아가는 것을 확인하고, 인계한다(관찰 0155: 완료는 결과를 쓸 수 있는 시점으로 정의한다).

- [ ] **Step 1: 착수 전 확인**

Run: `cd ~/rl_ws/hdgp-iker && git status --short && git log --oneline -1 && nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader`
Expected:
- `git status` 출력 없음.
- 마지막 커밋이 Task 4 커밋이다.
- GPU 사용 메모리가 전체의 절반 미만이다. 다른 로컬 작업이 GPU 를 쓰고 있으면 멈추고 보고한다.

- [ ] **Step 2: 250 에포크 학습을 백그라운드로 띄운다**

```bash
cd ~/rl_ws/hdgp-iker
RUN_LOG=$HOME/rl_ws/our_source/iker_render/train_iker_human_c00.log
RUN_LABEL=iker_human_c00 NOTE="IKER 사람 기준선 구성 0, 레거시 PPO 250 epoch, 4096 env (계획 2)" \
TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm \
setsid nohup ../IsaacLab/_isaac_sim/python.sh scripts/reinforcement_learning/rl_games/train.py \
  --task open-sens_r_iker_shoe --num_envs 4096 --headless > "$RUN_LOG" 2>&1 < /dev/null &
echo "PID=$!"
```
Expected: `PID=<숫자>`. 이 PID 가 학습 프로세스 그룹의 대표다. 종료는 이 PID 로만 한다.

- [ ] **Step 3: 첫 에포크를 확인한다**

Run: `sleep 150; grep -E "Logging experiment|fps total|Traceback|Error:" $HOME/rl_ws/our_source/iker_render/train_iker_human_c00.log | tail -5`
Expected:
- `Logging experiment in directory: .../log/rl_games/open-sens/right/iker-shoe`
- `fps total` 이 붙은 epoch 줄 1개 이상
- `Traceback` 없음
- 런 폴더 이름은 `iker_human_c00`

- [ ] **Step 4: 인계**

사용자에게 다음을 보고한다.
1. 커밋 목록(`git log --oneline` 계획 2 의 커밋 4개)과 테스트 수.
2. 뱅크 크기, 스모크 5종 결과, 학습 스모크 fps.
3. 학습 PID, 로그 경로, 런 폴더 `log/rl_games/open-sens/right/iker-shoe/iker_human_c00`, 예상 소요(250 에포크 × 131,072 프레임 ÷ 약 8,000 fps ≈ 70 분).
4. 모니터링 명령: `python3 scripts/tools/parse_tfevents.py <런 폴더>/summaries` 로 `Episode/iker/success_5cm`, `Episode/iker/keypoint_distance_m`, `rewards/iter` 확인.
5. 반대쪽 받침 자리는 뱅크 파지 자세로 도달 불가(114 mm)다. VLM 이 그 자리를 고르면 G6 이 걸러내지 못한다. 계획 3 결정 사항이다.
6. 브랜치는 push·병합하지 않았다.

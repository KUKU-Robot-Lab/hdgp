# IKER 학습 파지 1단계·왼팔 전환 구현 계획 (계획 2b)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 신발을 옮기는 팔을 왼팔로 바꾸고, 왼손이 신발에 접근해 쥐고 5 cm 들어 2 s 유지하는 1단계 파지 정책 환경을 만들어 로컬 GPU 에서 학습을 띄운다.

**Architecture:**
- `tasks/iker_shoe/grasp_bank.py`: 좌우 일반화(`side_sign`·`hand_side`·`hand_joint_name`). K1 뱅크는 다시 만들지 않는다 — 커밋된 `grasp_bank.json` 은 계획 2c 까지 오른팔 K1 이고 뱅크 파일 테스트가 측면 불일치를 skip 으로 드러낸다.
- `tasks/iker_shoe/grasp_stage.py`(순수 torch): 손 full-joint 법칙, 신발 표면 최근접 거리 파지 품질 q, 받침 제외 자유 들기 높이, 1단계 보상 스텝(불변 상태).
- `tasks/iker_shoe/iker_shoe_grasp_env{,_cfg}.py`: 2단계 env 의 장면·IK·잡음을 그대로 쓰고 행동 26(손바닥 6D + 손 20)·관측 78·에피소드 120. gym id `open-sens_l_iker_shoe_grasp`.
- `scripts/iker/{make_pregrasp_bank,measure_grasp_quality,grasp_smoke}.py`: 접근 자세 뱅크, A 단계 체크포인트의 q 보정 파일, 시뮬레이터 검사.

**Tech Stack:** Isaac Sim 5.1 / Isaac Lab 0.50.5, rl_games PPO, torch, pytest(시스템 python3 3.10).

**Spec:** `docs/superpowers/specs/2026-09-14-iker-learned-grasp-left-arm-design.md`(§3 왼팔 전환, §4 1단계 환경, §5 q, §6 보상·종료, §9 게이트, §10 테스트, §12 reward-audit). 기준 스펙 `docs/superpowers/specs/2026-09-13-iker-vlm-keypoint-reward-design.md`.

**선행 조건:** 계획 1·2 가 브랜치 `iker-front-end` 에 완료(HEAD `e80c1518` 이후). 오른팔 사람 기준선 런 `iker_human_c00_k1`·`_r750` 체크포인트는 비교 기준으로 남긴다(삭제 금지).

**실행 위치:** 워크트리 `~/rl_ws/hdgp-iker`(브랜치 `iker-front-end`). 공유 체크아웃 `~/rl_ws/hdgp` 는 건드리지 않는다.

**계획 분할:** 이 계획(2b)은 설계 [0]·[1]: 왼팔 전환, 1단계 모듈·env, A 단계(g(q) 끔) 학습, q 보정, B 단계(g(q) 켬) 착수. 계획 2c(학습 파지 수확 → 2단계 IKER 학습 → 평가 스크립트)는 B 단계 결과를 보고 쓴다.

**검증 상태(2026-09-14):** 이 계획의 코드 블록은 스크래치 미러(`scratchpad/plan2b/hdgp`, 워크트리 source 복사 + assets 심링크)에서 실제로 돌린 파일을 스크립트(`assemble_plan.py`)로 옮겨 넣은 것이다. 새 파일은 전문, 수정 파일은 워크트리 대비 `git diff` 다.
- 순수 테스트(미러): `150 passed, 1 skipped`(skip = 오른팔 K1 뱅크 측면 불일치).
- 접근 자세 뱅크(미러, 왼팔): 3 라운드 320 개, 채택 83 %. 첫 두 시도의 실패가 조건을 바꿨다 — K1 손목 기울기는 펴진 손가락을 신발에 박았고(간격 −50 mm), 스폰 자세 신발은 5.3 mm 가라앉았고, 엄지는 손바닥보다 아래로 늘어져 "윗면 위" 판정이 틀렸다.
- 1단계 스모크(미러, 16 env): `GRASP SMOKE passed True`. 첫 두 시도가 잡은 것 — 리셋 때 신발이 몇 mm 튀어 무행동 lift_progress 9.2(→ 1 cm 불감대), 공중에 써 넣은 신발이 정책 한 스텝 동안 5 cm 떨어지고 `set_disable_gravities` 가 효과 없음(→ 물리 없이 `_get_dones` 로 배선 검사).
- 학습 스모크(미러, 4096 env, 3 epoch): fps total 7,436~7,755, `MAX EPOCHS NUM!`.
- 쓰지 않은 검증: 사람 기준선 재생성(Task 2)은 미러에서 돌리지 않았다(장면 불변 가정 — Task 2 가 1 mm 비교로 확인). A/B 단계 학습·q 보정 스크립트는 체크포인트가 필요해 실행 중에 처음 돈다.

## Global Constraints

- IKER 보상 5항(`modules/iker/reward.py`)·2단계 env 의 보상과 종료는 바꾸지 않는다(사용자 결정 "IKER 원형 유지").
- t2r(`modules/t2r`, `scripts/reward_gen`)과 grasp_s2r 을 import 하지 않는다.
- 로봇 값은 `modules/robot_profiles.py`·`modules/vendor_gains.py` 에서만 읽는다(숫자 복사 금지). 손 관절 이름은 프로필 소유 — 코드에 `r_hj_`/`l_hj_` 를 박지 않는다.
- GPU 는 로컬만 쓴다. Isaac 프로세스는 한 번에 하나(학습 착수 뒤에는 학습 하나).
- 순수 테스트: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/modules/iker/tests source/openarm/openarm/agnostic/tasks/iker_shoe/tests -q -p no:cacheprovider` (scipy·requests 경고 줄은 무시).
- Isaac 실행: `cd ~/rl_ws/hdgp-iker && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm ../IsaacLab/_isaac_sim/python.sh <스크립트> --headless`. `isaaclab.sh` 는 쓰지 않는다(공유 체크아웃 openarm 을 PYTHONPATH 앞에 넣는다).
- 10 분이 넘을 수 있는 Isaac 실행은 백그라운드로 띄우고 짧은 Bash 호출로 로그를 폴링한다. **실행 중에 턴을 끝내지 않는다.**
- Isaac 스크립트 성공 판정은 출력 표식(`BANK ... passed True`, `PREGRASP ... passed True`, `QUALITY ... passed True`, `GRASP SMOKE passed True`, `SMOKE passed True`, `MAX EPOCHS NUM!`)과 산출 파일로 한다.
- 공유 체크아웃 `~/rl_ws/hdgp` 에서는 어떤 명령도 실행하지 않는다.
- `git add -A`·`git add .` 금지. 각 커밋 단계의 경로만 add, push·병합 금지. 커밋은 따로 Bash 호출(훅이 `git commit` 과 ` -n` 을 한 호출에서 막는다).
- 커밋 메시지 끝 두 줄: `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` / `Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4`
- 학습 프로세스 종료는 PID 로만 한다(`pkill -f` 금지).
- 서브에이전트를 띄우지 않는다(구현자).

## 파일 구조

| 파일 | 역할 | 태스크 |
|---|---|---|
| `tasks/iker_shoe/grasp_bank.py` (수정) | 좌우 부호·손 측면·손 관절 이름, 파지 전 자세의 y 반사 | 1 |
| `tasks/iker_shoe/tests/test_grasp_bank.py` (수정) | 좌측 반사·측면 판독 테스트 2개 추가 | 1 |
| `tasks/iker_shoe/tests/test_grasp_bank_file.py` (수정) | 엄지 `_3` 이름·범위를 프로필 측면에서 유도, 뱅크 측면이 다르면 skip | 1 |
| `scripts/iker/make_grasp_bank.py` (수정) | 측면 부호로 파지 전 자세·엄지 사전굽힘, 메타데이터 `side_sign` | 1 |
| `tasks/iker_shoe/layout.py` (수정) | `PROFILE_NAME = "tesollo_left_short"` | 1 |
| `iker_runs/shoe_place/config_00/{snapshot_raw.png,snapshot.png,keypoints.json,interaction_human.json}` (재생성) | 왼팔 로봇 장면·사람 기준선 | 2 |
| `tasks/iker_shoe/grasp_stage.py` (신규) | 1단계 순수 함수 | 3 |
| `tasks/iker_shoe/tests/test_grasp_stage.py` (신규) | 20개 | 3 |
| `tasks/iker_shoe/iker_shoe_grasp_env_cfg.py` (신규) | 1단계 env cfg | 4 |
| `tasks/iker_shoe/iker_shoe_grasp_env.py` (신규) | 1단계 env | 4 |
| `tasks/iker_shoe/config/__init__.py` (수정) | `open-sens_l_iker_shoe_grasp{,-play}` 등록 | 4 |
| `tasks/iker_shoe/config/agents/rl_games_grasp_ppo_cfg.yaml` (신규) | 1단계 PPO | 4 |
| `scripts/iker/make_pregrasp_bank.py` (신규) | 접근 자세 뱅크 | 5 |
| `scripts/iker/measure_grasp_quality.py` (신규) | A 단계 체크포인트에서 q_lo·q_hi 보정 파일 | 5·6 |
| `scripts/iker/grasp_smoke.py` (신규) | 1단계 시뮬레이터 검사 | 5 |
| `iker_runs/shoe_place/config_00/pregrasp_bank.json` (생성) | 1단계 리셋 | 5 |
| `iker_runs/shoe_place/config_00/grasp_quality_calibration.json` (생성) | B 단계 g(q) 보정 | 6 |

(경로 접두 `source/openarm/openarm/agnostic/` 생략.)

---

### Task 1: 좌우 일반화와 왼팔 선택(순수)

**Files:**
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/grasp_bank.py`
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_bank.py`
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_bank_file.py`
- Modify: `scripts/iker/make_grasp_bank.py`
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/layout.py`

**Interfaces:**
- Produces: `gb.SIDE_SIGNS`, `gb.hand_side(joint_names) -> str`, `gb.side_sign(joint_names) -> float`, `gb.hand_joint_name(joint_names, finger, k) -> str`, `gb.palm_rotations(tilt_deg, yaw_deg, side_sign=1.0)`, `gb.palm_goal_positions(pregrasp, shoe_xy, shoe_yaw_deg, shoe_half_width, shoe_top_z, side_sign=1.0)`. `gb.finger_index` 는 섞인 측면 이름을 거부한다.

- [ ] **Step 1: 실패하는 테스트 추가** — 아래 diff 를 `tests/test_grasp_bank.py` 에 적용한다.

```diff
diff --git a/source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_bank.py b/source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_bank.py
index e4a1b07..2350e8b 100644
--- a/source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_bank.py
+++ b/source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_bank.py
@@ -159,6 +159,36 @@ def test_palm_goal_sits_beside_the_near_side_above_the_top():
     assert torch.allclose(goal, torch.tensor([0.25 + 0.068, 0.14 + 0.01, 0.339]), atol=1e-6)
 
 
+def test_left_arm_pregrasp_is_the_right_one_mirrored_in_y():
+    tilt, yaw = torch.tensor([-20.0, -35.0]), torch.tensor([5.0, -8.0])
+    right = gb.palm_rotations(tilt, yaw)
+    left = gb.palm_rotations(tilt, yaw, side_sign=-1.0)
+    mirror = torch.diag(torch.tensor([1.0, -1.0, 1.0]))
+    assert torch.allclose(left[:, :, 0], (mirror @ right[:, :, 0:1]).squeeze(-1), atol=1e-6)  # palmar side
+    assert torch.allclose(left[:, :, 2], (mirror @ right[:, :, 2:3]).squeeze(-1), atol=1e-6)  # fingers
+    assert torch.allclose(torch.linalg.det(left), torch.ones(2), atol=1e-6)
+    flat = gb.palm_rotations(torch.tensor([0.0]), torch.tensor([0.0]), side_sign=-1.0)[0]
+    assert torch.allclose(flat[:, 0], torch.tensor([0.0, 0.0, -1.0]))  # palmar side still down
+    assert torch.allclose(flat[:, 2], torch.tensor([0.0, -1.0, 0.0]))  # fingers toward -y, across the shoe from the left
+    pre = gb.PreGrasp(*(torch.tensor([v]) for v in (-20.0, 0.03, 0.02, 0.01, 0.0, 0.4)))
+    goal = gb.palm_goal_positions(pre, (0.25, 0.14), 0.0, 0.048, 0.309, side_sign=-1.0)[0]
+    assert torch.allclose(goal, torch.tensor([0.25 + 0.01, 0.14 + 0.068, 0.339]), atol=1e-6)
+    with pytest.raises(ValueError, match="side_sign"):
+        gb.palm_rotations(tilt, yaw, side_sign=0.5)
+
+
+def test_hand_side_reads_the_prefix_and_rejects_mixed_hands():
+    left = ["l_hj_thumb_3", "l_hj_index_2"]
+    assert gb.hand_side(left) == "l" and gb.side_sign(left) == -1.0
+    assert gb.side_sign(FINGER_JOINTS) == 1.0
+    assert gb.hand_joint_name(left, "thumb", 3) == "l_hj_thumb_3"
+    assert gb.finger_index(left).tolist() == [gb.FINGERS.index("thumb"), gb.FINGERS.index("index")]
+    with pytest.raises(ValueError, match="one side prefix"):
+        gb.finger_index(["l_hj_thumb_3", "r_hj_index_2"])
+    with pytest.raises(ValueError, match="not one of the hand joints"):
+        gb.hand_joint_name(left, "pinky", 4)
+
+
 def test_samples_stay_in_their_ranges():
     pre = gb.sample_pregrasp(500, torch.Generator().manual_seed(0))
     for value, bounds in ((pre.tilt_deg, gb.TILT_DEG_RANGE), (pre.thumb3, gb.THUMB3_RANGE), (pre.along_length, gb.ALONG_LENGTH_RANGE)):
```

- [ ] **Step 2: 실패 확인** — `PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_bank.py -q -p no:cacheprovider`. 기대: 새 두 테스트가 `TypeError`(side_sign 인자 없음)·`AttributeError`(`hand_side` 없음)로 실패.

- [ ] **Step 3: 구현** — 아래 diff 를 `grasp_bank.py` 에 적용한다.

```diff
diff --git a/source/openarm/openarm/agnostic/tasks/iker_shoe/grasp_bank.py b/source/openarm/openarm/agnostic/tasks/iker_shoe/grasp_bank.py
index 9930448..9a06ffc 100644
--- a/source/openarm/openarm/agnostic/tasks/iker_shoe/grasp_bank.py
+++ b/source/openarm/openarm/agnostic/tasks/iker_shoe/grasp_bank.py
@@ -61,38 +61,81 @@ def sample_pregrasp(count: int, generator: torch.Generator, device: str | torch.
     )
 
 
-def palm_rotations(tilt_deg: torch.Tensor, yaw_deg: torch.Tensor) -> torch.Tensor:
+SIDE_SIGNS = {"r": 1.0, "l": -1.0}  # the left arm is the right arm reflected through the XZ plane (y -> -y)
+
+
+def _check_side_sign(side_sign: float) -> None:
+    if side_sign not in (1.0, -1.0):
+        raise ValueError(f"side_sign must be +1 (right arm) or -1 (left arm), got {side_sign}")
+
+
+def palm_rotations(tilt_deg: torch.Tensor, yaw_deg: torch.Tensor, side_sign: float = 1.0) -> torch.Tensor:
     """(N, 3, 3) palm orientations: palmar side down, fingers across the shoe toward +y, tilted about world x
-    (negative tilt turns the palmar side toward -y) and yawed about world z."""
+    (negative tilt turns the palmar side toward -y) and yawed about world z.
+
+    ``side_sign`` -1 gives the left arm's orientation, the right one reflected through the XZ plane (S R S with
+    S = diag(1, -1, 1)): palmar side and fingers are mirrored in y and the matrix stays a proper rotation."""
+    _check_side_sign(side_sign)
     down = torch.tensor([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]], dtype=tilt_deg.dtype, device=tilt_deg.device)
     t, y = torch.deg2rad(tilt_deg), torch.deg2rad(yaw_deg)
     zeros, ones = torch.zeros_like(t), torch.ones_like(t)
     rx = torch.stack([ones, zeros, zeros, zeros, t.cos(), -t.sin(), zeros, t.sin(), t.cos()], dim=-1).view(-1, 3, 3)
     rz = torch.stack([y.cos(), -y.sin(), zeros, y.sin(), y.cos(), zeros, zeros, zeros, ones], dim=-1).view(-1, 3, 3)
-    return rz @ rx @ down
+    rotation = rz @ rx @ down
+    if side_sign > 0:
+        return rotation
+    mirror = torch.diag(torch.tensor([1.0, -1.0, 1.0], dtype=tilt_deg.dtype, device=tilt_deg.device))
+    return mirror @ rotation @ mirror
 
 
 def palm_goal_positions(
-    pregrasp: PreGrasp, shoe_xy: Sequence[float], shoe_yaw_deg: float, shoe_half_width: float, shoe_top_z: float
+    pregrasp: PreGrasp,
+    shoe_xy: Sequence[float],
+    shoe_yaw_deg: float,
+    shoe_half_width: float,
+    shoe_top_z: float,
+    side_sign: float = 1.0,
 ) -> torch.Tensor:
-    """(N, 3) palm origin goals: above the shoe top, offset toward the shoe's near (-y) side."""
+    """(N, 3) palm origin goals: above the shoe top, offset toward the shoe's side facing the arm (-y for the right
+    arm, +y for the left arm)."""
+    _check_side_sign(side_sign)
     yaw = math.radians(shoe_yaw_deg)
     along = pregrasp.along_length
-    lateral = -(shoe_half_width + pregrasp.near_side_clearance)
+    lateral = -side_sign * (shoe_half_width + pregrasp.near_side_clearance)
     x = shoe_xy[0] + along * math.cos(yaw) - lateral * math.sin(yaw)
     y = shoe_xy[1] + along * math.sin(yaw) + lateral * math.cos(yaw)
     return torch.stack([x, y, shoe_top_z + pregrasp.height_above_top], dim=-1)
 
 
+def hand_side(joint_names: Sequence[str]) -> str:
+    """'r' or 'l', the common prefix of hand joint names shaped ``<side>_hj_<finger>_<k>``; mixed sides are an error."""
+    sides = {name.split("_")[0] for name in joint_names}
+    if len(sides) != 1 or next(iter(sides)) not in SIDE_SIGNS:
+        raise ValueError(f"hand joint names must share one side prefix out of {sorted(SIDE_SIGNS)}, got {sorted(sides)}")
+    return next(iter(sides))
+
+
+def side_sign(joint_names: Sequence[str]) -> float:
+    return SIDE_SIGNS[hand_side(joint_names)]
+
+
+def hand_joint_name(joint_names: Sequence[str], finger: str, k: int) -> str:
+    name = f"{hand_side(joint_names)}_hj_{finger}_{k}"
+    if name not in joint_names:
+        raise ValueError(f"{name!r} is not one of the hand joints {list(joint_names)}")
+    return name
+
+
 def finger_index(joint_names: Sequence[str]) -> torch.Tensor:
-    """(J,) long ids into ``FINGERS``, parsed from hand joint names shaped ``r_hj_<finger>_<k>``."""
+    """(J,) long ids into ``FINGERS``, parsed from hand joint names shaped ``<side>_hj_<finger>_<k>``."""
     ids = []
     for name in joint_names:
         parts = name.split("_")
         finger = parts[-2] if len(parts) >= 2 else ""
         if finger not in FINGERS:
-            raise ValueError(f"joint name {name!r} does not parse as r_hj_<finger>_<k>")
+            raise ValueError(f"joint name {name!r} does not parse as <side>_hj_<finger>_<k>")
         ids.append(FINGERS.index(finger))
+    hand_side(joint_names)
     return torch.tensor(ids, dtype=torch.long)
 
 
```

- [ ] **Step 4: 파일 테스트·생성 스크립트·프로필 선택** — 아래 세 diff 를 적용한다. 왼손 엄지 `_3` 는 음(−) 방향으로 조인다(프로필 미러 규칙) — 사전굽힘 범위에 측면 부호를 곱한다.

```diff
diff --git a/source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_bank_file.py b/source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_bank_file.py
index e74055f..10fc7da 100644
--- a/source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_bank_file.py
+++ b/source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_bank_file.py
@@ -4,6 +4,7 @@ import xml.etree.ElementTree as ET
 from pathlib import Path
 from typing import Sequence
 
+import pytest
 import torch
 
 from openarm.agnostic.modules import robot_profiles
@@ -31,19 +32,33 @@ def _hand_joint_limits(urdf_path: Path, names: Sequence[str]) -> tuple[torch.Ten
     return lower, upper
 
 
+def _skip_if_bank_is_for_the_other_arm(doc, profile) -> None:
+    # Banks record the grasping arm's side since 2026-09-14; older banks were all built with the right arm.
+    bank_side = float(doc["metadata"].get("side_sign", 1.0))
+    if bank_side != gb.side_sign(profile.hand_joint_names):
+        pytest.skip(
+            f"grasp_bank.json was built for side_sign {bank_side} but {profile.name} grasps with the other arm -- "
+            "the right-arm K1 bank stays committed until plan 2c replaces it with the learned left-arm bank"
+        )
+
+
 def test_bank_entries_have_no_finger_at_an_opposite_or_beyond_limit():
     doc = run_files.read_json(layout.RUNS_DIR / "config_00" / "grasp_bank.json")
     profile = robot_profiles.PROFILES[layout.PROFILE_NAME]
+    _skip_if_bank_is_for_the_other_arm(doc, profile)
 
     urdf_path = layout.HDGP_ROOT / "assets" / Path(profile.usd_relpath).with_suffix(".urdf")
     assert urdf_path.is_file(), f"missing URDF for profile {profile.name!r}: {urdf_path}"
 
-    thumb3_idx = profile.hand_joint_names.index("r_hj_thumb_3")
+    thumb3_name = gb.hand_joint_name(profile.hand_joint_names, "thumb", 3)
+    thumb3_idx = profile.hand_joint_names.index(thumb3_name)
     grip_thumb3 = profile.hand_grip_pose[thumb3_idx]
-    lo, hi = gb.THUMB3_RANGE
+    # The left hand is the right one mirrored: its thumb_3 pre-curl range carries the side sign.
+    sign = gb.side_sign(profile.hand_joint_names)
+    lo, hi = sorted((sign * gb.THUMB3_RANGE[0], sign * gb.THUMB3_RANGE[1]))
     assert not (lo <= grip_thumb3 <= hi), (
-        f"NEEDS_CONTEXT: THUMB3_RANGE {gb.THUMB3_RANGE} straddles the grip value {grip_thumb3} "
-        "for r_hj_thumb_3 -- the start pose used here would not have an unambiguous closing direction"
+        f"NEEDS_CONTEXT: THUMB3_RANGE {(lo, hi)} straddles the grip value {grip_thumb3} "
+        f"for {thumb3_name} -- the start pose used here would not have an unambiguous closing direction"
     )
 
     lower, upper = _hand_joint_limits(urdf_path, profile.hand_joint_names)
```

```diff
diff --git a/scripts/iker/make_grasp_bank.py b/scripts/iker/make_grasp_bank.py
index 6a561ea..84bfb80 100644
--- a/scripts/iker/make_grasp_bank.py
+++ b/scripts/iker/make_grasp_bank.py
@@ -100,7 +100,8 @@ def main() -> int:
     upper = arm.data.soft_joint_pos_limits[0, hand_ids, 1]
     open_pose = torch.tensor(prof.hand_open_pose, device=dev)
     grip_pose = torch.tensor(prof.hand_grip_pose, device=dev)
-    thumb3 = list(prof.hand_joint_names).index("r_hj_thumb_3")
+    thumb3 = list(prof.hand_joint_names).index(gb.hand_joint_name(prof.hand_joint_names, "thumb", 3))
+    side_sign = gb.side_sign(prof.hand_joint_names)  # +1 right arm, -1 left arm (y-mirrored pre-grasp)
     finger_ids = gb.finger_index(prof.hand_joint_names).to(dev)
     ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), num_envs=n, device=dev)
     home = arm.data.default_joint_pos.clone()
@@ -153,10 +154,10 @@ def main() -> int:
         arm.set_joint_position_target(home)
         ik.reset()
         pre = gb.sample_pregrasp(n, generator, dev)
-        palm_quat = quat_from_matrix(gb.palm_rotations(pre.tilt_deg, pre.yaw_deg))
-        goal = gb.palm_goal_positions(pre, (config.move_x, config.move_y), config.move_yaw_deg, half_width, top_z)
+        palm_quat = quat_from_matrix(gb.palm_rotations(pre.tilt_deg, pre.yaw_deg, side_sign))
+        goal = gb.palm_goal_positions(pre, (config.move_x, config.move_y), config.move_yaw_deg, half_width, top_z, side_sign)
         start_hand = open_pose.expand(n, -1).clone()
-        start_hand[:, thumb3] = pre.thumb3
+        start_hand[:, thumb3] = side_sign * pre.thumb3
         start_hand = torch.max(torch.min(start_hand, upper), lower)
         above = goal + torch.tensor([0.0, 0.0, 0.12], device=dev)
         run(APPROACH_STEPS, above, palm_quat, start_hand, parked=True)
@@ -229,6 +230,7 @@ def main() -> int:
         "shoe_meta_sha256": hashlib.sha256(layout.SHOE_META_PATH.read_bytes()).hexdigest(),
         "seed": args.seed,
         "rounds": stats,
+        "side_sign": side_sign,
         "closing": {
             "rule": "finger_stop",
             "trigger_backbend_rad": gb.FINGER_TRIGGER_BACKBEND_RAD,
```

```diff
diff --git a/source/openarm/openarm/agnostic/tasks/iker_shoe/layout.py b/source/openarm/openarm/agnostic/tasks/iker_shoe/layout.py
index 51a55d0..36b0ed2 100644
--- a/source/openarm/openarm/agnostic/tasks/iker_shoe/layout.py
+++ b/source/openarm/openarm/agnostic/tasks/iker_shoe/layout.py
@@ -25,7 +25,7 @@ SHOE_META_PATH = SHOE_ASSET_DIR / "shoe_meta.json"
 TABLE_USD_PATH = ASSETS_DIR / "simulation_setting" / "env_v1" / "usd" / "env_v1.usda"
 RUNS_DIR = HDGP_ROOT / "iker_runs" / "shoe_place"
 
-PROFILE_NAME = "tesollo_right_short"
+PROFILE_NAME = "tesollo_left_short"
 TASK_INSTRUCTION = "Place the shoe on the rack next to the other shoe."
 
 TABLE_TOP_Z = 0.205
```

- [ ] **Step 5: 순수 테스트** — Global Constraints 의 순수 테스트 명령. 기대: `130 passed, 1 skipped` — skip 사유 `grasp_bank.json was built for side_sign 1.0 but tesollo_left_short grasps with the other arm`. 커밋된 뱅크는 계획 2c 까지 오른팔 K1 이다.

- [ ] **Step 6: 커밋**

```bash
cd /home/user/rl_ws/hdgp-iker && git add source/openarm/openarm/agnostic/tasks/iker_shoe/grasp_bank.py source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_bank.py source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_bank_file.py scripts/iker/make_grasp_bank.py source/openarm/openarm/agnostic/tasks/iker_shoe/layout.py
```

```bash
cd /home/user/rl_ws/hdgp-iker && git commit -F - <<'MSG'
feat(iker): 파지 뱅크 좌우 일반화(side_sign·hand_side)와 왼팔(tesollo_left_short) 선택

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4
MSG
```

---

### Task 2: 왼팔 장면·사람 기준선 재생성(Isaac)

**Files:**
- Regenerate: `iker_runs/shoe_place/config_00/snapshot_raw.png`, `snapshot.png`, `keypoints.json`, `interaction_human.json`

**Interfaces:**
- Consumes: Task 1 의 `PROFILE_NAME = "tesollo_left_short"`.
- Produces: 왼팔 로봇 장면의 `keypoints.json`(Task 5 접근 자세 뱅크가 신발 정착 자세를 읽는다)·`interaction_human.json`(G6 좌 손바닥 상자).

`grasp_bank.json` 은 다시 만들지 않는다(사용자 결정 — 파지는 학습한다). 계획 2c 가 학습 뱅크로 바꿀 때까지 오른팔 K1 뱅크이고, 뱅크 파일 테스트는 이유를 적고 skip 한다. 2단계 env 스모크(`env_smoke.py`)는 이 계획에서 돌리지 않는다.

- [ ] **Step 1: 비교용 사본** — `mkdir -p /tmp/claude-1000/-home-user-rl-ws/5ca9e4f2-e022-4266-b0ca-84d6cbf13515/scratchpad/right_short_config00 && cp iker_runs/shoe_place/config_00/* /tmp/claude-1000/-home-user-rl-ws/5ca9e4f2-e022-4266-b0ca-84d6cbf13515/scratchpad/right_short_config00/`

- [ ] **Step 2: 스냅샷** — `scripts/iker/snapshot.py --config-index 0` 를 Isaac 실행 규칙으로 돌리고 `grep -E "SNAPSHOT|Traceback|Error"`.
  - 기대: `SNAPSHOT config 00 ... passed True`, `SNAPSHOT CHECK FAILED` 없음.
  - 저장한 사본 `keypoints.json` 과 비교: 물체별 라벨, 모든 키포인트 world xyz 최대 절대 차이, 옮길 신발 `position`·`quat_wxyz`, `checks` 블록. 장면이 같으므로 키포인트 차이는 1 mm 미만이어야 한다. 넘으면 멈추고 보고한다.
  - `snapshot.png` 를 Read 로 보고 팔이 프레임 밖인지 적는다.

- [ ] **Step 3: 사람 기준선** — `cd /home/user/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 scripts/iker/human_baseline.py --run-dir iker_runs/shoe_place/config_00 2>&1 | grep -v -i warn; echo "exit=$?"`
  - 기대: `gate passed=True failures=[]`, exit 0. 사본의 `target_keypoints` 와 최대 차이를 보고한다(1 mm 미만이어야 한다). G6 가 좌 프로필 손바닥 상자에서 실패하면 멈추고 BLOCKED 로 보고한다.

- [ ] **Step 4: 순수 테스트** — 기대 `130 passed, 1 skipped`(skip 사유: 오른팔 K1 뱅크).

- [ ] **Step 5: 커밋**

```bash
cd /home/user/rl_ws/hdgp-iker && git add iker_runs/shoe_place/config_00/snapshot_raw.png iker_runs/shoe_place/config_00/snapshot.png iker_runs/shoe_place/config_00/keypoints.json iker_runs/shoe_place/config_00/interaction_human.json
```

```bash
cd /home/user/rl_ws/hdgp-iker && git commit -F - <<'MSG'
feat(iker): 왼팔로 구성 0 스냅샷·사람 기준선 재생성

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4
MSG
```

---

### Task 3: 1단계 순수 모듈 `grasp_stage.py`

**Files:**
- Create: `source/openarm/openarm/agnostic/tasks/iker_shoe/grasp_stage.py`
- Test: `source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_stage.py`

**Interfaces:**
- Produces: `gs.HAND_EMA_ALPHA`, `gs.Stage1RewardCfg`(hydra 호환 비-frozen, `__post_init__` 검증), `gs.REWARD_TERMS`, `gs.hand_action_limits(names, hard_lo, hard_hi, override) -> (lo, hi)`, `gs.hand_targets(action, lo, hi, previous, alpha)`, `gs.normalized_targets(targets, lo, hi)`, `gs.surface_subsample(points, count=256)`, `gs.nearest_distance(query, surface)`, `gs.grasp_quality(link_pos, finger_sizes, surface, palm_pos, palm_normal, shoe_center) -> (q, w_f, palm_cos)`, `gs.g_factor(q, cfg)`, `gs.free_lift_height(surface, start_bottom_z, rack_x, rack_y, margin)`, `gs.Stage1State.start(n, device)`·`.reset_rows(env_ids)`, `gs.Stage1Step`, `gs.stage1_step(state, cfg, *, palm_gap, dz_free, palm_shoe_dist, rel_speed, shoe_speed, q, hand_floor_depth, arm_speed_sum, hand_speed_sum) -> Stage1Step`.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
"""Stage-1 grasp policy pure functions (design 2026-09-14 §4-§6, §10)."""

import math

import pytest
import torch

from openarm.agnostic.tasks.iker_shoe import grasp_stage as gs

NAMES = ("l_hj_thumb_3", "l_hj_index_3", "l_hj_index_4", "l_hj_index_1")
CFG = gs.Stage1RewardCfg(q_lo=0.2, q_hi=0.6)


def _step(state, **overrides):
    n = state.closest_palm.shape[0]
    inputs = dict(
        palm_gap=torch.full((n,), 0.10),
        dz_free=torch.zeros(n),
        palm_shoe_dist=torch.full((n,), 0.30),
        rel_speed=torch.zeros(n),
        shoe_speed=torch.zeros(n),
        q=torch.zeros(n),
        hand_floor_depth=torch.zeros(n),
        arm_speed_sum=torch.zeros(n),
        hand_speed_sum=torch.zeros(n),
    )
    inputs.update(overrides)
    return gs.stage1_step(state, CFG, **inputs)


def _held(n):
    return dict(dz_free=torch.full((n,), 0.06), palm_shoe_dist=torch.full((n,), 0.12), rel_speed=torch.zeros(n))


def test_hand_ema_alpha_matches_the_60hz_time_constant():
    assert gs.HAND_EMA_ALPHA == pytest.approx(1.0 - 0.9**6)
    assert (1.0 - gs.HAND_EMA_ALPHA) == pytest.approx(0.9**6)


def test_hand_action_limits_narrow_only_and_reject_unmatched_or_empty():
    lo, hi = torch.full((4,), -1.571), torch.full((4,), 1.571)
    out_lo, out_hi = gs.hand_action_limits(NAMES, lo, hi, {r"l_hj_index_[34]$": (0.0, None), r"l_hj_thumb_[34]$": (None, 0.0)})
    assert out_lo.tolist() == pytest.approx([-1.571, 0.0, 0.0, -1.571])
    assert out_hi.tolist() == pytest.approx([0.0, 1.571, 1.571, 1.571])
    widened, _ = gs.hand_action_limits(NAMES, lo, hi, {r"l_hj_index_1$": (-3.0, None)})
    assert widened[3] == pytest.approx(-1.571)
    with pytest.raises(ValueError, match="matches no joint"):
        gs.hand_action_limits(NAMES, lo, hi, {r"r_hj_index_[34]$": (0.0, None)})
    with pytest.raises(ValueError, match="zero width"):
        gs.hand_action_limits(NAMES, lo, hi, {r"l_hj_index_1$": (0.5, 0.5)})


def test_hand_targets_map_linearly_filter_and_clamp():
    lo, hi = torch.tensor([0.0, -1.0]), torch.tensor([2.0, 1.0])
    previous = torch.tensor([[1.0, 0.0]])
    full_close = gs.hand_targets(torch.tensor([[1.0, -1.0]]), lo, hi, previous, alpha=1.0)
    assert torch.allclose(full_close, torch.tensor([[2.0, -1.0]]))
    filtered = gs.hand_targets(torch.tensor([[1.0, 1.0]]), lo, hi, previous, alpha=0.5)
    assert torch.allclose(filtered, torch.tensor([[1.5, 0.5]]))
    beyond = gs.hand_targets(torch.tensor([[5.0, -5.0]]), lo, hi, torch.tensor([[3.0, -3.0]]), alpha=0.1)
    assert torch.allclose(beyond, torch.tensor([[2.0, -1.0]]))
    with pytest.raises(ValueError):
        gs.hand_targets(torch.zeros(1, 2), lo, hi, previous, alpha=0.0)


def test_normalized_targets_span_minus_one_to_one():
    lo, hi = torch.tensor([0.0, -1.0]), torch.tensor([2.0, 1.0])
    assert torch.allclose(gs.normalized_targets(torch.tensor([[0.0, 1.0]]), lo, hi), torch.tensor([[-1.0, 1.0]]))
    assert torch.allclose(gs.normalized_targets(torch.tensor([[1.0, 0.0]]), lo, hi), torch.tensor([[0.0, 0.0]]))


def test_surface_subsample_is_deterministic_and_keeps_both_ends():
    points = [[float(i), 0.0, 0.0] for i in range(1000)]
    first, second = gs.surface_subsample(points, 256), gs.surface_subsample(points, 256)
    assert first.shape == (256, 3) and torch.equal(first, second)
    assert first[0, 0] == 0.0 and first[-1, 0] == 999.0
    with pytest.raises(ValueError):
        gs.surface_subsample(points[:10], 256)


def _quality(link_pos, palm_normal=(0.0, 0.0, -1.0)):
    surface = torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]])
    return gs.grasp_quality(
        link_pos=link_pos,
        finger_sizes=(1, 1),
        surface=surface,
        palm_pos=torch.tensor([[0.05, 0.0, 0.1]]),
        palm_normal=torch.tensor([palm_normal]),
        shoe_center=torch.tensor([[0.05, 0.0, 0.0]]),
    )


def test_grasp_quality_is_a_soft_min_and_the_weakest_finger_dominates():
    touching, _, _ = _quality(torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]]))
    assert touching.item() == pytest.approx(1.0)
    one_far, w_f, _ = _quality(torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.2]]]))
    assert w_f[0, 1].item() == pytest.approx(math.exp(-0.2 / gs.KERNEL_TAU_M), abs=1e-6)
    assert w_f.min().item() - 1e-6 <= one_far.item() <= w_f.mean().item() + 1e-6
    assert one_far.item() < 0.2


def test_grasp_quality_is_zero_when_the_palm_faces_away():
    away, w_f, palm_cos = _quality(torch.tensor([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]]), palm_normal=(0.0, 0.0, 1.0))
    assert palm_cos.item() < 0.0 and away.item() == 0.0 and w_f.min().item() == pytest.approx(1.0)


def test_g_factor_runs_from_g_min_to_one_between_q_lo_and_q_hi():
    g = gs.g_factor(torch.tensor([0.0, 0.2, 0.4, 0.6, 1.0]), CFG)
    assert g.tolist() == pytest.approx([0.5, 0.5, 0.75, 1.0, 1.0])


def test_free_lift_height_is_zero_while_the_hull_touches_the_widened_rack_footprint():
    start = torch.tensor([0.20, 0.20, 0.20])
    lifted = torch.tensor([[[0.30, 0.10, 0.28], [0.30, 0.20, 0.30]]])
    beside = torch.tensor([[[0.30, 0.045, 0.28], [0.30, 0.20, 0.30]]])
    touching = torch.tensor([[[0.30, 0.035, 0.28], [0.30, 0.20, 0.30]]])
    surfaces = torch.cat([lifted, beside, touching])
    dz = gs.free_lift_height(surfaces, start, rack_x=(0.11, 0.43), rack_y=(-0.33, 0.03))
    assert dz.tolist() == pytest.approx([0.08, 0.08, 0.0])


def test_idle_policy_earns_nothing():
    state = gs.Stage1State.start(3)
    total = torch.zeros(3)
    for _ in range(120):
        step = _step(state)
        total, state = total + step.reward, step.state
    assert torch.equal(total, torch.zeros(3))


def test_progress_terms_pay_only_new_bests_and_never_for_retreat():
    state = gs.Stage1State.start(1)
    rewards = []
    for gap, dz in ((0.10, 0.0), (0.08, 0.02), (0.09, 0.015), (0.06, 0.03)):
        step = _step(state, palm_gap=torch.tensor([gap]), dz_free=torch.tensor([dz]))
        rewards.append((step.terms["palm_progress"].item(), step.terms["lift_progress"].item()))
        state = step.state
    assert rewards[0] == pytest.approx((0.0, 0.0))
    assert rewards[1] == pytest.approx((50 * 0.02, 2000 * 0.01))  # 2 cm rise, 1 cm of it past the dead band
    assert rewards[2] == pytest.approx((0.0, 0.0))
    assert rewards[3] == pytest.approx((50 * 0.02, 2000 * 0.01))


def test_a_reset_bounce_inside_the_dead_band_pays_nothing():
    state = gs.Stage1State.start(1)
    total = 0.0
    for dz in (0.0, 0.004, 0.009, 0.002, 0.0):
        step = _step(state, dz_free=torch.tensor([dz]))
        total, state = total + step.reward.item(), step.state
    assert total == 0.0
    lifted = _step(state, dz_free=torch.tensor([0.05]))
    assert lifted.terms["lift_progress"].item() == pytest.approx(2000 * (0.05 - CFG.lift_deadband_m))


def test_lift_bonus_needs_three_consecutive_held_steps_and_pays_once():
    state = gs.Stage1State.start(1)
    paid = []
    for held in (True, True, False, True, True, True, True):
        step = _step(state, **(_held(1) if held else {}), q=torch.tensor([0.6]))
        paid.append(step.terms["lift_bonus"].item())
        state = step.state
    assert paid == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 300.0, 0.0])


def test_a_fling_without_the_palm_near_or_moving_with_the_shoe_is_not_held():
    state = gs.Stage1State.start(2)
    for _ in range(5):
        step = _step(state, dz_free=torch.full((2,), 0.08), palm_shoe_dist=torch.tensor([0.30, 0.10]),
                     rel_speed=torch.tensor([0.0, 0.5]))
        assert not step.held.any() and step.terms["lift_bonus"].sum() == 0.0
        state = step.state


def test_success_needs_twenty_consecutive_held_steps_resets_on_a_break_and_pays_once():
    state = gs.Stage1State.start(1)
    success_steps = []
    pattern = [True] * 10 + [False] + [True] * 25
    for index, held in enumerate(pattern):
        step = _step(state, **(_held(1) if held else {}), q=torch.tensor([0.0]))
        if step.success.item():
            success_steps.append(index)
            assert step.terms["success_bonus"].item() == pytest.approx(500.0)
        state = step.state
    assert success_steps == [30]


def test_success_waits_for_the_shoe_to_be_slow():
    state = gs.Stage1State.start(1)
    for _ in range(25):
        step = _step(state, **_held(1), shoe_speed=torch.tensor([0.2]))
        assert not step.success.any()
        state = step.state


def test_progress_stops_after_the_latch():
    state = gs.Stage1State.start(1)
    for _ in range(3):
        state = _step(state, **_held(1)).state
    assert state.latched.item()
    after = _step(state, palm_gap=torch.tensor([0.0]), dz_free=torch.tensor([0.06]), palm_shoe_dist=torch.tensor([0.12]))
    assert after.terms["palm_progress"].item() == 0.0 and after.terms["lift_progress"].item() == 0.0


def test_hand_floor_penalty_is_proportional_and_capped():
    step = _step(gs.Stage1State.start(3), hand_floor_depth=torch.tensor([0.0, 0.1, 2.0]))
    assert step.terms["hand_floor"].tolist() == pytest.approx([0.0, -1.0, -5.0])


def test_reset_rows_restores_only_the_given_envs():
    state = gs.Stage1State.start(2)
    for _ in range(3):
        state = _step(state, **_held(2)).state
    reset = state.reset_rows(torch.tensor([1]))
    assert reset.latched.tolist() == [True, False]
    assert reset.hold_count.tolist() == [3, 0]
    assert reset.closest_palm.tolist() == pytest.approx([0.10, -1.0])
    assert state.latched.tolist() == [True, True]


def test_reward_cfg_rejects_inconsistent_values():
    with pytest.raises(ValueError):
        gs.Stage1RewardCfg(q_lo=0.5, q_hi=0.5)
    with pytest.raises(ValueError):
        gs.Stage1RewardCfg(g_min=0.0)
    with pytest.raises(ValueError):
        gs.Stage1RewardCfg(latch_steps=25, success_steps=20)
```

- [ ] **Step 2: 실패 확인** — `PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_stage.py -q -p no:cacheprovider`. 기대: `ImportError`(grasp_stage 없음).

- [ ] **Step 3: 구현**

```python
"""Stage-1 grasp policy of the IKER shoe task: hand action law, grasp quality and reward (design 2026-09-14 §4-§6).

Pure torch, importable without Isaac Sim. The environment computes the geometry (link positions, the shoe's surface
points, speeds) and feeds it here; every function returns new tensors and ``Stage1State`` is immutable, so the
environment owns the only copy of the episode state.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import torch

HAND_EMA_ALPHA = 1.0 - 0.9**6  # grasp_fj's 0.1 per 60 Hz step, same time constant at 10 Hz (= 0.468559)
SURFACE_POINT_COUNT = 256
KERNEL_TAU_M = 0.02
SOFTMIN_TAU = 0.1
PALM_COS_MIN = 0.0
RACK_MARGIN_M = 0.01  # a shoe hull point this close to the rack's footprint counts as supported by the rack


@dataclass
class Stage1RewardCfg:
    """Design §6. ``q_lo``/``q_hi`` are measured on the shoe before training (design §5).

    Not frozen: Isaac Lab's hydra round trip assigns every nested config field with ``setattr`` and a frozen
    dataclass raises there (the same reason as ``IkerRewardCfg``). The environment calls ``validate`` after the
    round trip; ``stage1_step`` never mutates it.
    """

    palm_scale: float = 50.0
    lift_progress_scale: float = 2000.0
    # The shoe's surface points sit ~4 mm below the table in its settled pose and restitution is randomised up to 1, so
    # a reset bounces the shoe a few mm: lift progress starts only past this dead band (grasp smoke, 2026-09-14).
    lift_deadband_m: float = 0.01
    lift_height_m: float = 0.05
    lift_bonus: float = 300.0
    success_bonus: float = 1000.0
    hold_radius_m: float = 0.15
    hold_rel_speed: float = 0.05
    latch_steps: int = 3
    success_steps: int = 20
    success_speed: float = 0.05
    hand_floor_scale: float = 10.0
    hand_floor_cap: float = 5.0
    arm_vel_scale: float = 0.0
    hand_vel_scale: float = 0.0
    g_min: float = 0.5
    q_lo: float = 0.13
    q_hi: float = 0.44

    def __post_init__(self):
        if not 0.0 < self.g_min <= 1.0:
            raise ValueError(f"g_min must be in (0, 1], got {self.g_min}")
        if not self.q_lo < self.q_hi:
            raise ValueError(f"q_lo {self.q_lo} must be < q_hi {self.q_hi}")
        if not 1 <= self.latch_steps <= self.success_steps:
            raise ValueError(f"need 1 <= latch_steps {self.latch_steps} <= success_steps {self.success_steps}")
        if min(self.palm_scale, self.lift_progress_scale, self.hand_floor_scale, self.hand_floor_cap,
               self.arm_vel_scale, self.hand_vel_scale) < 0.0:
            raise ValueError("reward scales must be non-negative")


REWARD_TERMS = ("palm_progress", "lift_progress", "lift_bonus", "success_bonus", "hand_floor", "arm_vel", "hand_vel")


def hand_action_limits(
    joint_names: Sequence[str], hard_lo: torch.Tensor, hard_hi: torch.Tensor, override: Mapping[str, tuple]
) -> tuple[torch.Tensor, torch.Tensor]:
    """(lo, hi) per hand joint: the URDF hard limits narrowed by the profile override (regex -> (lo|None, hi|None)).

    An override can only narrow a range. A regex that matches no joint and a range narrowed to zero width are
    errors, so a typo cannot silently leave a joint at its full (back-bending) range.
    """
    names = list(joint_names)
    if hard_lo.shape != (len(names),) or hard_hi.shape != (len(names),):
        raise ValueError(f"limits must be ({len(names)},), got {tuple(hard_lo.shape)} and {tuple(hard_hi.shape)}")
    lo, hi = hard_lo.clone(), hard_hi.clone()
    for pattern, (olo, ohi) in override.items():
        regex = re.compile(pattern)
        mask = torch.tensor([regex.fullmatch(n) is not None for n in names], device=lo.device)
        if not bool(mask.any()):
            raise ValueError(f"hand limit override {pattern!r} matches no joint of {names}")
        if olo is not None:
            lo = torch.where(mask, torch.maximum(lo, torch.full_like(lo, float(olo))), lo)
        if ohi is not None:
            hi = torch.where(mask, torch.minimum(hi, torch.full_like(hi, float(ohi))), hi)
    dead = [n for n, width in zip(names, (hi - lo).tolist()) if width <= 1e-6]
    if dead:
        raise ValueError(f"hand action range has zero width for {dead}")
    return lo, hi


def hand_targets(
    action: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, previous: torch.Tensor, alpha: float = HAND_EMA_ALPHA
) -> torch.Tensor:
    """One step of the full-joint hand law: raw = lo + (a+1)/2 (hi-lo), EMA toward raw, clamp to [lo, hi]."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    raw = lo + 0.5 * (action.clamp(-1.0, 1.0) + 1.0) * (hi - lo)
    return torch.max(torch.min(alpha * raw + (1.0 - alpha) * previous, hi), lo)


def normalized_targets(targets: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Post-EMA hand targets mapped to [-1, 1], the observation of the policy's own filter state."""
    return 2.0 * (targets - lo) / (hi - lo) - 1.0


def surface_subsample(points: Sequence[Sequence[float]], count: int = SURFACE_POINT_COUNT) -> torch.Tensor:
    """(count, 3) evenly spaced rows of the shoe's surface point list — deterministic, no RNG."""
    table = torch.as_tensor(points, dtype=torch.float32)
    if table.dim() != 2 or table.shape[1] != 3 or table.shape[0] < count:
        raise ValueError(f"need at least {count} points of shape (P, 3), got {tuple(table.shape)}")
    index = torch.linspace(0, table.shape[0] - 1, count).round().long()
    return table[index]


def nearest_distance(query: torch.Tensor, surface: torch.Tensor) -> torch.Tensor:
    """(N, L) distance from each query point (N, L, 3) to its nearest surface point (N, P, 3)."""
    return torch.cdist(query, surface).min(dim=-1).values


def grasp_quality(
    link_pos: torch.Tensor,
    finger_sizes: tuple[int, ...],
    surface: torch.Tensor,
    palm_pos: torch.Tensor,
    palm_normal: torch.Tensor,
    shoe_center: torch.Tensor,
    tau: float = KERNEL_TAU_M,
    tau_q: float = SOFTMIN_TAU,
    palm_cos_min: float = PALM_COS_MIN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(q (N,), w_f (N, F), palm_cos (N,)).

    k_l = exp(-d_l / tau) with d_l the nearest-surface distance of finger link l; w_f = mean over the finger's links;
    q = soft-min over fingers (min <= q <= mean), zero unless the palm normal points at the shoe centre.
    """
    sizes = tuple(int(s) for s in finger_sizes)
    if not sizes or min(sizes) < 1 or sum(sizes) != link_pos.shape[1]:
        raise ValueError(f"finger_sizes {sizes} must be positive and sum to {link_pos.shape[1]} links")
    if min(tau, tau_q) <= 0.0:
        raise ValueError("tau and tau_q must be positive")
    kernel = torch.exp(-nearest_distance(link_pos, surface) / tau)
    w_f = torch.stack([part.mean(dim=1) for part in torch.split(kernel, list(sizes), dim=1)], dim=1)
    q = -tau_q * (torch.logsumexp(-w_f / tau_q, dim=1) - math.log(len(sizes)))
    q = torch.minimum(torch.maximum(q, w_f.min(dim=1).values), w_f.mean(dim=1))
    toward = shoe_center - palm_pos
    palm_cos = (palm_normal * toward).sum(dim=-1) / (palm_normal.norm(dim=-1) * toward.norm(dim=-1)).clamp(min=1e-9)
    return q * (palm_cos > palm_cos_min).float(), w_f, palm_cos


def g_factor(q: torch.Tensor, cfg: Stage1RewardCfg) -> torch.Tensor:
    """Grasp factor on the one-shot bonuses: g_min for a flat grasp rising linearly to 1 between q_lo and q_hi."""
    ramp = ((q - cfg.q_lo) / (cfg.q_hi - cfg.q_lo)).clamp(0.0, 1.0)
    return cfg.g_min + (1.0 - cfg.g_min) * ramp


def free_lift_height(
    surface: torch.Tensor,
    start_bottom_z: torch.Tensor,
    rack_x: tuple[float, float],
    rack_y: tuple[float, float],
    margin: float = RACK_MARGIN_M,
) -> torch.Tensor:
    """(N,) rise of the shoe's lowest surface point since the episode start, or 0 while any surface point lies over the
    rack footprint widened by ``margin`` — height gained by pushing the shoe onto or against the rack is not a lift."""
    x, y = surface[..., 0], surface[..., 1]
    over_rack = ((x >= rack_x[0] - margin) & (x <= rack_x[1] + margin) & (y >= rack_y[0] - margin) & (y <= rack_y[1] + margin)).any(dim=-1)
    rise = surface[..., 2].min(dim=-1).values - start_bottom_z
    return torch.where(over_rack, torch.zeros_like(rise), rise)


@dataclass(frozen=True)
class Stage1State:
    closest_palm: torch.Tensor  # (N,) best palm-to-surface gap so far, -1 before the first step
    best_lift: torch.Tensor  # (N,) best clamp(dz_free, 0, lift_height) so far
    hold_count: torch.Tensor  # (N,) consecutive held steps
    latched: torch.Tensor  # (N,) bool, the lift bonus has been paid
    succeeded: torch.Tensor  # (N,) bool, the success bonus has been paid

    @classmethod
    def start(cls, num_envs: int, device: str | torch.device = "cpu") -> "Stage1State":
        return cls(
            closest_palm=torch.full((num_envs,), -1.0, device=device),
            best_lift=torch.zeros(num_envs, device=device),
            hold_count=torch.zeros(num_envs, dtype=torch.long, device=device),
            latched=torch.zeros(num_envs, dtype=torch.bool, device=device),
            succeeded=torch.zeros(num_envs, dtype=torch.bool, device=device),
        )

    def reset_rows(self, env_ids: torch.Tensor) -> "Stage1State":
        """A new state with the rows of ``env_ids`` back at their start values."""
        fresh = Stage1State.start(self.closest_palm.shape[0], self.closest_palm.device)

        def mix(old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
            out = old.clone()
            out[env_ids] = new[env_ids]
            return out

        return Stage1State(*(mix(getattr(self, f), getattr(fresh, f)) for f in ("closest_palm", "best_lift", "hold_count", "latched", "succeeded")))


@dataclass(frozen=True)
class Stage1Step:
    reward: torch.Tensor
    terms: dict[str, torch.Tensor]
    state: Stage1State
    held: torch.Tensor
    just_latched: torch.Tensor
    success: torch.Tensor


def stage1_step(
    state: Stage1State,
    cfg: Stage1RewardCfg,
    *,
    palm_gap: torch.Tensor,
    dz_free: torch.Tensor,
    palm_shoe_dist: torch.Tensor,
    rel_speed: torch.Tensor,
    shoe_speed: torch.Tensor,
    q: torch.Tensor,
    hand_floor_depth: torch.Tensor,
    arm_speed_sum: torch.Tensor,
    hand_speed_sum: torch.Tensor,
) -> Stage1Step:
    """One policy step of the stage-1 reward (design §6). All inputs are (N,) tensors."""
    n = state.closest_palm.shape[0]
    for name, value in dict(palm_gap=palm_gap, dz_free=dz_free, palm_shoe_dist=palm_shoe_dist, rel_speed=rel_speed,
                            shoe_speed=shoe_speed, q=q, hand_floor_depth=hand_floor_depth,
                            arm_speed_sum=arm_speed_sum, hand_speed_sum=hand_speed_sum).items():
        if value.shape != (n,):
            raise ValueError(f"{name} must be ({n},), got {tuple(value.shape)}")

    held = (dz_free >= cfg.lift_height_m) & (palm_shoe_dist <= cfg.hold_radius_m) & (rel_speed < cfg.hold_rel_speed)
    hold_count = torch.where(held, state.hold_count + 1, torch.zeros_like(state.hold_count))
    before_latch = (~state.latched).float()
    just_latched = ~state.latched & (hold_count >= cfg.latch_steps)
    latched = state.latched | just_latched
    success = ~state.succeeded & latched & (hold_count >= cfg.success_steps) & (shoe_speed < cfg.success_speed)

    first = state.closest_palm < 0.0
    palm_delta = torch.where(first, torch.zeros_like(palm_gap), (state.closest_palm - palm_gap).clamp(min=0.0))
    closest_palm = torch.where(first, palm_gap, torch.minimum(state.closest_palm, palm_gap))
    if not 0.0 <= cfg.lift_deadband_m < cfg.lift_height_m:
        raise ValueError(f"need 0 <= lift_deadband_m {cfg.lift_deadband_m} < lift_height_m {cfg.lift_height_m}")
    lift_level = (dz_free - cfg.lift_deadband_m).clamp(0.0, cfg.lift_height_m - cfg.lift_deadband_m)
    lift_delta = (lift_level - state.best_lift).clamp(min=0.0)
    best_lift = torch.maximum(state.best_lift, lift_level)
    g = g_factor(q, cfg)

    terms = {
        "palm_progress": cfg.palm_scale * palm_delta * before_latch,
        "lift_progress": cfg.lift_progress_scale * lift_delta * before_latch,
        "lift_bonus": cfg.lift_bonus * g * just_latched.float(),
        "success_bonus": cfg.success_bonus * g * success.float(),
        "hand_floor": -(cfg.hand_floor_scale * hand_floor_depth.clamp(min=0.0)).clamp(max=cfg.hand_floor_cap),
        "arm_vel": -cfg.arm_vel_scale * arm_speed_sum,
        "hand_vel": -cfg.hand_vel_scale * hand_speed_sum,
    }
    if tuple(terms) != REWARD_TERMS:
        raise RuntimeError(f"term order drifted: {tuple(terms)}")
    reward = torch.nan_to_num(torch.stack(list(terms.values())).sum(dim=0), nan=0.0, posinf=0.0, neginf=0.0)
    new_state = replace(state, closest_palm=closest_palm, best_lift=best_lift, hold_count=hold_count,
                        latched=latched, succeeded=state.succeeded | success)
    return Stage1Step(reward=reward, terms=terms, state=new_state, held=held, just_latched=just_latched, success=success)
```

- [ ] **Step 4: 통과 확인** — 새 파일 20 passed, 순수 테스트 전체 `150 passed, 1 skipped`.

- [ ] **Step 5: 커밋**

```bash
cd /home/user/rl_ws/hdgp-iker && git add source/openarm/openarm/agnostic/tasks/iker_shoe/grasp_stage.py source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_stage.py
```

```bash
cd /home/user/rl_ws/hdgp-iker && git commit -F - <<'MSG'
feat(iker): 1단계 파지 순수 모듈 — 손 full-joint 법칙·표면 최근접 파지 품질 q·받침 제외 자유 들기(1 cm 불감대)·진행형/1회성 보상

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4
MSG
```

---

### Task 4: 1단계 환경·cfg·gym 등록·PPO 설정

**Files:**
- Create: `source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_grasp_env_cfg.py`
- Create: `source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_grasp_env.py`
- Create: `source/openarm/openarm/agnostic/tasks/iker_shoe/config/agents/rl_games_grasp_ppo_cfg.yaml`
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/config/__init__.py`

**Interfaces:**
- Consumes: Task 3 전부, 2단계 `IkerShoeEnv._setup_scene/_keypoints_local/_noisy_quat`, `IkerShoeEnvCfg`, `modules/object_wrench.WrenchDR`.
- Produces: gym id `open-sens_l_iker_shoe_grasp`(+`-play`), `IkerShoeGraspEnv.grasp_quality_now() -> (q, w_f, palm_cos)`, `IkerShoeGraspEnv._shoe_surface()`, `_last: Stage1Step`, `_stage: Stage1State`, 모듈 상수 `PREGRASP_BANK_FILE`·`QUALITY_CALIBRATION_FILE`.

부팅 동작(스펙 §4·§6·관찰 대조 13·17행): 손 20관절 행동 한계 표 출력, 열린 손이 한계 밖으로 0.6 rad 넘게 벗어나면 오류, `pregrasp_bank.json` 부팅 대조(`robot_usd`·게인·`scene_config`·`shoe_meta_sha256`), `grasp_reward.g_min < 1` 인데 `grasp_quality_calibration.json` 이 없으면 오류, 무행동 스텝 수입이 0 이 아니면 오류. cfg 기본값은 A 단계(`g_min 1.0`).

- [ ] **Step 1: cfg**

```python
"""Isaac Lab configuration of the IKER shoe stage-1 grasp environment (design 2026-09-14 §4-§6)."""

from __future__ import annotations

from isaaclab.utils import configclass

from .grasp_stage import Stage1RewardCfg
from .iker_shoe_env_cfg import CONTROL_DT, IkerShoeEnvCfg

GRASP_EPISODE_STEPS = 120  # 12 s at 10 Hz: approach from 8-12 cm, close, lift and hold take 5-8 s
ARM_ACTION_DIM = 6
HAND_ACTION_DIM = 20


@configclass
class IkerShoeGraspEnvCfg(IkerShoeEnvCfg):
    """Same scene, physics, noise and startup randomisation as the stage-2 environment; a different task."""

    episode_length_s = GRASP_EPISODE_STEPS * CONTROL_DT
    action_space = ARM_ACTION_DIM + HAND_ACTION_DIM
    observation_space = 38 + 2 * HAND_ACTION_DIM

    # Phase A (user decision 2026-09-14): train with the grasp factor off (g_min 1.0), measure q at the latch and
    # success moments of a checkpoint that lifts (scripts/iker/measure_grasp_quality.py), then resume with
    # `env.grasp_reward.g_min=0.5`. Any g_min < 1 requires the calibration file.
    grasp_reward: Stage1RewardCfg = Stage1RewardCfg(g_min=1.0)

    # empty paths resolve to iker_runs/shoe_place/config_XX/
    pregrasp_bank_path = ""
    quality_calibration_path = ""

    start_noise_xy = 0.02  # legacy IKER interact-object position noise
    drop_z = 0.10  # env-local shoe height below which the shoe has fallen off the table
    hand_floor_offset = 0.01  # the hand-below-table penalty starts this far above the table top
    hand_reset_clamp_max_rad = 0.6  # the profile's open pose may sit this far outside the hand action range

    # grasp_fj disturbance (2.7 N/kg, 0.27 N m/kg) at 10 Hz: the per-step firing probability is x6 of its 60 Hz range
    wrench_force_per_kg = 2.7
    wrench_torque_per_kg = 0.27
    wrench_prob_range = (0.006, 0.6)


@configclass
class IkerShoeGraspPlayEnvCfg(IkerShoeGraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.add_noise = False
```

- [ ] **Step 2: env**

```python
"""IKER shoe stage-1 grasp environment (design 2026-09-14 §4-§6).

Episodes start from a pre-grasp arm pose 8-12 cm above the shoe with the hand open. The policy moves the palm with a
6-D delta pose through damped least-squares IK (as in stage 2) and commands all 20 hand joints through the grasp_fj
full-joint law. The reward is ``grasp_stage.stage1_step``: approach and lift progress, a lift bonus and a success
bonus scaled by the five-finger grasp quality. Success = the shoe held 5 cm above its start for 20 steps.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul, sample_uniform, subtract_frame_transforms

from openarm.agnostic.modules.iker import run_files
from openarm.agnostic.modules.iker.reward import NUM_KEYPOINTS
from openarm.agnostic.modules.object_wrench import WrenchDR

from . import grasp_bank as gb
from . import grasp_stage as gs
from . import layout, robot
from .iker_shoe_env import IkerShoeEnv
from .iker_shoe_env_cfg import FRICTION, PHYSICS_DT
from .iker_shoe_grasp_env_cfg import ARM_ACTION_DIM, IkerShoeGraspEnvCfg

PREGRASP_BANK_FILE = "pregrasp_bank.json"
QUALITY_CALIBRATION_FILE = "grasp_quality_calibration.json"
PALMAR_AXIS_LOCAL = (1.0, 0.0, 0.0)  # palm link +x is the grasping side (grasp_bank.palm_rotations, both hands)


def _artifact(path_text: str, default: Path) -> Path:
    return Path(path_text) if path_text else default


class IkerShoeGraspEnv(DirectRLEnv):
    cfg: IkerShoeGraspEnvCfg

    # Scene, keypoints and quaternion noise are the stage-2 environment's, unchanged.
    _setup_scene = IkerShoeEnv._setup_scene
    _keypoints_local = IkerShoeEnv._keypoints_local
    _noisy_quat = IkerShoeEnv._noisy_quat

    def __init__(self, cfg: IkerShoeGraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
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
        self._surface_local = gs.surface_subsample(meta["objects"][layout.MOVING_SHOE]["hull_local"]).to(dev)

        prof = robot.profile()
        joint_names, body_names = list(self._robot.data.joint_names), list(self._robot.data.body_names)
        self._arm_ids, _ = self._robot.find_joints(prof.arm_joint_regex)
        self._gravity_ids, _ = self._robot.find_joints(robot.GRAVITY_COMPENSATION_JOINTS)
        self._hand_ids = torch.tensor([joint_names.index(name) for name in prof.hand_joint_names], device=dev)
        self._palm = self._robot.find_bodies(prof.palm_body)[0][0]
        self._palm_jacobian = self._palm - 1  # fixed-base Jacobians omit the root body
        link_names = [body for finger in gb.FINGERS for body in prof.finger_sensor_bodies[finger]]
        self._finger_links = torch.tensor([body_names.index(body) for body in link_names], device=dev)
        self._finger_sizes = tuple(len(prof.finger_sensor_bodies[finger]) for finger in gb.FINGERS)
        self._palmar_axis = torch.tensor(PALMAR_AXIS_LOCAL, device=dev).expand(n, 3)

        hard = self._robot.data.joint_pos_limits[0, self._hand_ids]
        self._hand_lo, self._hand_hi = gs.hand_action_limits(prof.hand_joint_names, hard[:, 0], hard[:, 1], prof.hand_action_limit_override)
        default_hand = self._robot.data.default_joint_pos[0, self._hand_ids]
        self._hand_reset = torch.max(torch.min(default_hand, self._hand_hi), self._hand_lo)
        moved = (self._hand_reset - default_hand).abs()
        print("[iker_grasp] hand action range (joint: lo..hi, open pose -> reset)", flush=True)
        for name, lo, hi, q0, q1 in zip(prof.hand_joint_names, self._hand_lo.tolist(), self._hand_hi.tolist(), default_hand.tolist(), self._hand_reset.tolist()):
            print(f"[iker_grasp]   {name:16s} {lo:+.3f}..{hi:+.3f}  {q0:+.3f} -> {q1:+.3f}", flush=True)
        if float(moved.max()) > cfg.hand_reset_clamp_max_rad:
            raise ValueError(f"the open hand pose sits {float(moved.max()):.3f} rad outside the action range (max {cfg.hand_reset_clamp_max_rad})")

        scene_config = layout.sample_configs(cfg.config_index + 1)[cfg.config_index]
        expected = json.loads(json.dumps({
            "config_index": cfg.config_index,
            "physics_dt": PHYSICS_DT,
            "friction": FRICTION,
            "solver_position_iterations": robot.SOLVER_POSITION_ITERATIONS,
            "solver_velocity_iterations": robot.SOLVER_VELOCITY_ITERATIONS,
            "gains": gb.gains_metadata(joint_names, self._robot.data.joint_stiffness[0], self._robot.data.joint_damping[0]),
            "robot_usd": str(prof.usd_relpath),
            "shoe_meta_sha256": hashlib.sha256(layout.SHOE_META_PATH.read_bytes()).hexdigest(),
            "scene_config": vars(scene_config),
        }))
        bank_doc = run_files.read_json(_artifact(cfg.pregrasp_bank_path, run_dir / PREGRASP_BANK_FILE))
        self._bank = gb.load_bank(bank_doc, joint_names, expected, dev)

        reward_cfg = replace(cfg.grasp_reward)
        calibration_path = _artifact(cfg.quality_calibration_path, run_dir / QUALITY_CALIBRATION_FILE)
        if reward_cfg.g_min < 1.0:
            # The grasp factor acts only with q_lo/q_hi measured on this shoe (design §5, audit digest row 12).
            if not calibration_path.is_file():
                raise FileNotFoundError(f"g_min {reward_cfg.g_min} < 1 needs the grasp quality calibration {calibration_path} "
                                        "(scripts/iker/measure_grasp_quality.py on a phase-A checkpoint)")
            calibration = run_files.read_json(calibration_path)
            reward_cfg = replace(reward_cfg, q_lo=float(calibration["q_lo"]), q_hi=float(calibration["q_hi"]))
        reward_cfg.__post_init__()  # re-validate after the hydra round trip and the calibration
        self._reward_cfg = reward_cfg
        idle = gs.stage1_step(gs.Stage1State.start(1, dev), reward_cfg, **{k: torch.zeros(1, device=dev) for k in (
            "palm_gap", "dz_free", "rel_speed", "shoe_speed", "q", "hand_floor_depth", "arm_speed_sum", "hand_speed_sum")},
            palm_shoe_dist=torch.ones(1, device=dev))
        print(f"[iker_grasp] reward {reward_cfg} · idle income per step {float(idle.reward):.3f} · "
              f"calibration {calibration_path if calibration_path.is_file() else 'none'}", flush=True)
        if float(idle.reward) != 0.0:
            raise RuntimeError("the stage-1 reward pays an idle policy")

        self._ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"), num_envs=n, device=dev
        )
        self._action_scale = torch.tensor([cfg.action_pos_scale] * 3 + [cfg.action_rot_scale] * 3, device=dev)
        self._joint_targets = self._robot.data.default_joint_pos.clone()
        self._hand_targets = self._hand_reset.expand(n, -1).clone()
        self._stage = gs.Stage1State.start(n, dev)
        self._start_bottom_z = torch.zeros(n, device=dev)
        self._q_at_latch = torch.zeros(n, device=dev)
        self._q_at_success = torch.zeros(n, device=dev)
        self._shoe_mass = self._shoe.root_physx_view.get_masses()[:, 0].to(dev)
        self._wrench = WrenchDR(n, dev, force_scale=cfg.wrench_force_per_kg, torque_scale=cfg.wrench_torque_per_kg,
                                prob_range=cfg.wrench_prob_range)
        self._last: gs.Stage1Step | None = None
        self.actions = torch.zeros(n, cfg.action_space, device=dev)

    # --------------------------------------------------------------- action

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        command = self.actions
        if self.cfg.add_noise:
            command = (command + sample_uniform(-self.cfg.action_noise, self.cfg.action_noise, command.shape, self.device)).clamp(-1.0, 1.0)
        root = self._robot.data.root_pose_w
        palm = self._robot.data.body_pose_w[:, self._palm]
        palm_pos_b, palm_quat_b = subtract_frame_transforms(root[:, :3], root[:, 3:7], palm[:, :3], palm[:, 3:7])
        self._ik.set_command(command[:, :ARM_ACTION_DIM] * self._action_scale, ee_pos=palm_pos_b, ee_quat=palm_quat_b)
        jacobian = self._robot.root_physx_view.get_jacobians()[:, self._palm_jacobian, :, self._arm_ids]
        arm_targets = self._ik.compute(palm_pos_b, palm_quat_b, jacobian, self._robot.data.joint_pos[:, self._arm_ids])
        limits = self._robot.data.soft_joint_pos_limits[:, self._arm_ids]
        self._joint_targets[:, self._arm_ids] = torch.clamp(arm_targets, limits[..., 0], limits[..., 1])
        self._hand_targets = gs.hand_targets(command[:, ARM_ACTION_DIM:], self._hand_lo, self._hand_hi, self._hand_targets)
        self._joint_targets[:, self._hand_ids] = self._hand_targets
        forces, torques = self._wrench.step(self._shoe_mass, self._stage.latched)
        self._shoe.set_external_force_and_torque(forces, torques, is_global=True)

    def _apply_action(self):
        self._robot.set_joint_position_target(self._joint_targets)
        tau = self._robot.root_physx_view.get_gravity_compensation_forces()
        self._robot.set_joint_effort_target(tau[:, self._gravity_ids], joint_ids=self._gravity_ids)

    # ---------------------------------------------------------- observation

    def _shoe_surface(self) -> torch.Tensor:
        """(N, P, 3) env-local surface points of the moving shoe."""
        n, p = self.num_envs, self._surface_local.shape[0]
        quat = self._shoe.data.root_quat_w[:, None, :].expand(n, p, 4)
        points = quat_apply(quat.reshape(-1, 4), self._surface_local.expand(n, p, 3).reshape(-1, 3)).view(n, p, 3)
        return points + (self._shoe.data.root_pos_w - self.scene.env_origins)[:, None, :]

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
                self._robot.data.joint_pos[:, self._hand_ids],
                gs.normalized_targets(self._hand_targets, self._hand_lo, self._hand_hi),
            ],
            dim=-1,
        )
        if self.cfg.add_noise:
            obs = obs + sample_uniform(-self.cfg.observation_noise, self.cfg.observation_noise, obs.shape, self.device)
        return {"policy": obs}

    # ------------------------------------------------------- reward / dones

    def grasp_quality_now(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(q, w_f, palm_cos) of the current simulator state."""
        origins = self.scene.env_origins
        links = self._robot.data.body_pos_w[:, self._finger_links] - origins[:, None, :]
        palm_pos = self._robot.data.body_pos_w[:, self._palm] - origins
        palm_normal = quat_apply(self._robot.data.body_quat_w[:, self._palm], self._palmar_axis)
        shoe_center = self._shoe.data.root_pos_w - origins
        return gs.grasp_quality(links, self._finger_sizes, self._shoe_surface(), palm_pos, palm_normal, shoe_center)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        origins = self.scene.env_origins
        surface = self._shoe_surface()
        palm_pos = self._robot.data.body_pos_w[:, self._palm] - origins
        shoe_pos = self._shoe.data.root_pos_w - origins
        q, w_f, palm_cos = self.grasp_quality_now()
        hand_z = torch.cat([self._robot.data.body_pos_w[:, self._finger_links, 2], palm_pos[:, 2:3]], dim=1).min(dim=1).values
        dz_free = gs.free_lift_height(surface, self._start_bottom_z, layout.RACK_X_RANGE, layout.RACK_Y_RANGE)
        shoe_vel = self._shoe.data.root_lin_vel_w
        step = gs.stage1_step(
            self._stage,
            self._reward_cfg,
            palm_gap=gs.nearest_distance(palm_pos[:, None, :], surface)[:, 0],
            dz_free=dz_free,
            palm_shoe_dist=(palm_pos - shoe_pos).norm(dim=-1),
            rel_speed=(shoe_vel - self._robot.data.body_lin_vel_w[:, self._palm]).norm(dim=-1),
            shoe_speed=shoe_vel.norm(dim=-1),
            q=q,
            hand_floor_depth=(layout.TABLE_TOP_Z + self.cfg.hand_floor_offset - hand_z).clamp(min=0.0),
            arm_speed_sum=self._robot.data.joint_vel[:, self._arm_ids].abs().sum(dim=-1),
            hand_speed_sum=self._robot.data.joint_vel[:, self._hand_ids].abs().sum(dim=-1),
        )
        self._q_at_latch = torch.where(step.just_latched, q, self._q_at_latch)
        self._q_at_success = torch.where(step.success, q, self._q_at_success)
        self._stage, self._last = step.state, step
        over_rack = (dz_free == 0.0) & (surface[..., 2].min(dim=-1).values - self._start_bottom_z > 0.02)
        log = {f"grasp_reward/{name}": value.mean().item() for name, value in step.terms.items()}
        log.update({
            "grasp/q": q.mean().item(),
            "grasp/palm_cos": palm_cos.mean().item(),
            "grasp/dz_free": dz_free.mean().item(),
            "grasp/held_frac": step.held.float().mean().item(),
            "grasp/latched_frac": step.state.latched.float().mean().item(),
            "grasp/over_rack_raised_frac": over_rack.float().mean().item(),
            "grasp/arm_speed_sum": self._robot.data.joint_vel[:, self._arm_ids].abs().sum(dim=-1).mean().item(),
            "grasp/hand_speed_sum": self._robot.data.joint_vel[:, self._hand_ids].abs().sum(dim=-1).mean().item(),
            **{f"grasp/w_{finger}": w_f[:, i].mean().item() for i, finger in enumerate(gb.FINGERS)},
        })
        self.extras["log"] = log
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        dropped = shoe_pos[:, 2] < self.cfg.drop_z
        return (step.success | dropped) & ~truncated, truncated

    def _get_rewards(self) -> torch.Tensor:
        if self._last is None:
            raise RuntimeError("_get_rewards called before _get_dones")
        return self._last.reward

    # ----------------------------------------------------------------- reset

    def _log_episode_end(self, env_ids: torch.Tensor) -> None:
        finished = env_ids[self.episode_length_buf[env_ids] > 0]
        if len(finished) == 0:
            return
        latched, succeeded = self._stage.latched[finished], self._stage.succeeded[finished]
        log = self.extras.setdefault("log", {})
        log.update({
            "grasp_episode/success": succeeded.float().mean().item(),
            "grasp_episode/latched": latched.float().mean().item(),
            "grasp_episode/best_lift_m": self._stage.best_lift[finished].mean().item(),
            "grasp_episode/q_at_latch": self._q_at_latch[finished][latched].mean().item() if bool(latched.any()) else 0.0,
            "grasp_episode/q_at_success": self._q_at_success[finished][succeeded].mean().item() if bool(succeeded.any()) else 0.0,
        })

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        self._log_episode_end(env_ids)
        super()._reset_idx(env_ids)
        count, dev = len(env_ids), self.device
        pick = torch.randint(0, self._bank.size, (count,), device=dev)
        joint_pos = self._bank.joint_pos[pick].clone()
        joint_target = self._bank.joint_target[pick].clone()
        joint_pos[:, self._hand_ids] = self._hand_reset
        joint_target[:, self._hand_ids] = self._hand_reset
        self._robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        self._robot.set_joint_position_target(joint_target, env_ids=env_ids)
        self._joint_targets[env_ids] = joint_target
        self._hand_targets[env_ids] = self._hand_reset

        origins = self.scene.env_origins[env_ids]
        zero_velocity = torch.zeros(count, 6, device=dev)
        shoe_pose = self._bank.shoe_pose[pick].clone()
        shoe_pose[:, :2] += sample_uniform(-self.cfg.start_noise_xy, self.cfg.start_noise_xy, (count, 2), dev)
        start_points = quat_apply(
            shoe_pose[:, None, 3:].expand(count, self._surface_local.shape[0], 4).reshape(-1, 4),
            self._surface_local.expand(count, -1, 3).reshape(-1, 3),
        ).view(count, -1, 3) + shoe_pose[:, None, :3]
        self._start_bottom_z[env_ids] = start_points[..., 2].min(dim=-1).values
        shoe_pose[:, :3] += origins
        self._shoe.write_root_pose_to_sim(shoe_pose, env_ids=env_ids)
        self._shoe.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

        other_pose = torch.zeros(count, 7, device=dev)
        other_pose[:, :3] = self._other_pos + origins
        other_pose[:, :2] += sample_uniform(-self.cfg.other_shoe_pos_noise, self.cfg.other_shoe_pos_noise, (count, 2), dev)
        yaw = sample_uniform(-self.cfg.other_shoe_yaw_noise, self.cfg.other_shoe_yaw_noise, (count,), dev)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(count, 3)
        other_pose[:, 3:] = quat_mul(quat_from_angle_axis(yaw, z_axis), self._other_quat.expand(count, 4))
        self._other.write_root_pose_to_sim(other_pose, env_ids=env_ids)
        self._other.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

        self._stage = self._stage.reset_rows(env_ids)
        self._q_at_latch[env_ids] = 0.0
        self._q_at_success[env_ids] = 0.0
        self._wrench.reset(env_ids)
        self.actions[env_ids] = 0.0
        self._ik.reset(env_ids)
```

- [ ] **Step 3: PPO 설정**

```yaml
# IKER shoe stage-1 grasp policy (design 2026-09-14 §4). Legacy IKER PPO (rl_games_ppo_cfg.yaml) with a wider MLP for
# the 26-D arm+hand action; 500 epochs at 4096 envs = 65.5 M frames, gate at epoch 100 (design §9).
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
      units: [512, 256, 128]
      activation: elu
      d2rl: False
      initializer:
        name: default
      regularizer:
        name: None
  load_checkpoint: False
  load_path: ''
  config:
    name: iker_shoe_grasp
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
    max_epochs: 500
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
```

- [ ] **Step 4: gym 등록**

```diff
diff --git a/source/openarm/openarm/agnostic/tasks/iker_shoe/config/__init__.py b/source/openarm/openarm/agnostic/tasks/iker_shoe/config/__init__.py
index 2a135d4..4a04c0e 100644
--- a/source/openarm/openarm/agnostic/tasks/iker_shoe/config/__init__.py
+++ b/source/openarm/openarm/agnostic/tasks/iker_shoe/config/__init__.py
@@ -21,3 +21,19 @@ for _suffix, _cfg_name in (("", "IkerShoeEnvCfg"), ("-play", "IkerShoePlayEnvCfg
             "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
         },
     )
+
+# Stage-1 grasp policy (design 2026-09-14). The side letter is the left arm that grasps; logs go to
+# log/rl_games/open-sens/left/iker-shoe-grasp/.
+_GRASP_ENTRY = "openarm.agnostic.tasks.iker_shoe.iker_shoe_grasp_env:IkerShoeGraspEnv"
+_GRASP_CFG_MODULE = "openarm.agnostic.tasks.iker_shoe.iker_shoe_grasp_env_cfg"
+
+for _suffix, _cfg_name in (("", "IkerShoeGraspEnvCfg"), ("-play", "IkerShoeGraspPlayEnvCfg")):
+    gym.register(
+        id=f"open-sens_l_iker_shoe_grasp{_suffix}",
+        entry_point=_GRASP_ENTRY,
+        disable_env_checker=True,
+        kwargs={
+            "env_cfg_entry_point": f"{_GRASP_CFG_MODULE}:{_cfg_name}",
+            "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_grasp_ppo_cfg.yaml",
+        },
+    )
```

- [ ] **Step 5: 가져오기 검사** — Isaac 없이 가능한 범위: 순수 테스트 전체 `150 passed, 1 skipped`(등록 모듈은 자동 탐색에서 try/except 로 감싸여 순수 테스트에 영향이 없다). env 부팅 검증은 Task 5 스모크가 한다.

- [ ] **Step 6: 커밋**

```bash
cd /home/user/rl_ws/hdgp-iker && git add source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_grasp_env_cfg.py source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_grasp_env.py source/openarm/openarm/agnostic/tasks/iker_shoe/config/agents/rl_games_grasp_ppo_cfg.yaml source/openarm/openarm/agnostic/tasks/iker_shoe/config/__init__.py
```

```bash
cd /home/user/rl_ws/hdgp-iker && git commit -F - <<'MSG'
feat(iker): 1단계 파지 환경 open-sens_l_iker_shoe_grasp — 손바닥 6D IK + 손 20관절, 행동 26·관측 78·120 스텝, 들린 뒤 외란

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4
MSG
```

---

### Task 5: 접근 자세 뱅크·q 보정·1단계 스모크(Isaac)

**Files:**
- Create: `scripts/iker/make_pregrasp_bank.py`, `scripts/iker/measure_grasp_quality.py`, `scripts/iker/grasp_smoke.py`
- Generate: `iker_runs/shoe_place/config_00/pregrasp_bank.json`

**Interfaces:**
- Consumes: Task 2 `keypoints.json`(신발 정착 자세), Task 4 env.
- Produces: 1단계 리셋 뱅크. `measure_grasp_quality.py` 는 Task 6 에서 A 단계 체크포인트로 돌린다.

- [ ] **Step 1: 스크립트 3개 작성**

```python
"""Build the pre-grasp arm pose bank of one IKER configuration for the stage-1 grasp policy (design 2026-09-14 §4).

Every round, all envs drive the palm by IK from home to a sampled pose 8-12 cm above the shoe top (the K1 pre-grasp
geometry lifted, wrist near level, mirrored for the left arm) with the hand open and the shoe parked out of the way;
then the shoe is placed at its settled snapshot pose and the scene settles. States whose palm converged (position
< 5 mm, orientation < 3 deg), whose finger links all stay >= 1 cm from the shoe surface and whose shoe did not move
are kept.

Usage:
    cd ~/rl_ws/hdgp && PYTHONPATH=source/openarm ../IsaacLab/isaaclab.sh -p scripts/iker/make_pregrasp_bank.py \
        --config-index 0 --headless
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import sys
import time
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Build the IKER stage-1 pre-grasp pose bank of one configuration.")
parser.add_argument("--config-index", type=int, required=True)
parser.add_argument("--num-envs", type=int, default=128)
parser.add_argument("--min-entries", type=int, default=256)
parser.add_argument("--max-rounds", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    print("PREGRASP FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_from_matrix  # noqa: E402

from openarm.agnostic.modules.iker import run_files  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import grasp_bank as gb  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import grasp_stage as gs  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import layout, robot, scene_cfg  # noqa: E402

PHYSICS_DT = 1.0 / 120.0
APPROACH_STEPS, SETTLE_STEPS = 480, 60
# design §4: the palm starts 8-12 cm above the shoe top with the hand open and the wrist near level. The K1 tilt
# (-40..-15 deg) points the open fingers into the shoe. With the palm facing down the thumb hangs below the palm,
# so "no contact" is the distance from every finger link to the shoe's surface points, not height above the top.
START_HEIGHT_ABOVE_TOP_RANGE = (0.08, 0.12)
START_TILT_DEG_RANGE = (-10.0, 0.0)
MAX_PALM_ERROR_M = 0.005
MAX_PALM_ANGLE_DEG = 3.0
MIN_FINGER_SURFACE_GAP_M = 0.01
MAX_SHOE_SHIFT_M = 0.002  # the shoe starts at its settled snapshot pose, so it must not move
PARK_POS = (0.42, 0.40)
FRICTION = 1.0  # same contact material as the training environment


def build(start_pos, start_quat):
    @configclass
    class SceneCfg(InteractiveSceneCfg):
        table = scene_cfg.table_cfg()
        rack = scene_cfg.rack_cfg()
        light = scene_cfg.light_cfg()
        robot = robot.robot_cfg()
        shoe = scene_cfg.shoe_cfg(layout.MOVING_SHOE, (PARK_POS[0], PARK_POS[1], start_pos[2]), start_quat)

    sim = SimulationContext(
        SimulationCfg(
            dt=PHYSICS_DT,
            device=args.device,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=FRICTION, dynamic_friction=FRICTION, restitution=0.0),
        )
    )
    scene = InteractiveScene(SceneCfg(num_envs=args.num_envs, env_spacing=2.0, replicate_physics=True))
    sim.reset()
    return sim, scene


def quat_angle_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(2.0 * torch.acos((a * b).sum(dim=-1).abs().clamp(max=1.0)))


def main() -> int:
    config = layout.sample_configs(args.config_index + 1)[args.config_index]
    meta = run_files.load_shoe_meta(layout.SHOE_META_PATH)
    run_dir = layout.RUNS_DIR / f"config_{config.index:02d}"
    settled = run_files.read_json(run_dir / "keypoints.json")["objects"][layout.MOVING_SHOE]
    sim, scene = build(settled["position"], settled["quat_wxyz"])
    arm, shoe, dev, n = scene["robot"], scene["shoe"], sim.device, args.num_envs
    prof = robot.profile()
    origins = scene.env_origins
    arm_ids, _ = arm.find_joints(prof.arm_joint_regex)
    gravity_ids, _ = arm.find_joints(robot.GRAVITY_COMPENSATION_JOINTS)
    hand_ids = [arm.data.joint_names.index(name) for name in prof.hand_joint_names]
    body_names = list(arm.data.body_names)
    finger_links = [body_names.index(b) for f in gb.FINGERS for b in prof.finger_sensor_bodies[f]]
    palm = arm.find_bodies(prof.palm_body)[0][0]
    side_sign = gb.side_sign(prof.hand_joint_names)
    lower = arm.data.soft_joint_pos_limits[0, hand_ids, 0]
    upper = arm.data.soft_joint_pos_limits[0, hand_ids, 1]
    open_hand = torch.max(torch.min(torch.tensor(prof.hand_open_pose, device=dev), upper), lower)
    ik = DifferentialIKController(DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"), num_envs=n, device=dev)
    home = arm.data.default_joint_pos.clone()
    shoe_obj = meta["objects"][layout.MOVING_SHOE]
    top_z = layout.TABLE_TOP_Z + 2.0 * float(shoe_obj["rest_height"])
    half_width = abs(float(shoe_obj["keypoint_offsets"][2][0]))
    start = torch.tensor([*settled["position"], *settled["quat_wxyz"]], dtype=torch.float32, device=dev)
    surface = gs.surface_subsample(shoe_obj["hull_local"]).to(dev)
    start_surface = quat_apply(start[3:].expand(surface.shape[0], 4), surface) + start[:3]
    park = start.clone()
    park[:2] = torch.tensor(PARK_POS, device=dev)
    generator = torch.Generator().manual_seed(args.seed)
    zero_vel = torch.zeros(n, 6, device=dev)

    def write_shoe(local_pose):
        pose = local_pose.clone()
        pose[:, :3] += origins
        shoe.write_root_pose_to_sim(pose)
        shoe.write_root_velocity_to_sim(zero_vel)

    def run(steps, palm_goal, palm_quat, parked):
        for _ in range(steps):
            if parked:
                write_shoe(park.expand(n, 7))
            ee_pos, ee_quat = arm.data.body_pos_w[:, palm] - origins, arm.data.body_quat_w[:, palm]
            ik.set_command(torch.cat([palm_goal, palm_quat], dim=-1), ee_pos=ee_pos, ee_quat=ee_quat)
            jac = arm.root_physx_view.get_jacobians()[:, palm - 1, :, arm_ids]
            target = arm.data.joint_pos_target.clone()
            target[:, arm_ids] = ik.compute(ee_pos, ee_quat, jac, arm.data.joint_pos[:, arm_ids])
            target[:, hand_ids] = open_hand
            arm.set_joint_position_target(target)
            tau = arm.root_physx_view.get_gravity_compensation_forces()
            arm.set_joint_effort_target(tau[:, gravity_ids], joint_ids=gravity_ids)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(PHYSICS_DT)

    def uniform(bounds):
        return bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n, generator=generator).to(dev)

    kept = {key: [] for key in ("joint_pos", "joint_target", "shoe_pose", "palm_pose")}
    stats, t0 = [], time.time()
    quantiles = torch.tensor([0.1, 0.5, 0.9], device=dev)
    for round_index in range(args.max_rounds):
        start_pose = home.clone()
        start_pose[:, hand_ids] = open_hand
        arm.write_joint_state_to_sim(start_pose, torch.zeros_like(start_pose))
        arm.set_joint_position_target(start_pose)
        ik.reset()
        pre = dataclasses.replace(
            gb.sample_pregrasp(n, generator, dev),
            height_above_top=uniform(START_HEIGHT_ABOVE_TOP_RANGE),
            tilt_deg=uniform(START_TILT_DEG_RANGE),
        )
        palm_quat = quat_from_matrix(gb.palm_rotations(pre.tilt_deg, pre.yaw_deg, side_sign))
        goal = gb.palm_goal_positions(pre, (config.move_x, config.move_y), config.move_yaw_deg, half_width, top_z, side_sign)
        run(APPROACH_STEPS, goal, palm_quat, parked=True)
        write_shoe(start.expand(n, 7))
        run(SETTLE_STEPS, goal, palm_quat, parked=False)
        palm_pos = arm.data.body_pos_w[:, palm] - origins
        position_error = (palm_pos - goal).norm(dim=-1)
        angle_error = quat_angle_deg(arm.data.body_quat_w[:, palm], palm_quat)
        links = arm.data.body_pos_w[:, finger_links] - origins[:, None, :]
        finger_gap = gs.nearest_distance(links, start_surface.expand(n, -1, 3)).min(dim=1).values
        shoe_shift = (shoe.data.root_pos_w - origins - start[:3]).norm(dim=-1)
        ok = (position_error < MAX_PALM_ERROR_M) & (angle_error < MAX_PALM_ANGLE_DEG) & (finger_gap >= MIN_FINGER_SURFACE_GAP_M) & (shoe_shift < MAX_SHOE_SHIFT_M)
        record = {
            "joint_pos": arm.data.joint_pos.clone(),
            "joint_target": arm.data.joint_pos_target.clone(),
            "shoe_pose": torch.cat([shoe.data.root_pos_w - origins, shoe.data.root_quat_w], dim=-1),
            "palm_pose": torch.cat([palm_pos, arm.data.body_quat_w[:, palm]], dim=-1),
        }
        for key in kept:
            kept[key].append(record[key][ok])
        total = sum(int(t.shape[0]) for t in kept["joint_pos"])
        stats.append({"round": round_index, "kept": int(ok.sum()), "total": total})
        print(
            f"PREGRASP round {round_index} kept {int(ok.sum())}/{n} total {total} "
            f"pos err mm median {float(position_error.median() * 1000):.1f} angle deg median {float(angle_error.median()):.2f} "
            f"finger gap mm q10/50/90 {[round(float(v) * 1000, 1) for v in torch.quantile(finger_gap, quantiles)]} "
            f"shoe shift mm q90 {float(torch.quantile(shoe_shift, 0.9) * 1000):.2f} ({time.time() - t0:.0f} s)",
            flush=True,
        )
        if total >= args.min_entries:
            break

    entries = {key: torch.cat(parts) for key, parts in kept.items()}
    total = int(entries["joint_pos"].shape[0])
    metadata = {
        "config_index": config.index,
        "scene_config": vars(config),
        "robot_usd": str(prof.usd_relpath),
        "physics_dt": PHYSICS_DT,
        "friction": FRICTION,
        "solver_position_iterations": robot.SOLVER_POSITION_ITERATIONS,
        "solver_velocity_iterations": robot.SOLVER_VELOCITY_ITERATIONS,
        "gains": gb.gains_metadata(arm.data.joint_names, arm.data.joint_stiffness[0], arm.data.joint_damping[0]),
        "shoe_meta_sha256": hashlib.sha256(layout.SHOE_META_PATH.read_bytes()).hexdigest(),
        "seed": args.seed,
        "rounds": stats,
        "source": "pregrasp",
        "start_height_above_top_m": list(START_HEIGHT_ABOVE_TOP_RANGE),
        "start_tilt_deg": list(START_TILT_DEG_RANGE),
        "side_sign": side_sign,
    }
    out = run_dir / "pregrasp_bank.json"
    if total == 0:
        print(f"PREGRASP config {config.index:02d} entries 0 (min {args.min_entries}) passed False", flush=True)
        return 1
    run_files.write_json(out, gb.bank_document(entries, arm.data.joint_names, metadata))
    passed = total >= args.min_entries
    print(f"PREGRASP config {config.index:02d} entries {total} (min {args.min_entries}) passed {passed} -> {out}", flush=True)
    return 0 if passed else 1


os._exit(main())
```

```python
"""Measure the grasp quality q of a phase-A stage-1 checkpoint and write q_lo/q_hi (design 2026-09-14 §5).

Runs the checkpoint deterministically (no noise, no disturbance), takes q at every lift-latch and success moment of
each env's first episode, and writes q_lo = p25 and q_hi = p90 of those moments. Refuses to write with fewer than
``--min-events`` moments: a checkpoint that does not lift yet cannot calibrate anything.

Usage:
    cd ~/rl_ws/hdgp && PYTHONPATH=source/openarm ../IsaacLab/isaaclab.sh -p scripts/iker/measure_grasp_quality.py \
        --checkpoint <phase-A .pth> --headless
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure grasp quality q on a phase-A stage-1 checkpoint.")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--config-index", type=int, default=0)
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--min-events", type=int, default=64)
parser.add_argument("--seed", type=int, default=11)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    print("QUALITY FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

import openarm.agnostic.tasks.iker_shoe.config  # noqa: E402,F401  (registers the gym ids)
from openarm.agnostic.modules.iker import run_files  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import grasp_bank as gb  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import layout  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.iker_shoe_grasp_env import QUALITY_CALIBRATION_FILE  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.iker_shoe_grasp_env_cfg import IkerShoeGraspPlayEnvCfg  # noqa: E402

TASK = "open-sens_l_iker_shoe_grasp"
Q_LO_PERCENTILE, Q_HI_PERCENTILE = 0.25, 0.90


def main() -> int:
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cfg = IkerShoeGraspPlayEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.config_index = args.config_index
    cfg.seed = args.seed
    cfg.grasp_reward.g_min = 1.0  # measuring must not depend on a calibration
    cfg.wrench_prob_range = (1e-9, 1e-9)
    env = gym.make(TASK, cfg=cfg)
    u = env.unwrapped
    n, dev = u.num_envs, u.device
    agent_cfg = load_cfg_from_registry(TASK, "rl_games_cfg_entry_point")
    params = agent_cfg["params"]["env"]
    wrapped = RlGamesVecEnvWrapper(env, agent_cfg["params"]["config"].get("device", "cuda:0"), params.get("clip_observations", math.inf),
                                   params.get("clip_actions", math.inf), params.get("obs_groups"), params.get("concate_obs_groups", True))
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: wrapped})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(checkpoint)
    agent_cfg["params"]["config"]["num_actors"] = n
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(str(checkpoint))
    agent.reset()

    done_first = torch.zeros(n, dtype=torch.bool, device=dev)
    latch_q, success_q, w_at_latch = [], [], []
    obs = wrapped.reset()
    obs = obs["obs"] if isinstance(obs, dict) else obs
    _ = agent.get_batch_size(obs, 1)
    with torch.inference_mode():
        for _ in range(u.max_episode_length + 5):
            actions = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
            live = ~done_first
            obs, _, dones, _ = wrapped.step(actions)
            obs = obs["obs"] if isinstance(obs, dict) else obs
            step = u._last
            latched_now = step.just_latched & live
            if bool(latched_now.any()):
                latch_q.append(u._q_at_latch[latched_now].cpu())
                w_at_latch.append(u.grasp_quality_now()[1][latched_now].cpu())
            succeeded_now = step.success & live
            if bool(succeeded_now.any()):
                success_q.append(u._q_at_success[succeeded_now].cpu())
            done_first |= dones.bool()
            if bool(done_first.all()):
                break

    moments = torch.cat(latch_q + success_q) if (latch_q or success_q) else torch.zeros(0)
    events = int(moments.numel())
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "envs": n,
        "latch_events": int(sum(t.numel() for t in latch_q)),
        "success_events": int(sum(t.numel() for t in success_q)),
        "percentiles": {"lo": Q_LO_PERCENTILE, "hi": Q_HI_PERCENTILE},
    }
    if events < args.min_events:
        print(f"QUALITY config {args.config_index:02d} {result} passed False: {events} latch/success moments < {args.min_events}", flush=True)
        return 1
    result.update({
        "q_lo": float(torch.quantile(moments, Q_LO_PERCENTILE)),
        "q_hi": float(torch.quantile(moments, Q_HI_PERCENTILE)),
        "q_quantiles_10_25_50_75_90": [round(float(torch.quantile(moments, p)), 4) for p in (0.1, 0.25, 0.5, 0.75, 0.9)],
        "w_f_median_at_latch": {f: round(float(torch.cat(w_at_latch)[:, i].median()), 4) for i, f in enumerate(gb.FINGERS)} if w_at_latch else {},
    })
    passed = result["q_lo"] < result["q_hi"]
    out = layout.RUNS_DIR / f"config_{args.config_index:02d}" / QUALITY_CALIBRATION_FILE
    if passed:
        run_files.write_json(out, result)
    print(f"QUALITY config {args.config_index:02d} {result} passed {passed} -> {out}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    code = main()
    os._exit(code)
```

```python
"""Smoke-check the IKER stage-1 grasp environment before training (design 2026-09-14 §10).

1. boots with the pre-grasp bank; observations (N, 78) are finite; the palm starts 5-15 cm (surface gap) from the shoe;
2. zero actions for 3 s pay no lift or bonus term and latch nothing; the palm-progress ratchet may pay the arm's small
   zero-action sag once (bounded, not a per-step income);
3. a shoe pushed up onto the rack is not a free lift (dz_free 0, never held);
4. the reward wiring with the physics bypassed: a shoe written 6 cm above its start, clear of the hand, with zero velocity
   is held; the lift bonus latches on the third ``_get_dones`` call and the success bonus on the twentieth, each once;
5. random actions for 12 s keep rewards finite and write episode-end logs.

Usage:
    cd ~/rl_ws/hdgp && PYTHONPATH=source/openarm ../IsaacLab/isaaclab.sh -p scripts/iker/grasp_smoke.py --headless
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke-check the IKER stage-1 grasp environment.")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--config-index", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    print("GRASP SMOKE FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.agnostic.tasks.iker_shoe.config  # noqa: E402,F401  (registers the gym ids)
from openarm.agnostic.tasks.iker_shoe import grasp_stage as gs  # noqa: E402
from openarm.agnostic.tasks.iker_shoe import layout  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.iker_shoe_grasp_env_cfg import IkerShoeGraspEnvCfg  # noqa: E402

START_GAP_RANGE_M = (0.05, 0.18)  # palm 8-12 cm above the shoe top and offset to its side: measured 105-142 mm
MAX_IDLE_PALM_PROGRESS = 2.0  # 50 x 4 cm of zero-action sag; the ratchet cannot pay more than the start gap
FORCED_LIFT_M = 0.06
FORCED_SHIFT_X_M = 0.15  # clear of the open hand and of the rack footprint
FORCED_HOLD_RADIUS_M = 0.5  # the palm-distance condition is covered by the pure tests; this check targets the wiring
LATCH_CALL, SUCCESS_CALL = 2, 19  # zero-based: the third and the twentieth held call


def forced_hold(env, calls: int):
    """Write every env's shoe ``FORCED_LIFT_M`` above its start with zero velocity and evaluate ``_get_dones`` without
    stepping physics: a written pose falls ~5 cm during the 12 physics substeps of a policy step, so this check isolates
    the env's dz_free -> held -> latch -> success wiring from the simulator."""
    n, dev, origins = env.num_envs, env.device, env.scene.env_origins
    pose = env._shoe.data.root_state_w[:, :7].clone()
    pose[:, 0] += FORCED_SHIFT_X_M
    pose[:, 2] += FORCED_LIFT_M
    latch_calls, success_calls, paid = [], [], torch.zeros(n, device=dev)
    for index in range(calls):
        env._shoe.write_root_pose_to_sim(pose)
        env._shoe.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
        env.scene.update(env.physics_dt)
        env._get_dones()
        step = env._last
        if index < 3:
            dz = gs.free_lift_height(env._shoe_surface(), env._start_bottom_z, layout.RACK_X_RANGE, layout.RACK_Y_RANGE)
            shoe = env._shoe.data.root_pos_w - origins
            print(f"SMOKE forced hold call {index} env0: dz_free {float(dz[0]) * 1000:.1f} mm, shoe z {float(shoe[0, 2]):.3f} "
                  f"(target {float(pose[0, 2] - origins[0, 2]):.3f}), held {bool(step.held[0])}, hold count {int(step.state.hold_count[0])}",
                  flush=True)
        if bool(step.just_latched.any()):
            latch_calls.append(index)
        if bool(step.success.any()):
            success_calls.append(index)
        paid += step.terms["lift_bonus"] + step.terms["success_bonus"]
    return latch_calls, success_calls, paid


def main() -> int:
    cfg = IkerShoeGraspEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.config_index = args.config_index
    cfg.add_noise = False
    cfg.wrench_prob_range = (1e-9, 1e-9)
    cfg.grasp_reward.hold_radius_m = FORCED_HOLD_RADIUS_M
    env = gym.make("open-sens_l_iker_shoe_grasp", cfg=cfg).unwrapped
    n, dev = env.num_envs, env.device
    failures = []

    obs, _ = env.reset()
    policy = obs["policy"]
    origins = env.scene.env_origins
    palm = env._robot.data.body_pos_w[:, env._palm] - origins
    gap = gs.nearest_distance(palm[:, None, :], env._shoe_surface())[:, 0]
    print(f"SMOKE obs {tuple(policy.shape)} finite {bool(torch.isfinite(policy).all())} pregrasp bank {env._bank.size} "
          f"start palm gap mm min {float(gap.min()) * 1000:.1f} max {float(gap.max()) * 1000:.1f}", flush=True)
    if policy.shape != (n, 78) or not torch.isfinite(policy).all():
        failures.append("observation shape or finiteness")
    if not (START_GAP_RANGE_M[0] <= float(gap.min()) and float(gap.max()) <= START_GAP_RANGE_M[1]):
        failures.append(f"start palm gap outside {START_GAP_RANGE_M}")

    term_sums = {name: torch.zeros(n, device=dev) for name in gs.REWARD_TERMS}
    for _ in range(30):
        env.step(torch.zeros(n, env.cfg.action_space, device=dev))
        for name in gs.REWARD_TERMS:
            term_sums[name] += env._last.terms[name]
    latched = float(env._stage.latched.float().mean())
    sums = {k: round(float(v.abs().max()), 4) for k, v in term_sums.items()}
    print(f"SMOKE zero action 3 s: latched {latched:.2f}, per-term max |sum| {sums}", flush=True)
    if latched > 0.0 or any(sums[k] > 0.0 for k in ("lift_progress", "lift_bonus", "success_bonus")) or sums["palm_progress"] > MAX_IDLE_PALM_PROGRESS:
        failures.append("zero-action policy earned a lift or bonus term, latched, or more palm progress than the sag bound")

    env.reset()
    rack_pose = env._shoe.data.root_state_w[:, :7].clone()
    rack_pose[:, 0] = origins[:, 0] + sum(layout.RACK_X_RANGE) / 2
    rack_pose[:, 1] = origins[:, 1] + layout.RACK_Y_RANGE[1] - 0.08
    rack_pose[:, 2] = origins[:, 2] + layout.RACK_TOP_Z + (env._shoe.data.root_pos_w[:, 2] - origins[:, 2]) - layout.TABLE_TOP_Z + 0.002
    held_any = False
    for _ in range(10):
        env._shoe.write_root_pose_to_sim(rack_pose)
        env._shoe.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
        env.step(torch.zeros(n, env.cfg.action_space, device=dev))
        held_any |= bool(env._last.held.any())
    dz_on_rack = gs.free_lift_height(env._shoe_surface(), env._start_bottom_z, layout.RACK_X_RANGE, layout.RACK_Y_RANGE)
    print(f"SMOKE shoe on rack: dz_free max {float(dz_on_rack.max()) * 1000:.1f} mm, held {held_any}", flush=True)
    if float(dz_on_rack.max()) > 0.0 or held_any:
        failures.append("a shoe on the rack counted as a free lift")

    env.reset()
    latch_calls, success_calls, paid = forced_hold(env, 25)
    print(f"SMOKE forced hold: latch calls {latch_calls}, success calls {success_calls}, bonus paid min {float(paid.min()):.1f} max {float(paid.max()):.1f}", flush=True)
    if latch_calls != [LATCH_CALL] or success_calls != [SUCCESS_CALL] or float(paid.min()) != float(paid.max()):
        failures.append(f"forced hold latched at {latch_calls} (want [{LATCH_CALL}]) and succeeded at {success_calls} (want [{SUCCESS_CALL}])")

    env.reset()
    rewards, logs = [], []
    for _ in range(120):
        _, reward, _, _, extras = env.step(2.0 * torch.rand(n, env.cfg.action_space, device=dev) - 1.0)
        rewards.append(reward)
        if "grasp_episode/success" in extras.get("log", {}):
            logs.append(dict(extras["log"]))
    stacked = torch.stack(rewards)
    print(f"SMOKE random 12 s: reward finite {bool(torch.isfinite(stacked).all())} mean {float(stacked.mean()):.3f} "
          f"max {float(stacked.max()):.1f} episode logs {len(logs)}", flush=True)
    if not torch.isfinite(stacked).all() or not logs:
        failures.append("random-action rewards or episode logs")

    for failure in failures:
        print(f"SMOKE CHECK FAILED: {failure}", flush=True)
    print(f"GRASP SMOKE passed {not failures}", flush=True)
    return 0 if not failures else 1


os._exit(main())
```

- [ ] **Step 2: 접근 자세 뱅크** — `scripts/iker/make_pregrasp_bank.py --config-index 0` (Isaac, `grep -E "PREGRASP|Traceback|Error"`).
  - 기대: `PREGRASP config 00 entries N (min 256) passed True`. 미러 실측: 3 라운드 320 개(라운드당 106~107/128), 손바닥 IK 위치 오차 중앙값 0.4~0.5 mm·자세 0.2°, 손가락–표면 간격 q10/50/90 ≈ 26/39/57 mm, 신발 이동 q90 0.79 mm, 약 35 s.
  - `passed False` 면 모든 PREGRASP·diag 줄과 함께 BLOCKED 로 보고한다.

- [ ] **Step 3: 1단계 스모크** — `scripts/iker/grasp_smoke.py --num-envs 16` (Isaac, `grep -E "SMOKE|Traceback|Error|iker_grasp"`).
  - 기대: 손 행동 한계 20줄 표, 모든 `SMOKE` 줄, `GRASP SMOKE passed True`. 미러 실측: `SMOKE obs (16, 78) finite True pregrasp bank 320 start palm gap mm min 104.7 max 142.3` · 무행동 3 s `lift_progress`·`lift_bonus`·`success_bonus` 0, `palm_progress` 0.375(팔 처짐, 상한 2.0), 래치 0 · 받침 위 `dz_free max 0.0 mm, held False` · 강제 유지 `latch calls [2], success calls [19], bonus paid min 1300.0 max 1300.0`(A 단계 g=1) · 무작위 12 s 보상 유한·에피소드 로그 1.

- [ ] **Step 4: 커밋**

```bash
cd /home/user/rl_ws/hdgp-iker && git add scripts/iker/make_pregrasp_bank.py scripts/iker/measure_grasp_quality.py scripts/iker/grasp_smoke.py iker_runs/shoe_place/config_00/pregrasp_bank.json
```

```bash
cd /home/user/rl_ws/hdgp-iker && git commit -F - <<'MSG'
feat(iker): 1단계 접근 자세 뱅크·q 보정 스크립트(A 단계 체크포인트용)·시뮬레이터 스모크

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4
MSG
```

---

### Task 6: 학습 스모크·A 단계 학습·q 보정·B 단계 착수

**Files:**
- Generate: `iker_runs/shoe_place/config_00/grasp_quality_calibration.json`
- 런 산출물은 git 무시 대상 `log/`.

- [ ] **Step 1: 학습 스모크(4096 env, 3 epoch)**

```bash
cd /home/user/rl_ws/hdgp-iker && RUN_LABEL=iker_grasp_smoke TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm ../IsaacLab/_isaac_sim/python.sh scripts/reinforcement_learning/rl_games/train.py --task open-sens_l_iker_shoe_grasp --num_envs 4096 --max_iterations 3 --headless 2>&1 | grep -E "fps total|MAX EPOCHS|Traceback|Error|iker_grasp\] reward"
```
  - 기대: `iker_grasp] reward ... g_min=1.0 ... idle income per step 0.000`, epoch 줄 fps, `MAX EPOCHS NUM!`. 미러 실측: `fps total: 7436` → `7755` → `7490`, `MAX EPOCHS NUM!`, 부팅 줄 `lift_deadband_m=0.01 ... hold_radius_m=0.15 ...`(2단계 IKER env 4096 env ≈ 8,000 fps 와 같은 수준).
  - fps 가 2,000 미만이면 멈추고 보고한다(표면 거리 연산량 위험, 스펙 §13).
  - 스모크 런 폴더 `log/rl_games/open-sens/left/iker-shoe-grasp/iker_grasp_smoke` 는 지운다(스모크 산출물).

- [ ] **Step 2: A 단계 학습 착수(g(q) 끔, 500 epoch)**

```bash
cd /home/user/rl_ws/hdgp-iker && git status --short && RUN_LOG=$PWD/log/rl_games/open-sens/left/train_iker_grasp_c00_a.log && mkdir -p $(dirname $RUN_LOG) && RUN_LABEL=iker_grasp_c00_a NOTE="IKER 1단계 학습 파지 A 단계(g(q) 끔), 왼팔·손 20관절, 구성 0, 스펙 2026-09-14" TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm setsid nohup ../IsaacLab/_isaac_sim/python.sh scripts/reinforcement_learning/rl_games/train.py --task open-sens_l_iker_shoe_grasp --num_envs 4096 --headless > "$RUN_LOG" 2>&1 < /dev/null & sleep 15; ps -eo pid,ppid,etime,cmd | grep "train.py --task open-sens_l_iker_shoe_grasp" | grep -v grep | cut -c1-150
```
  - `git status --short` 가 비어 있어야 한다. python3 train.py PID 를 기록한다(종료는 이 PID 로만).

- [ ] **Step 3: epoch 100 게이트(스펙 §9)** — TFEvents(`log/rl_games/open-sens/left/iker-shoe-grasp/iker_grasp_c00_a/summaries/`)에서 `grasp_episode/latched`·`grasp_episode/success`·`grasp/over_rack_raised_frac`·`grasp/q`·`grasp_reward/*` 를 10구간 평균으로 뽑는다.
  - `grasp_episode/latched` 가 2 % 미만이면 학습을 PID 로 멈추고 보고한다(설계 문제).
  - `grasp/over_rack_raised_frac` 가 오르면 해킹 의심으로 보고한다.

- [ ] **Step 4: q 보정** — `grasp_episode/latched` 가 10 % 이상인 첫 저장 체크포인트(`nn/*_ep_*.pth`, 50 epoch 간격)로:

```bash
cd /home/user/rl_ws/hdgp-iker && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm ../IsaacLab/_isaac_sim/python.sh scripts/iker/measure_grasp_quality.py --checkpoint <A 체크포인트 절대경로> --headless 2>&1 | grep -E "QUALITY|Traceback|Error"
```
  - 학습과 동시에 돌린다(512 env, GPU 메모리 여유 확인 후). 기대: `QUALITY config 00 {...} passed True` 와 파일 생성. 래치·성공 순간이 64 개 미만이면 `passed False` — 다음 저장 체크포인트로 다시 한다.
  - q 분위수·손가락별 w_f 중앙값을 보고한다.
  - 첫 래치가 나온 체크포인트로 영상을 만든다: `scripts/reinforcement_learning/rl_games/play.py --task open-sens_l_iker_shoe_grasp-play --checkpoint <pth> --video --video_length 200 --num_envs 16 --headless`. 파지 형태(손가락이 신발을 감싸는지, 받침에 기대 올리는지)를 보고한다(스펙 §9).
  - 보정 파일을 커밋한다:

```bash
cd /home/user/rl_ws/hdgp-iker && git add iker_runs/shoe_place/config_00/grasp_quality_calibration.json
```

```bash
cd /home/user/rl_ws/hdgp-iker && git commit -F - <<'MSG'
feat(iker): 1단계 A 단계 체크포인트로 신발 파지 품질 q 보정(q_lo p25·q_hi p90)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4
MSG
```

- [ ] **Step 5: B 단계 착수(g(q) 켬, A 체크포인트에서 이어학습)** — A 런을 PID 로 멈추고, 보정에 쓴 체크포인트에서 이어간다.

```bash
cd /home/user/rl_ws/hdgp-iker && git status --short && CK=<보정에 쓴 A 체크포인트 절대경로> && RUN_LOG=$PWD/log/rl_games/open-sens/left/train_iker_grasp_c00_b.log && RUN_LABEL=iker_grasp_c00_b NOTE="IKER 1단계 B 단계(g(q) 켬, g_min 0.5, 보정 파일), A 체크포인트 이어학습" TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm setsid nohup ../IsaacLab/_isaac_sim/python.sh scripts/reinforcement_learning/rl_games/train.py --task open-sens_l_iker_shoe_grasp --num_envs 4096 --headless --checkpoint "$CK" --no-reset_epoch --max_iterations 1000 env.grasp_reward.g_min=0.5 > "$RUN_LOG" 2>&1 < /dev/null & sleep 60; grep -aE "iker_grasp\] reward|Traceback" "$RUN_LOG" | head -3 | cut -c1-300
```
  - 기대: 부팅 줄에 `g_min=0.5` 와 보정 파일의 `q_lo`·`q_hi`, `calibration .../grasp_quality_calibration.json`. 적용값이 다르면(hydra 덮어쓰기 실패) 멈추고 보고한다.
  - PID 를 기록한다.

- [ ] **Step 6: 인계** — 보고서에 A·B 런 PID·로그 경로·epoch 100 게이트 값·보정값·영상 판정을 적는다. 계획 2c(수확·2단계)는 B 단계 결과를 보고 쓴다.

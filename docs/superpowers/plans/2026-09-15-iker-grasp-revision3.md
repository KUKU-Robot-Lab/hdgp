# IKER 1단계 개정 3 (시작 자리 근처 잡힘 · thumb_3 한계) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 1단계 파지 정책이 신발을 시작 자리 근처에서 엄지를 조인 채 잡아야 성공하게 하고, thumb_3 역굴곡을 sim 관절 한계로 막고, 2단계 스모크의 무행동 판정을 손안 미끄럼으로 고친 뒤, 첫 루프 산출물을 보존해 루프를 1단계 A 부터 다시 돌릴 수 있게 한다.

**Architecture:** 판정은 순수 모듈 `grasp_stage.py`(held 조건 3 개 추가·`closing_travel`·`backstop_limits`)에 두고, env 는 입력(시작 xy·엄지 조임)만 계산한다. 관절 한계는 `robot.apply_hand_backstop` 한 곳에서 1·2단계 env 부팅 때 쓰고, 그 값을 뱅크 부팅 대조 키 `hand_backstop` 으로 남긴다. 스모크 두 개와 파일 이동은 마지막 두 태스크다.

**Tech Stack:** Python 3.11, PyTorch, Isaac Lab (DirectRLEnv, Articulation), pytest.

**Spec:** `docs/superpowers/specs/2026-09-14-iker-learned-grasp-left-arm-design.md` §16 (개정 3) + `docs/superpowers/specs/2026-09-14-iker-auto-loop-design.md` §13 (개정 1).

## Global Constraints

- 작업 위치는 워크트리 `~/rl_ws/hdgp-iker`(브랜치 `iker-front-end`)뿐이다. `~/rl_ws/hdgp` 에서는 어떤 명령도 실행하지 않는다.
- 로봇 값은 벤더·프로필에서만 읽는다. 제품 코드에 `r_hj_`·`l_hj_` 리터럴을 쓰지 않는다(역할 이름 `thumb_3` 과 `<side>_hj_<role>` 규칙을 쓴다). 공유 프로필 `tesollo_left_short`(`modules/robot_profiles.py`)는 수정하지 않는다.
- IKER 2단계 보상 5항(`modules/iker/reward.py`)은 바꾸지 않는다. reward-audit 은 IKER 에서 쓰지 않는다.
- 값: `lift_max_m = 0.15`, `hold_xy_radius_m = 0.10`, `thumb_curl_min_rad = 0.05`, `hand_backstop_joints = ("thumb_3",)`, `thumb_curl_role = "thumb_3"`, 2단계 스모크 미끄럼 기준 `0.03 m`·허용 비율 `0.25`.
- 트랙·라벨(태스크 4): `iker_shoe_c00_r2`, `iker_grasp_c00_r2_a`, `iker_grasp_c00_r2_b`, `iker_vlm_c00_r2_s1`. 보존 경로 `iker_runs/shoe_place/config_00/archive/2026-09-14_midair/`.
- Isaac 은 로컬 GPU 에서 한 번에 하나, 포그라운드로 `tee` 로그를 남기며 돌린다. 로그는 `log/` 아래에만 둔다. 프로세스 종료는 PID 로만(pkill·killall 금지). 학습은 띄우지 않는다.
- 순수 테스트 명령: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests source/openarm/openarm/agnostic/modules/iker/tests -q -p no:cacheprovider`. 착수 전 기준선: 1 failed(`test_grasp_bank_file.py::test_bank_entries_have_no_finger_at_an_opposite_or_beyond_limit`, 알려진 실패), 나머지 통과.
- Isaac 실행 형식: `cd ~/rl_ws/hdgp-iker && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm RUN_LABEL=<label> ../IsaacLab/_isaac_sim/python.sh <script> --headless 2>&1 | tee log/iker_r3/<name>.log`(Bash 타임아웃 600000).
- 커밋: 파일을 이름으로 `git add`(`-A`·`.` 금지), `--no-verify` 금지, `git commit` 은 `git add` 와 **다른** Bash 호출에서 한다. 메시지 끝 트레일러 두 줄:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` / `Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4`. 푸시하지 않는다.
- 작업 끝에 `git status --short` 로 만든 파일이 전부 커밋됐는지 본다.

---

### Task 1: 잡힘 판정 — 높이 상한·시작 자리 반경·엄지 조임

**Files:**
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/grasp_stage.py`
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_grasp_env.py`
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_grasp_env_cfg.py`
- Test: `source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_stage.py`

**Interfaces:**
- Produces:
  - `Stage1RewardCfg.lift_max_m: float = 0.15`, `.hold_xy_radius_m: float = 0.10`, `.thumb_curl_min_rad: float = 0.05`
  - `role_joint_index(joint_names: Sequence[str], role: str) -> int`
  - `closing_travel(joint_pos: torch.Tensor, open_pose: torch.Tensor, grip_pose: torch.Tensor) -> torch.Tensor`
  - `stage1_step(...)` 에 키워드 인자 `shoe_shift_xy: torch.Tensor`(N,), `thumb_curl: torch.Tensor`(N,) 추가
  - `IkerShoeGraspEnvCfg.thumb_curl_role = "thumb_3"`; env 속성 `_start_xy`(N, 2), `_thumb_curl_id: int`; 로그 키 `grasp/shift_xy`, `grasp/thumb_curl`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_grasp_stage.py` 의 `_step` 기본 입력에 두 줄을 더한다(`hand_speed_sum=torch.zeros(n),` 다음):

```python
        shoe_shift_xy=torch.zeros(n),
        thumb_curl=torch.full((n,), 0.5),
```

`test_success_waits_for_the_shoe_to_be_slow` 바로 뒤에 추가:

```python
def test_a_hold_counts_only_near_the_start_below_the_lift_ceiling_with_the_thumb_closing():
    state = gs.Stage1State.start(5)
    inputs = {
        **_held(5),
        "dz_free": torch.tensor([0.14, 0.16, 0.06, 0.06, 0.06]),
        "shoe_shift_xy": torch.tensor([0.09, 0.0, 0.11, 0.0, 0.0]),
        "thumb_curl": torch.tensor([0.06, 0.5, 0.5, 0.04, -0.3]),
    }
    step = _step(state, **inputs)
    assert step.held.tolist() == [True, False, False, False, False]


def test_a_hold_far_from_the_start_never_latches_or_succeeds():
    state = gs.Stage1State.start(1)
    for _ in range(25):
        step = _step(state, **_held(1), shoe_shift_xy=torch.tensor([0.5]))
        assert not step.held.any() and not step.success.any() and step.terms["lift_bonus"].item() == 0.0
        state = step.state


def test_reward_cfg_revision_3_defaults_and_rejections():
    cfg = gs.Stage1RewardCfg()
    assert (cfg.lift_max_m, cfg.hold_xy_radius_m, cfg.thumb_curl_min_rad) == (0.15, 0.10, 0.05)
    with pytest.raises(ValueError, match="lift_max_m"):
        gs.Stage1RewardCfg(lift_max_m=0.05)
    with pytest.raises(ValueError, match="hold_xy_radius_m"):
        gs.Stage1RewardCfg(hold_xy_radius_m=0.0)
    assert gs.Stage1RewardCfg(thumb_curl_min_rad=-1.0).thumb_curl_min_rad == -1.0  # negative switches the thumb condition off


def test_closing_travel_is_signed_by_the_closing_direction():
    open_pose, grip_pose = torch.tensor([0.0, 0.0]), torch.tensor([-1.8, 1.9])
    q = torch.tensor([[-0.3, 0.3], [0.95, -0.1]])
    assert gs.closing_travel(q, open_pose, grip_pose).tolist() == pytest.approx([[0.3, 0.3], [-0.95, -0.1]])
    with pytest.raises(ValueError, match="closing direction"):
        gs.closing_travel(q, open_pose, torch.tensor([0.0, 1.0]))


def test_role_joint_index_needs_exactly_one_match():
    names = ("l_hj_thumb_3", "l_hj_index_3", "l_hj_thumb_4")
    assert gs.role_joint_index(names, "thumb_3") == 0
    with pytest.raises(ValueError, match="matches 0"):
        gs.role_joint_index(names, "pinky_3")
    with pytest.raises(ValueError, match="matches 2"):
        gs.role_joint_index(("l_hj_thumb_3", "r_hj_thumb_3"), "thumb_3")
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_stage.py -q -p no:cacheprovider`
Expected: FAIL — `stage1_step() got an unexpected keyword argument 'shoe_shift_xy'`, `has no attribute 'closing_travel'` 등.

- [ ] **Step 3: `grasp_stage.py` 구현**

`Stage1RewardCfg` 에서 `success_speed: float = 0.05` 다음 줄에:

```python
    # learned-grasp spec §16 (revision 3): a hold counts only near the pick location, below a lift ceiling and with the thumb
    # closing; a negative thumb_curl_min_rad switches the thumb condition off (grasp_smoke's wiring check)
    lift_max_m: float = 0.15
    hold_xy_radius_m: float = 0.10
    thumb_curl_min_rad: float = 0.05
```

`validate` 에서 `lift_deadband_m` 검사 다음에:

```python
        if not self.lift_height_m < self.lift_max_m:
            raise ValueError(f"need lift_height_m {self.lift_height_m} < lift_max_m {self.lift_max_m}")
        if not self.hold_xy_radius_m > 0.0:
            raise ValueError(f"hold_xy_radius_m must be positive, got {self.hold_xy_radius_m}")
```

`frozen_hand_override` 앞에 새 함수를 두고, `frozen_hand_override` 의 루프 안 매칭 네 줄(`matches = ...` 부터 `value = float(open_pose[matches[0]])` 까지)과 `override[...]` 줄을 이것으로 바꾼다:

```python
def role_joint_index(joint_names: Sequence[str], role: str) -> int:
    """Index of the one hand joint named ``<side>_hj_<role>`` (``thumb_3``) — no side is spelled out."""
    matches = [i for i, name in enumerate(joint_names) if name.endswith(f"_hj_{role}")]
    if len(matches) != 1:
        raise ValueError(f"hand joint role {role!r} matches {len(matches)} joints of {list(joint_names)}")
    return matches[0]
```

`frozen_hand_override` 의 루프는 다음이 된다:

```python
    for role in roles:
        index = role_joint_index(names, role)
        value = float(open_pose[index])
        override[re.escape(names[index]) + "$"] = (value - halfwidth, value + halfwidth)
```

`normalized_targets` 다음에:

```python
def closing_travel(joint_pos: torch.Tensor, open_pose: torch.Tensor, grip_pose: torch.Tensor) -> torch.Tensor:
    """How far joints moved from the open pose toward the grip pose; negative = bent back past the open pose.

    Shapes broadcast (``joint_pos`` (N,) or (N, J) against (J,) or scalar poses). A joint whose open and grip poses coincide
    has no closing direction and is rejected.
    """
    direction = torch.sign(grip_pose - open_pose)
    if bool((direction == 0).any()):
        raise ValueError("closing_travel needs a closing direction: the open and grip poses coincide")
    return (joint_pos - open_pose) * direction
```

`stage1_step` 시그니처의 `palm_shoe_dist: torch.Tensor,` 다음에 두 인자를 넣고, 형상 검사 dict 와 held 를 바꾼다:

```python
    shoe_shift_xy: torch.Tensor,
    thumb_curl: torch.Tensor,
```

```python
    for name, value in dict(palm_gap=palm_gap, dz_free=dz_free, palm_shoe_dist=palm_shoe_dist, shoe_shift_xy=shoe_shift_xy,
                            thumb_curl=thumb_curl, rel_speed=rel_speed, shoe_speed=shoe_speed, q=q,
                            hand_floor_depth=hand_floor_depth, arm_speed_sum=arm_speed_sum, hand_speed_sum=hand_speed_sum).items():
        if value.shape != (n,):
            raise ValueError(f"{name} must be ({n},), got {tuple(value.shape)}")

    held = (
        (dz_free >= cfg.lift_height_m)
        & (dz_free <= cfg.lift_max_m)
        & (shoe_shift_xy <= cfg.hold_xy_radius_m)
        & (thumb_curl >= cfg.thumb_curl_min_rad)
        & (palm_shoe_dist <= cfg.hold_radius_m)
        & (rel_speed < cfg.hold_rel_speed)
    )
```

`stage1_step` docstring 은 `"""One policy step of the stage-1 reward (design §6, held narrowed by §16). All inputs are (N,) tensors."""` 로 바꾼다.

- [ ] **Step 4: 순수 테스트 통과 확인**

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_stage.py -q -p no:cacheprovider`
Expected: 전부 PASS(기존 `test_frozen_hand_override_pins_joint_roles_at_the_open_pose` 포함 — 오류 문구가 같다).

- [ ] **Step 5: env 배선**

`iker_shoe_grasp_env_cfg.py` 의 `frozen_hand_joints = ("thumb_2", "pinky_2")` 다음 줄에:

```python
    # learned-grasp spec §16: a hold needs this joint's closing travel >= grasp_reward.thumb_curl_min_rad (a thumb pressed
    # back against its backstop by the other four fingers is not a grasp)
    thumb_curl_role = "thumb_3"
```

`iker_shoe_grasp_env.py`:

1. 모듈 docstring 의 `Success = the shoe held 5 cm above its start for 20 steps.` 를 `Success = the shoe held 5-15 cm above and within 10 cm of its start, the thumb closing, for 20 steps (§6, §16).` 로 바꾼다.
2. `__init__` 의 `self._palmar_axis = torch.tensor(PALMAR_AXIS_LOCAL, device=dev).expand(n, 3)` 다음에:

```python
        curl = gs.role_joint_index(prof.hand_joint_names, cfg.thumb_curl_role)
        self._thumb_curl_id = int(self._hand_ids[curl])
        self._thumb_open = torch.tensor(float(prof.hand_open_pose[curl]), device=dev)
        self._thumb_grip = torch.tensor(float(prof.hand_grip_pose[curl]), device=dev)
        gs.closing_travel(self._thumb_open, self._thumb_open, self._thumb_grip)  # boot error when the role has no closing direction
```

3. 무행동 수입 검사 호출의 키 튜플을 `("palm_gap", "dz_free", "shoe_shift_xy", "thumb_curl", "rel_speed", "shoe_speed", "q", "hand_floor_depth", "arm_speed_sum", "hand_speed_sum")` 로 바꾼다.
4. `self._start_bottom_z = torch.zeros(n, device=dev)` 다음 줄에 `self._start_xy = torch.zeros(n, 2, device=dev)  # env-local shoe centre at the episode start (§16)`.
5. `_get_dones` 에서 `shoe_vel = self._shoe.data.root_lin_vel_w` 다음에:

```python
        shift_xy = (shoe_pos[:, :2] - self._start_xy).norm(dim=-1)
        thumb_curl = gs.closing_travel(self._robot.data.joint_pos[:, self._thumb_curl_id], self._thumb_open, self._thumb_grip)
```

   `gs.stage1_step(` 호출의 `palm_shoe_dist=(palm_pos - shoe_pos).norm(dim=-1),` 다음에 `shoe_shift_xy=shift_xy,` 와 `thumb_curl=thumb_curl,` 를 넣는다. 로그 dict 의 `"grasp/dz_free": ...` 다음에 `"grasp/shift_xy": shift_xy.mean().item(),` 와 `"grasp/thumb_curl": thumb_curl.mean().item(),` 를 넣는다.
6. `restore_success_captures` 의 `shoe_pose = self._capture.shoe_pose[env_ids].clone()` 다음 줄에 `self._start_xy[env_ids] = shoe_pose[:, :2]`.
7. `_reset_idx` 의 `shoe_pose[:, :2] += sample_uniform(...)` 다음 줄에 `self._start_xy[env_ids] = shoe_pose[:, :2]`(원점 더하기 전).

Run: `cd ~/rl_ws/hdgp-iker && python3 -m py_compile source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_grasp_env.py source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_grasp_env_cfg.py && grep -n "shoe_shift_xy\|thumb_curl\|_start_xy" source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_grasp_env.py`
Expected: 컴파일 오류 없음, 위 7 곳이 모두 보인다. Isaac 검증은 태스크 3 이 한다.

- [ ] **Step 6: 전체 순수 테스트**

Run: 전역 제약의 순수 테스트 명령.
Expected: 기준선과 같다 — 1 failed(알려진 뱅크 파일 테스트), 나머지 통과.

- [ ] **Step 7: 커밋**

```bash
git add source/openarm/openarm/agnostic/tasks/iker_shoe/grasp_stage.py source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_grasp_env.py source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_grasp_env_cfg.py source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_stage.py
```

(다른 Bash 호출에서)

```bash
git commit -m "feat(iker): 1단계 잡힘 판정 개정 3 — 자유 들기 5~15 cm·시작 자리 10 cm·엄지 조임 0.05 rad

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4"
```

---

### Task 2: thumb_3 관절 한계(backstop)와 부팅 대조 키

**Files:**
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/grasp_stage.py`
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/robot.py`
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_env_cfg.py`
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_env.py`
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_grasp_env.py`
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/grasp_bank.py`
- Test: `source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_stage.py`, `source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_bank.py`

**Interfaces:**
- Consumes: `grasp_stage.role_joint_index` (Task 1).
- Produces:
  - `grasp_stage.backstop_limits(joint_names, hard_lo: torch.Tensor, hard_hi: torch.Tensor, open_pose: Sequence[float], grip_pose: Sequence[float], roles: Sequence[str]) -> dict[str, tuple[float, float]]`
  - `robot.apply_hand_backstop(articulation, roles: Sequence[str]) -> dict[str, list[float]]`
  - `IkerShoeEnvCfg.hand_backstop_joints = ("thumb_3",)` (1단계 cfg 가 상속)
  - `grasp_bank.LEARNED_BOOT_KEYS = BOOT_METADATA_KEYS + ("hand_backstop",)`; 1단계 env `_boot_metadata["hand_backstop"]`, 2단계 env 부팅 대조 `expected["hand_backstop"]`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_grasp_stage.py` 의 `test_stage1_hand_limits_keep_the_profile_narrowing_on_frozen_joints` 다음에:

```python
def test_backstop_limits_move_the_opening_side_limit_to_the_open_pose():
    names = ("l_hj_thumb_3", "l_hj_index_3")
    lo, hi = torch.tensor([-1.5708, -1.5708]), torch.tensor([1.5708, 1.5708])
    open_pose, grip_pose = (0.0, 0.0), (-1.8, 1.8)
    thumb = gs.backstop_limits(names, lo, hi, open_pose, grip_pose, ("thumb_3",))
    assert list(thumb) == ["l_hj_thumb_3"] and thumb["l_hj_thumb_3"] == pytest.approx((-1.5708, 0.0))
    index = gs.backstop_limits(names, lo, hi, open_pose, grip_pose, ("index_3",))
    assert index["l_hj_index_3"] == pytest.approx((0.0, 1.5708))
    with pytest.raises(ValueError, match="matches 0"):
        gs.backstop_limits(names, lo, hi, open_pose, grip_pose, ("pinky_3",))
    with pytest.raises(ValueError, match="closing direction"):
        gs.backstop_limits(names, lo, hi, open_pose, (0.0, 1.8), ("thumb_3",))
    with pytest.raises(ValueError, match="outside"):
        gs.backstop_limits(names, lo, hi, (2.0, 0.0), grip_pose, ("thumb_3",))
```

`tests/test_grasp_bank.py`: `BOOT = {key: index for index, key in enumerate(gb.BOOT_METADATA_KEYS)}` 를 `BOOT = {key: index for index, key in enumerate(gb.LEARNED_BOOT_KEYS)}` 로 바꾸고, `test_learned_bank_metadata_keeps_the_boot_keys_and_records_the_origin` 의 `assert {key: meta[key] for key in gb.BOOT_METADATA_KEYS} == BOOT` 를 `assert {key: meta[key] for key in gb.LEARNED_BOOT_KEYS} == BOOT` 로 바꾼 뒤, 그 테스트 끝에 추가:

```python
    nine_keys = {key: BOOT[key] for key in gb.BOOT_METADATA_KEYS}  # a boot without the hand backstop (spec §16)
    with pytest.raises(ValueError, match="hand_backstop"):
        gb.learned_bank_metadata(nine_keys, side_sign=-1.0, checkpoint="c", checkpoint_sha256="s", stage1_reward={},
                                 seeds=(0,), captured=1, verified=1)
    assert gb.LEARNED_BOOT_KEYS == gb.BOOT_METADATA_KEYS + ("hand_backstop",)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_stage.py source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_bank.py -q -p no:cacheprovider`
Expected: FAIL — `has no attribute 'backstop_limits'`, `has no attribute 'LEARNED_BOOT_KEYS'`.

- [ ] **Step 3: 순수 구현**

`grasp_stage.py` 의 `stage1_hand_limits` 다음에:

```python
def backstop_limits(
    joint_names: Sequence[str],
    hard_lo: torch.Tensor,
    hard_hi: torch.Tensor,
    open_pose: Sequence[float],
    grip_pose: Sequence[float],
    roles: Sequence[str],
) -> dict[str, tuple[float, float]]:
    """Joint position limits that keep the hand joint ``roles`` from bending back past the profile's open pose (spec §16):
    the limit opposite each role's closing direction (open -> grip) moves to the open pose, the other stays hard.

    ``joint_names`` are the profile's hand joints, ``hard_lo``/``hard_hi`` their (J,) simulator limits, the poses (J,)
    profile values. Returns ``{joint name: (lo, hi)}`` in ``roles`` order.
    """
    names = list(joint_names)
    if hard_lo.shape != (len(names),) or hard_hi.shape != (len(names),):
        raise ValueError(f"limits must be ({len(names)},), got {tuple(hard_lo.shape)} and {tuple(hard_hi.shape)}")
    limits = {}
    for role in roles:
        index = role_joint_index(names, role)
        lo, hi = float(hard_lo[index]), float(hard_hi[index])
        q_open, q_grip = float(open_pose[index]), float(grip_pose[index])
        if q_grip == q_open:
            raise ValueError(f"{names[index]} has no closing direction: its open and grip poses are both {q_open}")
        if not lo <= q_open <= hi:
            raise ValueError(f"{names[index]} open pose {q_open} lies outside its limits [{lo}, {hi}]")
        limits[names[index]] = (lo, q_open) if q_grip < q_open else (q_open, hi)
    return limits
```

`grasp_bank.py` 의 `BOOT_METADATA_KEYS = (...)` 다음에:

```python
# A learned bank also records the thumb backstop its stage-1 environment ran with (learned-grasp spec §16); the pre-grasp
# bank keeps the nine keys — its hand is open, clear of the backstop.
LEARNED_BOOT_KEYS = BOOT_METADATA_KEYS + ("hand_backstop",)
```

`learned_bank_metadata` 안의 `BOOT_METADATA_KEYS` 세 곳(`missing`·`extra`·오류 문구, 반환 dict 의 `**{key: boot[key] for key in ...}`)을 모두 `LEARNED_BOOT_KEYS` 로 바꾼다.

Run: Step 2 명령. Expected: PASS.

- [ ] **Step 4: 한계를 시뮬레이터에 쓰는 함수**

`robot.py` 의 import 에 `from typing import Sequence`, `import torch` 를 더하고 `from . import layout` 를 `from . import grasp_stage, layout` 로 바꾼다. 파일 끝에:

```python
def apply_hand_backstop(articulation, roles: Sequence[str]) -> dict[str, list[float]]:
    """Write the learned-grasp spec §16 backstop (``grasp_stage.backstop_limits``) into the simulator and return it as boot
    metadata, ``{joint: [lo, hi]}`` rounded to 4 decimals. A runtime-written limit holds against the joint's own drive
    (probe 2026-09-15: thumb_3 target 0.8 rad past the open pose, measured -0.0003 rad)."""
    prof = profile()
    names = list(articulation.data.joint_names)
    hand = [names.index(name) for name in prof.hand_joint_names]
    hard = articulation.data.joint_pos_limits[0, hand]
    limits = grasp_stage.backstop_limits(prof.hand_joint_names, hard[:, 0], hard[:, 1], prof.hand_open_pose, prof.hand_grip_pose, roles)
    for name, bounds in limits.items():
        value = torch.tensor(bounds, device=articulation.device).repeat(articulation.num_instances, 1, 1)
        articulation.write_joint_position_limit_to_sim(value, joint_ids=[names.index(name)])
    return {name: [round(lo, 4), round(hi, 4)] for name, (lo, hi) in limits.items()}
```

- [ ] **Step 5: 두 env 배선**

`iker_shoe_env_cfg.py` 의 `eval_success_distance_m = 0.05` 다음에:

```python
    # learned-grasp spec §16: these hand joint roles cannot bend back past the profile's open pose (both stages)
    hand_backstop_joints = ("thumb_3",)
```

`iker_shoe_env.py`(2단계):
1. `self._palm_jacobian = self._palm - 1  # ...` 다음 줄에 `backstop = robot.apply_hand_backstop(self._robot, cfg.hand_backstop_joints)`.
2. `expected = {` dict 의 `"scene_config": vars(scene_config),` 다음에 `"hand_backstop": backstop,`.
3. 모듈 docstring 의 `the hand keeps the bank's commanded targets.` 를 `the hand keeps the bank's commanded targets, with the stage-1 thumb backstop (learned-grasp spec §16).` 로 바꾼다.

`iker_shoe_grasp_env.py`(1단계):
1. `hard = self._robot.data.joint_pos_limits[0, self._hand_ids]` **바로 앞** 줄에 `backstop = robot.apply_hand_backstop(self._robot, cfg.hand_backstop_joints)  # before the action limits read the hard limits`.
2. `self._boot_metadata = expected  # harvest_grasp_bank.py writes these comparison keys into the learned bank` 를 다음으로 바꾼다:

```python
        # harvest_grasp_bank.py writes these comparison keys into the learned bank; the pre-grasp bank above is compared
        # without the backstop (grasp_bank.LEARNED_BOOT_KEYS)
        self._boot_metadata = {**expected, "hand_backstop": backstop}
```

Run: `cd ~/rl_ws/hdgp-iker && python3 -m py_compile source/openarm/openarm/agnostic/tasks/iker_shoe/robot.py source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_env.py source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_grasp_env.py source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_env_cfg.py && grep -rn "l_hj_\|r_hj_" source/openarm/openarm/agnostic/tasks/iker_shoe/robot.py source/openarm/openarm/agnostic/tasks/iker_shoe/grasp_stage.py source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_env.py source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_grasp_env.py`
Expected: 컴파일 오류 없음, grep 출력 없음(측면 리터럴 없음).

- [ ] **Step 6: 전체 순수 테스트**

Run: 전역 제약의 순수 테스트 명령. Expected: 기준선과 같다(1 failed = 알려진 뱅크 파일 테스트).

- [ ] **Step 7: 커밋**

```bash
git add source/openarm/openarm/agnostic/tasks/iker_shoe/grasp_stage.py source/openarm/openarm/agnostic/tasks/iker_shoe/robot.py source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_env_cfg.py source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_env.py source/openarm/openarm/agnostic/tasks/iker_shoe/iker_shoe_grasp_env.py source/openarm/openarm/agnostic/tasks/iker_shoe/grasp_bank.py source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_stage.py source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_bank.py
```

(다른 Bash 호출에서)

```bash
git commit -m "feat(iker): thumb_3 역굴곡 관절 한계(펴진 자세) — 1·2단계 env 부팅, 뱅크 부팅 대조 hand_backstop

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4"
```

---

### Task 3: Isaac 스모크 — 1단계 한계·판정 배선, 2단계 무행동 미끄럼

**Files:**
- Modify: `scripts/iker/grasp_smoke.py`
- Modify: `scripts/iker/env_smoke.py`

**Interfaces:**
- Consumes: Task 1 cfg 필드·env 속성, Task 2 `_boot_metadata["hand_backstop"]`·`robot.profile()`.
- Produces: `GRASP SMOKE passed True` 가 새 검사 7·8 을 포함한다. `env_smoke.py` 검사 2 출력 줄 `SMOKE zero action 5 s: shoe slip in palm mm q10/50/90 [...], lost fraction X; palm dz mm q10/50/90 [...] (arm sag, reported only)`. 이 두 문구를 파싱하는 코드는 저장소에 없다(`grep -rn "dropped fraction" source scripts` 는 env_smoke.py 만).

- [ ] **Step 1: `grasp_smoke.py` 수정**

1. docstring 목록 끝(6 번 다음)에:

```
7. the thumb backstop (learned-grasp spec §16): with its target 0.8 rad past the open pose and the action path bypassed,
   thumb_3 stays at its open-pose limit, and the simulator limit equals the boot metadata;
8. with the default thumb condition, the forced hold of an open hand is never held (checks 4 and 6 run with the xy radius
   and the thumb condition switched off — their bounds are pure-test territory).
```

2. import 에 `from dataclasses import replace` 와 `from openarm.agnostic.tasks.iker_shoe import robot` 를 더한다(`# noqa: E402` 유지).
3. 상수에 추가:

```python
FORCED_THUMB_CURL_MIN_RAD = -1.0  # switches the thumb condition off for the wiring checks
BACKSTOP_PUSH_RAD = 0.8
BACKSTOP_STEPS = 20
BACKSTOP_SLACK_RAD = 0.02  # probe 2026-09-15 measured 0.0003 rad past the limit
```

4. `main` 의 `cfg.grasp_reward.hold_radius_m = FORCED_HOLD_RADIUS_M` 다음에:

```python
    cfg.grasp_reward.hold_xy_radius_m = FORCED_HOLD_RADIUS_M  # the forced hold shifts the shoe 15 cm (clear of the hand)
    cfg.grasp_reward.thumb_curl_min_rad = FORCED_THUMB_CURL_MIN_RAD
```

5. `forced_hold` 다음에 함수 추가:

```python
def backstop_check(env) -> tuple[str, float, list[float], list[float]]:
    """(joint, worst travel past the open pose in the opening direction, simulator limit, boot metadata limit) after
    ``BACKSTOP_STEPS`` policy steps with the joint's target ``BACKSTOP_PUSH_RAD`` past the open pose."""
    prof = robot.profile()
    backstop = env._boot_metadata["hand_backstop"]
    name = next(iter(backstop))
    k = prof.hand_joint_names.index(name)
    j = list(env._robot.data.joint_names).index(name)
    open_q, grip_q = float(prof.hand_open_pose[k]), float(prof.hand_grip_pose[k])
    opening = -1.0 if grip_q > open_q else 1.0
    env.reset()
    env._pre_physics_step = lambda actions: None  # hold the written targets; the hand law would pull them back into range
    try:
        env._joint_targets[:] = env._robot.data.joint_pos_target
        env._joint_targets[:, j] = open_q + opening * BACKSTOP_PUSH_RAD
        for _ in range(BACKSTOP_STEPS):
            env.step(torch.zeros(env.num_envs, env.cfg.action_space, device=env.device))
    finally:
        del env._pre_physics_step
    past = float(((env._robot.data.joint_pos[:, j] - open_q) * opening).max())
    return name, past, env._robot.data.joint_pos_limits[0, j].tolist(), backstop[name]
```

6. 성공 캡처 검사(검사 6) 블록 다음, `for failure in failures:` 앞에:

```python
    name, past, sim_limit, meta_limit = backstop_check(env)
    print(f"SMOKE thumb backstop: {name} worst travel past the open pose {past:+.4f} rad, simulator limit "
          f"{[round(v, 4) for v in sim_limit]}, boot metadata {meta_limit}", flush=True)
    if past > BACKSTOP_SLACK_RAD or [round(v, 4) for v in sim_limit] != meta_limit:
        failures.append(f"thumb backstop: {past:+.4f} rad past the open pose or simulator limit {sim_limit} != {meta_limit}")

    env._reward_cfg = replace(env._reward_cfg, thumb_curl_min_rad=gs.Stage1RewardCfg().thumb_curl_min_rad)
    env.reset()
    latch_calls, success_calls, _ = forced_hold(env, 5)
    print(f"SMOKE open thumb under the default thumb condition ({env._reward_cfg.thumb_curl_min_rad} rad): "
          f"latch calls {latch_calls}, success calls {success_calls}", flush=True)
    if latch_calls or success_calls:
        failures.append("a forced hold with the open thumb latched under the default thumb condition")
```

- [ ] **Step 2: `env_smoke.py` 검사 2 수정**

1. docstring 2 번을 `2. with zero actions for 5 s the shoe stays in the hand: its position in the palm frame moves < 3 cm (the arm's zero-action sag is reported, not judged — relative IK does not undo it, auto-loop spec §13);` 로 바꾼다.
2. import 에 `from isaaclab.utils.math import quat_apply_inverse, quat_from_matrix  # noqa: E402` (기존 `quat_from_matrix` import 줄을 이것으로 바꾼다).
3. 상수 `MAX_SLIP_M = 0.03` 을 `MAX_DROP_FRACTION_ZERO_ACTION` 다음에 둔다.
4. 함수 추가(`reach_slots` 앞):

```python
def shoe_in_palm(env) -> tuple[torch.Tensor, torch.Tensor]:
    """((N, 3) shoe position in the palm frame, (N,) env-local palm height)."""
    palm = env._robot.data.body_pose_w[:, env._palm]
    return quat_apply_inverse(palm[:, 3:7], env._shoe.data.root_pos_w - palm[:, :3]), palm[:, 2] - env.scene.env_origins[:, 2]


def quantiles_mm(values: torch.Tensor) -> list[int]:
    return [round(float(v) * 1000) for v in torch.quantile(values.float(), torch.tensor([0.1, 0.5, 0.9], device=values.device))]
```

5. `main` 의 기존 블록(`z0 = env._shoe.data.root_pos_w[:, 2].clone()` 부터 `failures.append(f"zero-action drop fraction {dropped:.2f}")` 까지)을 이것으로 바꾼다:

```python
    env.step(torch.zeros(n, 6, device=dev))  # a pose written at reset is read back only after a physics step
    rel0, palm_z0 = shoe_in_palm(env)
    for _ in range(49):
        env.step(torch.zeros(n, 6, device=dev))
    rel, palm_z = shoe_in_palm(env)
    slip = (rel - rel0).norm(dim=-1)
    lost = float((slip > MAX_SLIP_M).float().mean())
    print(f"SMOKE zero action 5 s: shoe slip in palm mm q10/50/90 {quantiles_mm(slip)}, lost fraction {lost:.2f}; "
          f"palm dz mm q10/50/90 {quantiles_mm(palm_z - palm_z0)} (arm sag, reported only)", flush=True)
    if not lost <= MAX_DROP_FRACTION_ZERO_ACTION:
        failures.append(f"zero-action lost fraction {lost:.2f}")
```

Run: `cd ~/rl_ws/hdgp-iker && python3 -m py_compile scripts/iker/grasp_smoke.py scripts/iker/env_smoke.py && grep -rn "dropped fraction" source scripts`
Expected: 컴파일 오류 없음, grep 출력 없음.

- [ ] **Step 3: 1단계 Isaac 스모크(합격 필수)**

Run: `mkdir -p ~/rl_ws/hdgp-iker/log/iker_r3` 후 전역 제약의 Isaac 실행 형식으로 `RUN_LABEL=iker_r3_grasp_smoke`, 스크립트 `scripts/iker/grasp_smoke.py`, 로그 `log/iker_r3/grasp_smoke.log`.
Expected:
- 부팅 손 한계 표에서 thumb_3 행이 `-1.571..+0.000`
- `SMOKE thumb backstop: ... worst travel past the open pose` 값 ≤ +0.02, simulator limit == boot metadata
- `SMOKE open thumb under the default thumb condition (0.05 rad): latch calls [], success calls []`
- 기존 검사 줄 전부, 마지막 `GRASP SMOKE passed True`
실패하면 고치고 다시 돌린다. 실패 원인이 이 계획의 설계(한계·판정 값) 자체라면 고치지 말고 BLOCKED 로 보고한다.

- [ ] **Step 4: 2단계 Isaac 스모크(배선 확인, 합격은 요구하지 않음)**

아직 한계를 기록한 학습 뱅크가 없으므로, 첫 루프 뱅크에 한계 metadata 만 더한 스크래치 사본으로 부팅·검사 2 배선을 본다. 이 뱅크는 공중 옆잡기라 도달 검사(5)는 실패가 예상이다.

```bash
cd ~/rl_ws/hdgp-iker && PYTHONPATH=source/openarm python3 - <<'EOF'
import torch
from openarm.agnostic.modules import robot_profiles
from openarm.agnostic.modules.iker import run_files
from openarm.agnostic.tasks.iker_shoe import grasp_stage as gs, layout
prof = robot_profiles.PROFILES[layout.PROFILE_NAME]
doc = run_files.read_json(layout.RUNS_DIR / "config_00" / "grasp_bank.json")
j = len(prof.hand_joint_names)
hard = torch.full((j,), -1.5708), torch.full((j,), 1.5708)  # USD limits of the DG-5F distal joints (probe 2026-09-15)
limits = gs.backstop_limits(prof.hand_joint_names, *hard, prof.hand_open_pose, prof.hand_grip_pose, ("thumb_3",))
doc["metadata"]["hand_backstop"] = {name: [round(lo, 4), round(hi, 4)] for name, (lo, hi) in limits.items()}
from pathlib import Path
print(run_files.write_json(Path("/tmp/claude-1000/-home-user-rl-ws/5ca9e4f2-e022-4266-b0ca-84d6cbf13515/scratchpad/r3_bank_with_backstop.json"), doc))
EOF
```

그다음 Isaac 실행 형식으로 `RUN_LABEL=iker_r3_env_smoke`, 스크립트 `scripts/iker/env_smoke.py --grasp-bank /tmp/claude-1000/-home-user-rl-ws/5ca9e4f2-e022-4266-b0ca-84d6cbf13515/scratchpad/r3_bank_with_backstop.json --interaction iker_runs/shoe_place/config_00/loop/stage_01/interaction_vlm.json`, 로그 `log/iker_r3/env_smoke_midair_bank.log`.
Expected: 부팅 대조 통과(`grasp bank was made with different settings` 오류 없음), 새 형식의 `SMOKE zero action 5 s: shoe slip in palm ...` 줄, 이어지는 검사 줄들, `SMOKE passed` 줄(True/False 어느 쪽이든). 보고에 SMOKE 줄 전부를 그대로 적는다.

- [ ] **Step 5: 커밋**

```bash
git add scripts/iker/grasp_smoke.py scripts/iker/env_smoke.py
```

(다른 Bash 호출에서)

```bash
git commit -m "test(iker): 스모크 개정 — 1단계 엄지 한계·조임 판정 배선, 2단계 무행동은 손안 미끄럼으로 판정

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4"
```

---

### Task 4: 첫 루프 산출물 보존과 재시작 준비

**Files:**
- Move: `iker_runs/shoe_place/config_00/loop/` → `iker_runs/shoe_place/config_00/archive/2026-09-14_midair/loop/`
- Move: `iker_runs/shoe_place/config_00/grasp_bank.json` → `iker_runs/shoe_place/config_00/archive/2026-09-14_midair/grasp_bank.json`
- Move: `iker_runs/shoe_place/config_00/grasp_quality_calibration.json` → `iker_runs/shoe_place/config_00/archive/2026-09-14_midair/grasp_quality_calibration.json`
- Modify: `source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_bank_file.py`
- Modify: `scripts/iker/LOOP_PROMPT.md:36`

**Interfaces:**
- Consumes: Task 3 까지의 커밋(이 태스크는 코드 동작을 바꾸지 않는다).
- Produces: 새 루프가 옛 보정·뱅크를 읽지 않는 `config_00`; 뱅크가 없으면 skip 하는 뱅크 파일 테스트.

- [ ] **Step 1: 이동 전 확인**

Run: `cd ~/rl_ws/hdgp-iker && git status --short && ls iker_runs/shoe_place/config_00/ && for pid in $(pgrep -f "_isaac_sim/kit/python/bin/python3" || true); do tr '\0' '\n' < /proc/$pid/environ | grep '^RUN_LABEL='; done`
Expected: 작업 트리 깨끗, `grasp_bank.json`·`grasp_quality_calibration.json`·`loop` 가 있고, `RUN_LABEL=iker_` 로 시작하는 Isaac 프로세스가 없다(있으면 멈추고 보고).

- [ ] **Step 2: 뱅크 파일 테스트가 없는 뱅크에서 skip 하게 한다(실패 테스트 먼저)**

Run: `cd ~/rl_ws/hdgp-iker && mkdir -p iker_runs/shoe_place/config_00/archive/2026-09-14_midair && git mv iker_runs/shoe_place/config_00/grasp_bank.json iker_runs/shoe_place/config_00/archive/2026-09-14_midair/grasp_bank.json && PYTHONPATH=source/openarm python3 -m pytest source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_bank_file.py -q -p no:cacheprovider`
Expected: 2 failed — `FileNotFoundError`(뱅크가 없으면 오류로 실패한다).

`test_grasp_bank_file.py` 의 import 다음에:

```python
BANK_PATH = layout.RUNS_DIR / "config_00" / "grasp_bank.json"


def _read_bank() -> dict:
    if not BANK_PATH.is_file():
        pytest.skip(f"{BANK_PATH} does not exist yet — the auto loop harvests it (auto-loop spec §13)")
    return run_files.read_json(BANK_PATH)
```

두 테스트의 `doc = run_files.read_json(layout.RUNS_DIR / "config_00" / "grasp_bank.json")` 를 `doc = _read_bank()` 로 바꾼다.

Run: 같은 pytest 명령. Expected: 2 skipped.

- [ ] **Step 3: 나머지 이동**

Run: `cd ~/rl_ws/hdgp-iker && git mv iker_runs/shoe_place/config_00/grasp_quality_calibration.json iker_runs/shoe_place/config_00/archive/2026-09-14_midair/grasp_quality_calibration.json && git mv iker_runs/shoe_place/config_00/loop iker_runs/shoe_place/config_00/archive/2026-09-14_midair/loop && ls iker_runs/shoe_place/config_00/ iker_runs/shoe_place/config_00/archive/2026-09-14_midair/`
Expected: `config_00` 에 `pregrasp_bank.json`·`interaction_human.json`·`keypoints.json`·`snapshot*.png`·`archive` 가 남고, 보존 폴더에 `grasp_bank.json`·`grasp_quality_calibration.json`·`loop/` 가 있다.

`scripts/iker/LOOP_PROMPT.md` 36 행의 `(track iker_shoe_c00)` 을 `(track iker_shoe_c00_r2)` 로 바꾼다(크론 프롬프트 예시 — 첫 루프는 `archive/2026-09-14_midair/loop` 에 보존).

- [ ] **Step 4: 전체 순수 테스트**

Run: 전역 제약의 순수 테스트 명령.
Expected: **0 failed**, 뱅크 파일 테스트 2 skipped.

- [ ] **Step 5: 커밋**

```bash
git add source/openarm/openarm/agnostic/tasks/iker_shoe/tests/test_grasp_bank_file.py scripts/iker/LOOP_PROMPT.md
git status --short   # the three git mv moves are already staged (R lines); nothing else may be listed
```

(다른 Bash 호출에서)

```bash
git commit -m "chore(iker): 첫 루프(공중 파지) 산출물 보존 — archive/2026-09-14_midair, 뱅크 파일 테스트는 수확 전 skip

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Hqg9n53yi9x4qtfFzXRMi4"
```

Run: `git status --short`. Expected: 출력 없음.

---

## 계획 뒤 운영(태스크 아님 — 사용자 확인 뒤 컨트롤러가 한다)

1. 1단계 A FRESH 기동: `RUN_LABEL=iker_grasp_c00_r2_a`, `loop.py` 의 `TRAIN_SCRIPT`·`STAGE1_TASK`·`STAGE1_NUM_ENVS` 와 같은 명령(`--task … --num_envs … --headless`), 로그 `log/rl_games/open-sens/left/train_iker_grasp_c00_r2_a.log`.
2. `loop.py init --track iker_shoe_c00_r2 --policy '{"labels": {"stage1_a": "iker_grasp_c00_r2_a", "stage1_b": "iker_grasp_c00_r2_b", "stage2": "iker_vlm_c00_r2_s1"}}' --adopt stage1_a`, 상태 파일 커밋.
3. 크론 `7,37 * * * *` 프롬프트 "IKER 자동 루프 틱: ~/rl_ws/hdgp-iker/scripts/iker/LOOP_PROMPT.md 절차대로 한 틱을 수행한다 (track iker_shoe_c00_r2)".

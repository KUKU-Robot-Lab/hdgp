"""grasp_s2r 계약 테스트 — Isaac Sim 없이 돈다(소스 텍스트·AST 검사).

cfg 는 isaaclab→pxr 를 끌어와 Isaac 앱 없이 import 가 안 되므로, 값이 아니라
**소스의 계약**을 잠근다. 각 테스트에 왜 이 계약이 생겼는지(어떤 사고를 막는지)를
적어 둔다 — 나중에 고칠 때 근거 없이 지우지 않도록.

실행:
    cd hdgp && PYTHONPATH=source/openarm python3 -m pytest \
        source/openarm/openarm/agnostic/tasks/grasp_s2r/tests -q
"""

from __future__ import annotations

import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
_ENV = (_HERE / "grasp_s2r_env.py").read_text(encoding="utf-8")
_CTRL = (_HERE / "grasp_s2r_control.py").read_text(encoding="utf-8")
_CFG = (_HERE / "grasp_s2r_env_cfg.py").read_text(encoding="utf-8")
_REW = (_HERE / "grasp_s2r_rewards.py").read_text(encoding="utf-8")
_REG = (_HERE / "config" / "__init__.py").read_text(encoding="utf-8")


def _code(src: str) -> str:
    """주석·docstring 을 뺀 실행 코드만.

    ★설명문에 적힌 이름이 계약을 통과시키면 안 된다 — 예컨대 "기각된 tip_cyl 분기를
      뺐다"는 **주석**이 "tip_cyl 이 없어야 한다"는 계약을 깨뜨리는 식이다.
    """
    import ast
    tree = ast.parse(src)
    doc_lines: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                and isinstance(first.value.value, str):
            doc_lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    out = []
    for i, line in enumerate(src.split("\n"), start=1):
        if i in doc_lines:
            continue
        s = line.split("#", 1)[0]
        if s.strip():
            out.append(s)
    return "\n".join(out)


def _assign_block(src: str, name: str) -> str:
    """`name = (` 로 시작하는 다중행 대입의 본문 — 괄호 균형으로 끝을 찾는다."""
    m = re.search(rf"\n\s*{re.escape(name)} = \(", src)
    assert m, f"{name} 대입 부재"
    i = src.index("(", m.start())
    depth, j = 0, i
    while j < len(src):
        if src[j] == "(":
            depth += 1
        elif src[j] == ")":
            depth -= 1
            if depth == 0:
                return src[i + 1:j]
        j += 1
    raise AssertionError(f"{name} 대입의 괄호가 안 닫힌다")


# ---------------------------------------------------------------- 등록
def test_four_task_ids_registered():
    """id 4종(train/play × mlp/lstm). `-play` 가 없으면 play.py·warm 수집이 죽는다."""
    assert '_ENTRY = "openarm.agnostic.tasks.grasp_s2r.grasp_s2r_env:GraspS2REnv"' in _REG
    for suffix in ('""', '"-play"', '"-lstm"', '"-play-lstm"'):
        assert suffix in _REG, f"태스크 id 접미사 {suffix} 미등록"
    assert 'f"open-{_tag}_grasp_s2r{_suffix}"' in _REG
    # 로그 분리 규약: 두 번째 슬롯이 r/l/b 여야 한다.
    assert re.search(r'"sens_r":\s*GraspS2RTesolloRightEnvCfg', _REG)


def test_registration_skips_profiles_without_fabric():
    """자산 없는 프로필을 조용히 빠뜨리지 않고 사유를 남긴다(record-loud)."""
    assert "SKIPPED" in _REG and "REGISTERED" in _REG
    assert "fabric_class is None" in _code(_REG)


# ---------------------------------------------------------------- 액션
def test_palm_action_is_home_relative_delta():
    """★palm 액션은 **홈 기준 델타**여야 한다 — `a=0` 이 홈.

    절대 매핑(`a=0` = 박스 중심)은 저장소 공통 σ=1.0 과 곱해지면 매 스텝 작업공간
    전역에서 목표를 재추첨해 접근이 랜덤워크가 된다(선행 트랙 실측: 클램프 전
    지령 요청량 0.33~0.36 m/step 상시 포화).
    """
    code = _code(_ENV)
    assert "self._home_palm.unsqueeze(0) + delta" in code, "palm 지령이 홈 기준 델타가 아니다"
    assert "palm_delta_xyz" in _code(_CFG) and "palm_delta_rot_deg" in _code(_CFG)
    # 홈이 박스에 잘리면 a=0 의 의미가 깨진다.
    assert "torch.minimum(self._palm_lo, self._home_palm)" in code


def test_command_rate_limiter_logs_preclamp_value():
    """클램프 **전** 원값을 남겨야 상한이 물리는 비율을 알 수 있다(reward-clamp 규칙)."""
    code = _code(_ENV)
    assert "_palm_cmd_step_raw" in code
    assert "palm_cmd_rate_limit_m" in code and "palm_cmd_rate_limit_rot_deg" in code
    # 리셋 직후 첫 지령은 초기화라 리미터를 걸지 않는다.
    assert "_palm_cmd_primed[env_ids] = False" in code


# ---------------------------------------------------------------- 래치
def test_latch_never_overrides_arm_command():
    """★래치는 **보상 단계 표시 전용**이다.

    grasp_v1 은 래치 후 팔 지령을 z 램프 스크립트로 대체했다(`torch.where(is_lift,
    _lift_palm, palm_pose)`). 이 트랙은 이송까지 정책이 fabric 으로 제어하므로
    그 오버라이드가 있으면 안 된다.
    """
    code = _code(_ENV)
    for banned in ("_lift_palm", "lift_start_step", "LIFT_PHASE_STEPS", "lift_height_delta"):
        assert banned not in code, f"래치 스크립트 잔재: {banned}"
    # palm_targets 를 쓰는 곳은 액션 매핑·리미터뿐이어야 한다.
    assert code.count("self.palm_targets =") == 1
    # 래치 자체는 살아 있어야 한다(보상 게이트).
    assert "self._latched" in code and "grasp_ready_hold_steps" in code


# ---------------------------------------------------------------- obs
def test_obs_has_no_object_identity():
    """policy obs 에 물체 정체성(onehot·치수·질량·클래스)이 없어야 한다(sim2real)."""
    code = _code(_ENV) + _code(_CFG)
    for banned in ("onehot", "object_class", "object_mass", "obj_scale"):
        assert banned not in code, f"obs 오염 경로: {banned}"


def test_obs_carries_tactile_and_goal():
    """촉각(tip-local 힘·관절 추종오차)과 이송 목표가 policy obs 에 있어야 한다.

    인벨롭이 잘 될수록 팁 F/T 가 0 을 읽으므로 `joint_pos_err` 가 주 파지력 관측이다.
    """
    m = re.search(r"_noisy = torch\.cat\(\[([\s\S]*?)\], dim=1\)", _ENV)
    assert m, "policy obs 결합식 부재"
    blk = m.group(1)
    for need in ("tip_force", "joint_err", "goal_rel", "palm_ax"):
        assert need in blk, f"policy obs 에 {need} 가 없다"


def test_obs_dim_formula_matches_layout():
    """cfg 의 obs 차원 산술식이 실제 결합 성분 수와 맞는지."""
    m = re.search(r"self\.observation_space = \(([\s\S]*?)\)", _CFG)
    assert m, "observation_space 식 부재"
    expr = m.group(1)
    for need in ("2 * n_arm", "2 * n_hand", "3 * num_tips", "self.action_space", "+ 3"):
        assert need in expr, f"obs 식에 {need} 가 없다"
    # 물체 뱅크·스케일에서 파생되면 안 된다.
    assert "bank" not in expr and "scale" not in expr


# ---------------------------------------------------------------- 보상
def test_reward_terms_include_transfer_and_stay():
    """이송 2항이 계약에 있어야 한다. 항 계약은 이 트랙 **로컬**이다."""
    code = _code(_REW)
    assert "GRASP_S2R_REWARD_TERMS" in code
    for term in ("transfer", "stay"):
        assert f'"{term}"' in code, f"보상 항 {term} 부재"
    # 공유 8항 계약(여러 트랙이 쓴다)을 끌어오면 안 된다.
    assert "GRASP_V2_REWARD_TERMS" not in code


def test_transfer_requires_contact_and_lift():
    """이송 보상은 접촉·리프트 없이는 0 이어야 한다 — 밀어 옮기기 차단."""
    blk = _assign_block(_code(_REW), "transfer")
    assert "lift_gate" in blk and "lifted_gate" in blk and "graded_contact" in blk


def test_stay_rewards_duration_not_touch():
    """stay 는 도달 순간이 아니라 **연속 유지 시간**에 비례해야 한다(찍고 빠지기 차단)."""
    assert "stay_frac" in _assign_block(_code(_REW), "stay")
    assert "self._stay_run" in _code(_ENV)


def test_approach_penalty_is_capped():
    """★approach 벌금은 상금(approach_weight)을 못 넘어야 한다 — approach 최솟값 0.

    상한이 없으면 컵에 닿을수록 손해가 되어 접촉 탐색이 금지되고, 스텝당 보상이
    순음수라 **조기 종료가 최적**이 된다(s2r_a1 실측: 16스텝 자살 경로 240 iter 고착,
    접촉 시작 시 grasp +0.43 vs approach −0.96→−2.02 로 순증분 음수).
    """
    code = _code(_REW)
    assert ".clamp(max=_aw)" in code, "밀림·기울기 벌금에 상한이 없다"
    m = re.search(r"_penalty = \(", code)
    assert m, "벌금 항이 분리돼 있지 않다"
    # 밀림 억제 자체는 disp_factor 가 계속 맡는다.
    assert "disp_factor" in code


def test_termination_causes_are_logged():
    """종료 원인별 비율이 있어야 무엇이 에피소드를 끝냈는지 역산 없이 안다."""
    code = _code(_ENV)
    for k in ("done/out_xy", "done/fell", "done/tipped", "done/abnormal"):
        assert f'"{k}"' in code, f"{k} 로깅 부재"


def test_disp_factor_uses_latch_snapshot():
    """★밀림 감쇠는 **래치 시점** 변위 기준이어야 한다.

    실시간 변위를 쓰면 이 트랙의 과제인 수평 이송이 통째로 처벌된다.
    """
    assert "cup_xy_disp_ref" in _REW
    m = re.search(r"_r = cup_xy_disp_ref / _limit", _REW)
    assert m, "감쇠가 래치 스냅샷을 안 쓴다"
    assert "self._disp_at_latch" in _code(_ENV)


# ---------------------------------------------------------------- 씬·기하
def test_goal_is_derived_from_settled_height():
    """goal 은 스폰점이 아니라 **정착고** 기준 — 스폰 패드가 리프트 기준에 실리면 안 된다."""
    code = _code(_ENV)
    assert "settled[:, 2] = float(self.cfg.table_surface_z)" in code
    assert "self.goal_pos[env_ids] = settled" in code
    assert "goal_offset_xyz" in code
    # 부팅에서 목표 도달성을 확인한다.
    assert "_assert_goal_reachable" in code


def test_spawn_height_has_single_source():
    """스폰 높이 파생은 cfg 한 곳에서만 — 이중 패딩 사고 차단."""
    assert _code(_CFG).count("self.object_spawn_z = (") == 1
    assert "object_spawn_pad" in _code(_CFG)


def test_contact_sensor_per_body():
    """body **하나당 센서 하나**. 다중 body 단일 센서는 force_matrix_w 가 무증상 0."""
    m = re.search(r"for body in bodies:([\s\S]*?)self\._finger_sensors\[finger\]", _CTRL)
    assert m and "ContactSensor(ContactSensorCfg(" in m.group(1)


# ---------------------------------------------------------------- agnosticism
def test_no_robot_joint_literals():
    """태스크 소스에 로봇 조인트/링크 리터럴이 없어야 한다 — 전부 프로필 경유."""
    pat = re.compile(r"\b[rl]_(aj|hj|hl)_")
    for name, src in (("env", _ENV), ("control", _CTRL), ("cfg", _CFG), ("rewards", _REW)):
        hit = pat.search(_code(src))
        assert hit is None, f"{name} 에 로봇 리터럴 '{hit.group(0)}'"


def test_hand_control_is_synergy_only():
    """손 제어 분기는 시너지 하나뿐이어야 한다(기각된 경로는 오해만 만든다)."""
    code = _code(_CTRL) + _code(_ENV) + _code(_CFG)
    # `use_hand_fabric=False` 는 fabric 생성자 인자라 남아야 한다 — 분기 플래그
    # (`self._hand_fabric`)만 금지한다.
    for banned in ("hand_control", "tip_cyl", "self._hand_fabric", "hand_attractor_gain"):
        assert banned not in code, f"기각된 손 제어 경로 잔재: {banned}"
    assert "use_hand_fabric=False" in _code(_CTRL)


def test_fabric_hand_state_is_synced():
    """fabric 은 실제 손 자세를 받아야 한다.

    끊으면 fabric 이 실재하지 않는 손으로 충돌구 FK 를 계산해 없는 자기충돌을
    피하려 팔을 민다(선행 트랙 실측: palm_err 475mm·joint_err 0.71rad·5kN).
    """
    code = _code(_ENV)
    assert "self._syn_to_fab(self._syn_target)" in code
    assert "_fab_home_hand" not in code, "fabric 손을 홈으로 고정하면 안 된다"


def test_fabric_integrates_once_per_policy_step():
    """적분은 `_step_fabric` 한 곳 — `_apply_action` 에서 돌리면 fabric 시간이 2배."""
    ctrl = _code(_CTRL)
    assert ctrl.count("self.integrator.step(") == 1
    m = re.search(r"def _apply_action\(self\)([\s\S]*?)\n    def ", ctrl)
    assert m and "integrator" not in m.group(1)

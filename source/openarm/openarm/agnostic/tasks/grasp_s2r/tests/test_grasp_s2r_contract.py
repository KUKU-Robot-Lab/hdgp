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


def test_opposition_axis_is_hand_derived():
    """★대향 중점은 **손 자신의 기하**에서 나와야 한다 — 임의 부호의 수직축 금지.

    구 수식 `axis = (−dir_y, dir_x)` 는 접근방향의 90° 회전이라 좌/우 부호가 임의였다.
    엄지 목표가 실제 엄지의 반대편에 놓이면 손목을 뒤집어야 도달 가능한 자세를 요구하고,
    정책은 그쪽으로 못 가서 엄지가 걸린 채 4지만 붙인다(실측: grip_frac 0.20 인데
    wrap_frac 이 2,228 iter 내내 0.000).
    """
    code = _code(_ENV)
    # ★★08.27: approach 의 cage_dist 는 **palm 강체**여야 한다. 실시간 손끝을 쓰면
    #   "쭉 편 손가락으로 팁을 컵 중심에 모으기"가 이 항의 최적이 되어 파지 예비자세를
    #   정면으로 방해한다 — s2r_a9 실측 corr(ch2 폐쇄, approach) = −0.702. ch2 가
    #   0.271 → 0.004 로 펴지는 동안 approach 0.61 → 0.75, touch_frac 0.000 유지.
    assert "cage_dist = self._cage_ctr_dist" in code
    assert "opp_mid" not in code, "보상용 케이지가 다시 실시간 손끝을 참조한다"
    # 임의 수직축·물체 반경 상수는 남아 있으면 안 된다.
    assert "axis[:, 0], axis[:, 1] = -_dir[:, 1], _dir[:, 0]" not in code
    for banned in ("object_grasp_radius", "enclosure_thumb_weight"):
        assert banned not in code + _code(_CFG), f"제거된 형상 상수 잔재: {banned}"


def test_closing_is_gated_on_cage_alignment():
    """★위치가 맞기 전에는 오므리지 않는다 — 닫는 방향만 게이트, 푸는 방향은 항상 허용.

    래치로는 못 막는다: 래치는 lift/transfer **보상**을 여는 신호일 뿐이고, 닫힘은
    정책의 손 액션이 직접 만든다. 실측(s2r_a5 iter13): cage_dist 0.293 = 케이지 반경의
    2.4배인데 syn_close 0.574 까지 닫혀 있었다.
    """
    env, ctrl = _code(_ENV), _code(_CTRL)
    assert "self._close_gate" in env and "close_gate_enabled" in env
    # 게이트는 손 액션을 만들기 **전에** 계산돼야 한다.
    assert env.index("self._close_gate =") < env.index("self._synergy_targets(")
    # 닫는 방향만 스케일 — 푸는 방향(delta<0)은 그대로여야 갇혔을 때 빠져나온다.
    assert "torch.where(delta > 0.0, delta * _g, delta)" in ctrl
    # 임계는 손 기하에서 부팅 실측한 케이지 반경이다(물체 상수 아님).
    assert "self._r_cage" in env


def test_close_gate_center_is_rigid_to_palm():
    """★★게이트 영역은 **손가락을 따라 움직이면 안 된다**.

    08.27 실측(s2r_a6, 202 iter): 중심을 실시간 손끝 평균으로 두니 팔이 정지한 구간
    (palm_to_cup 0.120~0.140, n=147)에서 corr(syn_close, cage_dist) = −0.974 —
    팔을 안 움직이고 손만 오므려도 중심이 컵 쪽으로 50mm(램프 폭의 83%) 당겨져
    게이트가 저절로 열렸다. "정렬되면 닫아라"가 아니라 "닫으면 닫아도 된다"는
    양의 되먹임이라 게이트가 아무것도 막지 못했다.

    또한 거리는 **3D** 여야 한다. xy 투영은 z 를 못 봐서 palm·검지가 컵보다 내려간
    잘못된 자세도 통과시켰다(사용자 GUI 관찰: 엄지가 컵에 걸린 채 접근).
    """
    env = _code(_ENV)
    blk = env[env.index("_obj = self._env_local(self.object.data.root_pos_w)"):]
    blk = blk[:blk.index("self._synergy_targets(")]
    # 중심 = palm + R_palm · (홈에서 실측한 고정 오프셋)
    assert "self._cage_offset_palm" in blk and "self._palm_ee_R()" in blk
    # 게이트 계산 구간에 손끝 위치가 등장하면 안 된다(되먹임 재발 방지).
    assert "_tip_ids_t" not in blk, "게이트가 다시 손끝을 참조한다 — 되먹임 재발"
    # 3D 거리 — xy 슬라이스로 되돌아가면 z 조건이 사라진다.
    # ★단 z 는 **데드밴드**를 통과한다(±grasp_z_deadband). 3D 노름이 z 를 xy 와 똑같이
    #   벌하는 바람에 palm 이 파지높이 아래로 눌려 내려갔다(s2r_b2 실측:
    #   palm_above_table mean 0.088 vs 파지중심 0.107, min 0.066 < 컵 원점 0.077).
    assert "self._cage_ctr_dist = self._banded_dist(_cage - _obj)" in blk
    ctrl_all = _code(_CTRL)
    assert "_dz = torch.relu(delta[:, 2].abs() - _b)" in ctrl_all, "z 데드밴드 부재"
    assert "palm_to_cup = self._banded_dist(palm_pos - grasp_center)" in env
    # 오프셋은 홈 자세에서 한 번만 실측한다(부팅 보고 안 — 게이트 블록 밖).
    assert "self._cage_offset_palm = _R.transpose(0, 1) @ (cage - _palm)" in env
    # 래치 후에는 해제 — 이송 중 컵이 흔들려도 다시 쥘 수 있어야 한다.
    assert "self._latched" in blk


def test_fabric_knows_about_the_table():
    """★★fabric 에 world 를 안 넘기면 테이블을 **아예 모르는 상태**로 계획한다.

    `WorldMeshesModel` 에 world_dict/world_filename 이 없으면 `object_indicator == 0`
    이라 반발 커널이 첫 줄에서 early-out 한다. 형제 tesollo 트랙은 전부
    `world_filename` 을 넘기는데 agnostic 트랙만 빠져 있었다 — 08.27 발견.
    사용자 GUI: "아예 테이블을 박히고 간다", 실측 palm_above_table min 0.066
    (컵 원점 0.077 보다 아래).

    ★params 의 body_repulsion.collision_sphere_frames 에 palm·5지 전 마디(소지 dg_5
      14개 포함)·팔 링크 충돌구가 이미 있어 테이블 하나로 손 전체가 보호된다 —
      params 파일은 건드리지 않는다.
    ★박스는 palm 도달영역에서 **파생**해야 한다. 숫자를 따로 적으면 물리 테이블과
      조용히 어긋난다.
    """
    ctrl, cfg = _code(_CTRL), _code(_CFG)
    assert "world_dict=self._build_fabric_world()" in ctrl, "fabric 이 빈 세계를 본다"
    assert "fabric_table_obstacle" in cfg
    # 상면은 table_surface_z 그 자체에서 파생 — 별도 상수 금지.
    assert "float(self.cfg.table_surface_z) - 0.5 * _th" in ctrl
    # 크기는 프로필 도달영역에서 파생.
    assert "_lo, _hi = p.palm_box_min, p.palm_box_max" in ctrl
    # 근거 없던 박스-바닥 클램프는 되돌렸다(fabric 반발이 정공법).
    assert "palm_min_above_table" not in ctrl + cfg


def test_grasp_has_pre_contact_gradient_gated_on_alignment():
    """★★`grasp` 는 **첫 접촉 전에도** 손가락을 내라는 gradient 를 줘야 한다.

    구판은 네 채널(팁접촉·전팁·지속·감쌈)이 전부 접촉 임계 뒤라 첫 접촉까지 정확히 0
    이었다. 그래서 접촉 전 손 모양을 정하는 보상이 approach 하나뿐이었고, approach 가
    실시간 손끝을 쓰는 바람에 최적 손 모양이 "쭉 편 손가락"이 됐다 — 손가락을 말면
    approach 가 즉시 깎이는데 grasp 는 닿아야 나오니 가는 길이 확실히 나쁜 **계곡**
    이었다(s2r_a9 526 iter: touch_frac 0.000 · wrap_frac 0.000 · ch2 0.004).

    계약: grasp = w · pre_lift · **close_gate** · [(1−ecred)·close_credit + ecred·wrap]
    · close_gate 곱 — 정렬 전 공중 폐쇄는 0 이어야 한다.
    · close_progress 는 **실측 관절**이어야 한다. 지령을 재면 손이 테이블에 눌려 쫙
      펴져도 만점이 나온다(s2r_b1: hand_joint_err_max 3.72 rad = 임계 0.30 의 12배인데
      grasp 4.69/step 지급). 실측은 물체에 막히면 스스로 멈추므로 인위적 포화 캡도
      필요 없다 — 캡을 뒀더니 **그 지점이 정지점**이 됐다(폐쇄도가 캡 0.5 에 고정).
    · 팁 제어 3채널은 폐기됐다 — 팔이 정밀 제어를 하는 지금은 불필요(사용자 확정).
    """
    rew = _code(_REW)
    blk = _assign_block(rew, "grasp_quality")
    assert "close_credit" in blk and "wrap_frac" in blk
    for banned in ("tip_contact_frac", "full_tip", "persistence"):
        assert banned not in blk, f"폐기된 팁 제어 채널이 grasp 에 되살아남: {banned}"
    assert "close_gate.clamp(0.0, 1.0) * grasp_quality" in rew, "grasp 가 정렬 게이트를 안 탄다"
    assert "_cref" not in rew, "포화 캡이 되살아남 — 그 지점이 정지점이 된다"
    # 폐쇄도는 **실측 관절**이어야 한다(지령 `_syn_close` 를 재면 테이블에 펴져도 만점).
    ctrl = _code(_CTRL)
    assert "_q = self.robot.data.joint_pos[:, self._syn_ids]" in ctrl
    assert "return _prog[:, self._syn_movable].mean(dim=1)" in ctrl
    assert "return self._syn_close[:, self._syn_movable]" not in ctrl, "폐쇄도가 다시 지령이다"
    # 가동폭 0° 관절(전 `_1`·pinky_2·thumb_2)이 분모에 섞이면 공짜 점수가 된다.
    assert "self._syn_movable = (self._syn_grip - self._syn_open).abs() > 1e-4" in ctrl
    # graded_contact(리프트 이후 "정말 쥐고 있나")는 팁을 계속 써야 한다 — 폐기 대상 아님.
    assert "graded_contact = (1.0 - _emix) * tip_contact_frac" in rew

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


def test_contact_freeze_is_per_joint_link():
    """★★동결은 관절마다 **자기 링크**가 닿았을 때만 걸려야 한다.

    구판은 (원위|팁) 접촉 하나로 `_3`·`_4` 를 통째로 얼렸다. 그런데 `_2` 가 굽으면
    손끝이 가장 먼저 닿으므로 **감쌈이 시작되기 직전에 감쌈 관절을 잠그는** 구조였다 —
    wrap_frac 이 전 런에서 정확히 0.000 이었고, syn_close 0.278 이 "채널1(`_2`)만
    폐쇄" 예측 0.250 과 일치했다(사용자 GUI: `_2` 완전굴곡·`_3`/`_4` 정지).

    또한 동결은 **닫는 방향에만** 걸려야 한다. 양방향을 막으면 잘못 얼린 자세에서
    빠져나올 수 없다(닫기 게이트와 같은 원칙).
    """
    ctrl = _code(_CTRL)
    assert "self._syn_freeze_mid" in ctrl and "self._syn_freeze_dist" in ctrl
    # `_3` 은 중간마디, `_4` 는 **원위 링크만** — 팁은 트리거가 아니다.
    # 팁은 원위와 별개 body 라, 팁으로 `_4` 를 얼리면 원위가 닿을 기회가 사라져
    # wrap(중간 AND 원위)이 영원히 0 이 된다(s2r_a8 817 iter 실측).
    assert "_h_dist = (_dist > _thr)[:, self._syn_fi]" in ctrl
    assert "self._tip_contact_forces() > _thr" not in ctrl, "팁이 동결 트리거로 되살아남"
    assert "_hold = (_h_mid & self._syn_freeze_mid) | (_h_dist & self._syn_freeze_dist)" in ctrl
    # 푸는 방향은 항상 허용.
    assert "torch.where(_hold & (delta > 0.0), torch.zeros_like(delta), delta)" in ctrl
    # 구판 배선이 되살아나면 실패시킨다.
    assert "delta * (~(_hold & self._syn_freeze)).float()" not in ctrl


def test_approach_targets_the_palm_not_the_cage():
    """★★접근 목표는 **palm** 이어야 한다 — 케이지를 목표로 두면 핀치가 강제된다.

    홈 실측: 케이지 중심이 palm 앞 **106mm**(cage−palm = 82.2, 66.4, 3.4 mm).
    approach 가 `cage_dist → 0` 을 요구하면 palm 은 컵에서 106mm 떨어져야 하므로
    "손바닥 밀착"과 **구조적으로 양립 불가**다. 실측 타협점 palm_to_cup 0.126 /
    cage_dist 0.041 이 사용자 GUI 관찰 "palm_ee → 손가락 → 컵 순서"의 정체다.

    · cage_dist 는 approach 에서 빠지고 **닫기 게이트 전용**으로 남는다.
    · 거리는 palm 프레임으로 분해해 법선(palm_ee_x)=밀착도를 더 날카롭게 본다.
      법선거리는 컵 표면에서 물리적으로 포화하므로 형상 상수가 필요 없다.
    · `palm_still` 을 곱해 **밀착한 채 멈추게** 한다 — 그래야 시너지 손가락이 말린다.
      "멀리서 정지" 회피는 성립 안 함: 홈(d 0.36) 정지 0.055 vs 밀착(d 0.05) 정지 0.67.
    """
    rew, env, cfg = _code(_REW), _code(_ENV), _code(_CFG)
    _i = rew.index("approach = pre_lift_gate")
    blk = rew[_i:rew.index("grasp_quality", _i)]
    assert "cage_dist" not in blk, "approach 가 다시 케이지를 목표로 삼는다(핀치 강제)"
    assert "palm_normal_dist" in blk and "palm_lateral_dist" in blk
    assert "palm_still" in blk, "밀착 후 정지 요건이 없다"
    assert "approach_sharpness_normal" in cfg and "palm_still_gain" in cfg
    # 법선은 palm 회전행렬 열 0(손바닥 법선)에서 나온다.
    assert "_dn = (_d * _R[:, :, 0]).sum(dim=-1)" in env
    # 정지는 **실측** palm 선속도다(액션 변화량이 아니다 — 손가락을 말면 안 되니까).
    assert "self.robot.data.body_lin_vel_w[:, self.palm_idx]" in env


def test_net_force_reading_is_diagnostic_only():
    """★★필터 없는 `net_forces_w` 는 **진단 전용**이다 — 보상 경로에 새면 안 된다.

    `force_matrix_w` 는 컵 baseLink 로 필터링된 접촉만 담고 `net_forces_w` 는 그 링크가
    받은 **모든** 접촉(테이블·자기충돌·다른 손가락)을 담는다. 후자를 보상에 쓰면
    "테이블을 짚고 있다"가 파지로 계상된다 — 08.22 envelope 판정 사고와 같은 부류.

    둘을 나란히 읽는 이유는 08.27 실측 때문이다: 원위(`_4`)가 다섯 손가락 전부·4,553
    기록점 내내 정확히 0.000 인데 영상에서는 감쌈이 성립한다. net 이 양수인데 matrix 가
    0 이면 필터 결함이고, 둘 다 0 이면 진짜 미접촉이다.
    """
    ctrl, env = _code(_CTRL), _code(_ENV)
    assert "net_forces_w" in ctrl, "무필터 판독이 없다 — 두 가설을 못 가른다"
    # 보상이 쓰는 두 진입점은 **필터판**만 봐야 한다.
    for _fn in ("_contact_forces_split", "_tip_contact_forces"):
        _i = ctrl.index(f"def {_fn}")
        _blk = ctrl[_i:ctrl.index("def ", _i + 10)]
        assert "_mag_filtered" in _blk, f"{_fn} 이 필터판을 안 쓴다"
        assert "_mag_net" not in _blk, f"{_fn} 에 무필터가 샜다 — 테이블 접촉이 파지로 계상된다"
    # 진단 로깅은 보상 총합이 정해진 **뒤**에 불린다(반환값 없음).
    assert "self._log_diagnostics(" in env
    assert env.index("self._log_diagnostics(") < env.index("return total")
    assert "def _log_diagnostics" in env


def test_blocked_needs_both_error_and_away_from_limit():
    """★"더 못 조인다"는 **한계 도달**과 **물체에 막힘** 둘 다에서 성립한다.

    `hand_grip_pose` 가 soft limit 을 넘겨 과지령이라(1.8 rad vs 1.571) 완전 폐쇄만으로
    모든 관절이 목표를 못 따라가는 상태가 된다 — 허공에서 주먹을 쥐어도 오차 조건은
    참이다. 관절이 **자기 한계에서 떨어져 있는지**를 함께 봐야 외부 차단이 확정된다.
    ★가동폭 0° 관절(전 `_1` + pinky_2 + thumb_2)은 항상 오차 상태라 분모에서 빠져야 한다.
    """
    ctrl, cfg = _code(_CTRL), _code(_CFG)
    _i = ctrl.index("def _hand_blocked")
    blk = ctrl[_i:ctrl.index("def ", _i + 10)]
    assert "_syn_lo" in blk and "_syn_hi" in blk, "한계 근접 판정이 없다"
    assert "blocked_err_thr_rad" in blk and "blocked_limit_eps_rad" in blk
    assert "self._syn_movable" in blk, "가동폭 0° 관절이 분모에 섞인다"
    assert "&" in blk, "두 조건이 AND 로 묶이지 않았다"
    assert "blocked_err_thr_rad" in cfg and "diag_contact_threshold_lo" in cfg


def test_goal_distance_is_logged_by_component():
    """★`goal_dist` 스칼라만으로는 높이 탓인지 수평 탓인지 못 가른다.

    08.27 실측 goal_dist 0.281 에서 높이 성분과 수평 성분의 비중이 처방을 가른다 —
    높이면 `lift_height_ref`(0.10) vs `goal_offset_xyz.z`(0.08) 충돌이고, 수평이면
    홈 복귀(`a=0` 이 홈)다. 두 원인은 처방이 완전히 다르다.
    """
    env = _code(_ENV)
    assert 'self.extras["task/goal_dz"]' in env
    assert 'self.extras["task/goal_dxy"]' in env
    # 홈 복귀 가설의 직접 관측량 — a=0 이 정확히 홈이라 액션 크기가 곧 홈 이탈량이다.
    assert 'self.extras["task/action_norm_arm"]' in env
    assert 'self.extras["task/palm_to_home"]' in env
    # 파지 자세가 명령 박스 안에 있는지 — 축별로 봐야 어느 축이 부족한지 안다.
    assert 'f"fabric/palm_cmd_box_sat_{_ax}"' in env
    assert 'self.extras["fabric/palm_cmd_rate_sat"]' in env

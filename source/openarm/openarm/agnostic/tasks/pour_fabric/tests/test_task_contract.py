"""pour_fabric 태스크 계약.

grasp_lift_fabric 의 계약을 계승 + pour 고유 계약(낙하 종료·증분 보상·warm 게이트).
Isaac 이 필요한 검사는 importorskip 으로 나눈다(모듈 레벨 skip 금지 — 정적 검사가
같이 죽는다).
"""
import ast
import re
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parents[1]
_ENV = (_PKG / "pour_fabric_env.py").read_text()
_CFG = (_PKG / "pour_fabric_env_cfg.py").read_text()
_REW = (_PKG / "rewards.py").read_text()
_BANK = (_PKG / "warm_bank.py").read_text()
_ALL = {"env": _ENV, "cfg": _CFG, "rewards": _REW,
        "bimanual": (_PKG / "bimanual.py").read_text(),
        "bead_flags": (_PKG / "bead_flags.py").read_text(),
        "warm_bank": _BANK}

# 9항 — rewards.compute_rewards 의 terms 키와 일치해야 한다.
EXPECTED_TERMS = (
    "hold_source", "hold_receiver", "aim", "tilt", "pour_delta",
    "success", "spill_penalty", "drop_penalty", "action_rate",
)


# =============================================================================
# robot-agnosticism / s2r
# =============================================================================
_LITERAL = re.compile(r"\b[rl]_(aj|hj|hl)_")


@pytest.mark.parametrize("name", sorted(_ALL))
def test_no_robot_specific_literals(name):
    for i, line in enumerate(_ALL[name].splitlines(), 1):
        assert not _LITERAL.search(line), f"{name}:{i}: 로봇 리터럴: {line.strip()}"


def test_env_does_not_import_specific_profiles():
    for sym in ("BIS_RIGHT", "BIS_LEFT", "BI_RIGHT", "SENS_", "RH56_"):
        assert sym not in _ENV


def test_fabric_state_is_not_in_observation():
    """fabric_q/qd 는 실기에 없는 내부 상태 — grasp_v1 s2r 실패의 직접 원인."""
    tree = ast.parse(_ENV)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_get_observations")
    src = ast.get_source_segment(_ENV, fn)
    assert "fabric_q" not in src and "fabric_qd" not in src


def test_no_reward_latch():
    """보상 래치 금지 — **식별자** 기준(주석/문서의 '래치 금지' 언급은 무해).
    (_crossed 는 비드의 물리 사건 기록이라 예외 — 보상 게이트가 아니라 이력이다.)"""
    for name in ("env", "rewards"):
        tree = ast.parse(_ALL[name])
        for node in ast.walk(tree):
            ident = (node.id if isinstance(node, ast.Name)
                     else node.attr if isinstance(node, ast.Attribute) else "")
            assert "latch" not in ident.lower(), f"{name}: 래치 식별자 {ident}"


def test_no_sim_mutation_during_reward():
    tree = ast.parse(_ENV)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_get_rewards")
    src = ast.get_source_segment(_ENV, fn)
    for bad in ("write_root_state_to_sim", "set_joint_position_target",
                "write_joint_state_to_sim", "write_object_state_to_sim"):
        assert bad not in src, f"_get_rewards 가 sim 을 변경한다: {bad}"


# =============================================================================
# 보상 구조
# =============================================================================
def test_nine_terms_exactly():
    m = re.search(r"terms = \{(.*?)\}", _REW, re.S)
    keys = re.findall(r'"(\w+)":', m.group(1))
    assert tuple(keys) == EXPECTED_TERMS


def test_pour_delta_is_relu_increment():
    """증분 보상만(레벨 금지·되돌림 음수 금지 — spill 이 담당)."""
    assert "d_in_target.clamp(min=0.0)" in _REW
    assert "d_released.clamp(min=0.0)" in _REW


def test_spill_penalty_is_capped():
    m = re.search(r"spill_penalty = .*?d_spill\.clamp\(\s*min=0\.0, max=", _REW, re.S)
    assert m, "spill 증분 페널티에 상한 클램프가 없다"


def test_release_weight_defaults_to_zero():
    """spill 페널티가 작은 초기에 release 보상은 '바닥 붓기'를 보상한다."""
    assert re.search(r'"pour_release_weight", 0\.0', _REW)
    assert re.search(r"pour_release_weight: float = 0\.0", _CFG)


def test_action_rate_is_mean_not_sum():
    m = re.search(r"action_rate = .*?torch\.(\w+)\(", _REW, re.S)
    assert m.group(1) == "mean", "robot-agnostic 트랙에서 sum 금지(차원 비례)"


def test_tilt_has_proximity_gate():
    """근접 게이트 없는 tilt 는 '빈 데서 기울이기' farming 을 연다."""
    assert re.search(r"tilt = .*prox.*tilt_prog", _REW, re.S)


def test_every_reward_cfg_key_is_declared():
    """_cfg(cfg,'x',기본) 이 cfg 에 미선언이면 조용히 기본값으로 돈다 — 금지."""
    used = dict(re.findall(r'_cfg\(cfg, "(\w+)", (-?[\d.]+)\)', _REW))
    declared = dict(re.findall(r"^    (\w+): float = (-?[\d.]+)", _CFG, re.M))
    for key, default in used.items():
        assert key in declared, f"cfg 미선언 보상 키: {key}"
        assert float(default) == float(declared[key]), (
            f"{key}: rewards 기본 {default} != cfg 선언 {declared[key]}")


# =============================================================================
# 종료 규약 (grasp 와 의도적으로 다름)
# =============================================================================
def test_drop_terminates():
    tree = ast.parse(_ENV)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_get_dones")
    src = ast.get_source_segment(_ENV, fn)
    assert "_dropped_now" in src and "runaway" in src


def test_success_does_not_terminate():
    """성공 후 유지가 5배 비드 수율(pour_v1 실측) — 성공은 종료 사유가 아니다."""
    tree = ast.parse(_ENV)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_get_dones")
    src = ast.get_source_segment(_ENV, fn)
    assert "_success" not in src


# =============================================================================
# warm 뱅크
# =============================================================================
def test_no_hardcoded_dataset_dirs():
    """pour_v1 의 /home/oem/… fallback 은 폐기 대상이었다 — 문자열 리터럴 기준
    (docstring 의 이력 언급은 무해)."""
    tree = ast.parse(_BANK)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            d = ast.get_docstring(node, clean=False)
            if d:
                docstrings.add(d)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in docstrings:
                continue
            assert "/home/" not in node.value, f"하드코딩 경로: {node.value!r}"


def test_bank_gates_are_hard_fail():
    for key in ("robot_usd", "enable_gravity", "enable_self_collisions"):
        assert key in _BANK
    assert "raise RuntimeError" in _BANK


def test_bank_writer_requires_physics_meta():
    m = re.search(r'required = \((.*?)\)', _BANK, re.S)
    for key in ("robot_usd", "enable_gravity", "enable_self_collisions"):
        assert key in m.group(1)


def test_bank_load_gates_runtime():
    """의도적 불일치 주입 — 게이트가 실제로 막는지 실행으로 확인."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("h5py")
    import tempfile

    from openarm.agnostic.tasks.pour_fabric.warm_bank import PourWarmBank, save_bank

    n, J = 64, 27
    kw = dict(joint_names=tuple(f"j{i}" for i in range(J)),
              joint_pos=np.zeros((n, J)), joint_target=np.zeros((n, J)),
              cup_pose=np.tile([0, 0, 0.3, 1, 0, 0, 0], (n, 1)),
              num_contacts=np.zeros(n), bead_state=None)
    meta = dict(robot_usd="asset_a.usd", enable_gravity=True,
                enable_self_collisions=True, profile="p", checkpoint="c")
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/bank.hdf5"
        save_bank(path, meta=meta, **kw)
        ok = PourWarmBank.load(path, expect_robot_usd="asset_a.usd",
                               expect_gravity=True, expect_self_collisions=True)
        assert len(ok) == n
        with pytest.raises(RuntimeError, match="robot_usd"):
            PourWarmBank.load(path, expect_robot_usd="asset_B.usd",
                              expect_gravity=True, expect_self_collisions=True)
        with pytest.raises(RuntimeError, match="enable_gravity"):
            PourWarmBank.load(path, expect_robot_usd="asset_a.usd",
                              expect_gravity=False, expect_self_collisions=True)
        with pytest.raises(ValueError, match="meta"):
            save_bank(f"{d}/b2.hdf5", meta={"robot_usd": "x"}, **kw)


# =============================================================================
# 액션 폭 불변
# =============================================================================
def test_action_width_is_constant_nine():
    """frozen↔learned 인계를 위해 receiver_control_mode 와 무관하게 9."""
    assert re.search(r"^NUM_ACTIONS = 9$", _CFG, re.M)
    m = re.search(r"cfg\.action_space = (\w+)", _CFG)
    assert m.group(1) == "NUM_ACTIONS"


# =============================================================================
# 물리 동일성 (grasp_lift_fabric 값 복사 검증)
# =============================================================================
def test_physics_values_match_grasp_lift_fabric():
    g = (_PKG.parent / "grasp_lift_fabric" / "grasp_lift_fabric_env_cfg.py").read_text()
    for pat in (r"max_depenetration_velocity=1\.0",
                r"solver_position_iteration_count=16",
                r"solver_velocity_iteration_count=1",
                r"fabrics_dt: float = 1\.0 / 60\.0",
                r"fabric_decimation: int = 2",
                r"fabrics_damping_gain: float = 20\.0",
                r"palm_slew_pos: float = 0\.004",
                r"palm_slew_rot_deg: float = 2\.0",
                r"ground_plane_z: float = -0\.10",
                r"runaway_joint_vel: float = 20\.0",
                r"contact_force_threshold: float = 1\.0"):
        assert re.search(pat, _CFG), f"pour cfg 에 없음: {pat}"
        assert re.search(pat, g), f"grasp cfg 에 없음(원본이 바뀌었다 — 동기화 필요): {pat}"
    assert re.search(r"dt=1\.0 / 120\.0", _CFG) and re.search(r"dt=1\.0 / 120\.0", g)
    # pour 기본 물리 스위치 = fab_test10 베이스라인 (grasp 는 CLI 로 켠다)
    assert re.search(r"enable_gravity: bool = True", _CFG)
    assert re.search(r"enable_self_collisions: bool = True", _CFG)


# =============================================================================
# 차원 (Isaac 필요 — resolve_cfg 결과와 대조. 공식 재구현 금지)
# =============================================================================
def _cfg_module():
    # isaaclab 은 앱 없이도 import 되지만 내부에서 pxr 을 끌어온다 —
    # pxr 은 Isaac 앱(AppLauncher) 안에서만 존재한다. 앱 밖에선 skip.
    pytest.importorskip("pxr")
    pytest.importorskip("isaaclab")
    from openarm.agnostic.tasks.pour_fabric import pour_fabric_env_cfg as C
    return C


def test_dims_from_resolve_cfg():
    C = _cfg_module()
    cfg = C.PourFabricEnvCfg()
    pair = __import__(
        "openarm.agnostic.tasks.pour_fabric.bimanual", fromlist=["get_pair"]
    ).get_pair(cfg.pair_name)
    j = (pair.source.num_arm_joints + pair.source.num_hand_joints
         + pair.receiver.num_arm_joints + pair.receiver.num_hand_joints)
    f = len(pair.source.fingers) + len(pair.receiver.fingers)
    assert cfg.action_space == 9
    assert cfg.observation_space == 3 * j + f + 7 + 7 + 3 + 3 + 18
    assert cfg.state_space == cfg.observation_space + 19


def test_bis_reference_dimensions():
    C = _cfg_module()
    cfg = C.PourFabricEnvCfg()
    assert (cfg.action_space, cfg.observation_space, cfg.state_space) == (9, 210, 229)


def test_receiver_mode_does_not_change_dims():
    C = _cfg_module()
    a = C.PourFabricEnvCfg()
    b = C.PourFabricEnvCfg()
    b.receiver_control_mode = "learned"
    C.resolve_cfg(b)
    assert (a.action_space, a.observation_space) == (b.action_space, b.observation_space)


def test_resolve_cfg_rejects_bad_receiver_mode():
    C = _cfg_module()
    cfg = C.PourFabricEnvCfg()
    cfg.receiver_control_mode = "scripted"
    with pytest.raises(ValueError):
        C.resolve_cfg(cfg)


def test_registered_cfg_classes_keep_own_pair_name():
    """configclass 상속 함정 회귀 가드 — grasp_lift_fabric 5e31d86 과 동일 결함.

    서브클래스에 일반 클래스 속성으로 pair_name 을 넣으면 베이스 데이터클래스
    __init__ 이 베이스 기본값("bis")을 인스턴스 속성으로 다시 써서 조용히 가린다.
    등록된 모든 쌍의 생성 cfg 를 실제 인스턴스화해 pair_name 을 대조한다.
    """
    _cfg_module()  # Isaac 게이트
    from openarm.agnostic.tasks.pour_fabric import config as reg

    assert reg.REGISTERED, "등록된 쌍이 없다"
    for short in reg.REGISTERED:
        for suffix in ("", "_PLAY"):
            cls = getattr(reg, f"PourFabric_{short}{suffix}_Cfg")
            cfg = cls()
            assert cfg.pair_name == short, (
                f"{cls.__name__}: pair_name={cfg.pair_name!r} != {short!r} — "
                "베이스 기본값에 가려짐(configclass 상속 함정)")
    play = getattr(reg, f"PourFabric_{sorted(reg.REGISTERED)[0]}_PLAY_Cfg")()
    assert play.scene.num_envs == 50


# =============================================================================
# grasp_lift_fabric 정렬 (08.23, 사용자 지시)
# =============================================================================
def test_envelope_uses_shared_function_not_a_copy():
    """감쌈 판정은 자매 트랙과 **같은 함수**여야 한다 — 복사하면 갈라진다."""
    src = _ENV
    assert "from ..grasp_sensor.rewards import envelope_fraction" in src
    # 로컬 재구현 금지
    assert "def envelope_fraction" not in src
    assert "def envelope_fraction" not in _REW


def test_envelope_denominator_excludes_non_flexing_fingers():
    """분모는 profile.envelope_fingers — 전 손가락을 쓰면 pinky 때문에 상한 0.8."""
    src = _ENV
    assert "profile.envelope_fingers" in src, "env_f 를 프로필에서 만들어야 한다"
    assert "self.env_f" in src
    # 구판 회귀 방지: wrap 인디케이터 전체 평균으로 감쌈을 세면 안 된다.
    assert "src_w.mean(dim=1)" not in src


def test_contact_thresholds_are_split_three_ways():
    """게이트/참여/엄격 임계 분리 — 하나로 쓰면 스침 차단과 참여 판정이 결합된다."""
    cfg = _CFG
    for pat in (r"contact_force_threshold: float = 1\.0",
                r"participation_force_threshold: float = 0\.1",
                r"envelope_force_threshold: float = 0\.5"):
        assert re.search(pat, cfg), f"{pat} 없음"
    env = _ENV
    assert "participation_force_threshold" in env, "참여 임계가 소비되지 않는다"


def test_thresholds_match_sibling_track():
    """세 임계값은 grasp_lift_fabric 과 같아야 한다(정렬 계약)."""
    ours = _CFG
    sib = (_PKG.parent / "grasp_lift_fabric" / "grasp_lift_fabric_env_cfg.py").read_text()
    for key in ("contact_force_threshold", "participation_force_threshold",
                "envelope_force_threshold"):
        m_o = re.search(rf"{key}: float = ([0-9.]+)", ours)
        m_s = re.search(rf"{key}: float = ([0-9.]+)", sib)
        assert m_o and m_s, f"{key} 파싱 실패"
        assert m_o.group(1) == m_s.group(1), (
            f"{key}: pour={m_o.group(1)} != grasp={m_s.group(1)}")


def test_fabric_owns_the_hand_in_direct_mode():
    """fabric 이 손 자세를 알아야 body_repulsion 이 맞는 형상으로 계산된다.

    실측(probe_pour_hand_drift): PCA zeros 를 주던 구판은 fabric 내부 손이 실제와
    step 400 에 16.7° 벌어졌고, direct + warm 자세 지령으로 6.6°(전부 fabric 내부
    경합, PD 층은 0.0001 rad)로 줄었다.
    """
    src = _ENV
    assert 'hand_mode="direct"' in src
    assert "use_hand_fabric=True" in src
    # 구판 회귀 방지: 5D PCA 지령 버퍼
    assert "fabric_hand_cmd = torch.zeros(self.num_envs, 5" not in src
    assert "rig.fabric_hand_cmd = rig.hand_hold[:, rig.fab_from_hand]" in src

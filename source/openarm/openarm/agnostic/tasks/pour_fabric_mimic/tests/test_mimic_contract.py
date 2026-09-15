"""pour_fabric_mimic 계약 — 저차원(언더액추) 손 전용 트랙이 원본 pour_fabric 과 갈라져야 하는 곳을 잠근다.

Isaac 불요(소스·프로필 데이터). 차원 대조는 Isaac 있을 때만.
"""
import io
import re
import tokenize
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from openarm.agnostic.modules import robot_profiles as _rp
from openarm.agnostic.tasks.pour_fabric_mimic import bimanual as _bm
from openarm.agnostic.tasks.pour_fabric_mimic import robot_profiles as _local

_PKG = Path(__file__).resolve().parents[1]


def _code_only(src: str) -> str:
    """주석·문자열(독스트링) 제거 — 리터럴 검사는 **코드**에만 건다(설명문은 로봇 이름을 써야 한다)."""
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        out.append(tok.string)
    return " ".join(out)


_ENV = (_PKG / "pour_fabric_env.py").read_text()
_RIG = (_PKG / "side_rig.py").read_text()
_CFG = (_PKG / "pour_fabric_env_cfg.py").read_text()
_ORIG = _PKG.parent / "pour_fabric"
_HDGP = Path(__file__).resolve().parents[7]
_ASSET = _HDGP / "assets" / "robot" / "openarm_rh56f1_bi_rl"
assert (_ASSET / "openarm_rh56f1_bi_rl.urdf").exists(), _ASSET


# =============================================================================
# 원본 불변 · 이 트랙은 별도 폴더
# =============================================================================
def test_original_pour_fabric_untouched_by_this_track():
    """원본에 mimic 문자열이 새로 들어가면 사용자 결정("원본 냅두고") 위반."""
    for f in ("side_rig.py", "pour_fabric_env.py", "pour_fabric_env_cfg.py", "bimanual.py"):
        assert "mimic" not in (_ORIG / f).read_text(), f
    assert "openarm_rh56f1_bi_rl" not in (_ORIG / "bimanual.py").read_text()


@pytest.mark.parametrize("name", ["env", "rig", "cfg"])
def test_no_robot_specific_literals(name):
    src = {"env": _ENV, "rig": _RIG, "cfg": _CFG}[name]
    body = _code_only("\n".join(l for l in src.splitlines() if not l.startswith("import fabrics_sim")))
    for lit in ("r_aj_", "l_aj_", "r_hj_", "l_hj_", "r_hl_", "l_hl_", "tesollo", "dg5f", "rh56f1_right", "RH56F1_"):
        assert lit not in body, f"{name}: 로봇 리터럴 '{lit}' — 프로필로 옮길 것"


# =============================================================================
# 프로필 · 쌍
# =============================================================================
def test_pair_rh_is_default_and_only_from_local_profiles():
    assert _bm.DEFAULT_PAIR == "rh" and set(_bm.PAIRS) == {"rh"}
    p = _bm.get_pair("rh")
    assert p.source.name == "rh56f1_right_fab" and p.receiver.name == "rh56f1_left_fab"
    assert p.usd_relpath.endswith("openarm_rh56f1_bi_rl/openarm_rh56f1_bi_rl.usd")
    # 모듈 레지스트리에는 안 들어간다(grasp config 가 그 dict 로 gym id 를 찍는다)
    assert "rh56f1_right_fab" not in _rp.PROFILES and "rh56f1_left_fab" not in _rp.PROFILES


def test_fabric_enabled_on_both_sides_with_26dof_order():
    p = _bm.get_pair("rh")
    assert p.source.fabric_class == "OpenArmRh56f1PoseFabric"
    assert p.receiver.fabric_class == "OpenArmRh56f1LeftPoseFabric"
    assert len(_local.FABRIC_JOINT_ORDER) == 26
    assert p.source.fabric_joint_order == p.receiver.fabric_joint_order == _local.FABRIC_JOINT_ORDER
    assert _bm.fabric_slots(p.source) == (slice(0, 7), slice(7, 13))
    assert _bm.fabric_slots(p.receiver) == (slice(13, 20), slice(20, 26))
    # 슬라이스 관절 = 프로필 관절(순서까지)
    for prof in (p.source, p.receiver):
        a, h = _bm.fabric_slots(prof)
        order = _local.FABRIC_JOINT_ORDER
        assert all(re.fullmatch(prof.arm_joint_regex, n) for n in order[a])
        assert tuple(order[h]) == prof.hand_joint_names


def test_left_is_mirror_of_right_arm_sign_and_hand_identity():
    """09.14 순수 FK: 팔 (−1,−1,−1,+1,−1,−1,−1) · 손 +1 → 4지 손끝 0.00 mm 일치."""
    p = _bm.get_pair("rh")
    init = p.init_joint_pos
    sign = (-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0)
    for i, s in enumerate(sign, start=1):
        assert init[f"l_aj_{i}"] == pytest.approx(s * init[f"r_aj_{i}"])
    for n in ("thumb_1", "thumb_2", "index_1", "middle_1", "ring_1", "pinky_1"):
        assert init[f"l_hj_{n}"] == pytest.approx(init[f"r_hj_{n}"])
    assert p.receiver.hand_open_pose == p.source.hand_open_pose
    assert p.receiver.hand_grip_pose == p.source.hand_grip_pose
    assert p.receiver.object_spawn_center == (p.source.object_spawn_center[0], -p.source.object_spawn_center[1])
    assert p.receiver.palm_box_min[1] == -p.source.palm_box_max[1]


def test_left_hand_limits_admit_the_mirrored_poses():
    """좌 URDF 한계 안에 open/grip 이 들어가는지(부호 +1 이 성립하는 전제)."""
    urdf = next(_ASSET.glob("*.urdf"))
    lim = {j.get("name"): (float(j.find("limit").get("lower")), float(j.find("limit").get("upper")))
           for j in ET.parse(urdf).getroot().iter("joint") if j.find("limit") is not None and j.get("type") == "revolute"}
    p = _bm.get_pair("rh").receiver
    for nm, o, g in zip(p.hand_joint_names, p.hand_open_pose, p.hand_grip_pose):
        lo, hi = lim[nm]
        assert lo - 1e-6 <= o <= hi + 1e-6 and lo - 1e-6 <= g <= hi + 1e-6, (nm, o, g, lo, hi)


def test_dependent_actuators_zero_gain_both_sides():
    p = _bm.get_pair("rh")
    mimic = {k: v for k, v in p.actuator_specs.items() if k.endswith("_hand_mimic")}
    assert set(mimic) == {"src_right_hand_mimic", "rcv_left_hand_mimic"}
    for v in mimic.values():
        assert (v["stiffness"], v["damping"]) == (0.0, 0.0)


def test_per_finger_slots_bijective_with_driven_joints():
    p = _bm.get_pair("rh")
    for prof in (p.source, p.receiver):
        slots = sorted(int(s) for m in prof.hand_finger_channels.values() for s in m.values())
        assert slots == list(range(prof.num_hand_joints))
        assert prof.hand_freeze_suffixes and set(prof.hand_freeze_suffixes) >= {"1", "2"}


# =============================================================================
# 소스 계약 — 원본과 갈라진 곳
# =============================================================================
def test_fabric_ctor_uses_bimanual_class_signature():
    assert 'use_hand_fabric=False, hand_mode="direct")' in _RIG
    code = _code_only(_RIG)
    assert "robot_dir_name" not in code and "tip_per_finger" not in code
    assert "def sync_other" in _RIG and "self.src.sync_other(self.rcv)" in _ENV and "self.rcv.sync_other(self.src)" in _ENV
    assert "self.fabric_q[:, self.arm_sl]" in _RIG and "[:, :n_arm]" not in _RIG


def test_hand_layout_is_per_finger_with_coupling_matrix():
    assert "self.syn_slot" in _RIG and "self.syn_couple" in _RIG
    assert "a_hand.view(N, nf, self.syn_nch)" not in _RIG
    assert "hold = (h_mid | h_dist) & self.syn_flex" in _RIG
    assert 'self.syn_flex = torch.tensor([s in p.hand_freeze_suffixes for s in sfx]' in _RIG
    assert 's == "3"' not in _RIG


def test_mimic_gates_present_and_read_from_asset():
    for fn in ("_load_mimic_pairs", "_assert_mimic_constraints_present", "_assert_hand_pose_usable", "_widen_dependent_joint_limits"):
        assert f"def {fn}" in _ENV
    assert "physxMimicJoint:rotZ:gearing" in _ENV and 'j.find("mimic")' in _ENV
    assert "ctrl/mimic_err_max" in _ENV
    assert "self._widen_dependent_joint_limits()" in _ENV


def test_cfg_kills_tesollo_only_knobs():
    assert re.search(r"oppose_grip_delta_rad:\s*float\s*=\s*0\.0", _CFG)
    assert re.search(r'synergy_freeze_scope:\s*str\s*=\s*"finger"', _CFG)
    m = float(re.search(r"mimic_dep_limit_margin_rad:\s*float\s*=\s*([0-9.]+)", _CFG).group(1))
    assert m >= 1.0
    assert "_validate_mimic_fields(cfg, pair)" in _CFG


def test_asset_actually_carries_mimic_multipliers_both_hands():
    urdf = next(_ASSET.glob("*.urdf"))
    pairs = {j.get("name"): float(j.find("mimic").get("multiplier"))
             for j in ET.parse(urdf).getroot().iter("joint") if j.find("mimic") is not None}
    for s in ("r", "l"):
        assert pairs[f"{s}_hj_thumb_3"] == pytest.approx(1.1425)
        assert pairs[f"{s}_hj_thumb_4"] == pytest.approx(0.7508)
        for f in ("index", "middle", "ring", "pinky"):
            assert pairs[f"{s}_hj_{f}_2"] == pytest.approx(1.1169)


def test_contact_filters_point_at_rigid_body_prim():
    """shaker 강체는 baseLink 하위 — 필터가 루트를 가리키면 힘이 조용히 0(관측 #0230)."""
    assert "cfg.source_contact_filter = (SOURCE_CUP_BODY,)" in _CFG
    assert "prim_path=_cfg.SOURCE_CUP_BODY, filter_prim_paths_expr=[_cfg.RECEIVER_CUP_BODY]" in _ENV
    assert "find_matching_prim_paths(expr)" in _ENV
    from pxr import Usd
    stage = Usd.Stage.Open(str(_HDGP / "assets" / "cup" / "shaker_closed_rl.usd"))
    rb = [p.GetName() for p in stage.Traverse() if "PhysicsRigidBodyAPI" in p.GetAppliedSchemas()]
    assert rb == ["baseLink"], rb


def test_mimic_blowup_terminates_episode():
    """09.14 사용자 결정: mimic 결합이 깨진 env(종속관절 속도 폭주)는 리셋한다 — 한계 여유는 처방이 아니었다."""
    assert re.search(r"mimic_runaway_dep_qd:\s*float\s*=\s*([0-9.]+)", _CFG)
    assert "terminated = runaway | mimic_runaway | self._dropped" in _ENV
    assert 'self.extras["done/mimic_runaway"]' in _ENV


def test_slow_mimic_drift_also_terminates():
    """09.14 라운드 1: 오차가 epoch 321-323(39→236 rad)·348-351(7→18 rad) 동안 속도 100 rad/s 아래로 천천히 벌어져
    속도 기준을 빠져나갔다 → 사용자 결정 "2 추가": 결합 오차 자체도 종료 조건."""
    m = re.search(r"mimic_runaway_err_rad:\s*float\s*=\s*([0-9.]+)", _CFG)
    assert m and 1.7 < float(m.group(1)) < 7.0, "정상 epoch 최대 1.7 rad · 느린 폭주 시작 7 rad 사이"
    code = _code_only(_ENV)
    assert "cfg . mimic_runaway_err_rad" in code
    assert "mimic_runaway = ( dep_qd > float ( cfg . mimic_runaway_dep_qd ) ) | ( mim_err > float ( cfg . mimic_runaway_err_rad ) )" in code
    assert 'self.extras["done/mimic_err_runaway"]' in _ENV
    assert "mimic_runaway_err_rad" in _CFG.split("def _validate_mimic_fields")[1]


def test_fingertip_tactile_obs_is_sim2real_shaped():
    """09.14 사용자 결정 "3 추가": RH56F1 실기 손끝 촉각(TouchData1.finger_forces[5], 0.01 N)에 대응하는 5칸/손.
    실기 센서는 무엇에 닿든 재므로 sim 도 컵 필터(force_matrix_w)가 아니라 손끝 링크 **전체** 접촉력(net_forces_w)."""
    rig = _code_only(_RIG)
    assert "def tip_tactile ( self )" in rig
    body = _RIG.split("def tip_tactile(self)")[1].split("\n    def ")[0]
    assert "net_forces_w" in body and "force_matrix_w" not in body
    assert "self.sensors[f][-1]" in body, "프로필 규약 (중간, 원위, 팁) — 마지막이 팁"
    env = _code_only(_ENV)
    assert env.count("self . _tactile ( self . src , noisy = True )") == 1
    assert env.count("self . _tactile ( self . rcv , noisy = True )") == 1
    assert env.count("self . _tactile ( self . src , noisy = False )") == 1
    assert env.count("self . _tactile ( self . rcv , noisy = False )") == 1
    assert re.search(r"tactile_obs_clip_n:\s*float\s*=\s*10\.0", _CFG), "실기 1024 = 10.24 N 포화"
    assert re.search(r"tactile_obs_noise_n:\s*float\s*=", _CFG)
    assert "+ f  # tactile" in _CFG, "관측 차원식에 손끝 촉각 f 칸"


def test_ctx_palm_axes_put_palm_normal_first():
    """09.15: RewardContext 계약은 "palm_axes 앞 3칸 = 손바닥 법선". RH56F1 palm_sensor 는 열 2 가 법선(URDF, 아래 테스트)이라
    열 0·1 을 그대로 넣으면 생성 보상(iter_03 orient)이 손 옆날을 컵으로 돌리는 방향을 보상한다 → env 가 [법선, 손가락 방향] 순서로 넣는다."""
    m = re.search(r"ctx_palm_normal_col:\s*int\s*=\s*(\d)", _CFG)
    m2 = re.search(r"ctx_palm_second_col:\s*int\s*=\s*(\d)", _CFG)
    assert m and int(m.group(1)) == 2
    assert m2 and int(m2.group(1)) == 1
    code = _code_only(_ENV)
    assert code.count("src_palm_axes = self . _ctx_palm_axes ( self . src )") == 1
    assert code.count("rcv_palm_axes = self . _ctx_palm_axes ( self . rcv )") == 1
    assert "palm_R ( ) [ : , : , 0 ] , self . src . palm_R ( ) [ : , : , 1 ]" not in code
    assert "palm_R ( ) [ : , : , 0 ] , self . rcv . palm_R ( ) [ : , : , 1 ]" not in code


def _rpy_matrix(rpy):
    import math
    r, p_, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p_), math.sin(p_), math.cos(y), math.sin(y)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


@pytest.mark.parametrize("s", ["r", "l"])
def test_rh56f1_palm_sensor_column2_is_palmar(s):
    """URDF 실측: palm_sensor 열 2 = 손 기저 +x, 엄지 기저가 네 손가락보다 +x 쪽(손바닥 쪽). 열 0 = 손가락 늘어선 가로(±y)."""
    root = ET.parse(_ASSET / "openarm_rh56f1_bi_rl.urdf").getroot()
    J = {j.get("name"): j for j in root.findall("joint")}
    def origin(name):
        o = J[name].find("origin")
        return [float(v) for v in o.get("xyz").split()], [float(v) for v in o.get("rpy").split()]
    for n in (f"{s}_hj_palm_1", f"{s}_hj_palm_2"):
        assert all(abs(v) < 1e-6 for v in origin(n)[1]), f"{n} 회전 없음 가정(palm_2 프레임 = 기저 프레임)"
    assert J[f"{s}_hj_palm_sensor"].find("parent").get("link") == f"{s}_hl_palm_2"
    R = _rpy_matrix(origin(f"{s}_hj_palm_sensor")[1])
    col = lambda k: [R[i][k] for i in range(3)]
    assert all(abs(a - b) < 1e-3 for a, b in zip(col(2), (1.0, 0.0, 0.0))), col(2)
    assert abs(abs(col(0)[1]) - 1.0) < 1e-3, col(0)
    thumb_x = origin(f"{s}_hj_thumb_1")[0][0]
    fingers_x = [origin(f"{s}_hj_{f}_1")[0][0] for f in ("index", "pinky")]
    assert thumb_x > max(fingers_x) + 0.01, "엄지 기저가 +x(손바닥) 쪽"


@pytest.mark.parametrize("s", ["r", "l"])
def test_rh56f1_contact_bodies_carry_the_distal_collider(s):
    """09.15: 손끝 마디 `_2`·`_sensor`·`_tip`(엄지 thumb_4·sensor·tip)은 벤더 STL 이 같은 입체라 collider 는 `_sensor` 에만 둔다.
    env 가 읽는 접촉 링크는 전부 collider 가 있어야 하고(없으면 접촉이 조용히 0), 사본 링크엔 없어야 한다(겹친 강체에 접촉이 나뉜다)."""
    P = pytest.importorskip("openarm.agnostic.tasks.pour_fabric_mimic.robot_profiles")
    prof = P.RH56F1_RIGHT_FAB if s == "r" else P.RH56F1_LEFT_FAB
    root = ET.parse(_ASSET / "openarm_rh56f1_bi_rl.urdf").getroot()
    has_col = {link.get("name"): bool(link.findall("collision")) for link in root.findall("link")}
    bodies = [b for bs in prof.finger_sensor_bodies.values() for b in bs] + [prof.palm_body]
    assert all(has_col[b] for b in bodies), [b for b in bodies if not has_col[b]]
    copies = [f"{s}_hl_thumb_4", f"{s}_hl_thumb_tip",
              *(f"{s}_hl_{f}_{k}" for f in ("index", "middle", "ring", "pinky") for k in ("2", "tip"))]
    assert not any(has_col[c] for c in copies), [c for c in copies if has_col[c]]


# 원본 계약 중 그대로 유지돼야 하는 것(보상 없음 · 성공은 env · a=0 = 앵커)
def test_inherited_contracts_hold():
    assert "load_reward_fn" in _ENV and "RewardContext(" in _ENV
    assert not re.search(r"_weight\b", _ENV)
    assert "success=self._success_now" in _ENV and "& (~self._cups_nested)" in _ENV
    assert "torch.where(a6 >= 0.0, a6 * self.delta_hi, -a6 * self.delta_lo)" in _RIG
    assert "torch.where(delta > 0.0, delta * g, delta)" in _RIG and "hold & (delta > 0.0)" in _RIG


# =============================================================================
# 차원 (Isaac 필요)
# =============================================================================
def test_dims_from_resolve_cfg():
    pytest.importorskip("pxr"); pytest.importorskip("isaaclab")
    from openarm.agnostic.tasks.pour_fabric_mimic import pour_fabric_env_cfg as C
    cfg = C.PourFabricMimicEnvCfg()
    assert cfg.action_space == 2 * (6 + 6) == 24
    per = 0
    for p in (cfg_pair := _bm.get_pair(cfg.pair_name)).source, cfg_pair.receiver:
        a, h, f = p.num_arm_joints, p.num_hand_joints, len(p.finger_sensor_bodies)
        per += 2 * a + h + 3 + 6 + 3 * f + 3 + 3 * f + h + 3 + f  # 09.14 손끝 촉각 f 칸
    assert cfg.observation_space == per + 6 + 24 == 182
    assert cfg.state_space == 182 + 12 + 4 + 3 + 12 + 1 + 10 == 224
    from openarm.agnostic.tasks.pour_fabric_mimic import config as reg
    assert reg.REGISTERED == {"rh": "open-rh_b_pour_fab_mimic"}


def test_thumb_rim_approach_is_logged_not_rewarded():
    """09.15 사용자 "다음부터 지표로깅으로 확인 가능하게": 엄지 입구 걸림 접근을 영상 없이 본다.
    양손 near 비율·near 중 엄지 입구 위 비율·엄지-입구 높이(mm) 를 로그로만 낸다 — obs·RewardContext 에는 넣지 않는다."""
    for side in ("src", "rcv"):
        for k in ("near_rate", "thumb_over_rim_near", "thumb_above_rim_mm_near"):
            assert f'f"task/{{side}}_{k}"' in _ENV or f"task/{side}_{k}" in _ENV
    assert _ENV.count("self._log_thumb_rim(") == 2
    for f in ("thumb_rim_near_m", "thumb_rim_band_m", "thumb_rim_radial_margin_m"):
        assert re.search(rf"{f}:\s*float\s*=", _CFG), f
    ctx_block = _ENV.split("return RewardContext(")[1].split("\n        )")[0]
    assert "thumb_rim" not in ctx_block and "thumb_over_rim" not in ctx_block

"""보상 게이트용 에피소드 단계 래치 — 순수 torch (Isaac 불요).

잠그는 것(09.15 사용자 "기본 핸드 자세에서 컵으로 접근 → 접근한 상태에서 인벨롭 파지 → 리프트"):
  · ★09.15 23:2x 사용자 결정 "C자 사전파지로 재정의" — 기본 손 자세의 엄지(thumb_2 대향 잠금 −1.57, 나머지 0)는 손바닥 법선
    방향으로 손바닥면보다 118 mm 앞으로 곧게 뻗는다(URDF FK). "손바닥 중심 2 cm + 기본 자세 + 무접촉"은 기하상 성립하지 않았고,
    i01 의 접근 래치는 엄지로 컵을 누르며 섰다(엄지 접촉 0.33–0.45). 접근 완료 = 컵이 엄지와 네 손가락 사이에 든 C자:
    손바닥면↔컵 옆면(법선 방향) ≤ 2 cm · 컵 축이 palm_ee 보다 손가락 방향으로 R−5 mm~R+20 mm 앞 · 손바닥 중심이 띠 높이 안 ·
    시작 방향(법선 +y · 손가락 +x, cos ≥ 0.7) · 움직이는 손 관절 기본 자세 ±0.15 · 손가락·엄지 무접촉 — 여섯 동시.
  · 인벨롭 완료 = 접근 완료 **뒤에만** · 손바닥 + 엄지 + 닿은 손가락 ≥ 4 가 5 스텝 연속.
  · 둘 다 에피소드 래치(한 번 서면 리셋까지 유지) — 끊기면 연속 카운트만 0.
  · 가까운 출발 IK 표(reach leaf cfg)가 컵 8종마다 C자 완료 자리 조금 앞(손바닥면 4.5 cm · 컵 축 R+2.5 cm · 띠 0.8 H)에 손을 둔다.
  · ctx 주석(생성기가 읽는 환경 설명)이 같은 수치를 적는다.

실행:
    cd hdgp && PYTHONPATH=source/openarm python3 -m pytest \
        source/openarm/openarm/agnostic/tasks/grasp_fj_t2r/tests -q
"""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import pytest
import torch

from openarm.agnostic.tasks.grasp_fj_t2r import grasp_gates as G

_HERE = Path(__file__).resolve().parent.parent
_CTX = (_HERE / "t2r" / "context.py").read_text(encoding="utf-8")
_CFG = (_HERE / "grasp_fj_t2r_env_cfg.py").read_text(encoding="utf-8")
_URDF_TOOL = Path.home() / "rl_ws" / "urdf" / "tools" / "solve_arm_reset_pose.py"
_URDF = Path.home() / "rl_ws" / "urdf" / "generated" / "rl" / "openarm_dg5f-m-short-tl_bi_rl.urdf"
#: palm_ee = r_hl_palm + (0.028, 0, 0.040) — URDF `r_hj_palm_ee` (palm_frame.py 가 부팅에서 읽는 값)
_PALM_EE_OFF = (0.028, 0.0, 0.040)
#: 작업면 높이 [m] — `fj_core_cfg.table_surface_z` (env_v1 CAD, memory real-table-30mm-higher-than-sim)
_TABLE_Z = 0.205


def _zeros(n):
    return (torch.zeros(n, dtype=torch.bool), torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.bool))


def _reach_cfg_value(name: str):
    tree = ast.parse(_CFG)
    cls = next(c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "GraspFJT2RReachEnvCfg")
    node = next(s for s in cls.body if isinstance(s, ast.AnnAssign) and getattr(s.target, "id", "") == name)
    return ast.literal_eval(node.value)


def _ik_tool():
    if not (_URDF_TOOL.exists() and _URDF.exists()):
        pytest.skip("urdf 도구·자산이 없는 호스트")
    spec = importlib.util.spec_from_file_location("_solve_arm_reset_pose", _URDF_TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Urdf(_URDF)


def _palm_frame(urdf, arm_q):
    T = urdf.pose("r_hl_palm", {f"r_aj_{i + 1}": float(v) for i, v in enumerate(arm_q)})
    R = torch.tensor(T[:3, :3], dtype=torch.float64)
    p = torch.tensor(T[:3, 3], dtype=torch.float64) + R @ torch.tensor(_PALM_EE_OFF, dtype=torch.float64)
    return p, R


def test_hand_orientation_is_the_worse_of_normal_and_finger_alignment():
    normal = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    finger = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(G.hand_orientation(normal, finger), torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)


def test_c_pregrasp_geometry_measures_palm_plane_finger_offset_and_band_height():
    # palm_ee 원점 · 법선 +y · 손가락 +x · 컵 축 z 가 (0.07, 0.08) · R 0.06 · 띠 중심 z 0.03
    #   → 손바닥면↔옆면 0.08−0.06 = 0.02 · 손가락 방향 0.07−0.06 = +0.01 · 손바닥 높이(띠 중심 기준) −0.03
    palm = torch.zeros(1, 3)
    normal, finger = torch.tensor([[0.0, 1.0, 0.0]]), torch.tensor([[1.0, 0.0, 0.0]])
    cup, axis = torch.tensor([[0.07, 0.08, 0.03]]), torch.tensor([[0.0, 0.0, 1.0]])
    gap, along, height = G.c_pregrasp_geometry(palm, normal, finger, cup, axis, torch.tensor([0.06]))
    assert torch.allclose(gap, torch.tensor([0.02]), atol=1e-6)
    assert torch.allclose(along, torch.tensor([0.01]), atol=1e-6)
    assert torch.allclose(height, torch.tensor([-0.03]), atol=1e-6)


def _ok(n, **over):
    kw = dict(plane_gap=torch.full((n,), 0.01), along_offset=torch.zeros(n), height=torch.zeros(n),
              half_height=torch.full((n,), 0.05), orient=torch.full((n,), 0.9), pose_dev=torch.zeros(n),
              digit_touch=torch.zeros(n, dtype=torch.bool))
    kw.update(over)
    return kw


def test_approach_conditions_name_each_check_for_the_log():
    assert G.APPROACH_CONDITIONS == ("gap", "along", "height", "orient", "pose", "no_touch")
    # env 0 은 전부 통과, env i(1..6) 는 조건 i−1 하나만 틀린다
    c = G.approach_conditions(**_ok(
        7,
        plane_gap=torch.tensor([0.01, 0.03, 0.01, 0.01, 0.01, 0.01, 0.01]),
        along_offset=torch.tensor([0.0, 0.0, -0.01, 0.0, 0.0, 0.0, 0.0]),
        height=torch.tensor([0.0, 0.0, 0.0, -0.06, 0.0, 0.0, 0.0]),
        orient=torch.tensor([0.9, 0.9, 0.9, 0.9, 0.5, 0.9, 0.9]),
        pose_dev=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.0]),
        digit_touch=torch.tensor([False, False, False, False, False, False, True])))
    assert c.shape == (7, 6) and c.dtype == torch.bool
    assert c[0].all()
    for i in range(1, 7):
        assert c[i].tolist() == [j != i - 1 for j in range(6)], i


def test_plane_gap_window_rejects_a_cup_behind_the_hand():
    # ★09.15 서버 부팅 스모크: 먼 출발 무작위 행동에서 gap 조건이 0.28 통과 — 상한만 있으면 컵이 손등 뒤(음수 간극)여도 선다.
    c = G.approach_conditions(**_ok(4, plane_gap=torch.tensor([-0.02, -0.009, 0.019, 0.021])))
    assert c[:, G.APPROACH_CONDITIONS.index("gap")].tolist() == [False, True, True, False]


def test_along_window_keeps_the_cup_clear_of_the_thumb_and_under_the_fingers():
    c = G.approach_conditions(**_ok(4, along_offset=torch.tensor([-0.006, -0.004, 0.019, 0.021])))
    assert c[:, G.APPROACH_CONDITIONS.index("along")].tolist() == [False, True, True, False]


def test_approach_latch_sets_when_all_conditions_hold_and_stays():
    n = 2
    quiet = dict(palm_touch=torch.zeros(n, dtype=torch.bool), finger_touch=torch.zeros(n, 5, dtype=torch.bool))
    appr, cnt, env = G.update_gates(*_zeros(n), approach_ok=torch.tensor([True, False]), **quiet)
    assert appr.tolist() == [True, False] and not env.any() and not cnt.any()
    appr, cnt, env = G.update_gates(appr, cnt, env, approach_ok=torch.tensor([False, False]), **quiet)
    assert appr.tolist() == [True, False], "접근 래치는 조건이 풀려도 유지된다"


def test_envelope_only_after_approach_and_after_consecutive_hold_steps():
    n = 3
    palm = torch.tensor([True, True, True])
    fingers = torch.zeros(n, 5, dtype=torch.bool)
    fingers[:, :4] = True                               # 엄지 + 검지·중지·약지
    no_thumb = fingers.clone()
    no_thumb[2, 0] = False
    no_thumb[2, 4] = True
    # env 0 은 접근 완료, env 1 은 접근 전(손바닥은 이미 닿음), env 2 는 접근 완료지만 엄지가 없다
    appr, cnt, env = torch.tensor([True, False, True]), torch.zeros(n, dtype=torch.long), torch.zeros(n, dtype=torch.bool)
    off = torch.zeros(n, dtype=torch.bool)
    for step in range(G.ENVELOPE_HOLD_STEPS):
        appr, cnt, env = G.update_gates(appr, cnt, env, approach_ok=off, palm_touch=palm, finger_touch=no_thumb)
        if step < G.ENVELOPE_HOLD_STEPS - 1:
            assert not env.any(), step
    assert env.tolist() == [True, False, False]
    assert appr.tolist() == [True, False, True]


def test_a_break_resets_the_hold_count_but_a_set_latch_stays():
    n = 1
    held, loose = torch.ones(n, 5, dtype=torch.bool), torch.zeros(n, 5, dtype=torch.bool)
    off = torch.tensor([False])
    appr, cnt, env = torch.tensor([True]), torch.zeros(n, dtype=torch.long), torch.tensor([False])
    for _ in range(G.ENVELOPE_HOLD_STEPS - 1):
        appr, cnt, env = G.update_gates(appr, cnt, env, approach_ok=off, palm_touch=torch.tensor([True]), finger_touch=held)
    appr, cnt, env = G.update_gates(appr, cnt, env, approach_ok=off, palm_touch=torch.tensor([False]), finger_touch=held)
    assert cnt.tolist() == [0] and not env.any()
    for _ in range(G.ENVELOPE_HOLD_STEPS):
        appr, cnt, env = G.update_gates(appr, cnt, env, approach_ok=off, palm_touch=torch.tensor([True]), finger_touch=held)
    assert env.tolist() == [True]
    appr, cnt, env = G.update_gates(appr, cnt, env, approach_ok=off, palm_touch=torch.tensor([False]), finger_touch=loose)
    assert env.tolist() == [True] and cnt.tolist() == [0]


def test_update_returns_new_tensors_without_mutating_inputs():
    appr, cnt, env = _zeros(2)
    out = G.update_gates(appr, cnt, env, approach_ok=torch.tensor([True, True]),
                         palm_touch=torch.tensor([True, True]), finger_touch=torch.ones(2, 5, dtype=torch.bool))
    assert not appr.any() and not cnt.any() and not env.any()
    assert out[0].all() and (out[1] == 1).all()


def test_pose_deviation_ignores_locked_joints():
    q = torch.tensor([[0.10, 0.90, 0.30], [0.12, 0.10, 0.50]])
    default = torch.tensor([0.10, 0.10, 0.30])
    movable = torch.tensor([True, False, True])
    assert torch.allclose(G.pose_deviation(q, default, movable), torch.tensor([0.0, 0.20]), atol=1e-6)


def test_gate_directions_are_the_hand_orientation_of_the_reach_start_pose():
    # ★09.15 사용자 "palm_ee_x(컵 쪽)·손가락 방향(palm_ee_z)" — 리셋 자세 FK(short-tl URDF)가 이미 법선 +y·손가락 +x 다.
    urdf = _ik_tool()
    q = [float(v) for v in re.search(r"arm_reset_joint_pos_override: tuple = \(([^)]*)\)", _CFG).group(1).split(",")]
    _, R = _palm_frame(urdf, q)
    assert float(R[:, 0] @ torch.tensor(G.APPROACH_PALM_NORMAL_DIR, dtype=torch.float64)) > 0.99
    assert float(R[:, 2] @ torch.tensor(G.APPROACH_FINGER_DIR, dtype=torch.float64)) > 0.99


def test_near_start_poses_put_the_hand_just_short_of_the_c_pregrasp_for_every_cup():
    # ★09.15 사용자 "시작 상태 커리큘럼 + env 고정" — 컵 8종마다 IK 한 자세. 컵 스폰 xy 는 매 리셋 ±2 cm 흔들린다.
    #   ★서버 스모크(32 중 4 env 가 첫 스텝에 접근 래치): 손바닥면 3 cm 는 −2 cm 흔들림에 1 cm 가 돼 공짜 래치였다 →
    #   손바닥면 4.5 cm(흔들려도 ≥ 2.5 cm > 2 cm 창) · 컵 축 R+2.5 cm(≥ R+0.5 cm 라 엄지와 겹치지 않음).
    from openarm.agnostic.modules import object_bank as ob
    from openarm.agnostic.modules.robot_profiles import TESOLLO_RIGHT_SHORT_TL as prof

    bank = ob.get("cup_family")
    species = _reach_cfg_value("near_start_species")
    table = _reach_cfg_value("near_start_arm_q")
    assert tuple(species) == tuple(s.id for s in bank.specs), "IK 표 순서 = 뱅크 순서(env_id % 8)"
    assert len(table) == len(bank.specs) and all(len(row) == 7 for row in table)
    urdf = _ik_tool()
    cx, cy = prof.object_spawn_center
    for spec, row in zip(bank.specs, table):
        palm, R = _palm_frame(urdf, row)
        cup = torch.tensor([[cx, cy, _TABLE_Z + spec.origin_offset_z]], dtype=torch.float64)
        gap, along, height = G.c_pregrasp_geometry(
            palm.unsqueeze(0), R[:, 0].unsqueeze(0), R[:, 2].unsqueeze(0), cup,
            torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64), torch.tensor([spec.grasp_radius_m], dtype=torch.float64))
        assert abs(float(gap) - 0.045) < 0.002, (spec.id, float(gap))
        assert abs(float(along) - 0.025) < 0.002, (spec.id, float(along))
        assert abs(float(height) - 0.8 * spec.grasp_halfheight_m) < 0.002, (spec.id, float(height))
        assert float(R[1, 0]) > 0.99 and float(R[0, 2]) > 0.99, (spec.id, "시작 방향")


def test_context_comments_state_the_same_gate_numbers():
    for tok in ("hand_default_q_norm", "approach_done", "envelope_done", "palm_finger_dir",
                "within 2 cm of the cup's side", "between -0.01 m and 0.02 m", "cup_radius - 0.005 m",
                "cup_radius + 0.02 m",
                "within about 45 degrees of +y", "within about 45 degrees of +x", "within 0.3 of hand_default_q_norm",
                # ★09.16 사용자 "가까운 출발 = 접근 완료로 시작" — 생성기가 래치가 첫 스텝부터 설 수 있음을 알아야 한다
                "approach_done already set on the first step",
                "no finger or thumb link touched the cup", "about 0.12 m", "at least 4", "5 consecutive steps", "0.1 N"):
        assert tok in _CTX, tok
    assert G.APPROACH_PLANE_GAP_M == 0.02 and G.APPROACH_PLANE_GAP_MIN_M == -0.01
    # ★09.16 사용자 "0.3 으로 완화" — 기본 자세가 액션 하한(a = −1)이고 탐색 σ = 1 이라 0.15 는 13관절 동시 통과 ≈ 0.8 %(i02 e240)
    assert G.APPROACH_POSE_TOL == 0.3 and G.APPROACH_ORIENT_MIN == 0.7
    assert G.APPROACH_ALONG_MIN_OFFSET_M == -0.005 and G.APPROACH_ALONG_MAX_OFFSET_M == 0.02
    assert G.APPROACH_PALM_NORMAL_DIR == (0.0, 1.0, 0.0) and G.APPROACH_FINGER_DIR == (1.0, 0.0, 0.0)
    assert G.ENVELOPE_MIN_DIGITS == 4 and G.ENVELOPE_HOLD_STEPS == 5 and G.TOUCH_N == 0.1

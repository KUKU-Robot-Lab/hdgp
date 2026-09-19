"""09.20 좌팔판(grasp_fj_rand_left) — 우 rand 의 y 반전 미러 계약. isaaclab 없이 돈다(시스템 python3 pytest).

rl-mirror-port 단계별 확인:
  1 기구학 — 좌 시작 자세 FK 가 우 시작 자세의 y 반전(URDF 가 있는 호스트에서만).
  2 부호 — 프로필 손 잠금·open 자세·유휴 우팔이 `_HAND_SIGN_L`·`_ARM_SIGN_L` 미러.
  3 제어·ctx — env 가 법선 방향·손 정규화 방향·손바닥 bbox 를 측(side)으로 고른다.
  4 보상 — 시드 보상은 우 i01 과 시작 방향 한 줄만 다르다.
  5 차원 — 손 관절 19 · 이름 순서가 우와 같은 접미사 순서.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from openarm.agnostic.modules import robot_profiles as RP
from openarm.agnostic.modules.source_contract import _fn_block

from .. import grasp_gates as G
from ..t2r import prompts as P
from ..t2r.context import context_stub_source

_HERE = Path(__file__).resolve().parents[1]
_HDGP = Path(__file__).resolve().parents[7]
_CFG = (_HERE / "grasp_fj_t2r_env_cfg.py").read_text(encoding="utf-8")
_ENV = (_HERE / "grasp_fj_t2r_env.py").read_text(encoding="utf-8")
_REG = (_HERE / "config" / "__init__.py").read_text(encoding="utf-8")
_URDF_TOOL = Path.home() / "rl_ws" / "urdf" / "tools" / "solve_arm_reset_pose.py"
_URDF = Path.home() / "rl_ws" / "urdf" / "generated" / "rl" / "openarm_dg5f-m-short-tl_bi_rl.urdf"

_R = RP.PROFILES["tesollo_right_short_tl"]
_L = RP.PROFILES["tesollo_left_short_tl"]
_SIGN = {f"{f}_{j}": s for (f, j), s in zip(((f, j) for f in RP._FINGERS for j in range(1, 5)), RP._HAND_SIGN_L)}


def _cls_value(cls_name: str, field: str):
    tree = ast.parse(_CFG)
    cls = next(c for c in tree.body if isinstance(c, ast.ClassDef) and c.name == cls_name)
    node = next(s for s in cls.body if isinstance(s, ast.AnnAssign) and getattr(s.target, "id", "") == field)
    return node.value


def _literal(cls_name: str, field: str):
    return ast.literal_eval(_cls_value(cls_name, field))


def _sfx(name: str) -> str:
    return name.split("_hj_", 1)[1]


# ---------------------------------------------------------------- 프로필(Step 2)
def test_left_tl_profile_is_the_sign_mirror_of_the_right_tl_hand():
    assert _L.num_hand_joints == 19 and len(_L.hand_joint_names) == 19
    assert [_sfx(n) for n in _L.hand_joint_names] == [_sfx(n) for n in _R.hand_joint_names]
    assert all(n.startswith("l_hj_") for n in _L.hand_joint_names) and "l_hj_thumb_1" not in _L.hand_joint_names
    for pose in ("hand_open_pose", "hand_grip_pose"):
        for n, vl, vr in zip(_R.hand_joint_names, getattr(_L, pose), getattr(_R, pose)):
            assert vl == pytest.approx(_SIGN[_sfx(n)] * vr, abs=1e-9), (pose, n)
    assert _L.usd_relpath == _R.usd_relpath and _L.palm_body == "l_hl_palm"
    assert not any("thumb_1" in k for k in _L.init_joint_pos)


def test_left_tl_action_limits_mirror_the_right_locks():
    ov = _L.hand_action_limit_override
    assert ov[r"l_hj_thumb_2$"] == (1.5608, 1.5808)
    assert _R.hand_action_limit_override[r"r_hj_thumb_2$"] == (-1.5808, -1.5608)
    assert ov[r"l_hj_thumb_[34]$"] == (None, 0.0)                 # 좌 엄지는 음의 각으로 조인다
    assert ov[r"l_hj_(index|middle|ring|pinky)_[34]$"] == (0.0, None)
    assert ov[r"l_hj_pinky_2$"] == (-0.01, 0.01)
    assert not any("thumb_1" in k or k.startswith("r_") for k in ov)


def test_idle_right_arm_is_the_mirrored_tucked_idle_left_arm_of_the_right_task():
    for i in range(1, 8):
        s = RP._ARM_SIGN_L[i - 1]
        assert _L.init_joint_pos[f"r_aj_{i}"] == pytest.approx(s * _R.init_joint_pos[f"l_aj_{i}"], abs=1e-9), i


# ---------------------------------------------------------------- cfg(Step 1·5)
def test_left_leaf_is_the_y_mirror_of_the_right_rand_leaf():
    assert "class GraspFJT2RRandLeftEnvCfg(GraspFJT2RRandEnvCfg)" in _CFG
    assert '("short_l", "grasp_fj_t2r_rand"): GraspFJT2RRandLeftEnvCfg' in _REG
    assert _literal("GraspFJT2RRandLeftEnvCfg", "profile_name") == "tesollo_left_short_tl"
    assert _literal("GraspFJT2RRandLeftEnvCfg", "hand_side") == "l"
    assert _literal("GraspFJT2RRightShortEnvCfg", "hand_side") == "r"
    cx, cy = _literal("GraspFJT2RRandEnvCfg", "object_spawn_center_override")
    assert _literal("GraspFJT2RRandLeftEnvCfg", "object_spawn_center_override") == (cx, -cy)
    x0, x1, y0, y1, zc = _literal("GraspFJT2RRandEnvCfg", "spawn_reject_hand_box")
    assert _literal("GraspFJT2RRandLeftEnvCfg", "spawn_reject_hand_box") == (x0, x1, -y1, -y0, zc)
    holes_r = _literal("GraspFJT2RRandEnvCfg", "spawn_reject_holes")
    holes_l = _literal("GraspFJT2RRandLeftEnvCfg", "spawn_reject_holes")
    assert sorted(holes_l) == sorted((x, -y + 0.0, r) for x, y, r in holes_r)
    sign = ast.literal_eval(_CFG.split("_ARM_SIGN_L = ", 1)[1].split("\n", 1)[0])
    assert tuple(sign) == RP._ARM_SIGN_L
    assert "zip(_ARM_SIGN_L, (-1.1974, 0.6707, 0.1866, 1.7310, 0.6920, 0.0416, 0.9460))" in _CFG
    assert _literal("GraspFJT2RReachEnvCfg", "arm_reset_joint_pos_override") == (
        -1.1974, 0.6707, 0.1866, 1.7310, 0.6920, 0.0416, 0.9460)


def _ik_tool():
    if not (_URDF_TOOL.exists() and _URDF.exists()):
        pytest.skip("urdf 도구·자산이 없는 호스트")
    spec = importlib.util.spec_from_file_location("_solve_arm_reset_pose", _URDF_TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Urdf(_URDF)


def test_left_start_pose_fk_is_the_y_mirror_with_palm_normal_minus_y():
    import numpy as np
    urdf = _ik_tool()
    q_r = _literal("GraspFJT2RReachEnvCfg", "arm_reset_joint_pos_override")
    q_l = tuple(s * v for s, v in zip(RP._ARM_SIGN_L, q_r))
    off = np.array((0.028, 0.0, 0.040))

    def frame(side, q):
        T = urdf.pose(f"{side}_hl_palm", {f"{side}_aj_{i + 1}": float(v) for i, v in enumerate(q)})
        return T[:3, 3] + T[:3, :3] @ off, T[:3, :3]

    p_r, R_r = frame("r", q_r)
    p_l, R_l = frame("l", q_l)
    M = np.diag([1.0, -1.0, 1.0])
    assert np.linalg.norm(M @ p_r - p_l) < 1e-5
    assert R_l[:, 0] @ np.array(G.APPROACH_PALM_NORMAL_DIR_LEFT) > 0.999     # 법선 = palm_ee x
    assert R_l[:, 2] @ np.array(G.APPROACH_FINGER_DIR) > 0.999               # 손가락 = palm_ee z


# ---------------------------------------------------------------- env ctx(Step 3)
def test_gate_orientation_takes_the_side_normal():
    normal, finger = torch.tensor([[0.0, -1.0, 0.0]]), torch.tensor([[1.0, 0.0, 0.0]])
    assert G.hand_orientation(normal, finger, G.APPROACH_PALM_NORMAL_DIR_LEFT).item() == pytest.approx(1.0)
    assert G.hand_orientation(normal, finger).item() == pytest.approx(-1.0)


def test_env_orients_every_hand_normalisation_and_the_palm_box_by_side():
    init, ctx = _fn_block(_ENV, "__init__"), _fn_block(_ENV, "_build_context")
    assert "self._t2r_default_q_norm = _hand_norm(self._t2r_norm_flip, " in init
    assert "q_norm = _hand_norm(self._t2r_norm_flip, ((q - lo) / span)" in ctx
    assert "hand_target_norm=_hand_norm(self._t2r_norm_flip, " in ctx
    assert "hand_orientation(R[:, :, 0], R[:, :, 2], self._t2r_normal_dir)" in ctx
    assert 'if self._t2r_side == "l":' in ctx and "(_lo[0], -_hi[1], _lo[2]), (_hi[0], -_lo[1], _hi[2])" in ctx
    # 프로필 손 ≠ hand_side 면 부팅에서 죽는다
    assert "hand_side {self._t2r_side!r} ≠ 프로필" in init


def test_left_flip_joints_are_exactly_the_minus_one_mirror_signs():
    flip = {s for s, v in _SIGN.items() if v < 0 and s != "thumb_1"}
    assert P.LEFT_FLIP_JOINTS == flip
    ranges = P.hand_joint_ranges("l")
    assert tuple(n for n, _, _ in ranges) == _L.hand_joint_names
    for (nl, lo_l, hi_l), (nr, lo_r, hi_r) in zip(ranges, P.HAND_JOINT_RANGES):
        if _sfx(nl) in flip:
            assert (lo_l, hi_l) == pytest.approx((-hi_r, -lo_r))
        else:
            assert (lo_l, hi_l) == (lo_r, hi_r)


# ---------------------------------------------------------------- 프롬프트·보상(Step 4·6)
def test_left_prompt_describes_the_left_hand_and_minus_y_start_normal():
    spec = P.PromptSpec(task="t", variant="rand_left")
    txt = P.render_prompt(spec)
    assert "(the left arm)" in txt and "(the right arm)" not in txt
    assert "l_hj_thumb_3" in txt and "r_hj_" not in txt
    assert "y between 0.00 m and 0.30 m" in txt and "cup's +y side" in txt
    assert "In the start pose palm_normal points along -y" in txt
    assert P.FLEX_RULE["l"] in txt
    right = P.render_prompt(P.PromptSpec(task="t", variant="rand"))
    assert "In the start pose palm_normal points along +y" in right and "(the right arm)" in right
    stub = context_stub_source("l")
    assert "left Tesollo" in stub and "within about 45 degrees of -y" in stub


def test_left_task_text_differs_only_in_the_y_direction():
    r = (_HDGP / "scripts/reward_gen/tasks/grasp_fj_rand.txt").read_text(encoding="utf-8")
    l_ = (_HDGP / "scripts/reward_gen/tasks/grasp_fj_rand_left.txt").read_text(encoding="utf-8")
    assert l_ == r.replace("ctx.palm_normal along +y", "ctx.palm_normal along -y").replace(
        "cup's -y side", "cup's +y side")


def test_left_seed_reward_is_the_right_i01_with_only_the_start_normal_flipped():
    root = _HDGP / "reward_gen"
    seed = root / "grasp_fj_rand_left" / "iter_00"
    if not (seed / "compute_reward.py").exists():
        pytest.skip("좌판 iter_00 ingest 전")
    right = (root / "grasp_fj_rand" / "iter_01" / "compute_reward.py").read_text(encoding="utf-8").splitlines()
    left = (seed / "compute_reward.py").read_text(encoding="utf-8").splitlines()
    diff = [(a, b) for a, b in zip(right, left) if a != b]
    assert len(right) == len(left) and len(diff) == 2, diff
    assert diff[1] == ("    ang_n = torch.acos(torch.clamp(n[:, 1], -1.0, 1.0))",
                       "    ang_n = torch.acos(torch.clamp(-n[:, 1], -1.0, 1.0))")
    assert json.loads((seed / "meta.json").read_text(encoding="utf-8"))["variant"] == "rand_left"

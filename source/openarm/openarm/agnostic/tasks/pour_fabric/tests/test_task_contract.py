"""pour_fabric 태스크 계약 (09.13 재작성 — 잡기부터 시작 · 보상은 t2r 생성 코드).

Isaac 없이 도는 소스 계약 + Isaac 있을 때만 도는 차원 대조.
"""
import ast
import re
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parents[1]
_ENV = (_PKG / "pour_fabric_env.py").read_text()
_RIG = (_PKG / "side_rig.py").read_text()
_CFG = (_PKG / "pour_fabric_env_cfg.py").read_text()
_ALL = {"env": _ENV, "rig": _RIG, "cfg": _CFG}


# =============================================================================
# robot-agnostic 계약
# =============================================================================
@pytest.mark.parametrize("name", sorted(_ALL))
def test_no_robot_specific_literals(name):
    # fabrics_sim 모듈 이름(openarm_tesollo_pose_fabric)은 클래스 탐색용 import 라 제외.
    src = "\n".join(l for l in _ALL[name].splitlines() if not l.startswith("import fabrics_sim"))
    for lit in ("r_aj_", "l_aj_", "r_hj_", "l_hj_", "r_hl_", "l_hl_", "tesollo", "dg5f"):
        assert lit not in src, f"{name}: 로봇 리터럴 '{lit}' — 프로필로 옮길 것"


def test_env_does_not_import_specific_profiles():
    assert "TESOLLO_" not in _ENV and "TESOLLO_" not in _RIG
    assert "from . import bimanual" in _ENV


def test_fabric_state_is_not_in_observation():
    """actor obs 에 fabric_q / 비드 ground truth 금지(실기에 없다)."""
    fn = next(n for n in ast.parse(_ENV).body[-1].body
              if isinstance(n, ast.FunctionDef) and n.name == "_get_observations")
    src = ast.get_source_segment(_ENV, fn)
    obs_part = src.split("obs = torch.cat")[0]
    assert "fabric_q" not in obs_part
    assert "beads." not in obs_part and "_prev_in_" not in obs_part


# =============================================================================
# 보상은 생성 코드 — env 에는 보상 항이 없다
# =============================================================================
def test_env_has_no_reward_terms_of_its_own():
    assert "load_reward_fn" in _ENV and "call_reward_fn" in _ENV
    assert "RewardContext(" in _ENV
    assert not re.search(r"_weight\b", _ENV), "env 에 보상 가중치가 있다 — 생성 코드로 옮길 것"
    assert not re.search(r"_weight\b", _CFG)


def test_success_is_env_judged_not_reward():
    """성공(fill·spill·xy)은 env 가 판정해 ctx.success 로 넘긴다 — 보상이 못 바꾼다."""
    assert "success_fill_ratio" in _ENV and "success_spill_max" in _ENV
    assert "success=self._success_now" in _ENV


def test_reward_is_zero_during_hold():
    assert "(~self._hold_mask()).float()" in _ENV


def test_no_sim_mutation_during_reward():
    fn = next(n for n in ast.parse(_ENV).body[-1].body
              if isinstance(n, ast.FunctionDef) and n.name == "_get_rewards")
    src = ast.get_source_segment(_ENV, fn)
    for bad in ("write_root_state_to_sim", "write_joint_state_to_sim", "set_joint_position_target"):
        assert bad not in src, f"_get_rewards 가 sim 을 변경한다: {bad}"


def test_success_does_not_terminate():
    fn = next(n for n in ast.parse(_ENV).body[-1].body
              if isinstance(n, ast.FunctionDef) and n.name == "_get_dones")
    src = ast.get_source_segment(_ENV, fn)
    assert "_success" not in src


def test_drop_terminates():
    assert "terminated = runaway | self._dropped" in _ENV


# =============================================================================
# 제어 규약 (grasp_s2r 이식분)
# =============================================================================
def test_palm_action_is_anchor_plus_delta():
    assert "raw = self.anchor + delta" in _RIG
    # a=0 = 앵커 (비대칭 박스라 선형 매핑 금지)
    assert "torch.where(a6 >= 0.0, a6 * self.delta_hi, -a6 * self.delta_lo)" in _RIG


def test_synergy_release_is_never_gated_or_frozen():
    """닫는 방향만 게이트·동결 — 푸는 방향은 항상 허용."""
    assert "torch.where(delta > 0.0, delta * g, delta)" in _RIG
    assert "hold & (delta > 0.0)" in _RIG


def test_fabric_hand_state_is_synced_from_synergy():
    assert "rig.sync_fabric_hand()" in _ENV
    assert "self.fabric_q[:, n_arm:] = self.syn_target[:, self.syn_to_fab_idx]" in _RIG


def test_close_gate_releases_after_grasp():
    assert "torch.where(grasped, torch.ones_like(g), g)" in _ENV


def test_oppose_delta_is_mirrored_per_side():
    assert "oppose_sign=1.0" in _ENV and "oppose_sign=-1.0" in _ENV
    assert "float(cfg.oppose_grip_delta_rad) * float(oppose_sign)" in _RIG


def test_contact_sensors_one_per_body_own_cup_only():
    assert 'filter_prim_paths_expr=flt' in _ENV
    assert "cfg.source_contact_filter = (SOURCE_CUP_PRIM,)" in _CFG
    assert "cfg.receiver_contact_filter = (RECEIVER_CUP_PRIM,)" in _CFG


def test_beads_use_verified_layout():
    assert "bead_offsets_in_cup" in _ENV


def test_cfg_has_no_warm_bank():
    assert "warm_bank" not in _CFG and "warm_bank" not in _ENV


# =============================================================================
# 차원 (Isaac 필요 — resolve_cfg 결과와 대조. 공식 재구현 금지)
# =============================================================================
def _cfg_module():
    pytest.importorskip("pxr")
    pytest.importorskip("isaaclab")
    from openarm.agnostic.tasks.pour_fabric import pour_fabric_env_cfg as C
    return C


def test_dims_from_resolve_cfg():
    C = _cfg_module()
    cfg = C.PourFabricEnvCfg()
    pair = __import__("openarm.agnostic.tasks.pour_fabric.bimanual",
                      fromlist=["get_pair"]).get_pair(cfg.pair_name)
    per = 0
    for p in (pair.source, pair.receiver):
        a, h, f = p.num_arm_joints, p.num_hand_joints, len(p.finger_sensor_bodies)
        per += 2 * a + 2 * h + 3 + 6 + 3 * f + 3 + 3 * f + h + 3
    assert cfg.action_space == 2 * (6 + 15)
    assert cfg.observation_space == per + 6 + cfg.action_space
    assert cfg.state_space == cfg.observation_space + 4 + 3 + 12 + 1 + 10


def test_short_reference_dimensions():
    C = _cfg_module()
    cfg = C.PourFabricEnvCfg()
    assert (cfg.action_space, cfg.observation_space, cfg.state_space) == (42, 286, 316)


def test_registered_cfg_classes_keep_own_pair_name():
    _cfg_module()
    from openarm.agnostic.tasks.pour_fabric import config as reg

    assert reg.REGISTERED, "등록된 쌍이 없다"
    for short in reg.REGISTERED:
        for suffix in ("", "_PLAY"):
            cls = getattr(reg, f"PourFabric_{short}{suffix}_Cfg")
            cfg = cls()
            assert cfg.pair_name == short
    play = getattr(reg, f"PourFabric_{sorted(reg.REGISTERED)[0]}_PLAY_Cfg")()
    assert play.scene.num_envs == 50

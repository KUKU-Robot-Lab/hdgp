"""Static and pure-Python contract tests for pre_pour_bc.

These tests intentionally avoid importing Isaac Lab so they can run in the
plain project Python environment.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _text(filename: str) -> str:
    return (_ROOT / filename).read_text(encoding="utf-8")


def test_init_registers_train_and_play_ids() -> None:
    t = _text("__init__.py")
    assert 'id="pre_pour_bc"' in t
    assert 'id="pre_pour_bc-play"' in t
    assert "pre_pour_bc_env:PrePourBCEnv" in t


def test_env_cfg_declares_91d_obs_and_18d_action() -> None:
    t = _text("pre_pour_bc_env_cfg.py")
    assert "observation_space = 91" in t
    assert "action_space = 18" in t
    assert "episode_length_s = 13.0" in t
    assert "self.sim.dt = 1.0 / 300.0" in t
    assert "self.decimation = 3" in t


def test_obs_cfg_keeps_dataset_compatible_91d_terms() -> None:
    t = _text("pre_pour_bc_obs_cfg.py")
    for term in (
        "right_joint_pos",
        "right_joint_vel",
        "left_joint_pos",
        "left_joint_vel",
        "tip_force_norm",
        "prev_actions",
    ):
        assert term in t
    assert "concatenate_terms = True" in t


def test_action_terms_keep_hdf5_slices() -> None:
    right = _text("fabrics_action_term.py")
    left = _text("left_arm_action_term.py")
    assert "return 11" in right
    assert "actions[:, :6]" in right
    assert "actions[:, 6:11]" in right
    assert "11:18" in left
    assert "_NUM_LEFT_ARM = 7" in left


def test_env_uses_pre_pour_joint_targets_not_pouring_reward() -> None:
    t = _text("pre_pour_bc_env.py")
    assert "target_joint_pos" in t
    assert "_get_rewards" in t
    assert "final_joint_error" in t
    assert "pour_done" not in t


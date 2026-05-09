"""Static checks for RL-Games playback overrides used by 5g_pour_right_v4."""

from __future__ import annotations

from pathlib import Path


_HDGP_ROOT = Path(__file__).resolve().parents[11]
_PLAY_TEXT = (_HDGP_ROOT / "scripts/reinforcement_learning/rl_games/play.py").read_text()
_CONFIG_TEXT = (
    _HDGP_ROOT
    / "source/openarm/openarm/tasks/manager_based/openarm_manipulation/pipeline/hand/right/5g_pour_right_v4/config/__init__.py"
).read_text()


def test_play_disable_adr_covers_v4_curriculum_flags() -> None:
    assert '"enable_noise_adr"' in _PLAY_TEXT
    assert '"enable_bead_count_adr"' in _PLAY_TEXT
    assert '"enable_success_adr"' in _PLAY_TEXT


def test_play_bead_fixed_supports_v4_single_count_curriculum() -> None:
    assert "env_cfg.bead_count = args_cli.bead_fixed" in _PLAY_TEXT
    assert "env_cfg.bead_count_stages = (args_cli.bead_fixed,)" in _PLAY_TEXT
    assert "env_cfg.enable_bead_count_adr = False" in _PLAY_TEXT


def test_play_can_freeze_grasp_hand_for_rendering() -> None:
    assert "--freeze_grasp_hand" in _PLAY_TEXT
    assert "env_cfg.freeze_grasp_hand_during_episode = True" in _PLAY_TEXT


def test_play_uses_same_bc_checkpoint_restore_patch_as_train() -> None:
    assert "from rl_games.common import a2c_common" in _PLAY_TEXT
    assert "def _patch_optimizer_restore()" in _PLAY_TEXT
    assert "_patch_optimizer_restore()" in _PLAY_TEXT
    assert "Skipping optimizer state restore" in _PLAY_TEXT


def test_play_installs_recurrent_gate_before_player_restore() -> None:
    assert "def _install_player_recurrent_gate(agent: BasePlayer, agent_cfg: dict) -> None:" in _PLAY_TEXT
    assert "recurrent_gate_enable" in _PLAY_TEXT
    assert "install_recurrent_gate(agent.model.a2c_network, obs_dim=actor_obs_dim)" in _PLAY_TEXT
    assert "_install_player_recurrent_gate(agent, agent_cfg)" in _PLAY_TEXT
    assert _PLAY_TEXT.index("_install_player_recurrent_gate(agent, agent_cfg)") < _PLAY_TEXT.index("agent.restore(resume_path)")


def test_play_normalizes_lowercase_play_task_name() -> None:
    assert "def _strip_play_task_name(task_name: str) -> str:" in _PLAY_TEXT
    assert '.replace("-play-", "-")' in _PLAY_TEXT
    assert "train_task_name = _strip_play_task_name(task_name)" in _PLAY_TEXT


def test_play_checkpoint_resolution_handles_relative_and_prefix_paths() -> None:
    assert "def _resolve_checkpoint_path(checkpoint: str) -> str:" in _PLAY_TEXT
    assert "Path(checkpoint).expanduser()" in _PLAY_TEXT
    assert "candidate.parent.glob(candidate.name + \"*.pth\")" in _PLAY_TEXT
    assert "Multiple checkpoint files match prefix" in _PLAY_TEXT
    assert "Parsed unique checkpoint prefix" in _PLAY_TEXT
    assert "resume_path = _resolve_checkpoint_path(args_cli.checkpoint)" in _PLAY_TEXT


def test_play_uses_plain_policy_step_loop_not_custom_replay() -> None:
    assert "read_from_sim" not in _PLAY_TEXT
    assert "write_to_sim" not in _PLAY_TEXT
    assert "any_bead_in_target" not in _PLAY_TEXT
    assert "if timestep == args_cli.video_length:" in _PLAY_TEXT
    assert "sleep_time = dt - (time.time() - start_time)" in _PLAY_TEXT


def test_play_env_uses_small_but_full_horizon_warmstart_cache_for_single_env_rendering() -> None:
    assert "class PourRightEnvCfg_PLAY(PourRightEnvCfg):" in _CONFIG_TEXT
    assert "self.warmstart_cache_size = 1" in _CONFIG_TEXT
    assert "self.warmstart_max_rollout_steps = 6000" in _CONFIG_TEXT

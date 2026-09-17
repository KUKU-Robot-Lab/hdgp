from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "iker_shoe_t2r_env.py").read_text(encoding="utf-8")
CFG = (ROOT / "iker_shoe_t2r_env_cfg.py").read_text(encoding="utf-8")
REG = (ROOT / "config" / "__init__.py").read_text(encoding="utf-8")


def test_env_subclasses_the_stage2_env_and_overrides_only_the_listed_hooks():
    assert "class IkerShoeT2rEnv(IkerShoeEnv)" in ENV
    # fix round 2: the boundary moved from 7 to 8 hooks — _log_episode_end was added to fix the parent's
    # full-replace self.extras["log"] silently dropping t2r_reward/*+place/* on almost every reset step.
    allowed = {"__init__", "_pre_physics_step", "_get_observations", "_get_dones",
               "_get_rewards", "_build_context", "_reset_idx", "_log_episode_end"}
    names = {line.split("def ")[1].split("(")[0] for line in ENV.splitlines() if line.strip().startswith("def ")}
    assert names <= allowed, names - allowed


def test_action_is_seven_and_observation_is_thirtynine():
    assert "action_space = 7" in CFG and "observation_space = 39" in CFG
    assert 'reward_code_path: str = ""' in CFG and "(IkerShoeEnvCfg)" in CFG


def test_grip_axis_uses_the_stage1_hand_law_not_a_new_one():
    assert "gs.hand_targets(" in ENV and "gs.normalized_targets(" in ENV and "actions[:, 6]" in ENV


def test_reward_comes_from_generated_code_and_place_flags_are_logged():
    block = ENV.split("def _get_rewards")[1]
    assert "call_reward_fn" in block and "nan_to_num" in block
    assert "t2r_reward/total" in ENV and "place/placed" in ENV and "place/released" in ENV


def test_context_tensors_are_copies():
    assert "v.clone() if isinstance(v, torch.Tensor) else v" in ENV.split("def _build_context")[1]


def test_reset_zeroes_previous_actions_and_restores_the_bank_grip():
    block = ENV.split("def _reset_idx")[1]
    assert "_t2r_prev_actions" in block and "_grip_targets" in block and "_place_window" in block


def test_registration_adds_the_t2r_ids():
    assert "openarm.agnostic.tasks.iker_shoe.iker_shoe_t2r_env:IkerShoeT2rEnv" in REG
    assert 'f"open-sens_l_iker_shoe_t2r{_suffix}"' in REG

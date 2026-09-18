"""The unified environment's contract, read from the source (no Isaac import): one policy, both halves of the task."""

from pathlib import Path

from openarm.agnostic.tasks.iker_shoe.t2r3.context import SCALAR_FIELDS, TENSOR_FIELDS
from openarm.agnostic.tasks.iker_shoe.t2r3 import prompts as P

ROOT = Path(__file__).resolve().parents[1]
ENV = (ROOT / "iker_shoe_unified_env.py").read_text(encoding="utf-8")
CFG = (ROOT / "iker_shoe_unified_env_cfg.py").read_text(encoding="utf-8")
REG = (ROOT / "config" / "__init__.py").read_text(encoding="utf-8")


def test_the_unified_env_extends_the_stage1_t2r_env_and_keeps_its_action_space():
    assert "class IkerShoeUnifiedEnv(IkerShoeGraspT2rEnv)" in ENV
    # the action space is stage-1's (6 palm + 20 finger joints) for both halves — user decision 2026-09-18
    assert "action_space" not in CFG and "observation_space" not in CFG


def test_the_episode_is_long_enough_for_both_halves():
    assert "UNIFIED_EPISODE_STEPS = 360" in CFG
    assert P.DEFAULT_EPISODE_STEPS == 360, "the prompt's episode length must equal the cfg's"
    assert "episode_length_s = UNIFIED_EPISODE_STEPS * CONTROL_DT" in CFG


def test_the_task_ends_on_the_placement_not_on_the_grasp():
    block = ENV.split("def _get_dones")[1]
    assert "step.success | dropped" in block, "termination must come from the placement predicate"
    assert "ps.place_step(" in block and "ps.retract_trigger(" in block


def test_reset_starts_from_the_rest_posture_with_a_drawn_shoe_position():
    block = ENV.split("def _reset_idx")[1]
    assert "default_joint_pos" in block and "_hand_reset" in block
    assert "spawn_noise_xy" in block


def test_start_mixture_draws_from_both_banks():
    block = ENV.split("def _reset_idx")[1]
    assert "self._held_bank" in block and "self._adjust_bank" in block
    assert "_start_held" in block
    for field in ("grasp_bank_start_path", "held_start_frac", "adjust_bank_path", "adjust_start_frac"):
        assert field in CFG


def test_success_per_start_kind_is_logged_on_every_step():
    # rl_games' observer indexes each step's log with the keys of the first one it saw (stage-2, 2026-09-18)
    assert '"place/success_table_start": 0.0' in ENV and '"place/success_held_start": 0.0' in ENV
    assert "log.update(self._start_log)" in ENV


def test_registration_adds_the_unified_ids():
    assert "openarm.agnostic.tasks.iker_shoe.iker_shoe_unified_env:IkerShoeUnifiedEnv" in REG
    assert 'f"open-sens_l_iker_shoe_unified{_suffix}"' in REG


def test_context_carries_both_halves():
    for name in ("held", "hold_count", "latched", "thumb_curl", "link_shoe_force", "slip_speed", "dz_free"):
        assert name in TENSOR_FIELDS, name
    for name in ("placed", "released", "resting", "still", "home", "stable_count", "success", "carried", "start_held"):
        assert name in TENSOR_FIELDS, name
    for name in ("lift_height", "thumb_curl_min", "place_tolerance", "home_joint_tol", "retract_open_min"):
        assert name in SCALAR_FIELDS, name


def test_prompt_renders_with_the_unified_action_table_and_both_predicates():
    text = P.render_prompt(P.PromptSpec(task=P.task_text()))
    assert "Box(-1, 1, (26,)" in text
    assert "actions[ 6] = hand joint  0" in text
    assert "ctx.held" in text and "ctx.placed" in text and "ctx.retracting" in text
    assert "{" not in text.split("```python")[0].replace("{{", ""), "an unfilled format placeholder is left in the prompt"

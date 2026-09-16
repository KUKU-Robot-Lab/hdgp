import torch

from openarm.agnostic.tasks.iker_shoe.place_stage import PlaceRewardCfg
from openarm.agnostic.tasks.iker_shoe.t2r2 import context as ctxmod
from openarm.agnostic.tasks.iker_shoe.t2r2 import loader, prompts, validator

GOOD = """
import torch

def compute_reward(ctx):
    near = 1.0 - torch.tanh(5.0 * ctx.keypoint_dist)
    let_go = ctx.released.float()
    return near + let_go, {"near": near, "let_go": let_go}
"""


def test_context_carries_the_stage2_fields_and_no_grasp_fields():
    required = {"grip_norm", "target_keypoints", "keypoints", "keypoint_err", "keypoint_dist",
                "shoe_bottom_z", "palm_shoe_dist", "placed", "released", "resting", "still",
                "stable_count", "success", "actions", "prev_actions"}
    assert required <= set(ctxmod.TENSOR_FIELDS)
    for gone in ("link_shoe_force", "thumb_curl", "dz_free", "hold_count", "latched", "hand_q"):
        assert gone not in ctxmod.TENSOR_FIELDS, gone


def test_good_code_passes_and_unknown_field_is_rejected():
    assert validator.static_check(GOOD).ok
    rep = validator.static_check(GOOD.replace("ctx.keypoint_dist", "ctx.distance_to_goal"))
    assert not rep.ok and any("distance_to_goal" in e for e in rep.errors)


def test_stage1_fields_are_unknown_here():
    rep = validator.static_check(GOOD.replace("ctx.released", "ctx.latched"))
    assert not rep.ok and any("latched" in e for e in rep.errors)


def test_in_place_mutation_and_bad_import_are_rejected():
    assert not validator.static_check(GOOD.replace("near = ", "ctx.keypoint_dist.clamp_(0.0)\n    near = ")).ok
    assert not validator.static_check("import os\n" + GOOD).ok


def test_fake_context_shapes_match_the_action_and_keypoint_counts():
    ctx = validator.make_fake_context(5)
    assert ctx.actions.shape == (5, 7) and ctx.prev_actions.shape == (5, 7)
    assert ctx.target_keypoints.shape == (5, 4, 3) and ctx.keypoint_err.shape == (5, 4)


def test_empty_path_loads_the_zero_reward():
    fn, origin = loader.load_reward_fn("")
    total, terms = loader.call_reward_fn(fn, validator.make_fake_context(3))
    assert total.shape == (3,) and terms == {} and "zero" in origin


def test_prompt_embeds_the_context_and_reads_numbers_from_cfg():
    cfg = PlaceRewardCfg()
    text = prompts.render_prompt(prompts.PromptSpec(task=prompts.task_text(cfg)))
    assert "class RewardContext" in text and "grip_norm" in text
    assert f"{cfg.place_tolerance}" in text and f"{cfg.release_radius}" in text and f"{cfg.stable_steps}" in text
    assert "actions[6]" in text and "actions[6:26]" not in text


def test_task_text_names_all_four_stages():
    for word in ("align", "set it down", "let go", "withdraw"):
        assert word in prompts.task_text(PlaceRewardCfg())

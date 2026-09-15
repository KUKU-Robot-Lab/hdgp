"""IKER stage-1 t2r fork — context, loader, validator, prompts and round files (spec 2026-09-15-iker-stage1-t2r §5, §6, §8; no Isaac)."""

import json
import re

import pytest
import torch

from openarm.agnostic.modules.robot_profiles import TESOLLO_LEFT_SHORT
from openarm.agnostic.tasks.iker_shoe import grasp_stage as gs
from openarm.agnostic.tasks.iker_shoe import layout
from openarm.agnostic.tasks.iker_shoe.t2r import context as C
from openarm.agnostic.tasks.iker_shoe.t2r import loader as L
from openarm.agnostic.tasks.iker_shoe.t2r import pipeline
from openarm.agnostic.tasks.iker_shoe.t2r import prompts as P
from openarm.agnostic.tasks.iker_shoe.t2r import validator as V

TASK_DIR = layout.HDGP_ROOT / "source" / "openarm" / "openarm" / "agnostic" / "tasks" / "iker_shoe"
SAMPLE = """import torch
import math


def compute_reward(ctx):
    approach = torch.exp(-5.0 * ctx.palm_gap)
    touch = torch.tanh(ctx.link_shoe_force.sum(dim=(1, 2)) / 5.0)
    lift = torch.clamp(ctx.dz_free / ctx.lift_height, 0.0, 1.0)
    return approach + touch + lift + 10.0 * ctx.success.float(), {"approach": approach, "touch": touch, "lift": lift}
"""


def _write(tmp_path, src):
    path = tmp_path / "compute_reward.py"
    path.write_text(src, encoding="utf-8")
    return str(path)


def test_context_stub_lists_every_field_without_helpers():
    stub = C.context_stub_source()
    assert "class RewardContext" in stub and "@property" not in stub
    for name in C.TENSOR_FIELDS + C.SCALAR_FIELDS:
        assert name in stub, name
    assert {"link_shoe_force", "palm_shoe_force", "slip_speed", "dz_free", "held", "hold_count", "latched", "success"} <= set(C.TENSOR_FIELDS)
    assert {"lift_height", "lift_max", "latch_steps", "success_steps", "rack_x_min", "rack_y_max", "episode_steps"} <= set(C.SCALAR_FIELDS)
    assert not any(name.startswith(("cup", "goal", "bead", "src_", "rcv_")) for name in C.TENSOR_FIELDS + C.SCALAR_FIELDS)


def test_fake_context_matches_the_documented_layout():
    ctx = V.make_fake_context(8)
    assert ctx.link_pos.shape == (8, 5, 3, 3) and ctx.link_shoe_gap.shape == (8, 5, 3) and ctx.link_shoe_force.shape == (8, 5, 3)
    assert ctx.hand_q.shape == (8, 20) and ctx.hand_target_norm.shape == (8, 20) and ctx.arm_q.shape == (8, 7)
    assert ctx.shoe_surface.shape == (8, gs.SURFACE_POINT_COUNT, 3) and ctx.shoe_start_xy.shape == (8, 2)
    assert ctx.actions.shape == (8, 26) and ctx.prev_actions.shape == (8, 26)
    assert ctx.held.dtype == ctx.latched.dtype == ctx.success.dtype == torch.bool and ctx.num_envs == 8


def test_validator_passes_a_wellformed_reward_and_rejects_bad_ones(tmp_path):
    rep = V.validate(_write(tmp_path, SAMPLE), devices=("cpu",))
    assert rep.ok, rep.errors
    assert set(rep.term_stats) == {"approach", "touch", "lift"} and "link_shoe_force" in rep.fields_used
    assert not V.validate(_write(tmp_path, SAMPLE.replace("ctx.palm_gap", "ctx.cup_pos")), devices=("cpu",)).ok
    assert not V.validate(_write(tmp_path, "import numpy\n" + SAMPLE), devices=("cpu",)).ok
    mutated = SAMPLE.replace("    approach =", "    ctx.dz_free.clamp_(min=0.0)\n    approach =")
    rep = V.validate(_write(tmp_path, mutated), devices=("cpu",))
    assert not rep.ok and any("dz_free" in error for error in rep.errors)


def test_loader_enforces_shapes_and_an_empty_path_is_zero_reward():
    ctx = V.make_fake_context(4)
    with pytest.raises(RuntimeError):
        L.call_reward_fn(lambda c: (torch.zeros(3), {}), ctx)
    fn, source = L.load_reward_fn("")
    total, terms = L.call_reward_fn(fn, ctx)
    assert torch.all(total == 0.0) and terms == {} and "zero" in source


def test_hand_action_table_has_one_row_per_profile_joint():
    names = TESOLLO_LEFT_SHORT.hand_joint_names
    assert len(P.HAND_ACTION_RANGES) == len(names) == 20
    assert all(lo <= hi for lo, hi in P.HAND_ACTION_RANGES)
    text = P.render_prompt(P.PromptSpec(task="T"))
    for index, name in enumerate(names):
        assert re.search(rf"actions\[\s*{6 + index}\] = hand joint\s+{index}: {name}\b", text), name


def test_prompt_numbers_come_from_the_config_and_the_hold_predicate():
    cfg = gs.Stage1RewardCfg()
    text = P.render_prompt(P.PromptSpec(task=P.task_text(cfg)))
    assert "Box(-1, 1, (26,), float32)" in text and "compute_reward(ctx: RewardContext)" in text
    assert f"{cfg.lift_height_m * 100:.0f} and {cfg.lift_max_m * 100:.0f} cm" in text
    assert "ctx.slip_speed < ctx.hold_slip_speed" in text and "ctx.latch_steps" in text
    env_cfg = (TASK_DIR / "iker_shoe_env_cfg.py").read_text(encoding="utf-8")
    grasp_cfg = (TASK_DIR / "iker_shoe_grasp_env_cfg.py").read_text(encoding="utf-8")
    assert f"action_pos_scale = {P.ACTION_POS_SCALE_M}" in env_cfg and f"action_rot_scale = {P.ACTION_ROT_SCALE_RAD}" in env_cfg
    assert f"observation_noise = {P.OBSERVATION_NOISE}" in env_cfg and f"action_noise = {P.ACTION_NOISE}" in env_cfg
    assert f"drop_z = {P.DROP_Z_M:.2f}" in grasp_cfg and f"wrench_force_per_kg = {P.WRENCH_FORCE_PER_KG}" in grasp_cfg
    assert f"start_noise_xy = {P.START_NOISE_XY_M}" in grasp_cfg and "GRASP_EPISODE_STEPS = 120" in grasp_cfg
    assert P.ARM_ACTION_DIM == 6 and P.HAND_ACTION_DIM == 20
    low = text.lower()
    assert "cup" not in low and "bead" not in low


def test_feedback_series_strips_prefixes_and_keeps_only_feedback_tags():
    events = {"Episode/t2r_reward/touch": [(1, 0.1), (2, 0.2)], "Episode/grasp_episode/success": [(1, 0.0)],
              "rewards/iter": [(1, 3.0)], "rewards/time": [(1, 3.0)], "Episode/grasp/w_thumb": [(1, 0.5)], "shaped_rewards/iter": [(1, 1.0)]}
    series = pipeline.feedback_series(events)
    assert series == {"t2r_reward/touch": [0.1, 0.2], "grasp_episode/success": [0.0], "rewards": [3.0]}
    table = P.render_feedback_table(series, n_points=10)
    assert "t2r_reward/touch: [0.1, 0.2]" in table and "min 0.1" in table


def test_round_files_render_ingest_and_reflect(tmp_path):
    first, second = tmp_path / "iter_00", tmp_path / "iter_01"
    prompt = pipeline.render(first)
    assert prompt.read_text(encoding="utf-8").startswith("You are an expert in robotics")
    with pytest.raises(FileExistsError):
        pipeline.render(first)
    (first / pipeline.RESPONSE).write_text("Stages...\n```python\nnot this\n```\nFinal:\n```python\n" + SAMPLE + "```\n", encoding="utf-8")
    report = pipeline.ingest(first, devices=("cpu",))
    assert report["ok"] and len(report["reward_sha256"]) == 64 and (first / pipeline.CODE).read_text(encoding="utf-8") == SAMPLE
    assert json.loads((first / pipeline.VALIDATION).read_text(encoding="utf-8"))["ok"]
    result = pipeline.reflect(first, second, {"Episode/t2r_reward/touch": [(1, 0.1)], "Episode/grasp_episode/success": [(1, 0.0)]})
    assert result["tags"] == 2 and (first / pipeline.FEEDBACK).is_file()
    text = (second / pipeline.PROMPT).read_text(encoding="utf-8")
    assert "The previous reward function was" in text and "t2r_reward/touch" in text and SAMPLE.strip().splitlines()[-1].strip() in text
    with pytest.raises(ValueError, match="feedback tag"):
        pipeline.reflect(first, tmp_path / "iter_02", {"rewards/time": [(1, 0.0)]})


def test_ingest_without_a_code_block_writes_a_failed_report(tmp_path):
    folder = tmp_path / "iter_00"
    folder.mkdir()
    (folder / pipeline.RESPONSE).write_text("no code here", encoding="utf-8")
    report = pipeline.ingest(folder, devices=("cpu",))
    assert not report["ok"] and "python" in report["errors"][0] and not (folder / pipeline.CODE).exists()

"""t2r(text2reward 계약) — Isaac 불요."""
import textwrap

import pytest
import torch

from openarm.agnostic.modules.t2r import context as C
from openarm.agnostic.modules.t2r import loader as L
from openarm.agnostic.modules.t2r import prompts as P
from openarm.agnostic.modules.t2r import validator as V

GOOD = textwrap.dedent("""
    import torch
    import math

    def compute_reward(ctx):
        d = (ctx.src_palm_pos - ctx.src_cup_pos).norm(dim=-1)
        approach = torch.exp(-5.0 * d)
        pour = 20.0 * ctx.d_in_target.clamp(min=0.0)
        lift = torch.where(ctx.src_grasped, ctx.src_cup_pos[:, 2] - ctx.src_cup_spawn_pos[:, 2],
                           torch.zeros_like(d))
        reward = approach + pour + lift + 10.0 * ctx.success.float()
        return reward, {"approach": approach, "pour": pour, "lift": lift}
""")


def _write(tmp_path, src, name="r.py"):
    p = tmp_path / name
    p.write_text(src)
    return str(p)


def test_context_fields_split_and_stub_has_every_field():
    stub = C.context_stub_source()
    for f in C.TENSOR_FIELDS + C.SCALAR_FIELDS:
        assert f in stub, f
    assert "@property" not in stub
    assert len(C.TENSOR_FIELDS) == 42 and len(C.SCALAR_FIELDS) == 5


def test_fake_context_shapes():
    ctx = V.make_fake_context(8, num_fingers=5, num_arm=7, num_actions=42)
    assert ctx.num_envs == 8
    assert tuple(ctx.src_tips_pos.shape) == (8, 5, 3)
    assert tuple(ctx.actions.shape) == (8, 42)
    assert ctx.success.dtype == torch.bool
    with pytest.raises(Exception):
        ctx.src_palm_pos = None    # frozen


def test_zero_reward_default():
    fn, src = L.load_reward_fn("")
    ctx = V.make_fake_context(4)
    total, terms = L.call_reward_fn(fn, ctx)
    assert tuple(total.shape) == (4,) and float(total.abs().sum()) == 0.0 and terms == {}
    assert "zero" in src


def test_good_code_validates(tmp_path):
    rep = V.validate(_write(tmp_path, GOOD))
    assert rep.ok, rep.errors
    assert set(rep.term_stats) == {"approach", "pour", "lift"}
    assert "src_grasped" in rep.fields_used


@pytest.mark.parametrize("bad, msg", [
    ("import os\ndef compute_reward(ctx):\n    return torch.zeros(ctx.num_envs), {}", "import"),
    ("import torch\ndef compute_reward(ctx):\n    return torch.zeros(ctx.num_envs) + ctx.nope, {}", "없는 필드"),
    ("import torch\ndef compute_reward(ctx):\n    ctx.src_palm_pos = 0\n    return torch.zeros(ctx.num_envs), {}", "읽기 전용"),
    ("import torch\ndef compute_reward(ctx):\n    return 1.0, {}", "(N,)"),
    ("import torch\ndef compute_reward(ctx):\n    return torch.zeros(ctx.num_envs), {'a': torch.zeros(3)}", "(N,)"),
    ("import torch\ndef other(ctx):\n    return torch.zeros(ctx.num_envs), {}", "정확히 하나"),
    ("import torch\ndef compute_reward(ctx):\n    return eval('1'), {}", "금지 호출"),
    ("import torch\ndef compute_reward(ctx):\n    return torch.zeros(ctx.num_envs) / 0 * 0, {}", "NaN"),
])
def test_bad_code_rejected(tmp_path, bad, msg):
    rep = V.validate(_write(tmp_path, "import torch\n" + bad))
    assert not rep.ok
    assert any(msg in e for e in rep.errors), rep.errors


def test_prompt_contains_stub_task_and_signature():
    txt = P.render_prompt(P.PromptSpec(task="Pour the beads."))
    assert "class RewardContext" in txt and "Pour the beads." in txt
    assert "def compute_reward(ctx: RewardContext)" in txt
    assert "Box(-1, 1, (42,)" in txt
    assert "previous reward function" not in txt


def test_prompt_feedback_round():
    fb = P.render_feedback_table({"reward/approach": [0.1, 0.2, 0.3], "task/success_now": [0.0] * 30})
    txt = P.render_prompt(P.PromptSpec(task="t", previous_code=GOOD, feedback=fb, user_notes="it tilts early"))
    assert "previous reward function" in txt and "reward/approach" in txt
    assert "min 0 · mean 0 · max 0" in txt
    assert "it tilts early" in txt

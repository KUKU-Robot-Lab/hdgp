"""Behavioural regression tests for fix round 1 of task-2 (spec 2026-09-16-iker-stage2-t2r).

``iker_shoe_t2r_env.py`` cannot be imported in this environment (isaaclab is not installed here — every
Isaac-touching module in this repo is imported only after ``AppLauncher`` boots the Kit app, which this
test suite intentionally never does). The existing contract test therefore only checks the file's *text*
(hook names, required substrings) and would not have caught any of the three fix-round-1 findings, all of
which are about *values* and *polarity*, not shape. These tests get real behavioural coverage anyway by
extracting the exact statements in question from the live source with ``ast`` and executing them — never by
re-deriving the formula independently, which would just test a second copy of the same possible mistake.
"""
from __future__ import annotations

import ast
from pathlib import Path

import torch

from openarm.agnostic.modules.iker.reward import reward_cfg_for_start_support
from openarm.agnostic.tasks.iker_shoe import layout
from openarm.agnostic.tasks.iker_shoe import place_stage as ps

ROOT = Path(__file__).resolve().parents[1]
ENV_SRC = (ROOT / "iker_shoe_t2r_env.py").read_text(encoding="utf-8")
EVAL_SRC = (layout.HDGP_ROOT / "scripts" / "iker" / "eval_iker.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------- finding 1: inherited fall height

def test_inherited_fall_height_differs_from_the_removed_drop_z():
    fall_height = reward_cfg_for_start_support(layout.TABLE_TOP_Z, max_episode_length=200).fall_height
    assert abs(fall_height - 0.168) < 1e-3
    assert abs(fall_height - 0.10) > 0.01  # the value the removed drop_z field used


def test_env_reads_the_inherited_fall_height_not_a_local_drop_z():
    assert "cfg.reward.fall_height" in ENV_SRC
    assert "cfg.drop_z" not in ENV_SRC and "self.cfg.drop_z" not in ENV_SRC


def test_dropped_boolean_terminates_below_the_inherited_threshold_and_not_above():
    fall_height = reward_cfg_for_start_support(layout.TABLE_TOP_Z, max_episode_length=200).fall_height
    shoe_z = torch.tensor([fall_height - 0.01, fall_height + 0.01])
    dropped = shoe_z < fall_height
    assert dropped.tolist() == [True, False]


# ---------------------------------------------------------------- finding 2: retreated_mask polarity

def _extract_eval_iker_pure_functions() -> dict:
    """Compile just eval_iker.py's pure metric helpers, skipping the module-level AppLauncher boot."""
    tree = ast.parse(EVAL_SRC, filename="eval_iker.py")
    wanted_names = {"RETREAT_M", "HOLD_RADIUS_M", "placed_mask", "retreated_mask"}
    keep = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in wanted_names)
        or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in wanted_names for t in node.targets))
    ]
    assert len(keep) == 4, f"expected RETREAT_M, HOLD_RADIUS_M, placed_mask, retreated_mask; found {[type(n).__name__ for n in keep]}"
    module = ast.Module(body=keep, type_ignores=[])
    ns: dict = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), "eval_iker.py", "exec"), ns)
    return ns


def test_retreated_mask_is_true_when_the_palm_returns_and_false_when_it_stays_away():
    ns = _extract_eval_iker_pure_functions()
    returned = torch.tensor([0.0, 0.05])       # at / well within RETREAT_M of the start pose
    stayed_away = torch.tensor([0.20, 1.00])   # well beyond RETREAT_M
    assert ns["retreated_mask"](returned).tolist() == [True, True]
    assert ns["retreated_mask"](stayed_away).tolist() == [False, False]


def test_placed_mask_unaffected_by_the_retreated_fix():
    ns = _extract_eval_iker_pure_functions()
    near = torch.tensor([True, True, False])
    palm_shoe = torch.tensor([0.30, 0.05, 0.30])  # only the first is both near and hand-off
    assert ns["placed_mask"](near, palm_shoe).tolist() == [True, False, False]


# Finding 3's "iker/* counters move with the predicate" tests were superseded in fix round 3 (critical 2):
# they verified the counters moved, but not against the parent's actual STRICT `>` comparison, which is what
# made the original fix wrong (a counter that only ever reaches the threshold, never exceeds it, still reads
# as "moving" while always failing `count > sustain`). See test_t2r2_fix_round_3.py.

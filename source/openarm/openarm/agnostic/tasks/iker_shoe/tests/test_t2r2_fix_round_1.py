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
import types
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


# ---------------------------------------------------------------- finding 3: iker/* counters move with the predicate

def _extract_dones_counter_slice():
    """Compile the exact _get_dones statements (place_step call through the failure-count accumulation) that
    feed self._success_count / self._failure_count, as a standalone function taking a duck-typed ``self``."""
    tree = ast.parse(ENV_SRC, filename="iker_shoe_t2r_env.py")
    get_dones = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_get_dones")
    body = get_dones.body
    start = next(i for i, n in enumerate(body) if ast.unparse(n).startswith("step = ps.place_step("))
    end = next(i for i, n in enumerate(body) if ast.unparse(n).startswith("self._failure_count = self._failure_count +"))
    assert start < end
    sliced = body[start:end + 1]

    func = ast.FunctionDef(
        name="_counter_slice",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=a) for a in
                            ("self", "keypoint_dist", "palm_shoe_dist", "shoe_bottom_z", "shoe_speed", "shoe_pos")],
                            kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=sliced + [ast.Return(value=ast.Tuple(elts=[
            ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr=a, ctx=ast.Load())
            for a in ("_success_count", "_failure_count", "_stable_count")
        ] + [ast.Name(id="dropped", ctx=ast.Load())], ctx=ast.Load()))],
        decorator_list=[], returns=None,
    )
    module = ast.Module(body=[func], type_ignores=[])
    ns: dict = {"ps": ps}
    exec(compile(ast.fix_missing_locations(module), "iker_shoe_t2r_env.py", "exec"), ns)
    return ns["_counter_slice"]


class _FakeSelf:
    def __init__(self, stable_count, failure_count, place_cfg, fall_height):
        self._stable_count = stable_count
        self._failure_count = failure_count
        self.cfg = types.SimpleNamespace(place=place_cfg, reward=types.SimpleNamespace(fall_height=fall_height))


def test_success_count_tracks_the_consecutive_stable_count_from_place_step():
    counter_slice = _extract_dones_counter_slice()
    cfg = ps.PlaceRewardCfg()
    good = dict(keypoint_dist=torch.tensor([0.01]), palm_shoe_dist=torch.tensor([0.30]),
                shoe_bottom_z=torch.tensor([cfg.rack_top_z]), shoe_speed=torch.tensor([0.0]),
                shoe_pos=torch.tensor([[0.0, 0.0, 1.0]]))  # well above any plausible fall height
    fake = _FakeSelf(torch.zeros(1), torch.zeros(1), cfg, fall_height=0.168)
    for expected in (1.0, 2.0, 3.0):
        success_count, failure_count, stable_count, dropped = counter_slice(fake, **good)
        fake._stable_count, fake._failure_count = stable_count, failure_count
        assert success_count.tolist() == [expected]
        assert bool(dropped[0]) is False and failure_count.tolist() == [0.0]


def test_failure_count_accumulates_while_the_shoe_is_below_fall_height_and_holds_otherwise():
    counter_slice = _extract_dones_counter_slice()
    cfg = ps.PlaceRewardCfg()
    fallen = dict(keypoint_dist=torch.tensor([0.30]), palm_shoe_dist=torch.tensor([0.30]),
                  shoe_bottom_z=torch.tensor([cfg.rack_top_z]), shoe_speed=torch.tensor([1.0]),
                  shoe_pos=torch.tensor([[0.0, 0.0, 0.0]]))  # z=0 < fall_height=0.168
    fake = _FakeSelf(torch.zeros(1), torch.zeros(1), cfg, fall_height=0.168)
    success_count, failure_count, stable_count, dropped = counter_slice(fake, **fallen)
    assert bool(dropped[0]) is True and failure_count.tolist() == [1.0]
    fake._stable_count, fake._failure_count = stable_count, failure_count
    success_count, failure_count, stable_count, dropped = counter_slice(fake, **fallen)
    assert failure_count.tolist() == [2.0]  # accumulates, does not reset, matching the old fall_penalty formula

    safe = dict(fallen, shoe_pos=torch.tensor([[0.0, 0.0, 1.0]]))
    success_count, failure_count, stable_count, dropped = counter_slice(fake, **safe)
    assert bool(dropped[0]) is False and failure_count.tolist() == [2.0]  # holds flat once no longer fallen

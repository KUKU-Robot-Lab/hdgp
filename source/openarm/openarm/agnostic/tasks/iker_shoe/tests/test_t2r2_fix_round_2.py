"""Behavioural regression test for fix round 2 of task-2 (spec 2026-09-16-iker-stage2-t2r).

``IkerShoeEnv._log_episode_end`` (inherited, unmodifiable, and inaccessible here — isaaclab is not installed
in this environment, see test_t2r2_fix_round_1.py's module docstring) does ``self.extras["log"] = {...}``, a
full replace, and runs after ``_get_rewards`` within the same ``step()``. Round 1 added ``_log_episode_end``
as an eighth override that calls the parent first and then merges ``t2r_reward/*``/``place/*`` back in. This
test builds a small fake parent with the same destructive full-replace shape, extracts the *real*
``_log_episode_end`` body from the live source with ``ast`` (never a reimplementation of it) as a method of a
class that actually subclasses the fake parent, and asserts the merged log still carries both key families
after a simulated "step where environments reset" — the exact scenario that was silently losing data before
round 2.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_SRC = (ROOT / "iker_shoe_t2r_env.py").read_text(encoding="utf-8")


class _FakeParent:
    """Stands in for IkerShoeEnv._log_episode_end: a full replace of self.extras["log"], exactly like the
    real one, simulating a step on which at least one environment's episode just ended."""

    def _log_episode_end(self, env_ids) -> None:
        self.extras["log"] = {"iker/keypoint_distance_m": 0.05, "iker/success_5cm": 1.0}


def _build_fake_t2r_env_class():
    tree = ast.parse(ENV_SRC, filename="iker_shoe_t2r_env.py")
    log_end = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_log_episode_end")
    class_def = ast.ClassDef(name="_FakeT2r", bases=[ast.Name(id="_FakeParent", ctx=ast.Load())], keywords=[],
                             body=[log_end], decorator_list=[])
    module = ast.Module(body=[class_def], type_ignores=[])
    ns: dict = {"_FakeParent": _FakeParent}
    exec(compile(ast.fix_missing_locations(module), "iker_shoe_t2r_env.py", "exec"), ns)
    return ns["_FakeT2r"]


def test_log_episode_end_merges_t2r_and_place_keys_back_after_the_parents_full_replace():
    fake_t2r_cls = _build_fake_t2r_env_class()
    obj = fake_t2r_cls()
    obj.extras = {}
    obj._t2r_log = {"t2r_reward/total": 1.23, "t2r_reward/near": 0.9, "place/placed": 0.5, "place/retreated": 0.0}

    obj._log_episode_end(env_ids=None)  # simulates the reset-step call from _reset_idx

    log = obj.extras["log"]
    assert any(k.startswith("t2r_reward/") for k in log), f"t2r_reward/* lost: {sorted(log)}"
    assert any(k.startswith("place/") for k in log), f"place/* lost: {sorted(log)}"
    assert any(k.startswith("iker/") for k in log), "the parent's own iker/* entries should still be present"
    assert log["t2r_reward/total"] == 1.23 and log["place/placed"] == 0.5


def test_log_episode_end_does_not_crash_before_the_first_get_rewards():
    """_t2r_log is None until _get_rewards has run at least once (e.g. the very first env.reset())."""
    fake_t2r_cls = _build_fake_t2r_env_class()
    obj = fake_t2r_cls()
    obj.extras = {}
    obj._t2r_log = None

    obj._log_episode_end(env_ids=None)

    assert obj.extras["log"] == {"iker/keypoint_distance_m": 0.05, "iker/success_5cm": 1.0}

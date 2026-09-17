"""Behavioural regression tests for fix round 3 (final whole-plan review) of task-2 (spec
2026-09-16-iker-stage2-t2r).

Same constraint and technique as the earlier fix-round test files: ``isaaclab`` is not installed in this
environment, so ``iker_shoe_t2r_env.py``/``iker_shoe_env.py`` cannot be imported, and these tests ``ast``-slice
the exact statements from the live source and execute them, rather than re-deriving the logic independently.
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
PARENT_SRC = (ROOT / "iker_shoe_env.py").read_text(encoding="utf-8")


# ================================================================== critical 1: log merge staleness


class _FakeCtx:
    def __init__(self, value: float = 1.0):
        self.placed = self.released = self.resting = self.still = torch.tensor([value > 0])


def _extract_get_rewards_log_slice():
    """The log: dict[str, float] = {} ... self._t2r_log = log statements from _get_rewards, as a standalone
    function taking a duck-typed self plus the reward-computation outputs it needs."""
    tree = ast.parse(ENV_SRC, filename="iker_shoe_t2r_env.py")
    retreat_m = next(
        n.value.value for n in tree.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RETREAT_M" for t in n.targets)
    )
    get_rewards = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_get_rewards")
    body = get_rewards.body
    start = next(i for i, n in enumerate(body) if ast.unparse(n).startswith("log: dict[str, float] = {}"))
    end = next(i for i, n in enumerate(body) if ast.unparse(n).startswith("self._t2r_log = log"))
    assert start < end
    sliced = body[start:end + 1]

    func = ast.FunctionDef(
        name="_log_slice",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=a) for a in ("self", "total", "terms", "nonfinite_frac", "ctx")],
                            kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=sliced, decorator_list=[], returns=None,
    )
    module = ast.Module(body=[func], type_ignores=[])
    ns: dict = {"RETREAT_M": retreat_m}
    exec(compile(ast.fix_missing_locations(module), "iker_shoe_t2r_env.py", "exec"), ns)
    return ns["_log_slice"]


class _FakeParentSeries:
    """Stands in for IkerShoeEnv._log_episode_end: a full replace of self.extras["log"], returning a
    DIFFERENT fresh iker/keypoint_distance_m each call — a real keypoint distance changing step to step.
    ``self._values`` is set on the instance after construction (not via __init__: the subclass built below
    inherits only this method, not a constructor, so the fixture is wired up after plain instantiation)."""

    def _log_episode_end(self, env_ids) -> None:
        self.extras["log"] = {"iker/keypoint_distance_m": next(self._values), "iker/success_5cm": 0.0}


def _build_fake_t2r_env_class(parent_cls):
    tree = ast.parse(ENV_SRC, filename="iker_shoe_t2r_env.py")
    log_end = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_log_episode_end")
    class_def = ast.ClassDef(name="_FakeT2r", bases=[ast.Name(id="_Parent", ctx=ast.Load())], keywords=[],
                             body=[log_end], decorator_list=[])
    module = ast.Module(body=[class_def], type_ignores=[])
    ns: dict = {"_Parent": parent_cls}
    exec(compile(ast.fix_missing_locations(module), "iker_shoe_t2r_env.py", "exec"), ns)
    return ns["_FakeT2r"]


def test_the_parents_fresh_iker_value_survives_the_merge_across_two_consecutive_steps():
    """Regression for critical 1: seeding _t2r_log from self.extras carried the previous step's iker/* forward
    into the merge, permanently pinning it at whatever value the parent first wrote. A fixture that DOES carry
    an iker/* key (unlike round 2's test, which used a fixture with none and could not have caught this) must
    show the parent's genuinely fresh value on every step, not the first one."""
    log_slice = _extract_get_rewards_log_slice()
    fake_cls = _build_fake_t2r_env_class(_FakeParentSeries)
    obj = fake_cls()
    obj.extras = {}
    obj._values = iter([0.28, 0.14])
    obj._t2r_last = {"palm_pos": torch.zeros(2, 3)}
    obj._palm_start = torch.zeros(2, 3)

    # step 1
    log_slice(obj, total=torch.tensor([1.0, 1.0]), terms={}, nonfinite_frac=torch.tensor(0.0), ctx=_FakeCtx())
    assert "iker/keypoint_distance_m" not in obj._t2r_log, "the child's own log must never carry an iker/* key"
    obj._log_episode_end(env_ids=None)
    assert obj.extras["log"]["iker/keypoint_distance_m"] == 0.28
    assert any(k.startswith("t2r_reward/") for k in obj.extras["log"])

    # step 2: a different, genuinely fresher keypoint distance
    log_slice(obj, total=torch.tensor([1.0, 1.0]), terms={}, nonfinite_frac=torch.tensor(0.0), ctx=_FakeCtx())
    assert "iker/keypoint_distance_m" not in obj._t2r_log
    obj._log_episode_end(env_ids=None)
    assert obj.extras["log"]["iker/keypoint_distance_m"] == 0.14, (
        "the merge pinned the stale step-1 value instead of the parent's fresh step-2 value"
    )
    assert any(k.startswith("t2r_reward/") for k in obj.extras["log"])


# ================================================================== critical 2: strict > sustain_steps


def _extract_dones_counter_slice():
    """The step = ps.place_step(...) ... self._failure_count = torch.where(...) statements from _get_dones."""
    tree = ast.parse(ENV_SRC, filename="iker_shoe_t2r_env.py")
    get_dones = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_get_dones")
    body = get_dones.body
    start = next(i for i, n in enumerate(body) if ast.unparse(n).startswith("step = ps.place_step("))
    end = next(i for i, n in enumerate(body) if ast.unparse(n).startswith("self._failure_count = torch.where("))
    assert start < end
    sliced = body[start:end + 1]

    func = ast.FunctionDef(
        name="_counter_slice",
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=a) for a in
                            ("self", "keypoint_dist", "palm_shoe_dist", "shoe_bottom_z", "shoe_speed", "shoe_pos",
                             "shoe_ang_speed")],
                            kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=sliced + [ast.Return(value=ast.Tuple(elts=[
            ast.Attribute(value=ast.Name(id="self", ctx=ast.Load()), attr=a, ctx=ast.Load())
            for a in ("_success_count", "_failure_count")
        ], ctx=ast.Load()))],
        decorator_list=[], returns=None,
    )
    module = ast.Module(body=[func], type_ignores=[])
    ns: dict = {"ps": ps, "torch": torch}
    exec(compile(ast.fix_missing_locations(module), "iker_shoe_t2r_env.py", "exec"), ns)
    return ns["_counter_slice"]


def _extract_parent_iker_bool_exprs() -> dict[str, str]:
    """The two boolean comparisons inside IkerShoeEnv._log_episode_end's iker/sustained_success and
    iker/dropped dict values, unparsed back to source text — not restated, read from the parent's own code."""
    tree = ast.parse(PARENT_SRC, filename="iker_shoe_env.py")
    log_end = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_log_episode_end")
    dict_node = next(n for n in ast.walk(log_end) if isinstance(n, ast.Dict))
    exprs = {}
    for key_node, value_node in zip(dict_node.keys, dict_node.values):
        if isinstance(key_node, ast.Constant) and key_node.value in ("iker/sustained_success", "iker/dropped"):
            compare = next(n for n in ast.walk(value_node) if isinstance(n, ast.Compare))
            exprs[key_node.value] = ast.unparse(compare)
    assert set(exprs) == {"iker/sustained_success", "iker/dropped"}, exprs
    return exprs


class _FakeSelfCfg:
    def __init__(self, place_cfg, fall_height, sustain_steps):
        self.place = place_cfg
        self.reward = type("_R", (), {"fall_height": fall_height, "sustain_steps": sustain_steps})()


def test_iker_counters_satisfy_the_parents_actual_strict_comparison_exactly_when_the_predicate_fires():
    """Regression for critical 2: the parent compares STRICTLY (count > sustain_steps). A counter that merely
    reaches sustain_steps (place_step's own stable_count ceilings at exactly stable_steps the moment success
    fires, and the episode ends that same step) fails a strict `>` forever. This exercises the parent's own
    comparison text, not a restated `> 20`, so an operator change in either file would fail it."""
    counter_slice = _extract_dones_counter_slice()
    bool_exprs = _extract_parent_iker_bool_exprs()
    cfg = ps.PlaceRewardCfg()
    sustain = reward_cfg_for_start_support(layout.TABLE_TOP_Z, max_episode_length=200).sustain_steps

    def run(shoe_pos_z: float, keypoint_dist: float, shoe_speed: float, stable_count_in: float):
        fake = type("_S", (), {})()
        # 창 방식(2026-09-17): "들어갈 때 이미 충족된 스텝 수" 를 창의 최근 쪽에 채운다. 가장 오래된
        # 칸(0번)이 비어 있어야 roll 이 0 을 버리고 이번 스텝 결과를 더해 count 가 +1 된다.
        window = ps.new_window(1, cfg, "cpu")
        if stable_count_in > 0:
            window[0, -int(stable_count_in):] = 1.0
        fake._place_window = window
        fake._success_count = torch.zeros(1)
        fake._failure_count = torch.zeros(1)
        fake.cfg = _FakeSelfCfg(cfg, fall_height=0.168, sustain_steps=sustain)
        success_count, failure_count = counter_slice(
            fake, keypoint_dist=torch.tensor([keypoint_dist]), palm_shoe_dist=torch.tensor([0.30]),
            shoe_bottom_z=torch.tensor([cfg.rack_top_z]), shoe_speed=torch.tensor([shoe_speed]),
            shoe_pos=torch.tensor([[0.0, 0.0, shoe_pos_z]]), shoe_ang_speed=torch.tensor([0.0]),
        )
        return success_count, failure_count

    def parent_bool(tag: str, count: torch.Tensor, sustain_value: int) -> bool:
        ns = {"self": type("_S", (), {"_success_count": count, "_failure_count": count})(),
              "finished": torch.tensor([0]), "sustain": sustain_value}
        return bool(eval(compile(bool_exprs[tag], "<parent>", "eval"), {}, ns).all())

    # the exact scenario that broke round 1's fix: success/dropped fires and the episode ends on the very step
    # stable_count first reaches cfg.stable_steps (here simulated as already at that ceiling going in).
    success_count, _ = run(shoe_pos_z=1.0, keypoint_dist=0.01, shoe_speed=0.0, stable_count_in=float(cfg.stable_steps - 1))
    assert parent_bool("iker/sustained_success", success_count, sustain) is True, (
        "the parent's strict count > sustain must read True on the exact step place_step declares success"
    )

    _, failure_count = run(shoe_pos_z=0.0, keypoint_dist=0.30, shoe_speed=1.0, stable_count_in=0.0)  # z=0 < fall_height
    assert parent_bool("iker/dropped", failure_count, sustain) is True, (
        "the parent's strict count > sustain must read True on the exact step the shoe first falls"
    )

    # and the negative: neither condition true this step -> both counters must read False through the parent's
    # own comparison, not just "small".
    success_count, failure_count = run(shoe_pos_z=1.0, keypoint_dist=0.30, shoe_speed=1.0, stable_count_in=0.0)
    assert parent_bool("iker/sustained_success", success_count, sustain) is False
    assert parent_bool("iker/dropped", failure_count, sustain) is False

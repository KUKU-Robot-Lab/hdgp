"""IKER stage-1 t2r environment contract — source text and AST (spec 2026-09-15-iker-stage1-t2r §7, §14; no Isaac)."""

from openarm.agnostic.modules.source_contract import _class_methods, _fn_block, _ordered
from openarm.agnostic.tasks.iker_shoe import layout

TASK_DIR = layout.HDGP_ROOT / "source" / "openarm" / "openarm" / "agnostic" / "tasks" / "iker_shoe"
ENV = (TASK_DIR / "iker_shoe_grasp_t2r_env.py").read_text(encoding="utf-8")
CFG = (TASK_DIR / "iker_shoe_grasp_t2r_env_cfg.py").read_text(encoding="utf-8")
REG = (TASK_DIR / "config" / "__init__.py").read_text(encoding="utf-8")


def test_env_subclasses_the_grasp_env_and_overrides_only_reward_sensor_and_reset_hooks():
    assert "class IkerShoeGraspT2rEnv(IkerShoeGraspEnv)" in ENV
    names = _class_methods(ENV, "IkerShoeGraspT2rEnv")
    allowed = {"__init__", "_setup_scene", "_shoe_contact_filter", "_link_shoe_forces", "_get_rewards", "_build_context", "_reset_idx"}
    assert names <= allowed, names - allowed
    for forbidden in ("_get_observations", "_get_dones", "_pre_physics_step", "_apply_action", "restore_success_captures"):
        assert forbidden not in names, forbidden


def test_generated_code_is_loaded_before_the_env_boots():
    _ordered(_fn_block(ENV, "__init__"), ["load_reward_fn(cfg.reward_code_path)", "super().__init__("])


def test_reward_comes_from_the_generated_code_on_the_step_state_the_predicate_used():
    block = _fn_block(ENV, "_get_rewards")
    _ordered(block, ["self._last is None", "self._build_context()", "call_reward_fn(self._reward_fn", "nan_to_num", "return total"])
    assert "grasp_reward/" in block and "t2r_reward/total" in block and "contact/fingers_touching" in block
    assert "self._t2r_prev_actions = self.actions.clone()" in block and "ctx.actions.clone()" not in block
    context = _fn_block(ENV, "_build_context")
    for token in ("self._last", "step.held", "step.state.hold_count", "step.state.latched", "step.success", "gs.palm_frame_slip_speed(",
                  "gs.free_lift_height(", "self._link_shoe_forces()", "self._t2r_prev_actions"):
        assert token in context, token
    assert "v.clone() if isinstance(v, torch.Tensor) else v" in context


def test_one_contact_sensor_per_body_filtered_to_the_shoe_prim_found_on_the_stage():
    scene = _fn_block(ENV, "_setup_scene")
    _ordered(scene, ["super()._setup_scene()", "self._shoe_contact_filter()", "ContactSensor(ContactSensorCfg("])
    assert "for body in" in scene and scene.count("filter_prim_paths_expr=shoe_filter") == 2 and "self.scene.sensors[" in scene
    shoe_filter = _fn_block(ENV, "_shoe_contact_filter")
    for token in ("get_all_matching_child_prims", "RigidBodyAPI", "self.cfg.shoe_move_cfg.prim_path", "len(prims) != 1", "raise RuntimeError"):
        assert token in shoe_filter, token


def test_reset_zeroes_the_previous_actions_after_the_parent_reset():
    _ordered(_fn_block(ENV, "_reset_idx"), ["super()._reset_idx(env_ids)", "= 0.0"])


def test_cfg_defaults_to_zero_reward_and_registration_uses_the_grasp_agent():
    assert 'reward_code_path: str = ""' in CFG and "(IkerShoeGraspEnvCfg)" in CFG
    assert "openarm.agnostic.tasks.iker_shoe.iker_shoe_grasp_t2r_env:IkerShoeGraspT2rEnv" in REG
    # 3 uses since 2026-09-18: the stage-1 grasp ids, the stage-1 t2r ids and the unified ids (same 26/78 spaces)
    assert 'f"open-sens_l_iker_shoe_grasp_t2r{_suffix}"' in REG and REG.count("rl_games_grasp_ppo_cfg.yaml") == 3

"""grasp_fj_t2r 계약 테스트 — 소스 텍스트·AST (Isaac 불요).

이 트랙은 Track B(`grasp_fj`) 위에 **보상만** text2reward 생성 코드로 바꿔 끼운다.
잠그는 것:
  ① 관측·액션·종료·성공 판정은 B 그대로다 — 덮는 훅은 보상 이음매·센서·로그뿐.
  ② 에피소드 상태(lifted 래치·추적기)는 B 모듈이 계속 만들고, 총보상만 생성 코드가 낸다.
  ③ 보상 전용 접촉 센서는 body 하나당 센서 하나, 컵에만 필터 — 관측에는 들어가지 않는다.
  ④ 생성 코드 경로 기본값은 비어 있다(영 보상) — 켜는 것은 런처의 `env.reward_code_path=`.

실행:
    cd hdgp && PYTHONPATH=source/openarm python3 -m pytest \
        source/openarm/openarm/agnostic/tasks/grasp_fj_t2r/tests -q
"""

from __future__ import annotations

from pathlib import Path

from openarm.agnostic.modules.source_contract import _class_methods, _fn_block, _ordered

_HERE = Path(__file__).resolve().parent.parent
_ENV = (_HERE / "grasp_fj_t2r_env.py").read_text(encoding="utf-8")
_CFG = (_HERE / "grasp_fj_t2r_env_cfg.py").read_text(encoding="utf-8")
_REG = (_HERE / "config" / "__init__.py").read_text(encoding="utf-8")


def test_env_is_track_b_with_only_reward_sensor_and_log_hooks():
    assert "class GraspFJT2REnv(GraspFJEnv)" in _ENV
    names = _class_methods(_ENV, "GraspFJT2REnv")
    allowed = {"__init__", "_setup_scene", "_cup_contact_filter", "_progress_reward", "_build_context",
               "_link_cup_forces", "_log_fabric_metrics", "_reset_idx"}
    assert names <= allowed, names - allowed
    for forbidden in ("_get_observations", "_get_dones", "_get_rewards", "_hand_command",
                      "_pre_physics_step", "_apply_action"):
        assert forbidden not in names, f"{forbidden} 를 덮으면 B 와 같은 과제가 아니다"


def test_reward_total_comes_from_generated_code_but_state_from_track_b():
    pr = _fn_block(_ENV, "_progress_reward")
    _ordered(pr, ["super()._progress_reward(", "self._build_context(", "call_reward_fn(self._reward_fn"])
    assert "nan_to_num" in pr, "폭발 env 의 NaN 이 PPO 전체를 오염시키지 않게"
    assert "return total, terms, out" in pr, "래치·추적기 되먹임(out)은 B 모듈 것을 그대로 돌려준다"


def test_generated_code_is_loaded_before_the_env_boots():
    _ordered(_fn_block(_ENV, "__init__"), ["load_reward_fn(cfg.reward_code_path)", "super().__init__("])


def test_one_contact_sensor_per_body_filtered_to_the_cup():
    blk = _fn_block(_ENV, "_setup_scene")
    _ordered(blk, ["super()._setup_scene()", "self._cup_contact_filter()", "ContactSensor(ContactSensorCfg("])
    assert blk.count("filter_prim_paths_expr=_filter") == 2, "마디·손바닥 센서 모두 실측 경로 필터"
    assert "for body in" in blk, "다중 body 를 한 센서에 묶으면 force_matrix_w 가 조용히 0 이다"
    assert "self.scene.sensors[" in blk


def test_cup_filter_comes_from_the_stage_not_the_cfg_string():
    # ★09.14 스모크: cfg `object_contact_filter` 가 shaker_sweep 에서 없는 경로(`Object/ShakerFDM5mm`)로
    #   풀려 PhysX 가 0개 매칭 에러만 찍고 힘이 전부 0 이었다. 스테이지의 RigidBodyAPI 프림으로 만든다.
    assert "object_contact_filter" not in _fn_block(_ENV, "_setup_scene")
    cf = _fn_block(_ENV, "_cup_contact_filter")
    for tok in ("get_all_matching_child_prims", "RigidBodyAPI", "self.cfg.object_cfg.prim_path",
                "len(prims) != 1", "raise RuntimeError"):
        assert tok in cf, tok


def test_context_is_built_from_the_same_step_state():
    bc = _fn_block(_ENV, "_build_context")
    for tok in ('kw["kp_dist"]', 'kw["is_success"]', 'out["lifted"]', "self._tol.tol",
                "self._obj_grasp_r", "self._obj_grasp_h", "self._palm_ee_R()",
                "self._link_cup_forces()", "self._trk.successes"):
        assert tok in bc, tok


def test_context_tensors_are_copies_not_live_env_buffers():
    # ★09.14 리뷰(HIGH): goal_pos·lifted·is_success·_obj_grasp_r 등은 env 가 보상 **뒤에** 다시 읽는 버퍼다.
    #   생성 코드가 제자리 연산을 하면 학습 내내 조용히 오염된다 — ctx 에는 복사본만 넣는다.
    assert "v.clone() if isinstance(v, torch.Tensor) else v" in _fn_block(_ENV, "_build_context")


def test_reward_code_path_defaults_empty_and_leaf_is_the_short_tl_hand():
    assert 'reward_code_path: str = ""' in _CFG
    assert "(GraspFJTesolloRightShortEnvCfg)" in _CFG


def test_registration_reuses_track_b_agents_with_the_t2r_entry():
    assert "openarm.agnostic.tasks.grasp_fj_t2r.grasp_fj_t2r_env:GraspFJT2REnv" in _REG
    assert "from ...grasp_fj.config import agents" in _REG
    assert "_grasp_fj_t2r" in _REG and "-play-lstm-sapg" in _REG

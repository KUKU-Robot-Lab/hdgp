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
    # ★09.14 reach 스모크: runaway env 에서 생성 속도 제곱 항이 −1e9 급 — 이번 스텝 abnormal env 는 보상·로그 항 모두 가린다
    _ordered(pr, ["call_reward_fn(self._reward_fn", "ok = ~self._abnormal", "torch.where(ok,", "terms = {k: torch.where(ok,"])


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


def test_contact_at_success_is_logged_for_the_loop_judge():
    # ★09.14 t2r 루프: "성공이 인벨롭이었나"는 스텝 평균으로 못 가른다(접근 중 env 가 뭉갠다) — 성공 순간 이벤트 EMA.
    blk = _fn_block(_ENV, "_log_fabric_metrics")
    for tok in ("self._success_now", "self._event_ema(", "_at_success"):
        assert tok in blk, tok


def test_stage_funnel_is_latched_per_episode_for_the_loop_ticks():
    # ★09.14 사용자 "컵에 접근, 파지, 리프트가 잘 되는지 틱을 확인" — 에피소드 래치 → 끝날 때 이벤트 EMA. 로그 전용.
    lg = _fn_block(_ENV, "_log_fabric_metrics")
    for tok in ("palm_band_gap(", "step_flags(", "self._t2r_stage_latch |=", '"stage/palm_cup_gap"',
                'f"stage/{name}_ep"'):
        assert tok in lg, tok
    rs = _fn_block(_ENV, "_reset_idx")
    _ordered(rs, ["self._event_ema(self._t2r_stage_ema", "latch[ids] = False", "super()._reset_idx(env_ids)"])
    assert "self.episode_length_buf[ids] > 0" in rs, "첫 reset()(길이 0)은 에피소드 끝이 아니다"
    assert "stage" not in _fn_block(_ENV, "_build_context"), "퍼널은 보상 ctx 에 들어가지 않는다"


def test_reward_code_path_defaults_empty_and_leaf_is_the_short_tl_hand():
    assert 'reward_code_path: str = ""' in _CFG
    assert "(GraspFJTesolloRightShortEnvCfg)" in _CFG


def test_reach_leaf_carries_the_user_decisions_of_0914():
    # ★사용자 결정(09.14): 테이블 앞 가장자리 위 시작 · 팔 속도 2배 + 15 s · cup_family. 셋 중 하나라도 빠지면 다른 과제다.
    assert "class GraspFJT2RReachEnvCfg(GraspFJT2RRightShortEnvCfg)" in _CFG
    blk = _CFG.split("class GraspFJT2RReachEnvCfg", 1)[1]
    for tok in ("arm_reset_joint_pos_override: tuple = (-1.1974, 0.6707, 0.1866, 1.7310, 0.6920, 0.0416, 0.9460)",
                "k_arm: float = 0.05", "arm_dof_speed_scale: float = 3.0", "arm_slew_rad_s: float = 0.3",
                "episode_length_s: float = 15.0", 'object_bank: str = "cup_family"', "start_palm_dist_band_m"):
        assert tok in blk, tok
    assert '("short_r", "grasp_fj_t2r_reach"): GraspFJT2RReachEnvCfg' in _REG


def test_start_pose_override_is_read_by_both_validator_and_env():
    # ★같은 자산 프로필 변종은 gym 슬롯이 겹친다 → cfg 덮어쓰기. 검증기(홈 1.5 rad)와 env 가 같은 값을 읽어야 한다.
    base = (_HERE.parent / "grasp_fj")
    cfg_src = (base / "grasp_fj_env_cfg.py").read_text(encoding="utf-8")
    env_src = (base / "grasp_fj_env.py").read_text(encoding="utf-8")
    assert "arm_reset_joint_pos_override: tuple = ()" in cfg_src
    assert "tuple(self.arm_reset_joint_pos_override) or tuple(profile.arm_reset_joint_pos)" in cfg_src
    assert "tuple(self.cfg.arm_reset_joint_pos_override) or tuple(self.profile.arm_reset_joint_pos)" in env_src


def test_registration_reuses_track_b_agents_with_the_t2r_entry():
    assert "openarm.agnostic.tasks.grasp_fj_t2r.grasp_fj_t2r_env:GraspFJT2REnv" in _REG
    assert "from ...grasp_fj.config import agents" in _REG
    assert "_grasp_fj_t2r" in _REG and "-play-lstm-sapg" in _REG

"""grasp_fj_t2r 의 text2reward 포크 — 컨텍스트·로더·검증기·프롬프트 (Isaac 불요, 순수 torch).

왜 포크인가: `modules/t2r` 는 양팔 붓기 전용(src/rcv 팔·컵 두 개·비드)이고 다른 세션의 루프가
쓰고 있다. fj 트랙은 공유 코드를 쓰지 않는다(09.10 사용자 확정). 여기서 잠그는 것:
  ① 컨텍스트 소스가 곧 프롬프트의 환경 설명이다 — 붓기 필드가 섞이지 않는다.
  ② 검증기가 없는 필드(환각·붓기 필드)·금지 import·ctx 변경을 시끄럽게 막는다.
  ③ 프롬프트가 단일 팔 + 19관절 full-joint 손 + 26차원 액션을 설명한다.

실행:
    cd hdgp && PYTHONPATH=source/openarm python3 -m pytest \
        source/openarm/openarm/agnostic/tasks/grasp_fj_t2r/tests -q
"""

from __future__ import annotations

import pytest
import torch

from openarm.agnostic.tasks.grasp_fj_t2r.t2r import context as C
from openarm.agnostic.tasks.grasp_fj_t2r.t2r import loader as L
from openarm.agnostic.tasks.grasp_fj_t2r.t2r import prompts as P
from openarm.agnostic.tasks.grasp_fj_t2r.t2r import validator as V

SAMPLE = """import torch
import math


def compute_reward(ctx):
    d = torch.norm(ctx.palm_pos - ctx.cup_pos, dim=-1)
    approach = torch.exp(-5.0 * d)
    touch = torch.tanh(ctx.link_cup_force.sum(dim=(1, 2)) / 5.0)
    return approach + touch + ctx.success.float(), {"approach": approach, "touch": touch}
"""


def _write(tmp_path, src: str) -> str:
    p = tmp_path / "compute_reward.py"
    p.write_text(src, encoding="utf-8")
    return str(p)


def test_context_stub_is_the_prompt_environment_description():
    stub = C.context_stub_source()
    assert "class RewardContext" in stub
    for name in ("palm_pos", "palm_normal", "palm_finger_dir", "link_pos", "link_cup_force", "palm_cup_force",
                 "hand_q_norm", "cup_axis", "cup_radius", "cup_half_height", "goal_dist",
                 "success_tol", "success", "lifted", "actions", "prev_actions"):
        assert name in stub, name
    assert "@property" not in stub, "헬퍼 속성은 프롬프트에 필요 없다"


def test_context_carries_no_pouring_fields():
    names = set(C.TENSOR_FIELDS) | set(C.SCALAR_FIELDS)
    assert names, "필드 목록이 비었다"
    assert not any(n.startswith(("src_", "rcv_", "bead", "d_in_", "d_spill")) for n in names), names


def test_fake_context_matches_the_documented_layout():
    ctx = V.make_fake_context(8)
    assert ctx.link_pos.shape == (8, 5, 3, 3)
    assert ctx.link_cup_force.shape == (8, 5, 3)
    assert ctx.hand_q_norm.shape == (8, 19) and ctx.hand_target_norm.shape == (8, 19)
    assert ctx.actions.shape == (8, 26) and ctx.prev_actions.shape == (8, 26)
    assert ctx.success.dtype == torch.bool and ctx.lifted.dtype == torch.bool
    # ★09.15 사용자 3단계 — 기본 손 자세 · 게이트 래치(접근 완료·인벨롭 완료)
    assert ctx.hand_default_q_norm.shape == (8, 19)
    assert ctx.approach_done.shape == (8,) and ctx.approach_done.dtype == torch.bool
    assert ctx.envelope_done.shape == (8,) and ctx.envelope_done.dtype == torch.bool
    # ★09.15 사용자 "손가락 방향(palm_ee_z)" — 접근 방향 게이트가 쓰는 손바닥 프레임 z 축
    assert ctx.palm_finger_dir.shape == (8, 3)
    assert ctx.num_envs == 8


def test_validator_passes_a_wellformed_reward(tmp_path):
    rep = V.validate(_write(tmp_path, SAMPLE))
    assert rep.ok, rep.errors
    assert set(rep.term_stats) == {"approach", "touch"}
    assert "link_cup_force" in rep.fields_used


def test_validator_rejects_hallucinated_and_pouring_fields(tmp_path):
    rep = V.validate(_write(tmp_path, SAMPLE.replace("ctx.palm_pos", "ctx.src_palm_pos")))
    assert not rep.ok
    assert any("src_palm_pos" in e for e in rep.errors), rep.errors


def test_validator_rejects_imports_and_ctx_mutation(tmp_path):
    rep = V.validate(_write(tmp_path, "import numpy\n" + SAMPLE))
    assert not rep.ok and any("numpy" in e for e in rep.errors)
    mutated = SAMPLE.replace("    d = torch", "    ctx.cup_pos = ctx.palm_pos\n    d = torch")
    rep = V.validate(_write(tmp_path, mutated))
    assert not rep.ok and any("읽기 전용" in e for e in rep.errors)


@pytest.mark.parametrize("line", [
    "    ctx.cup_radius.clamp_(min=0.02)\n",       # 제자리 메서드
    "    ctx.goal_pos[:, 2] += 0.01\n",            # 첨자 대입(AugAssign)
    "    ctx.cup_pos[:, 0].add_(1.0)\n",           # 첨자 뒤 제자리 메서드
])
def test_validator_rejects_in_place_ops_on_ctx(tmp_path, line):
    # ★09.14 리뷰(HIGH): ctx 필드는 env 가 보상 뒤에 다시 읽는 버퍼였다(env 는 이제 복사본을 넘기지만 이중으로 막는다).
    rep = V.validate(_write(tmp_path, SAMPLE.replace("    d = torch", line + "    d = torch")))
    assert not rep.ok and any("읽기 전용" in e for e in rep.errors), rep.errors


def test_validator_allows_in_place_ops_on_local_tensors(tmp_path):
    src = SAMPLE.replace("    approach = torch.exp", "    d.clamp_(max=1.0)\n    approach = torch.exp")
    rep = V.validate(_write(tmp_path, src))
    assert rep.ok, rep.errors


@pytest.mark.skipif(not V.cuda_usable(), reason="연산 가능한 cuda 없음(Isaac python 으로 돌리면 실행된다)")
def test_validator_catches_device_mismatch_on_cuda(tmp_path):
    # ★09.14 리뷰(HIGH): device 없이 만든 상수는 cpu 드라이런에서 통과하고 학습 첫 스텝에서 죽는다.
    src = SAMPLE.replace(
        "    touch = torch.tanh(ctx.link_cup_force.sum(dim=(1, 2)) / 5.0)",
        "    w = torch.tensor([1.0, 0.5, 0.5, 0.5, 0.5])\n"
        "    touch = torch.tanh((ctx.link_cup_force.sum(dim=2) * w).sum(dim=1) / 5.0)")
    path = _write(tmp_path, src)
    assert V.validate(path, devices=("cpu",)).ok
    rep = V.validate(path)
    assert not rep.ok and any(e.startswith("[cuda]") for e in rep.errors), rep.errors


def test_loader_enforces_the_return_shapes():
    ctx = V.make_fake_context(4)
    with pytest.raises(RuntimeError):
        L.call_reward_fn(lambda c: (torch.zeros(3), {}), ctx)
    with pytest.raises(RuntimeError):
        L.call_reward_fn(lambda c: (torch.zeros(4), {"x": torch.zeros(4, 2)}), ctx)
    total, terms = L.call_reward_fn(lambda c: (torch.zeros(4), {"x": torch.ones(4)}), ctx)
    assert total.shape == (4,) and terms["x"].shape == (4,)


def test_empty_code_path_is_zero_reward():
    fn, src = L.load_reward_fn("")
    total, terms = L.call_reward_fn(fn, V.make_fake_context(3))
    assert torch.all(total == 0.0) and terms == {}
    assert "zero" in src


def test_prompt_describes_single_arm_full_joint_hand_and_the_task():
    txt = P.render_prompt(P.PromptSpec(task="THE TASK SENTENCE"))
    assert "THE TASK SENTENCE" in txt
    assert "Box(-1, 1, (26,), float32)" in txt
    assert "class RewardContext" in txt and "compute_reward(ctx: RewardContext)" in txt
    for joint in ("r_hj_thumb_3", "r_hj_index_2", "r_hj_pinky_4"):
        assert joint in txt, joint
    low = txt.lower()
    assert "bead" not in low and "receiver" not in low, "붓기 프롬프트가 섞였다"


def test_prompt_joint_table_matches_the_profile_order():
    from openarm.agnostic.modules.robot_profiles import TESOLLO_RIGHT_SHORT_TL as prof
    assert tuple(P.HAND_JOINT_NAMES) == tuple(prof.hand_joint_names)
    assert len(P.HAND_JOINT_NAMES) == 19


def test_prompt_env_facts_follow_the_variant():
    # ★09.14 reach(최종 목표 env) 추가 — 트랙마다 생성기가 보는 환경 사실(시작 거리·컵·팔 속도·에피소드)이 다르다.
    env = P.render_prompt(P.PromptSpec(task="T"))
    for tok in ("29 mm to 44 mm", "roughly 0.16 m", "0.025 * a", "0.15 rad/s", "600 steps (10 s)"):
        assert tok in env, tok
    reach = P.render_prompt(P.PromptSpec(task="T", variant="reach"))
    for tok in ("44 mm to 81 mm", "roughly 0.38 m", "0.05 * a", "0.3 rad/s", "900 steps (15 s)",
                "+x points from the robot toward the table", "open cups of several sizes and a closed shaker",
                # ★09.15 시작 상태 커리큘럼 — 생성기가 두 출발을 알아야 한다(환경 사실)
                "From the second episode on", "with probability one half", "about 4.5 cm from the cup's side",
                "starts with approach_done already set"):
        assert tok in reach, tok
    for tok in ("29 mm to 44 mm", "0.16 m", "600 steps"):
        assert tok not in reach, tok
    with pytest.raises(KeyError):
        P.render_prompt(P.PromptSpec(task="T", variant="nope"))


def test_reach_variant_facts_match_the_registered_leaf_cfg():
    # 프롬프트의 팔 속도·에피소드 길이가 등록된 env cfg 와 어긋나면 생성기가 틀린 세계를 본다.
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "grasp_fj_t2r_env_cfg.py").read_text(encoding="utf-8")
    reach = src.split("class GraspFJT2RReachEnvCfg", 1)[1]
    f = P.VARIANTS["reach"]
    assert f"k_arm: float = {f.k_arm}" in reach and f"arm_slew_rad_s: float = {f.arm_slew}" in reach
    assert f"episode_length_s: float = {f.episode_s}" in reach
    assert f.episode_steps == round(f.episode_s * 60)


def test_rand_variant_facts_match_the_rand_leaf():
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "grasp_fj_t2r_env_cfg.py").read_text(encoding="utf-8")
    reach = src.split("class GraspFJT2RReachEnvCfg", 1)[1]
    f = P.VARIANTS["rand"]
    # rand leaf 는 reach 를 상속하고 팔 속도·에피소드를 바꾸지 않는다
    assert f"k_arm: float = {f.k_arm}" in reach and f"arm_slew_rad_s: float = {f.arm_slew}" in reach
    assert f"episode_length_s: float = {f.episode_s}" in reach and f.episode_steps == round(f.episode_s * 60)
    for tok in ("x between 0.10 m and 0.40 m", "y between -0.30 m and 0.00 m", "approach_done is never set"):
        assert tok in f.scene, tok
    assert "probability" not in f.scene, "rand 판에는 두 번째 출발이 없다"

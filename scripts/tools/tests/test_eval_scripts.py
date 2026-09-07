"""평가 스크립트의 태스크 종속성 — 소스 수준 검사.

두 스크립트 모두 상단에서 Isaac 앱을 띄우므로 import 할 수 없다. 그래서 소스 문자열로
계약을 잠근다(scripts/analysis/tests 의 기존 관례와 같다).

지키는 것:
  ① eval_sim2real 이 grasp_v1 두 개 말고 다른 태스크도 받는다
  ② eval_grasp_robustness 가 전 트랙을 등록하고, 못 재는 태스크는 **사유를 밝히고** 멈춘다
"""

from __future__ import annotations

from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2]
_S2R = _SCRIPTS / "eval_s2r/eval_sim2real.py"
_ROBUST = _SCRIPTS / "eval_grasp/eval_grasp_robustness.py"


# ---------------------------------------------------------------- eval_s2r
def test_s2r_accepts_an_explicit_task_id():
    src = _S2R.read_text(encoding="utf-8")
    assert 'parser.add_argument("--task"' in src
    assert "def resolve_task(args)" in src
    assert "return args.task or TASK_BY_ROBOT[args.robot]" in src


def test_s2r_no_longer_reads_the_hardcoded_map_directly():
    src = _S2R.read_text(encoding="utf-8")
    assert "TASK_BY_ROBOT[args_cli.robot]" not in src, (
        "기본값 조회가 남아 있으면 --task 가 무시되는 경로가 생긴다"
    )
    assert src.count("resolve_task(args_cli)") == 2


def test_s2r_keeps_the_previous_default_so_existing_calls_are_unchanged():
    src = _S2R.read_text(encoding="utf-8")
    assert '"left": "open-tesol_l_grasp_v1-play-lstm"' in src
    assert '"right": "open-tesol_r_grasp_v1-play-lstm"' in src


# ---------------------------------------------------------------- robustness
def test_robustness_registers_every_track_not_just_tesollo():
    """`import openarm.tesollo` 만으로는 agnostic·gripper 태스크가 gym 에 없다."""
    src = _ROBUST.read_text(encoding="utf-8")
    assert "import openarm.tasks" in src
    assert "import openarm.tesollo" not in src


def test_robustness_states_why_it_cannot_measure_an_unsupported_task():
    """루프 깊은 곳의 raw AttributeError 대신 진입 시점에 사유를 밝힌다."""
    src = _ROBUST.read_text(encoding="utf-8")
    assert "def assert_supported(uenv, task: str)" in src
    assert "assert_supported(uenv, args_cli.task)" in src
    for attr in ("lift_ready_latched_buf", "hand_dof_indices",
                 "wrench_max_accel", "hold_rotation_perturb_max_accel"):
        assert attr in src


def test_robustness_points_at_the_generic_harness_in_its_refusal():
    """막다른 길로 두지 않는다 — 일반 지표를 어디서 얻는지 알려준다."""
    src = _ROBUST.read_text(encoding="utf-8")
    assert "play.py --eval_episodes" in src

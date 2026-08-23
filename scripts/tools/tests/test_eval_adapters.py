"""eval_adapters — 평가 하네스가 태스크를 가리지 않게 하는 층.

배경: play.py 의 `--eval_episodes` 블록은 grasp_v1/v2 direct-env 속성
(`in_success_region`, `_obj_total_episodes`, `binary_contact_buf` …)을 `try` 없이
직참조한다. 업그레이드된 신규 태스크는 그 속성을 하나도 노출하지 않아 AttributeError 로
즉사한다. 재생은 되는데 정량 평가만 0 이었다.

여기서 지키는 것:
  ① grasp_v2 계열은 **예전 그대로** 그 블록으로 간다 (거동 보존)
  ② 그 밖의 태스크는 죽지 않고 공통 지표를 낸다
  ③ 손가락 라벨이 5지 하드코딩이 아니다 (2지 그리퍼에서도 의미가 산다)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import eval_adapters as EA  # noqa: E402


class _Stub:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _grasp_v2_env(n: int = 4):
    return _Stub(
        in_success_region=torch.zeros(n, dtype=torch.bool),
        object_pos=torch.zeros(n, 3),
        object_init_pos=torch.zeros(n, 3),
        binary_contact_buf=torch.zeros(n, 5, dtype=torch.bool),
        middle_binary_contact_buf=torch.zeros(n, 5, dtype=torch.bool),
        distal_binary_contact_buf=torch.zeros(n, 5, dtype=torch.bool),
        fingertip_pos=torch.zeros(n, 5, 3),
        _object_names=["cup", "shaker"],
        object_idx=torch.zeros(n, dtype=torch.long),
        _total_episodes=0,
        _successful_episodes=0,
        _obj_total_episodes=torch.zeros(2),
        _obj_success_episodes=torch.zeros(2),
    )


# ---------------------------------------------------------------- 어댑터 선택
def test_grasp_v2_env_still_routes_to_the_legacy_block():
    assert EA.select(_grasp_v2_env()) == EA.GRASP_V2


def test_env_missing_even_one_required_attribute_falls_back_to_common():
    """부분 노출을 grasp_v2 로 오인하면 그 블록 안에서 AttributeError 가 난다."""
    env = _grasp_v2_env()
    del env.binary_contact_buf
    assert EA.select(env) == EA.COMMON


def test_new_fabric_task_env_routes_to_common_instead_of_dying():
    """agnostic 태스크가 실제로 노출하는 것만 가진 env."""
    env = _Stub(object_idx=torch.zeros(4, dtype=torch.long), profile=_Stub(fingers=("thumb",)))
    assert EA.select(env) == EA.COMMON


def test_missing_attributes_are_reported_so_the_reason_is_visible():
    env = _grasp_v2_env()
    del env.in_success_region
    del env.fingertip_pos
    missing = EA.missing_grasp_v2_attrs(env)
    assert set(missing) == {"in_success_region", "fingertip_pos"}


# ---------------------------------------------------------------- 손가락 라벨
def test_finger_labels_come_from_the_profile_when_available():
    env = _Stub(profile=_Stub(fingers=("jaw1", "jaw2")))
    assert EA.finger_labels(env, 2) == ["jaw1", "jaw2"]


def test_finger_labels_fall_back_to_sensor_body_keys():
    env = _Stub(profile=_Stub(finger_sensor_bodies={"thumb": (), "index": ()}))
    assert EA.finger_labels(env, 2) == ["thumb", "index"]


def test_finger_labels_never_invent_names_for_a_two_jaw_gripper():
    """5지 하드코딩이면 2지 그리퍼에 thumb/index 가 붙어 오독을 부른다."""
    assert EA.finger_labels(_Stub(), 2) == ["f0", "f1"]


def test_finger_labels_are_truncated_to_the_actual_count():
    env = _Stub(profile=_Stub(fingers=("thumb", "index", "middle", "ring", "pinky")))
    assert EA.finger_labels(env, 3) == ["thumb", "index", "middle"]


# ---------------------------------------------------------------- 공통 지표
def test_common_accumulator_counts_episodes_on_done():
    acc = EA.CommonEvalAccumulator(num_envs=2)
    rew = torch.tensor([1.0, 2.0])
    none = torch.zeros(2, dtype=torch.bool)
    acc.add_step(rew, none, torch.zeros(2, 4))
    acc.add_step(rew, torch.tensor([True, False]), torch.zeros(2, 4))

    assert acc.episodes == 1
    assert acc.episode_returns == [2.0]      # env0 이 두 스텝 동안 1.0 씩
    assert acc.episode_lengths == [2]


def test_common_accumulator_resets_only_the_finished_env():
    acc = EA.CommonEvalAccumulator(num_envs=2)
    acc.add_step(torch.tensor([1.0, 1.0]), torch.tensor([True, False]), torch.zeros(2, 1))
    acc.add_step(torch.tensor([1.0, 1.0]), torch.tensor([False, True]), torch.zeros(2, 1))

    assert acc.episodes == 2
    assert acc.episode_returns == [1.0, 2.0]
    assert acc.episode_lengths == [1, 2]


def test_common_accumulator_tracks_action_saturation_and_nan():
    acc = EA.CommonEvalAccumulator(num_envs=1)
    acc.add_step(torch.zeros(1), torch.zeros(1, dtype=torch.bool),
                 torch.tensor([[1.0, 0.0, -1.0, float("nan")]]))

    assert acc.nan_steps == 1
    assert acc.saturated_frac() == pytest.approx(2 / 4)


def test_common_accumulator_separates_terminated_from_truncated():
    """실패 종료와 시간초과는 전혀 다른 사건이다 — dones 만으로는 못 가른다."""
    acc = EA.CommonEvalAccumulator(num_envs=2)
    env = _Stub(reset_terminated=torch.tensor([True, False]),
                reset_time_outs=torch.tensor([False, True]))
    acc.add_step(torch.zeros(2), torch.tensor([True, True]), torch.zeros(2, 1), env=env)

    assert acc.terminated == 1
    assert acc.truncated == 1


def test_common_report_states_the_numbers_and_the_missing_adapter():
    acc = EA.CommonEvalAccumulator(num_envs=1)
    acc.add_step(torch.tensor([1.5]), torch.tensor([True]), torch.zeros(1, 2))
    text = acc.report(task="open-sens_r_grasp_sensor-play-lstm",
                      missing=("in_success_region",))

    assert "1.5" in text
    assert "in_success_region" in text
    assert "open-sens_r_grasp_sensor-play-lstm" in text


def test_common_report_is_safe_with_zero_episodes():
    assert "에피소드 0" in EA.CommonEvalAccumulator(num_envs=1).report(task="t", missing=())


# ---------------------------------------------------------------- play.py 배선
_PLAY = _TOOLS.parents[0] / "reinforcement_learning/rl_games/play.py"


def test_play_enters_the_grasp_v2_block_only_through_the_adapter():
    src = _PLAY.read_text(encoding="utf-8")
    assert "if args_cli.eval_episodes > 0 and _eval_route == _eval_adapters.GRASP_V2:" in src
    assert "if args_cli.eval_episodes > 0:\n" not in src, (
        "무조건 진입하는 옛 진입점이 남아 있다 — 신규 태스크가 다시 즉사한다"
    )


def test_play_keeps_the_reward_it_used_to_discard():
    """공통 지표가 에피소드 리턴을 세려면 step 의 보상이 필요하다."""
    src = _PLAY.read_text(encoding="utf-8")
    assert "obs, _rew, dones, _ = env.step(actions)" in src


def test_play_builds_the_common_accumulator_from_the_unwrapped_env():
    src = _PLAY.read_text(encoding="utf-8")
    assert "_eval_adapters.CommonEvalAccumulator(_ev.num_envs)" in src


def test_grasp_v2_block_body_still_references_its_own_buffers():
    """본문을 옮기지 않았다는 확인 — 거동 보존이 이 변경의 전제다."""
    src = _PLAY.read_text(encoding="utf-8")
    for attr in ("_ge.in_success_region", "_ge._obj_total_episodes.zero_()",
                 "_ge.binary_contact_buf"):
        assert attr in src, f"{attr} 가 사라졌다 — grasp_v2 경로가 바뀌었다"


def test_finger_labels_keep_the_callers_legacy_names_when_no_profile_exists():
    """구 트랙 env 는 profile 을 노출하지 않는다(grasp_v1/v2/sensor 전부 실측 0건).

    default 를 주면 기존 출력이 한 글자도 바뀌지 않는다 — grasp_v2 리포트 보존의 근거다.
    """
    legacy = ("thumb", "index", "middle", "ring", "pinky")
    assert EA.finger_labels(_Stub(), 5, default=legacy) == list(legacy)


def test_profile_names_win_over_the_legacy_default():
    env = _Stub(profile=_Stub(fingers=("jaw1", "jaw2")))
    assert EA.finger_labels(env, 2, default=("thumb", "index")) == ["jaw1", "jaw2"]


def test_play_no_longer_hardcodes_five_finger_labels():
    src = _PLAY.read_text(encoding="utf-8")
    assert '["thumb", "index", "middle", "ring", "pinky"]' not in src
    assert src.count("_eval_adapters.finger_labels(") == 3

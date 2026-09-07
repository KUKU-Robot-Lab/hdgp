"""policy_contract — 학습된 정책의 실제 차원을 어디서 얻는가.

★소스 코드는 진실원천이 **아니다.** 네 신규 태스크 어디에도 `*_constants.py` 가 없고
  차원은 프로필에 따라 `__post_init__` 에서 파생된다. 런 산출물만이 사실을 안다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import policy_contract as PC  # noqa: E402


# ---------------------------------------------------------------- 체크포인트 판독
def _state_dict(obs: int, act: int) -> dict:
    import torch
    return {
        "running_mean_std.running_mean": torch.zeros(obs),
        "a2c_network.actor_mlp.0.weight": torch.zeros(256, obs),
        "a2c_network.mu.weight": torch.zeros(act, 64),
        "a2c_network.value.weight": torch.zeros(1, 64),
    }


def test_checkpoint_dims_come_from_the_first_actor_layer_and_the_mu_head():
    obs, act = PC.dims_from_state_dict(_state_dict(114, 23))
    assert (obs, act) == (114, 23)


def test_checkpoint_reader_raises_instead_of_guessing_when_shapes_are_absent():
    """차원을 못 읽으면 0 으로 채우지 않는다 — 조용히 틀린 계약이 최악이다."""
    with pytest.raises(PC.ContractError):
        PC.dims_from_state_dict({"something.else": None})


def test_checkpoint_reader_tolerates_an_lstm_network():
    import torch
    sd = _state_dict(114, 23)
    sd["a2c_network.rnn.rnn.weight_ih_l0"] = torch.zeros(4096, 626)
    assert PC.dims_from_state_dict(sd) == (114, 23)


# ---------------------------------------------------------------- 런 해석
def _run(tmp: Path, *, env_yaml: str | None, obs: int, act: int) -> Path:
    import torch
    run = tmp / "lstm_test1"
    (run / "nn").mkdir(parents=True)
    (run / "params").mkdir(parents=True)
    if env_yaml is not None:
        (run / "params/env.yaml").write_text(env_yaml)
    torch.save({"model": _state_dict(obs, act)}, run / "nn/policy.pth")
    return run


def test_env_yaml_is_preferred_and_the_source_is_recorded(tmp_path: Path):
    run = _run(tmp_path, env_yaml="observation_space: 114\nstate_space: 121\naction_space: 23\n",
               obs=114, act=23)
    rc = PC.resolve_run(run)
    assert (rc.observation_dim, rc.state_dim, rc.action_dim) == (114, 121, 23)
    assert rc.source == "env.yaml"


def test_manager_based_run_falls_back_to_the_checkpoint(tmp_path: Path):
    """gripper/left/grasp_sensor 는 ManagerBased 라 env.yaml 에 차원이 아예 없다(실측)."""
    run = _run(tmp_path, env_yaml="seed: 42\nobservations:\n  policy: {}\n", obs=36, act=8)
    rc = PC.resolve_run(run)
    assert (rc.observation_dim, rc.action_dim) == (36, 8)
    assert rc.state_dim is None
    assert rc.source == "checkpoint"


def test_disagreement_between_the_two_sources_is_reported_not_silently_preferred(tmp_path: Path):
    run = _run(tmp_path, env_yaml="observation_space: 114\naction_space: 23\n", obs=99, act=23)
    rc = PC.resolve_run(run, verify=True)
    assert rc.mismatch is not None
    assert "114" in rc.mismatch and "99" in rc.mismatch


def test_agreement_leaves_no_mismatch(tmp_path: Path):
    run = _run(tmp_path, env_yaml="observation_space: 114\naction_space: 23\n", obs=114, act=23)
    assert PC.resolve_run(run, verify=True).mismatch is None


def test_run_without_any_checkpoint_or_dims_raises(tmp_path: Path):
    run = tmp_path / "empty"
    (run / "nn").mkdir(parents=True)
    (run / "params").mkdir(parents=True)
    (run / "params/env.yaml").write_text("seed: 1\n")
    with pytest.raises(PC.ContractError):
        PC.resolve_run(run)


# ---------------------------------------------------------------- 실제 런
_REAL = PC.HDGP_ROOT / "log/rl_games/open-sens/right/grasp-sensor/lstm_test3"


@pytest.mark.skipif(not _REAL.is_dir(), reason="해당 런이 이 머신에 없다")
def test_real_agnostic_grasp_sensor_run_agrees_across_both_sources():
    rc = PC.resolve_run(_REAL, verify=True)
    assert (rc.observation_dim, rc.state_dim, rc.action_dim) == (114, 121, 23)
    assert rc.mismatch is None


# ---------------------------------------------------------------- 런 탐색
def test_discover_returns_run_directories_not_task_directories(tmp_path: Path):
    """`<robot>/<side>/<task>/<run>/{nn,params}` 에서 run 을 집어야 한다.

    한 단계 위(task)를 집으면 모든 행이 "계약 불가" 로 나온다 — 실제로 그랬다.
    """
    run = tmp_path / "open-sens/right/grasp-sensor/lstm_test3"
    (run / "nn").mkdir(parents=True)
    (run / "params").mkdir(parents=True)

    assert PC.discover_runs(tmp_path) == [run]

from __future__ import annotations

from pathlib import Path

import pytest

from vlm.pouring.checkpoint_resolver import CheckpointResolver, read_policy_contract


def _make_run(root: Path, task: str, run_name: str) -> Path:
    run = root / "log/rl_games/open-tesol/right" / task / run_name
    (run / "nn").mkdir(parents=True)
    (run / "params").mkdir()
    return run


def test_resolver_returns_checkpoint_and_neighbor_params(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "grasp-v1", "lstm_test1")
    (run / "nn/open-tesol_r_grasp_v1-lstm.pth").touch()
    (run / "params/agent.yaml").write_text("params: {}\n")
    (run / "params/env.yaml").write_text("scene: {}\n")

    result = CheckpointResolver(tmp_path).resolve(
        task_id="open-tesol_r_grasp_v1-lstm",
        run_dir="lstm_test1",
    )

    assert result.checkpoint.name == "open-tesol_r_grasp_v1-lstm.pth"
    assert result.agent_yaml == run / "params/agent.yaml"
    assert result.env_yaml == run / "params/env.yaml"


def test_resolver_rejects_missing_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="lstm_test1"):
        CheckpointResolver(tmp_path).resolve("open-tesol_r_grasp_v1-lstm", "lstm_test1")


def test_resolver_rejects_ambiguous_run_selector(tmp_path: Path) -> None:
    _make_run(tmp_path, "grasp-v1", "lstm_test1")
    _make_run(tmp_path, "grasp-v1", "lstm_test2")

    with pytest.raises(ValueError, match="exactly one"):
        CheckpointResolver(tmp_path).resolve("open-tesol_r_grasp_v1-lstm", "lstm_test*")


def test_resolver_accepts_explicit_checkpoint_but_still_requires_neighbor_params(tmp_path: Path) -> None:
    run = _make_run(tmp_path, "pour-v1", "manual")
    checkpoint = run / "nn/custom.pth"
    checkpoint.touch()
    (run / "params/agent.yaml").write_text("params: {}\n")
    (run / "params/env.yaml").write_text("scene: {}\n")

    result = CheckpointResolver(tmp_path).resolve(
        "open-tesol_b_pour_v1-lstm",
        "ignored",
        checkpoint=checkpoint,
    )

    assert result.checkpoint == checkpoint


def test_resolver_rejects_unknown_task_without_explicit_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="unsupported task"):
        CheckpointResolver(tmp_path).resolve("unknown-task", "test1")


def test_resolver_accepts_injected_task_map_for_retrained_runs(tmp_path: Path) -> None:
    run = tmp_path / "log/rl_games/pipeline/right/grasp-fabric/agn_test1"
    (run / "nn").mkdir(parents=True)
    (run / "params").mkdir()
    (run / "nn/agn_grasp-fabric.pth").touch()
    (run / "params/agent.yaml").write_text("params: {}\n")
    (run / "params/env.yaml").write_text("scene: {}\n")

    resolver = CheckpointResolver(
        tmp_path,
        task_logs={"agn_grasp-fabric": ("pipeline/right", "grasp-fabric")},
    )
    result = resolver.resolve("agn_grasp-fabric", "agn_test1")

    assert result.checkpoint == run / "nn/agn_grasp-fabric.pth"


# ---------------------------------------------------------------------------
# read_policy_contract — 학습된 정책의 실제 차원. 소스가 아니라 런 덤프가 진실원천이다.
# ---------------------------------------------------------------------------
_REAL_DUMP_HEAD = """\
seed: 42
observation_space: 114
observation_noise_model: null
state_space: 121
episode_length_s: 8.0
action_space: 23
"""


def test_contract_reads_all_three_dims_from_a_real_style_dump(tmp_path: Path) -> None:
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text(_REAL_DUMP_HEAD)

    contract = read_policy_contract(env_yaml)

    assert contract.observation_dim == 114
    assert contract.action_dim == 23
    assert contract.state_dim == 121


def test_contract_state_dim_is_none_when_the_run_had_no_critic_group(tmp_path: Path) -> None:
    """gripper/left/grasp_sensor 는 ManagerBased 라 policy 그룹만 있다."""
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text("observation_space: 36\naction_space: 8\n")

    assert read_policy_contract(env_yaml).state_dim is None


def test_contract_ignores_nested_keys_that_merely_share_the_name(tmp_path: Path) -> None:
    """Isaac 덤프는 python 태그를 품어 safe_load 가 깨진다 — 최상위 평문 줄만 읽는다."""
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text(
        "observation_space: 114\n"
        "action_space: 23\n"
        "some_block:\n"
        "  observation_space: 999\n"
        "  action_space: 999\n"
    )

    contract = read_policy_contract(env_yaml)

    assert (contract.observation_dim, contract.action_dim) == (114, 23)


def test_contract_refuses_a_dump_without_the_required_keys(tmp_path: Path) -> None:
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text("seed: 42\n")

    with pytest.raises(ValueError, match="lacks contract keys"):
        read_policy_contract(env_yaml)

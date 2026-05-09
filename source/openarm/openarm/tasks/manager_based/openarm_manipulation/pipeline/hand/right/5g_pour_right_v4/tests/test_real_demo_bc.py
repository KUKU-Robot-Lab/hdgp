from __future__ import annotations

from pathlib import Path

import pytest
import torch

from real_demo_bc import (
    DEFAULT_REAL_DEMO_PATHS,
    RealDemoBCBuffer,
    load_real_demo_episodes,
    pose_action_to_target,
    target_pose_to_pose_action,
)


_TASK_DIR = Path(__file__).parent.parent
_AGENT_CFG_TEXT = (_TASK_DIR / "config/agents/rl_games_ppo_lstm_bc_cfg.yaml").read_text()


def test_real_demo_bc_weight_init_matches_finetune_plan() -> None:
    assert "real_demo_bc_weight_init: 0.5" in _AGENT_CFG_TEXT


def test_a11_a20_hdf5_contract_and_timing() -> None:
    episodes = load_real_demo_episodes(DEFAULT_REAL_DEMO_PATHS, demo_stride=1)

    assert len(episodes) == 10
    for episode in episodes:
        assert episode.obs.shape[1] == 91
        assert episode.raw_actions.shape[1] == 18
        assert episode.actions.shape[1] == 11
        assert torch.isfinite(episode.obs).all()
        assert torch.isfinite(episode.actions).all()
        dt_ms = (episode.timestamps_ns[1:] - episode.timestamps_ns[:-1]).float() / 1.0e6
        assert float(dt_ms.median()) == pytest.approx(10.0, abs=1.5)


def test_real_demo_buffer_samples_padded_variable_length_sequences(tmp_path: Path) -> None:
    episodes = load_real_demo_episodes(DEFAULT_REAL_DEMO_PATHS[:2], demo_stride=2)
    buffer = RealDemoBCBuffer(episodes, device="cpu", pour_sample_ratio=0.6)

    batch = buffer.sample(batch_size=5, seq_len=32)

    assert batch is not None
    assert batch["obs"].shape == (5, 32, 91)
    assert batch["actions"].shape == (5, 32, 11)
    assert batch["mask"].shape == (5, 32)
    assert batch["mask"].dtype == torch.bool
    assert batch["mask"].any()
    assert 0.0 <= buffer.last_pour_phase_ratio <= 1.0


def test_target_pose_action_round_trip_matches_v4_delta_convention() -> None:
    base_pose = torch.zeros(2, 7)
    base_pose[:, 6] = 1.0
    target_pose = torch.tensor(
        [
            [0.05, -0.02, 0.03, 0.0, 0.0749297, 0.0, 0.9971888],
            [-0.04, 0.01, 0.02, 0.0499792, 0.0, 0.0, 0.9987503],
        ],
        dtype=torch.float32,
    )
    delta_mins = torch.tensor([-0.5, -0.5, -0.5, -2.0944, -2.0944, -2.0944])
    delta_maxs = torch.tensor([0.5, 0.5, 0.5, 2.0944, 2.0944, 2.0944])

    action = target_pose_to_pose_action(target_pose, base_pose, delta_mins, delta_maxs)
    reconstructed = pose_action_to_target(action, base_pose, delta_mins, delta_maxs)

    assert torch.max(torch.abs(reconstructed[:, :3] - target_pose[:, :3])) < 1.0e-5
    quat_dot = torch.abs(torch.sum(reconstructed[:, 3:7] * target_pose[:, 3:7], dim=-1))
    assert torch.min(quat_dot) > 0.999

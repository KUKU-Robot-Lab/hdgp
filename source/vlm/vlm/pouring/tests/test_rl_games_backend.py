from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from vlm.pouring.checkpoint_resolver import PolicyArtifacts, PolicyContract  # noqa: E402
from vlm.pouring.rl_games_backend import RlGamesPolicyBackend  # noqa: E402

_CONTRACT = PolicyContract(observation_dim=4, action_dim=2)


def _artifacts(tmp_path: Path | None = None) -> PolicyArtifacts:
    root = tmp_path or Path("run")
    return PolicyArtifacts(
        "open-tesol_b_pour_v1-lstm", root, root / "model.pth",
        root / "agent.yaml", root / "env.yaml",
    )


class FakePlayer:
    """Mimics the rl_games player surface the backend touches."""

    def __init__(self, num_envs: int, hidden: int = 3, rnn: bool = True) -> None:
        self.is_rnn = rnn
        self.states = [torch.ones(1, num_envs, hidden)] if rnn else None
        self.last_obs = None

    def obs_to_torch(self, obs):
        return obs

    def get_action(self, obs, is_deterministic):
        self.last_obs = obs
        assert is_deterministic
        # action = [row_sum, -row_sum] so gather correctness is observable.
        row_sum = obs.sum(dim=1, keepdim=True)
        return torch.cat([row_sum, -row_sum], dim=1)


def _backend(num_envs: int = 4, player: FakePlayer | None = None) -> RlGamesPolicyBackend:
    return RlGamesPolicyBackend(
        _artifacts(),
        num_envs=num_envs,
        device="cpu",
        contract=_CONTRACT,
        player=player if player is not None else FakePlayer(num_envs),
    )


def test_infer_scatters_subset_and_gathers_matching_rows() -> None:
    player = FakePlayer(num_envs=4)
    backend = _backend(player=player)

    actions = backend.infer(
        (2, 0),
        ((1.0, 1.0, 1.0, 1.0), (0.5, 0.5, 0.0, 0.0)),
    )

    assert actions == ((4.0, -4.0), (1.0, -1.0))
    assert player.last_obs is not None
    # Inactive envs run with zero rows so the full RNN batch stays aligned.
    assert player.last_obs.shape == (4, 4)
    assert float(player.last_obs[1].abs().sum()) == 0.0
    assert float(player.last_obs[3].abs().sum()) == 0.0


def test_reset_zeroes_only_the_given_envs_hidden_state() -> None:
    player = FakePlayer(num_envs=4)
    backend = _backend(player=player)

    backend.reset((1, 3))

    assert player.states is not None
    state = player.states[0]
    assert float(state[:, 1].abs().sum()) == 0.0
    assert float(state[:, 3].abs().sum()) == 0.0
    assert float(state[:, 0].abs().sum()) > 0.0
    assert float(state[:, 2].abs().sum()) > 0.0


def test_reset_is_a_no_op_before_load_and_for_feedforward_players() -> None:
    unloaded = RlGamesPolicyBackend(_artifacts(), num_envs=2, contract=_CONTRACT)
    unloaded.reset((0,))

    feedforward = _backend(num_envs=2, player=FakePlayer(2, rnn=False))
    feedforward.reset((0, 1))


def test_infer_validates_ids_and_dimensions() -> None:
    backend = _backend()

    with pytest.raises(ValueError, match="must align"):
        backend.infer((0,), ())
    with pytest.raises(ValueError, match="unique"):
        backend.infer((1, 1), ((0.0,) * 4, (0.0,) * 4))
    with pytest.raises(IndexError, match="out of range"):
        backend.infer((4,), ((0.0,) * 4,))
    with pytest.raises(ValueError, match="4D"):
        backend.infer((0,), ((0.0, 0.0),))
    with pytest.raises(IndexError, match="out of range"):
        backend.reset((9,))


def test_contract_falls_back_to_env_yaml(tmp_path: Path) -> None:
    env_yaml = tmp_path / "env.yaml"
    env_yaml.write_text("observation_space: 51\naction_space: 15\n")
    backend = RlGamesPolicyBackend(
        PolicyArtifacts("open-tesol_b_pour_v1-lstm", tmp_path, tmp_path / "model.pth",
                        tmp_path / "agent.yaml", env_yaml),
        num_envs=2,
        player=FakePlayer(2),
    )
    assert (backend.contract.observation_dim, backend.contract.action_dim) == (51, 15)


def test_num_envs_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        RlGamesPolicyBackend(_artifacts(), num_envs=0, contract=_CONTRACT)

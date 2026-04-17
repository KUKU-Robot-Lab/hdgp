from __future__ import annotations

import pytest
import torch

from recurrent_gate import RecurrentGateState, install_recurrent_gate, readiness_score, resolve_success_rate


class _FakeRNN(torch.nn.Module):
    def forward(self, out, states, dones, bptt_len):
        return out + 10.0, states


class _FakeNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.has_rnn = True
        self.separate = False
        self.is_rnn_before_mlp = True
        self.rnn_units = 2
        self.rnn_ln = False
        self.rnn_concat_output = False
        self.has_cnn = False
        self.permute_input = False
        self.central_value = False
        self.is_discrete = False
        self.is_multi_discrete = False
        self.is_continuous = True
        self.fixed_sigma = True
        self.actor_cnn = torch.nn.Identity()
        self.actor_mlp = torch.nn.Identity()
        self.rnn = _FakeRNN()
        self.mu = torch.nn.Identity()
        self.mu_act = torch.nn.Identity()
        self.sigma = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
        self.sigma_act = torch.nn.Identity()
        self.value = torch.nn.Linear(2, 1, bias=False)
        self.value_act = torch.nn.Identity()


class TestReadinessScore:
    def test_score_clamps_to_unit_interval(self):
        assert readiness_score(0.0, 20.0) == pytest.approx(0.0)
        assert readiness_score(10.0, 20.0) == pytest.approx(0.5)
        assert readiness_score(30.0, 20.0) == pytest.approx(1.0)

    def test_zero_threshold_is_ready(self):
        assert readiness_score(0.0, 0.0) == pytest.approx(1.0)


class TestRecurrentGateState:
    def test_both_signals_required_before_activation(self):
        gate = RecurrentGateState(success_ema_tau=0.0, success_threshold=0.2, traj_threshold=4.0, ramp_epochs=10)
        alpha, traj_score, success_score, success_ema = gate.update(epoch=5, traj_size=4.0, success_rate=0.1)
        assert alpha == pytest.approx(0.0)
        assert traj_score == pytest.approx(1.0)
        assert success_score == pytest.approx(0.5)
        assert success_ema == pytest.approx(0.1)

    def test_gate_ramps_after_both_signals_are_ready(self):
        gate = RecurrentGateState(success_ema_tau=0.0, success_threshold=0.2, traj_threshold=4.0, ramp_epochs=4)
        alpha1, _, _, _ = gate.update(epoch=7, traj_size=4.0, success_rate=0.2)
        alpha2, _, _, _ = gate.update(epoch=8, traj_size=6.0, success_rate=0.3)
        alpha4, _, _, _ = gate.update(epoch=10, traj_size=8.0, success_rate=0.4)
        assert alpha1 == pytest.approx(0.25)
        assert alpha2 == pytest.approx(0.5)
        assert alpha4 == pytest.approx(1.0)

    def test_success_ema_is_smoothed(self):
        gate = RecurrentGateState(success_ema_tau=0.5, success_threshold=0.2, traj_threshold=1.0, ramp_epochs=1)
        _, _, _, ema1 = gate.update(epoch=1, traj_size=1.0, success_rate=0.4)
        _, _, _, ema2 = gate.update(epoch=2, traj_size=1.0, success_rate=0.0)
        assert ema1 == pytest.approx(0.2)
        assert ema2 == pytest.approx(0.1)


class TestResolveSuccessRate:
    def test_empty_window_is_zero(self):
        assert resolve_success_rate([]) == pytest.approx(0.0)

    def test_window_mean_is_computed(self):
        assert resolve_success_rate([1, 0, 1, 1]) == pytest.approx(0.75)


class TestInstallRecurrentGate:
    def test_alpha_zero_uses_skip_projection(self):
        network = _FakeNetwork()
        install_recurrent_gate(network, obs_dim=2)
        with torch.no_grad():
            network.recurrent_gate_skip.weight.copy_(torch.eye(2))
            network.recurrent_gate_skip.bias.zero_()
            network.recurrent_gate_alpha = 0.0
        mu, sigma, value, states = network(
            {
                "obs": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                "seq_length": 1,
                "rnn_states": (torch.zeros(1, 1, 2), torch.zeros(1, 1, 2)),
            }
        )
        assert mu.tolist() == pytest.approx([[1.0, 2.0]])
        assert sigma.tolist() == pytest.approx([1.0, 1.0])
        assert value.shape == (1, 1)
        assert len(states) == 2

    def test_alpha_one_uses_recurrent_features(self):
        network = _FakeNetwork()
        install_recurrent_gate(network, obs_dim=2)
        with torch.no_grad():
            network.recurrent_gate_skip.weight.zero_()
            network.recurrent_gate_skip.bias.zero_()
            network.recurrent_gate_alpha = 1.0
        mu, _, _, _ = network(
            {
                "obs": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                "seq_length": 1,
                "rnn_states": (torch.zeros(1, 1, 2), torch.zeros(1, 1, 2)),
            }
        )
        assert mu.tolist() == pytest.approx([[11.0, 12.0]])

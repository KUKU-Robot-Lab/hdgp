from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Sequence

import torch
import torch.nn as nn


def readiness_score(current: float, threshold: float) -> float:
    """Normalize a readiness signal into [0, 1]."""
    if threshold <= 0.0:
        return 1.0
    return max(0.0, min(float(current) / float(threshold), 1.0))


@dataclass
class RecurrentGateState:
    success_ema_tau: float = 0.95
    success_threshold: float = 0.02
    traj_threshold: float = 20.0
    ramp_epochs: int = 500
    success_ema: float = 0.0
    activation_epoch: int | None = None

    def update(self, *, epoch: int, traj_size: float, success_rate: float) -> tuple[float, float, float, float]:
        """Return alpha, traj_score, success_score, success_ema."""
        tau = max(0.0, min(self.success_ema_tau, 1.0))
        self.success_ema = tau * self.success_ema + (1.0 - tau) * float(success_rate)

        traj_score = readiness_score(traj_size, self.traj_threshold)
        success_score = readiness_score(self.success_ema, self.success_threshold)

        if traj_score >= 1.0 and success_score >= 1.0:
            if self.activation_epoch is None:
                self.activation_epoch = int(epoch)
            elapsed = max(int(epoch) - self.activation_epoch + 1, 0)
            if self.ramp_epochs <= 0:
                alpha = 1.0
            else:
                alpha = max(0.0, min(elapsed / float(self.ramp_epochs), 1.0))
        else:
            alpha = 0.0

        return alpha, traj_score, success_score, self.success_ema


def install_recurrent_gate(network: nn.Module, obs_dim: int) -> bool:
    """Attach skip projections + gated recurrent forward to an rl_games A2C network.

    Supports the non-separate A2CBuilder network used by the v4 pouring task.
    """
    if getattr(network, "_recurrent_gate_installed", False):
        return True
    if not getattr(network, "has_rnn", False):
        return False
    if getattr(network, "separate", False):
        return False
    if not getattr(network, "is_rnn_before_mlp", False):
        return False

    network.recurrent_gate_skip = nn.Linear(int(obs_dim), int(network.rnn_units))
    network.recurrent_gate_alpha = 0.0
    network._recurrent_gate_installed = True

    init_factory = getattr(network, "init_factory", None)
    initializer = getattr(network, "initializer", None)
    if init_factory is not None and initializer is not None:
        gate_init = init_factory.create(**initializer)
        gate_init(network.recurrent_gate_skip.weight)
        if network.recurrent_gate_skip.bias is not None:
            torch.nn.init.zeros_(network.recurrent_gate_skip.bias)

    def _forward_with_recurrent_gate(self, obs_dict):
        obs = obs_dict["obs"]
        states = obs_dict.get("rnn_states", None)
        dones = obs_dict.get("dones", None)
        bptt_len = obs_dict.get("bptt_len", 0)

        if self.has_cnn:
            if self.permute_input and len(obs.shape) == 4:
                obs = obs.permute((0, 3, 1, 2))

        out = self.actor_cnn(obs)
        out = out.flatten(1)

        if self.has_rnn:
            seq_length = obs_dict.get("seq_length", 1)
            cnn_out = out
            batch_size = out.size(0)
            num_seqs = batch_size // seq_length

            if len(states) == 1:
                states = states[0]

            out = out.reshape(num_seqs, seq_length, -1)
            out = out.transpose(0, 1)

            if dones is not None:
                dones = dones.reshape(num_seqs, seq_length, -1)
                dones = dones.transpose(0, 1)

            out, states = self.rnn(out, states, dones, bptt_len)
            out = out.transpose(0, 1)
            out = out.contiguous().reshape(out.size(0) * out.size(1), -1)

            if self.rnn_ln:
                out = self.layer_norm(out)

            alpha = float(max(0.0, min(getattr(self, "recurrent_gate_alpha", 1.0), 1.0)))
            skip = self.recurrent_gate_skip(cnn_out)
            out = alpha * out + (1.0 - alpha) * skip

            if self.rnn_concat_output:
                out = torch.cat([out, cnn_out], dim=1)
            if self.is_rnn_before_mlp:
                out = self.actor_mlp(out)
            if not isinstance(states, tuple):
                states = (states,)
        else:
            out = self.actor_mlp(out)

        value = self.value_act(self.value(out))

        if self.central_value:
            return value, states
        if self.is_discrete:
            logits = self.logits(out)
            return logits, value, states
        if self.is_multi_discrete:
            logits = [logit(out) for logit in self.logits]
            return logits, value, states
        if self.is_continuous:
            mu = self.mu_act(self.mu(out))
            if self.fixed_sigma:
                sigma = mu * 0.0 + self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma(out))
            return mu, sigma, value, states

        raise RuntimeError("Unsupported action space for recurrent gate forward patch.")

    network.forward = MethodType(_forward_with_recurrent_gate, network)
    return True


def resolve_success_rate(success_window: Sequence[int]) -> float:
    if not success_window:
        return 0.0
    return float(sum(int(v) for v in success_window)) / float(len(success_window))

# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PourLstmBCAgent: LSTM PPO + Behavioral Cloning auxiliary loss.

성공 에피소드 궤적(SuccessTrajectoryBuffer) 에서 (obs, action) 시퀀스를 샘플링하여
BC aux loss 를 PPO loss 에 더한다.

L_total = L_PPO + λ_BC(t) × L_BC
  L_BC  = -E_{(s,a)~D_success}[log π_θ(a|s)]  (NLL over padded-masked sequence)
  λ_BC(t): warmup → decay 스케줄

등록:
  rl_games Runner 의 algo_factory 에 'a2c_continuous_lstm_bc' 이름으로 등록.
  이 파일의 register_pour_lstm_bc_agent(runner) 를 train 스크립트에서 호출하거나,
  v4 __init__.py 의 Runner monkeypatch 를 통해 자동 등록된다.
"""

from __future__ import annotations

import math
import torch
from torch import Tensor

from rl_games.algos_torch.a2c_continuous import A2CAgent

from .recurrent_gate import RecurrentGateState, install_recurrent_gate, resolve_success_rate
from .real_demo_bc import DEFAULT_REAL_DEMO_PATHS, RealDemoBCBuffer, load_real_demo_episodes


# ---------------------------------------------------------------------------
# BC 가중치 스케줄
# ---------------------------------------------------------------------------

def _bc_weight(
    epoch: int,
    warmup_epochs: int,
    decay_epochs: int,
    weight_init: float,
    weight_final: float,
) -> float:
    """epoch 에 따른 BC loss 가중치 계산.

    0 → warmup_epochs  : 0 에서 weight_init 으로 선형 상승
    warmup → decay     : weight_init 에서 weight_final 로 선형 하강
    decay 이후         : weight_final 유지
    """
    if epoch <= 0:
        return 0.0
    if epoch <= warmup_epochs:
        return weight_init * (epoch / warmup_epochs)
    decay_end = warmup_epochs + decay_epochs
    if epoch <= decay_end:
        frac = (epoch - warmup_epochs) / decay_epochs
        return weight_init + (weight_final - weight_init) * frac
    return weight_final


# ---------------------------------------------------------------------------
# PourLstmBCAgent
# ---------------------------------------------------------------------------

class PourLstmBCAgent(A2CAgent):
    """LSTM PPO + BC auxiliary loss agent.

    rl_games A2CAgent 를 상속하여 calc_gradients() 를 override:
      1. 표준 PPO gradient 계산 (super 호출 없이 직접 구현, 단 super 의 로직을 재사용)
      2. success_trajectory_buffer 에서 BC minibatch 샘플
      3. NLL BC loss 계산 + λ_BC 가중치 적용
      4. L_total = L_PPO + L_BC 합산 후 single backward

    PourRightEnv.success_trajectory_buffer 접근:
      self.vec_env.env.unwrapped 또는 self.vec_env.env 체인을 통해 접근.
      환경이 준비되기 전(첫 epoch) 에는 BC 를 비활성화.
    """

    def __init__(self, base_name: str, params: dict) -> None:
        super().__init__(base_name, params)
        cfg = params.get("config", {})
        self._bc_warmup    = int(cfg.get("bc_loss_warmup_epochs", 500))
        self._bc_decay     = int(cfg.get("bc_loss_decay_epochs", 3000))
        self._bc_w_init    = float(cfg.get("bc_loss_weight_init", 1.0))
        self._bc_w_final   = float(cfg.get("bc_loss_weight_final", 0.1))
        self._bc_min_buf   = int(cfg.get("bc_min_buffer_size", 20))
        self._bc_seq_len   = int(cfg.get("bc_seq_len", 16))
        self._bc_batch     = int(cfg.get("bc_batch_size", 64))
        self._last_bc_loss = 0.0
        self._last_bc_success_sim_loss = 0.0
        self._last_bc_real_demo_loss = 0.0
        self._last_real_demo_weight = 0.0
        self._last_demo_pour_phase_ratio = 0.0
        self._last_demo_action_reconstruction_error = 0.0
        self._traj_buf     = None   # lazy resolve: 첫 calc_gradients 호출 시 탐색
        self._real_demo_buf = None
        self._real_demo_load_error = ""
        self._real_demo_enabled = bool(cfg.get("real_demo_bc_enable", False))
        self._real_demo_paths = tuple(cfg.get("real_demo_bc_paths", [str(p) for p in DEFAULT_REAL_DEMO_PATHS]))
        self._real_demo_stride = int(cfg.get("real_demo_stride", cfg.get("demo_stride", 2)))
        self._real_demo_pour_sample_ratio = float(cfg.get("real_demo_pour_sample_ratio", 0.6))
        self._real_demo_min_buf = int(cfg.get("real_demo_bc_min_buffer_size", 1))
        self._real_demo_warmup = int(cfg.get("real_demo_bc_warmup_epochs", 100))
        self._real_demo_decay = int(cfg.get("real_demo_bc_decay_epochs", 3000))
        self._real_demo_w_init = float(cfg.get("real_demo_bc_weight_init", 0.5))
        self._real_demo_w_final = float(cfg.get("real_demo_bc_weight_final", 0.05))
        self._real_demo_obs_mismatch_policy = str(
            cfg.get("real_demo_bc_obs_mismatch_policy", "pad_or_crop")
        )
        self._task_env     = None
        self._last_recurrent_alpha = 1.0
        self._last_recurrent_traj_score = 1.0
        self._last_recurrent_success_score = 1.0
        self._last_recurrent_success_ema = 0.0
        self._last_recurrent_epoch = -1
        self._recurrent_gate_enabled = bool(cfg.get("recurrent_gate_enable", True))
        self._recurrent_gate = RecurrentGateState(
            success_ema_tau=float(cfg.get("recurrent_gate_success_ema_tau", 0.95)),
            success_threshold=float(cfg.get("recurrent_gate_success_threshold", 0.02)),
            traj_threshold=float(cfg.get("recurrent_gate_traj_min_size", self._bc_min_buf)),
            ramp_epochs=int(cfg.get("recurrent_gate_ramp_epochs", 500)),
        )
        self._install_recurrent_gates()

    def _install_recurrent_gates(self) -> None:
        if not self._recurrent_gate_enabled:
            return
        actor_obs_shape = self.model.obs_shape
        actor_obs_dim = int(actor_obs_shape[0] if isinstance(actor_obs_shape, (tuple, list)) else actor_obs_shape)
        install_recurrent_gate(self.model.a2c_network, obs_dim=actor_obs_dim)

        if self.has_central_value:
            critic_obs_shape = self.central_value_net.model.obs_shape
            critic_obs_dim = int(
                critic_obs_shape[0] if isinstance(critic_obs_shape, (tuple, list)) else critic_obs_shape
            )
            install_recurrent_gate(self.central_value_net.model.a2c_network, obs_dim=critic_obs_dim)

        self._set_recurrent_alpha(0.0)

    # ------------------------------------------------------------------
    def _resolve_traj_buffer(self):
        """vec_env 체인을 탐색하여 success_trajectory_buffer 를 반환.

        못 찾으면 None 반환 (BC 비활성).
        """
        if self._traj_buf is not None:
            return self._traj_buf

        env = getattr(self, "vec_env", None)
        for _ in range(8):          # 최대 8단계 unwrap
            if env is None:
                break
            buf = getattr(env, "success_trajectory_buffer", None)
            if buf is not None:
                self._traj_buf = buf
                return buf
            env = getattr(env, "env", None) or getattr(env, "unwrapped", None)
        return None

    def _resolve_real_demo_buffer(self):
        """Lazily load external HDF5 real-demo BC episodes."""
        if not self._real_demo_enabled:
            return None
        if self._real_demo_buf is not None:
            return self._real_demo_buf
        if self._real_demo_load_error:
            return None

        try:
            episodes = load_real_demo_episodes(
                self._real_demo_paths,
                demo_stride=self._real_demo_stride,
                device=self.ppo_device,
            )
            self._real_demo_buf = RealDemoBCBuffer(
                episodes,
                device=self.ppo_device,
                pour_sample_ratio=self._real_demo_pour_sample_ratio,
            )
        except Exception as exc:  # noqa: BLE001 - keep PPO running if external demos are unavailable.
            self._real_demo_load_error = str(exc)
            self._real_demo_buf = None
        return self._real_demo_buf

    def _resolve_task_env(self):
        if self._task_env is not None:
            return self._task_env

        env = getattr(self, "vec_env", None)
        for _ in range(8):
            if env is None:
                break
            if hasattr(env, "_success_window"):
                self._task_env = env
                return env
            env = getattr(env, "env", None) or getattr(env, "unwrapped", None)
        return None

    def _set_recurrent_alpha(self, alpha: float) -> None:
        alpha = float(max(0.0, min(alpha, 1.0)))
        actor_network = getattr(self.model, "a2c_network", None)
        if actor_network is not None and hasattr(actor_network, "recurrent_gate_alpha"):
            actor_network.recurrent_gate_alpha = alpha

        if self.has_central_value:
            critic_network = getattr(self.central_value_net.model, "a2c_network", None)
            if critic_network is not None and hasattr(critic_network, "recurrent_gate_alpha"):
                critic_network.recurrent_gate_alpha = alpha

        self._last_recurrent_alpha = alpha

    def _update_recurrent_gate_if_needed(self) -> None:
        if not self._recurrent_gate_enabled:
            return
        if self._last_recurrent_epoch == int(self.epoch_num):
            return

        env = self._resolve_task_env()
        buf = self._resolve_traj_buffer()

        traj_size = float(len(buf)) if buf is not None else 0.0
        success_rate = 0.0
        if env is not None:
            success_rate = resolve_success_rate(getattr(env, "_success_window", ()))

        alpha, traj_score, success_score, success_ema = self._recurrent_gate.update(
            epoch=int(self.epoch_num),
            traj_size=traj_size,
            success_rate=success_rate,
        )
        self._last_recurrent_traj_score = traj_score
        self._last_recurrent_success_score = success_score
        self._last_recurrent_success_ema = success_ema
        self._last_recurrent_epoch = int(self.epoch_num)
        self._set_recurrent_alpha(alpha)

    # ------------------------------------------------------------------
    def _make_zero_rnn_states(self, batch_size: int, device):
        """BC forward 용 초기 LSTM/GRU 상태 (zeros) 생성.

        rl_games network_builder 는 rnn_states=None 을 허용하지 않으므로
        (len(None) → TypeError) 명시적으로 zeros 를 전달해야 한다.

        Returns:
            LSTM: [h, c] 각 (num_layers, batch_size, hidden_size)
            GRU : [h]    (num_layers, batch_size, hidden_size)
        """
        rnn = getattr(self.model.a2c_network, "rnn", None)
        if rnn is None:
            return None
        num_layers  = int(getattr(rnn, "num_layers", 1))
        hidden_size = int(getattr(rnn, "hidden_size", 256))
        rnn_name    = str(getattr(rnn, "rnn_name", "lstm")).lower()
        z = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        return [z.clone(), z.clone()] if rnn_name == "lstm" else [z.clone()]

    # ------------------------------------------------------------------
    def _compute_bc_loss(self, demo_batch: dict) -> Tensor:
        """BC NLL loss 계산.

        Args:
            demo_batch: SuccessTrajectoryBuffer.sample() 결과
              "obs"     : (B, T, obs_dim)
              "actions" : (B, T, act_dim)
              "mask"    : (B, T) bool

        Returns:
            scalar loss tensor
        """
        obs_seq  = demo_batch["obs"]      # (B, T, obs_dim)
        act_seq  = demo_batch["actions"]  # (B, T, act_dim)
        mask     = demo_batch["mask"]     # (B, T)

        obs_seq = self._adapt_bc_obs_to_model(obs_seq)

        B, T, _ = obs_seq.shape
        device   = obs_seq.device

        # obs 전처리 (normalize_input 적용)
        obs_flat = obs_seq.reshape(B * T, -1)
        obs_flat = self._preproc_obs(obs_flat)

        # rl_games LSTM model 은 (B*T, obs) 입력 + seq_length 파라미터로
        # 내부에서 reshape 처리함. is_rnn=True 일 때 rnn_states 를 명시 전달
        # (None 이면 network_builder 내부 len() 체크에서 TypeError 발생).
        if self.is_rnn:
            batch_dict = {
                "is_train": True,
                "obs": obs_flat,
                "prev_actions": act_seq.reshape(B * T, -1),
                "seq_length": T,
                "rnn_states": self._make_zero_rnn_states(B, device),
            }
        else:
            batch_dict = {
                "is_train": True,
                "obs": obs_flat,
                "prev_actions": act_seq.reshape(B * T, -1),
            }

        res = self.model(batch_dict)
        mu    = res["mus"]    # (B*T, act_dim)
        sigma = res["sigmas"] # (B*T, act_dim)

        # NLL: 0.5 × ((a - μ)/σ)² + log σ  (Gaussian NLL per dim)
        act_flat   = act_seq.reshape(B * T, -1)
        nll        = 0.5 * ((act_flat - mu) / (sigma + 1e-8)).pow(2) + (sigma + 1e-8).log()
        nll_per_step = nll.sum(dim=-1)                        # (B*T,)
        nll_per_step = nll_per_step.reshape(B, T)

        # 유효 스텝 마스킹
        mask_f = mask.float()
        valid_sum = mask_f.sum().clamp(min=1.0)
        loss = (nll_per_step * mask_f).sum() / valid_sum
        return loss

    def _adapt_bc_obs_to_model(self, obs_seq: Tensor) -> Tensor:
        """Adapt external 91D demo observations to the current policy input dim.

        The preferred training mode is a 91D actor.  During transition, this
        keeps old 110D configs runnable while making the mismatch explicit via
        config and TensorBoard metrics.
        """
        model_shape = self.model.obs_shape
        model_dim = int(model_shape[0] if isinstance(model_shape, (tuple, list)) else model_shape)
        obs_dim = int(obs_seq.shape[-1])
        if obs_dim == model_dim:
            return obs_seq
        if self._real_demo_obs_mismatch_policy == "error":
            raise RuntimeError(
                f"Real demo obs dim {obs_dim} does not match policy obs dim {model_dim}. "
                "Use a 91D actor or set real_demo_bc_obs_mismatch_policy=pad_or_crop."
            )
        if self._real_demo_obs_mismatch_policy != "pad_or_crop":
            raise RuntimeError(f"Unsupported real_demo_bc_obs_mismatch_policy={self._real_demo_obs_mismatch_policy!r}")
        if obs_dim > model_dim:
            return obs_seq[..., :model_dim]
        pad = torch.zeros(*obs_seq.shape[:-1], model_dim - obs_dim, device=obs_seq.device, dtype=obs_seq.dtype)
        return torch.cat([obs_seq, pad], dim=-1)

    # ------------------------------------------------------------------
    def calc_gradients(self, input_dict: dict) -> None:
        """PPO gradient + BC aux loss (override).

        PPO 부분은 super().calc_gradients() 를 호출해서 계산하되,
        BC loss 를 별도 forward 로 추가한 후 single backward 를 수행한다.

        주의: super() 의 calc_gradients 는 내부적으로 backward() 를 호출한다.
        이를 막기 위해 loss 계산까지만 super 를 호출하지 않고, 전체 로직을
        직접 재구현하여 BC loss 를 합산한다.
        """
        from rl_games.algos_torch import torch_ext
        from rl_games.common import common_losses

        self._update_recurrent_gate_if_needed()

        value_preds_batch          = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage                  = input_dict["advantages"]
        old_mu_batch               = input_dict["mu"]
        old_sigma_batch            = input_dict["sigma"]
        return_batch               = input_dict["returns"]
        actions_batch              = input_dict["actions"]
        obs_batch                  = input_dict["obs"]
        obs_batch                  = self._preproc_obs(obs_batch)

        curr_e_clip = self.e_clip
        rnn_masks   = None

        batch_dict = {
            "is_train": True,
            "prev_actions": actions_batch,
            "obs": obs_batch,
        }
        if self.is_rnn:
            rnn_masks = input_dict["rnn_masks"]
            batch_dict["rnn_states"]  = input_dict["rnn_states"]
            batch_dict["seq_length"]  = self.seq_length
            if self.zero_rnn_on_done:
                batch_dict["dones"] = input_dict["dones"]

        # ── PPO forward + BC forward (single AMP context) ────────────
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict           = self.model(batch_dict)
            action_log_probs   = res_dict["prev_neglogp"]
            values             = res_dict["values"]
            entropy            = res_dict["entropy"]
            mu                 = res_dict["mus"]
            sigma              = res_dict["sigmas"]

            a_loss = self.actor_loss_func(
                old_action_log_probs_batch, action_log_probs, advantage,
                self.ppo, curr_e_clip
            )
            if self.has_value_loss:
                c_loss = common_losses.critic_loss(
                    self.model, value_preds_batch, values,
                    curr_e_clip, return_batch, self.clip_value
                )
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)

            if self.bound_loss_type == "regularisation":
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == "bound":
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)

            losses, _ = torch_ext.apply_masks(
                [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)],
                rnn_masks,
            )
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = (
                a_loss
                + 0.5 * c_loss * self.critic_coef
                - entropy * self.entropy_coef
                + b_loss * self.bounds_loss_coef
            )

            # ── aux loss from model (standard hook) ──────────────────
            aux_loss = self.model.get_aux_loss()
            self.aux_loss_dict = {}
            if aux_loss is not None:
                for k, v in aux_loss.items():
                    loss += v
                    self.aux_loss_dict[k] = [v.detach()]

            # ── BC auxiliary loss ─────────────────────────────────────
            bc_loss_val = 0.0
            bc_success_sim_loss_val = 0.0
            bc_real_demo_loss_val = 0.0
            buf = self._resolve_traj_buffer()
            lam = _bc_weight(
                self.epoch_num,
                self._bc_warmup, self._bc_decay,
                self._bc_w_init, self._bc_w_final,
            )
            if lam > 0.0 and buf is not None and buf.is_warm(self._bc_min_buf):
                demo = buf.sample(self._bc_batch, self._bc_seq_len)
                if demo is not None:
                    bc_loss = self._compute_bc_loss(demo)
                    loss    = loss + lam * bc_loss
                    bc_loss_val = float(bc_loss.detach().item())
                    bc_success_sim_loss_val = bc_loss_val

            real_lam = _bc_weight(
                self.epoch_num,
                self._real_demo_warmup, self._real_demo_decay,
                self._real_demo_w_init, self._real_demo_w_final,
            )
            real_buf = self._resolve_real_demo_buffer()
            if real_lam > 0.0 and real_buf is not None and real_buf.is_warm(self._real_demo_min_buf):
                real_demo = real_buf.sample(self._bc_batch, self._bc_seq_len)
                if real_demo is not None:
                    real_bc_loss = self._compute_bc_loss(real_demo)
                    loss = loss + real_lam * real_bc_loss
                    bc_real_demo_loss_val = float(real_bc_loss.detach().item())
                    bc_loss_val += bc_real_demo_loss_val
                    self._last_demo_pour_phase_ratio = float(real_buf.last_pour_phase_ratio)
                    # Reconstruction diagnostic: converted real-demo actions are already
                    # normalized to v4's 11D action contract; clipping error catches bad HDF5 data.
                    recon_err = torch.relu(real_demo["actions"].abs() - 1.0)
                    self._last_demo_action_reconstruction_error = float(recon_err.mean().detach().item())

        self._last_bc_loss = bc_loss_val
        self._last_bc_success_sim_loss = bc_success_sim_loss_val
        self._last_bc_real_demo_loss = bc_real_demo_loss_val
        self._last_real_demo_weight = real_lam

        # ── backward ─────────────────────────────────────────────────
        if self.multi_gpu:
            self.optimizer.zero_grad()
        else:
            for param in self.model.parameters():
                param.grad = None

        self.scaler.scale(loss).backward()
        self.trancate_gradients_and_step()

        # ── KL (no_grad) ──────────────────────────────────────────────
        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist   = torch_ext.policy_kl(
                mu.detach(), sigma.detach(),
                old_mu_batch, old_sigma_batch, reduce_kl,
            )
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()

        self.diagnostics.mini_batch(
            self,
            {
                "values":      value_preds_batch,
                "returns":     return_batch,
                "new_neglogp": action_log_probs,
                "old_neglogp": old_action_log_probs_batch,
                "masks":       rnn_masks,
            },
            curr_e_clip, 0,
        )

        self.train_result = (
            a_loss, c_loss, entropy,
            kl_dist, self.last_lr, 1.0,
            mu.detach(), sigma.detach(), b_loss,
        )

    # ------------------------------------------------------------------
    def write_stats(self, *args, **kwargs):
        """BC loss 를 TensorBoard 에 추가 로깅.

        부모 시그니처:
          write_stats(total_time, epoch_num, step_time, play_time, update_time,
                      a_losses, c_losses, entropies, kls, last_lr, lr_mul,
                      frame, scaled_time, scaled_play_time, curr_frames)
        인수 수가 버전마다 바뀔 수 있으므로 *args/**kwargs 로 전달한다.
        """
        super().write_stats(*args, **kwargs)
        # args 인덱스 (부모 시그니처 순서)
        epoch_num = args[1] if len(args) > 1 else kwargs.get("epoch_num", 0)
        frame     = args[11] if len(args) > 11 else kwargs.get("frame", 0)
        if hasattr(self, "writer") and self.writer is not None:
            self.writer.add_scalar("bc/loss", self._last_bc_loss, frame)
            self.writer.add_scalar("bc/loss_success_sim", self._last_bc_success_sim_loss, frame)
            self.writer.add_scalar("bc/loss_real_demo", self._last_bc_real_demo_loss, frame)
            self.writer.add_scalar(
                "bc/weight",
                _bc_weight(epoch_num, self._bc_warmup, self._bc_decay,
                           self._bc_w_init, self._bc_w_final),
                frame,
            )
            self.writer.add_scalar("bc/weight_real_demo", self._last_real_demo_weight, frame)
            self.writer.add_scalar("demo/pour_phase_ratio", self._last_demo_pour_phase_ratio, frame)
            self.writer.add_scalar(
                "demo/action_reconstruction_error",
                self._last_demo_action_reconstruction_error,
                frame,
            )
            real_buf = self._resolve_real_demo_buffer()
            if real_buf is not None:
                self.writer.add_scalar("demo/real_buffer_size", len(real_buf), frame)
            buf = self._resolve_traj_buffer()
            if buf is not None:
                self.writer.add_scalar("bc/buffer_size", len(buf), frame)
            self.writer.add_scalar("recurrent/alpha", self._last_recurrent_alpha, frame)
            self.writer.add_scalar("recurrent/traj_score", self._last_recurrent_traj_score, frame)
            self.writer.add_scalar("recurrent/success_score", self._last_recurrent_success_score, frame)
            self.writer.add_scalar("recurrent/success_ema", self._last_recurrent_success_ema, frame)


# ---------------------------------------------------------------------------
# 등록 헬퍼
# ---------------------------------------------------------------------------

def register_pour_lstm_bc_agent(runner) -> None:
    """runner.algo_factory 에 PourLstmBCAgent 를 등록.

    Usage (train 스크립트에서):
        from openarm.tasks...pour_lstm_bc_agent import register_pour_lstm_bc_agent
        runner = Runner(...)
        register_pour_lstm_bc_agent(runner)
        runner.load(agent_cfg)
    """
    runner.algo_factory.register_builder(
        "a2c_continuous_lstm_bc",
        lambda **kwargs: PourLstmBCAgent(**kwargs),
    )

# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PourLstmBCAgent: LSTM PPO + Behavioral Cloning auxiliary loss (v5-native).

L_total = L_PPO + λ_sim(t) × L_BC_sim + λ_demo(t) × L_BC_demo

λ_sim(t):  warmup→plateau→decay 스케줄 — sim 성공 궤적 BC
λ_demo(t): warmup→decay 스케줄 — rollout-conditioned 실제 데모 teacher BC

등록:
  v5 __init__.py 가 Runner monkeypatch 로 자동 등록:
    algo_factory  → 'a2c_continuous_lstm_bc' (학습용)
    player_factory→ 'a2c_continuous_lstm_bc' (추론용)
"""

from __future__ import annotations

import torch
from torch import Tensor

from rl_games.algos_torch.a2c_continuous import A2CAgent

from .demo_bc_buffer import DEFAULT_DEMO_PATHS, DEFAULT_WARM_STATE_PATHS, DemoBCBuffer


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
    """epoch 에 따른 BC loss 가중치.

    [0, warmup]       : 0 → weight_init 선형 상승
    [warmup, warmup+decay]: weight_init → weight_final 선형 하강
    이후               : weight_final 유지
    """
    if epoch <= 0:
        return 0.0
    if epoch <= warmup_epochs:
        return weight_init * (epoch / max(warmup_epochs, 1))
    decay_end = warmup_epochs + decay_epochs
    if epoch <= decay_end:
        frac = (epoch - warmup_epochs) / max(decay_epochs, 1)
        return weight_init + (weight_final - weight_init) * frac
    return weight_final


# ---------------------------------------------------------------------------
# PourLstmBCAgent
# ---------------------------------------------------------------------------

class PourLstmBCAgent(A2CAgent):
    """LSTM PPO + BC auxiliary loss.

    calc_gradients() 를 override:
      1. PPO loss (a_loss + c_loss + entropy + b_loss)
      2. SuccessTrajBuffer 에서 sim BC loss (λ_sim × NLL)
      3. rollout env의 demo_id/time/base에 맞춘 real demo BC loss
      4. 합산 후 single backward()

    SuccessTrajBuffer 탐색:
      vec_env 체인을 순회하며 success_trajectory_buffer 속성을 찾는다.
      PourRightEnv.__init__ 에서 cfg.enable_trajectory_capture=True 이면 할당됨.

    DemoBCBuffer 초기화:
      real_demo_offline_bc_enable=True 일 때만 legacy offline BC fallback 으로 lazy load.
    """

    def __init__(self, base_name: str, params: dict) -> None:
        super().__init__(base_name, params)
        cfg = params.get("config", {})

        # sim BC 하이퍼파라미터
        self._bc_warmup   = int(cfg.get("bc_loss_warmup_epochs",   500))
        self._bc_decay    = int(cfg.get("bc_loss_decay_epochs",   3000))
        self._bc_w_init   = float(cfg.get("bc_loss_weight_init",    1.0))
        self._bc_w_final  = float(cfg.get("bc_loss_weight_final",   0.1))
        self._bc_min_buf  = int(cfg.get("bc_min_buffer_size",        20))
        self._bc_seq_len  = int(cfg.get("bc_seq_len",                16))
        self._bc_batch    = int(cfg.get("bc_batch_size",             64))

        # real demo BC 하이퍼파라미터
        self._demo_enabled     = bool(cfg.get("real_demo_bc_enable",               True))
        self._demo_offline_enabled = bool(cfg.get("real_demo_offline_bc_enable",  False))
        self._demo_warmup      = int(cfg.get("real_demo_bc_warmup_epochs",            10))
        self._demo_decay       = int(cfg.get("real_demo_bc_decay_epochs",          3000))
        self._demo_w_init      = float(cfg.get("real_demo_bc_weight_init",          30.0))
        self._demo_w_final     = float(cfg.get("real_demo_bc_weight_final",        1.0))
        self._demo_min_buf     = int(cfg.get("real_demo_bc_min_buffer_size",          1))
        self._demo_stride      = int(cfg.get("real_demo_stride",                      2))
        self._demo_pour_ratio  = float(cfg.get("real_demo_pour_sample_ratio",       0.0))
        self._demo_time_bin_weights = list(
            cfg.get("real_demo_time_bin_weights", [0.20, 0.20, 0.25, 0.35])
        )
        self._demo_start_mode  = str(cfg.get("real_demo_bc_start_mode", "warm_state_match"))
        self._demo_start_fraction = float(cfg.get("real_demo_bc_start_fraction", 0.0))
        self._demo_warm_state_paths = list(
            cfg.get("real_demo_bc_warm_state_paths", [str(p) for p in DEFAULT_WARM_STATE_PATHS])
        )
        self._demo_paths       = list(
            cfg.get("real_demo_bc_paths", [str(p) for p in DEFAULT_DEMO_PATHS])
        )
        self._demo_teacher_palm_weight = float(cfg.get("real_demo_teacher_palm_weight", 1.0))
        self._demo_teacher_finger_weight = float(cfg.get("real_demo_teacher_finger_weight", 0.0))

        # 상태
        self._traj_buf            = None
        self._rollout_env         = None
        self._demo_buf            = None
        self._demo_load_error     = ""
        self._last_bc_sim_loss    = 0.0
        self._last_bc_demo_loss   = 0.0
        self._last_demo_pour_ratio = 0.0
        self._last_demo_bin_ratios = [0.0 for _ in self._demo_time_bin_weights]
        self._last_demo_teacher_mask_ratio = 0.0

    # ------------------------------------------------------------------
    # 버퍼 lazy resolve
    # ------------------------------------------------------------------

    def _resolve_traj_buffer(self):
        """vec_env 체인을 순회하여 success_trajectory_buffer 를 반환."""
        if self._traj_buf is not None:
            return self._traj_buf
        env = getattr(self, "vec_env", None)
        for _ in range(8):
            if env is None:
                break
            buf = getattr(env, "success_trajectory_buffer", None)
            if buf is not None:
                self._traj_buf = buf
                return buf
            env = getattr(env, "env", None) or getattr(env, "unwrapped", None)
        return None

    def _resolve_demo_buffer(self):
        """Legacy offline BC가 명시적으로 켜졌을 때 HDF5 데모 버퍼를 반환."""
        if not self._demo_enabled or not self._demo_offline_enabled:
            return None
        if self._demo_buf is not None:
            return self._demo_buf
        if self._demo_load_error:
            return None
        try:
            self._demo_buf = DemoBCBuffer(
                paths=self._demo_paths,
                stride=self._demo_stride,
                device=self.ppo_device,
                pour_ratio=self._demo_pour_ratio,
                time_bin_weights=self._demo_time_bin_weights,
                start_mode=self._demo_start_mode,
                start_fraction=self._demo_start_fraction,
                warm_state_paths=self._demo_warm_state_paths,
            )
        except Exception as exc:  # noqa: BLE001 — demo 로드 실패 시 BC만 비활성
            self._demo_load_error = str(exc)
            print(
                f"[PourLstmBCAgent] WARNING: real demo buffer load failed — {exc}",
                flush=True,
            )
            self._demo_buf = None
        return self._demo_buf

    def _resolve_rollout_env(self):
        """vec_env 체인에서 task env의 rollout teacher API를 찾는다."""
        if self._rollout_env is not None:
            return self._rollout_env

        env = getattr(self, "vec_env", None)
        seen: set[int] = set()
        for _ in range(12):
            if env is None or id(env) in seen:
                break
            seen.add(id(env))
            if hasattr(env, "get_demo_teacher_actions"):
                self._rollout_env = env
                return env

            next_env = None
            for attr in ("unwrapped", "env"):
                candidate = getattr(env, attr, None)
                if candidate is not None and id(candidate) not in seen:
                    next_env = candidate
                    break
            env = next_env
        return None

    # ------------------------------------------------------------------
    # rollout-conditioned demo teacher storage
    # ------------------------------------------------------------------

    def init_tensors(self) -> None:
        super().init_tensors()
        action_buf = self.experience_buffer.tensor_dict["actions"]
        self.experience_buffer.tensor_dict["demo_teacher_actions"] = torch.zeros_like(action_buf)
        self.experience_buffer.tensor_dict["demo_teacher_masks"] = torch.zeros(
            action_buf.shape[:-1] + (1,),
            dtype=action_buf.dtype,
            device=action_buf.device,
        )
        for name in ("demo_teacher_actions", "demo_teacher_masks"):
            if name not in self.update_list:
                self.update_list.append(name)
            if name not in self.tensor_list:
                self.tensor_list.append(name)

    def _empty_demo_teacher_like(self, actions: Tensor) -> tuple[Tensor, Tensor]:
        return (
            torch.zeros_like(actions),
            torch.zeros(actions.shape[:-1] + (1,), dtype=actions.dtype, device=actions.device),
        )

    def _get_rollout_demo_teacher_like(self, actions: Tensor) -> tuple[Tensor, Tensor]:
        if not self._demo_enabled:
            return self._empty_demo_teacher_like(actions)
        env = self._resolve_rollout_env()
        if env is None:
            return self._empty_demo_teacher_like(actions)

        try:
            teacher_actions, teacher_masks = env.get_demo_teacher_actions()
        except Exception as exc:  # noqa: BLE001
            if not self._demo_load_error:
                self._demo_load_error = f"rollout demo teacher failed: {exc}"
                print(f"[PourLstmBCAgent] WARNING: {self._demo_load_error}", flush=True)
            return self._empty_demo_teacher_like(actions)

        teacher_actions = teacher_actions.to(device=actions.device, dtype=actions.dtype)
        teacher_masks = teacher_masks.to(device=actions.device, dtype=actions.dtype)
        if teacher_masks.ndim == 1:
            teacher_masks = teacher_masks.unsqueeze(-1)
        return teacher_actions, teacher_masks

    def get_action_values(self, obs):
        res_dict = super().get_action_values(obs)
        teacher_actions, teacher_masks = self._get_rollout_demo_teacher_like(res_dict["actions"])
        res_dict["demo_teacher_actions"] = teacher_actions
        res_dict["demo_teacher_masks"] = teacher_masks
        return res_dict

    def prepare_dataset(self, batch_dict):
        teacher_actions = batch_dict.get("demo_teacher_actions", None)
        teacher_masks = batch_dict.get("demo_teacher_masks", None)
        super().prepare_dataset(batch_dict)
        if teacher_actions is not None and teacher_masks is not None:
            self.dataset.values_dict["demo_teacher_actions"] = teacher_actions
            self.dataset.values_dict["demo_teacher_masks"] = teacher_masks

    # ------------------------------------------------------------------
    # LSTM 초기 상태 생성
    # ------------------------------------------------------------------

    def _make_zero_rnn_states(self, batch_size: int, device) -> tuple[Tensor, ...] | None:
        """BC forward 용 zero LSTM/GRU 상태 생성.

        rl_games network_builder 는 rnn_states=None 허용 안 함.
        """
        rnn = getattr(self.model.a2c_network, "rnn", None)
        if rnn is None:
            return None
        # rl_games wraps torch RNN modules inside RnnWithDones as `rnn.rnn`.
        core_rnn = getattr(rnn, "rnn", rnn)
        num_layers = int(
            getattr(core_rnn, "num_layers", getattr(self.model.a2c_network, "rnn_layers", 1))
        )
        hidden_size = int(
            getattr(core_rnn, "hidden_size", getattr(self.model.a2c_network, "rnn_units", 256))
        )
        rnn_name = str(getattr(self.model.a2c_network, "rnn_name", "lstm")).lower()
        z = torch.zeros(num_layers, batch_size, hidden_size, device=device)
        return (z.clone(), z.clone()) if rnn_name == "lstm" else (z.clone(),)

    # ------------------------------------------------------------------
    # BC NLL loss
    # ------------------------------------------------------------------

    def _compute_bc_loss(self, demo_batch: dict) -> Tensor:
        """Gaussian NLL BC loss on a padded sequence batch.

        Args:
            demo_batch: {"obs": (B,T,obs), "actions": (B,T,act), "mask": (B,T) bool}

        Returns:
            scalar loss
        """
        obs_seq = demo_batch["obs"]       # (B, T, obs_dim)
        act_seq = demo_batch["actions"]   # (B, T, act_dim)
        mask    = demo_batch["mask"]      # (B, T) bool

        B, T, _ = obs_seq.shape
        device  = obs_seq.device

        obs_flat = self._preproc_obs(obs_seq.reshape(B * T, -1))

        if self.is_rnn:
            batch_dict = {
                "is_train":     True,
                "obs":          obs_flat,
                "prev_actions": act_seq.reshape(B * T, -1),
                "seq_length":   T,
                "rnn_states":   self._make_zero_rnn_states(B, device),
            }
        else:
            batch_dict = {
                "is_train":     True,
                "obs":          obs_flat,
                "prev_actions": act_seq.reshape(B * T, -1),
            }

        res   = self.model(batch_dict)
        mu    = res["mus"]    # (B*T, act_dim)

        act_flat  = act_seq.reshape(B * T, -1)
        # MSE loss: sigma collapse 없이 안정적 (NLL은 sigma→0 시 -log(sigma) 발산)
        nll_steps = 0.5 * (act_flat - mu).pow(2).sum(dim=-1).reshape(B, T)  # (B, T)

        mask_f    = mask.float()
        valid     = mask_f.sum().clamp(min=1.0)
        return (nll_steps * mask_f).sum() / valid

    def _compute_rollout_demo_teacher_loss(
        self,
        mu: Tensor,
        teacher_actions: Tensor,
        teacher_masks: Tensor,
        rnn_masks: Tensor | None,
    ) -> tuple[Tensor | None, float]:
        """MSE BC loss for teacher actions sampled from the same rollout state."""
        mask = teacher_masks.reshape(-1).float()
        if rnn_masks is not None:
            mask = mask * rnn_masks.reshape(-1).float()
        valid = mask.sum()
        mask_ratio = float((mask > 0.0).float().mean().detach().item())
        if float(valid.detach().item()) <= 0.0:
            return None, mask_ratio

        diff = mu - teacher_actions
        palm_loss = diff[:, :6].pow(2).sum(dim=-1) * self._demo_teacher_palm_weight
        finger_loss = diff[:, 6:].pow(2).sum(dim=-1) * self._demo_teacher_finger_weight
        step_loss = 0.5 * (palm_loss + finger_loss)
        return (step_loss * mask).sum() / valid.clamp(min=1.0), mask_ratio

    # ------------------------------------------------------------------
    # calc_gradients override (PPO + BC)
    # ------------------------------------------------------------------

    def calc_gradients(self, input_dict: dict) -> None:
        """PPO gradient + BC aux loss, single backward."""
        from rl_games.algos_torch import torch_ext
        from rl_games.common import common_losses

        value_preds_batch          = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage                  = input_dict["advantages"]
        old_mu_batch               = input_dict["mu"]
        old_sigma_batch            = input_dict["sigma"]
        return_batch               = input_dict["returns"]
        actions_batch              = input_dict["actions"]
        obs_batch                  = self._preproc_obs(input_dict["obs"])

        curr_e_clip = self.e_clip
        rnn_masks   = None

        batch_dict = {"is_train": True, "prev_actions": actions_batch, "obs": obs_batch}
        if self.is_rnn:
            rnn_masks = input_dict["rnn_masks"]
            batch_dict["rnn_states"] = input_dict["rnn_states"]
            batch_dict["seq_length"] = self.seq_length
            if self.zero_rnn_on_done:
                batch_dict["dones"] = input_dict["dones"]

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict         = self.model(batch_dict)
            action_log_probs = res_dict["prev_neglogp"]
            values           = res_dict["values"]
            entropy          = res_dict["entropy"]
            mu               = res_dict["mus"]
            sigma            = res_dict["sigmas"]

            a_loss = self.actor_loss_func(
                old_action_log_probs_batch, action_log_probs, advantage,
                self.ppo, curr_e_clip,
            )
            c_loss = (
                common_losses.critic_loss(
                    self.model, value_preds_batch, values,
                    curr_e_clip, return_batch, self.clip_value,
                )
                if self.has_value_loss
                else torch.zeros(1, device=self.ppo_device)
            )

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
            a_loss, c_loss, entropy, b_loss = losses

            loss = (
                a_loss
                + 0.5 * c_loss * self.critic_coef
                - entropy * self.entropy_coef
                + b_loss * self.bounds_loss_coef
            )

            # model-level aux loss hook
            aux = self.model.get_aux_loss()
            self.aux_loss_dict = {}
            if aux is not None:
                for k, v in aux.items():
                    loss += v
                    self.aux_loss_dict[k] = [v.detach()]

            # ── sim 성공 궤적 BC ──────────────────────────────────────
            sim_lam        = _bc_weight(self.epoch_num, self._bc_warmup, self._bc_decay, self._bc_w_init, self._bc_w_final)
            bc_sim_loss_v  = 0.0
            traj_buf       = self._resolve_traj_buffer()
            if sim_lam > 0.0 and traj_buf is not None and traj_buf.is_warm(self._bc_min_buf):
                sim_batch = traj_buf.sample(self._bc_batch, self._bc_seq_len)
                if sim_batch is not None:
                    sim_loss   = self._compute_bc_loss(sim_batch)
                    loss       = loss + sim_lam * sim_loss
                    bc_sim_loss_v = float(sim_loss.detach().item())

            # ── rollout-conditioned 실제 데모 teacher BC ───────────────
            demo_lam       = _bc_weight(self.epoch_num, self._demo_warmup, self._demo_decay, self._demo_w_init, self._demo_w_final)
            bc_demo_loss_v = 0.0
            self._last_demo_teacher_mask_ratio = 0.0
            if demo_lam > 0.0 and self._demo_enabled:
                teacher_actions = input_dict.get("demo_teacher_actions", None)
                teacher_masks = input_dict.get("demo_teacher_masks", None)
                if teacher_actions is not None and teacher_masks is not None:
                    demo_loss, mask_ratio = self._compute_rollout_demo_teacher_loss(
                        mu,
                        teacher_actions,
                        teacher_masks,
                        rnn_masks,
                    )
                    self._last_demo_teacher_mask_ratio = mask_ratio
                    if demo_loss is not None:
                        loss = loss + demo_lam * demo_loss
                        bc_demo_loss_v = float(demo_loss.detach().item())

            # ── legacy offline HDF5 demo BC (explicit fallback only) ─────
            demo_buf = self._resolve_demo_buffer()
            if (
                demo_lam > 0.0
                and self._demo_offline_enabled
                and demo_buf is not None
                and demo_buf.is_warm(self._demo_min_buf)
            ):
                demo_batch = demo_buf.sample(self._bc_batch, self._bc_seq_len)
                demo_loss = self._compute_bc_loss(demo_batch)
                loss = loss + demo_lam * demo_loss
                bc_demo_loss_v = float(demo_loss.detach().item())
                self._last_demo_pour_ratio = float(getattr(demo_buf, "_last_pour_ratio", 0.0))
                self._last_demo_bin_ratios = list(getattr(demo_buf, "_last_bin_ratios", self._last_demo_bin_ratios))

        self._last_bc_sim_loss  = bc_sim_loss_v
        self._last_bc_demo_loss = bc_demo_loss_v

        # ── backward ──────────────────────────────────────────────────
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
    # write_stats override (TensorBoard BC 지표)
    # ------------------------------------------------------------------

    def write_stats(self, *args, **kwargs) -> None:
        super().write_stats(*args, **kwargs)
        epoch_num = args[1] if len(args) > 1 else kwargs.get("epoch_num", 0)
        frame     = args[11] if len(args) > 11 else kwargs.get("frame", 0)
        if not (hasattr(self, "writer") and self.writer is not None):
            return

        sim_w  = _bc_weight(epoch_num, self._bc_warmup, self._bc_decay, self._bc_w_init, self._bc_w_final)
        demo_w = _bc_weight(epoch_num, self._demo_warmup, self._demo_decay, self._demo_w_init, self._demo_w_final)

        self.writer.add_scalar("bc/loss_sim",          self._last_bc_sim_loss,    frame)
        self.writer.add_scalar("bc/loss_demo",         self._last_bc_demo_loss,   frame)
        self.writer.add_scalar("bc/weight_sim",        sim_w,                      frame)
        self.writer.add_scalar("bc/weight_demo",       demo_w,                     frame)
        self.writer.add_scalar("demo/teacher_mask_ratio", self._last_demo_teacher_mask_ratio, frame)
        self.writer.add_scalar("demo/offline_bc_enabled", float(self._demo_offline_enabled), frame)
        if self._demo_offline_enabled:
            self.writer.add_scalar("demo/pour_phase_ratio", self._last_demo_pour_ratio, frame)
            for i, ratio in enumerate(self._last_demo_bin_ratios):
                self.writer.add_scalar(f"demo/sample_bin_{i}_ratio", float(ratio), frame)

        traj_buf = self._resolve_traj_buffer()
        if traj_buf is not None:
            self.writer.add_scalar("bc/sim_buffer_size",  float(len(traj_buf)), frame)

        if self._demo_offline_enabled:
            demo_buf = self._resolve_demo_buffer()
            if demo_buf is not None:
                self.writer.add_scalar("bc/demo_buffer_size", float(len(demo_buf)), frame)


# ---------------------------------------------------------------------------
# 등록 헬퍼
# ---------------------------------------------------------------------------

def register_pour_lstm_bc_agent(runner) -> None:
    """runner.algo_factory 에 PourLstmBCAgent 등록 (train 스크립트에서 직접 호출 가능)."""
    from rl_games.algos_torch import players as _players

    runner.algo_factory.register_builder(
        "a2c_continuous_lstm_bc",
        lambda **kw: PourLstmBCAgent(**kw),
    )
    runner.player_factory.register_builder(
        "a2c_continuous_lstm_bc",
        lambda **kw: _players.PpoPlayerContinuous(**kw),
    )

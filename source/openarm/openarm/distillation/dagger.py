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

"""DAgger teacher→student distillation (DEXTRAH distillation.py 이식, mono 카메라 전용).

teacher = state-based LSTM (privileged obs, 물체 pose 포함)
student = vision (D435i mono 카메라 + proprio, 물체 pose 미관측)

student 인코더가 실제로 보는 건 **RGB** 다 — DEXTRAH 의 a2c_* 학생망은 전부
use_depth=False 로 RGB 를 입력받는다. depth 는 (stereo_recon 변종에서) aux 재구성
타깃으로 쓰일 뿐이다. 그래서 RGB 증강 + env 시각 DR(visual_dr.py)이 필수다.
이걸 빼면 student 가 고정된 단일 외형에만 맞는 정책이 된다.

원본 대비 제거: stereo 분기, wandb, matplotlib 시각화.
원본 유지: DAgger beta 혼합, student/teacher RNN hidden state 관리, DDP, aux loss,
          RGB/depth 증강 (img_aug_type 으로 분기).

env 계약 (cfg.distillation=True 일 때 obs dict):
    "policy"        student obs  (proprio + goal + action + fabric)
    "expert_policy" teacher obs  (+ 물체 privileged state)
    "img"           depth  (N,1,H,W)
    "rgb"           rgb    (N,3,H,W)
    "aux_info"      dict — aux head 지도 대상 (object_pos 등)
    "mask"          depth 무효 픽셀 마스크 (N,1,H,W)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torchvision.utils as vutils
import warp as wp
import yaml
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.model_builder import ModelBuilder
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from openarm.distillation.depth_augs import DepthAug
from openarm.distillation.rgb_augs import RgbAug

# DAgger 스케줄 (DEXTRAH 원본 상수)
DEFAULT_NUM_ITERS = 100_000
BACKBONE_FREEZE_ITERS = 5_000   # 초반엔 인코더 동결 (헤드만 학습)
CKPT_INTERVAL = 5_000
GRAD_CLIP_NORM = 1.0
LEARNING_RATE = 1e-4
GAMES_TO_TRACK = 100


def l2(model: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.norm(model - target, p=2, dim=-1)


def weighted_l2(
    model: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    return torch.sum((model - target) * (weights * (model - target)), dim=-1) ** 0.5


def adjust_state_dict_keys(
    checkpoint_state_dict: dict, model_state_dict: dict
) -> dict:
    """torch.compile 흔적(_orig_mod)을 넣거나 빼서 체크포인트 키를 모델에 맞춘다."""
    adjusted = {}
    for key, value in checkpoint_state_dict.items():
        if key in model_state_dict:
            adjusted[key] = value
            continue

        parts = key.split(".")
        parts.insert(2, "_orig_mod")
        key_with_orig_mod = ".".join(parts)
        key_without_orig_mod = key.replace("_orig_mod.", "")

        if key_with_orig_mod in model_state_dict:
            adjusted[key_with_orig_mod] = value
        elif key_without_orig_mod in model_state_dict:
            adjusted[key_without_orig_mod] = value
        else:
            print(f"[dagger] 매칭 실패한 체크포인트 키: {key}")
            adjusted[key] = value
    return adjusted


class Dagger:
    """teacher 행동을 student가 모방하도록 online 학습한다.

    config = {
        "student": {"cfg": <yaml path>, "ckpt": <path|None>, "obs_type": "policy",
                    "data_aug": bool},
        "teacher": {"cfg": <yaml path>, "ckpt": <path>, "obs_type": "expert_policy"},
        "play_policy": bool,
    }
    """

    def __init__(self, env, config: dict, summaries_dir: str, nn_dir: str):
        # torchrun 필수: 단일 GPU도 `torchrun --nproc_per_node=1` 로 기동
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)
        wp.set_device(f"cuda:{self.local_rank}")

        self.env = env
        # gym.make 가 OrderEnforcing 등 wrapper 를 씌우므로 unwrapped 로 베이스
        # DirectRLEnv 까지 벗긴다(num_envs·cfg·robot 등 직접 속성 접근용).
        self.ov_env = env.env.unwrapped
        self.num_envs = self.ov_env.num_envs
        self.num_actions = self.ov_env.num_actions
        self.device = self.local_rank
        self.config = config

        self.student_network_params = self._load_param_dict(config["student"]["cfg"])["params"]
        self.teacher_network_params = self._load_param_dict(config["teacher"]["cfg"])["params"]
        student_network = self._load_network(self.student_network_params)
        teacher_network = self._load_network(self.teacher_network_params)

        self.value_size = 1
        student_cfg = self.student_network_params["config"]
        self.horizon_length = student_cfg["horizon_length"]
        self.normalize_value = student_cfg["normalize_value"]
        self.normalize_input = student_cfg["normalize_input"]

        self.student_model = student_network.build({
            "actions_num": self.num_actions,
            "input_shape": (self.ov_env.num_observations,),
            "batch_size": self.num_envs,
            "num_seqs": self.num_envs,
            "value_size": self.value_size,
            "normalize_value": self.normalize_value,
            "normalize_input": self.normalize_input,
        }).to(self.device)
        # rank 0 의 초기 가중치로 전 rank 동기화 후 DDP 래핑
        for param in self.student_model.parameters():
            dist.broadcast(param.data, src=0)
        self.student_model_ddp = DDP(
            self.student_model,
            device_ids=[self.local_rank],
            find_unused_parameters=True,
        )

        self.teacher_model = teacher_network.build({
            "actions_num": self.num_actions,
            "input_shape": (self.ov_env.num_teacher_observations,),
            "num_seqs": self.num_envs,
            "value_size": self.value_size,
            "normalize_value": self.normalize_value,
            "normalize_input": self.normalize_input,
        }).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.student_model_ddp.parameters(), lr=LEARNING_RATE, eps=1e-8
        )
        self.num_iters = DEFAULT_NUM_ITERS

        if config["student"]["ckpt"] is not None:
            self._set_weights(config["student"]["ckpt"], "student")
        if config["teacher"]["ckpt"] is None:
            raise ValueError("teacher 체크포인트가 필요하다 (--teacher)")
        self._set_weights(config["teacher"]["ckpt"], "teacher")

        self.student_obs_type = config["student"]["obs_type"]
        self.teacher_obs_type = config["teacher"]["obs_type"]

        self.is_rnn = self.student_model.is_rnn()
        self.is_teacher_rnn = self.teacher_model.is_rnn()
        if self.is_rnn:
            # BPTT 윈도우 길이. 원본 DEXTRAH 은 1(매 스텝 flush + hidden detach)이라
            # LSTM 이 시간 credit assignment 를 못 배운다 — occlusion 전에 물체 위치를
            # hidden 에 저장했다 나중에 꺼내 쓰는 걸 학습할 gradient 경로가 없다.
            # config.seq_length>1 이면 hidden 을 N 스텝 유지 후 한 번에 backward →
            # LSTM 이 시간 메모리를 학습. 에피소드 리셋은 flush 하지 않고 done env
            # hidden 만 마스크 곱으로 0 처리(_on_done). 경계에서만 flush(distill()).
            self.seq_length = int(student_cfg.get("seq_length", 1))
        self.is_aux = bool(
            getattr(self.student_model.a2c_network, "is_aux", False)
        )
        self.finetune_backbone = False

        self.play_policy = config["play_policy"]
        # play 시 teacher(expert) action 으로 rollout(beta=1). teacher 상한 확인·비교 렌더용.
        self.play_teacher = bool(config.get("play_teacher", False))
        self.frame = 0
        self.epoch_num = 0
        self.game_rewards = torch_ext.AverageMeter(
            self.value_size, GAMES_TO_TRACK
        ).to(self.device)
        self.game_lengths = torch_ext.AverageMeter(1, GAMES_TO_TRACK).to(self.device)

        if self.rank == 0:
            self.writer = SummaryWriter(summaries_dir)
            self.nn_dir = nn_dir
            os.makedirs(self.nn_dir, exist_ok=True)

        wp.init()
        self.aux_coeff = self.ov_env.cfg.aux_coeff
        self.aux_loss_names: Any = []

        # 이미지 증강 — 어느 텐서를 증강할지는 student 인코더가 뭘 보느냐로 정해진다.
        #   img_aug_type="rgb"   : 인코더 입력이 RGB (a2c_* 전부 use_depth=False)
        #   img_aug_type="depth" : 구 a2c_with_aux_depth 처럼 depth 를 직접 입력받는 경우
        # 엉뚱한 쪽을 증강하면 조용히 헛돈다 — 인코더가 안 보는 텐서에 노이즈를 넣게 된다.
        self.img_aug_type = self.ov_env.cfg.img_aug_type
        # 인코더가 보는 modality(use_depth)와 증강 대상(img_aug_type)이 어긋나면 조용히 헛돈다
        # (안 보는 텐서에 노이즈를 넣거나, 정규화 채널수가 어긋난다) → 시작 시 즉시 실패시킨다.
        _net_use_depth = bool(getattr(self.student_model.a2c_network, "use_depth", False))
        if (self.img_aug_type == "depth") != _net_use_depth:
            raise ValueError(
                f"modality 불일치: env.img_aug_type={self.img_aug_type!r} 인데 "
                f"student network use_depth={_net_use_depth}. "
                "둘을 함께 맞춰라(depth ↔ True, rgb ↔ False)."
            )
        self.use_data_aug = config["student"]["data_aug"]

        self.depth_aug = DepthAug(f"cuda:{self.local_rank}")
        self.use_depth_aug = (
            self.ov_env.cfg.aug_depth and self.img_aug_type == "depth"
        )
        self.depth_aug_cfg = self.ov_env.cfg.depth_randomization_cfg_dict

        self.rgb_aug = None
        if self.use_data_aug and self.img_aug_type == "rgb":
            self.rgb_aug = RgbAug(
                device=self.device,
                all_env_inds=self.ov_env.robot._ALL_INDICES,
                use_stereo=False,
                background_cfg={
                    "dir": os.path.join(
                        self.ov_env.cfg.texture_root, "background_imgs", "voc_resized"
                    ),
                    "height": self.ov_env.cfg.img_height,
                    "width": self.ov_env.cfg.img_width,
                    "aug_prob": 0.5,
                },
                color_cfg={
                    "aug_prob": 1.0,
                    "saturation_range": [0.5, 1.5],
                    "contrast_range": [0.5, 1.5],
                    "brightness_range": [0.5, 1.5],
                    "hue_range": [-0.15, 0.15],
                },
                motion_blur_cfg={
                    "aug_prob": 0.1,
                    "kernel_sizes": [9, 11, 13, 15, 17],
                    "angle_range": [0, 2 * np.pi],
                    "direction_range": [-1, 1],
                },
            )

        self._init_tensors()

    def _init_tensors(self) -> None:
        # neglogp 계산이 요구하는 더미 (값 자체는 쓰지 않음)
        self.prev_actions_student = torch.zeros(
            (self.num_envs, self.num_actions), dtype=torch.float32, device=self.device
        )
        self.prev_actions_teacher = torch.zeros(
            (self.num_envs, self.num_actions), dtype=torch.float32, device=self.device
        )
        self.current_rewards = torch.zeros(
            (self.num_envs, self.value_size), dtype=torch.float32, device=self.device
        )
        self.current_lengths = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.dones = torch.ones(
            (self.num_envs,), dtype=torch.uint8, device=self.device
        )
        self.actions_teacher = torch.zeros(
            (self.num_envs, self.num_actions), dtype=torch.float32, device=self.device
        )

        if self.is_rnn:
            self.student_hidden_states = [
                s.to(self.device)
                for s in self.student_model.get_default_rnn_state()
            ]
            self.hidden_state_means = [
                RunningMeanStd((s.shape[0], s.shape[-1])).to(
                    device=self.device, dtype=s.dtype
                )
                for s in self.student_hidden_states
            ]

        if self.is_teacher_rnn:
            self.teacher_hidden_states = [
                s.to(self.device)
                for s in self.teacher_model.get_default_rnn_state()
            ]

    # ------------------------------------------------------------------
    # 학습 루프
    # ------------------------------------------------------------------
    def distill(self) -> None:
        self.student_model.train()
        self.teacher_model.eval()

        obs = self.env.reset()[0]
        log_counter = 0
        total_loss = 0.0
        self.optimizer.zero_grad()

        while log_counter < self.num_iters:
            # beta = teacher action으로 rollout할 확률. 원본은 0 고정 (student on-policy
            # rollout + teacher 지도 = DAgger). play 시엔 student만 스텝.
            beta = 0.0
            if self.play_policy:
                self.optimizer.param_groups[0]["lr"] = 0.0
                if self.play_teacher:
                    beta = 1.0   # 전 env teacher action 으로 스텝(상한 비교 렌더)
            self.finetune_backbone = log_counter >= BACKBONE_FREEZE_ITERS

            if self.use_depth_aug:
                obs["img"] = self._augment_depth(obs["img"])
                obs["img"][obs["img"] > self.ov_env.cfg.d_max] = 0.0
                obs["img"][obs["img"] < self.ov_env.cfg.d_min] = 0.0

            if self.rgb_aug is not None:
                # mask = 배경(depth 밴드 초과) 픽셀 → 배경만 VOC 이미지로 교체
                obs["rgb"] = self.rgb_aug.apply(obs["rgb"], obs["mask"])

            # play(eval)에선 autocast 끔(fp32) — bf16 이 teacher sigma 를 음수로 반올림해
            # 모델 내부 sample(models.py normal std>=0) 이 죽는다. 학습만 bf16 유지.
            import contextlib as _ctx
            _amp = (
                _ctx.nullcontext() if self.play_policy
                else torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            )
            with _amp:
                with torch.no_grad():
                    actions_teacher = self._get_actions(obs, "teacher")
                    self.actions_teacher = actions_teacher["actions"]

                actions_student = self._get_actions(obs, "student")
                aux_loss = self._compute_aux_loss(actions_student, obs)

                # teacher sigma의 역수 제곱으로 가중 — teacher가 확신하는 축을 더 강하게 모방
                weights = (1.0 / actions_teacher["sigmas"][0]) ** 2
                student_loss = self._loss(
                    actions_student["mus"], actions_teacher["mus"],
                    fn="weighted_l2", weights=weights,
                ) + self._loss(actions_student["sigmas"], actions_teacher["sigmas"])
                total_loss += student_loss + self.aux_coeff * sum(aux_loss)

            if self.rank == 0:
                self._log_information(log_counter, total_loss, aux_loss, beta)

            log_counter += 1

            if self.is_rnn:
                if log_counter % self.seq_length == 0:
                    total_loss = self._optimizer_step(total_loss)
            else:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                total_loss = 0.0

            stepping_actions = self._mix_actions(actions_student, actions_teacher, beta)
            obs, rew, out_of_reach, timed_out, _ = self.env.step(
                stepping_actions.detach()
            )

            self.frame += self.num_envs
            self.current_rewards += rew.unsqueeze(-1)
            self.current_lengths += 1
            self.dones = out_of_reach | timed_out
            all_done_indices = self.dones.nonzero(as_tuple=False)

            if len(all_done_indices) > 0:
                total_loss = self._on_done(total_loss, all_done_indices)
                if self.rgb_aug is not None:
                    self.rgb_aug.reset(all_done_indices)

            self.game_rewards.update(self.current_rewards[all_done_indices])
            self.game_lengths.update(self.current_lengths[all_done_indices])
            not_dones = 1.0 - self.dones.float()
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
            self.actions_teacher[all_done_indices] *= 0.0

            if not self.play_policy and self.rank == 0 and log_counter % CKPT_INTERVAL == 0:
                self.save(
                    os.path.join(self.nn_dir, f"grasp_student_{log_counter}_iters")
                )

    def _optimizer_step(self, total_loss):
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), GRAD_CLIP_NORM)
        self.optimizer.step()
        self.optimizer.zero_grad()
        # 다음 스텝의 BPTT가 이미 소비한 그래프를 다시 타지 않도록 hidden 분리
        for i, s in enumerate(self.student_hidden_states):
            self.student_hidden_states[i] = s.detach()
        torch.cuda.empty_cache()
        return 0.0

    def _on_done(self, total_loss, all_done_indices):
        """에피소드 종료 env의 RNN hidden state를 0으로 리셋한다.

        seq_length>1(BPTT 윈도우)에서는 여기서 optimizer flush 를 하지 않는다.
        예전엔 done 이 하나라도 있으면 매 스텝 flush 해서, 리셋이 잦은 다중 env 에선
        유효 BPTT 윈도우가 1 로 붕괴했다(memory 학습 불가). flush 는 distill() 의
        경계(log_counter % seq_length == 0)에서만 한다.

        done env 의 hidden 은 0 으로 만들되, 윈도우 그래프에 걸린 텐서를 in-place 로
        건드리면 autograd 가 죽으므로 마스크 곱(새 텐서)으로 처리한다.
        """
        if self.is_rnn:
            done_keep = (~self.dones.bool()).to(
                self.student_hidden_states[0].dtype
            ).view(1, -1, 1)                       # (1, num_envs, 1)
            for i in range(len(self.student_hidden_states)):
                with torch.no_grad():
                    self.hidden_state_means[i](
                        self.student_hidden_states[i][:, all_done_indices[0]].permute(
                            (1, 0, 2)
                        )
                    )
                # in-place 대신 마스크 곱 — 윈도우 그래프 손상 없이 done env hidden 0
                self.student_hidden_states[i] = self.student_hidden_states[i] * done_keep

        if self.is_teacher_rnn:
            for s in self.teacher_hidden_states:
                s[:, all_done_indices, ...] *= 0.0

        return total_loss

    def _mix_actions(self, actions_student, actions_teacher, beta: float):
        """DAgger 혼합: env별로 beta 확률만큼 teacher action으로 스텝."""
        p = torch.rand(self.num_envs, device=self.device)
        stepping_actions = torch.zeros_like(actions_student["actions"])
        student_inds = p > beta
        teacher_inds = ~student_inds
        if torch.any(student_inds):
            stepping_actions[student_inds] = actions_student["actions"][student_inds]
        if torch.any(teacher_inds):
            stepping_actions[teacher_inds] = actions_teacher["actions"][teacher_inds]
        return stepping_actions

    def _compute_aux_loss(self, actions_student, obs) -> list:
        """aux head(object_pos 등) 지도 손실. aux 미사용 네트워크면 [0.]."""
        if not self.is_aux or actions_student["aux"] is None:
            return [0.0]

        aux_out = actions_student["aux"]
        aux_gt = obs["aux_info"]
        self.aux_loss_names = list(aux_out.keys())

        aux_loss = []
        for name in self.aux_loss_names:
            if "img" in name:
                if "depth" in name:
                    d_min, d_max = self.ov_env.cfg.d_min, self.ov_env.cfg.d_max
                    aux_out[name] = aux_out[name] * (d_max - d_min) + d_min
                aux_loss.append(
                    torch.mean(
                        torch.norm(aux_out[name] - aux_gt[name], p=2, dim=(1, 2, 3))
                    )
                )
                if self.rank == 0:
                    self._log_img(aux_out[name][:5], aux_gt[name][:5])
            elif "uv" in name:
                uv_mask = ((aux_out[name] >= 0) & (aux_out[name] <= 1)).all(dim=-1)
                aux_loss.append(
                    self._loss(
                        aux_out[name][uv_mask],
                        aux_gt[name][uv_mask].reshape(len(uv_mask), -1),
                    )
                )
            else:
                aux_loss.append(
                    self._loss(
                        aux_out[name], aux_gt[name].reshape(self.num_envs, -1)
                    )
                )
        return aux_loss

    def _get_actions(self, obs: dict, policy_type: str) -> dict:
        aux = None
        if policy_type == "student":
            batch_dict = {
                "is_train": True,
                "obs": obs[self.student_obs_type],
                "prev_actions": self.prev_actions_student,
                "img": obs["img"],
                "rgb": obs["rgb"],
                "rgb_data": obs["rgb"],
                "finetune_backbone": self.finetune_backbone,
            }
            if self.is_rnn:
                batch_dict["rnn_states"] = self.student_hidden_states
                batch_dict["seq_length"] = 1
                batch_dict["rnn_masks"] = None
            res_dict = self.student_model_ddp(batch_dict)
            if self.is_rnn:
                # aux 네트워크는 rnn_states 슬롯에 (states, aux) 튜플을 싣는다
                if self.is_aux:
                    self.student_hidden_states = list(res_dict["rnn_states"][0])
                else:
                    self.student_hidden_states = list(res_dict["rnn_states"])
            if self.is_aux:
                aux = res_dict["rnn_states"][1]
        else:
            batch_dict = {
                "is_train": False,
                "obs": obs[self.teacher_obs_type],
                "prev_actions": self.prev_actions_teacher,
            }
            if self.is_teacher_rnn:
                batch_dict["rnn_states"] = self.teacher_hidden_states
                batch_dict["seq_length"] = 1
                batch_dict["rnn_masks"] = None
            res_dict = self.teacher_model(batch_dict)
            if self.is_teacher_rnn:
                self.teacher_hidden_states = res_dict["rnn_states"]

        mus, sigmas = res_dict["mus"], res_dict["sigmas"]
        distr = torch.distributions.Normal(mus, sigmas, validate_args=False)
        selected_action = torch.clamp(distr.sample().squeeze(), -1.0, 1.0)

        return {"mus": mus, "sigmas": sigmas, "actions": selected_action, "aux": aux}

    def _augment_depth(self, depth_nchw: torch.Tensor) -> torch.Tensor:
        """D435i 스테레오 깊이 노이즈 모사 (상관노이즈·정규노이즈·dropout·스틱)."""
        depth_nhw = depth_nchw[:, 0]
        depths = torch.clone(depth_nhw)
        self.depth_aug.add_correlated_noise(
            depth_nhw, depths, **self.depth_aug_cfg["correlated_noise"]
        )
        self.depth_aug.add_normal_noise(depths, **self.depth_aug_cfg["normal_noise"])
        self.depth_aug.add_pixel_dropout_and_randu(
            depths, **self.depth_aug_cfg["pixel_dropout_and_randu"]
        )
        self.depth_aug.add_sticks(depths, **self.depth_aug_cfg["sticks"])
        return depths.unsqueeze(1)

    def _loss(self, student_result, target_result, fn="l2", weights=None):
        if fn == "l2":
            loss = l2(student_result, target_result)
        else:
            loss = weighted_l2(student_result, target_result, weights)
        losses, _ = torch_ext.apply_masks([loss.unsqueeze(1)], None)
        return losses[0]

    # ------------------------------------------------------------------
    # 로깅 / 체크포인트
    # ------------------------------------------------------------------
    def _log_information(self, log_counter, total_loss, aux_loss, beta) -> None:
        student_loss = total_loss - self.aux_coeff * sum(aux_loss)
        perf = self.ov_env.in_success_region.float().mean().cpu().numpy()

        if self.game_rewards.current_size > 0:
            mean_rewards = self.game_rewards.get_mean()
            self.writer.add_scalar("rewards/step", mean_rewards[0], self.frame)
        self.writer.add_scalar("total_loss", float(total_loss), self.frame)
        self.writer.add_scalar("imitation_loss", float(student_loss), self.frame)
        self.writer.add_scalar("beta", beta, self.frame)
        self.writer.add_scalar("in_success_region", perf, self.frame)
        if self.is_aux:
            for idx, name in enumerate(self.aux_loss_names):
                self.writer.add_scalar(
                    f"aux_loss_{name}", float(aux_loss[idx]), self.frame
                )

        if log_counter % 10 == 0:
            print("=" * 10)
            print(f"ITERATION: {log_counter}")
            print(f"  imitation_loss: {float(student_loss):.5f}")
            print(f"  total_loss: {float(total_loss):.5f}")
            print(f"  in_success_region: {perf}")
            if self.game_rewards.current_size > 0:
                print(f"  mean_rewards: {self.game_rewards.get_mean()}")
                print(f"  mean_length: {self.game_lengths.get_mean()}")

    def _log_img(self, pred_images, gt_images) -> None:
        grid = vutils.make_grid(
            torch.cat((pred_images, gt_images), dim=0),
            nrow=pred_images.shape[0],
            normalize=True,
            scale_each=True,
        )
        self.writer.add_image("pred_vs_gt", grid, global_step=self.frame)

    def _set_weights(self, ckpt: str, policy_type: str) -> None:
        weights = torch_ext.load_checkpoint(ckpt)
        if policy_type == "student":
            model = self.student_model
            weights["model"] = adjust_state_dict_keys(
                weights["model"], model.state_dict()
            )
        else:
            model = self.teacher_model
        model.load_state_dict(weights["model"])
        if self.normalize_input and "running_mean_std" in weights:
            model.running_mean_std.load_state_dict(weights["running_mean_std"])

    def save(self, filename: str) -> None:
        torch_ext.save_checkpoint(filename, {
            "model": self.student_model.state_dict(),
            "epoch": self.epoch_num,
            "optimizer": self.optimizer.state_dict(),
            "frame": self.frame,
        })

    @staticmethod
    def _load_network(params: dict):
        return ModelBuilder().load(params)

    @staticmethod
    def _load_param_dict(cfg_path: str) -> dict:
        with open(cfg_path) as f:
            return yaml.safe_load(f)

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

"""Dual-head PPO configuration for bimanual pouring.

This configuration uses separate encoders for left and right arms,
allowing rollout of pre-trained grasp policies (2g_grasp_left_v1, 2g_grasp_right_v1)
without distribution shift conflicts.

Key parameters:
- dof_split_index: 8 (left arm actions [0:8], right arm actions [8:])
- actor_obs_split_index: 40 (left obs [0:40], right obs [40:])
- separate_actor_encoders: True (each arm has its own encoder)
"""

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils import configclass

from sbm.rl import SbmDualHeadActorCriticCfg


@configclass
class OpenArmPouringV1DualHeadPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "pipeline/both/2g_pouring_v1_dualhead"
    run_name = ""
    resume = False
    empirical_normalization = False
    policy = SbmDualHeadActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        # Action space split: left arm [0:8], right arm [8:]
        dof_split_index=8,
        # Observation space split: left obs [0:40], right obs [40:]
        actor_obs_split_index=40,
        critic_obs_split_index=40,
        # Separate encoders for each arm
        separate_actor_encoders=True,
        separate_critic_encoders=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=2.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

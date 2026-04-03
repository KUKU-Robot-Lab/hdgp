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

"""grasp_right checkpoint에서 grasp phase 후반 관절 위치를 수집하여 통계를 출력한다.

수집 타이밍: step 실행 전, episode_length_buf >= collect_from_step 이고
            num_contacts_buf >= min_contacts 인 env에서 에피소드당 1회 수집.
            (step 후 reset으로 버퍼가 클리어되기 전에 저장)

사용법:
    ./isaaclab.sh -p ../hdgp/scripts/reinforcement_learning/rl_games/extract_grasp_pose.py \
        --task 5g_grasp_right-v7 \
        --num_envs 100 \
        --headless \
        --extract_episodes 100 \
        --min_contacts 2 \
        --output /tmp/v7_grasp_pose_stats.yaml
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Extract grasp pose statistics from a trained checkpoint.")
parser.add_argument("--num_envs",            type=int,  default=4)
parser.add_argument("--task",                type=str,  default="5g_grasp_right-v7")
parser.add_argument("--checkpoint",          type=str,  default=None)
parser.add_argument("--seed",                type=int,  default=42)
parser.add_argument("--use_last_checkpoint", action="store_true")
parser.add_argument("--extract_episodes",    type=int,  default=100,
                    help="수집할 에피소드 수")
parser.add_argument("--min_contacts",        type=int,  default=2,
                    help="수집 시 최소 fingertip 접촉 수")
parser.add_argument("--collect_from_step",   type=int,  default=450,
                    help="수집 시작 episode step (grasp phase 후반, default=450)")
parser.add_argument("--output",              type=str,  default="/tmp/v7_grasp_pose_stats.yaml")
parser.add_argument("--disable_adr",         action="store_true", default=False)
parser.add_argument("--bead_fixed",          type=int,  default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os
import re

import gymnasium as gym
import torch
import yaml

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, DirectMARLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def _force_local_openarm_path() -> str:
    hdgp_root   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    openarm_src = os.path.join(hdgp_root, "source", "openarm")
    openarm_pkg = os.path.join(openarm_src, "openarm")
    if not os.path.isdir(openarm_pkg):
        raise RuntimeError(f"Local openarm package not found: {openarm_pkg}")
    if openarm_src in sys.path:
        sys.path.remove(openarm_src)
    sys.path.insert(0, openarm_src)
    return os.path.abspath(openarm_pkg)


_EXPECTED_OPENARM_DIR = _force_local_openarm_path()
import openarm  # noqa: E402
if not os.path.abspath(getattr(openarm, "__file__", "")).startswith(_EXPECTED_OPENARM_DIR + os.sep):
    raise RuntimeError(f"openarm resolved to unexpected location: {getattr(openarm, '__file__', None)}")
import openarm.tasks  # noqa: F401,E402


def _resolve_pipeline_log_components(task_name: str) -> tuple[str, str]:
    task_key       = task_name.split(":")[-1].replace("-Play", "")
    fallback_folder = task_key.replace("-", "_")
    try:
        spec = gym.spec(task_key)
        env_cfg_entry = spec.kwargs.get("env_cfg_entry_point", "")
        if isinstance(env_cfg_entry, str):
            match = re.search(
                r"\.pipeline\.(?:gripper|hand)\.(left|right|both)\.([A-Za-z0-9_]+)\.",
                env_cfg_entry,
            )
            if match:
                return match.group(1), match.group(2)
    except Exception:
        pass
    if "_right" in fallback_folder.lower():
        return "right", fallback_folder
    return "left", fallback_folder


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    task_name       = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed           = args_cli.seed

    if args_cli.disable_adr and hasattr(env_cfg, "enable_adr"):
        env_cfg.enable_adr = False
    if args_cli.bead_fixed is not None and hasattr(env_cfg, "bead_count_min"):
        env_cfg.bead_count_min = args_cli.bead_fixed
        env_cfg.bead_count_max = args_cli.bead_fixed

    side_dir, task_dir_name = _resolve_pipeline_log_components(train_task_name)
    sbm_root      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    log_root_path = os.path.join(sbm_root, "log", "rl_games", "pipeline", side_dir, task_dir_name)

    if args_cli.checkpoint is None:
        run_dir         = agent_cfg["params"]["config"].get("full_experiment_name", "test.*")
        checkpoint_file = (
            ".*" if args_cli.use_last_checkpoint
            else f"{agent_cfg['params']['config']['name']}.pth"
        )
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)

    print(f"[INFO] Checkpoint: {resume_path}")

    rl_device          = agent_cfg["params"]["config"]["device"]
    clip_obs           = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions       = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups         = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"]      = True
    agent_cfg["params"]["load_path"]            = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    # -----------------------------------------------------------------------
    # raw env 접근
    # -----------------------------------------------------------------------
    raw_env        = env.unwrapped
    n_envs         = args_cli.num_envs
    target         = args_cli.extract_episodes
    min_cont       = args_cli.min_contacts
    collect_from   = args_cli.collect_from_step

    print(f"[DEBUG] raw_env type     : {type(raw_env).__name__}")
    print(f"[DEBUG] has hand_dof_indices: {hasattr(raw_env, 'hand_dof_indices')}")
    print(f"[DEBUG] has num_contacts_buf: {hasattr(raw_env, 'num_contacts_buf')}")
    print(f"[DEBUG] max_episode_length  : {raw_env.max_episode_length}")

    # 각 env별로 이번 에피소드에서 이미 수집했는지 여부
    already_collected = torch.zeros(n_envs, dtype=torch.bool)

    collected_poses: list[torch.Tensor] = []
    total_steps = 0

    # -----------------------------------------------------------------------
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    print(f"\n[INFO] 수집 시작 — 목표: {target}개  min_contacts={min_cont}  collect_from_step={collect_from}")
    print(f"[INFO] 수집 타이밍: step 실행 전, ep_len >= {collect_from} AND contacts >= {min_cont}")
    print("-" * 60)

    while simulation_app.is_running() and len(collected_poses) < target:

        # ---- step 전: ep_len / contacts / hand_pos 스냅샷 ----
        pre_ep_len  = raw_env.episode_length_buf.clone()       # (N,)
        pre_contacts = raw_env.num_contacts_buf.clone()        # (N,)
        pre_hand_pos = raw_env.robot.data.joint_pos[
            :, raw_env.hand_dof_indices
        ].clone()                                              # (N, 20)

        # ---- 수집 체크 (step 전 데이터 기준) ----
        for ei in range(n_envs):
            if already_collected[ei]:
                continue
            el = pre_ep_len[ei].item()
            ct = pre_contacts[ei].item()
            if el >= collect_from and ct >= min_cont:
                already_collected[ei] = True
                collected_poses.append(pre_hand_pos[ei].cpu())
                print(
                    f"  [OK  {len(collected_poses):3d}/{target}]"
                    f"  env={ei}  ep_len={el:.0f}  contacts={ct}"
                    f"  thumb2={pre_hand_pos[ei, 1].item():.3f}"
                    f"  idx2={pre_hand_pos[ei, 5].item():.3f}"
                    f"  mid2={pre_hand_pos[ei, 9].item():.3f}"
                )
                if len(collected_poses) >= target:
                    break

        if len(collected_poses) >= target:
            break

        # ---- env.step ----
        with torch.inference_mode():
            obs_t   = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_t, is_deterministic=True)
            obs, _, dones, _ = env.step(actions)

            if agent.is_rnn and agent.states is not None:
                for s in agent.states:
                    s[:, dones, :] = 0.0

        total_steps += 1

        # ---- 100 스텝마다 진행 상황 ----
        if total_steps % 100 == 0:
            ep_len_mean = raw_env.episode_length_buf.float().mean().item()
            ct_mean     = raw_env.num_contacts_buf.float().mean().item()
            print(
                f"  [step {total_steps:6d}]"
                f"  collected={len(collected_poses)}/{target}"
                f"  ep_len={ep_len_mean:.0f}"
                f"  contacts_mean={ct_mean:.2f}"
            )

        # ---- 에피소드 종료 시 already_collected 리셋 ----
        if isinstance(dones, torch.Tensor):
            done_mask = dones.bool().cpu()
        else:
            done_mask = torch.tensor(dones, dtype=torch.bool)
        already_collected[done_mask] = False

    env.close()

    # -----------------------------------------------------------------------
    # 통계 계산 및 출력
    # -----------------------------------------------------------------------
    if len(collected_poses) == 0:
        print("[WARN] 수집된 pose 없음.")
        print(f"       collect_from_step({collect_from})을 낮추거나 min_contacts({min_cont})를 줄여보세요.")
        return

    all_poses = torch.stack(collected_poses, dim=0)  # (N, 20)
    mean_pose = all_poses.mean(dim=0)
    std_pose  = all_poses.std(dim=0)

    joint_names   = [f"rj_dg_{f}_{j}" for f in range(1, 6) for j in range(1, 5)]
    finger_labels = ["thumb", "index", "middle", "ring", "pinky"]

    print("\n" + "=" * 60)
    print(f"수집 완료: {len(collected_poses)}개")
    print("=" * 60)
    print(f"  {'Joint':<16} {'Mean':>8} {'Std':>8}")
    print("  " + "-" * 34)
    for i in range(20):
        fi, ji = divmod(i, 4)
        print(f"  {finger_labels[fi]}_{ji+1:<11}  {mean_pose[i].item():+8.4f}  ±{std_pose[i].item():6.4f}")

    # preset.py 붙여넣기용 포맷
    print("\n# ---- HAND_GRASP_POSE (preset.py 교체용) ----")
    print("HAND_GRASP_POSE = [")
    for fi, fname in enumerate(finger_labels):
        vals = ", ".join(f"{mean_pose[fi * 4 + j].item():+.3f}" for j in range(4))
        print(f"    {vals},   # {fname}")
    print("]")

    # YAML 저장
    stats = {
        "num_samples":       len(collected_poses),
        "task":              args_cli.task,
        "checkpoint":        resume_path,
        "collect_from_step": collect_from,
        "min_contacts":      min_cont,
        "joint_names":       joint_names,
        "mean":              [round(v.item(), 5) for v in mean_pose],
        "std":               [round(v.item(), 5) for v in std_pose],
        "hand_grasp_pose_per_finger": {
            fname: [round(mean_pose[fi * 4 + j].item(), 4) for j in range(4)]
            for fi, fname in enumerate(finger_labels)
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)), exist_ok=True)
    with open(args_cli.output, "w") as f:
        yaml.dump(stats, f, default_flow_style=False, allow_unicode=True)
    print(f"\n[INFO] 저장 완료: {args_cli.output}")


if __name__ == "__main__":
    main()
    simulation_app.close()

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

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from distutils.util import strtobool

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--reset_epoch",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="On --checkpoint resume, load weights only and reset epoch/frame to 0 (default). "
    "Use --no-reset_epoch to continue the checkpoint's epoch count.",
)
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--wandb-project-name", type=str, default=None, help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
parser.add_argument("--wandb-name", type=str, default=None, help="the name of wandb's run")
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="if toggled, this experiment will be tracked with Weights and Biases",
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import random
import re

import omni
from rl_games.common import env_configurations, vecenv
from rl_games.common import a2c_common
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rl_games import MultiObserver, PbtAlgoObserver, RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


class SingleAxisIsaacObserver(IsaacAlgoObserver):
    """env의 log/* 진단 지표를 frame/iter/time 3중 대신 iter(epoch) 1개만 기록.

    IsaacAlgoObserver는 direct_info를 3개 x축(frame/iter/time)에 모두 쓴다.
    여기서는 super()가 direct_info를 건너뛰게 한 뒤 iter 축에만 한 번 기록한다.
    (Episode/* 는 원래 1축, scores/episode_lengths/rewards 3중은 rl_games 코어 소관.)
    """

    def after_print_stats(self, frame, epoch_num, total_time):
        saved = self.direct_info
        self.direct_info = {}
        super().after_print_stats(frame, epoch_num, total_time)
        for k, v in saved.items():
            self.writer.add_scalar(f"{k}/iter", v, epoch_num)


def _force_local_openarm_path() -> str:
    """Force import path to hdgp/source/openarm so openarm resolves to local source only."""
    hdgp_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
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
    raise RuntimeError(
        f"openarm resolved to unexpected location: {getattr(openarm, '__file__', None)} "
        f"(expected under {_EXPECTED_OPENARM_DIR})"
    )
import openarm.tasks  # noqa: F401,E402

# PLACEHOLDER: Extension template (do not remove this comment)


def _resolve_pipeline_log_components(task_name: str) -> tuple[str, str]:
    """Resolve <robot>/<side>/<task-ver> or <side>/<folder> from task name.

    신규 형식 open-<robot>_<side>_<task>_<ver>  →  (robot, side, task-ver)
      예: open-tesol_r_pour_v3  →  log/rl_games/open-tesol/right/pour-v3/
    레거시 형식은 env_cfg_entry_point 경로에서 파싱.
    """
    SIDE_MAP = {"r": "right", "l": "left", "b": "both"}
    task_key = task_name.split(":")[-1]

    # 신규 형식 감지: open-<robot>_<side>_<task>_... 패턴
    new_fmt = re.match(r"^(open-[A-Za-z0-9]+)_([rbl])_(.+?)(?:-play|-lstm|-bc|-il.*|-diffusion)?$", task_key, re.IGNORECASE)
    if new_fmt:
        robot = new_fmt.group(1)          # open-tesol
        side = SIDE_MAP.get(new_fmt.group(2), new_fmt.group(2))   # right
        task_ver = new_fmt.group(3).replace("_", "-")             # pour-v3
        # log path: <robot>/<side>/<task-ver>  — 반환값 재사용을 위해 (side_dir, task_dir_name) 형태 유지
        # side_dir에 robot/side를 담고 task_dir_name에 task-ver를 담는다
        return f"{robot}/{side}", task_ver

    # 레거시 형식: env_cfg_entry_point 경로 파싱
    task_key_legacy = task_key.replace("-Play", "")
    fallback_folder = task_key_legacy.replace("-", "_")
    try:
        spec = gym.spec(task_key_legacy)
        env_cfg_entry = spec.kwargs.get("env_cfg_entry_point", "")
        if isinstance(env_cfg_entry, str):
            match = re.search(r"\.pipeline\.(?:gripper|hand)\.([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)\.", env_cfg_entry)
            if match:
                return match.group(1), match.group(2)
    except Exception:
        pass
    if "_right" in fallback_folder.lower():
        return "right", fallback_folder
    if "_both" in fallback_folder.lower():
        return "both", fallback_folder
    return "left", fallback_folder


def _patch_optimizer_restore() -> None:
    """Make rl_games checkpoint restore tolerate optimizer-state shape drift.

    For BC-pretrained checkpoints we want to reuse actor / critic weights but do
    not require the exact optimizer slot layout from the offline pretraining run.
    """
    if getattr(a2c_common.A2CBase, "_hdgp_optimizer_restore_patched", False):
        return

    original = a2c_common.A2CBase.set_full_state_weights

    def _set_full_state_weights(self, weights, set_epoch=True):
        self.set_weights(weights)
        # --reset_epoch(기본 on): 가중치만 받고 epoch/frame은 0 유지 → max_epochs를 처음부터.
        if set_epoch and not getattr(args_cli, "reset_epoch", False):
            self.epoch_num = weights["epoch"]
            self.frame = weights["frame"]

        if self.has_central_value:
            self.central_value_net.load_state_dict(weights["assymetric_vf_nets"])

        try:
            self.optimizer.load_state_dict(weights["optimizer"])
        except ValueError as exc:
            print(f"[WARN] Skipping optimizer state restore: {exc}")

        self.last_mean_rewards = weights.get("last_mean_rewards", -1000000000)

        if self.vec_env is not None:
            env_state = weights.get("env_state", None)
            self.vec_env.set_env_state(env_state)

    a2c_common.A2CBase.set_full_state_weights = _set_full_state_weights
    a2c_common.A2CBase._hdgp_optimizer_restore_patched = True


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with RL-Games agent."""
    _patch_optimizer_restore()
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    # multi-gpu training config
    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # LOG PATH RULE:
    #   신규 형식: log/rl_games/<robot>/<side>/<task-ver>/testN
    #     예: open-tesol_r_pour_v3 → log/rl_games/open-tesol/right/pour-v3/
    #   레거시 side(left|right|both): log/rl_games/pipeline/<side>/<task_dir_name>/testN
    #   레거시 명명형(inspire_r 등): log/rl_games/<variant>/<task_dir_name>/testN
    task_name = args_cli.task
    side_dir, task_dir_name = _resolve_pipeline_log_components(task_name)
    sbm_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if "/" in side_dir:
        # 신규 형식: side_dir = "<robot>/<side>"
        log_root_path = os.path.join(sbm_root, "log", "rl_games", side_dir, task_dir_name)
    elif side_dir in ("left", "right", "both"):
        log_root_path = os.path.join(sbm_root, "log", "rl_games", "pipeline", side_dir, task_dir_name)
    else:
        # 레거시 명명형: inspire_r 등
        log_root_path = os.path.join(sbm_root, "log", "rl_games", side_dir, task_dir_name)
    if "pbt" in agent_cfg:
        if agent_cfg["pbt"]["directory"] == ".":
            log_root_path = os.path.abspath(log_root_path)
        else:
            log_root_path = os.path.join(agent_cfg["pbt"]["directory"], log_root_path)

    os.makedirs(log_root_path, exist_ok=True)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # LOG RUN-NAME RULE:
    #   auto-increment "test1", "test2", ...
    # Modify this if you prefer timestamp/custom run names.
    existing = []
    for name in os.listdir(log_root_path):
        if name.startswith("test"):
            suffix = name[4:]
            if suffix.isdigit():
                existing.append(int(suffix))
    next_idx = (max(existing) + 1) if existing else 1
    log_dir = f"test{next_idx}"
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir
    config_name = agent_cfg["params"]["config"]["name"]
    wandb_project = config_name if args_cli.wandb_project_name is None else args_cli.wandb_project_name
    experiment_name = log_dir if args_cli.wandb_name is None else args_cli.wandb_name

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        print(
            "[WARN] IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # Keep env log_dir as absolute run path so play/tools can reuse this folder.
    env_cfg.log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games

    if "pbt" in agent_cfg and agent_cfg["pbt"]["enabled"]:
        observers = MultiObserver([SingleAxisIsaacObserver(), PbtAlgoObserver(agent_cfg, args_cli)])
        runner = Runner(observers)
    else:
        runner = Runner(SingleAxisIsaacObserver())

    runner.load(agent_cfg)

    # reset the agent and env
    runner.reset()
    # train the agent

    global_rank = int(os.getenv("RANK", "0"))
    if args_cli.track and global_rank == 0:
        if args_cli.wandb_entity is None:
            raise ValueError("Weights and Biases entity must be specified for tracking.")
        import wandb

        wandb.init(
            project=wandb_project,
            entity=args_cli.wandb_entity,
            name=experiment_name,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        if not wandb.run.resumed:
            wandb.config.update({"env_cfg": env_cfg.to_dict()})
            wandb.config.update({"agent_cfg": agent_cfg})

    if args_cli.checkpoint is not None:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

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

"""Evaluate a pour checkpoint across all envs and report per-env bead transfer.

play.py의 체크포인트 로딩부를 그대로 사용하되, env-0 진단 렌더 대신
전 환경을 결정론으로 롤아웃하여 환경별 bead 이송 개수·성공률을 집계하고
Markdown 리포트를 생성한다.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--disable_adr",
    action="store_true",
    default=False,
    help="Disable ADR for visualization/eval-style playback.",
)
parser.add_argument(
    "--bead_fixed",
    type=int,
    default=None,
    help="Fix bead count for playback. Supports bead_count_min/max and single bead_count curriculum tasks.",
)
parser.add_argument(
    "--freeze_grasp_hand",
    action="store_true",
    default=False,
    help="Freeze grasp-hand joints during playback for stable rendering/evaluation.",
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--eval_steps", type=int, default=1600,
    help="Number of deterministic rollout steps to collect per-env episode outcomes.",
)
parser.add_argument(
    "--eval_out", type=str, default=None,
    help="Path to write the per-env bead-transfer analysis Markdown report.",
)
parser.add_argument(
    "--success_fill", type=float, default=None,
    help="Success threshold on bead-in-target fraction (default: env success_target_fill_ratio).",
)
# --- 궤적 기록 옵션 (Phase 2) ---
parser.add_argument(
    "--record_out", type=str, default=None,
    help="컵 포즈 + 로봇 관절각 궤적을 저장할 hdf5 경로. (미지정 시 log_dir/pour_traj_<stamp>.hdf5)",
)
parser.add_argument(
    "--record_episodes", type=int, default=16,
    help="저장할 에피소드 수(bead frac 상위부터 선별).",
)
parser.add_argument(
    "--record_collect", type=int, default=0,
    help="선별 전 수집할 총 에피소드 수(0=record_episodes*3).",
)
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
import importlib
import math
import os
from pathlib import Path
import random
import re
import time
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common import a2c_common
from rl_games.common.player import BasePlayer
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
from isaaclab.utils.io import load_yaml
try:
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    try:
        from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
    except ModuleNotFoundError:
        def get_published_pretrained_checkpoint(workflow: str, task_name: str):
            raise ModuleNotFoundError(
                "Pretrained checkpoint utility is unavailable in this IsaacLab install. "
                "Use --checkpoint or disable --use_pretrained_checkpoint."
            )

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


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
    """Resolve <robot>/<side>/<task-ver> or <side>/<folder> from task name."""
    side_map = {"r": "right", "l": "left", "b": "both"}
    task_key = _strip_play_task_name(task_name)

    new_fmt = re.match(r"^(open-[A-Za-z0-9]+)_([rbl])_(.+?)(?:-lstm|-bc|-il.*|-diffusion)?$", task_key, re.IGNORECASE)
    if new_fmt:
        robot = new_fmt.group(1)
        side = side_map.get(new_fmt.group(2), new_fmt.group(2))
        task_ver = new_fmt.group(3).replace("_", "-")
        return f"{robot}/{side}", task_ver

    fallback_folder = task_key.replace("-", "_")
    try:
        spec = gym.spec(task_key)
        env_cfg_entry = spec.kwargs.get("env_cfg_entry_point", "")
        if isinstance(env_cfg_entry, str):
            match = re.search(r"\.pipeline\.(?:gripper|hand)\.(left|right|both)\.([A-Za-z0-9_]+)\.", env_cfg_entry)
            if match:
                return match.group(1), match.group(2)
    except Exception:
        pass
    if "_right" in fallback_folder.lower():
        return "right", fallback_folder
    if "_both" in fallback_folder.lower():
        return "both", fallback_folder
    return "left", fallback_folder


def _strip_play_task_name(task_name: str) -> str:
    """Map play task ids back to their train task id."""
    task_key = task_name.split(":")[-1]
    return (
        task_key
        .replace("-Play-", "-")
        .replace("-play-", "-")
        .replace("-Play", "")
        .replace("-play", "")
    )


def _resolve_checkpoint_name(task_name: str) -> str:
    """Use the Gym task id as the rl-games checkpoint basename.

    Must stay identical to train.py so play.py's default ("best") lookup matches
    the file train.py actually saved (`{name}.pth`).
    """
    task_key = task_name.split(":")[-1]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", task_key)


def _resolve_run_dir_prefix(task_name: str) -> str:
    task_key = task_name.split(":")[-1]
    return "lstm_test" if task_key.endswith("-lstm") else "test"


def _checkpoint_sort_key(path: Path) -> tuple[float, int, float]:
    """Prefer higher reward, then later epoch, then newer mtime for prefix matches."""
    match = re.search(r"_ep_(\d+)_rew_([-+]?\d+(?:\.\d+)?)", path.name)
    if match:
        return (float(match.group(2)), int(match.group(1)), path.stat().st_mtime)
    return (float("-inf"), -1, path.stat().st_mtime)


def _resolve_checkpoint_path(checkpoint: str) -> str:
    """Resolve checkpoint paths and common truncated-prefix CLI input."""
    try:
        return retrieve_file_path(checkpoint)
    except FileNotFoundError:
        pass

    candidate = Path(checkpoint).expanduser()
    search_candidates = [candidate]
    if not candidate.is_absolute():
        sbm_root = Path(__file__).resolve().parents[3]
        search_candidates.append((sbm_root / candidate).resolve())

    prefix_matches: list[Path] = []
    for candidate in search_candidates:
        if candidate.is_file():
            return str(candidate)
        if candidate.parent.is_dir():
            prefix_matches.extend(sorted(candidate.parent.glob(candidate.name + "*.pth")))

    if prefix_matches:
        unique_matches = sorted(set(prefix_matches), key=_checkpoint_sort_key, reverse=True)
        if len(unique_matches) > 1:
            preview = "\n".join(f"  - {path}" for path in unique_matches[:10])
            raise FileNotFoundError(
                f"Multiple checkpoint files match prefix: {checkpoint}\n"
                f"Use the full checkpoint path. Top matches:\n{preview}"
            )
        resolved = unique_matches[0]
        print(f"[INFO] Parsed unique checkpoint prefix: {checkpoint} -> {resolved}")
        return str(resolved)

    raise FileNotFoundError(f"Unable to find the checkpoint file or prefix: {checkpoint}")


def _rebase_logged_paths(value, *, workspace_root: str):
    """Map absolute paths from another machine's logged cfg onto this workspace."""
    if isinstance(value, dict):
        return {k: _rebase_logged_paths(v, workspace_root=workspace_root) for k, v in value.items()}
    if isinstance(value, list):
        return [_rebase_logged_paths(v, workspace_root=workspace_root) for v in value]
    if isinstance(value, tuple):
        return tuple(_rebase_logged_paths(v, workspace_root=workspace_root) for v in value)
    if not isinstance(value, str) or os.path.exists(value):
        return value

    marker = "/rl_ws/"
    if marker not in value:
        return value
    rel = value.split(marker, 1)[1]
    candidate = os.path.join(workspace_root, rel)
    return candidate if os.path.exists(candidate) else value


def _apply_logged_env_cfg(target, logged: dict) -> None:
    """Recursively copy logged env config values onto the Hydra config object."""
    if not isinstance(logged, dict):
        return
    for key, value in logged.items():
        if key == "func":
            continue
        if isinstance(target, dict):
            if key not in target:
                continue
            current = target[key]
        elif hasattr(target, key):
            current = getattr(target, key)
        else:
            continue
        if callable(current):
            continue

        if isinstance(value, dict) and (isinstance(current, dict) or hasattr(current, "__dict__")):
            _apply_logged_env_cfg(current, value)
        else:
            try:
                if isinstance(target, dict):
                    target[key] = value
                else:
                    setattr(target, key, value)
            except Exception:
                pass


def _restore_run_cfg_if_available(env_cfg, agent_cfg: dict, *, resume_path: str, workspace_root: str) -> dict:
    """Use params saved next to the checkpoint so playback matches training."""
    run_dir = os.path.dirname(os.path.dirname(resume_path))
    params_dir = os.path.join(run_dir, "params")
    env_yaml = os.path.join(params_dir, "env.yaml")
    agent_yaml = os.path.join(params_dir, "agent.yaml")

    if os.path.exists(env_yaml):
        logged_env = _rebase_logged_paths(load_yaml(env_yaml), workspace_root=workspace_root)
        _apply_logged_env_cfg(env_cfg, logged_env)
        print(f"[INFO] Restored playback env cfg from: {env_yaml}")
    else:
        print(f"[WARN] Run env cfg not found; using current source cfg: {env_yaml}")

    if os.path.exists(agent_yaml):
        logged_agent = _rebase_logged_paths(load_yaml(agent_yaml), workspace_root=workspace_root)
        if isinstance(logged_agent, dict) and "params" in logged_agent:
            print(f"[INFO] Restored playback agent cfg from: {agent_yaml}")
            return logged_agent
        print(f"[WARN] Ignoring malformed run agent cfg: {agent_yaml}")
    else:
        print(f"[WARN] Run agent cfg not found; using current source cfg: {agent_yaml}")
    return agent_cfg


def _apply_playback_env_overrides(env_cfg) -> None:
    """Apply CLI-only playback overrides after any logged cfg restore."""
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    if args_cli.disable_adr:
        disabled_attrs = []
        for adr_attr in (
            "enable_adr",
            "enable_noise_adr",
            "enable_bead_count_adr",
            "enable_success_adr",
            "enable_spill_adr",
        ):
            if hasattr(env_cfg, adr_attr):
                setattr(env_cfg, adr_attr, False)
                disabled_attrs.append(adr_attr)
        if disabled_attrs:
            print(f"[INFO] ADR disabled for playback: {', '.join(disabled_attrs)}")
        else:
            print("[WARN] --disable_adr ignored: env does not expose ADR flags")

    if args_cli.bead_fixed is not None:
        if hasattr(env_cfg, "bead_count_min") and hasattr(env_cfg, "bead_count_max"):
            env_cfg.bead_count_min = args_cli.bead_fixed
            env_cfg.bead_count_max = args_cli.bead_fixed
            print(f"[INFO] bead count fixed for playback: {args_cli.bead_fixed}")
        elif hasattr(env_cfg, "bead_count"):
            env_cfg.bead_count = args_cli.bead_fixed
            if hasattr(env_cfg, "bead_count_stages"):
                env_cfg.bead_count_stages = (args_cli.bead_fixed,)
            if hasattr(env_cfg, "enable_bead_count_adr"):
                env_cfg.enable_bead_count_adr = False
            print(f"[INFO] bead count fixed for playback: {args_cli.bead_fixed}")
        else:
            print("[WARN] --bead_fixed ignored: env does not expose bead count settings")

    if args_cli.freeze_grasp_hand:
        if hasattr(env_cfg, "freeze_grasp_hand_during_episode"):
            env_cfg.freeze_grasp_hand_during_episode = True
            print("[INFO] grasp hand frozen during playback.")
        else:
            print("[WARN] --freeze_grasp_hand ignored: env does not expose freeze_grasp_hand_during_episode")


def _patch_optimizer_restore() -> None:
    """Make checkpoint restore match train.py for BC-pretrained weights."""
    if getattr(a2c_common.A2CBase, "_hdgp_optimizer_restore_patched", False):
        return

    def _set_full_state_weights(self, weights, set_epoch=True):
        self.set_weights(weights)
        if set_epoch:
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


def _install_player_recurrent_gate(agent: BasePlayer, agent_cfg: dict) -> None:
    """Install v4 recurrent gate modules before loading gated checkpoints."""
    cfg = agent_cfg.get("params", {}).get("config", {})
    if not bool(cfg.get("recurrent_gate_enable", False)):
        return

    try:
        recurrent_gate = importlib.import_module(
            "openarm.tasks.manager_based.openarm_manipulation"
            ".pipeline.hand.right.5g_pour_right_v4.recurrent_gate"
        )
    except Exception as exc:  # noqa: BLE001 - keep non-v4 playback usable.
        print(f"[WARN] Unable to import recurrent gate for player restore: {exc}")
        return

    actor_obs_shape = agent.model.obs_shape
    actor_obs_dim = int(actor_obs_shape[0] if isinstance(actor_obs_shape, (tuple, list)) else actor_obs_shape)
    install_recurrent_gate = recurrent_gate.install_recurrent_gate
    install_recurrent_gate(agent.model.a2c_network, obs_dim=actor_obs_dim)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with RL-Games agent."""
    _patch_optimizer_restore()
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = _strip_play_task_name(task_name)

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # CHECKPOINT SEARCH ROOT RULE:
    #   new:    <sbm_root>/log/rl_games/<robot>/<side>/<task-ver>
    #   legacy: <sbm_root>/log/rl_games/pipeline/<left|right|both>/<task_dir_name>
    side_dir, task_dir_name = _resolve_pipeline_log_components(train_task_name)
    sbm_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if "/" in side_dir:
        log_root_path = os.path.join(sbm_root, "log", "rl_games", side_dir, task_dir_name)
    elif side_dir in ("left", "right", "both"):
        log_root_path = os.path.join(sbm_root, "log", "rl_games", "pipeline", side_dir, task_dir_name)
    else:
        log_root_path = os.path.join(sbm_root, "log", "rl_games", side_dir, task_dir_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # specify directory for logging runs
        # Default: search test-style runs. Change run_dir pattern here if needed.
        run_prefix = _resolve_run_dir_prefix(train_task_name)
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", f"{run_prefix}.*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint — resolve the basename exactly like
            # train.py (it overwrites config.name at runtime), since the source yaml
            # config.name can be stale after a task rename.
            checkpoint_file = f"{_resolve_checkpoint_name(train_task_name)}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = _resolve_checkpoint_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    workspace_root = os.path.abspath(os.path.join(sbm_root, ".."))
    agent_cfg = _restore_run_cfg_if_available(
        env_cfg,
        agent_cfg,
        resume_path=resume_path,
        workspace_root=workspace_root,
    )
    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]
    _apply_playback_env_overrides(env_cfg)

    # pour 태스크 한정: pour_point(빨강) 마커 강제 표시.
    #   logged env.yaml이 enable_visual_markers=False로 덮어쓰므로 cfg 복원 이후에 주입.
    if "pour" in train_task_name.lower() and hasattr(env_cfg, "enable_visual_markers"):
        env_cfg.enable_visual_markers = True
        print("[INFO] pour playback: visual marker (pour_point=red) enabled.")

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
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

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    _install_player_recurrent_gate(agent, agent_cfg)
    agent.restore(resume_path)
    agent.reset()

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    # ------------------------------------------------------------------
    # [Phase 2 기록] 결정론 롤아웃하며 컵 포즈 + 로봇 관절각을 에피소드별로 저장.
    #   유체 재생(replay_pour_fluid.py)에서 이 궤적으로 컵/로봇을 kinematic 구동한다.
    # ------------------------------------------------------------------
    import numpy as np

    try:
        import h5py
    except ImportError as exc:  # noqa: BLE001
        raise SystemExit(f"[REC] h5py 필요: {exc}")

    _raw_env = env.unwrapped
    if hasattr(_raw_env, "env"):
        _raw_env = _raw_env.env.unwrapped

    N = int(_raw_env.num_envs)
    if N != 1:
        raise SystemExit("[REC] 기록은 --num_envs 1 로 실행하세요 (에피소드 궤적 단일 추적).")

    num_beads = int(getattr(_raw_env, "num_beads", getattr(_raw_env.cfg, "bead_count", 0)))
    step_dt = float(getattr(_raw_env, "step_dt",
                            getattr(_raw_env, "physics_dt", 1.0 / 60.0) * getattr(_raw_env.cfg, "decimation", 1)))
    env_origin = _raw_env.scene.env_origins[0].detach().cpu().numpy().astype(np.float32)
    joint_names = list(_raw_env.robot.joint_names)
    body_names = list(_raw_env.robot.data.body_names)

    # 로봇 USD 경로 (replay 에서 로봇 재생용)
    robot_usd = ""
    try:
        robot_usd = str(_raw_env.cfg.robot_cfg.spawn.usd_path)
    except Exception:  # noqa: BLE001
        pass

    def _cup_pose(cup):
        p = (cup.data.root_pos_w[0] - _raw_env.scene.env_origins[0]).detach().cpu().numpy()
        q = cup.data.root_quat_w[0].detach().cpu().numpy()   # wxyz
        return np.concatenate([p, q]).astype(np.float32)

    def _robot_root():
        p = (_raw_env.robot.data.root_pos_w[0] - _raw_env.scene.env_origins[0]).detach().cpu().numpy()
        q = _raw_env.robot.data.root_quat_w[0].detach().cpu().numpy()
        return np.concatenate([p, q]).astype(np.float32)

    def _to_np(attr):
        t = getattr(_raw_env, attr, None)
        return t.detach().float().cpu().numpy() if t is not None else np.zeros(N)

    collect = args_cli.record_collect or (args_cli.record_episodes * 3)
    print(f"[REC] 수집 목표 {collect} 에피소드 → 상위 {args_cli.record_episodes} 저장 "
          f"(beads/env={num_beads}, dt={step_dt:.4f})", flush=True)

    episodes = []                       # 완료된 에피소드 [{frac, source[T,7], target[T,7], joints[T,J]}]
    cur = {"source": [], "target": [], "joints": []}

    for t in range(args_cli.eval_steps):
        # 현재(pre-step) 상태 기록 — env.step 이 done env 를 내부 reset 하므로 스텝 전에 캡처.
        cur["source"].append(_cup_pose(_raw_env.cup))
        cur["target"].append(_cup_pose(_raw_env.left_target_cup))
        cur["joints"].append(_raw_env.robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32))

        with torch.inference_mode():
            obs_t = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_t, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)
            if agent.is_rnn and agent.states is not None and len(dones) > 0:
                for s in agent.states:
                    s[:, dones, :] = 0.0

        if bool(dones[0].item()) if hasattr(dones[0], "item") else bool(dones[0]):
            frac = float(_to_np("_last_done_bead")[0])
            spill = float(_to_np("_last_done_spill")[0])
            episodes.append({
                "frac": frac, "spill": spill,
                "source": np.asarray(cur["source"], dtype=np.float32),
                "target": np.asarray(cur["target"], dtype=np.float32),
                "joints": np.asarray(cur["joints"], dtype=np.float32),
            })
            cur = {"source": [], "target": [], "joints": []}
            if len(episodes) % 5 == 0:
                print(f"[REC] 수집 {len(episodes)}/{collect} (마지막 frac={frac:.2f})", flush=True)
            if len(episodes) >= collect:
                break

    if not episodes:
        raise SystemExit("[REC] 완료된 에피소드가 없습니다 (eval_steps 를 늘리세요).")

    # bead frac 상위 선별
    episodes.sort(key=lambda e: -e["frac"])
    best = episodes[: args_cli.record_episodes]

    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args_cli.record_out or os.path.join(log_dir, f"pour_traj_{stamp}.hdf5")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with h5py.File(out_path, "w") as f:
        f.attrs["dt"] = step_dt
        f.attrs["num_beads"] = num_beads
        f.attrs["env_origin"] = env_origin
        f.attrs["robot_root"] = _robot_root()
        f.attrs["robot_usd"] = robot_usd
        f.attrs["task"] = task_name
        f.attrs["checkpoint"] = os.path.basename(resume_path)
        f.attrs["joint_names"] = np.array(joint_names, dtype="S64")
        f.attrs["body_names"] = np.array(body_names, dtype="S64")
        f.attrs["n_episodes"] = len(best)
        for i, e in enumerate(best):
            g = f.create_group(f"ep_{i:03d}")
            g.create_dataset("source_pose", data=e["source"], compression="gzip")  # [T,7] pos(3)+quat wxyz(4), env-rel, meters
            g.create_dataset("target_pose", data=e["target"], compression="gzip")
            g.create_dataset("joint_pos", data=e["joints"], compression="gzip")     # [T,J]
            g.attrs["bead_frac"] = e["frac"]
            g.attrs["bead_spill"] = e["spill"]
            g.attrs["n_steps"] = int(e["source"].shape[0])

    fracs = [e["frac"] for e in best]
    print("\n" + "=" * 60)
    print(f"[REC] 저장 {len(best)} 에피소드 → {out_path}")
    print(f"[REC] bead frac: 평균 {np.mean(fracs):.3f}  범위 [{min(fracs):.2f}, {max(fracs):.2f}]  "
          f"길이 평균 {np.mean([e['source'].shape[0] for e in best]):.0f} step")
    print("=" * 60)


    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

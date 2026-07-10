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

"""Script to play a checkpoint if an RL agent from RL-Games."""

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
    "--cam_eye", type=str, default=None,
    help="Viewer camera position 'x,y,z' (env-local). pour 태스크는 기본 근접뷰 자동 적용.",
)
parser.add_argument(
    "--cam_lookat", type=str, default=None,
    help="Viewer camera lookat 'x,y,z' (env-local).",
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

import yaml as _yaml


class _RunCfgYamlLoader(_yaml.FullLoader):
    """logged env.yaml 전용 loader.

    EventCfg의 SceneEntityCfg 기본값 slice(None)이
    `!!python/object/apply:builtins.slice` 태그로 덤프되는데 FullLoader가
    거부하므로 해당 태그만 명시적으로 복원한다.
    """


_RunCfgYamlLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice",
    lambda loader, node: slice(*loader.construct_sequence(node, deep=True)),
)


def _load_run_yaml(path: str):
    try:
        return load_yaml(path)
    except _yaml.constructor.ConstructorError:
        with open(path) as f:
            return _yaml.load(f, Loader=_RunCfgYamlLoader)


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
        logged_env = _rebase_logged_paths(_load_run_yaml(env_yaml), workspace_root=workspace_root)
        _apply_logged_env_cfg(env_cfg, logged_env)
        print(f"[INFO] Restored playback env cfg from: {env_yaml}")
    else:
        print(f"[WARN] Run env cfg not found; using current source cfg: {env_yaml}")

    if os.path.exists(agent_yaml):
        logged_agent = _rebase_logged_paths(_load_run_yaml(agent_yaml), workspace_root=workspace_root)
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

    # pour_point(빨강) 마커 off: train.py가 학습 시 True로 강제 → logged env.yaml에 true가 박혀
    #   복원 단계에서 다시 켜진다. 복원 이후인 여기서 명시적으로 꺼야 렌더/비디오에 안 나온다.
    if hasattr(env_cfg, "enable_visual_markers") and env_cfg.enable_visual_markers:
        env_cfg.enable_visual_markers = False
        print("[INFO] pour_point 마커 표시 off (playback).")
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # 뷰어 카메라: logged env.yaml이 기본(먼) 카메라를 복원하므로 복원 이후 여기서 덮어쓴다.
    #   pour 태스크는 컵/손이 작아 기본 7.5m 뷰가 너무 멀다 → env-따라가는 근접뷰 기본 적용.
    def _parse_xyz(text):
        parts = [float(v) for v in text.split(",")]
        if len(parts) != 3:
            raise ValueError(f"expected 'x,y,z', got: {text}")
        return tuple(parts)

    task_name = args_cli.task.split(":")[-1]
    is_pour = "pour" in task_name.lower()
    cam_eye = _parse_xyz(args_cli.cam_eye) if args_cli.cam_eye else (None if not is_pour else (1.4, 1.0, 0.9))
    cam_lookat = _parse_xyz(args_cli.cam_lookat) if args_cli.cam_lookat else (None if not is_pour else (0.3, -0.15, 0.45))
    if cam_eye is not None and hasattr(env_cfg, "viewer"):
        env_cfg.viewer.eye = cam_eye
        env_cfg.viewer.lookat = cam_lookat
        # env 0 원점 기준 상대좌표로 근접뷰 고정 (world 원점 기준이면 여전히 멀어짐)
        if hasattr(env_cfg.viewer, "origin_type"):
            env_cfg.viewer.origin_type = "env"
        if hasattr(env_cfg.viewer, "env_index"):
            env_cfg.viewer.env_index = 0
        print(f"[INFO] viewer camera override: eye={cam_eye} lookat={cam_lookat} (origin=env0)")

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

    # pour_point(빨강) 마커는 렌더에 표시하지 않는다 (cfg 기본 enable_visual_markers=False 존중).

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # grasp_v2: hydra 가 MultiAssetSpawnerCfg.assets_cfg(중첩 SpawnerCfg list)를 dict 로
    # 직렬화해 spawn 시 'dict has no attribute func' 로 깨지는 문제 → gym.make 직전 복원.
    if str(args_cli.task).startswith("open-tesol_r_grasp_v2"):
        from openarm.tesollo.right.grasp_v2.grasp_right_env_cfg import _GRASP_OBJECT_SPAWN
        env_cfg.cup_cfg.spawn = _GRASP_OBJECT_SPAWN
    elif str(args_cli.task).startswith("open-rh56f1_r_grasp_v2"):
        from openarm.rh56f1.right.grasp_v2.grasp_right_env_cfg import _GRASP_OBJECT_SPAWN
        env_cfg.cup_cfg.spawn = _GRASP_OBJECT_SPAWN

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

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    _episode_step = 0
    _episode_buf = []
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    _mimic_meas = {"n": 0, "drive": {}, "mimic_act": {}, "mimic_tgt": {}}  # 파지 중 mimic 추종 측정
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)

            # === HOLD 안정성 측정 (결정론 eval: lift_started 중 cup_ang/lin_vel) ===
            try:
                _re2 = env.unwrapped
                if hasattr(_re2, "env"):
                    _re2 = _re2.env.unwrapped
                _lifted = (_re2.object_pos[:, 2] - _re2.object_init_pos[:, 2]) > 0.03
                _gripped = _re2.num_contacts_buf >= 2
                _holding = _re2._lift_started_buf & _lifted & _gripped   # 실제로 들고+잡은 env만
                if bool(_holding.any()) and not args_cli.video:
                    _w = _re2.cup.data.root_ang_vel_w[_holding]           # (M,3) world 컵 각속도
                    _pidx = _re2.palm_body_index
                    _pw = _re2.robot.data.body_ang_vel_w[_holding, _pidx] # (M,3) palm 각속도
                    _rel = (_w - _pw).norm(dim=-1).mean().item()          # 상대(그립 슬립/구름)
                    _av = _w.norm(dim=-1).mean().item()
                    _mimic_meas.setdefault("hold_angvel", []).append(_av)
                    _mimic_meas.setdefault("hold_rel", []).append(_rel)
                    _mimic_meas.setdefault("hold_linvel", []).append(
                        _re2.cup.data.root_lin_vel_w[_holding].norm(dim=-1).mean().item())
                    # env0 (영상 env) 개별 — 들고 있으면
                    if bool(_holding[0]):
                        _av0 = _re2.cup.data.root_ang_vel_w[0].norm().item()
                        _mimic_meas.setdefault("hold_angvel_env0", []).append(_av0)
                    if len(_mimic_meas["hold_angvel"]) >= 120:
                        import numpy as _np2, os as _os2
                        _hav = _np2.mean(_mimic_meas["hold_angvel"]); _hlv = _np2.mean(_mimic_meas["hold_linvel"])
                        _hav_med = float(_np2.median(_mimic_meas["hold_angvel"]))
                        _e0 = _mimic_meas.get("hold_angvel_env0", [])
                        _e0m = float(_np2.median(_e0)) if _e0 else -1.0
                        _relm = float(_np2.median(_mimic_meas.get("hold_rel", [0])))
                        print("\n" + "HOLDSTAB" + "=" * 55)
                        print(f"HOLD 안정성 (결정론 eval, 실제 들고+잡은 env, {len(_mimic_meas['hold_angvel'])} frame)")
                        print(f"  world cup_ang_vel  median={_hav_med:.3f} (팔 동작+슬립 합산)")
                        print(f"  ★ 상대(컵-palm) 각속도 median={_relm:.3f} = 그립 안 실제 구름/슬립")
                        print(f"  cup_lin_vel  mean={_hlv:.4f}")
                        print(f"  → {'그립 FIRM(안 구름), world 회전은 팔 동작 = 로그오진' if _relm < 0.5 else '그립서 실제 구름/슬립 = 로그 맞음'}")
                        print("HOLDSTAB" + "=" * 55, flush=True)
                        _os2._exit(0)
            except SystemExit:
                raise
            except Exception:
                pass

            # === mimic 추종 측정 (실제 파지 중 drive vs mimic) ===
            try:
                _re = env.unwrapped
                if hasattr(_re, "env"):
                    _re = _re.env.unwrapped
                _rb = _re.robot
                _mi = _re._hand_mimic_dof_indices
                _hi = _re.hand_dof_indices
                _src = _re._hand_mimic_src_idx
                _mult = _re._hand_mimic_mult
                _dp = _rb.data.joint_pos[:, _hi]                 # (N,6) drive
                _mp = _rb.data.joint_pos[:, _mi]                 # (N,6) mimic actual
                _mt = _dp[:, _src] * _mult.unsqueeze(0)          # mimic 기대(=drive×mult)
                _grasp = (_re.num_contacts_buf >= 2)             # 파지 중 env
                if bool(_grasp.any()):
                    _mimic_meas["n"] += int(_grasp.sum())
                    _mimic_meas.setdefault("frames", 0)
                    _mimic_meas["frames"] += 1
                    # mimic 조인트 순서: [thumb_3, thumb_4, index_2, middle_2, ring_2, pinky_2]
                    for _nm, _k in [("index_2", 2), ("middle_2", 3), ("ring_2", 4), ("pinky_2", 5), ("thumb_4", 1)]:
                        _drv = _dp[_grasp, _src[_k]].mean().item()
                        _act = _mp[_grasp, _k].mean().item()
                        _tgt = _mt[_grasp, _k].mean().item()
                        _mimic_meas["drive"].setdefault(_nm, []).append(_drv)
                        _mimic_meas["mimic_act"].setdefault(_nm, []).append(_act)
                        _mimic_meas["mimic_tgt"].setdefault(_nm, []).append(_tgt)
                    # hold(lift_started) 중 컵 각속도/선속도 — 결정론적 eval서 실제 홀드가 안정한지
                    try:
                        _ls = _re._lift_started_buf
                        if bool(_ls.any()):
                            _av = _re.cup.data.root_ang_vel_w[_ls].norm(dim=-1).mean().item()
                            _lv = _re.cup.data.root_lin_vel_w[_ls].norm(dim=-1).mean().item()
                            _mimic_meas.setdefault("hold_angvel", []).append(_av)
                            _mimic_meas.setdefault("hold_linvel", []).append(_lv)
                    except Exception:
                        pass
                    # 충분히 모이면 즉시 출력 후 강제 종료 (play.py loop-break가 pour continue로 무효)
                    # video 렌더 시엔 끝까지 돌려야 하므로 조기종료 안 함.
                    if _mimic_meas["frames"] >= 60 and not args_cli.video and _mimic_meas.get("_want_mimic_exit", False):
                        import numpy as _np, os as _os
                        print("\n" + "MIMICSUMMARY" + "=" * 55)
                        print("MIMIC 추종 측정 (실제 파지 중, num_contacts>=2 env 평균)")
                        print(f"  파지 프레임: {_mimic_meas['frames']}, env·프레임 누적: {_mimic_meas['n']}")
                        print(f"  {'조인트':10s} {'drive':>8s} {'기대(xmult)':>12s} {'실제mimic':>10s} {'gap':>8s}  판정")
                        for _n2 in ["index_2", "middle_2", "ring_2", "pinky_2", "thumb_4"]:
                            _d = _np.mean(_mimic_meas["drive"][_n2]); _t = _np.mean(_mimic_meas["mimic_tgt"][_n2]); _a = _np.mean(_mimic_meas["mimic_act"][_n2]); _g = _a - _t
                            _v = "MOT-FOLLOW" if _g < -0.25 else ("FOLLOW" if abs(_g) < 0.25 else "OVER")
                            print(f"  {_n2:10s} {_d:8.3f} {_t:12.3f} {_a:10.3f} {_g:+8.3f}  {_v}")
                        if _mimic_meas.get("hold_angvel"):
                            _hav = _np.mean(_mimic_meas["hold_angvel"]); _hlv = _np.mean(_mimic_meas["hold_linvel"])
                            print(f"  [HOLD 결정론 eval] cup_ang_vel={_hav:.3f} (임계 0.5), cup_lin_vel={_hlv:.4f} (임계 0.04)")
                            print(f"    → {'안정(홀드 OK, 학습 1.67은 노이즈/artifact)' if _hav < 0.5 else '실제 굴림(홀드 불안정)'}")
                        print("MIMICSUMMARY" + "=" * 55, flush=True)
                        _os._exit(0)
            except SystemExit:
                raise
            except Exception:
                pass

            # --- 진단 로깅 (env 0 기준) ---
            # 주의: env.step()은 done인 env를 내부에서 자동 reset한다. done frame의 self.* 는
            #   reset 후 값(비드 hidden=z-10, bead_in=0)이므로, done이 아닌 frame만 기록하고
            #   done 시점엔 직전까지 쌓인 buf(=done 직전 유효 frame)를 출력한다.
            _raw_env = env.unwrapped
            if hasattr(_raw_env, 'env'):
                _raw_env = _raw_env.env.unwrapped
            _done0 = (len(dones) > 0 and bool(dones[0]))

            if agent.is_rnn and agent.states is not None and len(dones) > 0:
                for s in agent.states:
                    s[:, dones, :] = 0.0

            if not _done0:
                def _g0(attr):
                    """env 0의 텐서 값을 python(float/list)으로. 없으면 None."""
                    t = getattr(_raw_env, attr, None)
                    if t is None:
                        return None
                    try:
                        v = t[0]
                        return v.cpu().tolist() if hasattr(v, "cpu") else float(v)
                    except Exception:
                        return None

                try:
                    _origin = _raw_env.scene.env_origins[0].cpu().tolist()
                except Exception:
                    _origin = [0.0, 0.0, 0.0]

                _pp = _g0("_source_pour_point_w")
                _to = _g0("_target_opening_w")
                _sud = _g0("_source_up_dot_world")
                _arm_idx = getattr(_raw_env, "arm_dof_indices", None)
                _jpos = None
                if _arm_idx is not None:
                    try:
                        _jpos = _raw_env.robot.data.joint_pos[0, _arm_idx].cpu().tolist()
                    except Exception:
                        _jpos = None

                # 비드 per-bead local 좌표 (done 아닌 유효 frame에서 계산)
                _bead_rows = None
                _beads = getattr(_raw_env, "beads", None)
                if _beads is not None:
                    try:
                        from isaaclab.utils.math import quat_apply_inverse as _qai
                        bp = _beads.data.object_pos_w[0]
                        kk = bp.shape[0]
                        cq = _raw_env.cup.data.root_quat_w[0].unsqueeze(0).expand(kk, -1)
                        cp = _raw_env.cup.data.root_pos_w[0].unsqueeze(0)
                        sl = _qai(cq, bp - cp)
                        lq = _raw_env.left_target_cup.data.root_quat_w[0].unsqueeze(0).expand(kk, -1)
                        lp = _raw_env.left_target_cup.data.root_pos_w[0].unsqueeze(0)
                        tl = _qai(lq, bp - lp)
                        sxy = sl[:, :2].norm(dim=-1); txy = tl[:, :2].norm(dim=-1)
                        _bead_rows = [
                            (sxy[bi].item(), sl[bi, 2].item(), txy[bi].item(), tl[bi, 2].item(), bp[bi, 2].item())
                            for bi in range(kk)
                        ]
                    except Exception:
                        _bead_rows = None

                _episode_buf.append({
                    "pp": [_pp[i] - _origin[i] for i in range(3)] if _pp else None,
                    "to": [_to[i] - _origin[i] for i in range(3)] if _to else None,
                    "cup_up_dot": _sud,
                    "tilt_deg": math.degrees(math.acos(max(-1.0, min(1.0, _sud)))) if _sud is not None else None,
                    "mouth_xy": _g0("_mouth_xy_distance"),
                    "mouth_z": _g0("_mouth_z_clearance"),
                    "bead_in": _g0("_bead_in_target_fraction"),
                    "bead_src": _g0("_bead_in_source_fraction"),
                    "spill": _g0("_spill_ratio"),
                    "clamp_xy": _g0("_palm_clamp_viol_xy"),
                    "clamp_z": _g0("_palm_clamp_viol_z"),
                    "introt_gate": _g0("_internal_rot_gate"),
                    "rim_cos": _g0("_rim_facing_cos"),
                    "joints": _jpos,
                    "beads": _bead_rows,
                })
            elif _episode_buf:
                # 에피소드 종료: buf 마지막 = done 직전 유효 frame
                tail = _episode_buf[-100:]
                offset = len(_episode_buf) - len(tail)
                n = len(_episode_buf)
                print(f"\n=== episode done (유효 frame {n}개, done 직전 기준) — last {len(tail)} ===")

                def _f(v, p="+.4f"):
                    return ("{:" + p + "}").format(v) if v is not None else "   --  "

                # [1] pour/bead 진단 표
                print(f"{'frame':>6} {'pp_x':>8} {'pp_y':>8} {'pp_z':>8} "
                      f"{'cupUpDot':>8} {'tilt°':>6} {'mouthXY':>8} {'mouthZ':>8} "
                      f"{'beadIn':>7} {'beadSrc':>7} {'spill':>6} {'clampXY':>8} {'clampZ':>8} "
                      f"{'introtG':>8} {'rimCos':>8}")
                for fi, r in enumerate(tail):
                    pp = r["pp"] or [None, None, None]
                    print(f"{offset+fi:>6} {_f(pp[0]):>8} {_f(pp[1]):>8} {_f(pp[2]):>8} "
                          f"{_f(r['cup_up_dot'],'+.3f'):>8} {_f(r['tilt_deg'],'5.1f'):>6} "
                          f"{_f(r['mouth_xy'],'.4f'):>8} {_f(r['mouth_z'],'+.4f'):>8} "
                          f"{_f(r['bead_in'],'.3f'):>7} {_f(r['bead_src'],'.3f'):>7} "
                          f"{_f(r['spill'],'.3f'):>6} {_f(r['clamp_xy'],'.4f'):>8} {_f(r['clamp_z'],'.4f'):>8} "
                          f"{_f(r['introt_gate'],'.3f'):>8} {_f(r['rim_cos'],'+.3f'):>8}")

                # [2] joint 표
                if any(r["joints"] for r in tail):
                    print(f"\n{'frame':>6}  {'j0':>7} {'j1':>7} {'j2':>7} {'j3':>7} {'j4':>7} {'j5':>7} {'j6':>7}")
                    for fi, r in enumerate(tail):
                        jv = r["joints"]
                        if jv is None:
                            continue
                        vals = " ".join(f"{v:+.4f}" for v in jv[:7])
                        print(f"{offset+fi:>6}  {vals}")

                # [3] 마지막 유효 frame 요약
                last = tail[-1]
                print(f"\n--- final (done 직전) frame summary ---")
                print(f"  tilt={_f(last['tilt_deg'],'5.1f')}°  cup_up_dot={_f(last['cup_up_dot'],'+.3f')}  "
                      f"mouth_xy={_f(last['mouth_xy'],'.4f')}  mouth_z={_f(last['mouth_z'],'+.4f')}")
                print(f"  bead_in_target={_f(last['bead_in'],'.3f')}  bead_in_source={_f(last['bead_src'],'.3f')}  "
                      f"spill={_f(last['spill'],'.3f')}")
                # palm 위치 박스 클램프(rim-pivot hinge 파손) + joint 한계 포화도
                print(f"  palm_clamp_viol_xy={_f(last['clamp_xy'],'.4f')}  palm_clamp_viol_z={_f(last['clamp_z'],'.4f')}"
                      f"   (>0 → rim-pivot hinge 기계적 파손 = palm 박스가 틸트 벽)")
                print(f"  internal_rot_gate={_f(last['introt_gate'],'.3f')}  rim_facing_cos={_f(last['rim_cos'],'+.3f')}"
                      f"   (gate가 깊은 tilt에서 급락 → rot_dir 곱이 r_tilt를 깎아 tilt 정지)")
                if last.get("joints"):
                    _jl = [(-1.40, 3.49), (-0.17, 3.32), (-1.57, 1.57), (-1e-6, 2.44),
                           (-1.57, 1.57), (-0.79, 0.79), (-1.57, 1.57)]
                    _sat = [max(j / _up, j / _lo) for j, (_lo, _up) in zip(last["joints"][:7], _jl)]
                    print("  joint_sat(|j|/limit, 1.0=포화): "
                          + "  ".join(f"j{k + 1}={s:+.2f}" for k, s in enumerate(_sat)))

                # [4] 비드별 로컬좌표 분포 (done 직전 유효 frame 기준)
                _cfg = _raw_env.cfg
                rows = last.get("beads")
                if rows:
                    print(f"\n--- 비드 분포 (경계 r<{_cfg.source_inner_radius} "
                          f"z∈[{_cfg.source_inside_z_min},{_cfg.source_inside_z_max}]) ---")
                    print(f"{'bead':>4} {'srcXY':>7} {'srcZ':>8} {'inSrc':>5} "
                          f"{'tgtXY':>7} {'tgtZ':>8} {'inTgt':>5} {'status':>10}")
                    n_in_src = n_in_tgt = n_void = n_hidden = 0
                    for bi, (sxv, szv, txv, tzv, wz) in enumerate(rows):
                        if wz < -5.0:
                            n_hidden += 1
                            continue
                        insrc = (sxv <= _cfg.source_inner_radius
                                 and _cfg.source_inside_z_min <= szv <= _cfg.source_inside_z_max)
                        intgt = (txv <= _cfg.target_inner_radius
                                 and _cfg.target_inside_z_min <= tzv <= _cfg.target_inside_z_max)
                        spl = (not insrc) and (tzv < _cfg.target_inside_z_min)
                        if insrc: st = "SOURCE"; n_in_src += 1
                        elif intgt: st = "TARGET"; n_in_tgt += 1
                        elif spl: st = "spill"
                        else: st = "**VOID**"; n_void += 1
                        print(f"{bi:>4} {sxv:>7.4f} {szv:>+8.4f} {str(insrc):>5} "
                              f"{txv:>7.4f} {tzv:>+8.4f} {str(intgt):>5} {st:>10}")
                    print(f"  요약: SOURCE={n_in_src} TARGET={n_in_tgt} VOID(사각지대)={n_void} hidden={n_hidden}")
                _episode_buf = []

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # === mimic 추종 측정 요약 ===
    if _mimic_meas["n"] > 0:
        import numpy as _np
        print("\n" + "=" * 66)
        print("MIMIC 추종 측정 (실제 파지 중, num_contacts>=2 env 평균)")
        print(f"  파지 프레임·env 누적: {_mimic_meas['n']}")
        print(f"  {'조인트':10s} {'drive':>8s} {'기대(×mult)':>12s} {'실제mimic':>10s} {'gap':>8s}  판정")
        for _nm in ["index_2", "middle_2", "ring_2", "pinky_2", "thumb_4"]:
            if _nm in _mimic_meas["drive"]:
                _d = _np.mean(_mimic_meas["drive"][_nm])
                _t = _np.mean(_mimic_meas["mimic_tgt"][_nm])
                _a = _np.mean(_mimic_meas["mimic_act"][_nm])
                _gap = _a - _t
                _verdict = "◄못따라감" if _gap < -0.25 else ("추종" if abs(_gap) < 0.25 else "초과")
                print(f"  {_nm:10s} {_d:8.3f} {_t:12.3f} {_a:10.3f} {_gap:+8.3f}  {_verdict}")
        print("=" * 66)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

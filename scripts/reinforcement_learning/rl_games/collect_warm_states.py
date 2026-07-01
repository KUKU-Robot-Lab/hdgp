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

"""Dedicated entry point for warm-state HDF5 collection from a trained checkpoint.

play.py 는 eval/render 전용이다. warm-state 수집은 관심사가 다르고, play.py 의
logged-cfg 복원이 ``env.enable_warm_state_export`` 같은 설정을 덮어써 수집을 조용히
무력화하는 사고가 있었다(2026-06-30). 이 스크립트는 수집만을 위해:

  1. 명시적 --checkpoint 로 정책/cfg 로드 (자동 체크포인트 탐색 없음).
  2. 학습 run 의 logged env.yaml 복원 후, warm-state export 설정을 **강제로 주입**
     (복원이 덮어쓰지 못하도록 항상 마지막에 설정).
  3. deterministic rollout 을 돌려 env 가 성공 상태를 누적, target_count 도달 시
     env 가 HDF5 를 원자적으로 1회 기록한다.
  4. 출력 HDF5 가 생기면 즉시 종료(GPU 누수 방지).

video/pour 진단 로깅은 일절 없다.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Collect grasp success warm-states into an HDF5 cache from a checkpoint."
)
parser.add_argument("--task", type=str, required=True, help="Play task id (e.g. open-rh56f1_r_grasp_v1-play-lstm).")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint (.pth).")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="RL agent configuration entry point."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument(
    "--disable_adr",
    action="store_true",
    default=False,
    help="Disable ADR so collection uses the nominal (non-randomized-difficulty) regime.",
)
# warm-state export (first-class CLI — NOT hydra env.* so logged-cfg restore cannot clobber it)
parser.add_argument("--warm_export_path", type=str, required=True, help="Output HDF5 path.")
parser.add_argument("--warm_target_count", type=int, default=2048, help="Number of success states to collect.")
parser.add_argument(
    "--warm_success_source",
    type=str,
    default=None,
    help="Override warm_state_success_source if the env exposes it (e.g. stage/lift/stabilize).",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=200000,
    help="Safety cap on policy steps before aborting if the cache never fills.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import importlib
import math
import os
from pathlib import Path
import re
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common import a2c_common
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import load_yaml

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
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


def _checkpoint_sort_key(path: Path) -> tuple[float, int, float]:
    """Prefer higher reward, then later epoch, then newer mtime for prefix matches."""
    match = re.search(r"_ep_(\d+)_rew_([-+]?\d+(?:\.\d+)?)", path.name)
    if match:
        return (float(match.group(2)), int(match.group(1)), path.stat().st_mtime)
    return (float("-inf"), -1, path.stat().st_mtime)


def _resolve_checkpoint_path(checkpoint: str) -> str:
    """Resolve a checkpoint path or a unique truncated-prefix CLI input."""
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
    """Use params saved next to the checkpoint so the policy runs in its training regime."""
    run_dir = os.path.dirname(os.path.dirname(resume_path))
    params_dir = os.path.join(run_dir, "params")
    env_yaml = os.path.join(params_dir, "env.yaml")
    agent_yaml = os.path.join(params_dir, "agent.yaml")

    if os.path.exists(env_yaml):
        logged_env = _rebase_logged_paths(load_yaml(env_yaml), workspace_root=workspace_root)
        _apply_logged_env_cfg(env_cfg, logged_env)
        print(f"[INFO] Restored env cfg from: {env_yaml}")
    else:
        print(f"[WARN] Run env cfg not found; using current source cfg: {env_yaml}")

    if os.path.exists(agent_yaml):
        logged_agent = _rebase_logged_paths(load_yaml(agent_yaml), workspace_root=workspace_root)
        if isinstance(logged_agent, dict) and "params" in logged_agent:
            print(f"[INFO] Restored agent cfg from: {agent_yaml}")
            return logged_agent
        print(f"[WARN] Ignoring malformed run agent cfg: {agent_yaml}")
    else:
        print(f"[WARN] Run agent cfg not found; using current source cfg: {agent_yaml}")
    return agent_cfg


def _force_warm_state_export_cfg(env_cfg) -> None:
    """Force warm-state export ON after cfg restore (restore must not clobber it).

    play.py 사고 재발 방지의 핵심: 이 설정은 hydra `env.*` 가 아니라 CLI 인자로 받아
    logged-cfg 복원 **이후** 강제 주입한다.
    """
    if not hasattr(env_cfg, "enable_warm_state_export"):
        raise RuntimeError(
            "env_cfg has no 'enable_warm_state_export' — this task does not support warm-state export."
        )
    env_cfg.enable_warm_state_export = True
    env_cfg.warm_state_export_path = args_cli.warm_export_path
    if hasattr(env_cfg, "warm_state_target_count"):
        env_cfg.warm_state_target_count = args_cli.warm_target_count
    if args_cli.warm_success_source is not None and hasattr(env_cfg, "warm_state_success_source"):
        env_cfg.warm_state_success_source = args_cli.warm_success_source
    src = getattr(env_cfg, "warm_state_success_source", None)
    print(
        "[INFO] warm-state export forced ON: "
        f"path={env_cfg.warm_state_export_path} target={args_cli.warm_target_count} source={src}"
    )


def _apply_env_overrides(env_cfg) -> None:
    """Apply CLI playback-style overrides after cfg restore."""
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.disable_adr:
        for adr_attr in (
            "enable_adr",
            "enable_noise_adr",
            "enable_bead_count_adr",
            "enable_success_adr",
            "enable_spill_adr",
        ):
            if hasattr(env_cfg, adr_attr):
                setattr(env_cfg, adr_attr, False)


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
            self.vec_env.set_env_state(weights.get("env_state", None))

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
    except Exception as exc:  # noqa: BLE001 - keep non-v4 collection usable.
        print(f"[WARN] Unable to import recurrent gate for player restore: {exc}")
        return
    actor_obs_shape = agent.model.obs_shape
    actor_obs_dim = int(actor_obs_shape[0] if isinstance(actor_obs_shape, (tuple, list)) else actor_obs_shape)
    recurrent_gate.install_recurrent_gate(agent.model.a2c_network, obs_dim=actor_obs_dim)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: dict):
    """Roll out a deterministic policy until the warm-state cache fills."""
    _patch_optimizer_restore()

    if args_cli.seed is not None:
        agent_cfg["params"]["seed"] = args_cli.seed
    env_cfg.seed = agent_cfg["params"]["seed"]

    resume_path = _resolve_checkpoint_path(args_cli.checkpoint)
    print(f"[INFO] Collecting warm-states with checkpoint: {resume_path}")

    sbm_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    workspace_root = os.path.abspath(os.path.join(sbm_root, ".."))

    # 1) 학습 regime 복원 → 2) playback override → 3) warm-state export 강제(마지막).
    agent_cfg = _restore_run_cfg_if_available(
        env_cfg, agent_cfg, resume_path=resume_path, workspace_root=workspace_root
    )
    env_cfg.seed = agent_cfg["params"]["seed"]
    _apply_env_overrides(env_cfg)
    _force_warm_state_export_cfg(env_cfg)

    out_path = Path(args_cli.warm_export_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    _install_player_recurrent_gate(agent, agent_cfg)
    agent.restore(resume_path)
    agent.reset()

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    print(
        f"[collect_warm_states] rollout 시작: num_envs={env.unwrapped.num_envs} "
        f"target={args_cli.warm_target_count} out={out_path}",
        flush=True,
    )

    steps = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)
            if agent.is_rnn and agent.states is not None and len(dones) > 0:
                for s in agent.states:
                    s[:, dones, :] = 0.0

        steps += 1
        if out_path.is_file():
            print(f"[collect_warm_states] DONE: cache written → {out_path} ({steps} steps)", flush=True)
            break
        if steps >= args_cli.max_steps:
            print(
                f"[collect_warm_states] ABORT: {args_cli.max_steps} steps reached without a full cache.",
                flush=True,
            )
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

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

"""Bead 무게별 파지 품질 평가 스크립트 (5g_grasp_right_v8)

[평가 지표]
  - success_rate : 에피소드 성공률
  - grip_intensity: lift 직전(step 480) 시점의 finger action 평균 (−1=open, +1=closed)
  - grip_std      : grip_intensity 표준편차

[기대 패턴 - 정상 적응적 파지]
  bead 0개:  grip_intensity ≈ 0.4~0.6  (가볍게 잡아도 충분)
  bead 10개: grip_intensity ≈ 0.7~1.0  (세게 잡아야 성공)
  → grip_intensity가 bead 수에 따라 증가해야 함

[비정상 패턴 - 항상 최대 파지]
  모든 bead 수에서 grip_intensity ≈ 1.0 고정
  → bead_mass_normalized obs를 활용하지 않고 있음

[실행 예시]
  # 기본 실행 (랜덤 bead 0~10개, 500 에피소드)
  ./isaaclab.sh -p ../hdgp/scripts/reinforcement_learning/rl_games/eval_bead.py \
    --task 5g_grasp_right-v8 \
    --checkpoint /path/to/model.pth \
    --num_envs 50 \
    --headless

  # 특정 bead 수 고정 평가
  ./isaaclab.sh -p ... --bead_fixed 5
"""

import argparse
import csv
import math
import os
import random
import re
import sys
from collections import defaultdict

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Bead 무게별 파지 품질 평가 (5g_grasp_right_v8)")
parser.add_argument("--task",       type=str,   default="5g_grasp_right-v8")
parser.add_argument("--checkpoint", type=str,   default=None)
parser.add_argument("--num_envs",   type=int,   default=50)
parser.add_argument("--num_episodes", type=int, default=500, help="총 수집 에피소드 수")
parser.add_argument("--bead_fixed", type=int,   default=None,
                    help="특정 bead 수 고정 (None이면 0~10 랜덤)")
parser.add_argument("--seed",       type=int,   default=42)
parser.add_argument("--output_csv", type=str,   default="eval_bead_results.csv",
                    help="결과 CSV 저장 경로")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- Isaac Lab & task imports ----
import gymnasium as gym
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectRLEnvCfg, DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401


def _force_local_openarm_path() -> str:
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
    raise RuntimeError(f"openarm resolved to unexpected location: {getattr(openarm, '__file__', None)}")
import openarm.tasks  # noqa: F401, E402


def _resolve_log_path(task_name: str) -> str:
    fallback = task_name.replace("-", "_")
    try:
        spec = gym.spec(task_name)
        entry = spec.kwargs.get("env_cfg_entry_point", "")
        if isinstance(entry, str):
            m = re.search(r"\.pipeline\.(?:gripper|hand)\.(left|right|both)\.([A-Za-z0-9_]+)\.", entry)
            if m:
                side, folder = m.group(1), m.group(2)
                sbm_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
                return os.path.join(sbm_root, "log", "rl_games", "pipeline", side, folder)
    except Exception:
        pass
    sbm_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    return os.path.join(sbm_root, "log", "rl_games", "pipeline", "right", fallback)


def _print_table(records: list[dict], bead_max: int = 10) -> None:
    """bead 수(0~10)별로 집계해서 표 출력."""
    bins: dict[int, list] = defaultdict(list)
    for r in records:
        bead_count = round(r["bead_mass"] * bead_max)
        bins[bead_count].append(r)

    header = f"{'bead':>6} | {'n_ep':>5} | {'success%':>9} | {'grip_mean':>10} | {'grip_std':>9} | 판정"
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    prev_grip = None
    for bead_count in sorted(bins.keys()):
        ep_list = bins[bead_count]
        n = len(ep_list)
        success_rate = sum(r["success"] for r in ep_list) / n * 100
        grips = [r["grip"] for r in ep_list]
        grip_mean = sum(grips) / n
        grip_std = (sum((g - grip_mean) ** 2 for g in grips) / max(n - 1, 1)) ** 0.5

        # 적응적 판정: bead가 늘어날수록 grip이 증가하는지
        if prev_grip is None:
            verdict = "  (기준)"
        elif grip_mean > prev_grip + 0.05:
            verdict = "✓ 증가"
        elif grip_mean < prev_grip - 0.05:
            verdict = "✗ 감소"
        else:
            verdict = "  유지"

        print(f"{bead_count:>6} | {n:>5} | {success_rate:>8.1f}% | {grip_mean:>10.4f} | {grip_std:>9.4f} | {verdict}")
        prev_grip = grip_mean

    print(sep)

    # 전체 요약
    all_success = sum(r["success"] for r in records) / len(records) * 100
    all_grips = [r["grip"] for r in records]
    all_grip_mean = sum(all_grips) / len(all_grips)
    print(f"{'전체':>6} | {len(records):>5} | {all_success:>8.1f}% | {all_grip_mean:>10.4f} |")
    print(sep)

    # 적응성 요약
    grip_vals = [
        sum(r["grip"] for r in bins[k]) / len(bins[k])
        for k in sorted(bins.keys())
        if len(bins[k]) > 0
    ]
    if len(grip_vals) >= 2:
        grip_range = max(grip_vals) - min(grip_vals)
        if grip_range > 0.15:
            print(f"\n[결과] 적응적 파지 확인 ✓  (grip 범위: {grip_range:.3f})")
        elif grip_range > 0.05:
            print(f"\n[결과] 약한 적응성 (grip 범위: {grip_range:.3f}) — 더 학습 필요")
        else:
            print(f"\n[결과] 항상 동일한 grip (grip 범위: {grip_range:.3f}) — bead obs 미활용 의심")


def _save_csv(records: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bead_mass", "grip", "success"])
        writer.writeheader()
        writer.writerows(records)
    print(f"[INFO] 결과 저장: {path}")


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    task_name = args_cli.task
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg["params"]["seed"] = args_cli.seed

    # eval 시 ADR 비활성화: ADR counter=0이면 bead_count_max=0 → 항상 0개가 되는 버그 방지
    env_cfg.enable_adr = False

    # bead 수 고정 옵션
    if args_cli.bead_fixed is not None:
        env_cfg.bead_count_min = args_cli.bead_fixed
        env_cfg.bead_count_max = args_cli.bead_fixed
        print(f"[INFO] bead 수 고정: {args_cli.bead_fixed}개")
    else:
        print(f"[INFO] bead 수 랜덤: {env_cfg.bead_count_min}~{env_cfg.bead_count_max}개")

    # 체크포인트 경로 결정
    log_root = _resolve_log_path(task_name)
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root, "test.*", f"{agent_cfg['params']['config']['name']}.pth",
            other_dirs=["nn"],
        )
    print(f"[INFO] 체크포인트: {resume_path}")

    # 환경 생성
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    rl_device  = agent_cfg["params"]["config"]["device"]
    clip_obs   = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_acts  = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_acts)

    vecenv.register("IsaacRlgWrapper",
                    lambda cfg_name, n_actors, **kw: RlGamesGpuEnv(cfg_name, n_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                           "env_creator": lambda **kw: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    # 기반 환경 접근 (eval 기록용)
    base_env = env.unwrapped  # GraspRightEnv
    base_env._eval_records.clear()

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    target_episodes = args_cli.num_episodes
    print(f"[INFO] {target_episodes} 에피소드 수집 시작...")

    while simulation_app.is_running():
        with torch.inference_mode():
            obs_t = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_t, is_deterministic=True)
            obs, _, dones, _ = env.step(actions)

            if isinstance(obs, dict):
                obs = obs["obs"]

            if agent.is_rnn and agent.states is not None:
                for s in agent.states:
                    s[:, dones, :] = 0.0

        n_collected = len(base_env._eval_records)
        if n_collected >= target_episodes:
            print(f"[INFO] {n_collected} 에피소드 수집 완료")
            break

        if n_collected % 50 == 0 and n_collected > 0:
            print(f"  진행: {n_collected}/{target_episodes} 에피소드")

    # ---- 결과 출력 ----
    records = base_env._eval_records[:target_episodes]
    _print_table(records, bead_max=env_cfg.bead_count_max)
    _save_csv(records, args_cli.output_csv)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""left grasp_v2 성공 파지 hand pose 수집기 (tesollo PCA 역추론 원료).

학습 완료 정책(rl_games LSTM)을 결정론으로 rollout 하면서, 에피소드마다
최초 성공(success_flag = in_success_region & valid) 순간의 손 20관절 자세와
물체 인덱스를 기록한다. 출력 npz 는 compute_tesollo_left_grasp_pca.py 의 입력.

실행 (IsaacLab 번들 python 필요):
  /home/user/rl_ws/IsaacLab/isaaclab.sh -p scripts/pca/collect_left_grasp_poses.py \
      --checkpoint log/rl_games/open-tesol/left/grasp-v2/lstm_test1/nn/<ckpt>.pth \
      --num_envs 512 --episodes 3060 --headless
"""
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

REPO = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(description="left grasp_v2 성공 hand pose 수집")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--episodes", type=int, default=3060, help="집계 대상 총 에피소드 수")
parser.add_argument("--adr_level", type=int, default=0, help="ADR 증분 고정(0=클린)")
parser.add_argument("--out", type=str, default=str(REPO / "data" / "left_grasp_poses_lstm_test7.npz"))
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

import openarm  # noqa: F401,E402 — task 등록
from openarm.tesollo.left.grasp_v2.grasp_left_env_cfg import GraspLeftEnvCfg  # noqa: E402

AGENT_YAML = (
    REPO / "source/openarm/openarm/tesollo/left/grasp_v2/config/agents/rl_games_ppo_lstm_cfg.yaml"
)


def main() -> None:
    ckpt = str(Path(args.checkpoint).resolve())
    agent_cfg = yaml.safe_load(AGENT_YAML.read_text())

    # cfg 를 직접 생성 — hydra 직렬화를 거치지 않아 MultiAsset(153종) 이 그대로 살아있다.
    env_cfg = GraspLeftEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = agent_cfg["params"].get("seed", 42)

    env = gym.make("open-tesol_l_grasp_v2-lstm", cfg=env_cfg)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", float("inf"))
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", float("inf"))
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate = agent_cfg["params"]["env"].get("concate_obs_groups", True)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = ckpt
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(ckpt)
    agent.reset()

    raw = env.unwrapped
    if hasattr(raw, "env"):
        raw = raw.env.unwrapped

    # ADR 고정 + 집계 카운터 리셋 (play.py --eval_episodes 와 동일 절차)
    if getattr(raw, "grasp_adr", None) is not None:
        raw.grasp_adr.set_increment(int(args.adr_level))
    raw._total_episodes = 0
    raw._successful_episodes = 0

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    n = raw.num_envs
    captured = torch.zeros(n, dtype=torch.bool, device=raw.device)
    poses: list[np.ndarray] = []
    obj_idx: list[np.ndarray] = []

    while simulation_app.is_running():
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=True)
            obs, _r, dones, _i = env.step(actions)

            # 에피소드 최초 성공 순간의 손 자세 기록
            new_succ = raw.success_flag & ~captured
            if bool(new_succ.any()):
                q_hand = raw.robot.data.joint_pos[new_succ][:, raw.hand_dof_indices]
                poses.append(q_hand.cpu().numpy())
                obj_idx.append(raw.object_idx[new_succ].cpu().numpy())
                captured |= new_succ

            if len(dones) > 0:
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                captured[done_ids] = False
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, done_ids, :] = 0.0

            if raw._total_episodes >= args.episodes:
                break

    q = np.concatenate(poses, axis=0) if poses else np.zeros((0, len(raw.hand_dof_indices)))
    oi = np.concatenate(obj_idx, axis=0) if obj_idx else np.zeros((0,), dtype=np.int64)
    names = np.array(raw._object_names)
    hand_joint_names = np.array([raw.robot.joint_names[j] for j in raw.hand_dof_indices])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        q_hand_left=q.astype(np.float32),
        object_idx=oi.astype(np.int64),
        object_names=names,
        hand_joint_names=hand_joint_names,
        checkpoint=np.array(ckpt),
        adr_level=np.array(args.adr_level),
        episodes=np.array(int(raw._total_episodes)),
    )
    per_obj = np.bincount(oi, minlength=len(names))
    print("COLLECTSUMMARY" + "=" * 50)
    print(f"  샘플 {q.shape[0]}개 (에피소드 {raw._total_episodes}, 성공률 "
          f"{raw._successful_episodes / max(raw._total_episodes, 1):.3f})")
    print(f"  물체 커버리지: {(per_obj > 0).sum()}/{len(names)}종, "
          f"물체당 중앙값 {int(np.median(per_obj))}개")
    print(f"  저장: {out}")
    print("COLLECTSUMMARY" + "=" * 50, flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

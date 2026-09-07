"""학습된 정책의 **기하**를 잰다 — approach 고원의 원인 분리.

approach = exp(-s·(palm_to_obj + tip_side_dist)) 은 **합**만 보여준다.
합이 0.29m 로 멈췄을 때 그것이
  (a) palm 은 붙었는데 손끝이 멀다      → 손 자세 문제
  (b) palm 이 멀다                      → 팔 접근 문제
  (c) 손끝이 닿았는데 힘이 임계 미달     → 게이트 임계 문제
  (d) 물리적으로 못 닿는 자세           → 워크스페이스/자세 문제
중 어느 것인지 가른다. 접촉력 **원값**을 찍는 게 핵심(env 로깅에 빠져 있다).

    isaaclab.sh -p .../probe_policy_geometry.py --checkpoint <path> --steps 300
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym      # noqa: E402
import torch                 # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg   # noqa: E402
from rl_games.torch_runner import Runner         # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry               # noqa: E402

import openarm.tasks         # noqa: E402,F401

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
agent_cfg = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")

env = gym.make(args.task, cfg=env_cfg)
raw = env.unwrapped
env = RlGamesVecEnvWrapper(env, args.device, agent_cfg["params"]["env"].get("clip_observations", 5.0),
                           # ★params.config.clip_actions 는 rl_games 내부 플래그(False)다. 래퍼가 쓰는 것은
                           #   params.env.clip_actions(1.0). 잘못 읽으면 Box(0,0) 이 되어 **모든 액션이 0**
                           #   → 지표가 전부 정확히 0.0000 으로 나온다(play.py 와 같은 키를 쓸 것).
                           agent_cfg["params"]["env"].get("clip_actions", 1.0))
gym.vector.register("rlgpu", lambda cfg_name, nenv, **kw: RlGamesGpuEnv(cfg_name, nenv, **kw))
from rl_games.common import env_configurations, vecenv  # noqa: E402
vecenv.register("IsaacRlgWrapper", lambda cn, ne, **kw: env)
env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})

agent_cfg["params"]["config"]["env_info"] = env.get_number_of_agents and None
runner = Runner()
agent_cfg["params"]["config"]["num_actors"] = args.num_envs
runner.load(agent_cfg)
agent = runner.create_player()
agent.restore(args.checkpoint)
agent.reset()

obs = env.reset()
if isinstance(obs, tuple):
    obs = obs[0]

import collections   # noqa: E402
acc = collections.defaultdict(list)
for i in range(args.steps):
    with torch.no_grad():
        act = agent.get_action(obs, is_deterministic=True)
    obs, _, _, _ = env.step(act)[:4]

    obj = raw.object.data.root_pos_w - raw.scene.env_origins
    palm = raw.robot.data.body_pos_w[:, raw.palm_idx] - raw.scene.env_origins
    tips = raw.robot.data.body_pos_w[:, raw._tip_t] - raw.scene.env_origins[:, None, :]
    tipd = (tips - obj[:, None, :]).norm(dim=-1)
    force, wrapped, _, _ = raw._contact()

    acc["palm_dist"].append((palm - obj).norm(dim=-1).mean().item())
    acc["tip_mean"].append(tipd.mean().item())
    acc["tip_min"].append(tipd.min(dim=1).values.mean().item())
    acc["force_max"].append(force.max().item())
    acc["force_mean_best"].append(force.max(dim=1).values.mean().item())
    acc["n_over_1N"].append((force > 1.0).float().sum(dim=1).mean().item())
    acc["n_over_0.1N"].append((force > 0.1).float().sum(dim=1).mean().item())
    acc["palm_z"].append(palm[:, 2].mean().item())
    acc["obj_z"].append(obj[:, 2].mean().item())

last = slice(-args.steps // 3, None)      # 마지막 1/3 구간 평균
print("\n" + "=" * 64)
print("정책 기하 (마지막 1/3 구간 평균)")
print("=" * 64)
def m(k): return sum(acc[k][last]) / len(acc[k][last])
pd, tm, tn = m("palm_dist"), m("tip_mean"), m("tip_min")
print(f"  palm ↔ 물체 거리        {pd:.4f} m")
print(f"  손끝 평균 ↔ 물체        {tm:.4f} m   (approach 식의 tip_side_dist)")
print(f"  손끝 **최소** ↔ 물체     {tn:.4f} m   ← 가장 가까운 손가락")
print(f"  합(palm+tip_mean)      {pd + tm:.4f} m  → approach = exp(-4·합) = {2.718281828**(-4*(pd+tm)):.4f}")
print()
print(f"  접촉력 최대(전 env)      {m('force_max'):.4f} N")
print(f"  env 별 최대 손가락 힘 평균 {m('force_mean_best'):.4f} N")
print(f"  1.0N 초과 손가락 수      {m('n_over_1N'):.3f}   ← 게이트 임계")
print(f"  0.1N 초과 손가락 수      {m('n_over_0.1N'):.3f}   ← 임계를 10배 낮추면")
print()
print(f"  palm z {m('palm_z'):.4f}   물체 z {m('obj_z'):.4f}   (작업면 0.200)")
print()
print("판정 힌트:")
print("  손끝최소 < 0.02 이고 1.0N 초과 = 0 이면 → **게이트 임계 문제**")
print("  손끝최소 > 0.05 이면            → 아직 물리적으로 안 닿음(접근/자세 문제)")
env.close()
app.close()

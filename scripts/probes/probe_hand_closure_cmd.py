"""학습 정책의 손가락별 **폐쇄 지령** 분포 — 어느 손가락을 안 닫는지 직독.

h7 우팔: 350ep 동안 검지·중지·약지 접촉 정확히 0.00, close_bridge 0.00.
정책이 그 관절들을 여는 쪽으로 지령하는지(폐쇄도 0), 닫는데 안 닿는지 가른다.
폐쇄도 = (fabric 손 지령 − 홈)/(닫힘한계 − 홈), 손가락별 유효 관절 평균.
"""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=240)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app
import math, yaml                                  # noqa: E402
from pathlib import Path                           # noqa: E402
import gymnasium as gym                            # noqa: E402
import torch                                       # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg     # noqa: E402
from rl_games.torch_runner import Runner           # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
import openarm.tasks                               # noqa: E402,F401

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
_genv = gym.make(args.task, cfg=env_cfg)
env = _genv.unwrapped
N = args.num_envs
agent_cfg = yaml.safe_load(open(Path(args.checkpoint).parents[1] / "params" / "agent.yaml"))
agent_cfg["params"]["config"]["num_actors"] = N
_ce = agent_cfg["params"].get("env", {})
_wrapped = RlGamesVecEnvWrapper(_genv, env.device, _ce.get("clip_observations", math.inf),
                                _ce.get("clip_actions", math.inf),
                                _ce.get("obs_groups"), _ce.get("concate_obs_groups", True))
vecenv.register("IsaacRlgWrapper", lambda cn, na, **kw: RlGamesGpuEnv(cn, na, **kw))
env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                      "env_creator": lambda **kw: _wrapped})
runner = Runner(); runner.load(agent_cfg)
player = runner.create_player(); player.restore(args.checkpoint); player.reset()
obs = _wrapped.reset()
_ = player.get_batch_size(obs if not isinstance(obs, dict) else obs["obs"], 1)
if player.is_rnn: player.init_rnn()

p = env.profile
fingers = list(p.fingers)
# fabric 손 슬롯 → 손가락 매핑(env 부팅 로직과 동일 규약)
jn = list(env.robot.data.joint_names)
n_arm = p.num_arm_joints
slot_f = [ [f for f in fingers if f in jn[int(env._fab_t[n_arm + i])]][0]
           for i in range(p.num_hand_joints) ]
den = env._close_den; valid = env._close_valid
acc = {f: [] for f in fingers}
act_acc = []
with torch.no_grad():
    for t in range(args.steps):
        o = obs["obs"] if isinstance(obs, dict) else obs
        a = player.get_action(o, is_deterministic=True)
        obs, _, _, _ = _wrapped.step(a)
        if t < args.steps // 2: continue
        close = ((env._fabric_hand_cmd - env._fab_hand_home) / den).clamp(0, 1)
        for i, f in enumerate(slot_f):
            if bool(valid[i]): acc[f].append(close[:, i])
        act_acc.append(a[:, 6:].abs().mean(0))
print("\n" + "=" * 64)
print(f"손가락별 폐쇄 **지령** — {args.task} · {Path(args.checkpoint).name[:40]}")
print("=" * 64)
for f in fingers:
    if acc[f]:
        v = torch.stack(acc[f])
        print(f"  {f:>8s}  폐쇄도 평균 {float(v.mean()):.3f}  p95 {float(v.quantile(0.95)):.3f}")
    else:
        print(f"  {f:>8s}  (유효 관절 없음 — 전부 고정)")
aa = torch.stack(act_acc).mean(0)
print(f"  손 액션 |a| 평균(관절별): {[round(float(x),2) for x in aa]}")
print("=" * 64 + "\n", flush=True)
env.close(); app.close()

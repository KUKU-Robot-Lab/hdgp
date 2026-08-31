"""팁 **지령**이 컵 표면에 가는가 — 학습된 정책의 손끝 커맨드 배치 실측.

08.26 사용자 관찰: "컵 표면에 팁 커맨드 액션이 가야 하는데 이상한 곳에 있음."
마커로는 '이상하다'까지만 보이므로 컵 기준 좌표로 수치화한다.

측정(손가락별, λ 열린 스텝만):
  · 지령: 컵 축 기준 반경 r_cmd, 컵 바닥 기준 높이 h_cmd, 컵 표면까지 거리
  · 실제: 같은 것(실 손끝) — 지령↔실제 차이가 곧 IK 추종오차
  · 액션 박스 안에서 지령이 차지한 위치(박스 중앙 대비) — a=0 쏠림 진단
컵 반경/높이는 자산이 아니라 **접촉 실측**으로 잡는다(형상 상수 미사용 규약):
표면 반경 = 접촉이 실제로 일어난 반경의 중앙값. 접촉이 없으면 스폰 bbox 로 대체.
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import math                                     # noqa: E402
import gymnasium as gym                         # noqa: E402
import torch                                    # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg   # noqa: E402
from rl_games.torch_runner import Runner         # noqa: E402
import openarm.tasks                             # noqa: E402,F401

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
_genv = gym.make(args.task, cfg=env_cfg)
env = _genv.unwrapped
dev = env.device
N, A = args.num_envs, env.cfg.action_space
p = env.profile
fingers = list(p.fingers)

# ── 정책 로드 — play.py 와 **같은 경로**(Runner→create_player→restore).
#   model_builder.build() 직접 호출은 rl_games 버전에 따라 시그니처가 달라 깨진다.
import yaml                                      # noqa: E402
from pathlib import Path                         # noqa: E402
from rl_games.torch_runner import Runner         # noqa: E402

_cfg_path = Path(args.checkpoint).parents[1] / "params" / "agent.yaml"
agent_cfg = yaml.safe_load(open(_cfg_path))
agent_cfg["params"]["config"]["num_actors"] = N
# ★rl_games Runner 는 "rlgpu" 등록을 요구한다(play.py 와 같은 배선). 없으면
#   create_player 가 KeyError: 'rlgpu' 로 죽는다.
from rl_games.common import env_configurations, vecenv          # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
# ★clip/obs_groups 는 params["env"] 에 있다(params["config"] 아님 — 잘못 읽어
#   obs_groups 에 문자열이 들어가 rand() 가 죽었다).
_ce = agent_cfg["params"].get("env", {})
_wrapped = RlGamesVecEnvWrapper(
    _genv, dev, _ce.get("clip_observations", math.inf),
    _ce.get("clip_actions", math.inf),
    _ce.get("obs_groups"), _ce.get("concate_obs_groups", True))
vecenv.register("IsaacRlgWrapper",
                lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                      "env_creator": lambda **kw: _wrapped})
runner = Runner()
runner.load(agent_cfg)
player = runner.create_player()
player.restore(args.checkpoint)
player.reset()
print(f"[정책] {Path(args.checkpoint).name} · rnn={player.is_rnn}", flush=True)

_o0 = _wrapped.reset()
obs = _o0
_ = player.get_batch_size(obs if not isinstance(obs, dict) else obs["obs"], 1)
if player.is_rnn:
    player.init_rnn()
acc_cmd_r, acc_cmd_h, acc_act_r, acc_act_h, acc_box = [], [], [], [], []
contact_r = []
with torch.no_grad():
    for t in range(args.steps):
        o = obs["obs"] if isinstance(obs, dict) else obs
        a = player.get_action(o, is_deterministic=True)
        obs, _, _, _ = _wrapped.step(a)
        if t < args.steps // 2:
            continue
        cup = env.object.data.root_pos_w                      # (N,3)
        cmd_w = env._tip_target.view(N, -1, 3) + env.scene.env_origins[:, None, :]
        act_w = env.robot.data.body_pos_w[:, env._tip_t]
        for arr, src in ((0, cmd_w), (1, act_w)):
            d = src - cup[:, None, :]
            r = d[:, :, :2].norm(dim=-1)                      # 컵 축(수직) 기준 반경
            h = d[:, :, 2]                                    # 컵 원점 기준 높이
            (acc_cmd_r if arr == 0 else acc_act_r).append(r)
            (acc_cmd_h if arr == 0 else acc_act_h).append(h)
        # 액션 박스 안 위치(0=lo, 1=hi) — a=0(박스중앙=0.5) 쏠림 진단
        tl = env._tip_cmd if getattr(env, "_tip_cmd", None) is not None else None
        if tl is not None:
            acc_box.append(((tl - env._tip_lo) / (env._tip_hi - env._tip_lo)).clamp(0, 1))
        c, _, mid, dist = env._contact()
        m = c > float(env.cfg.stage_contact_threshold)
        if bool(m.any()):
            dd = act_w - cup[:, None, :]
            contact_r.append(dd[:, :, :2].norm(dim=-1)[m])

CR = torch.cat(acc_cmd_r); CH = torch.cat(acc_cmd_h)
AR = torch.cat(acc_act_r); AH = torch.cat(acc_act_h)
surf = float(torch.cat(contact_r).median()) if contact_r else float("nan")
print("\n" + "=" * 88)
print(f"팁 지령 배치 — {args.task} · {N}env × {args.steps//2}스텝")
print(f"  컵 표면 반경(접촉 실측 중앙값) = {surf*1000:.0f}mm" if surf == surf
      else "  ★접촉 표본 없음 — 표면 반경 미상")
print("=" * 88)
print(f"  {'손가락':>8s} | {'지령 r':>7s} {'지령 h':>7s} | {'실제 r':>7s} {'실제 h':>7s} | "
      f"{'지령→실제':>9s} | 박스위치(x,y,z)")
for i, f in enumerate(fingers):
    bx = (torch.cat(acc_box)[:, i].mean(dim=0) if acc_box else torch.full((3,), float("nan")))
    gap = float(((CR[:, i] - AR[:, i]) ** 2 + (CH[:, i] - AH[:, i]) ** 2).sqrt().mean())
    print(f"  {f:>8s} | {float(CR[:,i].mean())*1000:6.0f}mm {float(CH[:,i].mean())*1000:+6.0f}mm | "
          f"{float(AR[:,i].mean())*1000:6.0f}mm {float(AH[:,i].mean())*1000:+6.0f}mm | "
          f"{gap*1000:8.0f}mm | "
          + " ".join(f"{float(v):.2f}" for v in bx))
print(f"\n  해석: 지령 r 이 표면 반경({surf*1000:.0f}mm) 근처면 정상. 훨씬 크면 컵 밖 허공,"
      f"\n        훨씬 작으면 컵 내부를 지시(관통 요구). 지령 h 0 부근 = 컵 몸통 높이.")
print(f"  박스위치 0.5 쏠림 = a≈0(무행동) · 0/1 붙음 = 포화")
print("=" * 88 + "\n", flush=True)
env.close(); app.close()

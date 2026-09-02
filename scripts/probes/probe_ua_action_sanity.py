"""★계측기 검증 — 액션이 손바닥을 실제로 옮기는가.

09.03: 스크립트 롤아웃 프로브가 "파지 0/48" 을 냈는데 팔 도달오차가 110~218mm,
리프트 지령 80mm 에 손바닥 실제 상승 2.6mm 였다. 즉 **팔이 안 갔다** — 파지에
대해 아무것도 못 잰 것이다. 결과를 해석하기 전에 계측기를 시험한다.

여기서 확인하는 것:
  · 알려진 palm 델타 지령 → 손바닥이 실제로 그만큼 가는가(축별)
  · 롤아웃 중 종료(terminated/truncated)가 몇 번 나는가 — 리셋되면 측정 무의미
  · 박스 클램프·속도 리미터가 지령을 얼마나 깎는가
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="open-rh_r_grasp_ua-play-lstm")
parser.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.agnostic.tasks.grasp_ua.config  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> int:
    n = 6
    cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=n)
    cfg.scene.num_envs = n
    cfg.enable_events = False
    cfg.enable_adr = False
    cfg.episode_length_s = 120.0
    env = gym.make(args_cli.task, cfg=cfg)
    u = env.unwrapped
    env.reset()
    dev = u.device

    # env 별로 서로 다른 한 축 델타를 준다.
    deltas = torch.tensor([[0.05, 0, 0], [-0.05, 0, 0], [0, 0.05, 0],
                           [0, -0.05, 0], [0, 0, 0.05], [0, 0, -0.05]], device=dev)
    anchor = u._palm_anchor().clone()
    tgt = anchor[:, :3] + deltas
    lo, hi = u._delta_lo, u._delta_hi
    d6 = torch.zeros(n, 6, device=dev); d6[:, :3] = deltas
    a6 = (2.0 * (d6 - lo) / (hi - lo).clamp(min=1e-9) - 1.0).clamp(-1.0, 1.0)
    action = torch.cat([a6, torch.full((n, u.cfg.action_space - 6), -1.0, device=dev)],
                       dim=1)

    p0 = u._env_local(u.robot.data.body_pos_w[:, u.palm_idx]).clone()
    n_term = torch.zeros(n, device=dev)
    for i in range(args_cli.steps):
        _, _, term, trunc, _ = env.step(action)
        n_term += (term | trunc).float()

    p1 = u._env_local(u.robot.data.body_pos_w[:, u.palm_idx])
    print("\n" + "=" * 96, flush=True)
    print(f"[sanity] 앵커 {[round(float(x), 4) for x in anchor[0, :3]]} · "
          f"델타범위 lo {[round(float(x), 3) for x in lo[:3]]} "
          f"hi {[round(float(x), 3) for x in hi[:3]]}", flush=True)
    print(f"[sanity] 박스 lo {[round(float(x), 3) for x in u._box_lo[:3]]} "
          f"hi {[round(float(x), 3) for x in u._box_hi[:3]]}", flush=True)
    for i in range(n):
        got = (p1[i] - p0[i]) * 1000.0
        want = deltas[i] * 1000.0
        # 지령이 박스에 잘렸는지
        clamped = (u.palm_targets[i, :3] - (anchor[i, :3] + deltas[i])).norm() * 1000.0
        print(f"[sanity] 지령 ({want[0]:5.0f},{want[1]:5.0f},{want[2]:5.0f})mm → "
              f"실제 ({got[0]:6.1f},{got[1]:6.1f},{got[2]:6.1f})mm · "
              f"목표클램프 {float(clamped):5.1f}mm · 종료 {int(n_term[i])}회", flush=True)
    print(f"[sanity] fabric joint_err_max {float(u.robot.data.joint_pos.new_zeros(1)):.0f} "
          f"· palm_targets[0] {[round(float(x), 4) for x in u.palm_targets[0, :3]]}",
          flush=True)
    print("=" * 96, flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    import traceback
    try:
        _rc = main()
    except BaseException:
        traceback.print_exc()
        _rc = 3
    _app.close()
    raise SystemExit(_rc)

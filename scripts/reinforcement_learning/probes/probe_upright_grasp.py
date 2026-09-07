"""**수직 palm 으로도 이 컵을 잡고 들 수 있는가** — 보상을 바꾸기 전의 물리 검증.

fab_test9 실측(probe_finger_participation): 정책은 palm 정렬축을 world z 에서 **37.5°**
기울여 접근하고, 그 결과 컵이 13.6° 로 기운 채 들린다. 홈 자세의 palm 정렬축은 0.64°
(수직)이므로 기구학적 강제가 아니라 **정책의 선택**이다. 이 기울기가 up_mul 을 0.77 로
떨어뜨려 lift+success(전체 보상의 84%)에서 상시 22% 를 깎는다.

여기서 답해야 할 것: tilt 페널티를 강화해도 되는가?
  · 수직 palm 으로도 파지·리프트가 되면  → 정책의 국소최적. 페널티 강화가 유효하다.
  · 수직 palm 이면 파지가 깨지면        → 기울기는 파지의 대가. 페널티 강화는
                                          정책이 **파지를 팔아버리게** 만든다.

절차(정책 없이 스크립트): 컵을 손 안에 **수직으로** 놓고 → 자세를 고정한 채 손을 닫고
→ palm 을 +15cm 올린다. 자세별로 접촉·상승·최종 컵 기울기를 잰다.

    isaaclab.sh -p .../probe_upright_grasp.py --num_envs 32
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--close", type=int, default=80)
parser.add_argument("--lift", type=int, default=200)
parser.add_argument("--self_collisions", action="store_true",
                    help="enabled_self_collisions=True 로 재해석해 실행(자산 갱신 후 검증용)")
parser.add_argument("--gravity", action="store_true",
                    help="로봇 중력 ON (기본 cfg 는 disable_gravity=True)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym      # noqa: E402
import torch                 # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg          # noqa: E402
from isaaclab.utils.math import quat_apply              # noqa: E402

import openarm.tasks         # noqa: E402,F401

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
# ★스위치는 **cfg 필드**로만 켠다. `robot_cfg.spawn.*` 을 직접 고치면 env.__init__ 의
#   resolve_cfg 가 robot_cfg 를 재생성하며 조용히 되돌린다(08.22 실측: 중력을 False 로
#   바꿔 로그까지 찍었는데 USD 는 True 였다). 필드는 resolve_cfg 가 읽으므로 살아남는다.
if args.self_collisions or args.gravity:
    from openarm.agnostic.tasks.grasp_lift_fabric.grasp_lift_fabric_env_cfg import resolve_cfg
    env_cfg.enable_self_collisions = bool(args.self_collisions)
    env_cfg.enable_gravity = bool(args.gravity)
    resolve_cfg(env_cfg)
    print(f"[probe] self_collisions={env_cfg.robot_cfg.spawn.articulation_props.enabled_self_collisions}"
          f" · disable_gravity={env_cfg.robot_cfg.spawn.rigid_props.disable_gravity}")
env = gym.make(args.task, cfg=env_cfg).unwrapped
N, A = args.num_envs, env.cfg.action_space
dev = env.device


def palm_axis_deg() -> torch.Tensor:
    """palm 로컬 y(=컵 정렬축, probe_finger_participation 에서 확정) ↔ world z."""
    e = torch.zeros(N, 3, device=dev); e[:, 1] = 1.0
    ax = quat_apply(env.robot.data.body_quat_w[:, env.palm_idx], e)
    wz = torch.zeros(N, 3, device=dev); wz[:, 2] = 1.0
    d = torch.rad2deg(torch.acos((ax * wz).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)))
    return torch.minimum(d, 180.0 - d)


# 자세 케이스: a[3:6] 은 palm euler(zyx). a=0 이 홈(=수직).
CASES = [
    ("홈(수직)",      (0.0, 0.0, 0.0)),
    ("ey +0.5",       (0.0, 0.5, 0.0)),
    ("ey -0.5",       (0.0, -0.5, 0.0)),
    ("ex +0.5",       (0.0, 0.0, 0.5)),
    ("ex -0.5",       (0.0, 0.0, -0.5)),
    ("ez +0.5",       (0.5, 0.0, 0.0)),
]

print("\n" + "=" * 86)
print(f"{'자세':<12s}{'palm↔z°':>9s}{'접촉N':>8s}{'손가락':>7s}{'컵 dz':>9s}{'컵 기울기°':>11s}{'판정':>10s}")
print("=" * 86)
for name, rot in CASES:
    env.reset()
    act = torch.zeros(N, A, device=dev)
    act[:, 3], act[:, 4], act[:, 5] = rot
    for _ in range(60):                      # 자세부터 잡는다
        env.step(act)
    pa = palm_axis_deg().mean().item()

    # 컵을 손 안에 **수직으로**(identity quat) 배치
    palm0 = env.robot.data.body_pos_w[:, env.palm_idx]
    tips0 = env.robot.data.body_pos_w[:, env._tip_t].mean(dim=1)
    root = torch.zeros(N, 13, device=dev)
    root[:, :3] = 0.5 * (palm0 + tips0)
    root[:, 3] = 1.0
    env.object.write_root_state_to_sim(root)
    env.step(act)
    z0 = (env.object.data.root_pos_w - env.scene.env_origins)[:, 2].clone()

    for i in range(args.close):              # 손 폐합 (자세 유지)
        act[:, 6:] = min(1.0, i / (args.close * 0.5))
        env.step(act)
    f, _, _, _ = env._contact()
    force = f.max(dim=1).values.mean().item()
    nfing = (f > 1.0).float().sum(dim=1).mean().item()

    home_z, hi_z = env.home_palm[0, 2].item(), env.palm_hi[0, 2].item()
    for i in range(args.lift):               # +15cm 상승
        frac = min(1.0, i / (args.lift * 0.5))
        act[:, 2] = min(1.0, (0.15 * frac) / max(hi_z - home_z, 1e-6))
        env.step(act)

    z = (env.object.data.root_pos_w - env.scene.env_origins)[:, 2]
    dz = (z - z0).mean().item()
    tilt = env._object_tilt_deg().mean().item()
    verdict = "PASS" if dz > 0.05 else ("부분" if dz > 0.01 else "FAIL")
    print(f"{name:<12s}{pa:9.2f}{force:8.2f}{nfing:7.2f}{dz:+9.4f}{tilt:11.2f}{verdict:>10s}")

print("=" * 86)
print("★홈(수직) 이 PASS 이고 기울기가 낮으면 → 정책의 국소최적. tilt 페널티 강화가 유효하다.")
print("★홈(수직) 이 FAIL 이면 → 기울기는 파지의 대가. 페널티 강화는 파지를 팔아버린다.")
env.close()
app.close()

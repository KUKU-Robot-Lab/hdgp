"""self-collision 을 끄면 **어느 마디 쌍이** 관통하는가 — body_repulsion 쌍 선별 근거.

PhysX self-collision 을 끄려면 손가락 관통을 Fabrics `body_repulsion` 이 계획 단계에서
막아야 한다. 그런데 손가락 충돌 구를 전부 넣으면 쌍이 수백 개로 늘고, 나란히 붙은
손가락(index-middle-ring)은 구조적으로 가까워 상시 오탐이 된다.
→ **실제로 관통하는 쌍만** 덮는 것이 답이고, 그걸 여기서 잰다.

마디를 **선분**으로 본다(링크 원점 → 자식 링크 원점). 원점 간 거리로는 길쭉한 마디의
실제 근접을 못 잡는다. 선분-선분 최소거리가 두 마디 반경 합보다 작으면 겹침이다.
실측 마디 단면 16.1 x 19.6mm → 반경 8~10mm, 합 ~18mm 를 겹침 기준으로 쓴다.

self-collision ON/OFF 를 같은 자세로 비교해 "OFF 에서만 생기는 관통"을 가른다.

    isaaclab.sh -p .../probe_finger_penetration.py --self_coll 0
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--self_coll", type=int, default=0)
parser.add_argument("--hand_control", default="pd", choices=["pd", "fabric"],
                    help="fabric = 손을 Fabrics 가 제어(body_repulsion 이 관통을 막는지 본다)")
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--hand_repulsion", type=int, default=0,
                    help="1 = 손가락 Fabrics 반발을 켠다(계획 단계 겹침 회피)")
parser.add_argument("--overlap_mm", type=float, default=18.0, help="겹침 기준(두 마디 반경 합)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym      # noqa: E402
import torch                 # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg   # noqa: E402

import openarm.tasks         # noqa: E402,F401
from openarm.agnostic.tasks.grasp_lift_fabric.grasp_lift_fabric_env_cfg import (  # noqa: E402
    resolve_cfg,
)

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env_cfg.enable_self_collisions = bool(args.self_coll)
env_cfg.enable_gravity = True
# ★body_repulsion 은 **fabric 이 제어하는 관절**에만 유효하다. pd 모드에서는
#   손이 fabric 밖이라 쌍을 넣어도 막을 수 없다 — 그 차이를 이 인자로 가른다.
env_cfg.hand_control = args.hand_control
env_cfg.use_hand_repulsion = bool(args.hand_repulsion)
env_cfg.use_tip_fabric = False
resolve_cfg(env_cfg)
env = gym.make(args.task, cfg=env_cfg).unwrapped
N, A = args.num_envs, env.cfg.action_space
dev = env.device
P = env.profile
FING = list(P.fingers)
SEGS = ["1", "2", "3", "4", "tip"]


def _body(name: str):
    ids, _ = env.robot.find_bodies(name)
    return ids[0] if len(ids) == 1 else None


# 손가락별 마디 체인 — 이름 규약은 프로필의 wrap/tip body 에서 접두사를 얻는다.
_any = P.finger_wrap_bodies[FING[0]][0]          # 예: r_hl_thumb_3
_pre = _any[: _any.index(FING[0])]               # "r_hl_"
chain = {}
for f in FING:
    ids = [_body(f"{_pre}{f}_{s}") for s in SEGS]
    chain[f] = [i for i in ids if i is not None]
print("마디 체인:", {f: len(v) for f, v in chain.items()}, flush=True)

act = torch.zeros(N, A, device=dev)
env.reset()
for i in range(args.steps):
    act[:, 6:] = min(1.0, (i + 1) / 60.0)        # 개방 → 완전 폐합
    env.step(act)

# ★repulsion 은 **fabric_q(계획)** 에 작용한다. 실제 관절(body_pos_w)은 거기에 PD 추종과
#   물리 접촉까지 얹힌 결과라, 그것만 재면 repulsion 효과를 볼 수 없다.
#   "정책이 관통 해를 쓰지 못하게 탐색 공간에서 제거하는가" = fabric_q 에서의 겹침이다.
if getattr(env.fabric, "hand_fabric_repulsion", None) is not None:
    _hp, _ = env.fabric.get_taskmap("hand_points")(env.fabric_q.detach(), None)
    _hp = _hp.reshape(N, -1, 3)
    _d = torch.cdist(_hp, _hp)
    _eye = torch.eye(_hp.shape[1], device=dev, dtype=torch.bool)
    _d = _d.masked_fill(_eye.unsqueeze(0), float("inf"))
    # 같은 손가락 구는 캡슐이라 항상 겹친다 — 프레임 이름으로 손가락을 갈라 제외한다.
    import re as _re
    _fr = env.fabric.get_taskmap("hand_points").link_names
    _own = torch.tensor([int(_re.search(r"dg_(\d+)_", n).group(1)) for n in _fr], device=dev)
    _cross = (_own[:, None] != _own[None, :])
    _d = _d.masked_fill(~_cross.unsqueeze(0), float("inf"))
    _mind = _d.view(N, -1).min(dim=1).values
    print(f"[fabric_q] 다른 손가락 구 최소거리 {float(_mind.mean())*1000:.1f}mm "
          f"· 18mm 미만 {float((_mind < 0.018).float().mean())*100:.1f}%  "
          f"← repulsion 이 계획 단계에서 막는가", flush=True)

_palm = env.robot.data.body_pos_w[:, env.palm_idx] - env.scene.env_origins
print(f"palm 추종오차 {float((env.palm_targets[:, :3] - _palm).norm(dim=-1).mean())*1000:.1f}mm "
      f"(반발 강도가 팔 회피와 공유되므로 함께 본다)", flush=True)
pos = env.robot.data.body_pos_w                   # (N, B, 3)


def seg_dist(p1, q1, p2, q2):
    """선분(p1,q1) ↔ 선분(p2,q2) 최소거리. 배치 (N,3) 텐서."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = (d1 * d1).sum(-1); e = (d2 * d2).sum(-1); f = (d2 * r).sum(-1)
    c = (d1 * r).sum(-1); b = (d1 * d2).sum(-1)
    den = (a * e - b * b).clamp(min=1e-9)
    s = ((b * f - c * e) / den).clamp(0, 1)
    t = ((b * s + f) / e.clamp(min=1e-9)).clamp(0, 1)
    s = ((b * t - c) / a.clamp(min=1e-9)).clamp(0, 1)
    return ((p1 + d1 * s[:, None]) - (p2 + d2 * t[:, None])).norm(dim=-1)


thr = args.overlap_mm / 1000.0
print(f"\n=== 마디 쌍 근접 · self_coll={bool(args.self_coll)} · hand={args.hand_control} · {N}env "
      f"(완전 폐합, 겹침기준 {args.overlap_mm:.0f}mm) ===", flush=True)
rows = []
for i, fa in enumerate(FING):
    for fb in FING[i + 1:]:
        for ka in range(len(chain[fa]) - 1):
            for kb in range(len(chain[fb]) - 1):
                d = seg_dist(pos[:, chain[fa][ka]], pos[:, chain[fa][ka + 1]],
                             pos[:, chain[fb][kb]], pos[:, chain[fb][kb + 1]])
                rows.append((float(d.mean()) * 1000, float((d < thr).float().mean()) * 100,
                             f"{fa}_{SEGS[ka]}", f"{fb}_{SEGS[kb]}"))
rows.sort(key=lambda r: r[0])
print(f"{'마디 A':>12s} {'마디 B':>12s} {'평균거리[mm]':>12s} {'겹침율':>8s}")
for d, v, a_, b_ in rows[:18]:
    print(f"{a_:>12s} {b_:>12s} {d:12.1f} {v:7.1f}%", flush=True)
n_ov = sum(1 for d, v, _, _ in rows if v > 5.0)
print(f"\n겹침율 5% 초과 쌍: {n_ov} / {len(rows)}", flush=True)

env.close()
app.close()

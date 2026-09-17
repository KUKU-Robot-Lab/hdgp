"""RH56F1 손 기하 계측 (09.17) — 보상이 "컵이 손 안에 있다"를 손바닥 좌표로 정의할 수 있게 사실만 뽑는다.

grasp_lift 격자(열린 루프 대본)는 접근 경로가 컵을 쳐서 파지 가능성을 판정하지 못했다. 보상은 접근 경로가 아니라
"닫기 직전에 컵 축이 손바닥 좌표의 어디에 있어야 하는가"를 알아야 한다. 그 값은 손만 있으면 나온다:
홈 자세에서 팔을 세워둔 채 손만 연 상태 → 끝까지 닫은 상태로 보내며 손끝·원위 링크 위치를 손바닥 좌표로 기록한다.

손바닥 좌표(cfg 주석과 같은 규약): a0 = palm_R 열 0(손가락이 늘어선 가로), a1 = 열 1(손가락 길이), a2 = 열 2(손바닥 법선).
보상 입력 ctx.*_palm_axes = [a2, a1] 이다.

사용:
  ./IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/probes/probe_rh_hand_geometry.py --num_envs 4 \
      --out_json log/pour_fabric_mimic/hand_geometry.json
"""
from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-rh_b_pour_fab_mimic")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--close_steps", type=int, default=240)
parser.add_argument("--every", type=int, default=10)
parser.add_argument("--out_json", default="log/pour_fabric_mimic/hand_geometry.json")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import openarm.tasks  # noqa: E402,F401
import openarm.agnostic.tasks.pour_fabric_mimic.config  # noqa: E402,F401

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=cfg).unwrapped
N, A = env.num_envs, env.cfg.action_space
HALF = A // 2
dev = env.device
bn = env.robot.data.body_names

_orig_dones = env._get_dones


def _dones_off():
    term, trunc = _orig_dones()
    return torch.zeros_like(term), torch.zeros_like(trunc)


env._get_dones = _dones_off


def r1(v):
    return [round(float(x) * 1000.0, 1) for x in v]


def in_palm(rig, pos_w_local):
    """(N,K,3) env-local 위치 → 손바닥 좌표 (N,K,3) [a0,a1,a2] (m)."""
    R = rig.palm_R()                                   # (N,3,3) 열이 축
    d = pos_w_local - rig.palm_pos()[:, None, :]
    return torch.einsum("nkj,nji->nki", d, R)          # d · 열 i


sides = (("src", env.src, 0), ("rcv", env.rcv, 1))
links = {}
for tag, rig, si in sides:
    names = []
    for finger, bodies in rig.profile.finger_sensor_bodies.items():
        for b in bodies:
            names.append((finger, b))
    links[tag] = names
    print(f"[{tag}] fingers={list(rig.profile.finger_sensor_bodies.keys())} tips={list(rig.profile.fingertip_bodies)}", flush=True)
print(f"[cfg] N={N} action={A} half={HALF} hand_dims={HALF - 6} hold={int(env.cfg.hold_steps)} "
      f"close_speed={env.cfg.synergy_close_speed} cup_radius={env.cfg.cup_inner_radius} mouth_z={env.cfg.cup_mouth_z}", flush=True)


def snap(rig, tag):
    env_o = env.scene.env_origins
    tips = in_palm(rig, rig.tips_pos())[0]                                            # (F,3)
    idx = [bn.index(b) for _, b in links[tag]]
    lp = env.robot.data.body_pos_w[:, idx] - env_o[:, None, :]
    lk = in_palm(rig, lp)[0]
    cup = (env.source_cup if tag == "src" else env.receiver_cup).data.root_pos_w - env_o
    cup_p = in_palm(rig, cup[:, None, :])[0, 0]
    return {
        "closure": round(float(rig.closure()[0]), 3),
        "tips_mm": [r1(t) for t in tips],
        "links_mm": {f"{f}:{b}": r1(lk[i]) for i, (f, b) in enumerate(links[tag])},
        "cup_origin_mm": r1(cup_p),
        "finger_force_max": round(float(rig.finger_forces().max()), 3),
        "palm_world_mm": r1(rig.palm_pos()[0]),
    }


out = {"frame": "a0=palm_R col0 (across fingers), a1=col1 (finger length), a2=col2 (palm normal); mm from palm link origin",
       "series": {"src": [], "rcv": []}}
env.reset()
for _ in range(int(env.cfg.hold_steps) + 30):
    env.step(torch.zeros(N, A, device=dev))
for tag, rig, si in sides:
    s = snap(rig, tag)
    s["step"] = 0
    out["series"][tag].append(s)
    print(f"[{tag}] open  closure={s['closure']} tips(a0,a1,a2)mm={s['tips_mm']} cup={s['cup_origin_mm']}", flush=True)

for u in range(1, args.close_steps + 1):
    a = torch.zeros(N, A, device=dev)
    for tag, rig, si in sides:
        a[:, si * HALF + 6:(si + 1) * HALF] = 1.0
    env.step(a)
    if u % args.every == 0:
        for tag, rig, si in sides:
            s = snap(rig, tag)
            s["step"] = u
            out["series"][tag].append(s)
for tag, rig, si in sides:
    s = out["series"][tag][-1]
    print(f"[{tag}] close closure={s['closure']} tips(a0,a1,a2)mm={s['tips_mm']} fmax={s['finger_force_max']}", flush=True)
    for row in out["series"][tag]:
        t = row["tips_mm"]
        print(f"[{tag}] u={row['step']:3d} clo={row['closure']:.2f} thumb(a1,a2)=({t[0][1]:6.1f},{t[0][2]:6.1f}) "
              f"index(a1,a2)=({t[1][1]:6.1f},{t[1][2]:6.1f}) a0 thumb/index/pinky={t[0][0]:6.1f}/{t[1][0]:6.1f}/{t[-1][0]:6.1f}", flush=True)

os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
with open(args.out_json, "w") as fh:
    json.dump(out, fh, ensure_ascii=False, indent=1)
print(f"[out] {args.out_json}", flush=True)
env.close()
app.close()

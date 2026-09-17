"""RH56F1 붓기 트랙 — 손이 닿기 전에 컵이 왜 넘어지는가(09.17). 사건 순서 계측.

  probe_rh_grasp_lift 격자에서 기준 조합까지 약 90 % env 가 진입 전(step 41~120)에 컵이 넘어져 바닥으로 떨어졌다.
  종료를 끄고 매 스텝 신호를 env 별로 기록해 "무엇이 먼저 일어났는가"를 센다.

  mode zero   : 전 구간 a=0 (양팔 앵커 유지, 손 명령 0)
  mode open   : 대본 없이 한쪽 손만 hand_cmd(기본 -1) — 팔 위치 명령 0
  mode script : 격자 기준 조합의 above→down→enter (손 -1)

    ../IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/probes/probe_rh_cup_disturb_trace.py --num_envs 64
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-rh_b_pour_fab_mimic")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--side", default="src", choices=["src", "rcv"])
parser.add_argument("--modes", default="zero,open,script")
parser.add_argument("--steps", type=int, default=240, help="hold 이후 기록 스텝 수")
parser.add_argument("--hand_cmd", type=float, default=-1.0)
parser.add_argument("--dz", type=float, default=0.027)
parser.add_argument("--side_m", type=float, default=0.08)
parser.add_argument("--safe_dz", type=float, default=0.08)
parser.add_argument("--table_obstacle", type=int, default=1)
parser.add_argument("--out_json", default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import openarm.tasks  # noqa: E402,F401
import openarm.agnostic.tasks.pour_fabric_mimic.config  # noqa: E402,F401

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
cfg.fabric_table_obstacle = bool(args.table_obstacle)
env = gym.make(args.task, cfg=cfg).unwrapped
N, A = env.num_envs, env.cfg.action_space
HALF = A // 2
dev = env.device
hold = int(env.cfg.hold_steps)
bn = env.robot.data.body_names
jn = env.robot.data.joint_names

_orig_dones = env._get_dones


def _dones_off():
    term, trunc = _orig_dones()
    q, qd = env.robot.data.joint_pos, env.robot.data.joint_vel
    nonfinite = ~(torch.isfinite(q).all(dim=-1) & torch.isfinite(qd).all(dim=-1))
    return term & nonfinite, torch.zeros_like(trunc)


env._get_dones = _dones_off

SI = 0 if args.side == "src" else 1
rig, other = (env.src, env.rcv) if SI == 0 else (env.rcv, env.src)
cup, cup_o = (env.source_cup, env.receiver_cup) if SI == 0 else (env.receiver_cup, env.source_cup)
n_sign = 1.0 if SI == 0 else -1.0
prefix = rig.profile.hand_joint_names[0].split("_hj_")[0]
prefix_o = other.profile.hand_joint_names[0].split("_hj_")[0]
i_th = bn.index(rig.profile.fingertip_bodies[0])
i_ix = bn.index(rig.profile.fingertip_bodies[1])
i_ix1 = bn.index(rig.profile.finger_sensor_bodies["index"][0])
hand_b = [i for i, b in enumerate(bn) if b.startswith(f"{prefix}_hl_")]
hand_b_o = [i for i, b in enumerate(bn) if b.startswith(f"{prefix_o}_hl_")]
arm_b = [i for i, b in enumerate(bn) if b.startswith(f"{prefix}_") and i not in hand_b]
dep_names = [jn[int(i)] for i in env._mim_dep_t]
dep_mine = torch.tensor([k for k, nm in enumerate(dep_names) if nm.startswith(f"{prefix}_")], device=dev)
dep_oth = torch.tensor([k for k, nm in enumerate(dep_names) if nm.startswith(f"{prefix_o}_")], device=dev)
print(f"[cfg] N={N} side={args.side} prefix={prefix} hold={hold} steps={args.steps} hand_cmd={args.hand_cmd} "
      f"hand_bodies={len(hand_b)} arm_bodies={len(arm_b)} dep_mine={len(dep_mine)} dep_other={len(dep_oth)} "
      f"runaway_qd={env.cfg.mimic_runaway_dep_qd} runaway_err={env.cfg.mimic_runaway_err_rad}", flush=True)


def cup_tilt(c):
    up = matrix_from_quat(c.data.root_quat_w)[:, :, 2]
    return torch.rad2deg(torch.acos(up[:, 2].clamp(-1.0, 1.0)))


def hand_frame():
    pos = env.robot.data.body_pos_w - env.scene.env_origins[:, None, :]
    pocket = 0.5 * (pos[:, i_th] + pos[:, i_ix])
    f = pos[:, i_ix, :2] - pos[:, i_ix1, :2]
    f = f / (f.norm(dim=-1, keepdim=True) + 1e-9)
    n = n_sign * torch.stack([-f[:, 1], f[:, 0]], dim=1)
    return pocket, n


def scaled(v, lo, hi):
    return torch.where(v >= 0, (v / hi).clamp(max=1.0), -(v / lo).clamp(max=1.0))


def servo(goal_pocket, hand_cmd, extra_z=0.0):
    pocket, _ = hand_frame()
    tgt = rig.palm_pos() + (goal_pocket - pocket)
    tgt[:, 2] += extra_z
    d = tgt - rig.anchor_env[:, :3]
    a = torch.zeros(N, A, device=dev)
    for k in range(3):
        a[:, SI * HALF + k] = scaled(d[:, k], float(rig.delta_lo[k]), float(rig.delta_hi[k]))
    a[:, SI * HALF + 6:(SI + 1) * HALF] = hand_cmd
    return a


def bead_speed():
    try:
        v = env.beads.data.object_lin_vel_w
    except AttributeError:
        return torch.full((N,), float("nan"), device=dev)
    return v.norm(dim=-1).max(dim=-1).values


# 신호 이름 → (임계, 방향) : 첫 교차 스텝을 센다
THR = {
    "cup_tilt": (5.0, ">"), "cup_speed": (0.05, ">"), "cup_dz": (-0.03, "<"),
    "ocup_tilt": (5.0, ">"), "ocup_dz": (-0.03, "<"),
    "dep_qd": (100.0, ">"), "dep_qd_other": (100.0, ">"), "mim_err": (1.0, ">"),
    "arm_qd": (float(env.cfg.runaway_joint_vel), ">"),
    "finger_f": (0.5, ">"), "palm_f": (0.5, ">"), "foreign_f": (0.5, ">"), "foreign_f_other": (0.5, ">"),
    "hand_cup_mm": (45.0, "<"), "arm_cup_mm": (60.0, "<"), "ohand_ocup_mm": (45.0, "<"),
    "bead_speed": (1.0, ">"),
}


def measure(rest, rest_o):
    q, qd = env.robot.data.joint_pos, env.robot.data.joint_vel
    pos = env.robot.data.body_pos_w
    c_w, co_w = cup.data.root_pos_w, cup_o.data.root_pos_w
    dep_qd = qd[:, env._mim_dep_t].abs()
    err = (q[:, env._mim_dep_t] - env._mim_mult * q[:, env._mim_lead_t]).abs()
    return {
        "cup_tilt": cup_tilt(cup), "cup_speed": cup.data.root_lin_vel_w.norm(dim=-1),
        "cup_dz": env._local(c_w)[:, 2] - rest[:, 2],
        "ocup_tilt": cup_tilt(cup_o), "ocup_dz": env._local(co_w)[:, 2] - rest_o[:, 2],
        "dep_qd": dep_qd[:, dep_mine].max(dim=-1).values, "dep_qd_other": dep_qd[:, dep_oth].max(dim=-1).values,
        "mim_err": err.max(dim=-1).values,
        "arm_qd": qd[:, rig.arm_t].abs().max(dim=-1).values,
        "finger_f": rig.finger_forces().max(dim=-1).values, "palm_f": rig.palm_force(),
        "foreign_f": rig.foreign_force(), "foreign_f_other": other.foreign_force(),
        "hand_cup_mm": (pos[:, hand_b] - c_w[:, None, :]).norm(dim=-1).min(dim=-1).values * 1000.0,
        "arm_cup_mm": (pos[:, arm_b] - c_w[:, None, :]).norm(dim=-1).min(dim=-1).values * 1000.0,
        "ohand_ocup_mm": (pos[:, hand_b_o] - co_w[:, None, :]).norm(dim=-1).min(dim=-1).values * 1000.0,
        "bead_speed": bead_speed(),
    }


def run(mode: str) -> dict:
    env.reset()
    for _ in range(hold):
        env.step(torch.zeros(N, A, device=dev))
    rest = env._local(cup.data.root_pos_w).clone()
    rest_o = env._local(cup_o.data.root_pos_w).clone()
    series = {k: [] for k in THR}
    for u in range(args.steps):
        if mode == "zero":
            a = torch.zeros(N, A, device=dev)
        elif mode == "open":
            a = torch.zeros(N, A, device=dev)
            a[:, SI * HALF + 6:(SI + 1) * HALF] = args.hand_cmd
        else:
            c = env._local(cup.data.root_pos_w)
            _, n = hand_frame()
            n3 = torch.cat([n, torch.zeros(N, 1, device=dev)], dim=1)
            grip = torch.cat([c[:, :2], (c[:, 2] + args.dz).unsqueeze(1)], dim=1)
            if u < 90:
                a = servo(grip + args.side_m * n3, args.hand_cmd, extra_z=args.safe_dz)
            elif u < 160:
                a = servo(grip + args.side_m * n3, args.hand_cmd)
            else:
                a = servo(grip, args.hand_cmd)
        env.step(a)
        m = measure(rest, rest_o)
        for k in THR:
            series[k].append(m[k].detach().float().cpu())
    S = {k: torch.stack(v, dim=1) for k, v in series.items()}     # (N, T)

    first = {}
    for k, (thr, op) in THR.items():
        hit = (S[k] > thr) if op == ">" else (S[k] < thr)
        hit = hit & torch.isfinite(S[k])
        idx = torch.where(hit.any(dim=1), hit.float().argmax(dim=1), torch.full((N,), -1, dtype=torch.long))
        first[k] = idx
    tag = f"{args.side}/{mode}"
    print(f"\n===== [{tag}] 첫 교차 스텝(hold 이후 0 기준) — 발생 env 수 / 중앙 / 최소 =====", flush=True)
    for k, (thr, op) in THR.items():
        v = first[k][first[k] >= 0]
        if len(v) == 0:
            print(f"[{tag}] {k:16s} {op}{thr:<7g} 0/{N}", flush=True)
            continue
        print(f"[{tag}] {k:16s} {op}{thr:<7g} {len(v)}/{N} · 중앙 {int(v.median())} · 최소 {int(v.min())} · "
              f"구간 최대 {float(torch.nan_to_num(S[k], nan=0.0).abs().max()):.1f}", flush=True)

    # 컵이 기운 env 에서: 기울기 첫 교차 직전까지 어떤 신호가 이미 교차했는가
    tilt_first = first["cup_tilt"]
    moved = (tilt_first >= 0).nonzero().flatten().tolist()
    before = {k: 0 for k in THR if not k.startswith("cup_")}
    for i in moved:
        for k in before:
            if 0 <= int(first[k][i]) <= int(tilt_first[i]):
                before[k] += 1
    print(f"[{tag}] 컵이 5° 넘게 기운 env {len(moved)}/{N} — 그 시점까지 이미 교차한 신호(env 수): "
          + " · ".join(f"{k} {v}" for k, v in before.items()), flush=True)

    # 가장 먼저 기운 env 3 개의 시계열(기울기 첫 교차 -12 ~ +8 스텝)
    order = sorted(moved, key=lambda i: int(tilt_first[i]))[:3]
    cols = ["cup_tilt", "cup_dz", "cup_speed", "hand_cup_mm", "arm_cup_mm", "finger_f", "palm_f", "foreign_f",
            "dep_qd", "mim_err", "arm_qd", "bead_speed"]
    for i in order:
        t0 = int(tilt_first[i])
        print(f"[{tag}] env{i} 기울기 첫 교차 u={t0} · " + " ".join(cols), flush=True)
        for u in range(max(0, t0 - 12), min(args.steps, t0 + 9), 2):
            print(f"[{tag}]   u={u:3d} " + " ".join(f"{float(S[k][i, u]):9.3f}" for k in cols), flush=True)
    return {"mode": mode, "first": {k: v.tolist() for k, v in first.items()},
            "max": {k: float(torch.nan_to_num(S[k], nan=0.0).abs().max()) for k in THR}}


out = [run(m.strip()) for m in args.modes.split(",") if m.strip()]
if args.out_json:
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(out, fh)
env.close()
app.close()

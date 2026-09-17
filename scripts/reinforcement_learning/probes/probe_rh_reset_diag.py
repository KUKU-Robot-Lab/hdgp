"""RH56F1 붓기 트랙 리셋 진단·게이트(09.17) — 보상과 무관한 원인을 재고, 학습 전 통과 조건을 확인한다.

  diag    : a=0 유지 오차(회전 wrap 전후)·축별 도달 비율·폐루프 대본 파지(손가락별 힘). Phase 1 판정용.
  extract : 컵을 치운 채 가상 컵 옆 사전파지 대기 자세로 손 포켓을 보내고, 그 팔 관절값과 컵 상대 위치를
            JSON 으로 저장한다(시작 상태 혼합 P2 의 원본 데이터).
  gate    : 학습 전 통과 게이트 — 위에서 컵 옆으로 내려와(쓸고 지나가지 않게) 포켓에 넣고 오므린 뒤 10 cm 들고
            유지한다. 판정: 들기·유지 구간 파지 플래그 평균 ≥ 0.9 · 유지 구간 최소 들기 ≥ 10 cm · 기울기 ≤ 20° · 놓침 없음.

    ../IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/probes/probe_rh_reset_diag.py --side src --mode extract,gate
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-rh_b_pour_fab_mimic")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--side", default="src", choices=["src", "rcv"])
parser.add_argument("--mode", default="extract,gate", help="쉼표 구분: diag,extract,gate")
parser.add_argument("--n_sign", type=float, default=0.0, help="열린 쪽 법선 부호(0 = src +1 / rcv −1, 좌손 미러)")
parser.add_argument("--table_obstacle", type=int, default=1)
parser.add_argument("--side_m", type=float, default=0.08, help="사전파지 대기 오프셋(열린 쪽) [m]")
parser.add_argument("--safe_dz", type=float, default=0.08, help="gate: 위에서 내려오는 높이 여유 [m]")
parser.add_argument("--out_json", default="source/openarm/openarm/agnostic/tasks/pour_fabric_mimic/data/pregrasp_start.json")
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
cfg.fabric_table_obstacle = bool(args.table_obstacle)
env = gym.make(args.task, cfg=cfg).unwrapped
N, A = env.num_envs, env.cfg.action_space
HALF = A // 2
dev = env.device
SI = 0 if args.side == "src" else 1
rig = env.src if SI == 0 else env.rcv
cup_asset = env.source_cup if SI == 0 else env.receiver_cup
n_sign = args.n_sign if args.n_sign != 0.0 else (1.0 if SI == 0 else -1.0)
modes = [m.strip() for m in args.mode.split(",") if m.strip()]
hold = int(env.cfg.hold_steps)
mid_z = 0.5 * float(env.cfg.cup_mouth_z)
rest_z = float(env.cfg.table_surface_z) + float(env.cfg.object_origin_offset_z)
bn = env.robot.data.body_names
i_th = bn.index(rig.profile.fingertip_bodies[0])
i_ix = bn.index(rig.profile.fingertip_bodies[1])
i_ix1 = bn.index(rig.profile.finger_sensor_bodies["index"][0])


def wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def fmt(v, nd=3):
    return [round(float(x), nd) for x in v]


def cup_local():
    return env._local(cup_asset.data.root_pos_w)


def to_action(d):
    a = torch.zeros(N, 3, device=dev)
    for k in range(3):
        v = d[:, k]
        a[:, k] = torch.where(v >= 0, (v / rig.delta_hi[k]).clamp(max=1.0), -(v / rig.delta_lo[k]).clamp(max=1.0))
    return a


def hand_frame():
    """손 포켓(엄지 끝·검지 끝 중점)과 열린 쪽 법선 n(검지 방향의 수평 수직, 좌손은 부호 반전)."""
    pos = env.robot.data.body_pos_w - env.scene.env_origins[:, None, :]
    pocket = 0.5 * (pos[:, i_th] + pos[:, i_ix])
    f = pos[:, i_ix, :2] - pos[:, i_ix1, :2]
    f = f / (f.norm(dim=-1, keepdim=True) + 1e-9)
    n = n_sign * torch.stack([-f[:, 1], f[:, 0]], dim=1)
    return pocket, n


def servo(goal_pocket, hand_cmd, extra_z=0.0):
    """포켓을 goal 로 보내는 이 팔 액션(나머지 팔은 a=0)."""
    pocket, _ = hand_frame()
    tgt = rig.palm_pos() + (goal_pocket - pocket)
    tgt[:, 2] += extra_z
    a = torch.zeros(N, A, device=dev)
    a[:, SI * HALF:SI * HALF + 3] = to_action(tgt - rig.anchor_env[:, :3])
    a[:, SI * HALF + 6:(SI + 1) * HALF] = hand_cmd
    return a


def settle():
    env.reset()
    for _ in range(hold):
        env.step(torch.zeros(N, A, device=dev))


print(f"[cfg] side={args.side} n_sign={n_sign:+.0f} modes={modes} freeze_thr={env.cfg.contact_freeze_threshold} "
      f"grasp_thr={env.cfg.contact_force_threshold} table_obstacle={args.table_obstacle}", flush=True)

# ---- diag -------------------------------------------------------------------------------
if "diag" in modes:
    settle()
    perr = float((rig.palm_targets[:, :3] + rig.fab_to_env - rig.palm_pos()).norm(dim=-1).mean())
    raw = rig.palm_targets[:, 3:] - rig.palm_pose_6d()[:, 3:]
    print(f"[diag hold] pos_err={perr*1000:.1f}mm rot_raw={float(torch.rad2deg(raw.abs().max(dim=1).values).mean()):.1f}deg "
          f"rot_wrapped={float(torch.rad2deg(wrap(raw).abs().max(dim=1).values).mean()):.1f}deg", flush=True)

# ---- gate -------------------------------------------------------------------------------
if "gate" in modes:
    settle()
    plan = [("above", 90), ("down", 70), ("enter", 110), ("close", 130), ("lift", 90), ("hold", 70)]
    rec = {k: [] for k, _ in plan}
    for name, steps in plan:
        for u in range(steps):
            c = cup_local()
            _, n = hand_frame()
            if name == "above":
                goal = torch.cat([c[:, :2] + args.side_m * n, (c[:, 2] + mid_z + args.safe_dz).unsqueeze(1)], dim=1)
                a = servo(goal, -1.0)
            elif name == "down":
                goal = torch.cat([c[:, :2] + args.side_m * n, (c[:, 2] + mid_z).unsqueeze(1)], dim=1)
                a = servo(goal, -1.0)
            elif name == "enter":
                goal = torch.cat([c[:, :2], (c[:, 2] + mid_z).unsqueeze(1)], dim=1)
                a = servo(goal, -1.0)
            elif name == "close":
                goal = torch.cat([c[:, :2], (c[:, 2] + mid_z).unsqueeze(1)], dim=1)
                a = servo(goal, 1.0)
            else:
                spawn_z = (env._src_spawn if SI == 0 else env._rcv_spawn)[:, 2]
                goal = torch.cat([c[:, :2], (spawn_z + mid_z).unsqueeze(1)], dim=1)
                k_up = min(1.0, (u + 1) / 60.0) if name == "lift" else 1.0
                a = servo(goal, 1.0, extra_z=0.10 * k_up)
            _, _, term, _, ex = env.step(a)
            tag = "src" if SI == 0 else "rcv"
            if bool(term.any()):
                print(f"[gate {name} u={u} TERM] drop={float(env._dropped.float().mean()):.2f} "
                      f"arm_runaway={float(ex['task/runaway_rate']):.2f} mimic_runaway={float(ex['done/mimic_runaway']):.2f} "
                      f"mimic_err_runaway={float(ex['done/mimic_err_runaway']):.2f} mimic_err_max={float(ex['ctrl/mimic_err_max']):.2f} "
                      f"hand_dep_qd_max={float(ex['ctrl/hand_dep_qd_max']):.1f}", flush=True)
            pocket, _ = hand_frame()
            rec[name].append(dict(grasp=float(ex[f"task/{tag}_grasped"]), lift=float(ex[f"task/{tag}_cup_lift"]),
                                  tilt=float(ex[f"task/{tag}_tilt_deg"]), term=bool(term.any()),
                                  pc=float((pocket[:, :2] - c[:, :2]).norm(dim=-1).mean()),
                                  f=[round(v, 2) for v in rig.finger_forces()[0].tolist()]))
        if name == "down" and "extract" in modes:
            q = env.robot.data.joint_pos[:, rig.arm_ids]
            palm = rig.palm_pos()
            c = cup_local()
            pocket, n = hand_frame()
            rel = (c[:, :2] - palm[:, :2]).mean(dim=0)
            names = [env.robot.data.joint_names[i] for i in rig.arm_ids]
            resid = float((pocket[:, :2] - (c[:, :2] + args.side_m * n)).norm(dim=-1).mean())
            data = {}
            if os.path.exists(args.out_json):
                with open(args.out_json) as fh:
                    data = json.load(fh)
            data[args.side] = {"arm_joint_names": names, "q_arm": fmt(q.mean(0), 5), "q_arm_std_max": round(float(q.std(dim=0).max()), 5),
                               "palm_env": fmt(palm.mean(0), 5), "cup_rel_xy": fmt(rel, 5), "side_m": args.side_m,
                               "pocket_residual_mm": round(resid * 1000, 2), "cup_tilt_deg": round(float(env.extras[f"task/{'src' if SI == 0 else 'rcv'}_tilt_deg"]), 2),
                               "note": "probe_rh_reset_diag gate 'down' 끝 09.17 — 컵 옆 열린 쪽 대기 자세(위에서 내려옴)"}
            os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
            with open(args.out_json, "w") as fh:
                json.dump(data, fh, indent=1, ensure_ascii=False)
            print(f"[extract] 저장 {args.side} 포켓 잔여 {resid*1000:.1f}mm q 산포 {float(q.std(dim=0).max()):.4f}rad "
                  f"palm={fmt(palm.mean(0))} cup_rel_xy={fmt(rel)} q={fmt(q.mean(0), 4)}", flush=True)
        last = rec[name][-1]
        print(f"[gate {name:5s}] 포켓-컵 {last['pc']*1000:.0f}mm · 파지 {last['grasp']:.2f} · 들기 {last['lift']*100:+.1f}cm · "
              f"기울기 {last['tilt']:.1f}° · 힘[th,ix,md,rg,pk] {last['f']} · 구간 최대 기울기 "
              f"{max(r['tilt'] for r in rec[name]):.1f}°", flush=True)
    lw = rec["lift"] + rec["hold"]
    g_mean = sum(r["grasp"] for r in lw) / len(lw)
    lift_min = min(r["lift"] for r in rec["hold"])
    tilt_max = max(r["tilt"] for k in ("enter", "close", "lift", "hold") for r in rec[k])
    lost = any(r["term"] for r in lw) or lift_min < 0.02
    ok = g_mean >= 0.9 and lift_min >= 0.10 and tilt_max <= 20.0 and not lost
    print(f"[gate {args.side} {'PASS' if ok else 'FAIL'}] 파지 평균(들기·유지) {g_mean:.2f} ≥0.9 · 유지 최소 들기 "
          f"{lift_min*100:.1f}cm ≥10 · 최대 기울기 {tilt_max:.1f}° ≤20 · 놓침 {lost}", flush=True)

env.close()
app.close()

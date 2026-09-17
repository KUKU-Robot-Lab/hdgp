"""RH56F1 붓기 트랙 리셋 진단(09.17) — 보상과 무관한 원인 3종을 한 세션에서 잰다.

  hold  : a=0(앵커 유지) 동안 fabric 손바닥 추종 오차. 회전 오차는 **wrap 없이**(env 로그 방식)와 **wrap 후**를 같이 낸다.
  sweep : 위치 액션 슬롯(x,y,z)을 ±1 로 하나씩 명령하고 명령 목표 vs 실제 도달을 잰다(fabric 이 따라가는지·어디서 막히는지).
  grasp : 폐루프 대본 — 손 포켓(엄지 끝·검지 끝 중점)을 컵 축·컵 중간 높이로 보낸 뒤 오므리고 들어올린다.
          손가락별 접촉력·파지 플래그·들기·기울기를 기록한다(이 손·이 컵으로 파지가 되는지).

    ../IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/probes/probe_rh_reset_diag.py --num_envs 4 --table_obstacle 1
"""
import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-rh_b_pour_fab_mimic")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--table_obstacle", type=int, default=1, help="0 = fabric 테이블 장애물 OFF")
parser.add_argument("--hold_steps_probe", type=int, default=120)
parser.add_argument("--sweep_steps", type=int, default=90)
parser.add_argument("--side_m", type=float, default=0.08, help="grasp: 대기 오프셋(열린 쪽) [m]")
parser.add_argument("--wait_steps", type=int, default=120)
parser.add_argument("--enter_steps", type=int, default=120)
parser.add_argument("--close_steps", type=int, default=120)
parser.add_argument("--lift_steps", type=int, default=100)
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
rig = env.src
dev = env.device


def wrap(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def cup():
    return env._local(env.source_cup.data.root_pos_w)


def to_action(d):
    """env 프레임 위치 델타(앵커 기준) → 위치 액션 슬롯 0..2 (a=0=앵커, 부호별 스케일)."""
    a = torch.zeros(N, 3, device=dev)
    for k in range(3):
        v = d[:, k]
        a[:, k] = torch.where(v >= 0, (v / rig.delta_hi[k]).clamp(max=1.0), -(v / rig.delta_lo[k]).clamp(max=1.0))
    return a


def target_env():
    return rig.palm_targets[:, :3] + rig.fab_to_env


def rot_errs():
    raw = rig.palm_targets[:, 3:] - rig.palm_pose_6d()[:, 3:]
    return (torch.rad2deg(raw.abs().max(dim=1).values).mean().item(),
            torch.rad2deg(wrap(raw).abs().max(dim=1).values).mean().item())


def fmt(v):
    return [round(float(x), 3) for x in v]


env.reset()
hold = int(env.cfg.hold_steps)
print(f"[cfg] table_obstacle={args.table_obstacle} hold_steps={hold} delta_lo={fmt(rig.delta_lo)} delta_hi={fmt(rig.delta_hi)} "
      f"box_lo={fmt(rig.box_lo)} box_hi={fmt(rig.box_hi)} box_verified={rig.profile.palm_box_verified}", flush=True)

# ---- hold -------------------------------------------------------------------------------
for t in range(hold + args.hold_steps_probe):
    env.step(torch.zeros(N, A, device=dev))
perr = (target_env() - rig.palm_pos()).norm(dim=-1)
r_raw, r_wrap = rot_errs()
print(f"[hold] anchor_env={fmt(rig.anchor_env[0, :3])} palm={fmt(rig.palm_pos()[0])} cup={fmt(cup()[0])} "
      f"palm-cup={float((rig.palm_pos()-cup()).norm(dim=-1).mean())*100:.1f}cm "
      f"pos_err={float(perr.mean())*1000:.1f}mm rot_err_raw={r_raw:.1f}deg rot_err_wrapped={r_wrap:.1f}deg "
      f"euler_tgt={fmt(torch.rad2deg(rig.palm_targets[0, 3:]))} euler_now={fmt(torch.rad2deg(rig.palm_pose_6d()[0, 3:]))}", flush=True)

# ---- sweep ------------------------------------------------------------------------------
for k, name in enumerate(("x", "y", "z")):
    for sgn in (+1.0, -1.0):
        env.reset()
        for _ in range(hold):
            env.step(torch.zeros(N, A, device=dev))
        p0 = rig.palm_pos().clone()
        a = torch.zeros(N, A, device=dev)
        a[:, k] = sgn
        for _ in range(args.sweep_steps):
            env.step(a)
        want = (target_env() - p0)[:, k].mean().item()
        got = (rig.palm_pos() - p0)[:, k].mean().item()
        err = float((target_env() - rig.palm_pos()).norm(dim=-1).mean())
        print(f"[sweep {name}{'+' if sgn > 0 else '-'}] 명령 {want*100:+.1f}cm 도달 {got*100:+.1f}cm 비율 "
              f"{(got/want if abs(want) > 1e-4 else float('nan')):.2f} 잔여오차 {err*1000:.0f}mm", flush=True)

# ---- grasp ------------------------------------------------------------------------------
env.reset()
for _ in range(hold):
    env.step(torch.zeros(N, A, device=dev))
bn = env.robot.data.body_names
i_th = bn.index(rig.profile.fingertip_bodies[0])
i_ix = bn.index(rig.profile.fingertip_bodies[1])
i_ix1 = bn.index(rig.profile.finger_sensor_bodies["index"][0])
mid_z = 0.5 * float(env.cfg.cup_mouth_z)
stats = dict(min_d=9.0, fmax=[0.0] * 5, grasp=0.0, lift=-1.0, tilt=0.0, minpocket=9.0)
T = args.wait_steps + args.enter_steps + args.close_steps + args.lift_steps
for u in range(T):
    pos = env.robot.data.body_pos_w - env.scene.env_origins[:, None, :]
    pocket = 0.5 * (pos[:, i_th] + pos[:, i_ix])
    f = pos[:, i_ix, :2] - pos[:, i_ix1, :2]
    f = f / (f.norm(dim=-1, keepdim=True) + 1e-9)
    n = torch.stack([-f[:, 1], f[:, 0]], dim=1)
    c = cup()
    phase = "wait" if u < args.wait_steps else "enter" if u < args.wait_steps + args.enter_steps else \
        "close" if u < args.wait_steps + args.enter_steps + args.close_steps else "lift"
    off = args.side_m * n if phase == "wait" else torch.zeros_like(n)
    goal_pocket = torch.cat([c[:, :2] + off, (c[:, 2] + mid_z).unsqueeze(1)], dim=1)
    tgt_palm = rig.palm_pos() + (goal_pocket - pocket)
    if phase == "lift":
        tgt_palm[:, 2] += 0.10
    a = torch.zeros(N, A, device=dev)
    a[:, 0:3] = to_action(tgt_palm - rig.anchor_env[:, :3])
    a[:, 6:HALF] = 1.0 if phase in ("close", "lift") else -1.0
    _, _, term, _, ex = env.step(a)
    d_pc = float((pocket[:, :2] - c[:, :2]).norm(dim=-1).mean())
    stats["minpocket"] = min(stats["minpocket"], d_pc)
    stats["min_d"] = min(stats["min_d"], float((rig.palm_pos() - c).norm(dim=-1).mean()))
    ff = rig.finger_forces()[0].tolist()
    stats["fmax"] = [max(a_, b_) for a_, b_ in zip(stats["fmax"], ff)]
    stats["grasp"] = max(stats["grasp"], float(ex["task/src_grasped"]))
    stats["lift"] = max(stats["lift"], float(ex["task/src_cup_lift"]))
    stats["tilt"] = max(stats["tilt"], float(ex["task/src_tilt_deg"]))
    if u % 20 == 0 or u == T - 1:
        print(f"[grasp {phase:5s} u={u:3d}] pocket-cup_xy={d_pc*1000:.0f}mm palm-cup={float((rig.palm_pos()-c).norm(dim=-1).mean())*100:.1f}cm "
              f"pos_err={float(ex['fabric/src_palm_err'])*1000:.0f}mm closure={float(rig.closure()[0]):.2f} "
              f"force[th,ix,md,rg,pk]={[round(v,2) for v in ff]} grasped={float(ex['task/src_grasped']):.2f} "
              f"lift={float(ex['task/src_cup_lift'])*100:+.1f}cm tilt={float(ex['task/src_tilt_deg']):.1f} "
              f"{'TERM' if bool(term.any()) else ''}", flush=True)
print(f"[grasp summary] 최소 포켓-컵 xy {stats['minpocket']*1000:.0f}mm · 최소 palm-컵 {stats['min_d']*100:.1f}cm · "
      f"손가락별 최대 힘 [th,ix,md,rg,pk]={[round(v,2) for v in stats['fmax']]} · 파지 최대 {stats['grasp']:.2f} · "
      f"들기 최대 {stats['lift']*100:+.1f}cm · 기울기 최대 {stats['tilt']:.1f}deg", flush=True)
env.close()
app.close()

"""RH56F1 붓기 트랙 대본 파지·들기 존재 증명 스윕(09.17) — "이 손·컵·제어 구성으로 들기가 가능한가"를 보상과 무관하게 잰다.

  env 마다 파지 높이(dz)·깊이(dn)·손 회전(yaw,pitch,roll)을 다르게 주고 같은 대본(위→내려옴→진입→오므림→들기→유지)을
  돌린다. 종료 판정은 끄되(비유한 상태만 유지) "켜져 있었다면 언제·무슨 사유로 끊겼을지"를 env 별로 기록한다.
  한 프로세스에서 측(src/rcv)·동결 임계를 바꿔 가며 반복한다(부팅 1 회).

  판정(env 별): 들기·유지 구간 파지 플래그 평균 >= 0.9 · 유지 구간 최소 들기 >= lift_ok · 최대 기울기 <= tilt_ok.
  strict = 위 + 진입 이후 종료 사유 없음.

    ../IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/probes/probe_rh_grasp_lift.py --num_envs 224
"""
import argparse
import itertools
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-rh_b_pour_fab_mimic")
parser.add_argument("--num_envs", type=int, default=224)
parser.add_argument("--sides", default="src,rcv")
parser.add_argument("--freeze", default="4.0", help="접촉 동결 임계 [N] 쉼표 목록(측마다 반복)")
parser.add_argument("--dz", default="-0.010,0.010,0.027,0.045", help="파지 높이(컵 원점 기준) [m]")
parser.add_argument("--dn", default="-0.020,-0.010,0.0,0.010", help="포켓 깊이 오프셋(열린 쪽 법선 방향) [m]")
parser.add_argument("--rot", default="0,0,0;0,15,0;0,-15,0;0,0,15;0,0,-15;15,0,0;-15,0,0",
                    help="손 회전 델타 [deg] yaw,pitch,roll 세미콜론 목록")
parser.add_argument("--side_m", type=float, default=0.08)
parser.add_argument("--safe_dz", type=float, default=0.08)
parser.add_argument("--lift_m", type=float, default=0.10)
parser.add_argument("--lift_ok", type=float, default=0.08)
parser.add_argument("--tilt_ok", type=float, default=20.0)
parser.add_argument("--close_steps", type=int, default=220)
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

DZ = [float(x) for x in args.dz.split(",")]
DN = [float(x) for x in args.dn.split(",")]
ROT = [tuple(float(v) for v in r.split(",")) for r in args.rot.split(";")]
COMBOS = list(itertools.product(range(len(DZ)), range(len(DN)), range(len(ROT))))

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
cfg.fabric_table_obstacle = bool(args.table_obstacle)
env = gym.make(args.task, cfg=cfg).unwrapped
N, A = env.num_envs, env.cfg.action_space
HALF = A // 2
dev = env.device
hold = int(env.cfg.hold_steps)
bn = env.robot.data.body_names

combo_id = torch.arange(N, device=dev) % len(COMBOS)
p_dz = torch.tensor([DZ[COMBOS[int(c)][0]] for c in combo_id], device=dev)
p_dn = torch.tensor([DN[COMBOS[int(c)][1]] for c in combo_id], device=dev)
p_rot = torch.tensor([ROT[COMBOS[int(c)][2]] for c in combo_id], device=dev)          # (N,3) deg yaw,pitch,roll

# ---- 종료 끄기(비유한 상태만 유지) + 사유 기록 ------------------------------------------------
CAUSES = ("drop", "arm_runaway", "mimic_qd", "mimic_err")
state = {"step": 0}
first_step = torch.full((N,), -1, device=dev, dtype=torch.long)
first_cause = torch.full((N,), -1, device=dev, dtype=torch.long)
_orig_dones = env._get_dones


def _dones_off():
    term, trunc = _orig_dones()
    c = env.cfg
    q, qd = env.robot.data.joint_pos, env.robot.data.joint_vel
    arm_qd = torch.cat([qd[:, env.src.arm_t], qd[:, env.rcv.arm_t]], dim=1)
    flags = torch.stack([
        env._dropped,
        (arm_qd.abs() > float(c.runaway_joint_vel)).any(dim=-1),
        env._mim_qd_streak >= int(c.mimic_runaway_qd_steps),
        (q[:, env._mim_dep_t] - env._mim_mult * q[:, env._mim_lead_t]).abs().max(dim=-1).values > float(c.mimic_runaway_err_rad),
    ], dim=1)
    new = flags.any(dim=1) & (first_step < 0)
    first_step[new] = state["step"]
    first_cause[new] = flags[new].float().argmax(dim=1)
    nonfinite = ~(torch.isfinite(q).all(dim=-1) & torch.isfinite(qd).all(dim=-1))
    return term & nonfinite, torch.zeros_like(trunc)


env._get_dones = _dones_off


def fmt(v, nd=3):
    return [round(float(x), nd) for x in v]


def run_trial(side: str, freeze_thr: float) -> list[dict]:
    env.cfg.contact_freeze_threshold = float(freeze_thr)
    SI = 0 if side == "src" else 1
    rig = env.src if SI == 0 else env.rcv
    cup = env.source_cup if SI == 0 else env.receiver_cup
    n_sign = 1.0 if SI == 0 else -1.0
    i_th = bn.index(rig.profile.fingertip_bodies[0])
    i_ix = bn.index(rig.profile.fingertip_bodies[1])
    i_ix1 = bn.index(rig.profile.finger_sensor_bodies["index"][0])
    fingers = list(rig.fingers)
    grp_b = [fingers.index(f) for f in rig.profile.contact_group_b]

    def cup_local():
        return env._local(cup.data.root_pos_w)

    def cup_tilt_deg():
        up = matrix_from_quat(cup.data.root_quat_w)[:, :, 2]
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
            r = torch.deg2rad(p_rot[:, k])
            a[:, SI * HALF + 3 + k] = scaled(r, float(rig.delta_lo[3 + k]), float(rig.delta_hi[3 + k]))
        a[:, SI * HALF + 6:(SI + 1) * HALF] = hand_cmd
        return a

    first_step.fill_(-1)
    first_cause.fill_(-1)
    state["step"] = 0
    env.reset()
    for _ in range(hold):
        env.step(torch.zeros(N, A, device=dev))
        state["step"] += 1
    rest = cup_local().clone()
    plan = [("above", 90), ("down", 70), ("enter", 110), ("close", args.close_steps), ("lift", 100), ("hold", 90)]
    t_enter = state["step"] + 160
    grasp_sum = torch.zeros(N, device=dev)
    grasp_n = 0
    tilt_max = torch.zeros(N, device=dev)
    lift_min_hold = torch.full((N,), 1e9, device=dev)
    snap = {}
    for name, steps in plan:
        for u in range(steps):
            c = cup_local()
            _, n = hand_frame()
            n3 = torch.cat([n, torch.zeros(N, 1, device=dev)], dim=1)
            grip = torch.cat([c[:, :2], (c[:, 2] + p_dz).unsqueeze(1)], dim=1) + p_dn.unsqueeze(1) * n3
            if name == "above":
                a = servo(grip + args.side_m * n3, -1.0, extra_z=args.safe_dz)
            elif name == "down":
                a = servo(grip + args.side_m * n3, -1.0)
            elif name == "enter":
                a = servo(grip, -1.0)
            elif name == "close":
                a = servo(grip, 1.0)
            else:
                k_up = min(1.0, (u + 1) / 60.0) if name == "lift" else 1.0
                a = servo(grip, 1.0, extra_z=args.lift_m * k_up)
            env.step(a)
            state["step"] += 1
            if name in ("lift", "hold"):
                grasp_sum += rig.grasped(rig.finger_forces()).float()
                grasp_n += 1
                tilt_max = torch.maximum(tilt_max, cup_tilt_deg())
            if name == "hold":
                lift_min_hold = torch.minimum(lift_min_hold, cup_local()[:, 2] - rest[:, 2])
        if name == "close":
            # 오므림 끝 스냅샷: 폐쇄도·손가락 힘·컵 좌표계 손끝 위치·컵이 밀린 정도.
            c = cup_local()
            R = matrix_from_quat(cup.data.root_quat_w)
            tips = torch.einsum("nji,nfj->nfi", R, rig.tips_pos() - c.unsqueeze(1))      # (N,F,3) 컵 좌표계
            th = tips[:, 0, :2]
            op = tips[:, grp_b[:2], :2].mean(dim=1)
            chord = (th[:, 0] * op[:, 1] - th[:, 1] * op[:, 0]).abs() / ((op - th).norm(dim=-1) + 1e-9)
            snap = dict(
                closure=rig.closure().clone(), forces=rig.finger_forces().clone(), palm_f=rig.palm_force().clone(),
                tips=tips.clone(), chord=chord, shift=(c[:, :2] - rest[:, :2]).norm(dim=-1), tilt=cup_tilt_deg().clone(),
                rel_z=(c[:, 2] - rig.palm_pos()[:, 2]).clone(), euler=torch.rad2deg(rig.palm_pose_6d()[:, 3:]).clone(),
            )
    c = cup_local()
    lift_end = c[:, 2] - rest[:, 2]
    slip = (c[:, 2] - rig.palm_pos()[:, 2]) - snap["rel_z"]
    grasp = grasp_sum / max(grasp_n, 1)
    phys = (grasp >= 0.9) & (lift_min_hold >= args.lift_ok) & (tilt_max <= args.tilt_ok)
    term_after = (first_step >= t_enter)
    strict = phys & ~term_after
    fr = cup.root_physx_view.get_material_properties().to(dev)[:, :, 0].mean(dim=1)
    rows = []
    for i in range(N):
        zi, ni, ri = COMBOS[int(combo_id[i])]
        rows.append(dict(
            side=side, freeze=freeze_thr, env=i, dz=DZ[zi], dn=DN[ni], rot=list(ROT[ri]),
            phys=bool(phys[i]), strict=bool(strict[i]), grasp=round(float(grasp[i]), 3),
            lift_min=round(float(lift_min_hold[i]), 4), lift_end=round(float(lift_end[i]), 4),
            tilt_max=round(float(tilt_max[i]), 1), slip=round(float(slip[i]), 4),
            term_step=int(first_step[i]), term_cause=(CAUSES[int(first_cause[i])] if int(first_cause[i]) >= 0 else ""),
            term_after_enter=bool(term_after[i]), mu=round(float(fr[i]), 3),
            close=dict(closure=round(float(snap["closure"][i]), 3), forces=fmt(snap["forces"][i], 2),
                       palm_f=round(float(snap["palm_f"][i]), 2), chord_mm=round(float(snap["chord"][i]) * 1000, 1),
                       tip_z_mm=fmt(snap["tips"][i, :, 2] * 1000, 1),
                       tip_r_mm=fmt(snap["tips"][i, :, :2].norm(dim=-1) * 1000, 1),
                       shift_mm=round(float(snap["shift"][i]) * 1000, 1), tilt=round(float(snap["tilt"][i]), 1),
                       euler=fmt(snap["euler"][i], 1)),
        ))
    return rows


def rate(rows, key):
    return sum(1 for r in rows if r[key]) / max(len(rows), 1)


def mean(rows, f):
    v = [f(r) for r in rows]
    return sum(v) / max(len(v), 1)


def report(rows, side, thr):
    tag = f"{side} thr={thr:g}"
    n = len(rows)
    print(f"\n==== [{tag}] env {n} · phys {sum(r['phys'] for r in rows)} · strict {sum(r['strict'] for r in rows)} ====", flush=True)
    hist = {}
    for r in rows:
        if r["term_after_enter"]:
            hist[r["term_cause"]] = hist.get(r["term_cause"], 0) + 1
    print(f"[{tag}] 진입 이후 종료 사유(꺼 둠): {hist} / {n}", flush=True)
    print(f"[{tag}] 전체 평균: grasp {mean(rows, lambda r: r['grasp']):.2f} · lift_min {mean(rows, lambda r: r['lift_min'])*100:.1f}cm · "
          f"tilt_max {mean(rows, lambda r: r['tilt_max']):.1f}° · slip {mean(rows, lambda r: r['slip'])*1000:.1f}mm · "
          f"오므림 끝 closure {mean(rows, lambda r: r['close']['closure']):.2f} · 밀림 {mean(rows, lambda r: r['close']['shift_mm']):.1f}mm · "
          f"오므림 끝 기울기 {mean(rows, lambda r: r['close']['tilt']):.1f}°", flush=True)
    for label, key, vals in (("dz", "dz", DZ), ("dn", "dn", DN), ("rot", "rot", [list(r) for r in ROT])):
        for v in vals:
            sub = [r for r in rows if r[key] == v]
            print(f"[{tag}] {label}={v}: phys {rate(sub, 'phys'):.2f} strict {rate(sub, 'strict'):.2f} grasp {mean(sub, lambda r: r['grasp']):.2f} "
                  f"lift_min {mean(sub, lambda r: r['lift_min'])*100:.1f}cm tilt {mean(sub, lambda r: r['tilt_max']):.1f}° "
                  f"slip {mean(sub, lambda r: r['slip'])*1000:.1f}mm chord {mean(sub, lambda r: r['close']['chord_mm']):.1f}mm "
                  f"term {rate(sub, 'term_after_enter'):.2f}", flush=True)
    groups = {}
    for r in rows:
        groups.setdefault((r["dz"], r["dn"], tuple(r["rot"])), []).append(r)
    ranked = sorted(groups.items(), key=lambda kv: (-rate(kv[1], "strict"), -rate(kv[1], "phys"),
                                                    mean(kv[1], lambda r: r["tilt_max"])))
    print(f"[{tag}] 상위 조합(dz, dn, rot):", flush=True)
    for (dz, dn, rot), sub in ranked[:8]:
        r0 = sub[0]
        print(f"[{tag}]   dz={dz:+.3f} dn={dn:+.3f} rot={list(rot)} · strict {rate(sub, 'strict'):.2f} phys {rate(sub, 'phys'):.2f} · "
              f"grasp {mean(sub, lambda r: r['grasp']):.2f} lift_min {mean(sub, lambda r: r['lift_min'])*100:.1f}cm "
              f"tilt {mean(sub, lambda r: r['tilt_max']):.1f}° slip {mean(sub, lambda r: r['slip'])*1000:.1f}mm · "
              f"힘 {r0['close']['forces']} palm {r0['close']['palm_f']} closure {r0['close']['closure']} "
              f"chord {r0['close']['chord_mm']}mm tip_z {r0['close']['tip_z_mm']}", flush=True)
    base = [r for r in rows if abs(r["dz"] - 0.027) < 1e-6 and r["dn"] == 0.0 and r["rot"] == [0.0, 0.0, 0.0]]
    for r in base[:2]:
        print(f"[{tag}] 기준(dz=0.027 dn=0 rot=0) env{r['env']}: phys={r['phys']} strict={r['strict']} grasp={r['grasp']} "
              f"lift_min={r['lift_min']*100:.1f}cm tilt={r['tilt_max']}° slip={r['slip']*1000:.1f}mm term={r['term_cause']}@{r['term_step']} "
              f"mu={r['mu']} close={r['close']}", flush=True)


all_rows = []
print(f"[cfg] N={N} combos={len(COMBOS)} dz={DZ} dn={DN} rot={ROT} close_steps={args.close_steps} "
      f"grasp_thr={env.cfg.contact_force_threshold} close_speed={env.cfg.synergy_close_speed}", flush=True)
for side in [s.strip() for s in args.sides.split(",") if s.strip()]:
    for thr in [float(x) for x in args.freeze.split(",")]:
        rows = run_trial(side, thr)
        report(rows, side, thr)
        all_rows += rows
if args.out_json:
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(all_rows, fh, ensure_ascii=False)
    print(f"[out] {args.out_json} ({len(all_rows)} rows)", flush=True)
env.close()
app.close()

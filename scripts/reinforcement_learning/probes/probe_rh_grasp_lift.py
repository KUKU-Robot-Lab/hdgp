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
parser.add_argument("--v_max", type=float, default=0.10,
                    help="포켓 목표 이동 속도 상한 [m/s]. 0 이면 무제한(09.17 cup_trace: 무제한은 ~1.3 m/s 로 날아가 u=10 에 손가락이 컵을 친다)")
parser.add_argument("--ref", default="rest", choices=["rest", "live"],
                    help="파지점 기준 컵 위치. rest=정지 위치 고정, live=매 스텝 현재 위치(넘어진 컵을 쫓아간다)")
parser.add_argument("--plans", default="full",
                    help="쉼표 구분. full=above->down->enter->close->lift->hold, direct=홈에서 바로 enter(홈이 이미 컵 옆이면 우회가 컵을 친다)")
parser.add_argument("--side_sign", default="auto", choices=["auto", "pos", "neg"],
                    help="옆 대기점 부호. auto=정지 자세에서 포켓이 컵의 n 어느 쪽에 있는지 재서 그쪽으로 물러난다")
parser.add_argument("--above_steps", type=int, default=200)
parser.add_argument("--down_steps", type=int, default=90)
parser.add_argument("--enter_steps", type=int, default=110)
parser.add_argument("--pre_tilt_deg", type=float, default=5.0, help="오므리기 전 컵 교란 판정: 기울기 [deg]")
parser.add_argument("--pre_shift_m", type=float, default=0.010, help="오므리기 전 컵 교란 판정: 수평 밀림 [m]")
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


def run_trial(side: str, freeze_thr: float, plan_name: str = "full") -> list[dict]:
    env.cfg.contact_freeze_threshold = float(freeze_thr)
    SI = 0 if side == "src" else 1
    rig = env.src if SI == 0 else env.rcv
    cup = env.source_cup if SI == 0 else env.receiver_cup
    n_sign = 1.0 if SI == 0 else -1.0
    i_th = bn.index(rig.profile.fingertip_bodies[0])
    i_ix = bn.index(rig.profile.fingertip_bodies[1])
    i_ix1 = bn.index(rig.profile.finger_sensor_bodies["index"][0])
    fingers = list(rig.fingers)
    global _FINGERS_ORDER
    _FINGERS_ORDER = fingers
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

    def servo(goal_pocket, hand_cmd, rot_k=1.0):
        pocket, _ = hand_frame()
        tgt = rig.palm_pos() + (goal_pocket - pocket)
        d = tgt - rig.anchor_env[:, :3]
        a = torch.zeros(N, A, device=dev)
        for k in range(3):
            a[:, SI * HALF + k] = scaled(d[:, k], float(rig.delta_lo[k]), float(rig.delta_hi[k]))
            r = torch.deg2rad(p_rot[:, k]) * rot_k
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
    # 09.17 cup_trace: 목표를 한 번에 멀리 주면 손이 ~1.3 m/s 로 날아가 u=10 에 손가락이 컵을 친다(오므리기 전에 컵이 넘어짐).
    # 그래서 (1) 포켓 목표를 v_max 로 속도 제한해 흘리고 (2) 파지점 기준을 정지 위치로 고정한다(넘어진 컵을 쫓지 않게).
    step_dt = float(env.cfg.sim.dt) * int(env.cfg.decimation)
    max_step = args.v_max * step_dt if args.v_max > 0 else float("inf")
    plan = [("above", args.above_steps), ("down", args.down_steps), ("enter", args.enter_steps),
            ("close", args.close_steps), ("lift", 100), ("hold", 90)]
    if plan_name == "direct":
        plan = [p for p in plan if p[0] not in ("above", "down")]
    # 09.17 정지 자세 기하. 컵이 포켓에서 본 n 의 어느 쪽인가(=옆 대기점 부호)와 열린 손끝-컵 간격.
    pk0, n0 = hand_frame()
    f0 = n_sign * torch.stack([n0[:, 1], -n0[:, 0]], dim=1)
    def _nfz(v):
        return torch.stack([(v[..., :2] * n0.view(N, *([1] * (v.dim() - 2)), 2)).sum(-1),
                            (v[..., :2] * f0.view(N, *([1] * (v.dim() - 2)), 2)).sum(-1), v[..., 2]], dim=-1)
    pk_rel = _nfz(pk0 - rest)
    palm_rel = _nfz(rig.palm_pos() - rest)
    tips_rel = _nfz(rig.tips_pos() - rest.unsqueeze(1))
    if args.side_sign == "auto":
        sgn = torch.sign(pk_rel[:, 0])
        sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    else:
        sgn = torch.full((N,), 1.0 if args.side_sign == "pos" else -1.0, device=dev)
    gtag = f"{side} {plan_name}"
    print(f"[{gtag}] 정지 기하(컵 기준 n,f,z mm): 포켓 {[round(float(x) * 1000, 1) for x in pk_rel.mean(0)]} "
          f"손바닥 {[round(float(x) * 1000, 1) for x in palm_rel.mean(0)]} · 옆 대기점 부호 +{int((sgn > 0).sum())}/-{int((sgn < 0).sum())}", flush=True)
    for fi, fname in enumerate(fingers):
        t = tips_rel[:, fi]
        print(f"[{gtag}]   손끝 {fname}: n,f,z {[round(float(x) * 1000, 1) for x in t.mean(0)]} mm · 컵 축까지 수평 "
              f"{float(t[:, :2].norm(dim=-1).mean()) * 1000:.1f} (최소 {float(t[:, :2].norm(dim=-1).min()) * 1000:.1f}) mm", flush=True)
    pre_rel = torch.zeros(N, 3, device=dev)
    pre_fid = torch.full((N,), -1, device=dev, dtype=torch.long)
    n_total = hold + sum(k for _, k in plan)
    if n_total >= int(env.max_episode_length):
        raise SystemExit(f"대본 {n_total} 스텝 >= 에피소드 {int(env.max_episode_length)} 스텝 - time_out 리셋이 끼어든다")
    t_enter = state["step"] + sum(k for nm, k in plan if nm in ("above", "down"))
    g_cmd = hand_frame()[0].clone()
    pre_step = torch.full((N,), -1, device=dev, dtype=torch.long)
    pre_phase = torch.full((N,), -1, device=dev, dtype=torch.long)
    pre_finger = torch.zeros(N, device=dev)
    pre_palm = torch.zeros(N, device=dev)
    pre_speed = torch.zeros(N, device=dev)
    phase_end = {}
    # 오므리는 동안 컵이 처음 교란된 시점(오므림 시작 자세 기준)과 그때의 손가락별 힘.
    cl_step = torch.full((N,), -1, device=dev, dtype=torch.long)
    cl_forces = torch.zeros(N, len(fingers), device=dev)
    cl_palm = torch.zeros(N, device=dev)
    cl_closure = torch.zeros(N, device=dev)
    cl_first_step = torch.full((N,), -1, device=dev, dtype=torch.long)
    cl_first_fid = torch.full((N,), -1, device=dev, dtype=torch.long)
    cl_ref, cl_tilt0 = None, None
    grasp_sum = torch.zeros(N, device=dev)
    grasp_n = 0
    tilt_max = torch.zeros(N, device=dev)
    lift_min_hold = torch.full((N,), 1e9, device=dev)
    snap = {}
    for pi, (name, steps) in enumerate(plan):
        for u in range(steps):
            c = rest if args.ref == "rest" else cup_local()
            pocket_now, n = hand_frame()
            n3 = torch.cat([n, torch.zeros(N, 1, device=dev)], dim=1)
            grip = torch.cat([c[:, :2], (c[:, 2] + p_dz).unsqueeze(1)], dim=1) + p_dn.unsqueeze(1) * n3
            ez = 0.0
            if name == "above":
                goal, hand_cmd, ez = grip + args.side_m * sgn.unsqueeze(1) * n3, -1.0, args.safe_dz
            elif name == "down":
                goal, hand_cmd = grip + args.side_m * sgn.unsqueeze(1) * n3, -1.0
            elif name == "enter":
                goal, hand_cmd = grip, -1.0
            elif name == "close":
                goal, hand_cmd = grip, 1.0
            else:
                k_up = min(1.0, (u + 1) / 60.0) if name == "lift" else 1.0
                goal, hand_cmd, ez = grip, 1.0, args.lift_m * k_up
            goal = goal.clone()
            goal[:, 2] += ez
            d = goal - g_cmd
            dist = d.norm(dim=-1, keepdim=True)
            g_cmd = g_cmd + d * torch.clamp(max_step / (dist + 1e-9), max=1.0)
            # 회전 델타도 첫 60 스텝에 걸쳐 올린다(한 번에 주면 손끝이 수 cm 휘둘린다).
            a = servo(g_cmd, hand_cmd, rot_k=(min(1.0, (u + 1) / 60.0) if pi == 0 else 1.0))
            prev_pocket = pocket_now.clone()
            if name == "close" and u == 0:
                cl_ref, cl_tilt0 = cup_local().clone(), cup_tilt_deg().clone()
            env.step(a)
            state["step"] += 1
            if name == "close":
                ff = rig.finger_forces()
                first = (cl_first_step < 0) & (ff.max(dim=1).values > 0.05)
                if bool(first.any()):
                    cl_first_step[first] = u
                    cl_first_fid[first] = ff.argmax(dim=1)[first]
                hit = (cl_step < 0) & ((cup_tilt_deg() - cl_tilt0 > args.pre_tilt_deg)
                                       | ((cup_local()[:, :2] - cl_ref[:, :2]).norm(dim=-1) > args.pre_shift_m))
                if bool(hit.any()):
                    cl_step[hit] = u
                    cl_forces[hit] = ff[hit]
                    cl_palm[hit] = rig.palm_force()[hit]
                    cl_closure[hit] = rig.closure()[hit]
            if name in ("above", "down", "enter"):
                # 오므리기 전에 컵이 이미 교란됐는가(손이 지나가며 침). 첫 교란 시점의 단계·손가락 힘·손 속도를 남긴다.
                cl = cup_local()
                hit = (pre_step < 0) & ((cup_tilt_deg() > args.pre_tilt_deg)
                                        | ((cl[:, :2] - rest[:, :2]).norm(dim=-1) > args.pre_shift_m))
                if bool(hit.any()):
                    pre_step[hit] = state["step"]
                    pre_phase[hit] = pi
                    pre_finger[hit] = rig.finger_forces().max(dim=1).values[hit]
                    pre_palm[hit] = rig.palm_force()[hit]
                    pre_speed[hit] = ((hand_frame()[0] - prev_pocket).norm(dim=-1) / step_dt)[hit]
                    pre_rel[hit] = _nfz(hand_frame()[0] - rest)[hit]
                    pre_fid[hit] = rig.finger_forces().argmax(dim=1)[hit]
            if name in ("lift", "hold"):
                grasp_sum += rig.grasped(rig.finger_forces()).float()
                grasp_n += 1
                tilt_max = torch.maximum(tilt_max, cup_tilt_deg())
            if name == "hold":
                lift_min_hold = torch.minimum(lift_min_hold, cup_local()[:, 2] - rest[:, 2])
        pk = hand_frame()[0]
        phase_end[name] = (float((g_cmd - goal).norm(dim=-1).max()), float((pk - g_cmd).norm(dim=-1).mean()),
                           float((pk - g_cmd).norm(dim=-1).max()))
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
            side=side, freeze=freeze_thr, plan=plan_name, env=i, dz=DZ[zi], dn=DN[ni], rot=list(ROT[ri]),
            pre_rel_mm=[round(float(x) * 1000, 1) for x in pre_rel[i]],
            pre_fid=(fingers[int(pre_fid[i])] if int(pre_fid[i]) >= 0 else ""),
            phys=bool(phys[i]), strict=bool(strict[i]), grasp=round(float(grasp[i]), 3),
            lift_min=round(float(lift_min_hold[i]), 4), lift_end=round(float(lift_end[i]), 4),
            tilt_max=round(float(tilt_max[i]), 1), slip=round(float(slip[i]), 4),
            term_step=int(first_step[i]), term_cause=(CAUSES[int(first_cause[i])] if int(first_cause[i]) >= 0 else ""),
            term_after_enter=bool(term_after[i]), mu=round(float(fr[i]), 3),
            pre_step=int(pre_step[i]), pre_phase=(plan[int(pre_phase[i])][0] if int(pre_phase[i]) >= 0 else ""),
            pre_finger=round(float(pre_finger[i]), 2), pre_palm=round(float(pre_palm[i]), 2),
            pre_speed=round(float(pre_speed[i]), 3), clean=bool(pre_step[i] < 0),
            cl=dict(step=int(cl_step[i]), forces=fmt(cl_forces[i], 2), palm=round(float(cl_palm[i]), 2),
                    closure=round(float(cl_closure[i]), 3), first_step=int(cl_first_step[i]),
                    first_fid=(fingers[int(cl_first_fid[i])] if int(cl_first_fid[i]) >= 0 else "")),
            close=dict(closure=round(float(snap["closure"][i]), 3), forces=fmt(snap["forces"][i], 2),
                       palm_f=round(float(snap["palm_f"][i]), 2), chord_mm=round(float(snap["chord"][i]) * 1000, 1),
                       tip_z_mm=fmt(snap["tips"][i, :, 2] * 1000, 1),
                       tip_r_mm=fmt(snap["tips"][i, :, :2].norm(dim=-1) * 1000, 1),
                       shift_mm=round(float(snap["shift"][i]) * 1000, 1), tilt=round(float(snap["tilt"][i]), 1),
                       euler=fmt(snap["euler"][i], 1)),
        ))
    for nm, (gap_goal, lag_mean, lag_max) in phase_end.items():
        print(f"[{side} thr={freeze_thr:g}] 단계 끝 {nm}: 목표-지령 잔차 최대 {gap_goal*1000:.1f}mm · "
              f"지령-포켓 추종 오차 평균 {lag_mean*1000:.1f}mm 최대 {lag_max*1000:.1f}mm", flush=True)
    return rows


def rate(rows, key):
    return sum(1 for r in rows if r[key]) / max(len(rows), 1)


def mean(rows, f):
    v = [f(r) for r in rows]
    return sum(v) / max(len(v), 1)


def report(rows, side, thr):
    fingers_order = list(_FINGERS_ORDER)
    tag = f"{side} thr={thr:g}"
    n = len(rows)
    print(f"\n==== [{tag}] env {n} · phys {sum(r['phys'] for r in rows)} · strict {sum(r['strict'] for r in rows)} ====", flush=True)
    hist = {}
    for r in rows:
        if r["term_after_enter"]:
            hist[r["term_cause"]] = hist.get(r["term_cause"], 0) + 1
    print(f"[{tag}] 진입 이후 종료 사유(꺼 둠): {hist} / {n}", flush=True)
    pre = {}
    for r in rows:
        if not r["clean"]:
            pre[r["pre_phase"]] = pre.get(r["pre_phase"], 0) + 1
    dirty = [r for r in rows if not r["clean"]]
    clean = [r for r in rows if r["clean"]]
    print(f"[{tag}] 오므리기 전 컵 교란(기울기>{args.pre_tilt_deg:g}° 또는 밀림>{args.pre_shift_m*1000:g}mm): {len(dirty)}/{n} 단계별 {pre}"
          + (f" · 교란 시점 손가락 힘 평균 {mean(dirty, lambda r: r['pre_finger']):.2f}N 손 속도 평균 {mean(dirty, lambda r: r['pre_speed']):.3f}m/s"
             if dirty else ""), flush=True)
    if dirty:
        fid = {}
        for r in dirty:
            fid[r["pre_fid"]] = fid.get(r["pre_fid"], 0) + 1
        zero_f = sum(1 for r in dirty if r["pre_finger"] < 0.05 and r["pre_palm"] < 0.05)
        print(f"[{tag}]   교란 시점 최대 힘 손가락 {fid} · 손 힘 0(손가락·손바닥 <0.05N) {zero_f}/{len(dirty)} · "
              f"그때 포켓 위치(컵 기준 n,f,z mm) 평균 {[round(mean(dirty, lambda r, k=k: r['pre_rel_mm'][k]), 1) for k in range(3)]} · "
              f"교란 스텝 중앙 {sorted(r['pre_step'] for r in dirty)[len(dirty) // 2]}", flush=True)
    print(f"[{tag}] 교란 없는 env {len(clean)}: phys {rate(clean, 'phys'):.2f} strict {rate(clean, 'strict'):.2f} "
          f"grasp {mean(clean, lambda r: r['grasp']):.2f} lift_min {mean(clean, lambda r: r['lift_min'])*100:.1f}cm "
          f"tilt_max {mean(clean, lambda r: r['tilt_max']):.1f}° slip {mean(clean, lambda r: r['slip'])*1000:.1f}mm", flush=True)
    med = lambda v: (sorted(v)[len(v) // 2] if v else float("nan"))
    hitc = [r for r in clean if r["cl"]["step"] >= 0]
    print(f"[{tag}] 오므리는 중 컵 교란(오므림 시작 자세 기준, 접근 교란 없는 env): {len(hitc)}/{len(clean)}", flush=True)
    if hitc:
        fid, first = {}, {}
        for r in hitc:
            f = r["cl"]["forces"]
            k = fingers_order[max(range(len(f)), key=lambda j: f[j])] if max(f) > 0.05 else "none"
            fid[k] = fid.get(k, 0) + 1
            first[r["cl"]["first_fid"] or "none"] = first.get(r["cl"]["first_fid"] or "none", 0) + 1
        one = sum(1 for r in hitc if (r["cl"]["forces"][0] > 0.05) != (max(r["cl"]["forces"][1:]) > 0.05))
        two = sum(1 for r in hitc if r["cl"]["forces"][0] > 0.05 and max(r["cl"]["forces"][1:]) > 0.05)
        zero = sum(1 for r in hitc if max(r["cl"]["forces"]) <= 0.05 and r["cl"]["palm"] <= 0.05)
        print(f"[{tag}]   교란 시점 최대 힘 손가락 {fid} · 첫 접촉 손가락 {first} · 한쪽만 접촉 {one} 양쪽 접촉 {two} 손 힘 0 {zero} · "
              f"최대 손가락 힘 중앙 {med([max(r['cl']['forces']) for r in hitc]):.2f}N · closure 중앙 {med([r['cl']['closure'] for r in hitc]):.2f} · "
              f"오므림 스텝 중앙 {med([r['cl']['step'] for r in hitc])} (첫 접촉 스텝 중앙 {med([r['cl']['first_step'] for r in hitc])})", flush=True)
    keep = [r for r in clean if r["cl"]["step"] < 0]
    if keep:
        two = [r for r in keep if r["close"]["forces"][0] > 0.05 and max(r["close"]["forces"][1:]) > 0.05]
        print(f"[{tag}]   오므림 끝까지 컵 안 움직인 env {len(keep)}: 양쪽 접촉 {len(two)} · 그중 lift_min>2cm {sum(1 for r in two if r['lift_min'] > 0.02)} · "
              f"힘 합 중앙 {med([sum(r['close']['forces']) for r in two]):.2f}N · 엄지 끝 z 중앙 {med([r['close']['tip_z_mm'][0] for r in two]):.0f}mm", flush=True)
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
print(f"[cfg] N={N} combos={len(COMBOS)} dz={DZ} dn={DN} rot={ROT} close_steps={args.close_steps} v_max={args.v_max} ref={args.ref} "
      f"grasp_thr={env.cfg.contact_force_threshold} close_speed={env.cfg.synergy_close_speed}", flush=True)
for side in [s.strip() for s in args.sides.split(",") if s.strip()]:
    for thr in [float(x) for x in args.freeze.split(",")]:
        for plan_name in [x.strip() for x in args.plans.split(",") if x.strip()]:
            rows = run_trial(side, thr, plan_name)
            report(rows, f"{side} {plan_name}", thr)
            all_rows += rows
if args.out_json:
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(all_rows, fh, ensure_ascii=False)
    print(f"[out] {args.out_json} ({len(all_rows)} rows)", flush=True)
env.close()
app.close()

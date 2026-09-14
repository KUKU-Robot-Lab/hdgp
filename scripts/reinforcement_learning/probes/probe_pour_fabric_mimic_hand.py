"""pour_fabric_mimic 손 진단 — 팔 고정(a=0) 상태에서 손만 닫고 리더/종속 관절·접촉을 관절별로 찍는다.

    ./isaaclab.sh -p scripts/reinforcement_learning/probes/probe_pour_fabric_mimic_hand.py --num_envs 4
"""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-rh_b_pour_fab_mimic")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=260)
parser.add_argument("--approach", type=str, default="", help="'cup' = 케이지를 컵 원점+z_off 로 보낸다")
parser.add_argument("--z_off", type=float, default=0.04, help="케이지 목표 높이 = 컵 원점 + z_off [m]")
parser.add_argument("--dey", type=float, default=0.0, help="palm pitch(ey) 델타 [deg] (음수 = 손가락을 수평 쪽으로)")
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--pocket_n", type=float, default=0.03)
parser.add_argument("--palm_dz", type=float, default=0.0, help="cupf: palm z = cup z + palm_dz")
parser.add_argument("--quiet2", action="store_true")
parser.add_argument("--cup_scale", type=float, default=-1.0, help="컵 스케일 override(<0 = cfg 값)")
parser.add_argument("--side_axis", type=str, default="x", choices=["x", "y"])
parser.add_argument("--side_m", type=float, default=0.10, help="cup2: 컵 옆 대기 오프셋(−y, 바깥쪽) [m]")
parser.add_argument("--table_obstacle", type=int, default=1, help="0 = fabric 테이블 장애물 OFF(팔이 못 내려가는 원인 분리용)")
parser.add_argument("--cage_dz", type=float, default=-0.018, help="palm→케이지 z 오프셋 [m] (Track B 실측 −0.018)")
parser.add_argument("--print_q", action="store_true", help="마지막에 팔 관절값·palm pose 를 찍는다(홈 캘리브)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(); args.headless = True
app = AppLauncher(args).app
import gymnasium as gym, torch
from isaaclab_tasks.utils import parse_env_cfg
import openarm.tasks  # noqa
import openarm.agnostic.tasks.pour_fabric_mimic.config  # noqa
cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
cfg.fabric_table_obstacle = bool(args.table_obstacle)
if args.cup_scale > 0:
    cfg.cup_scale = args.cup_scale
    from openarm.agnostic.tasks.pour_fabric_mimic.pour_fabric_env_cfg import resolve_cfg as _rc; _rc(cfg)
env = gym.make(args.task, cfg=cfg).unwrapped
obs, _ = env.reset(); N, A, hold, half = env.num_envs, env.cfg.action_space, int(env.cfg.hold_steps), env.cfg.action_space // 2
rig = env.src; jn = env.robot.data.joint_names
side = rig.profile.hand_joint_names[0].split("_hj_")[0]
hand_all = [i for i, n in enumerate(jn) if n.startswith(f"{side}_hj_")]
def cup_local(): return env._local(env.source_cup.data.root_pos_w)
ex_prev = None
for t in range(args.steps):
    a = torch.zeros(N, A, device=env.device)
    u = t - hold
    # 케이지(손가락 사이 공간) 중심 = palm + (+0.071, +0.034, −0.018) (grasp_fj_rh 09.07 실측, 우손).
    # 목표: 케이지가 컵 원점 +4 cm 높이에 오도록 palm 을 보낸다. 델타는 앵커 기준(a=0=앵커).
    CAGE_OFF = torch.tensor([0.071, 0.034, args.cage_dz], device=env.device)
    if u >= 0 and args.approach == "cupf":
        # ★손가락 방향 기반 접근(09.14 기하 덤프: 이 손의 '열린' 손가락은 palm 프레임에서 45° 굽어 있고
        #   포켓은 손가락 체인의 **오목한 쪽**(체인 진행 방향의 왼쪽 n)에 있다). 포켓 = 검지 체인 중점 + 0.03·n,
        #   대기 = 포켓 + side_m·n (n 쪽은 엄지가 위(z)에 있어 비어 있다), 진입 = −n 방향.
        bn = env.robot.data.body_names; pos = env.robot.data.body_pos_w[:, :, :] - env.scene.env_origins[:, None, :]
        i1 = bn.index(rig.profile.finger_sensor_bodies["index"][0]); it = bn.index(rig.profile.fingertip_bodies[1])
        f = pos[:, it, :2] - pos[:, i1, :2]; f = f / (f.norm(dim=-1, keepdim=True) + 1e-9)
        n = torch.stack([-f[:, 1], f[:, 0]], dim=1)
        pocket = 0.5 * (pos[:, it, :2] + pos[:, i1, :2]) + args.pocket_n * n
        wait = args.side_m * n if u < 80 else torch.zeros_like(n)
        d_xy = cup_local()[:, :2] + wait - pocket                      # 포켓을 컵으로 옮기는 xy 이동
        cur_palm = rig.palm_pos()
        tgt_palm = torch.cat([cur_palm[:, :2] + d_xy, (cup_local()[:, 2] + args.palm_dz).unsqueeze(1)], dim=1)
        d = tgt_palm - rig.anchor_env[:, :3]
        for k in range(3):
            v = d[:, k]
            a[:, k] = torch.where(v >= 0, (v / rig.delta_hi[k]).clamp(max=1.0), (v / rig.delta_lo[k]).clamp(max=1.0) * -1.0)
        if u in (79, 100, 120, 140) and not args.quiet2:
            print(f"[cupf u={u}] f={[round(float(x),2) for x in f[0]]} n={[round(float(x),2) for x in n[0]]} pocket-cup={[round(float(x),3) for x in (pocket[0]-cup_local()[0,:2])]} tilt={float(ex_prev['task/src_tilt_deg']) if ex_prev else 0:.1f}", flush=True)
    if u >= 0 and args.approach == "trackb2":
        # 2단: A(0~100) xy 만 컵 위로(케이지 xy = 컵, palm z 는 홈 유지 → 소지 손끝이 림 위) · B(100~200) z 하강.
        goal = cup_local() + torch.tensor([0.0, 0.0, args.z_off], device=env.device) - CAGE_OFF
        d = goal - rig.anchor_env[:, :3]
        fa = min(1.0, (u + 1) / 100.0); fb = max(0.0, min(1.0, (u - 100) / 100.0))
        for k in range(3):
            v = d[:, k] * (fa if k < 2 else fb)
            a[:, k] = torch.where(v >= 0, (v / rig.delta_hi[k]).clamp(max=1.0), (v / rig.delta_lo[k]).clamp(max=1.0) * -1.0)
    if u >= 0 and args.approach == "trackb":
        # grasp_fj_rh 09.07 실측 경로 재현: 홈 케이지−물체 = (+0.3, −105, +95) mm → 케이지를 (0, +0.105, −0.095) 옮긴다.
        #   물체는 스폰 (0.38, −0.16) 그대로. 0~120 스텝 y·z 동시 접근(속도 제한: 스텝당 최대 slow_m), 그 뒤 폐쇄.
        goal = cup_local() + torch.tensor([0.0, 0.0, args.z_off], device=env.device) - CAGE_OFF
        d = goal - rig.anchor_env[:, :3]
        frac = min(1.0, (u + 1) / 120.0)
        for k in range(3):
            v = d[:, k] * frac
            a[:, k] = torch.where(v >= 0, (v / rig.delta_hi[k]).clamp(max=1.0), (v / rig.delta_lo[k]).clamp(max=1.0) * -1.0)
    if u >= 0 and args.approach in ("cup", "cup2"):
        # cup2: 컵 **옆**(바깥쪽 −y 로 side_m)에 먼저 서서 높이를 맞춘 뒤(u<80) 옆에서 밀어 넣는다(80~140).
        #   straight 접근(cup)은 아래로 향한 손가락이 컵 벽을 위에서 치며 들어가 컵을 넘어뜨린다(09.14 실측 tilt 50~80°).
        # 손 개구(엄지↔4지 사이)는 **손끝 쪽(+x)** 으로 열려 있다(palm 법선 +y·손가락 +x, Track B 기준자세).
        #   컵은 손끝 쪽에서 케이지로 들어와야 하므로 대기 위치는 컵 뒤(−x). (−y 옆에서 밀면 4지가 컵 벽을 민다 — 09.14 실측 15~55 N·전도)
        _ax = {"x": [-args.side_m, 0.0, 0.0], "y": [0.0, -args.side_m, 0.0]}[args.side_axis]
        wait_off = torch.tensor(_ax, device=env.device) if (args.approach == "cup2" and u < 80) else torch.zeros(3, device=env.device)
        tgt_palm = cup_local() + torch.tensor([0.0, 0.0, args.z_off], device=env.device) + wait_off - CAGE_OFF
        d = tgt_palm - rig.anchor_env[:, :3]
        for k in range(3):
            v = d[:, k]
            a[:, k] = torch.where(v >= 0, (v / rig.delta_hi[k]).clamp(max=1.0), (v / rig.delta_lo[k]).clamp(max=1.0) * -1.0)
        if args.dey != 0.0:
            import math as _m
            v = _m.radians(args.dey)
            a[:, 4] = v / float(rig.delta_hi[4]) if v >= 0 else -v / float(rig.delta_lo[4])
    close_at = {"cup": 120, "cup2": 160, "cupf": 160, "trackb": 150, "trackb2": 220}.get(args.approach, 40)
    if args.approach == "cup2" and u == close_at - 1 and args.print_q:
        q = env.robot.data.joint_pos[0]; jn = env.robot.data.joint_names
        print("[pre-close] arm q =", {jn[i]: round(float(q[i]), 4) for i in rig.arm_ids}, "palm6 =", [round(float(v), 4) for v in rig.palm_pose_6d()[0].tolist()],
              "palm-cup =", [round(float(v), 3) for v in (rig.palm_pos() - cup_local())[0].tolist()], flush=True)
    if u >= close_at:
        a[:, 6:half] = 1.0
    if args.approach in ("cup", "cup2", "cupf", "trackb", "trackb2") and u >= close_at + 150:
        a[:, 2] = a[:, 2] + 0.12 / float(rig.delta_hi[2])
    obs, rew, term, trunc, ex = env.step(a)
    ex_prev = ex
    if bool(term.any()):
        print(f"[reset t={t}] terminated envs={int(term.sum())} runaway={float(ex['task/runaway_rate']):.2f} drop={float(ex['done/drop']):.2f} "
              f"tiltS={float(ex['task/src_tilt_deg']):.1f} mimic={float(ex['ctrl/mimic_err_max']):.2f}", flush=True)
    if args.approach == "cup2" and (u in (79, 85, 90, 95, 100, 110)):
        # 손 링크 기하 덤프(컵 원점 기준)
        c = cup_local()[0]; bn = env.robot.data.body_names
        rows = []
        for i, n in enumerate(bn):
            d = (env.robot.data.body_pos_w[0, i] - env.scene.env_origins[0]) - c
            if float(d.norm()) < 0.16:          # 컵 근방 링크만(팔·손·센서 프레임 전부)
                rows.append(f"{n}({float(d[0]):+.3f},{float(d[1]):+.3f},{float(d[2]):+.3f})")
        nets = {f: [round(float(s_.data.net_forces_w.view(N, -1, 3).sum(dim=1).norm(dim=-1)[0]), 1) for s_ in rig.sensors[f]] for f in rig.fingers}
        pn = round(float(rig.palm_sensor.data.net_forces_w.view(N, -1, 3).sum(dim=1).norm(dim=-1)[0]), 1)
        print(f"[geom u={u}] tilt={float(ex['task/src_tilt_deg']):.1f} palm_net={pn} net={nets}\n   " + " ".join(rows), flush=True)
    if args.quiet and t % 30 == 0:
        pc = (rig.palm_pos() - cup_local())[0].tolist()
        print(f"[t={t:3d}] clos={float(rig.closure()[0]):.2f} mimic={float(ex['ctrl/mimic_err_max']):.2f} grasped={float(ex['task/src_grasped']):.2f} "
              f"lift={float(ex['task/src_cup_lift']):+.3f} tilt={float(ex['task/src_tilt_deg']):.1f} palm-cup={[round(v,3) for v in pc]} palm_err={float(ex['fabric/src_palm_err'])*1000:.0f}mm "
              f"rotErr={float(ex['fabric/src_rot_err_deg']):.1f} ff={[round(v,1) for v in rig.finger_forces()[0].tolist()]}", flush=True)
    if t == 5:
        for f in rig.fingers:
            for sname, sensor in zip(rig.profile.finger_sensor_bodies[f], rig.sensors[f]):
                net = sensor.data.net_forces_w.view(N, -1, 3).sum(dim=1).norm(dim=-1)[0].item()
                if net > 0.5: print(f"[net t=5] {sname}: {net:.1f} N")
        ps = rig.palm_sensor.data.net_forces_w.view(N, -1, 3).sum(dim=1).norm(dim=-1)[0].item(); print(f"[net t=5] palm: {ps:.1f} N")
    if (not args.quiet) and t in (0, hold, hold + 60, hold + 120, hold + 180, hold + 240, hold + 270, hold + 300, hold + 340) or t == args.steps - 1:
        q = env.robot.data.joint_pos[0]
        vals = {jn[i].split("_hj_")[1]: round(float(q[i]), 3) for i in hand_all}
        tgt = {rig.profile.hand_joint_names[i].split("_hj_")[1]: round(float(rig.syn_target[0, i]), 3) for i in range(len(rig.syn_ids))}
        ff = rig.finger_forces()[0].tolist(); fo = rig.foreign_force()[0].item()
        pc = (rig.palm_pos() - cup_local())[0].tolist()
        print(f"[t={t:3d}] q={vals}\n        tgt={tgt}\n        closure={float(rig.closure()[0]):.2f} mimic_err={float(ex['ctrl/mimic_err_max']):.3f} "
              f"finger_force={[round(v,2) for v in ff]} foreign={fo:.2f} palm-cup={[round(v,3) for v in pc]} "
              f"palm_err={float(ex['fabric/src_palm_err'])*1000:.1f}mm grasped={float(ex['task/src_grasped']):.2f} lift={float(ex['task/src_cup_lift']):+.3f}", flush=True)
if args.print_q:
    q = env.robot.data.joint_pos[0]; jn = env.robot.data.joint_names
    print("[home] arm q =", {jn[i]: round(float(q[i]), 4) for i in rig.arm_ids})
    print("[home] palm pose6 =", [round(float(v), 4) for v in rig.palm_pose_6d()[0].tolist()])
    tips = rig.tips_pos()[0]; c = cup_local()[0]
    print("[home] tips-cup =", [[round(float(v), 3) for v in (tips[i] - c).tolist()] for i in range(tips.shape[0])])
    print("[home] palm_err mm =", round(float(ex['fabric/src_palm_err']) * 1000, 1), "foreign N =", round(float(rig.foreign_force()[0]), 2))
env.close(); app.close()

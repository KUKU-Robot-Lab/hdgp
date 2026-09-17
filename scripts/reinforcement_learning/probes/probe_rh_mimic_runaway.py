"""RH56F1 종속관절(mimic) 폭주 원인 분리(09.17) — 접촉 없이 팔만 움직여 폭주가 무엇에 달렸는지 잰다.

리셋 게이트에서 컵에 닿기 전 자유 공간 이동 중 hand_dep_qd 가 1,118~11,178 rad/s 로 튀어 에피소드가 계속 리셋됐다.
자산의 mimic nf 500 은 09.15 에 손만 고정한 손가락 스윕으로 검증됐고 팔 가속 상황은 검증된 적이 없다.

변종(한 세션 안에서 차례로):
  step       팔 목표를 한 번에 점프(게이트와 같은 조건)
  ramp       같은 거리를 구간 전체에 선형으로 나눠 이동(속도 제한)
  step_damp  step + 손 구동관절 감쇠 --damping(기본 1.5, 자산 0.3)
  ramp_damp  ramp + 감쇠
--mimic_nf 를 주면 로봇 USD 위에 mimic naturalFrequency 오버라이드 층을 씌운 사본으로 같은 변종을 돈다.

    ../IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/probes/probe_rh_mimic_runaway.py --num_envs 8
    ../IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/probes/probe_rh_mimic_runaway.py --num_envs 8 --mimic_nf 200
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-rh_b_pour_fab_mimic")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--variants", default="step,ramp,step_damp,ramp_damp")
parser.add_argument("--damping", type=float, default=1.5)
parser.add_argument("--mimic_nf", type=float, default=0.0, help=">0 이면 mimic naturalFrequency 를 이 값으로 덮은 USD 사본 사용")
parser.add_argument("--sensor_mass", type=float, default=0.0, help=">0 이면 *_sensor 링크 질량을 이 값[kg]으로(관성은 같은 비율) 덮은 USD 사본 사용")
parser.add_argument("--cycles", type=int, default=5)
parser.add_argument("--leg_steps", type=int, default=60)
parser.add_argument("--scenario", default="free", choices=["free", "contact"], help="free=접촉 없는 팔 이동 · contact=소스 손을 컵 입구/벽으로 쓸기")
parser.add_argument("--contact_variants", default="rim_bead,rim_nobead,wall_bead,rim_bead_damp")
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

if args.mimic_nf > 0.0 or args.sensor_mass > 0.0:
    from pxr import Gf, Sdf, Usd
    orig = cfg.robot_cfg.spawn.usd_path
    tmp = f"/tmp/rh56f1_override_nf{int(args.mimic_nf)}_sm{args.sensor_mass:g}.usda"
    stage = Usd.Stage.CreateNew(tmp)
    stage.GetRootLayer().subLayerPaths.append(orig)
    dp = Sdf.Layer.FindOrOpen(orig).defaultPrim
    if dp:
        stage.SetDefaultPrim(stage.GetPrimAtPath(f"/{dp}"))
    n_nf = n_m = 0
    for prim in stage.Traverse():
        if args.mimic_nf > 0.0:
            for sch in prim.GetAppliedSchemas():
                if sch.startswith("PhysxMimicJointAPI:"):
                    attr = prim.GetAttribute(f"physxMimicJoint:{sch.split(':', 1)[1]}:naturalFrequency")
                    if attr:
                        attr.Set(float(args.mimic_nf))
                        n_nf += 1
        if args.sensor_mass > 0.0 and prim.GetName().endswith("_sensor"):
            m = prim.GetAttribute("physics:mass")
            if m and m.Get() is not None:
                old = float(m.Get())
                m.Set(float(args.sensor_mass))
                inert = prim.GetAttribute("physics:diagonalInertia")
                if inert and inert.Get() is not None and old > 0.0:
                    k = float(args.sensor_mass) / old
                    v = inert.Get()
                    inert.Set(Gf.Vec3f(v[0] * k, v[1] * k, v[2] * k))
                n_m += 1
                if n_m <= 3:
                    print(f"[usd] {prim.GetPath()} mass {old:g} → {args.sensor_mass:g}", flush=True)
    stage.GetRootLayer().Save()
    print(f"[usd] override nf→{args.mimic_nf or '-'}({n_nf}) sensor_mass→{args.sensor_mass or '-'}({n_m}) · {tmp}", flush=True)
    if (args.mimic_nf > 0.0 and n_nf == 0) or (args.sensor_mass > 0.0 and n_m == 0):
        raise SystemExit("오버라이드 대상 prim 을 못 찾았다 — 무효")
    cfg.robot_cfg.spawn.usd_path = tmp

env = gym.make(args.task, cfg=cfg).unwrapped
N, A = env.num_envs, env.cfg.action_space
HALF = A // 2
dev = env.device
hold = int(env.cfg.hold_steps)
hand_ids = sorted(set(env.src.syn_ids) | set(env.rcv.syn_ids))
base_damp = env.robot.data.joint_damping[:, hand_ids].clone()

# 자유 공간 이동: 두 팔을 앵커에서 위·바깥쪽으로 크게 옮겼다 돌아온다(컵·몸통·테이블과 안 닿는 쪽).
#   소스(우) +y 는 몸 안쪽이라 −x·+z 위주, 리시버(좌)는 미러.
LEG = torch.zeros(A, device=dev)
LEG[0:3] = torch.tensor([-0.9, 0.0, 0.9], device=dev)
LEG[HALF:HALF + 3] = torch.tensor([-0.9, 0.0, 0.9], device=dev)
LEG[6:HALF] = -1.0
LEG[HALF + 6:A] = -1.0

print(f"[cfg] envs={N} cycles={args.cycles} leg_steps={args.leg_steps} mimic_nf={args.mimic_nf or 'asset(500)'} "
      f"thr dep_qd={env.cfg.mimic_runaway_dep_qd} err={env.cfg.mimic_runaway_err_rad} hand_damp_asset={float(base_damp.mean()):.2f}", flush=True)

FREE_VARIANTS = [v.strip() for v in args.variants.split(",") if v.strip()] if args.scenario == "free" else []
for var in FREE_VARIANTS:
    damp = torch.full_like(base_damp, args.damping) if var.endswith("_damp") else base_damp
    env.robot.write_joint_damping_to_sim(damp, joint_ids=hand_ids)
    env.reset()
    for _ in range(hold):
        env.step(torch.zeros(N, A, device=dev))
    ramp = var.startswith("ramp")
    terms = mim = arm = 0.0
    qd_max = err_max = 0.0
    qd_trace = []
    for c in range(args.cycles):
        for leg in (1.0, -1.0):
            for u in range(args.leg_steps):
                if ramp:
                    frac = (u + 1) / args.leg_steps
                    s = frac if leg > 0 else 1.0 - frac
                else:
                    s = 1.0 if leg > 0 else 0.0
                a = LEG.clone()
                a[0:3] *= s
                a[HALF:HALF + 3] *= s
                _, _, term, _, ex = env.step(a.unsqueeze(0).repeat(N, 1))
                terms += float(term.float().sum())
                mim += float(ex["done/mimic_runaway"]) * N
                arm += float(ex["task/runaway_rate"]) * N
                qd = float(ex["ctrl/hand_dep_qd_max"])
                qd_max = max(qd_max, qd)
                err_max = max(err_max, float(ex["ctrl/mimic_err_max"]))
                qd_trace.append(qd)
    steps = args.cycles * 2 * args.leg_steps
    moved = float((env.src.palm_pos() - env.src.anchor_env[:, :3]).norm(dim=-1).mean())
    over100 = sum(1 for q in qd_trace if q > 100.0)
    print(f"[{var:9s}] 종료 {terms:.0f}회({terms/N/steps*900:.2f}/에피소드 환산) · mimic 폭주 {mim:.0f} · 팔 폭주 {arm:.0f} · "
          f"(마지막 palm-앵커 {moved*100:.1f}cm) · hand_dep_qd 최대 {qd_max:.2f} rad/s · >100 rad/s 스텝 {over100}/{steps} · mimic 오차 최대 {err_max:.2f} rad", flush=True)

# ---- contact ------------------------------------------------------------------------------
if args.scenario == "contact":
    rig, cup_asset = env.src, env.source_cup
    bn = env.robot.data.body_names
    i_th = bn.index(rig.profile.fingertip_bodies[0]); i_ix = bn.index(rig.profile.fingertip_bodies[1])
    i_ix1 = bn.index(rig.profile.finger_sensor_bodies["index"][0])
    mouth, mid = float(env.cfg.cup_mouth_z), 0.5 * float(env.cfg.cup_mouth_z)

    def hand_frame():
        pos = env.robot.data.body_pos_w - env.scene.env_origins[:, None, :]
        pocket = 0.5 * (pos[:, i_th] + pos[:, i_ix])
        f = pos[:, i_ix, :2] - pos[:, i_ix1, :2]
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-9)
        return pocket, torch.stack([-f[:, 1], f[:, 0]], dim=1)

    def to_action(d):
        a = torch.zeros(N, 3, device=dev)
        for k in range(3):
            v = d[:, k]
            a[:, k] = torch.where(v >= 0, (v / rig.delta_hi[k]).clamp(max=1.0), -(v / rig.delta_lo[k]).clamp(max=1.0))
        return a

    for var in [v.strip() for v in args.contact_variants.split(",") if v.strip()]:
        damp = torch.full_like(base_damp, args.damping) if var.endswith("_damp") else base_damp
        env.robot.write_joint_damping_to_sim(damp, joint_ids=hand_ids)
        env.reset()
        for _ in range(hold):
            env.step(torch.zeros(N, A, device=dev))
        if "nobead" in var:
            st = env.beads.data.object_state_w.clone()
            st[:, :, 0] += 1.5
            st[:, :, 7:] = 0.0
            env.beads.write_object_state_to_sim(st)
        z_off = mouth if var.startswith("rim") else mid
        terms = mim = 0.0
        qd_max = err_max = fmax = 0.0
        over = 0
        steps = 0
        for c in range(args.cycles):
            for leg in (1.0, -1.0):
                for u in range(args.leg_steps):
                    frac = (u + 1) / args.leg_steps
                    s_ = 0.12 - 0.16 * frac if leg > 0 else -0.04 + 0.16 * frac
                    cp = env._local(cup_asset.data.root_pos_w)
                    pocket, n = hand_frame()
                    goal = torch.cat([cp[:, :2] + s_ * n, (cp[:, 2] + z_off).unsqueeze(1)], dim=1)
                    a = torch.zeros(N, A, device=dev)
                    a[:, 0:3] = to_action(rig.palm_pos() + (goal - pocket) - rig.anchor_env[:, :3])
                    a[:, 6:HALF] = -1.0
                    _, _, term, _, ex = env.step(a)
                    steps += 1
                    terms += float(term.float().sum())
                    mim += float(ex["done/mimic_runaway"]) * N
                    q = float(ex["ctrl/hand_dep_qd_max"])
                    qd_max = max(qd_max, q); over += int(q > 100.0)
                    err_max = max(err_max, float(ex["ctrl/mimic_err_max"]))
                    fmax = max(fmax, float(rig.finger_forces().max()))
        print(f"[contact {var:13s}] 종료 {terms:.0f}회({terms/N/steps*900:.2f}/에피소드 환산) · mimic 폭주 {mim:.0f} · "
              f"hand_dep_qd 최대 {qd_max:.0f} rad/s · >100 rad/s 스텝 {over}/{steps} · mimic 오차 최대 {err_max:.2f} rad · "
              f"손가락 접촉력 최대 {fmax:.0f} N", flush=True)

env.close()
app.close()

"""RH56F1 손가락 1개 스윕 → 다른 손 관절이 흔들리는지 (09.15 사용자 보고: 검지 조인트를 움직이면 옆 링크가 흔들림).

로봇을 단독으로 띄우고(고정 베이스·중력 ON) 오른손 구동관절 하나(기본 r_hj_index_1)의 위치 목표만
0 → HI → 0 으로 벤더 최대 속도(속도값 2000 = 무부하 전범위 1000 ms ≈ 1.5 rad/s)에 맞춰 움직인다.
나머지 관절은 초기 목표를 유지한다. 기록: 다른 손 관절의 최대 이탈·속도, 스윕 관절의 종속(mimic) 추종 오차,
손바닥 링크 위치 이탈, 스윕 관절 토크 포화.

손 드라이브 게인은 --hand_kp/--hand_kd 로 바꾼다(미지정 = USD 에 적힌 값). 종속관절은 드라이브 없이 둔다.

    PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/probes/probe_rh56f1_finger_sweep.py \
        --headless [--usd <path>] [--self_collision on|off] [--hand_kp 5 --hand_kd 2] [--joint r_hj_index_1]
결과 한 줄: `[SWEEP] {json}`
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

HDGP = Path(__file__).resolve().parents[3]
DEFAULT_USD = HDGP / "assets" / "robot" / "openarm_rh56f1_bi_rl" / "openarm_rh56f1_bi_rl.usd"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd", default=str(DEFAULT_USD))
parser.add_argument("--self_collision", choices=["on", "off"], default="on")
parser.add_argument("--hand_kp", type=float, default=None, help="손 구동관절 강성 [N·m/rad] (미지정=USD 값)")
parser.add_argument("--hand_kd", type=float, default=None, help="손 구동관절 감쇠 [N·m·s/rad] (미지정=USD 값)")
parser.add_argument("--joint", default="r_hj_index_1")
parser.add_argument("--hi", type=float, default=1.2, help="스윕 최대 각 [rad]")
parser.add_argument("--rate", type=float, default=1.5, help="스윕 속도 [rad/s] (벤더 최대 ≈ 1.5)")
parser.add_argument("--mimic_nf", type=float, default=None, help="PhysX mimic naturalFrequency 덮어쓰기(미지정=USD 값)")
parser.add_argument("--mimic_dr", type=float, default=None, help="PhysX mimic dampingRatio 덮어쓰기(미지정=USD 값)")
parser.add_argument("--contacts", action="store_true", help="오른손 전 링크 접촉 센서로 링크별 최대 접촉력 기록")
parser.add_argument("--contact_partners", default="",
                    help="쉼표 구분 상대 링크명. 상대마다 필터 1개짜리 센서(순서 모호성 없음)로 오른손 링크↔상대 최대 접촉력 기록")
parser.add_argument("--label", default="")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402

DT = 1.0 / 120.0
HAND_DRIVEN = r"[rl]_hj_(thumb_[12]|index_1|middle_1|ring_1|pinky_1)"
HAND_MIMIC = r"[rl]_hj_(thumb_[34]|index_2|middle_2|ring_2|pinky_2)"
MIMIC_OF = {"index_2": ("index_1", 1.1169), "middle_2": ("middle_1", 1.1169), "ring_2": ("ring_1", 1.1169),
            "pinky_2": ("pinky_1", 1.1169), "thumb_3": ("thumb_2", 1.1425), "thumb_4": ("thumb_3", 0.7508)}


def build_robot() -> Articulation:
    hand_gain = {} if args.hand_kp is None else {"stiffness": args.hand_kp, "damping": args.hand_kd or 0.0}
    actuators = {
        "rest": ImplicitActuatorCfg(joint_names_expr=[r"^(?![rl]_hj_).*"], stiffness=None, damping=None),
        "hand_driven": ImplicitActuatorCfg(joint_names_expr=[HAND_DRIVEN],
                                           stiffness=hand_gain.get("stiffness"), damping=hand_gain.get("damping")),
        "hand_mimic": ImplicitActuatorCfg(joint_names_expr=[HAND_MIMIC], stiffness=0.0, damping=0.0),
    }
    return Articulation(ArticulationCfg(
        prim_path="/World/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=args.usd,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=args.self_collision == "on",
                solver_position_iteration_count=16, solver_velocity_iteration_count=1),
            activate_contact_sensors=args.contacts or bool(args.contact_partners),
        ),
        actuators=actuators,
    ))


def override_mimic(root: str = "/World/robot") -> dict:
    """spawn 직후·sim.reset 전에 PhysX mimic 스프링 속성을 덮어쓴다. 바꾼 개수와 실제 값(첫 관절)을 돌려준다."""
    import omni.usd
    from pxr import Usd
    stage = omni.usd.get_context().get_stage()
    changed, seen = 0, {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
        for at in prim.GetAttributes():
            nm = at.GetName()
            if not nm.startswith("physxMimicJoint:"):
                continue
            if nm.endswith(":naturalFrequency"):
                if args.mimic_nf is not None and not prim.IsInstanceProxy():
                    at.Set(float(args.mimic_nf)); changed += 1
                seen.setdefault("naturalFrequency", at.Get())
            elif nm.endswith(":dampingRatio"):
                if args.mimic_dr is not None and not prim.IsInstanceProxy():
                    at.Set(float(args.mimic_dr)); changed += 1
                seen.setdefault("dampingRatio", at.Get())
    return {"changed": changed, **seen}


def schedule(n_hold: int, n_ramp: int, hi: float) -> list[float]:
    up = [hi * (i + 1) / n_ramp for i in range(n_ramp)]
    return [0.0] * n_hold + up + [hi] * n_hold + up[::-1] + [0.0] * n_hold


def main() -> dict:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT))
    robot = build_robot()
    mimic = override_mimic()
    sensor = ContactSensor(ContactSensorCfg(prim_path="/World/robot/r_hl_.*", update_period=0.0, history_length=1)) \
        if args.contacts else None
    partners = [p for p in args.contact_partners.split(",") if p]
    partner_sensors = {p: ContactSensor(ContactSensorCfg(prim_path="/World/robot/r_hl_.*", update_period=0.0, history_length=1,
                                                         filter_prim_paths_expr=[f"/World/robot/{p}"])) for p in partners}
    sim.reset()
    robot.update(DT)
    names = robot.joint_names
    ji = names.index(args.joint)
    hand = [i for i, n in enumerate(names) if n.startswith("r_hj_")]
    suffix = args.joint[len("r_hj_"):]
    chain, lead = [], suffix                               # 종속 연쇄 전체(thumb_2 → thumb_3 → thumb_4)는 흔들림이 아니라 추종
    while (nxt := next((f for f, (ld, _) in MIMIC_OF.items() if ld == lead), None)) is not None:
        chain.append(names.index(f"r_hj_{nxt}")); lead = nxt
    follower = chain[0] if chain else None
    mult = next((m for f, (lead, m) in MIMIC_OF.items() if lead == suffix), None)
    others = [i for i in hand if i != ji and i not in chain]
    palm = robot.body_names.index("r_hl_palm_2")

    target0 = robot.data.joint_pos.clone()
    for _ in range(120):                                   # 정착(중력 ON, 초기 목표 유지)
        robot.set_joint_position_target(target0); robot.write_data_to_sim(); sim.step(); robot.update(DT)
    q0 = robot.data.joint_pos.clone()
    p0 = robot.data.body_pos_w[:, palm].clone()
    n_ramp = max(1, int(round(args.hi / args.rate / DT)))
    dev = torch.zeros(len(names), device=q0.device); vel = torch.zeros_like(dev)
    vel_sq = torch.zeros_like(dev); n_steps = 0
    track_err = 0.0; mimic_err = 0.0; palm_dev = 0.0; sat = 0.0
    contact_max = None
    pair_max: dict[str, torch.Tensor] = {}
    effort = robot.data.joint_effort_limits[0, ji].item() if hasattr(robot.data, "joint_effort_limits") else float("nan")
    for tgt in schedule(60, n_ramp, args.hi):
        target = q0.clone(); target[:, ji] = tgt
        robot.set_joint_position_target(target); robot.write_data_to_sim(); sim.step(); robot.update(DT)
        q, qd = robot.data.joint_pos[0], robot.data.joint_vel[0]
        dev = torch.maximum(dev, (q - q0[0]).abs()); vel = torch.maximum(vel, qd.abs())
        vel_sq += qd ** 2; n_steps += 1
        if sensor is not None:
            sensor.update(DT)
            f = sensor.data.net_forces_w[0].norm(dim=-1)
            contact_max = f if contact_max is None else torch.maximum(contact_max, f)
        for p, s in partner_sensors.items():
            s.update(DT)
            fp = s.data.force_matrix_w[0, :, 0].norm(dim=-1)
            pair_max[p] = fp if p not in pair_max else torch.maximum(pair_max[p], fp)
        track_err = max(track_err, abs(q[ji].item() - tgt))
        if follower is not None:
            mimic_err = max(mimic_err, abs(q[follower].item() - mult * q[ji].item()))
        palm_dev = max(palm_dev, (robot.data.body_pos_w[0, palm] - p0[0]).norm().item())
        tau = robot.data.applied_torque[0, ji].item()
        if effort == effort and effort > 0:
            sat = max(sat, abs(tau) / effort)
    worst = sorted(((dev[i].item(), names[i]) for i in others), reverse=True)[:5]
    return {
        "label": args.label, "usd": Path(args.usd).name, "self_collision": args.self_collision,
        "hand_kp": args.hand_kp, "hand_kd": args.hand_kd, "joint": args.joint, "hi": args.hi, "rate": args.rate,
        "stiffness_used": robot.data.joint_stiffness[0, ji].item(), "damping_used": robot.data.joint_damping[0, ji].item(),
        "effort_limit": effort, "torque_saturation_max": round(sat, 3),
        "sweep_track_err_max_rad": round(track_err, 4), "mimic_follower_err_max_rad": round(mimic_err, 4),
        "others_dev_max_rad": round(max(d for d, _ in worst), 5), "others_vel_max_rad_s": round(max(vel[i].item() for i in others), 4),
        "others_worst5": [(n, round(d, 5)) for d, n in worst], "palm_pos_dev_mm": round(1000 * palm_dev, 3),
        "others_vel_rms_rad_s": round(float((vel_sq[others] / max(n_steps, 1)).sqrt().mean().item()), 5),
        "mimic": mimic,
        "contact_top_N": None if sensor is None else sorted(((round(contact_max[i].item(), 3), n) for i, n in enumerate(sensor.body_names)
                                                            if contact_max[i].item() > 1e-3), reverse=True)[:8],
        "contact_pairs_top": sorted(((round(fp[i].item(), 3), n, p) for p, fp in pair_max.items()
                                     for i, n in enumerate(partner_sensors[p].body_names)
                                     if n != p and fp[i].item() > 1e-3), reverse=True)[:12],
    }


if __name__ == "__main__":
    out = main()
    print("[SWEEP]", json.dumps(out), flush=True)
    sys.stdout.flush()
    os._exit(0)                                            # Isaac app.close() 멈춤 회피(메모리: isaac-probe-nohup-unbuffered)

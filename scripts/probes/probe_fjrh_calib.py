#!/usr/bin/env python3
"""RH56F1 + Track B(grasp_fj_rh) 캘리브 프로브 — 프로필 상수를 **실측으로** 정한다.

왜 태스크 env 가 아니라 articulation 만 띄우나
---------------------------------------------
태스크 env 는 부팅에서 홈 palm 이 워크스페이스 박스 안인지, 케이지가 컵을 관통하지
않는지를 **이미 검사한다**. 그 상수를 정하려고 도는 프로브가 그 검사에 걸리면
아무것도 못 잰다. 그래서 여기서는 `_build_robot_cfg` 로 로봇만 스폰한다.

무엇을 재는가 (프로필의 어느 필드로 가는지 함께 적는다)
-----------------------------------------------------
1. **mimic 건전성** — 리더 목표를 여러 점으로 스윕해 `q_dep / q_lead` 가 URDF 배율
   (1.1425 · 0.7508 · 1.1169)에 붙는지. ★한 지점 판정 금지(09.02: 0.1 rad 에서는
   임포터 기본값도 맞아 보였고 0.4 에서 부호가 뒤집혔다).
2. **홈 palm 포즈** — `palm_rot_center_deg` · `palm_box_*` 가 홈을 담는지.
   `_palm_pose_6d` 와 같은 규약(env-local xyz + euler_zyx)으로 잰다.
3. **케이지 중심·반경** — `object_spawn_center` (케이지 x 를 컵에 정렬).
4. **손 최하단 z** — `rw_hand_floor_z` · `hand_floor_terminate_depth` 여유.
5. **파지 창** — 엄지↔4지 간극(open/grip)과 물체 뱅크 지름의 대조.
6. **접촉 스트레스** — 고정 원통을 케이지 중심에 두고 손을 완전 폐쇄 → 종속관절이
   한계를 넘거나 속도가 폭주하는지(09.02 씬 발산 재발 확인).

사용
----
    cd hdgp && ~/rl_ws/IsaacLab/isaaclab.sh -p scripts/probes/probe_fjrh_calib.py

출력은 stdout + `our_source/rh56f1_fj_calib/probe_fjrh_calib.json`.
★값을 프로필에 **자동 반영하지 않는다** — 홈은 체크포인트에 딸린 값이라 조용히 바뀌면
  재생이 깨진다. 사람이 보고 옮겨 적는다.
"""

from __future__ import annotations

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--profile", default="rh56f1_right")
parser.add_argument("--settle", type=int, default=120, help="자세 정착 스텝")
parser.add_argument("--press-steps", type=int, default=400, help="접촉 스트레스 스텝")
parser.add_argument("--press-radius", type=float, default=0.028, help="스트레스 원통 반지름 [m]")
parser.add_argument("--out", default="our_source/rh56f1_fj_calib/probe_fjrh_calib.json")
parser.add_argument("--depen", type=float, default=1000.0,
                    help="max_depenetration_velocity — 09.02 RH56F1 은 1 을 썼다")
parser.add_argument("--dep-damping", type=float, default=0.0,
                    help="종속관절 점성감쇠. kp=0 이라 위치 제약과 싸우지 않고 속도만 잡는다")
parser.add_argument("--press-mass", type=float, default=0.0,
                    help="0 이면 고정(kinematic) 원통, >0 이면 그 질량의 자유 강체")
parser.add_argument("--only-press", action="store_true", help="스트레스 구간만 돈다")
parser.add_argument("--skip-press", action="store_true",
                    help="접촉 스트레스를 건너뛴다. ★홈을 풀 때는 반드시 켠다 — 스트레스용 물체가 "
                         "씬에 남아 손을 눌러 케이지 반경이 작게 측정된다(09.07 실측 53 → 29mm)")
parser.add_argument("--skip-reach", action="store_true", help="목표 박스 도달성 검사를 건너뛴다")
parser.add_argument("--press-mode", choices=("cyl", "table"), default="cyl",
                    help="cyl=케이지 중심 원통 · table=팔로 손을 테이블에 밀어넣는다(학습 실패 재현)")
parser.add_argument("--press-depth", type=float, default=0.08, help="table 모드에서 상판 아래로 지령할 깊이 [m]")
parser.add_argument("--solve-home", default="",
                    help="케이지를 '컵 + dx,dy,dz'(m)에 두는 팔 관절값을 IK 로 푼다. "
                         "예: 0,-0.060,0.055 — 홈 접근거리를 케이지 반경 기준으로 다시 잡을 때")
parser.add_argument("--dep-margin", type=float, default=0.0,
                    help="종속관절 한계를 이만큼 넓힌다(env cfg `mimic_dep_limit_margin_rad` 와 같은 처방)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
_app = AppLauncher(args).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat, matrix_from_quat  # noqa: E402

from openarm.agnostic.tasks.grasp_s2r.grasp_s2r_env_cfg import _build_robot_cfg  # noqa: E402
from openarm.agnostic.tasks.grasp_s2r.robot_profiles import PROFILES  # noqa: E402

# ★테이블·물체 상수는 **이 트랙의 cfg** 에서 읽는다. 부모 기본 cfg 를 쓰면 물체 뱅크가 달라
#   컵 높이가 31 mm 어긋나고(기본 cup_family 최대 원점 0.1005 vs shaker_small 0.0691), 그 위에서
#   푼 홈이 의도한 접근거리를 못 낸다(09.07 실측: 목표 비율 1.60 → 실제 1.96).
#   같은 계열의 실수를 중력보상에서 이미 한 번 했다 — 계측기는 대상 시스템과 **같은 상수·같은
#   제어**를 써야 한다.
from openarm.agnostic.tasks.grasp_fj_rh.grasp_fj_rh_env_cfg import (  # noqa: E402
    GraspFJRH56F1RightEnvCfg as _TrackCfg,
)

_DEG = 180.0 / math.pi
#: URDF `<mimic>` 표 — 자산이 PhysX 제약으로 들고 있는 값과 대조한다.
_MIMIC = {
    "r_hj_thumb_3": ("r_hj_thumb_2", 1.1425),
    "r_hj_thumb_4": ("r_hj_thumb_3", 0.7508),
    **{f"r_hj_{f}_2": (f"r_hj_{f}_1", 1.1169) for f in ("index", "middle", "ring", "pinky")},
}
#: 리더 스윕 지점 — 한 점만 보면 오판한다(09.02).
_SWEEP = (0.1, 0.4, 0.8, 1.2, 1.5)


def _fmt(v, n=4):
    if torch.is_tensor(v):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return [round(float(x), n) for x in v]
    return round(float(v), n)


def _settle(sim, robot, steps: int, dt: float, grav_ids=None) -> None:
    """★중력보상을 env 와 **같게** 걸어야 한다.

    태스크 env 는 `gravity_compensation=1.0` 으로 팔 14관절에 보상 토크를 넣는다.
    이걸 빼고 재면 팔이 처진 자세를 홈으로 착각한다 — 09.07 첫 실행에서 palm z 가
    0.4169(env 보고) 대신 0.3446 으로 나왔고, 케이지·스폰 정렬 결론이 통째로 틀어졌다.
    """
    for _ in range(steps):
        if grav_ids is not None:
            tau = robot.root_physx_view.get_gravity_compensation_forces()
            robot.set_joint_effort_target(tau[:, grav_ids], joint_ids=grav_ids.tolist())
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt)


def _ik(sim, robot, dt, grav, palm_idx, cage_off_palm, q, goal, arm_ids, iters: int = 300):
    """케이지 중심을 `goal` 에 두는 팔 관절값(DLS). 텔레포트로 푼다 — 접촉 없이 기하만 본다."""
    arm_t = torch.tensor(arm_ids, device=robot.device, dtype=torch.long)
    lo = robot.data.soft_joint_pos_limits[0, arm_t, 0]
    hi = robot.data.soft_joint_pos_limits[0, arm_t, 1]
    for _ in range(iters):
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt)
        R = matrix_from_quat(robot.data.body_quat_w[:, palm_idx])[0]
        r = R @ cage_off_palm
        err = goal - (robot.data.body_pos_w[0, palm_idx] + r)
        if float(err.norm()) < 2e-3:
            break
        jac = robot.root_physx_view.get_jacobians()[0]
        _bi = palm_idx - 1 if jac.shape[0] == len(robot.data.body_names) - 1 else palm_idx
        J = jac[_bi][:, arm_t]
        Jc = J[:3] - torch.stack([torch.linalg.cross(r, J[3:][:, k])
                                  for k in range(J.shape[1])], dim=1)
        dq = Jc.T @ torch.linalg.solve(Jc @ Jc.T + 1e-4 * torch.eye(3, device=Jc.device), err)
        q[:, arm_t] = (q[:, arm_t] + 0.5 * dq).clamp(lo, hi)
    return q


def _press_loop(sim, robot, dt, grav, steps, dep_ids, syn_ids, arm_ids, lo, hi) -> dict:
    """지령을 건 채 물리를 굴리며 **최악값**을 모은다(발산은 순간값으로만 보인다)."""
    worst = {"dep_qd_max": 0.0, "dep_violation_rad": 0.0, "drive_below_lo_rad": 0.0,
             "arm_qd_max": 0.0, "nan": False}
    for _ in range(int(steps)):
        _tau = robot.root_physx_view.get_gravity_compensation_forces()
        robot.set_joint_effort_target(_tau[:, grav], joint_ids=grav.tolist())
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt)
        q, qd = robot.data.joint_pos[0], robot.data.joint_vel[0]
        if bool(torch.isnan(q).any()):
            worst["nan"] = True
            break
        worst["dep_qd_max"] = max(worst["dep_qd_max"], float(qd[dep_ids].abs().max()))
        worst["arm_qd_max"] = max(worst["arm_qd_max"], float(qd[arm_ids].abs().max()))
        worst["dep_violation_rad"] = max(worst["dep_violation_rad"], float(
            torch.maximum(q[dep_ids] - hi[dep_ids], lo[dep_ids] - q[dep_ids]).max().clamp(min=0)))
        worst["drive_below_lo_rad"] = max(worst["drive_below_lo_rad"], float(
            torch.relu(lo[syn_ids] - q[syn_ids]).max()))
    return {k: (v if isinstance(v, bool) else round(v, 4)) for k, v in worst.items()}


def _goal_box_reach(sim, robot, profile, dt, grav, palm_idx, cage_off_palm, q0) -> dict:
    """목표 박스 8꼭짓점 + 중심에 **케이지 중심**을 놓을 수 있는가(팔 관절 IK).

    감쇠 최소자승(DLS)으로 팔 7관절만 푼다. 손은 홈 자세 고정 — 케이지는 palm 에 강체로
    붙어 있으므로 손이 목표 도달성을 바꾸지 않는다.
    ★케이지 야코비안 = J_v − r × J_w (r = 월드에서 palm→케이지 벡터). 위치 야코비안만
      쓰면 55mm 지렛대를 무시해 도달성을 낙관한다.
    """
    from openarm.agnostic.tasks.grasp_fj_rh.grasp_fj_rh_env_cfg import GraspFJRH56F1RightEnvCfg

    cfg = GraspFJRH56F1RightEnvCfg()
    cfg.finalize_after_overrides()
    lo3 = [float(v) for v in cfg.goal_box_min]
    hi3 = [float(v) for v in cfg.goal_box_max]
    targets = {f"corner_{i}": [lo3[k] if (i >> k) & 1 == 0 else hi3[k] for k in range(3)]
               for i in range(8)}
    targets["center"] = [0.5 * (lo3[k] + hi3[k]) for k in range(3)]

    arm_ids, _ = robot.find_joints(profile.arm_joint_regex, preserve_order=True)
    arm_t = torch.tensor(arm_ids, device=robot.device, dtype=torch.long)
    lo = robot.data.soft_joint_pos_limits[0, arm_t, 0]
    hi = robot.data.soft_joint_pos_limits[0, arm_t, 1]
    rows = {}
    for name, goal in targets.items():
        q = q0.clone()
        g = torch.tensor(goal, device=robot.device)
        for _ in range(300):
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            robot.set_joint_position_target(q)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(dt)
            R = matrix_from_quat(robot.data.body_quat_w[:, palm_idx])[0]
            r = R @ cage_off_palm                                   # palm → 케이지 (월드)
            cage = robot.data.body_pos_w[0, palm_idx] + r
            err = g - cage
            if float(err.norm()) < 2e-3:
                break
            jac = robot.root_physx_view.get_jacobians()[0]          # (bodies, 6, dof)
            _bi = palm_idx - 1 if jac.shape[0] == len(robot.data.body_names) - 1 else palm_idx
            J = jac[_bi][:, arm_t]                                  # (6, 7)
            Jv, Jw = J[:3], J[3:]
            Jc = Jv - torch.linalg.cross(r.expand(3, 3).T.contiguous().T * 0 + r, Jw.T).T \
                if False else Jv - torch.stack([torch.linalg.cross(r, Jw[:, k])
                                                for k in range(Jw.shape[1])], dim=1)
            dq = Jc.T @ torch.linalg.solve(Jc @ Jc.T + 1e-4 * torch.eye(3, device=Jc.device), err)
            q[:, arm_t] = (q[:, arm_t] + 0.5 * dq).clamp(lo, hi)
        rows[name] = {
            "goal": _fmt(goal), "cage": _fmt(cage), "err_mm": round(float(err.norm()) * 1000, 1),
            "at_limit": [n for k, n in enumerate([f"j{i+1}" for i in range(len(arm_ids))])
                         if abs(float(q[0, arm_t[k]] - lo[k])) < 1e-4
                         or abs(float(q[0, arm_t[k]] - hi[k])) < 1e-4],
        }
        print(f"[probe] 도달 {name}: 목표 {_fmt(goal)} 오차 {rows[name]['err_mm']}mm "
              f"한계관절 {rows[name]['at_limit']}", flush=True)
    worst = max(v["err_mm"] for v in rows.values())
    rows["worst_err_mm"] = worst
    print(f"[probe] 목표 박스 최악 도달오차 {worst}mm "
          f"({'OK' if worst < 20 else '⚠ 목표열이 조용히 멈출 수 있다'})", flush=True)
    return rows


def main() -> None:
    profile = PROFILES[args.profile]
    cfg = _TrackCfg()                            # 상수 출처(테이블·물체 뱅크 정착고)
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cuda:0"))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg(),
                                    translation=(0.0, 0.0, -0.05))
    sim_utils.DomeLightCfg(intensity=800.0).func("/World/Light",
                                                 sim_utils.DomeLightCfg(intensity=800.0))

    robot_cfg = _build_robot_cfg(profile, enable_self_collisions=False, enable_gravity=True)
    robot_cfg.prim_path = "/World/Robot"
    robot_cfg.spawn.rigid_props.max_depenetration_velocity = float(args.depen)
    for _n in ("right_hand_mimic", "left_hand_mimic"):
        if _n in robot_cfg.actuators:
            robot_cfg.actuators[_n].damping = float(args.dep_damping)
    robot = Articulation(robot_cfg)
    sim.reset()
    dt = sim.get_physics_dt()
    out: dict = {"profile": profile.name, "usd": profile.usd_relpath}

    jn = robot.data.joint_names
    arm_ids, _ = robot.find_joints(profile.arm_joint_regex, preserve_order=False)
    hand_ids, hand_names = robot.find_joints(profile.hand_joint_regex, preserve_order=False)
    syn_ids = [jn.index(n) for n in profile.hand_joint_names]
    dep_ids = [jn.index(n) for n in _MIMIC]
    out["dof"] = {"total": len(jn), "arm": len(arm_ids), "hand_driven": len(hand_ids),
                  "hand_dependent": len(dep_ids), "hand_names": sorted(hand_names)}
    print(f"[probe] DOF 총 {len(jn)} · 팔 {len(arm_ids)} · 손 구동 {len(hand_ids)} · 종속 {len(dep_ids)}",
          flush=True)

    grav, _ = robot.find_joints("[rl]_aj_[1-7]")          # env 와 같은 중력보상 대상
    grav = torch.tensor(grav, device=robot.device, dtype=torch.long)
    if float(args.dep_margin) > 0.0:
        _lim = robot.data.joint_pos_limits[:, torch.tensor(dep_ids, device=robot.device), :].clone()
        _lim[..., 0] -= float(args.dep_margin)
        _lim[..., 1] += float(args.dep_margin)
        robot.write_joint_limits_to_sim(_lim, joint_ids=dep_ids)
        print(f"[probe] 종속 한계 ±{args.dep_margin} rad 확장 적용", flush=True)
        lo = robot.data.soft_joint_pos_limits[0, :, 0]
        hi = robot.data.soft_joint_pos_limits[0, :, 1]

    q0 = robot.data.default_joint_pos.clone()
    lo = robot.data.soft_joint_pos_limits[0, :, 0]
    hi = robot.data.soft_joint_pos_limits[0, :, 1]

    def _home() -> None:
        robot.write_joint_state_to_sim(q0, torch.zeros_like(q0))
        robot.set_joint_position_target(q0)
        _settle(sim, robot, args.settle, dt, grav)

    out["knobs"] = {"depen": float(args.depen), "dep_damping": float(args.dep_damping),
                    "press_mass": float(args.press_mass), "press_radius": float(args.press_radius)}

    # ---------------------------------------------------------------- ① mimic 스윕
    _home()
    sweep = []
    for lead_val in ([] if args.only_press else _SWEEP):
        tgt = q0.clone()
        for dep, (lead, _mult) in _MIMIC.items():
            if lead in jn:                       # thumb_4 의 리더는 종속(thumb_3) — 지령 대상 아님
                li = jn.index(lead)
                if li in syn_ids:
                    tgt[:, li] = min(lead_val, float(hi[li]))
        robot.set_joint_position_target(tgt)
        _settle(sim, robot, args.settle, dt, grav)
        q = robot.data.joint_pos[0]
        qd = robot.data.joint_vel[0]
        row = {"lead_cmd": lead_val, "pairs": {}}
        for dep, (lead, mult) in _MIMIC.items():
            di, li = jn.index(dep), jn.index(lead)
            ql, qdep = float(q[li]), float(q[di])
            row["pairs"][dep] = {
                "q_lead": round(ql, 4), "q_dep": round(qdep, 4),
                "ratio": round(qdep / ql, 4) if abs(ql) > 1e-3 else None,
                "expect": mult,
                "err_rad": round(qdep - mult * ql, 4),
            }
        row["dep_qd_max"] = round(float(qd[dep_ids].abs().max()), 3)
        row["dep_limit_violation_rad"] = round(
            float(torch.relu(q[dep_ids] - hi[dep_ids]).max().clamp(min=0)
                  + torch.relu(lo[dep_ids] - q[dep_ids]).max().clamp(min=0)), 5)
        sweep.append(row)
        print(f"[probe] mimic lead={lead_val}: "
              + " · ".join(f"{k.split('_hj_')[1]} {v['ratio']}" for k, v in row["pairs"].items())
              + f" | dep|qd|max {row['dep_qd_max']}", flush=True)
    out["mimic_sweep"] = sweep

    # ---------------------------------------------------------------- ② 홈 palm / 케이지 / 바닥
    _home()
    palm_idx = robot.find_bodies(profile.palm_body)[0][0]
    tip_ids = [robot.find_bodies(n)[0][0] for n in profile.fingertip_bodies]
    hand_body_ids = [i for i, n in enumerate(robot.data.body_names) if n.startswith("r_hl_")]

    def _pose6() -> list[float]:
        pos = robot.data.body_pos_w[0, palm_idx]
        r, p_, y = euler_xyz_from_quat(robot.data.body_quat_w[:, palm_idx])
        # `_palm_pose_6d` 규약: [x, y, z, ez, ey, ex]. euler 는 (−π, π] 로 감아 비교 가능하게.
        def _w(a):
            a = float(a)
            return (a + math.pi) % (2 * math.pi) - math.pi
        return [float(pos[0]), float(pos[1]), float(pos[2]), _w(y[0]), _w(p_[0]), _w(r[0])]

    def _cage() -> tuple[list[float], float]:
        tips = robot.data.body_pos_w[0, tip_ids]          # 0=thumb, 1..4 = 4지
        ctr = 0.5 * (tips[0] + tips[1:].mean(dim=0))
        rad = 0.5 * float((tips[0] - tips[1:].mean(dim=0)).norm())
        return ctr.tolist(), rad

    home6 = _pose6()
    cage_c, cage_r = _cage()
    palm_w = robot.data.body_pos_w[0, palm_idx]
    hand_z_min_open = float(robot.data.body_pos_w[0, hand_body_ids, 2].min())
    R = matrix_from_quat(robot.data.body_quat_w[:, palm_idx])[0]
    cage_off_palm = (R.transpose(0, 1) @ (torch.tensor(cage_c, device=palm_w.device) - palm_w))

    cup_xy = profile.object_spawn_center
    cup_z = (float(cfg.table_surface_z) + float(cfg.object_origin_offset_z)
             + float(cfg.object_grasp_z_offset))
    out["home"] = {
        "arm_q": {n: round(float(q0[0, jn.index(n)]), 4) for n in jn if n.startswith("r_aj_")},
        "palm_pose6": _fmt(home6),
        "palm_euler_deg": _fmt([home6[3] * _DEG, home6[4] * _DEG, home6[5] * _DEG], 1),
        "cage_center": _fmt(cage_c),
        "cage_radius_mm": round(cage_r * 1000, 1),
        "cage_offset_palm": _fmt(cage_off_palm),
        "hand_z_min_open": round(hand_z_min_open, 4),
        "palm_minus_hand_z": round(float(palm_w[2]) - hand_z_min_open, 4),
        "cup_grasp_center": _fmt([cup_xy[0], cup_xy[1], cup_z]),
        "cage_minus_cup_mm": _fmt([(cage_c[i] - [cup_xy[0], cup_xy[1], cup_z][i]) * 1000
                                   for i in range(3)], 1),
        "cage_cup_gap_xy_mm": round(float(
            (torch.tensor(cage_c[:2]) - torch.tensor(list(cup_xy))).norm()) * 1000, 1),
        "palm_box_min": _fmt(profile.palm_box_min),
        "palm_box_max": _fmt(profile.palm_box_max),
        "palm_rot_center_deg": _fmt(profile.palm_rot_center_deg, 1),
    }
    # ★손의 **벌림 축**(엄지 ↔ 4지 중심)과 **접근 방향**(홈 케이지 → 물체)의 각도.
    #   이 둘이 어긋나면 손가락이 물체를 정면으로 감싸지 못하고 옆으로 쓸어 넘긴다 —
    #   대본 파지에서 리셋의 90%가 `tipped` 였던 것의 후보 원인이다.
    _tips = robot.data.body_pos_w[0, tip_ids]
    _axis = (_tips[0] - _tips[1:].mean(dim=0))                      # 엄지 → 4지 반대방향
    _axis = _axis / _axis.norm()
    _appr = (torch.tensor([cup_xy[0], cup_xy[1], cup_z], device=_tips.device)
             - torch.tensor(cage_c, device=_tips.device))
    _appr = _appr / _appr.norm()
    _cos = float(abs((_axis * _appr).sum()))
    out["home"]["open_axis_world"] = _fmt(_axis)
    out["home"]["approach_dir_world"] = _fmt(_appr)
    out["home"]["axis_vs_approach_deg"] = round(math.degrees(math.acos(min(1.0, _cos))), 1)
    print(f"[probe] 홈 palm={_fmt(home6)} (euler deg {out['home']['palm_euler_deg']})", flush=True)
    print(f"[probe] 벌림축 {_fmt(_axis)} · 접근방향 {_fmt(_appr)} · 사잇각 "
          f"{out['home']['axis_vs_approach_deg']}° "
          f"({'평행 = 정면 감쌈' if out['home']['axis_vs_approach_deg'] < 30 else '어긋남 = 옆으로 쓸어낸다'})",
          flush=True)
    print(f"[probe] 케이지 중심={_fmt(cage_c)} 반경 {cage_r*1000:.0f}mm · "
          f"케이지−컵 {out['home']['cage_minus_cup_mm']}mm · xy간격 "
          f"{out['home']['cage_cup_gap_xy_mm']}mm", flush=True)
    print(f"[probe] 손 최하단 z(open) {hand_z_min_open:.4f} · palm−손 "
          f"{out['home']['palm_minus_hand_z']:.4f} m", flush=True)

    # ---------------------------------------------------------------- ③ 파지 창(open/grip)
    def _hand_pose(pose) -> dict:
        tgt = q0.clone()
        for k, name in enumerate(profile.hand_joint_names):
            tgt[:, jn.index(name)] = float(pose[k])
        robot.set_joint_position_target(tgt)
        _settle(sim, robot, args.settle, dt, grav)
        tips = robot.data.body_pos_w[0, tip_ids]
        gap = float((tips[0] - tips[1:].mean(dim=0)).norm())
        per = {profile.fingertip_bodies[i + 1].split("r_hl_")[1]:
               round(float((tips[0] - tips[i + 1]).norm()) * 1000, 1) for i in range(4)}
        q = robot.data.joint_pos[0]
        return {
            "thumb_to_four_gap_mm": round(gap * 1000, 1),
            "thumb_to_each_mm": per,
            "hand_z_min": round(float(robot.data.body_pos_w[0, hand_body_ids, 2].min()), 4),
            "driven_q": {n: round(float(q[jn.index(n)]), 4) for n in profile.hand_joint_names},
            "dep_q": {n: round(float(q[jn.index(n)]), 4) for n in _MIMIC},
        }

    _home()
    out["grasp_window"] = {"open": _hand_pose(profile.hand_open_pose)}
    _home()
    out["grasp_window"]["grip"] = _hand_pose(profile.hand_grip_pose)
    print(f"[probe] 엄지-4지 간극 open {out['grasp_window']['open']['thumb_to_four_gap_mm']}mm → "
          f"grip {out['grasp_window']['grip']['thumb_to_four_gap_mm']}mm", flush=True)

    # ---------------------------------------------------------------- ④ 접촉 스트레스
    # 케이지 중심에 **고정** 원통을 두고 손을 완전 폐쇄한다. 09.02 발산은 구동관절이
    # 접촉에 밀려 하한 밖으로 나갈 때 종속 제약이 동시에 만족 불가가 되며 났다.
    _home()
    if args.press_mode == "table":
        # ★학습 실패 재현: 팔이 손을 **상판 아래로** 민다. rh_b1 실측에서 손 최하단이
        #   0.15~0.175(상판 0.205)일 때만 mimic 이 폭주했다 — 그 조건을 그대로 만든다.
        _tz = float(cfg.table_surface_z)
        _box = sim_utils.CuboidCfg(
            size=(0.8, 0.8, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.4, 0.3)))
        _box.func("/World/PressTable", _box, translation=(0.45, -0.25, _tz - 0.1))
        _goal = torch.tensor([cage_c[0], cage_c[1], _tz - float(args.press_depth)],
                             device=robot.device)
        q_push = _ik(sim, robot, dt, grav, palm_idx, cage_off_palm, q0.clone(), _goal, arm_ids)
        _home()
        tgt = q_push.clone()
        for k, name in enumerate(profile.hand_joint_names):
            tgt[:, jn.index(name)] = float(profile.hand_grip_pose[k])
        robot.set_joint_position_target(tgt)
        worst = _press_loop(sim, robot, dt, grav, args.press_steps, dep_ids, syn_ids,
                            arm_ids, lo, hi)
        q = robot.data.joint_pos[0]
        worst["hand_z_min"] = round(float(robot.data.body_pos_w[0, hand_body_ids, 2].min()), 4)
        worst["final_driven_q"] = {n: round(float(q[jn.index(n)]), 4)
                                   for n in profile.hand_joint_names}
        worst["final_dep_err_rad"] = {
            d: round(float(q[jn.index(d)] - m * q[jn.index(l)]), 4)
            for d, (l, m) in _MIMIC.items()}
        out["press_stress"] = worst
        print(f"[probe] 테이블 프레스: 손최하단 {worst['hand_z_min']} · 종속|qd|max "
              f"{worst['dep_qd_max']} · 한계위반 {worst['dep_violation_rad']} rad · "
              f"구동 하한이탈 {worst['drive_below_lo_rad']} rad · mimic오차 "
              f"{max(abs(v) for v in worst['final_dep_err_rad'].values()):.3f} · NaN {worst['nan']}",
              flush=True)
        path = os.path.join(os.getcwd(), args.out)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[probe] 저장 {path}", flush=True)
        return

    _kin = float(args.press_mass) <= 0.0
    cyl = sim_utils.CylinderCfg(
        radius=float(args.press_radius), height=0.12,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=_kin),
        mass_props=None if _kin else sim_utils.MassPropertiesCfg(mass=float(args.press_mass)),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.9)))
    # ★skip-press 면 물체를 손이 닿지 않는 곳에 두고 루프를 0 스텝으로 돌린다 —
    #   씬에 남은 스트레스 물체가 손을 눌러 케이지 반경이 작게 측정된다(09.07: 53 → 29mm).
    _pos = (cage_c[0], cage_c[1], cage_c[2] - 5.0) if args.skip_press else tuple(cage_c)
    cyl.func("/World/PressObj", cyl, translation=_pos)
    tgt = q0.clone()
    for k, name in enumerate(profile.hand_joint_names):
        tgt[:, jn.index(name)] = float(profile.hand_grip_pose[k])
    robot.set_joint_position_target(tgt)
    worst = {"dep_qd_max": 0.0, "dep_violation_rad": 0.0, "drive_below_lo_rad": 0.0,
             "arm_qd_max": 0.0, "nan": False}
    for _ in range(0 if args.skip_press else int(args.press_steps)):
        _tau = robot.root_physx_view.get_gravity_compensation_forces()
        robot.set_joint_effort_target(_tau[:, grav], joint_ids=grav.tolist())
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt)
        q, qd = robot.data.joint_pos[0], robot.data.joint_vel[0]
        if bool(torch.isnan(q).any()):
            worst["nan"] = True
            break
        worst["dep_qd_max"] = max(worst["dep_qd_max"], float(qd[dep_ids].abs().max()))
        worst["arm_qd_max"] = max(worst["arm_qd_max"], float(qd[arm_ids].abs().max()))
        worst["dep_violation_rad"] = max(worst["dep_violation_rad"], float(
            torch.maximum(q[dep_ids] - hi[dep_ids], lo[dep_ids] - q[dep_ids]).max().clamp(min=0)))
        worst["drive_below_lo_rad"] = max(worst["drive_below_lo_rad"], float(
            torch.relu(lo[syn_ids] - q[syn_ids]).max()))
    q = robot.data.joint_pos[0]
    worst = {k: (v if isinstance(v, bool) else round(v, 4)) for k, v in worst.items()}
    worst["final_driven_q"] = {n: round(float(q[jn.index(n)]), 4)
                               for n in profile.hand_joint_names}
    worst["final_dep_err_rad"] = {
        d: round(float(q[jn.index(d)] - m * q[jn.index(l)]), 4) for d, (l, m) in _MIMIC.items()}
    out["press_stress"] = worst
    print(f"[probe] 스트레스: 종속|qd|max {worst['dep_qd_max']} · 한계위반 "
          f"{worst['dep_violation_rad']} rad · 구동 하한이탈 {worst['drive_below_lo_rad']} rad · "
          f"NaN {worst['nan']}", flush=True)

    # ---------------------------------------------------------------- ④' 새 홈 풀기
    if args.solve_home:
        _home()                                  # ★스트레스 자세를 지우고 개방 홈에서 시작
        _off = [float(v) for v in args.solve_home.split(",")]
        _goal = torch.tensor([cup_xy[0] + _off[0], cup_xy[1] + _off[1], cup_z + _off[2]],
                             device=robot.device)
        _arm_ids, _ = robot.find_joints(profile.arm_joint_regex, preserve_order=True)
        q_new = _ik(sim, robot, dt, grav, palm_idx, cage_off_palm, q0.clone(), _goal, _arm_ids)
        robot.write_joint_state_to_sim(q_new, torch.zeros_like(q_new))
        robot.set_joint_position_target(q_new)
        _settle(sim, robot, args.settle, dt, grav)
        _c, _r = _cage()
        _p6 = _pose6()
        _hz = float(robot.data.body_pos_w[0, hand_body_ids, 2].min())
        _gap = float((torch.tensor(_c[:2]) - torch.tensor(list(cup_xy))).norm())
        _lo6 = list(profile.palm_box_min) + [math.radians(v - profile.palm_rot_half_deg)
                                             for v in profile.palm_rot_center_deg]
        _hi6 = list(profile.palm_box_max) + [math.radians(v + profile.palm_rot_half_deg)
                                             for v in profile.palm_rot_center_deg]
        _out_ax = [ax for k, ax in enumerate(("x", "y", "z", "ez", "ey", "ex"))
                   if _p6[k] < _lo6[k] or _p6[k] > _hi6[k]]
        out["solve_home"] = {
            "target_cage": _fmt(_goal), "arm_q": {n: round(float(q_new[0, jn.index(n)]), 4)
                                                  for n in jn if n.startswith("r_aj_")},
            "cage": _fmt(_c), "cage_radius_mm": round(_r * 1000, 1),
            "cage_minus_cup_mm": _fmt([(_c[i] - [cup_xy[0], cup_xy[1], cup_z][i]) * 1000
                                       for i in range(3)], 1),
            "standoff_mm": round(float((torch.tensor(_c) - torch.tensor(
                [cup_xy[0], cup_xy[1], cup_z])).norm()) * 1000, 1),
            "standoff_over_radius": round(float((torch.tensor(_c) - torch.tensor(
                [cup_xy[0], cup_xy[1], cup_z])).norm()) / max(_r, 1e-6), 2),
            "gap_xy_mm": round(_gap * 1000, 1), "hand_z_min": round(_hz, 4),
            "palm_pose6": _fmt(_p6), "palm_out_of_box_axes": _out_ax,
        }
        for k, v in out["solve_home"].items():
            print(f"[probe] home/{k}: {v}", flush=True)
        print("[probe] 판정: "
              f"{'OK' if (not _out_ax and _gap > _r and _hz > 0.215) else '⚠ 조건 위반'} "
              f"(박스 안 · 간격 {_gap*1000:.0f}mm > 반경 {_r*1000:.0f}mm · 손 {_hz:.3f} > 0.215)",
              flush=True)

    # ---------------------------------------------------------------- ⑤ 목표 박스 도달성
    # ★왜 여기서 재나: Track B 는 팔이 관절공간이라 부모의 `_assert_goal_box_in_arm_reach`
    #   가 **일찍 반환한다**(palm 박스가 지령 한계가 아니므로). 즉 목표열이 팔 사거리
    #   밖으로 나가도 부팅이 안 잡는다 — 목표가 안 닿으면 목표열이 조용히 멈추고
    #   "학습이 안 되는 것"으로 위장된다.
    if not (args.only_press or args.skip_reach):
        out["goal_reach"] = _goal_box_reach(sim, robot, profile, dt, grav, palm_idx,
                                            cage_off_palm, q0)

    path = os.path.join(os.getcwd(), args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[probe] 저장 {path}", flush=True)


main()
_app.close()

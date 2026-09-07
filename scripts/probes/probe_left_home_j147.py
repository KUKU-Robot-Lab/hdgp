# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""왼팔 홈 후보 탐색 — **J1·J4·J7 만** 쓰는 뒤로 뺀 자세.

`probe_left_home_search.py` 는 7 관절 전부를 훑는다. 이 프로브는 사용자 지시대로
**j2=j3=j5=j6=0 으로 고정**하고 j1(base yaw) · j4(elbow pitch) · j7(wrist pitch)
세 축만 격자로 훑는다. 세 축은 각각 축이 z · y · -y 라, 이 조합은 "한 평면 안에서
팔꿈치·손목으로 뒤로 접었다 앞으로 펴는" 동작 + yaw 하나가 된다.

목적은 S2R 에서 드러난 **TCP 가 테이블 상면을 긁는 문제**다. 홈이 판 위 낮은 높이에
있으면 리셋 직후와 접근 초기에 그리퍼가 상면을 쓸고 지나간다.

판정 조건(전부 통과해야 후보):
  1. j2=j3=j5=j6 = 0 (구성상 자동)
  2. **모든 좌팔·좌손 링크 z > TABLE_SURFACE_Z + 0.010** — 판을 긁지 않는다
  3. **TCP z > TABLE_SURFACE_Z + 0.010** — 사용자 지시(상면보다 1 cm 이상)
  4. **TCP x < 컵 스폰 하한** — 컵보다 뒤에 있어야 "앞으로 전진"이 된다
  5. 관절 한계에서 0.5 rad 여유 — 액션 범위(±0.5 rad)를 온전히 쓴다
  6. 컵 관통 없음 — 홈이 컵 자리를 점유하면 스폰 즉시 튕겨나간다

실행:
    PYTHONUNBUFFERED=1 python -u scripts/probes/probe_left_home_j147.py
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--n1", type=int, default=9, help="j1 격자 점 수")
parser.add_argument("--n4", type=int, default=13, help="j4 격자 점 수")
parser.add_argument("--n7", type=int, default=13, help="j7 격자 점 수")
parser.add_argument("--tcp_clear", type=float, default=0.010,
                    help="TCP·링크가 상면보다 최소 이만큼 위 (m)")
parser.add_argument("--tcp_z_max", type=float, default=0.50,
                    help="TCP z 상한 — 이보다 높으면 팔을 세운 쓸모없는 해")
parser.add_argument("--tcp_x_min", type=float, default=0.05,
                    help="TCP x 하한 — 로봇 앞쪽이어야 한다(음수면 뒤로 넘어간 자세)")
parser.add_argument("--d_cup", type=str, default="0.10,0.35",
                    help="TCP-컵 거리 허용 구간 (m) — 뒤로 뺐되 전진으로 닿을 거리")
parser.add_argument("--jaw_max", type=float, default=15.0, help="jaw 수평 이탈 상한 (deg)")
parser.add_argument("--appr_max", type=float, default=45.0, help="접근축 이탈 상한 (deg)")
parser.add_argument("--top", type=int, default=8, help="출력할 상위 후보 수")
parser.add_argument("--poses", type=str, default="",
                    help="격자 대신 지정한 자세만 평가. 'j1,j4,j7' 을 ';' 로 구분")
parser.add_argument("--reach", type=str, default="",
                    help="도달성 검사 모드. 'j1,j4,j7' 하나를 홈으로 두고 액션 상자"
                         "(±--reach_span rad, 7 관절)를 무작위로 훑어 컵·목표 도달을 잰다")
parser.add_argument("--num_reach", type=int, default=4096,
                    help="도달성 검사 표본 수")
parser.add_argument("--reach_span", type=float, default=0.5,
                    help="액션 반범위 (rad) — arm_action scale 과 같아야 한다")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import xml.etree.ElementTree as ET  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from openarm import OPENARM_ROOT_DIR  # noqa: E402
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P  # noqa: E402

TASK = "open-grip_l_grasp_sensor"
ACTION_HALF_RANGE = 0.5           # scale=0.5 → 액션 ±1 이 ±0.5 rad
FREE_JOINTS = ("l_aj_1", "l_aj_4", "l_aj_7")
ZERO_JOINTS = ("l_aj_2", "l_aj_3", "l_aj_5", "l_aj_6")


def _joint_limits() -> dict[str, tuple[float, float]]:
    urdf = Path(OPENARM_ROOT_DIR).resolve().parents[2] / (
        "assets/robot/openarm_tesollo_sensor_rl/openarm_tesollo_sensor_rl.urdf"
    )
    out = {}
    for j in ET.parse(urdf).getroot().iter("joint"):
        name = j.get("name") or ""
        lim = j.find("limit")
        if name.startswith("l_aj_") and lim is not None:
            out[name] = (float(lim.get("lower")), float(lim.get("upper")))
    return out


def _explicit_poses() -> torch.Tensor:
    """`--poses` 로 받은 j1,j4,j7 조합을 그대로 쓴다(격자 없음)."""
    rows = []
    for chunk in args.poses.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals = [float(v) for v in chunk.split(",")]
        if len(vals) != 3:
            raise SystemExit(f"--poses 항목은 'j1,j4,j7' 3 개여야 한다: {chunk!r}")
        rows.append(vals)
    if not rows:
        raise SystemExit("--poses 에 유효한 자세가 없다")
    return torch.tensor(rows)


def _grid(limits: dict[str, tuple[float, float]]) -> torch.Tensor:
    """한계여유 0.5 를 지키는 구간 안에서 j1·j4·j7 격자를 만든다."""
    axes = []
    for jn, n in zip(FREE_JOINTS, (args.n1, args.n4, args.n7)):
        lo = limits[jn][0] + ACTION_HALF_RANGE
        hi = limits[jn][1] - ACTION_HALF_RANGE
        axes.append(torch.linspace(lo, hi, n))
    g1, g4, g7 = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([g1.reshape(-1), g4.reshape(-1), g7.reshape(-1)], dim=-1)


def _reach_main(limits: dict[str, tuple[float, float]]) -> None:
    """홈을 고정하고 **액션 상자 안에서** 컵·목표에 닿는지 잰다.

    액션은 홈 기준 ±`reach_span` rad 국소 오프셋이다. 그래서 홈이 멀면 액션 상자 안에
    컵이 안 들어올 수 있다. 위치만 보면 안 된다 — 물기 자세(jaw 수평·접근축)까지
    동시에 만족하는 표본이 있어야 실제로 잡을 수 있다.
    """
    vals = [float(v) for v in args.reach.split(",")]
    if len(vals) == 3:
        home = dict(zip(FREE_JOINTS, vals))
        for jn in ZERO_JOINTS:
            home[jn] = 0.0
    elif len(vals) == 7:
        # 7 개면 j1..j7 전체를 그대로 쓴다 — 현재 홈을 대조군으로 재기 위한 경로다.
        home = {f"l_aj_{i + 1}": v for i, v in enumerate(vals)}
    else:
        raise SystemExit("--reach 는 'j1,j4,j7'(3 개) 또는 j1..j7(7 개)여야 한다")
    n = args.num_reach
    print(f"[reach] 홈 " + " ".join(f"{k}={home[k]:+.4f}" for k in sorted(home))
          + f" · 액션 상자 ±{args.reach_span} rad · 표본 {n}")

    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=n)
    env_cfg.events.reset_object_position.params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)
    }
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    obj = env.scene["object"]
    ee = env.scene["ee_frame"]
    origins = env.scene.env_origins
    arm_ids, arm_names = robot.find_joints(P.LEFT_ARM_JOINT_NAMES, preserve_order=True)
    base_i = robot.body_names.index(P.GRIPPER_BASE_BODY)

    gen = torch.Generator(device="cpu").manual_seed(0)
    target = robot.data.default_joint_pos.clone()
    for j, jn in enumerate(arm_names):
        lo, hi = limits[jn]
        off = (torch.rand(n, generator=gen) * 2.0 - 1.0) * args.reach_span
        q = (home[jn] + off).clamp(lo, hi)
        q[0] = home[jn]                       # env 0 = 홈 그대로(액션 0)
        target[:, arm_ids[j]] = q.to(target.device)
    robot.data.default_joint_pos[:] = target
    robot.write_joint_state_to_sim(target, torch.zeros_like(target))
    env.action_manager.get_term("arm_action")._offset[:] = target[:, arm_ids]

    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(args.steps):
        env.step(zero.clone())

    tcp = ee.data.target_pos_w[:, 0, :] - origins
    q0 = robot.data.body_quat_w[:, base_i, :]
    w, x, y, z = q0.unbind(-1)
    jaw = torch.rad2deg(torch.asin((2.0 * (y * z + w * x)).abs().clamp(max=1.0)))
    appr = torch.rad2deg(torch.asin((1.0 - 2.0 * (x * x + y * y)).abs().clamp(max=1.0)))
    cup = (obj.data.root_pos_w - origins)[0]
    goal = torch.tensor(P.GOAL_POINT, device=tcp.device)

    d_cup = (tcp - cup).norm(dim=-1)
    d_goal = (tcp - goal).norm(dim=-1)
    pose_ok = (jaw < 10.0) & (appr < 35.0)

    print(f"\n=== 도달성 (홈에서 액션 상자 안) ===")
    print(f"  컵(참값) ({cup[0]:.3f}, {cup[1]:.3f}, {cup[2]:.3f})  "
          f"목표점 ({P.GOAL_POINT[0]:.3f}, {P.GOAL_POINT[1]:.3f}, {P.GOAL_POINT[2]:.3f})")
    print(f"  홈에서: TCP-컵 {d_cup[0] * 1e3:.1f}mm · TCP-목표 {d_goal[0] * 1e3:.1f}mm")
    for tag, d in (("컵", d_cup), ("목표", d_goal)):
        for thr in (0.03, 0.05, 0.08):
            hit = (d < thr)
            print(f"  {tag} {thr * 1e3:3.0f}mm 이내 도달  "
                  f"{100.0 * hit.float().mean():6.2f}%   "
                  f"+물기자세 동시 {100.0 * (hit & pose_ok).float().mean():6.2f}%")
        print(f"    → 최소 거리 {d.min() * 1e3:.1f}mm   "
              f"자세까지 만족하는 표본 중 최소 "
              f"{(d[pose_ok].min() * 1e3 if pose_ok.any() else float('nan')):.1f}mm")
    # ★리프트 여유 — 컵 근처에 도달한 표본이 **위로 얼마나 더 갈 수 있는가**.
    #   파지만 되고 리프트가 0 인 현상(라운드 17 갈래 A)의 직접 진단이다.
    near = (d_cup < 0.08)
    if near.any():
        dz = (tcp[near, 2] - cup[2])
        q = lambda t: float(dz.quantile(t)) * 1e3
        print(f"\n  ★컵 80 mm 이내 표본 {int(near.sum())} 개의 TCP z − 컵 z (mm)")
        print(f"    p10 {q(0.1):+.1f} · 중앙 {q(0.5):+.1f} · p90 {q(0.9):+.1f} · "
              f"최대 {float(dz.max()) * 1e3:+.1f}")
        for h in (0.02, 0.04, 0.06):
            print(f"    +{h * 1e3:.0f} mm 이상 들 수 있는 표본  "
                  f"{100.0 * (dz > h).float().mean():6.2f}%")
    else:
        print("\n  ★컵 80 mm 이내 표본 없음 — 리프트 여유를 잴 수 없다")
    print("\nJ147DONE")
    env.close()


def main() -> None:
    limits = _joint_limits()
    if args.reach:
        _reach_main(limits)
        return
    explicit = bool(args.poses)
    grid = _explicit_poses() if explicit else _grid(limits)
    # env 0 은 대조군(현재 홈)이라 후보 앞에 한 칸 붙인다
    n_samples = grid.shape[0] + 1
    if explicit:
        print(f"[poses] 지정 자세 {grid.shape[0]} 개를 평가한다")
        for row in grid.tolist():
            slack = min(min(v - limits[jn][0], limits[jn][1] - v)
                        for v, jn in zip(row, FREE_JOINTS))
            mark = "" if slack >= ACTION_HALF_RANGE else "  ← 한계여유 부족"
            print(f"       j1={row[0]:+.4f} j4={row[1]:+.4f} j7={row[2]:+.4f} "
                  f"한계여유 {slack:.4f}{mark}")
    else:
        print(f"[grid] j1×j4×j7 = {args.n1}×{args.n4}×{args.n7} = {grid.shape[0]} 후보")
    for jn in FREE_JOINTS:
        lo = limits[jn][0] + ACTION_HALF_RANGE
        hi = limits[jn][1] - ACTION_HALF_RANGE
        print(f"       {jn}  액션범위 확보구간 [{lo:+.4f}, {hi:+.4f}]  (한계 {limits[jn]})")

    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=n_samples)
    env_cfg.events.reset_object_position.params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)
    }
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    obj = env.scene["object"]
    ee = env.scene["ee_frame"]
    origins = env.scene.env_origins
    arm_ids, arm_names = robot.find_joints(P.LEFT_ARM_JOINT_NAMES, preserve_order=True)
    base_i = robot.body_names.index(P.GRIPPER_BASE_BODY)
    left_idx = [i for i, nm in enumerate(robot.body_names) if nm.startswith(("l_hl_", "l_al_"))]

    target = robot.data.default_joint_pos.clone()
    free_cols = [arm_names.index(jn) for jn in FREE_JOINTS]
    for jn in ZERO_JOINTS:
        target[:, arm_ids[arm_names.index(jn)]] = 0.0
    for c, jn in zip(free_cols, FREE_JOINTS):
        target[1:, arm_ids[c]] = grid[:, FREE_JOINTS.index(jn)].to(target.device)
    # env 0 = 대조군(현재 홈)
    for j, jn in enumerate(arm_names):
        target[0, arm_ids[j]] = P.LEFT_ARM_HOME_JOINT_POS[jn]

    robot.data.default_joint_pos[:] = target
    robot.write_joint_state_to_sim(target, torch.zeros_like(target))
    env.action_manager.get_term("arm_action")._offset[:] = target[:, arm_ids]

    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    env.step(zero.clone())
    cup0 = (obj.data.root_pos_w - origins).clone()
    tcp = (ee.data.target_pos_w[:, 0, :] - origins).clone()
    q0 = robot.data.body_quat_w[:, base_i, :].clone()
    for _ in range(args.steps):
        env.step(zero.clone())

    cup = obj.data.root_pos_w - origins
    cup_moved = (cup[:, :2] - cup0[:, :2]).norm(dim=-1)
    cx, cy = obj.data.root_quat_w[:, 1], obj.data.root_quat_w[:, 2]
    cup_tilt = torch.rad2deg(torch.acos((1 - 2 * (cx * cx + cy * cy)).clamp(-1.0, 1.0)))
    w, x, y, z = q0.unbind(-1)
    jaw_deg = torch.rad2deg(torch.asin((2.0 * (y * z + w * x)).abs().clamp(max=1.0)))
    appr_deg = torch.rad2deg(torch.asin((1.0 - 2.0 * (x * x + y * y)).abs().clamp(max=1.0)))
    d_cup = (tcp - cup0).norm(dim=-1)
    low_z = (robot.data.body_pos_w[:, left_idx, 2] - origins[:, 2:3]).min(dim=1).values

    z_floor = P.TABLE_SURFACE_Z + args.tcp_clear
    x_behind = P.CUP_SPAWN_X_CENTER - P.CUP_SPAWN_X_RANGE

    print(f"\n=== J1·J4·J7 전용 홈 탐색 ===")
    print(f"  테이블 상면 z={P.TABLE_SURFACE_Z}  판 x={P.WORK_SURFACE_X}")
    print(f"  요구: 모든 좌팔 링크 z > {z_floor:.3f} · TCP z > {z_floor:.3f} · TCP x < {x_behind:.3f}")
    print(f"  컵(참값) {cup0[1].tolist()}")
    k0 = 0
    print(f"  [대조군=현재 홈] TCP=({tcp[k0, 0]:.3f},{tcp[k0, 1]:.3f},{tcp[k0, 2]:.3f}) "
          f"최저링크z={low_z[k0]:.4f} TCP-컵={d_cup[k0] * 1e3:.1f}mm "
          f"jaw={jaw_deg[k0]:.1f}° 접근={appr_deg[k0]:.1f}°")

    d_lo, d_hi = (float(v) for v in args.d_cup.split(","))
    if explicit:
        print("\n  --- 지정 자세 실측 ---")
        for k in range(1, n_samples):
            j1 = float(target[k, arm_ids[arm_names.index('l_aj_1')]])
            j4 = float(target[k, arm_ids[arm_names.index('l_aj_4')]])
            j7 = float(target[k, arm_ids[arm_names.index('l_aj_7')]])
            flags = []
            if float(low_z[k]) <= z_floor:
                flags.append(f"링크가 판에 {(z_floor - float(low_z[k])) * 1e3:.0f}mm 못미침")
            if float(tcp[k, 2]) <= z_floor:
                flags.append("TCP z 미달")
            if float(tcp[k, 0]) >= x_behind:
                flags.append("TCP 가 컵보다 앞")
            if float(cup_moved[k]) >= 0.002 or float(cup_tilt[k]) >= 1.0:
                flags.append(f"컵 관통(이동 {float(cup_moved[k]) * 1e3:.1f}mm)")
            print(f"  j1={j1:+.4f} j4={j4:+.4f} j7={j7:+.4f}")
            print(f"    TCP=({tcp[k, 0]:.3f}, {tcp[k, 1]:.3f}, {tcp[k, 2]:.3f})  "
                  f"판 위 {(float(tcp[k, 2]) - P.TABLE_SURFACE_Z) * 1e3:+.0f}mm")
            print(f"    최저 링크 z={low_z[k]:.4f} (판 위 "
                  f"{(float(low_z[k]) - P.TABLE_SURFACE_Z) * 1e3:+.0f}mm)  "
                  f"TCP-컵 {d_cup[k] * 1e3:.1f}mm")
            print(f"    jaw {jaw_deg[k]:.1f}°  접근 {appr_deg[k]:.1f}°  "
                  f"{'· '.join(flags) if flags else '조건 통과'}")
    ranked = []
    fail = {"link_z": 0, "tcp_z": 0, "tcp_x": 0, "d_cup": 0, "pose": 0, "penetrate": 0}
    for k in range(1, n_samples):
        quiet = float(cup_moved[k]) < 0.002 and float(cup_tilt[k]) < 1.0
        ok_link = float(low_z[k]) > z_floor
        ok_tcpz = z_floor < float(tcp[k, 2]) < args.tcp_z_max
        ok_tcpx = args.tcp_x_min < float(tcp[k, 0]) < x_behind
        ok_dcup = d_lo < float(d_cup[k]) < d_hi
        ok_pose = float(jaw_deg[k]) < args.jaw_max and float(appr_deg[k]) < args.appr_max
        if not ok_link:
            fail["link_z"] += 1
        if not ok_tcpz:
            fail["tcp_z"] += 1
        if not ok_tcpx:
            fail["tcp_x"] += 1
        if not ok_dcup:
            fail["d_cup"] += 1
        if not ok_pose:
            fail["pose"] += 1
        if not quiet:
            fail["penetrate"] += 1
        if not (quiet and ok_link and ok_tcpz and ok_tcpx and ok_dcup and ok_pose):
            continue
        # 점수: 물기 자세(jaw·접근 수평)가 최우선. 판 여유는 **넉넉하면 충분**이라
        #   30 mm 까지만 인정한다 — 그 이상은 팔을 세우는 쓸모없는 해로 끌려간다.
        clear_bonus = min(float(low_z[k]) - z_floor, 0.030)
        score = float(jaw_deg[k]) + float(appr_deg[k]) - 100.0 * clear_bonus
        ranked.append((score, k))

    print(f"\n  조건 통과 {len(ranked)}/{n_samples - 1}"
          f"   (탈락: 링크z {fail['link_z']} · TCPz {fail['tcp_z']} · TCPx {fail['tcp_x']} · "
          f"컵거리 {fail['d_cup']} · 자세 {fail['pose']} · 관통 {fail['penetrate']})")
    if not ranked:
        print("  → 후보 없음. --tcp_clear 를 낮추거나 격자를 조밀하게 할 것.")
    else:
        ranked.sort()
        print(f"\n{'순위':>4} {'j1':>9} {'j4':>9} {'j7':>9} {'TCP x':>7} {'TCP z':>7} "
              f"{'최저링크z':>9} {'TCP-컵':>9} {'jaw':>7} {'접근':>7}")
        for rank, (_s, k) in enumerate(ranked[:args.top], 1):
            j1 = float(target[k, arm_ids[arm_names.index('l_aj_1')]])
            j4 = float(target[k, arm_ids[arm_names.index('l_aj_4')]])
            j7 = float(target[k, arm_ids[arm_names.index('l_aj_7')]])
            print(f"{rank:4d} {j1:+9.4f} {j4:+9.4f} {j7:+9.4f} "
                  f"{tcp[k, 0]:7.3f} {tcp[k, 2]:7.3f} {low_z[k]:9.4f} "
                  f"{d_cup[k] * 1e3:8.1f} {jaw_deg[k]:6.1f}° {appr_deg[k]:6.1f}°")
    print("\nJ147DONE")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

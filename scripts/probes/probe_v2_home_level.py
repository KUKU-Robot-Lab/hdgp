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

"""라운드 22 · Part 2 — **접근축을 수평으로 세운 홈**을 현재 홈 근처에서 찾는다.

`probe_left_home_j147.py` 와 목적이 다르다. 그 프로브는 j2·j3·j5·j6 을 0 으로
고정한 "뒤로 뺀 자세"를 훑었고, 그 결과 도달성이 무너져 기각됐다. 여기서는
**현재 홈을 최소로 흔들어** 접근각만 세운다.

현재 홈: 접근축이 world +z 와 **103.9°** (= 수평보다 13.9° 아래로 기욺).
G(w=7.0) 실측에서 정책이 이 14° 를 스스로 세우는 데 액션 6축 중 4축을 포화시키고도
파지점 48 mm 앞에서 멈췄다. 즉 **각도 유지 비용을 홈이 만들고 있다** — 홈에서 이미
수평이면 그 비용이 0 이 된다.

판정(전부 통과해야 후보):
  1. 접근각이 `--target_deg ± --deg_tol` 안
  2. TCP 이동 ≤ `--tcp_tol` (현재 홈 대비) — 도달 봉투를 지키기 위한 핵심 조건
  3. 모든 좌팔·좌그리퍼 링크 z > 상면 + `--clear`
  4. 관절 한계 여유 ≥ 0.5 rad (액션 상자를 온전히 쓴다)
  5. 컵 관통 없음(컵 이동 ≈ 0)

실행:
    PYTHONUNBUFFERED=1 python -u scripts/probes/probe_v2_home_level.py --num 4096
    PYTHONUNBUFFERED=1 python -u scripts/probes/probe_v2_home_level.py \
        --reach "j1,j2,...,j7" --num_reach 8192
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num", type=int, default=4096, help="탐색 표본 수")
parser.add_argument("--span", type=float, default=0.35, help="홈 대비 관절 변위 한계 (rad)")
parser.add_argument("--free", type=str, default="2,4,6,7",
                    help="흔들 관절 번호 (쉼표). 기본은 pitch 사슬 j2·j4·j6·j7")
parser.add_argument("--target_deg", type=float, default=93.0,
                    help="목표 접근각(도). 90 = 완전 수평, 90 미만 = 위로")
parser.add_argument("--deg_tol", type=float, default=2.0)
parser.add_argument("--tcp_tol", type=float, default=0.010, help="TCP 이동 상한 (m)")
parser.add_argument("--clear", type=float, default=0.010, help="상면 여유 (m)")
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--top", type=int, default=10)
parser.add_argument("--min_slack", type=float, default=0.5,
                    help="관절 한계 여유 하한(rad). 홈 후보를 고를 땐 0.5 가 맞지만, "
                         "**기구학 한계 자체**를 물을 땐 0 으로 둬야 한다 — 정책은 홈 "
                         "선정 기준에 묶이지 않고 관절 한계까지 쓸 수 있다")
parser.add_argument("--tcp_z_target", type=float, default=0.0,
                    help="0 이 아니면 TCP 이동이 아니라 **TCP z 를 이 값에 맞추는** 홈을 "
                         "찾는다. 라운드 22 가설: 수평 접근이 안 되는 이유가 홈 TCP 가 "
                         "파지점보다 45~80mm 위라서라면, 처음부터 파지 높이에 두면 된다")
parser.add_argument("--tcp_z_tol", type=float, default=0.015)
parser.add_argument("--reach", type=str, default="",
                    help="도달성 검사 모드. j1..j7 7 개 값을 홈으로 두고 액션 상자를 훑는다")
parser.add_argument("--num_reach", type=int, default=8192)
parser.add_argument("--reach_span", type=float, default=0.5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from openarm import OPENARM_ROOT_DIR  # noqa: E402
from openarm.gripper.left.grasp_sensor_v2 import v2_preset as P  # noqa: E402

TASK = "open-grip_l_grasp_sensor_v2"
ACTION_HALF_RANGE = 0.5


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


def _approach_deg(quat: torch.Tensor) -> torch.Tensor:
    """`v2_rewards._approach_az` 와 **같은 식** — 그리퍼 +z 의 world z 성분."""
    az = (1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(az))


def _build(n: int):
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=n)
    env_cfg.events.reset_object_position.params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)
    }
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()
    return env


def _hold(env, target):
    """관절을 써 넣고 **그 자세를 목표로 잡아** 물리를 몇 스텝 정착시킨다.

    ⚠ v2 의 `arm_action` 은 `FabricPalmAction` 이라 v1 프로브처럼 `_offset` 을
      건드릴 수 없다(그 경로로 `env.step` 을 부르면 fabric 이 팔을 딴 데로 끌고 간다).
      여기서는 액션 매니저를 아예 우회하고 물리만 돌린다.
    """
    robot = env.scene["robot"]
    robot.write_joint_state_to_sim(target, torch.zeros_like(target))
    robot.data.default_joint_pos[:] = target
    dt = env.sim.get_physics_dt()
    for _ in range(args.steps):
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt)


def _search() -> None:
    limits = _joint_limits()
    free = [int(v) for v in args.free.split(",")]
    home = [P.LEFT_ARM_HOME_JOINT_POS[f"l_aj_{i}"] for i in range(1, 8)]
    print(f"[home] 현재 홈 " + " ".join(f"j{i+1}={home[i]:+.4f}" for i in range(7)))
    print(f"[search] 흔들 관절 j{free} · ±{args.span} rad · 표본 {args.num}")
    print(f"[target] 접근각 {args.target_deg}±{args.deg_tol}° · TCP 이동 ≤ {args.tcp_tol*1e3:.0f}mm")

    n = args.num + 1
    gen = torch.Generator(device="cpu").manual_seed(0)
    q = torch.tensor(home).repeat(n, 1)
    for j in free:
        jn = f"l_aj_{j}"
        lo = limits[jn][0] + ACTION_HALF_RANGE
        hi = limits[jn][1] - ACTION_HALF_RANGE
        d = (torch.rand(n, generator=gen) * 2.0 - 1.0) * args.span
        q[:, j - 1] = (q[:, j - 1] + d).clamp(lo, hi)
    q[0] = torch.tensor(home)                      # env 0 = 대조군

    env = _build(n)
    robot = env.scene["robot"]
    obj = env.scene["object"]
    ee = env.scene["ee_frame"]
    origins = env.scene.env_origins
    arm_ids, _ = robot.find_joints(P.LEFT_ARM_JOINT_NAMES, preserve_order=True)
    base_i = robot.body_names.index(P.GRIPPER_BASE_BODY)
    cup0 = (obj.data.root_pos_w - origins).clone()

    target = robot.data.default_joint_pos.clone()
    target[:, arm_ids] = q.to(target.device)
    _hold(env, target)

    tcp = ee.data.target_pos_w[:, 0, :] - origins
    deg = _approach_deg(robot.data.body_quat_w[:, base_i, :])
    lnames = robot.body_names
    lsel = [i for i, nm in enumerate(lnames) if nm.startswith("l_")]
    minz = (robot.data.body_pos_w[:, lsel, 2] - origins[:, 2:3]).min(dim=1).values
    cup_move = ((obj.data.root_pos_w - origins) - cup0).norm(dim=-1)

    d_tcp = (tcp - tcp[0:1]).norm(dim=-1)
    slack = torch.full((n,), 9.0)
    for j in range(1, 8):
        jn = f"l_aj_{j}"
        lo, hi = limits[jn]
        s = torch.minimum(q[:, j - 1] - lo, hi - q[:, j - 1])
        slack = torch.minimum(slack, s)
    slack = slack.to(deg.device)

    print(f"\n[대조군] 현재 홈  접근각 {deg[0]:.2f}°  TCP({tcp[0,0]:.4f},{tcp[0,1]:.4f},{tcp[0,2]:.4f})"
          f"  최저링크z {minz[0]:.4f}  상면 {P.TABLE_SURFACE_Z:.3f}")

    # ★목표각 ↔ TCP 이동의 교환관계. 어떤 각도가 얼마의 이동을 요구하는지가
    #   "홈으로 각도를 세울 수 있는가"의 답이다.
    base_ok = (minz > P.TABLE_SURFACE_Z + args.clear) & (slack >= args.min_slack) \
        & (cup_move < 0.002)
    base_ok[0] = False
    print(f"\n[교환관계] 목표각별 **최소 TCP 이동** (링크여유·한계여유·무관통 통과 표본만)")
    print(f"  {'목표각':>7} {'표본':>6} {'최소':>8} {'p10':>8} {'중앙':>8}  최소이동 후보(j1..j7)")
    for t in range(90, 113, 2):
        m = base_ok & ((deg - float(t)).abs() <= 1.0)
        if not bool(m.any()):
            print(f"  {t:>6}° {0:>6}        —")
            continue
        dm = d_tcp[m] * 1e3
        bi = torch.nonzero(m).flatten()[dm.argmin()].item()
        js = ",".join(f"{q[bi, k]:.4f}" for k in range(7))
        print(f"  {t:>6}° {int(m.sum()):>6} {dm.min():>7.1f}mm {dm.quantile(0.1):>7.1f}mm "
              f"{dm.quantile(0.5):>7.1f}mm  z={minz[bi]:.4f}  {js}")

    if args.tcp_z_target > 0.0:
        dz = (tcp[:, 2] - args.tcp_z_target).abs()
        print(f"\n[높이 모드] TCP z 를 {args.tcp_z_target:.4f}±{args.tcp_z_tol*1e3:.0f}mm 로 "
              f"맞추면서 접근각을 세운다 (대조군 TCPz {tcp[0,2]:.4f})")
        print(f"  {'목표각':>7} {'표본':>6} {'최소|Δz|':>9}  후보(j1..j7)")
        for t in range(88, 113, 2):
            m = base_ok & ((deg - float(t)).abs() <= 1.0)
            if not bool(m.any()):
                print(f"  {t:>6}° {0:>6}         —")
                continue
            dm = dz[m]
            bi = torch.nonzero(m).flatten()[dm.argmin()].item()
            js = ",".join(f"{q[bi, k]:.4f}" for k in range(7))
            print(f"  {t:>6}° {int(m.sum()):>6} {dm.min()*1e3:>8.1f}mm  z={tcp[bi,2]:.4f} "
                  f"link_z={minz[bi]:.4f}  {js}")
        ok = ((deg - args.target_deg).abs() <= args.deg_tol) & (dz <= args.tcp_z_tol) & base_ok
        d_tcp = dz
    else:
        ok = ((deg - args.target_deg).abs() <= args.deg_tol)
        ok &= (d_tcp <= args.tcp_tol)
    ok &= (minz > P.TABLE_SURFACE_Z + args.clear)
    ok &= (slack >= args.min_slack)
    ok &= (cup_move < 0.002)
    ok[0] = False
    print(f"\n[필터] 각도 {int(((deg-args.target_deg).abs()<=args.deg_tol).sum())} · "
          f"+TCP {int((((deg-args.target_deg).abs()<=args.deg_tol)&(d_tcp<=args.tcp_tol)).sum())} · "
          f"전부통과 {int(ok.sum())} / {n-1}")

    if not bool(ok.any()):
        m = base_ok & ((deg - args.target_deg).abs() <= args.deg_tol)
        if bool(m.any()):
            idx2 = torch.nonzero(m).flatten()
            order2 = idx2[d_tcp[idx2].argsort()][: args.top]
            print(f"\n[TCP 허용 무시] 목표각 {args.target_deg}±{args.deg_tol}° 후보 상위")
            print(f"{'순':>3} {'접근각':>7} {'TCPΔ':>7} {'최저z':>7} {'여유':>6}   관절(j1..j7)")
            for r, i in enumerate(order2.tolist()):
                js = " ".join(f"{q[i, k]:+.4f}" for k in range(7))
                print(f"{r+1:>3} {deg[i]:>6.2f}° {d_tcp[i]*1e3:>6.1f}mm {minz[i]:>7.4f} "
                      f"{slack[i]:>6.3f}   {js}")
        print("\n★후보 없음 — TCP 허용을 늘리거나 목표각을 완화해야 한다.")
        print("  참고: 각도만 만족하는 표본의 TCP 이동 분포 (mm)")
        m = (deg - args.target_deg).abs() <= args.deg_tol
        if bool(m.any()):
            dm = d_tcp[m] * 1e3
            print(f"    최소 {dm.min():.1f} · p10 {dm.quantile(0.1):.1f} · 중앙 {dm.quantile(0.5):.1f}")
        print("HOMELEVEL_DONE")
        env.close()
        return

    idx = torch.nonzero(ok).flatten()
    order = idx[d_tcp[idx].argsort()][: args.top]
    print(f"\n{'순':>3} {'접근각':>7} {'TCPΔ':>7} {'최저z':>7} {'여유':>6}   관절(j1..j7)")
    for r, i in enumerate(order.tolist()):
        js = " ".join(f"{q[i, k]:+.4f}" for k in range(7))
        print(f"{r+1:>3} {deg[i]:>6.2f}° {d_tcp[i]*1e3:>6.1f}mm {minz[i]:>7.4f} "
              f"{slack[i]:>6.3f}   {js}")
    best = order[0].item()
    print("\n★채택 후보 (v2_preset 붙여넣기 형식)")
    print("LEFT_ARM_HOME_LEVEL = {")
    for k in range(7):
        print(f"    \"l_aj_{k+1}\": {q[best, k]:+.4f},")
    print("}")
    print(f"# 접근각 {deg[best]:.2f}° (현재 홈 {deg[0]:.2f}°) · "
          f"TCP 이동 {d_tcp[best]*1e3:.1f}mm · 최저링크z {minz[best]:.4f} · 한계여유 {slack[best]:.3f}")
    print("HOMELEVEL_DONE")
    env.close()


def _reach() -> None:
    limits = _joint_limits()
    vals = [float(v) for v in args.reach.split(",")]
    if len(vals) != 7:
        raise SystemExit("--reach 는 j1..j7 7 개여야 한다")
    n = args.num_reach
    print(f"[reach] 홈 " + " ".join(f"j{i+1}={vals[i]:+.4f}" for i in range(7))
          + f" · 액션 상자 ±{args.reach_span} rad · 표본 {n}")

    env = _build(n)
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
        qq = (vals[j] + off).clamp(lo, hi)
        qq[0] = vals[j]
        target[:, arm_ids[j]] = qq.to(target.device)
    _hold(env, target)

    tcp = ee.data.target_pos_w[:, 0, :] - origins
    deg = _approach_deg(robot.data.body_quat_w[:, base_i, :])
    cup = (obj.data.root_pos_w - origins)[0]
    grasp = cup.clone()
    grasp[2] = grasp[2] + P.CUP_ORIGIN_TO_GRASP_Z
    d = (tcp - grasp).norm(dim=-1)
    lvl = deg <= 100.0

    print(f"\n  컵 ({cup[0]:.3f},{cup[1]:.3f},{cup[2]:.3f}) · 파지점 z {grasp[2]:.4f}")
    print(f"  홈에서: 접근각 {deg[0]:.2f}° · TCP–파지점 {d[0]*1e3:.1f}mm")
    print(f"\n  {'문턱':>8} {'도달':>9} {'+접근각≤100° 동시':>20}")
    for thr in (0.0225, 0.030, 0.050, 0.080):
        hit = d < thr
        print(f"  {thr*1e3:>6.1f}mm {100.0*hit.float().mean():>8.2f}% "
              f"{100.0*(hit & lvl).float().mean():>19.2f}%")
    print(f"    → 최소거리 {d.min()*1e3:.1f}mm · 접근각≤100° 표본 중 최소 "
          f"{(d[lvl].min()*1e3 if bool(lvl.any()) else float('nan')):.1f}mm")
    near = d < 0.08
    if bool(near.any()):
        dz = tcp[near, 2] - cup[2]
        print(f"\n  ★리프트 여유 — 컵 80mm 이내 {int(near.sum())} 표본의 TCPz − 컵z (mm)")
        print(f"    중앙 {float(dz.quantile(0.5))*1e3:+.1f} · p90 {float(dz.quantile(0.9))*1e3:+.1f} "
              f"· 최대 {float(dz.max())*1e3:+.1f}")
        for h in (0.02, 0.04, 0.06):
            print(f"    +{h*1e3:.0f}mm 이상  {100.0*(dz > h).float().mean():6.2f}%")
    print("HOMELEVEL_DONE")
    env.close()


def main() -> None:
    _reach() if args.reach else _search()


main()
simulation_app.close()

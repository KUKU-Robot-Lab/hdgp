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

"""open-grip_l_grasp_sensor zero-action 스모크.

학습을 태우기 전에 **씬과 액션 배선이 물리적으로 멀쩡한지**만 본다. 여기서 이상하면
epoch 를 더 태워도 의미가 없다(pour_v1 이력: 2442 epoch 를 1 분 probe 가 대체했다).

보는 것:
  1. 컵이 리셋 후 제자리에 있는가 — zero action 에서 밀리거나 넘어지면 씬 결함이다.
  2. 액션 0 자세에서 TCP 가 컵에서 얼마나 떨어져 있는가 — lift 레시피는 초기 자세가
     해답 근처여야 학습된다(±0.5 rad 국소 탐색).
  3. 그리퍼 이진 지령이 실제 관절 이동으로 이어지는가 — mimic 제약과 싸우면 여기서 보인다.
  4. 조기 종료가 나는가 — object_dropping 이 초반부터 켜지면 스폰 높이가 틀린 것이다.

★계측 함정(반복해서 밟았다): reset 직후 위치 버퍼는 stale 이다. 반드시 1 스텝 굴린 뒤 읽는다.

실행:
    PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/probes/probe_lift_left_gripper_smoke.py
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--sweep_beyond", action="store_true",
                    help="스폰 박스 바깥까지 훑어 관통 경계를 다시 잰다")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import openarm.tasks  # noqa: F401  (gym 등록)
from isaaclab_tasks.utils import parse_env_cfg
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P

TASK = "open-grip_l_grasp_sensor"


def _quat_tilt_deg(quat: torch.Tensor) -> torch.Tensor:
    """컵 로컬 +z 와 월드 +z 사이 각도. 넘어짐 판정."""
    w, x, y, z = quat.unbind(-1)
    axis_z = torch.stack(
        [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=-1
    )
    return torch.rad2deg(torch.acos(axis_z[..., 2].clamp(-1.0, 1.0)))


def main() -> None:
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    env = gym.make(TASK, cfg=env_cfg).unwrapped

    obs, _ = env.reset()
    action_dim = env.action_manager.total_action_dim
    print(f"[dim] action={action_dim} obs={obs['policy'].shape[-1]}")
    assert action_dim == 8, f"액션 차원이 8 이 아니다: {action_dim}"

    robot = env.scene["robot"]
    obj = env.scene["object"]
    ee = env.scene["ee_frame"]
    origins = env.scene.env_origins
    grip_ids, _ = robot.find_joints(P.GRIPPER_JOINT_NAMES, preserve_order=True)

    zero = torch.zeros(env.num_envs, action_dim, device=env.device)

    arm_ids, _ = robot.find_joints(P.LEFT_ARM_JOINT_NAMES, preserve_order=True)
    home = torch.tensor(
        [P.LEFT_ARM_HOME_JOINT_POS[n] for n in P.LEFT_ARM_JOINT_NAMES], device=env.device
    )

    print("\n=== 1) zero action: 컵이 제자리에 있는가 ===")
    print(
        "  ★팔 처짐과 컵 이동의 **시간 순서**를 본다. 팔이 먼저 움직이고 컵이 따라가면\n"
        "    원인은 팔이 컵을 치는 것이고, 컵이 혼자 움직이면 씬(마찰·스폰 높이) 결함이다."
    )
    print("step |  cup dxy(mm)  cup z(m)  tilt(deg) | TCP-cup(mm) | 팔 처짐 max(deg) | done")
    ref_xy = None
    for i in range(args.steps):
        env.step(zero.clone())
        # ★1 스텝 굴린 뒤에 읽는다(reset 직후 버퍼는 stale).
        cup = obj.data.root_pos_w - origins
        if ref_xy is None:
            ref_xy = cup[:, :2].clone()
        dxy = (cup[:, :2] - ref_xy).norm(dim=-1)
        tilt = _quat_tilt_deg(obj.data.root_quat_w)
        tcp = ee.data.target_pos_w[:, 0, :] - origins
        d_tcp = (tcp - cup).norm(dim=-1)
        droop = torch.rad2deg((robot.data.joint_pos[:, arm_ids] - home).abs().max(dim=-1).values)
        if i < 12 or i % 25 == 0 or i == args.steps - 1:
            print(
                f"{i:4d} | {dxy.mean() * 1e3:8.2f}  {cup[:, 2].mean():8.4f}  "
                f"{tilt.mean():7.2f} | {d_tcp.mean() * 1e3:8.1f} | "
                f"{droop.mean():10.2f} | "
                f"{int(env.termination_manager.dones.sum())}"
            )

    print(f"\n  스폰 z 기대 {P.CUP_SPAWN_Z:.5f} / 실측 {cup[:, 2].mean():.5f}")
    print(f"  리프트 임계 {P.MINIMAL_LIFT_HEIGHT:.3f} (테이블 상면 {P.TABLE_SURFACE_Z:.3f})")

    print("\n=== 1a) 유휴 오른팔이 프리셋 자세를 지키는가 ===")
    print(
        "  ★렌더에서 오른팔이 **바닥에 닿아** 있는 것으로 관찰됐다. 유휴 팔이 처지면\n"
        "    씬이 통째로 신뢰할 수 없고, 나중에 양팔로 갈 때 그대로 문제가 된다."
    )
    # cfg 에 쓴 값이 실제로 적용됐는지부터 본다 — USD drive 가 이기는 경우가 있다.
    for name, act in robot.actuators.items():
        def _s(v):
            try:
                return f"{float(v.min()):.1f}~{float(v.max()):.1f}"
            except Exception:
                return str(v)
        print(f"  [actuator] {name:<16} stiffness {_s(act.stiffness)}  "
              f"damping {_s(act.damping)}  effort {_s(act.effort_limit)}")

    r_arm_ids, r_arm_names = robot.find_joints(
        list(P.RIGHT_ARM_REST_JOINT_POS), preserve_order=True
    )
    r_home = torch.tensor(
        [P.RIGHT_ARM_REST_JOINT_POS[n] for n in r_arm_names], device=env.device
    )
    err = robot.data.joint_pos[:, r_arm_ids] - r_home
    print(f"  오른팔 관절 오차(도): 최대 {torch.rad2deg(err.abs()).max():.2f}, "
          f"평균 {torch.rad2deg(err.abs()).mean():.2f}")
    for j, n in enumerate(r_arm_names):
        print(f"    {n}: 목표 {r_home[j]:+.4f}  실제 {robot.data.joint_pos[:, r_arm_ids[j]].mean():+.4f}  "
              f"오차 {torch.rad2deg(err[:, j]).mean():+7.2f}°")
    right_bodies = [
        (i, n) for i, n in enumerate(robot.body_names)
        if n.startswith("r_hl_") or n.startswith("r_al_")
    ]
    lows = sorted(
        ((robot.data.body_pos_w[:, i, 2] - origins[:, 2]).min().item(), n)
        for i, n in right_bodies
    )
    print("  오른팔 링크 최저 z (바닥 0.0, 테이블 상면 %.3f):" % P.TABLE_SURFACE_Z)
    for z, n in lows[:5]:
        mark = "  ← 바닥/테이블 아래" if z < 0.02 else ""
        print(f"    {n:<26} {z:+.4f}{mark}")

    print("\n=== 1a2) 컵이 모든 env 에 제대로 스폰되는가 ===")
    print("  ★일부 env 에서 컵이 안 보인다는 관찰. 위치·개수를 전 env 에서 직접 센다.")
    cup_local = obj.data.root_pos_w - origins
    print(f"  x: [{cup_local[:, 0].min():.4f}, {cup_local[:, 0].max():.4f}] "
          f"기대 [{P.CUP_SPAWN_X_CENTER - P.CUP_SPAWN_X_RANGE:.4f}, "
          f"{P.CUP_SPAWN_X_CENTER + P.CUP_SPAWN_X_RANGE:.4f}]")
    print(f"  y: [{cup_local[:, 1].min():.4f}, {cup_local[:, 1].max():.4f}] "
          f"기대 [{P.CUP_SPAWN_Y_CENTER - P.CUP_SPAWN_Y_RANGE:.4f}, "
          f"{P.CUP_SPAWN_Y_CENTER + P.CUP_SPAWN_Y_RANGE:.4f}]")
    print(f"  z: [{cup_local[:, 2].min():.5f}, {cup_local[:, 2].max():.5f}] "
          f"기대 {P.CUP_SPAWN_Z:.5f}")
    out = (
        (cup_local[:, 0] < P.CUP_SPAWN_X_CENTER - P.CUP_SPAWN_X_RANGE - 1e-3)
        | (cup_local[:, 0] > P.CUP_SPAWN_X_CENTER + P.CUP_SPAWN_X_RANGE + 1e-3)
        | (cup_local[:, 1] < P.CUP_SPAWN_Y_CENTER - P.CUP_SPAWN_Y_RANGE - 1e-3)
        | (cup_local[:, 1] > P.CUP_SPAWN_Y_CENTER + P.CUP_SPAWN_Y_RANGE + 1e-3)
    )
    print(f"  스폰 박스 밖: {int(out.sum())}/{env.num_envs}")
    print(f"  env 원점 간격: x {origins[:, 0].unique().numel()} 종, "
          f"y {origins[:, 1].unique().numel()} 종")

    print("\n=== 1a3) 쓰러진 컵이 종료되는가 ===")
    print(
        "  ★렌더에서 컵이 테이블에 쓰러진 채 종료되지 않는 것으로 관찰됐다.\n"
        "    쓰러뜨려 놓고 원점 z 를 재서, 현재 종료 임계로 잡히는지 확인한다."
    )
    env.reset()
    root = obj.data.default_root_state.clone()
    root[:, :3] += origins
    # ★테이블에 **얹힌** 상태로 눕혀야 한다. 스폰 높이 그대로 눕히면 컵이 공중에 뜬 채라
    #   z 가 아직 안 내려가 종료가 안 걸린다(그렇게 재서 한 번 오판했다).
    root[:, 2] = origins[:, 2] + P.CUP_TIPPED_ORIGIN_Z
    root[:, 3:] = 0.0
    # y 축 90° 회전 = 옆으로 누움
    root[:, 3] = 0.70710678
    root[:, 5] = 0.70710678
    obj.write_root_pose_to_sim(root[:, :7])
    obj.write_root_velocity_to_sim(root[:, 7:])
    # ★리셋이 일어나기 전에 읽어야 한다. 종료가 걸리면 컵이 곧바로 정립 스폰으로 돌아가
    #   "쓰러진 컵"을 영영 관측하지 못한다(그 자체가 수정이 먹혔다는 증거이긴 하다).
    env.step(zero.clone())
    lying = (obj.data.root_pos_w - origins)[:, 2]
    lying_tilt = _quat_tilt_deg(obj.data.root_quat_w)
    fired = int(env.termination_manager.terminated.sum())
    print(f"  눕힌 직후 컵 원점 z = {lying.mean():.5f} (tilt {lying_tilt.mean():.1f}°)")
    print(f"  낙하/쓰러짐 종료 임계 {P.OBJECT_DROP_HEIGHT:.5f} → "
          f"종료 발화 {fired}/{env.num_envs} env "
          f"{'✓ 잡힌다' if fired == env.num_envs else '← **안 잡힌다** (에피소드가 계속된다)'}")
    print(f"  리프트 임계 {P.MINIMAL_LIFT_HEIGHT:.5f} → "
          f"{'lifted 판정 참(!)' if lying.mean() > P.MINIMAL_LIFT_HEIGHT else 'lifted 판정 거짓'}")

    print("\n=== 1b) 테이블 상면 실측 (낙하 정착) ===")
    print(
        "  자산 해석(extent·BBoxCache)을 두 번 틀렸으므로, 여기서는 **물리로 잰다**.\n"
        "  컵을 상면 바로 위에서 떨어뜨려, 정착한 z 에서 bottom→원점을 빼면 그것이 상면이다."
    )
    # ★낙하 높이는 작아야 한다. shaker 는 가늘고 길어(높이 175 mm, 바닥 반경 29.5 mm)
    #   10 cm 에서 떨어뜨리면 충격으로 넘어져 측정이 통째로 오염된다(실측: tilt 18.7°).
    for drop in (0.005, 0.020):
        env.reset()
        root = obj.data.default_root_state.clone()
        root[:, :3] += origins
        root[:, 2] = origins[:, 2] + P.CUP_SPAWN_Z + drop
        root[:, 3:] = 0.0
        root[:, 3] = 1.0                  # 정립 쿼터니언, 속도 0
        obj.write_root_pose_to_sim(root[:, :7])
        obj.write_root_velocity_to_sim(root[:, 7:])
        for _ in range(400):
            env.step(zero.clone())
        settled = (obj.data.root_pos_w - origins)[:, 2]
        tilt = _quat_tilt_deg(obj.data.root_quat_w)
        upright = tilt < 1.0               # 넘어진 env 는 상면 추정에 쓸 수 없다
        n = int(upright.sum())
        if n == 0:
            print(f"  drop {drop * 1e3:4.0f} mm: 정립 env 0 개 (tilt 평균 {tilt.mean():.2f}°) — 측정 불가")
            continue
        z = settled[upright]
        surface = z.mean().item() - P.CUP_BOTTOM_TO_ORIGIN
        print(
            f"  drop {drop * 1e3:4.0f} mm: 정립 {n}/{env.num_envs}, 정착 z={z.mean():.5f} "
            f"(±{z.std():.5f}) → 상면 {surface:.5f}  "
            f"(프리셋 {P.TABLE_SURFACE_Z:.5f}, 차이 {(surface - P.TABLE_SURFACE_Z) * 1e3:+.1f} mm)"
        )

    print("\n=== 1c) 대조군: 컵을 팔에서 멀리 두면 가만히 있는가 ===")
    print("  스폰 위치에서만 흔들리고 여기서는 멀쩡하면, 원인은 씬이 아니라 **팔/손가락 접촉**이다.")
    for tag, dx in (("스폰 위치", 0.0), ("팔에서 +30cm", 0.30)):
        env.reset()
        root = obj.data.default_root_state.clone()
        root[:, :3] += origins
        root[:, 0] += dx
        root[:, 2] = origins[:, 2] + P.CUP_SPAWN_Z
        root[:, 3:] = 0.0
        root[:, 3] = 1.0
        obj.write_root_pose_to_sim(root[:, :7])
        obj.write_root_velocity_to_sim(root[:, 7:])
        start = None
        for _ in range(150):
            env.step(zero.clone())
            if start is None:
                start = (obj.data.root_pos_w - origins)[:, :2].clone()
        cup = obj.data.root_pos_w - origins
        print(
            f"  {tag:>12}: 이동 {(cup[:, :2] - start).norm(dim=-1).mean() * 1e3:6.2f} mm, "
            f"tilt {_quat_tilt_deg(obj.data.root_quat_w).mean():5.2f}°, "
            f"z {cup[:, 2].mean():.5f}"
        )

    print("\n=== 1d) 홈 자세에서 어느 링크가 컵과 겹치는가 ===")
    print(
        "  컵을 축 반경 44 mm·높이 [-92, +83] mm 의 원기둥으로 근사해, 왼팔 링크 원점에서\n"
        "  표면까지의 부호 있는 거리를 잰다. 음수 = 컵 안쪽(관통)."
    )
    env.reset()
    env.step(zero.clone())               # ★1 스텝 후에 읽는다
    cup = obj.data.root_pos_w - origins
    left_bodies = [
        (i, n) for i, n in enumerate(robot.body_names)
        if n.startswith("l_hl_") or n.startswith("l_al_")
    ]
    rows = []
    for i, name in left_bodies:
        p = robot.data.body_pos_w[:, i, :] - origins
        d = p - cup
        radial = d[:, :2].norm(dim=-1) - 0.044                     # 옆면까지
        below = -0.09209 - d[:, 2]                                 # 바닥면 아래로
        above = d[:, 2] - 0.08291                                  # 윗면 위로
        outside = torch.maximum(torch.maximum(radial, below), above)
        rows.append((outside.mean().item(), name))
    rows.sort()
    for dist, name in rows[:8]:
        mark = "  ← 관통" if dist < 0 else ""
        print(f"  {name:<28} {dist * 1e3:+8.1f} mm{mark}")

    print("\n=== 1e) 스폰 박스 전체가 조용한가 (관통 = 홈 자세의 팔과 충돌) ===")
    print("  링크 **원점**은 전부 컵 바깥인데도 밀린다 = 팔·손가락 **메시**가 닿는 것이다.")
    # env 마다 다른 오프셋을 줘서 스윕 전체를 **한 번의 롤아웃**으로 끝낸다.
    # 조합마다 reset+120 스텝을 반복하면 수 분이 걸린다.
    # 기본은 **실제 스폰 박스 전체**를 훑는다(중심만 보면 랜덤화 하한에서 터진다).
    # 경계를 다시 재고 싶으면 --sweep_beyond 로 박스 바깥까지 넓힌다.
    rx, ry = P.CUP_SPAWN_X_RANGE, P.CUP_SPAWN_Y_RANGE
    if args.sweep_beyond:
        dxs = [-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06]
        dys = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03]
    else:
        dxs = [-rx, -rx / 2, 0.0, rx / 2, rx]
        dys = [-ry, -ry / 2, 0.0, ry / 2, ry]
    combos = [(dx, dy) for dy in dys for dx in dxs][: env.num_envs]
    if len(combos) < len(dxs) * len(dys):
        print(f"  ⚠ num_envs={env.num_envs} 라 {len(combos)}/{len(dxs) * len(dys)} 조합만 본다 "
              f"(--num_envs {len(dxs) * len(dys)} 로 올려 전수 확인 가능)")

    env.reset()
    root = obj.data.default_root_state.clone()
    root[:, :3] += origins
    off = torch.tensor(combos, device=env.device)
    root[: len(combos), 0] += off[:, 0]
    root[: len(combos), 1] += off[:, 1]
    root[:, 2] = origins[:, 2] + P.CUP_SPAWN_Z
    root[:, 3:] = 0.0
    root[:, 3] = 1.0
    obj.write_root_pose_to_sim(root[:, :7])
    obj.write_root_velocity_to_sim(root[:, 7:])
    start = None
    for _ in range(120):
        env.step(zero.clone())
        if start is None:
            start = (obj.data.root_pos_w - origins)[:, :2].clone()
    d = (obj.data.root_pos_w - origins)[:, :2] - start
    tilt = _quat_tilt_deg(obj.data.root_quat_w)
    for k, (dx, dy) in enumerate(combos):
        moved = d[k].norm().item()
        quiet = "  ✓조용" if moved < 2e-3 and tilt[k] < 1.0 else "  ← 관통"
        print(
            f"  {dx * 1e3:6.0f}  {dy * 1e3:6.0f} | {moved * 1e3:8.2f}  "
            f"({d[k, 0] * 1e3:+6.1f},{d[k, 1] * 1e3:+6.1f})  {tilt[k]:8.2f}{quiet}"
        )

    print("\n=== 2) 그리퍼 이진 지령이 관절을 움직이는가 ===")
    for label, val in (("close(-1)", -1.0), ("open(+1)", +1.0)):
        act = zero.clone()
        act[:, 7] = val
        for _ in range(40):
            env.step(act.clone())
        grip = robot.data.joint_pos[:, grip_ids]
        print(f"  {label:>9}: j1={grip[:, 0].mean() * 1e3:6.2f} mm  j2={grip[:, 1].mean() * 1e3:6.2f} mm")
    print(f"  (URDF 스트로크 {P.GRIPPER_CLOSED_POS * 1e3:.0f}~{P.GRIPPER_OPEN_POS * 1e3:.0f} mm, "
          f"gripper_2 는 mimic gearing -1)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

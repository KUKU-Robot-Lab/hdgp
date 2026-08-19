"""[P0-3] 좌팔 그리퍼 파지력 / effort 포화 + zero-action 도달 점검 (Isaac 필요).

두 가지를 한 번에 본다.

1) **zero-action 도달** — action=0 이면 Fabrics 가 홈에서 컵-정준 pregrasp 까지 스스로
   가야 한다. 여기서 안 가면 그건 보상 문제가 아니라 **제어 문제**이고, 학습을 아무리
   돌려도 못 고친다. 이 저장소에서 2442 epoch 를 1분 probe 가 대체한 이력이 있다.

2) **파지력 / effort 포화** — 우측 손에서 effort limit 포화(80.6%)가 파지력 제어를 통째로
   무효화한 이력이 있다(목표를 더 밀어도 힘이 안 오름 = sim2real 무효). 그리퍼 액추에이터
   게인·effort 한계가 같은 함정에 빠지지 않는지 **학습 전에** 확인한다.

판정 게이트:
  · zero-action 종료 시 TCP–pregrasp 거리 < 3 cm
  · 폐쇄 후 두 핑거 모두 컵 접촉
  · 그리퍼 effort 포화율 < 20 %
  · 리프트 후 컵 상승 > 4 cm

사용:
  cd ~/rl_ws/hdgp
  PYTHONUNBUFFERED=1 ../IsaacLab/isaaclab.sh -p scripts/probes/probe_gripper_grip_force.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-grip_l_grasp_sensor-play")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--approach_steps", type=int, default=180, help="zero-action 접근 스텝")
parser.add_argument("--close_steps", type=int, default=120, help="그리퍼 폐쇄 유지 스텝")
parser.add_argument("--lift_steps", type=int, default=120, help="이후 들어올리는 스텝")
parser.add_argument("--freeze_steps", type=int, default=0,
                    help="팔을 홈에 완전히 고정한 채 물리만 돌리는 스텝 (씬 문제 분리용)")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

_p = print


def print(*a, **kw):  # noqa: A001
    _p(*a, **kw, flush=True)


SAT_GATE = 0.20
REACH_GATE = 0.03
LIFT_GATE = 0.04


def main() -> int:
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    obs, _ = env.reset()

    robot = env.robot
    gidx = env.gripper_cmd_index
    effort_limit = float(env.cfg.robot_cfg.actuators["left_gripper"].effort_limit_sim)
    zero = torch.zeros(env.num_envs, env.cfg.action_space, device=env.device)

    print(f"=== 설정 ===")
    print(f"  task={args.task} envs={env.num_envs}  그리퍼 effort_limit_sim={effort_limit} N")
    print(f"  Fabrics cspace={env.fabric.num_joints} (팔 7 DOF 여야 한다)")

    # ── 0. 팔 고정 관측 (선택) ─────────────────────────────────────
    # ★"컵이 움직인다"의 원인을 씬/물리 vs 팔 움직임으로 **가르는** 유일한 방법이다.
    #   env.step 은 항상 Fabrics 를 돌려 팔을 움직이므로, 여기서는 액션 파이프라인을
    #   건너뛰고 물리만 굴린다. 팔이 가만히 있는데도 컵이 움직이면 그건 씬 문제다.
    if args.freeze_steps > 0:
        print("\n=== 0. 팔 고정 (액션 파이프라인 우회, 물리만) ===")
        print(f"  {'step':>5} {'컵밀림[mm]':>11} {'컵기울기[°]':>11} {'컵 z[m]':>9}")
        home_arm = env.q_home_arm.unsqueeze(0).repeat(env.num_envs, 1)
        for k in range(args.freeze_steps):
            robot.set_joint_position_target(home_arm, joint_ids=env.arm_dof_indices)
            robot.set_joint_position_target(
                torch.full((env.num_envs, 1), float(env.gripper_cmd_buf[0]), device=env.device),
                joint_ids=[env.gripper_cmd_index])
            robot.set_joint_position_target(env.idle_rest_pos, joint_ids=env.idle_dof_indices)
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
            if (k + 1) % 20 == 0 or k == 0:
                env._compute_intermediate_values()
                disp = (env.object_pos[:, :2] - env.cup_spawn_pos[:, :2]).norm(dim=-1)
                up = torch.zeros_like(env.object_pos); up[:, 2] = 1.0
                from isaaclab.utils.math import quat_apply as _qa
                tilt = torch.rad2deg(torch.acos(_qa(env.object_rot, up)[:, 2].clamp(-1, 1)))
                print(f"  {k+1:5d} {disp.mean()*1000:11.1f} {tilt.mean():11.1f} "
                      f"{env.object_pos[:, 2].mean():9.4f}")
        print("  → 여기서 이미 움직이면 그리퍼와 무관한 **씬/물리** 문제다")

    # ── 1. zero-action 접근 ────────────────────────────────────────
    # ★수렴 실패 시 원인을 세 갈래로 **분리**해서 봐야 한다:
    #     (a) Fabrics 가 목표에 수렴 안 함  → 거리(err)가 평탄해짐
    #     (b) 팔이 Fabrics 지령을 못 따라감 → track(지령−실측 관절) 이 큼
    #     (c) TCP 계산식이 틀림            → 위 둘 다 정상인데 err 만 큼
    #   셋을 안 나누면 "안 간다"만 보이고 어디를 고칠지 알 수 없다.
    print("\n=== 1. zero-action 접근 (Fabrics 가 홈 → pregrasp) ===")
    print(f"  {'step':>5} {'TCP-pregrasp[mm]':>17} {'관절추종[rad]':>14} "
          f"{'컵밀림[mm]':>11} {'컵기울기[°]':>11} {'래치':>5}")
    for k in range(args.approach_steps):
        obs, _, _, _, _ = env.step(zero)
        if (k + 1) % 20 == 0 or k == 0:
            d = (env.tcp_pos - env.pregrasp_palm_pose_buf[:, :3]).norm(dim=-1)
            track = (
                env.fabric_q[:, :7] - robot.data.joint_pos[:, env.arm_dof_indices]
            ).abs().max(dim=-1).values
            disp = (env.object_pos[:, :2] - env.cup_spawn_pos[:, :2]).norm(dim=-1)
            # 컵 기울기: 컵이 밀린 게 아니라 **넘어지는** 것이면 여기서 먼저 커진다
            up = torch.zeros_like(env.object_pos); up[:, 2] = 1.0
            from isaaclab.utils.math import quat_apply as _qa
            cup_up = _qa(env.object_rot, up)
            tilt = torch.rad2deg(torch.acos(cup_up[:, 2].clamp(-1.0, 1.0)))
            print(f"  {k+1:5d} {d.mean()*1000:17.1f} {track.mean():14.4f} "
                  f"{disp.mean()*1000:11.1f} {tilt.mean():11.1f} "
                  f"{env.lift_latched_buf.float().mean():5.2f}")
    tcp_err = (env.tcp_pos - env.pregrasp_palm_pose_buf[:, :3]).norm(dim=-1)
    print(f"  TCP–pregrasp 거리  평균 {tcp_err.mean()*1000:7.1f} mm  "
          f"최대 {tcp_err.max()*1000:7.1f} mm   (게이트 < {REACH_GATE*1000:.0f} mm)")
    print(f"  목표 palm(첫 env)  {[round(v, 4) for v in env.palm_pose_targets[0].tolist()]}")
    print(f"  실측 TCP(첫 env)   {[round(v, 4) for v in env.tcp_pos[0].tolist()]}")
    print(f"  컵 위치(첫 env)    {[round(v, 4) for v in env.object_pos[0].tolist()]}")

    # ── 자세 오차 ──────────────────────────────────────────────────
    # palm attractor 는 원점 + ±0.25 m 축점 7개로 자세를 잰다. 즉 **1° 자세 오차가
    # 4.4 mm 점 변위**로 환산돼 위치와 한 저울에 올라간다. 위치만 보면 원인을 못 찾는다.
    import math as _m
    q = env.tcp_quat[0]
    w, x, y_, z_ = (float(v) for v in q)
    R = [
        [1 - 2 * (y_ * y_ + z_ * z_), 2 * (x * y_ - w * z_), 2 * (x * z_ + w * y_)],
        [2 * (x * y_ + w * z_), 1 - 2 * (x * x + z_ * z_), 2 * (y_ * z_ - w * x)],
        [2 * (x * z_ - w * y_), 2 * (y_ * z_ + w * x), 1 - 2 * (x * x + y_ * y_)],
    ]
    ey = _m.asin(max(-1.0, min(1.0, -R[2][0])))
    ez = _m.atan2(R[1][0], R[0][0])
    ex = _m.atan2(R[2][1], R[2][2])
    tgt = env.palm_pose_targets[0][3:].tolist()
    print(f"  자세 목표 euler_zyx[°] "
          f"{[round(_m.degrees(v), 1) for v in tgt]}")
    print(f"  자세 실측 euler_zyx[°] "
          f"{[round(_m.degrees(v), 1) for v in (ez, ey, ex)]}")
    print(f"  jaw축(실측) {[round(R[i][1], 3) for i in range(3)]}  "
          f"접근축(실측) {[round(R[i][2], 3) for i in range(3)]}")
    print(f"  ※ jaw z 성분이 0 에서 멀면 수평 파지가 안 되고 있다는 뜻")

    # ── 관절 상태 ──────────────────────────────────────────────────
    qa = robot.data.joint_pos[0, env.arm_dof_indices]
    lo = robot.data.soft_joint_pos_limits[0, env.arm_dof_indices, 0]
    hi = robot.data.soft_joint_pos_limits[0, env.arm_dof_indices, 1]
    margin = torch.minimum(qa - lo, hi - qa)
    print(f"  관절(첫 env)  {[round(float(v), 3) for v in qa]}")
    print(f"  한계여유      {[round(float(v), 3) for v in margin]}  "
          f"최소 {float(margin.min()):.3f} rad")
    print(f"  TCP–컵 거리        평균 "
          f"{(env.object_pos - env.tcp_pos).norm(dim=-1).mean()*1000:7.1f} mm")
    print(f"  컵 밀림            평균 "
          f"{(env.object_pos[:, :2] - env.cup_spawn_pos[:, :2]).norm(dim=-1).mean()*1000:7.1f} mm")

    # ── 2. 그리퍼 폐쇄 ─────────────────────────────────────────────
    close = zero.clone()
    close[:, -1] = 1.0                      # 그리퍼 action +1 = 완전 폐쇄
    sat_hits, samples = 0, 0
    force_sum = torch.zeros(env.num_envs, device=env.device)
    for _ in range(args.close_steps):
        obs, _, _, _, _ = env.step(close)
        tau = robot.data.applied_torque[:, gidx].abs()
        sat_hits += int((tau >= effort_limit * 0.99).sum().item())
        samples += env.num_envs
        force_sum += env.finger_force_buf.norm(dim=-1).sum(dim=-1)

    sat_ratio = sat_hits / max(1, samples)
    contact = env.contact_binary_buf
    gripper_actual = robot.data.joint_pos[:, gidx]
    gripper_err = (env.gripper_cmd_buf - gripper_actual).clamp(min=0.0)
    print("\n=== 2. 그리퍼 폐쇄 ===")
    print(f"  두 핑거 접촉 비율   {contact.all(dim=-1).float().mean():.3f}")
    print(f"  핑거별 접촉 비율    {contact.float().mean(dim=0).tolist()}")
    print(f"  접촉력 합 평균      {(force_sum / args.close_steps).mean():7.2f} N")
    print(f"  그리퍼 관절 오차    {gripper_err.mean()*1000:7.2f} mm "
          f"(지령 {env.gripper_cmd_buf.mean():.4f} / 실측 {gripper_actual.mean():.4f})")
    print(f"  effort 포화율       {sat_ratio:.3f}   (게이트 < {SAT_GATE:.2f})")

    # ── 3. 리프트 ──────────────────────────────────────────────────
    h0 = env.object_pos[:, 2].clone()
    for _ in range(args.lift_steps):
        obs, _, _, _, _ = env.step(close)
    rise = env.object_pos[:, 2] - h0
    print("\n=== 3. 리프트 ===")
    print(f"  래치 비율          {env.lift_latched_buf.float().mean():.3f}")
    print(f"  컵 상승            평균 {rise.mean()*1000:7.1f} mm  "
          f"최대 {rise.max()*1000:7.1f} mm   (게이트 > {LIFT_GATE*1000:.0f} mm)")
    print(f"  유지 접촉          {env.contact_binary_buf.all(dim=-1).float().mean():.3f}")

    # ── 판정 ───────────────────────────────────────────────────────
    checks = {
        "zero-action 도달": bool((tcp_err.mean() < REACH_GATE).item()),
        "양 핑거 접촉": bool((contact.all(dim=-1).float().mean() > 0.5).item()),
        "effort 비포화": sat_ratio < SAT_GATE,
        "리프트": bool((rise.mean() > LIFT_GATE).item()),
    }
    print("\n=== 판정 ===")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    env.close()
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    code = main()
    app.close()
    raise SystemExit(code)

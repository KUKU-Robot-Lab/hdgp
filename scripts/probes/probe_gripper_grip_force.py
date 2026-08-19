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

    # ── 1. zero-action 접근 ────────────────────────────────────────
    for _ in range(args.approach_steps):
        obs, _, _, _, _ = env.step(zero)
    tcp_err = (env.tcp_pos - env.pregrasp_palm_pose_buf[:, :3]).norm(dim=-1)
    print("\n=== 1. zero-action 접근 (Fabrics 가 홈 → pregrasp) ===")
    print(f"  TCP–pregrasp 거리  평균 {tcp_err.mean()*1000:7.1f} mm  "
          f"최대 {tcp_err.max()*1000:7.1f} mm   (게이트 < {REACH_GATE*1000:.0f} mm)")
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

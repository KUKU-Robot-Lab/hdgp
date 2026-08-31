# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""이송이 **물리적으로 가능한가** — 정책을 빼고 제어기만으로 컵을 goal 까지 몬다.

질문(사용자 08.27): "현재 어느 정도 제자리에 멈추는 것까지는 되는데 이송을 못한다.
  학습을 멈추고 실험으로 원인을 파악하자."

방법:
  ① 학습된 정책으로 `--switch_at` 스텝까지 돌려 **파지·리프트를 완성**시킨다.
  ② 그 시점의 (palm 지령 − 컵 위치) 오프셋을 고정한다 = 파지 기하.
  ③ 이후 위치 액션 3성분을 **스크립트가 덮어써** palm 지령을 `goal + 오프셋` 으로 몬다.
     회전·그리퍼 액션은 정책 값을 그대로 둔다(파지 자세를 흔들지 않기 위해).
     지령 변화율 리미터가 켜져 있으면 자동으로 램프가 된다.
  ④ 컵–목표 거리·관절 추종오차·컵 기울기를 재고, 마지막 구간의 도달률을 낸다.

판정:
  · 도달한다  → 제어기는 할 수 있다 = **정책·보상 문제**
  · 못 간다   → **물리·액추에이터 문제**(그때 어느 관절이 포화하는지가 같이 나온다)

액추에이터 조건은 인자로 바꾼다(재빌드 불필요):
  --wrist_effort 50 --arm_kp 400 --arm_kd 80
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-grip_l_grasp_sensor_fab")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--switch_at", type=int, default=120,
                    help="이 스텝부터 위치 지령을 스크립트가 가져간다")
parser.add_argument("--wrist_effort", type=float, default=-1.0,
                    help="j5~7 effort_limit_sim 오버라이드 (음수=프리셋 값 유지)")
parser.add_argument("--arm_effort", type=float, default=-1.0,
                    help="j1~4 도 함께 올릴 때")
parser.add_argument("--arm_kp", type=float, default=-1.0)
parser.add_argument("--arm_kd", type=float, default=-1.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym                              # noqa: E402
import torch                                         # noqa: E402
import openarm.tasks                                 # noqa: E402,F401
from isaaclab.utils.math import combine_frame_transforms   # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper   # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P    # noqa: E402
from rl_games.common import env_configurations, vecenv   # noqa: E402
from rl_games.torch_runner import Runner             # noqa: E402


def _quat_tilt_deg(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    cos = 1.0 - 2.0 * (x * x + y * y)
    return torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))


def main() -> None:
    cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    # ⚠ episode_length_s 는 건드리지 않는다. 1e9 로 늘렸더니 정책이 리프트를 유지하지
    #   못한 채로 계속 굴러가 컵이 스폰 높이(0.296)에 머물렀다. 대신 --steps 를
    #   에피소드(250 스텝) 안으로 잡아 리셋 오염을 피한다.

    # ── 액추에이터 오버라이드 ────────────────────────────────────────
    act_cfg = cfg.scene.robot.actuators["left_arm"]
    if args.wrist_effort > 0 or args.arm_effort > 0:
        eff = dict(P.ARM_EFFORT_LIMIT)
        if args.arm_effort > 0:
            eff["l_aj_[1-2]"] = args.arm_effort
            eff["l_aj_[3-4]"] = args.arm_effort
        if args.wrist_effort > 0:
            eff["l_aj_[5-7]"] = args.wrist_effort
        act_cfg.effort_limit_sim = eff
    if args.arm_kp > 0:
        act_cfg.stiffness = args.arm_kp
    if args.arm_kd > 0:
        act_cfg.damping = args.arm_kd
    print(f"[액추에이터] effort={act_cfg.effort_limit_sim}  kp={act_cfg.stiffness}"
          f"  kd={act_cfg.damping}")

    env = gym.make(args.task, cfg=cfg)
    raw = env.unwrapped
    agent_cfg = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")
    wrapped = RlGamesVecEnvWrapper(env, "cuda:0",
                                   agent_cfg["params"]["config"].get("clip_observations", 100.0),
                                   agent_cfg["params"]["config"].get("clip_actions", 100.0))
    vecenv.register("IsaacRlgWrapper", lambda cn, nw, **kw: RlGamesGpuEnv(cn, nw, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: wrapped})
    agent_cfg["params"]["config"]["env_info"] = wrapped.get_env_info()
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(args.checkpoint)
    agent.reset()

    # RlGamesVecEnvWrapper 는 {'obs': tensor} 를 준다. player 는 텐서를 기대한다.
    def _tensor(o):
        return o["obs"] if isinstance(o, dict) else o

    obs = _tensor(wrapped.reset())
    # ★play.py 와 같은 준비 절차. 없으면 player 가 배치를 1 로 보고 flatten 한다.
    agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()
    at = raw.action_manager.get_term("arm_action")
    robot = raw.scene["robot"]
    obj = raw.scene["object"]
    origins = raw.scene.env_origins
    aid = at._arm_joint_ids
    center, half = at._box_center, at._box_half

    offset = None
    dist_hist, err_hist, tilt_hist = [], [], []

    for step in range(args.steps):
        with torch.inference_mode():
            act_raw = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
        # ★★inference_mode 안에서 만든 텐서를 in-place 로 고치면 step 에 반영되지 않는다
        #   (실측: 액션은 +0.89 인데 지령이 박스 중심에 머물렀다). 밖에서 clone 한다.
        act = act_raw.clone()
        cup = (obj.data.root_pos_w - origins).clone()
        cmd_now = at._palm_pose_target[:, :3].clone()
        if step == args.switch_at:
            offset = (cmd_now - cup).clone()      # 파지 기하 + 지령 선행량
            print(f"[전환] step {step}  오프셋(palm−컵) 평균 "
                  f"{[round(v, 4) for v in offset.mean(0).tolist()]}"
                  f"  컵 z 평균 {float(cup[:, 2].mean()):.3f}")
        if offset is not None:
            g = raw.command_manager.get_command("object_pose")
            des_w, _ = combine_frame_transforms(
                robot.data.root_pos_w, robot.data.root_quat_w, g[:, :3])
            des_palm = (des_w - origins) + offset
            act[:, :3] = ((des_palm - center) / half).clamp(-1.0, 1.0)

        obs, _, _, _ = wrapped.step(act)
        obs = _tensor(obs)

        if offset is not None and (step - args.switch_at) in (0, 5, 20, 60, 100):
                g = raw.command_manager.get_command("object_pose")
                des_w2, _ = combine_frame_transforms(
                    robot.data.root_pos_w, robot.data.root_quat_w, g[:, :3])
                want = (des_w2 - origins + offset).mean(0)
                got = at._palm_pose_target[:, :3].mean(0)
                cupm = (obj.data.root_pos_w - origins).mean(0)
                a3 = act[:, :3].mean(0)
                print(f"  [+{step-args.switch_at:3d}] 원하는지령 "
                      f"[{want[0]:.3f},{want[1]:.3f},{want[2]:.3f}]"
                      f"  실제지령 [{got[0]:.3f},{got[1]:.3f},{got[2]:.3f}]"
                      f"  액션 [{a3[0]:+.2f},{a3[1]:+.2f},{a3[2]:+.2f}]"
                      f"  컵 [{cupm[0]:.3f},{cupm[1]:.3f},{cupm[2]:.3f}]")
        if offset is not None:
            cup = obj.data.root_pos_w - origins
            g = raw.command_manager.get_command("object_pose")
            des_w, _ = combine_frame_transforms(
                robot.data.root_pos_w, robot.data.root_quat_w, g[:, :3])
            dist_hist.append((des_w - obj.data.root_pos_w).norm(dim=-1).detach().cpu())
            err_hist.append(
                (robot.data.joint_pos_target[:, aid]
                 - robot.data.joint_pos[:, aid]).abs().detach().cpu())
            tilt_hist.append(_quat_tilt_deg(obj.data.root_quat_w).detach().cpu())

    if not dist_hist:
        print("전환이 일어나지 않았다 (--switch_at 이 --steps 보다 큰가)")
        env.close(); app.close(); return

    D = torch.stack(dist_hist)          # (T, E)
    E = torch.stack(err_hist)           # (T, E, 7)
    T = torch.stack(tilt_hist)
    n = D.shape[0]
    tail = slice(max(0, n - 30), n)

    print("\n" + "=" * 80)
    print("스크립트 지령으로 컵을 goal 까지 몰았을 때")
    print("=" * 80)
    print(f"  전환 후 {n} 스텝 · env {D.shape[1]}")
    print(f"  {'구간':<10}{'컵–목표':>12}{'컵 기울기':>12}")
    for lab, a, b in (("전환 직후", 0, n // 4), ("중반", n // 4, n // 2),
                      ("후반", n // 2, 3 * n // 4), ("종반", 3 * n // 4, n)):
        print(f"  {lab:<10}{float(D[a:b].mean())*1e3:9.0f} mm{float(T[a:b].mean()):10.1f}°")
    fin = D[tail].mean(dim=0)
    print(f"\n  ★최종(마지막 30스텝) 컵–목표: 중앙값 {float(fin.median())*1e3:.0f} mm"
          f" · 최소 {float(fin.min())*1e3:.0f} mm")
    print(f"  ★30 mm 이내 도달 env {float((fin < 0.03).float().mean()):.1%}"
          f" · 50 mm 이내 {float((fin < 0.05).float().mean()):.1%}")
    print(f"  컵을 놓친 env(기울기>60°) {float((T[tail].mean(dim=0) > 60).float().mean()):.1%}")

    lim = [0.100, 0.100, 0.0675, 0.0675, 0.0175, 0.0175, 0.0175]
    print(f"\n  {'관절':<6}{'추종오차(mrad)':>16}{'포화기준':>10}{'대비':>8}")
    for j in range(7):
        v = float(E[tail][:, :, j].mean())
        print(f"  j{j+1:<5}{v*1e3:16.1f}{lim[j]*1e3:10.1f}{v/lim[j]:7.0%}")
    print("  → 도달했으면 제어기는 가능한 것이다(정책·보상 문제).")
    print("     못 갔으면 여기서 포화한 관절이 물리 병목이다.")

    env.close()
    app.close()


main()

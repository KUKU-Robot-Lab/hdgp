"""Fabrics 추종 격리 probe — 제어기만 본다.

스모크 probe 에서 "팔 관절속도 20 rad/s 포화인데 palm 은 2초에 60mm" 라는 모순이
나왔다. 목표가 도달 불가인지, 제어기 배선이 틀렸는지, PD 가 싸우는지 분리한다.

3단계로 목표를 준다:
  A) 측정된 **현재 palm pose** 그대로 (완전한 no-op — 여기서 안 되면 배선 문제)
  B) 현재 pose 에서 z +5cm  (작고 확실히 도달 가능)
  C) 박스 중심            (스모크 probe 가 쓴 목표)
각 단계마다 palm 오차 / 관절속도 / fabric_q vs 실제 q 괴리를 찍는다.
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=90)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym          # noqa: E402
import torch                     # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg       # noqa: E402

import openarm.tasks             # noqa: E402,F401

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()
zero = torch.zeros(args.num_envs, env.cfg.action_space, device=env.device)
env.step(zero)


def palm_pose():
    pos = env.robot.data.body_pos_w[:, env.palm_idx] - env.scene.env_origins
    quat = env.robot.data.body_quat_w[:, env.palm_idx]
    r, p_, y = euler_xyz_from_quat(quat)
    return pos, torch.stack([y, p_, r], dim=1)      # euler_zyx 순서


# ---- 관절 인덱스/한계 정합 검사 -------------------------------------------
_, arm_names = env.robot.find_joints(env.profile.arm_joint_regex)
jl = env.robot.data.soft_joint_pos_limits[0]
print("\n=== 팔 관절 인덱스 · 한계 (find_joints 반환 순서 그대로) ===")
for k, (idx, nm) in enumerate(zip(env._arm_t.tolist(), arm_names)):
    lo, hi = jl[idx].tolist()
    print(f"  fabric슬롯 {k} ← articulation idx {idx:2d}  {nm:10s}  [{lo:+7.3f}, {hi:+7.3f}]")
print("  ★fabric 슬롯 순서(j1..jN)와 이름 순서가 어긋나면 지령이 다른 관절로 간다.")

# actuator 가 이 관절들을 실제로 덮는지 (커버리지 숫자만으론 부족 — 어느 그룹인지 본다)
print("\n=== actuator 그룹별 관절 ===")
for name, act in env.robot.actuators.items():
    idxs = act.joint_indices
    idxs = idxs.tolist() if hasattr(idxs, "tolist") else list(idxs)
    hit = [n for i, n in zip(idxs, act.joint_names)] if hasattr(act, "joint_names") else idxs
    print(f"  {name:22s} n={len(idxs):2d}  {str(hit)[:90]}")


def _limit_violations():
    q = env.robot.data.joint_pos[0]
    bad = []
    for idx, nm in zip(env._arm_t.tolist(), arm_names):
        lo, hi = jl[idx].tolist()
        v = q[idx].item()
        if v < lo - 1e-3 or v > hi + 1e-3:
            bad.append(f"{nm}={v:+.2f}∉[{lo:+.2f},{hi:+.2f}]")
    return bad


pos0, eul0 = palm_pose()
print(f"\n측정된 홈 palm : pos {pos0[0].tolist()}")
print(f"                euler_zyx {[f'{v:.3f}' for v in eul0[0].tolist()]} rad "
      f"= {[f'{torch.rad2deg(v):.1f}' for v in eul0[0]]} deg")
print(f"박스 lo        : {[f'{v:.3f}' for v in env.palm_lo[0].tolist()]}")
print(f"박스 hi        : {[f'{v:.3f}' for v in env.palm_hi[0].tolist()]}")
inside = ((pos0[0] >= env.palm_lo[0, :3]) & (pos0[0] <= env.palm_hi[0, :3])).all()
inside_r = ((eul0[0] >= env.palm_lo[0, 3:]) & (eul0[0] <= env.palm_hi[0, 3:])).all()
print(f"★홈이 박스 안인가? 위치 {bool(inside)} / 자세 {bool(inside_r)}")

targets = {
    "A no-op (현재 pose)": torch.cat([pos0, eul0], dim=1).clone(),
    "B z+5cm":            torch.cat([pos0 + torch.tensor([0, 0, 0.05], device=env.device),
                                     eul0], dim=1).clone(),
    "C 박스 중심":         (0.5 * (env.palm_lo + env.palm_hi)).repeat(args.num_envs, 1).clone(),
}

for name, tgt in targets.items():
    env.reset()
    env.step(zero)
    print(f"\n{'=' * 62}\n{name}\n{'=' * 62}")
    print(f"  목표 pos {[f'{v:.3f}' for v in tgt[0, :3].tolist()]}  "
          f"eul {[f'{v:.3f}' for v in tgt[0, 3:].tolist()]}")
    for i in range(args.steps):
        # 액션 경로를 우회해 fabric 에 목표를 직접 준다(액션 매핑을 배제)
        env.palm_targets = tgt
        env.fabric.set_features(
            env._fabric_hand_cmd, env.palm_targets, "euler_zyx",
            env.fabric_q.detach(), env.fabric_qd.detach(),
            env._world_ids, env._world_indicator, env._fabric_damping,
        )
        env._step_fabric()
        for _ in range(env.cfg.decimation):
            env._apply_action()
            # ★set_joint_position_target 는 버퍼에만 쓴다. write_data_to_sim 을 빠뜨리면
            #   목표가 PhysX 에 도달하지 않는다(이 probe 의 1차 실행이 그래서 무효였다).
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
        if i in (0, 4, 19, args.steps - 1):
            pos, eul = palm_pose()
            perr = (tgt[:, :3] - pos).norm(dim=-1).mean().item()
            # ★자세 오차도 잰다 — 위치만 보면 손목 롤 발산을 놓친다.
            derr = torch.atan2(torch.sin(tgt[:, 3:] - eul),
                               torch.cos(tgt[:, 3:] - eul)).abs()
            qd = env.robot.data.joint_vel[:, env._arm_t].abs().max().item()
            q_real = env.robot.data.joint_pos[:, env._arm_t]
            q_fab = env.fabric_q[:, : env.profile.num_arm_joints]
            gap = (q_real - q_fab).abs().mean(dim=0)          # 관절별
            print(f"  step {i:3d}: pos오차 {perr:.4f} m | 자세오차(deg) "
                  f"{[f'{torch.rad2deg(v).item():.1f}' for v in derr.mean(dim=0)]} | "
                  f"qd_max {qd:6.2f}")
            print(f"            |q_real-q_fab| 관절별(rad) "
                  f"{[f'{v.item():.2f}' for v in gap]}")
            print(f"            q_fab  {[f'{v:.2f}' for v in q_fab[0].tolist()]}")
            print(f"            q_real {[f'{v:.2f}' for v in q_real[0].tolist()]}")
            _bad = _limit_violations()
            if _bad:
                print(f"            ★관절한계 위반: {_bad}")

env.close()
app.close()

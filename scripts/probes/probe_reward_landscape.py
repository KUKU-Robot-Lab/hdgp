#!/usr/bin/env python3
"""현재 정책이 있는 자리 → 파지 자세로 가는 **직선 위의 보상 지형**을 잰다.

왜
--
t33 이 936 epoch 동안 턱-컵 0.18~0.21 m 에서 정체했다. 그런데 실측하면:
  · 낙하는 **페널티가 아니라 종료**다(음의 보상 없음)
  · 페널티 총합이 양의 보상의 **1.2%**(−0.00056 vs +0.046), `palm_cmd_rate` 는 0
  · 접근 성공 시 에피소드 보상이 ~6.0, 가만히 있으면 0.41 (**15 배**)
즉 접근을 막는 **벌이 없고**, 산수로는 접근이 압도적으로 이득이다. 그런데 안 간다.

남는 설명은 둘뿐이다 — ① 보상 지형이 실제로는 안 오른다 ② 지형은 오르는데 탐색이 못 찾는다.
이 프로브가 ①을 직접 배제하거나 확정한다. 정책 없이 **지령을 직접 찍어** 잰다.

방법
----
정책이 실제로 머무는 지령(t33 실측 (0.461, 0.329, 0.450))에서 파지 지령까지 직선으로
N 등분해 각 지점을 env 에 하나씩 배정하고, **지령 고정 후 정착**시켜 보상 항을 읽는다.
⚠ 리셋 오염 차단(episode_length_s 무한 · 종료 제거 · ADR 제거) · 라운드마다 컵 재배치.

사용:
  TERM=xterm PYTHONUNBUFFERED=1 PYTHONPATH=<hdgp>/source/openarm \\
    ~/rl_ws/IsaacLab/isaaclab.sh -p scripts/probes/probe_reward_landscape.py --num_envs 64
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--settle", type=int, default=60)
parser.add_argument("--from_cmd", type=float, nargs=3, default=[0.461, 0.329, 0.450],
                    help="정책이 실제로 머무는 지령 (t33 실측)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import openarm.gripper.left.grasp_sensor  # noqa: F401,E402
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P  # noqa: E402
from openarm.gripper.left.grasp_sensor import grasp_left_rewards as R  # noqa: E402

TASK = "open-grip_l_grasp_sensor_fab"


def main() -> None:
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.episode_length_s = 1.0e9
    for t in ("time_out", "object_dropping", "object_out_of_workspace"):
        setattr(cfg.terminations, t, None)
    cfg.curriculum.adr = None
    cfg.events.reset_object_position.params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}
    env = gym.make(TASK, cfg=cfg).unwrapped
    env.reset()
    dev, n = env.device, env.num_envs
    act = env.action_manager.get_term("arm_action")
    obj = env.scene["object"]
    spawn = obj.data.default_root_state.clone()
    spawn[:, :3] += env.scene.env_origins

    start = torch.tensor(args.from_cmd, device=dev)
    # 파지 지령 ≈ 컵 스폰 xy + 파지 대역 높이 (실측: 지령 ≈ 턱 위치)
    goal = torch.tensor([P.CUP_SPAWN_X_CENTER, P.CUP_SPAWN_Y_CENTER,
                         P.GRASP_TARGET_Z], device=dev)
    t = torch.linspace(0.0, 1.0, n, device=dev).unsqueeze(-1)
    pts = start * (1 - t) + goal * t
    a = torch.zeros(n, env.action_manager.total_action_dim, device=dev)
    a[:, :3] = ((pts - act._box_center) / act._box_half).clamp(-1.0, 1.0)
    a[:, 6:] = -1.0

    obj.write_root_pose_to_sim(spawn[:, :7])
    obj.write_root_velocity_to_sim(torch.zeros_like(spawn[:, 7:]))
    for _ in range(args.settle):
        env.step(a)

    rm = env.reward_manager
    names = rm.active_terms
    vals = {}
    for i, nm in enumerate(names):
        cfg_ = rm.get_term_cfg(nm)
        try:
            v = cfg_.func(env, **cfg_.params)
        except Exception:
            continue
        if torch.is_tensor(v) and v.shape[:1] == (n,):
            vals[nm] = (v.float() * float(cfg_.weight)).detach()

    jaw = SceneEntityCfg("robot", body_names=list(P.GRIPPER_FINGER_BODIES))
    jaw.resolve(env.scene)
    dcup = R.diag_jaw_cup_dist(env, P.JAW_PAD_OFFSET, jaw)
    clamped = (a[:, :3].abs() >= 0.999).any(dim=-1)

    print(f"\n시작 지령 {args.from_cmd} → 파지 지령 "
          f"[{goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f}]\n")
    show = [k for k in ("reaching_object", "cup_between_jaws",
                        "grip_closure_when_enclosed", "lifting_object") if k in vals]
    print(f"{'t':>5}{'지령 z':>9}{'턱-컵(mm)':>11}" +
          "".join(f"{k[:14]:>15}" for k in show) + f"{'합':>9}  clamp")
    total = sum(vals.values())
    for i in range(0, n, max(1, n // 16)):
        print(f"{float(t[i]):>5.2f}{float(pts[i,2]):>9.3f}{float(dcup[i])*1000:>11.1f}" +
              "".join(f"{float(vals[k][i]):>15.4f}" for k in show) +
              f"{float(total[i]):>9.4f}" + ("   ★clamp" if bool(clamped[i]) else ""))
    j = int(torch.argmax(total))
    print(f"\n최대 합 t={float(t[j]):.2f} · 턱-컵 {float(dcup[j])*1000:.1f} mm · {float(total[j]):.4f}")
    print(f"시작(t=0) {float(total[0]):.4f} → 끝(t=1) {float(total[-1]):.4f}   "
          f"비 {float(total[-1])/max(float(total[0]),1e-9):.1f}배")
    if int(clamped.sum()):
        print(f"⚠ 액션 clamp 에 걸린 지점 {int(clamped.sum())}/{n} — 박스 밖이라 그 지령은 못 낸다")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

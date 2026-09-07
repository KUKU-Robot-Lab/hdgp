"""액션 → palm 지령 → 실제 자세 **매핑이 의미대로인가**.

왜 필요한가: 액션 규약은 "절대 palm 6D"(a=0 이면 항상 같은 pose)다. 그런데 그 지령이
실제로 그 자리로 가지 않으면 상위 계측·보상 해석이 통째로 흔들린다. 축별 부호·배율·
오프셋과 회전 규약(xyzw)을 **직접 지령해 재본다**.

★에피소드 타임아웃을 끈다 — 250 스텝마다 리셋되면 수렴 전에 값이 튄다(3번 당했다).

실행: ./isaaclab.sh -p scripts/probes/probe_fab_action_mapping.py
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--settle", type=int, default=200, help="지령 후 수렴 대기 스텝")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: E402,F401
from isaaclab.utils.math import matrix_from_quat  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P  # noqa: E402

TASK = "open-grip_l_grasp_sensor_fab"
cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
env = gym.make(TASK, cfg=cfg).unwrapped
env.cfg.episode_length_s = 100000.0     # ★리셋 오염 차단
env.reset()

robot = env.scene["robot"]
ee = env.scene["ee_frame"]
org = env.scene.env_origins
base_i = robot.body_names.index(P.GRIPPER_BASE_BODY)
term = env.action_manager.get_term("arm_action")
obj = env.scene["object"]


def park_cup() -> None:
    """컵을 공중 고정 — 접촉이 매핑 측정을 오염시키지 않게."""
    st = obj.data.default_root_state.clone()
    st[:, :3] = org + torch.tensor([0.30, -0.30, 0.50], device=env.device)
    obj.write_root_pose_to_sim(st[:, :7])
    obj.write_root_velocity_to_sim(torch.zeros_like(st[:, 7:]))


def hold(action: torch.Tensor, steps: int):
    a = action.unsqueeze(0).repeat(env.num_envs, 1)
    for _ in range(steps):
        env.step(a)
        park_cup()
    tcp = (ee.data.target_pos_w[:, 0, :] - org).mean(dim=0)
    base = (robot.data.body_pos_w[:, base_i, :] - org).mean(dim=0)
    cmd = term.processed_actions[0, :3].clone()
    q_cmd = term.processed_actions[0, 3:7].clone()          # xyzw
    q_act = robot.data.body_quat_w[:, base_i, :].mean(dim=0)  # wxyz
    return cmd, tcp, base, q_cmd, q_act


A = env.action_manager.total_action_dim
print(f"\n액션 차원 {A} · PALM_BOX x{P.PALM_BOX_X} y{P.PALM_BOX_Y} z{P.PALM_BOX_Z}")
print(f"박스 중심 ({0.5*(P.PALM_BOX_X[0]+P.PALM_BOX_X[1]):.3f}, "
      f"{0.5*(P.PALM_BOX_Y[0]+P.PALM_BOX_Y[1]):.3f}, {0.5*(P.PALM_BOX_Z[0]+P.PALM_BOX_Z[1]):.3f})")

print("\n=== 1) 위치 매핑: a=0 (박스 중심) ===")
z = torch.zeros(A, device=env.device)
cmd0, tcp0, base0, qc0, qa0 = hold(z, args.settle)
print(f"  지령      ({cmd0[0]:.4f}, {cmd0[1]:.4f}, {cmd0[2]:.4f})")
print(f"  TCP       ({tcp0[0]:.4f}, {tcp0[1]:.4f}, {tcp0[2]:.4f})  오차 {(tcp0-cmd0).norm()*1e3:6.1f} mm")
print(f"  base      ({base0[0]:.4f}, {base0[1]:.4f}, {base0[2]:.4f})  오차 {(base0-cmd0).norm()*1e3:6.1f} mm")
print("  → 어느 쪽 오차가 작은지가 **fabric 이 실제로 어느 프레임을 겨냥하는가**를 말한다.")

print("\n=== 2) 축별 부호·배율 (a=±0.5, 기대 변위 = 0.5 × 박스 반폭) ===")
half = [0.5*(P.PALM_BOX_X[1]-P.PALM_BOX_X[0]), 0.5*(P.PALM_BOX_Y[1]-P.PALM_BOX_Y[0]),
        0.5*(P.PALM_BOX_Z[1]-P.PALM_BOX_Z[0])]
print(f"  {'축':<4}{'기대 Δ':>10}{'지령 Δ':>10}{'TCP Δ':>10}{'배율':>8}{'부호':>6}")
for i, ax in enumerate("xyz"):
    a = torch.zeros(A, device=env.device); a[i] = 0.5
    c1, t1, b1, _, _ = hold(a, args.settle)
    a[i] = -0.5
    c2, t2, b2, _, _ = hold(a, args.settle)
    exp = 2 * 0.5 * half[i]
    dcmd = float((c1 - c2)[i]); dtcp = float((t1 - t2)[i])
    print(f"  {ax:<4}{exp*1e3:9.1f}mm{dcmd*1e3:9.1f}mm{dtcp*1e3:9.1f}mm"
          f"{dtcp/exp if exp else 0:8.3f}{'+' if dtcp*exp > 0 else '−':>6}")
print("  → 배율 1.0·부호 + 가 정상. 배율이 낮으면 지령을 다 못 따라간 것이다.")

print("\n=== 3) 회전 매핑 (a_rot=0 → 기준 자세와 일치하는가) ===")
def q_angle(q_wxyz_a, q_wxyz_b):
    d = (q_wxyz_a * q_wxyz_b).sum().abs().clamp(max=1.0)
    return float(torch.rad2deg(2 * torch.acos(d)))
ref = torch.tensor(P.PALM_REF_QUAT_WXYZ, device=env.device)
print(f"  지령 quat(xyzw)  ({qc0[0]:+.4f}, {qc0[1]:+.4f}, {qc0[2]:+.4f}, {qc0[3]:+.4f})")
qc0_wxyz = torch.tensor([qc0[3], qc0[0], qc0[1], qc0[2]], device=env.device)
print(f"  ↳ wxyz 변환      ({qc0_wxyz[0]:+.4f}, {qc0_wxyz[1]:+.4f}, {qc0_wxyz[2]:+.4f}, {qc0_wxyz[3]:+.4f})")
print(f"  기준 REF(wxyz)   ({ref[0]:+.4f}, {ref[1]:+.4f}, {ref[2]:+.4f}, {ref[3]:+.4f})")
print(f"  지령 vs 기준     {q_angle(qc0_wxyz, ref):6.2f}°   (0 이어야 절대 규약이 맞다)")
print(f"  실제 palm vs 지령 {q_angle(qa0, qc0_wxyz):6.2f}°   (작을수록 회전 추종이 좋다)")
appr = matrix_from_quat(qa0.unsqueeze(0))[0, :, 2]
print(f"  실제 접근축      ({appr[0]:+.3f}, {appr[1]:+.3f}, {appr[2]:+.3f})  (홈 실측 +0.94,+0.26,−0.24)")

env.close()
simulation_app.close()

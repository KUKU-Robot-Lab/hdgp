"""fabric URDF FK ↔ sim USD palm 교차검증 (P1 — robots.py "probe 4").

fabric 은 자산 USD 와 **별개의 URDF** 로 FK/IK 를 푼다. 둘의 기구학이 어긋나면
attractor 가 존재하지 않는 palm 을 향해 풀고, 오차는 자세 의존이라 상수 보정도
불가능하다. 우팔은 0.000mm 검증 기록(08.17)이 있고 좌팔은 기록이 없다.

방법: 다양한 팔 자세(랜덤 palm 목표 세그먼트 주행)에서 매 스텝
  (측정 관절각 → fabric taskmap FK palm) vs (sim USD palm 링크 위치)
를 대조한다. 같은 관절각에서의 순수 FK 비교라 추종 오차와 독립이다.

게이트: 최대 오차 < 5mm (우팔 기록 0.000mm — 유의미하게 크면 URDF 결함).

    ./isaaclab.sh -p scripts/reinforcement_learning/probes/probe_fabric_fk_crosscheck.py \
        --task open-bis_l_grasp_lift_fab
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-bis_l_grasp_lift_fab")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--segments", type=int, default=5, help="랜덤 palm 목표 구간 수")
parser.add_argument("--steps", type=int, default=60, help="구간당 스텝")
parser.add_argument("--gate_mm", type=float, default=5.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym      # noqa: E402
import torch                 # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg   # noqa: E402

import openarm.tasks         # noqa: E402,F401

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()

p = env.profile
print(f"[fk] task={args.task} profile={p.name} fabric={p.fabric_robot_dir}")

taskmap = env.fabric.get_taskmap("palm")
N = env.num_envs
torch.manual_seed(0)

errs = []
for seg in range(args.segments):
    a = torch.zeros(N, env.cfg.action_space, device=env.device)
    # 팔 6D 만 랜덤(박스 안), 손은 열어 둔다 — FK 비교라 손 자세는 무관하지만
    # 자기충돌 반발로 자세 다양성이 줄지 않게 고정.
    a[:, :6] = (torch.rand(N, 6, device=env.device) * 2.0 - 1.0) * 0.8
    a[:, 6:] = -1.0
    for t in range(args.steps):
        env.step(a)
        if t < 20:              # slew 이동 중 자세도 표본에 포함하되 초반만 제외
            continue
        q_fab = env._fabric_order(env.robot.data.joint_pos)
        pts, _ = taskmap(q_fab, None)
        fk_palm = pts[:, :3]
        sim_palm = (env.robot.data.body_pos_w[:, env.palm_idx]
                    - env.scene.env_origins)
        errs.append((fk_palm - sim_palm).norm(dim=-1))
    e_seg = errs[-1]
    print(f"[fk] seg {seg}: 이번 구간 오차 mean {e_seg.mean()*1000:.3f}mm "
          f"max {e_seg.max()*1000:.3f}mm")

e = torch.cat(errs)
mean_mm = float(e.mean()) * 1000.0
p95_mm = float(torch.quantile(e, 0.95)) * 1000.0
max_mm = float(e.max()) * 1000.0
n_samp = e.numel()

print("=" * 70)
print(f"[fk] 표본 {n_samp} (env {N} × seg {args.segments})")
print(f"[fk] 오차 mean {mean_mm:.3f}mm · p95 {p95_mm:.3f}mm · max {max_mm:.3f}mm")
verdict = "PASS" if max_mm < args.gate_mm else "FAIL"
print(f"[fk] 게이트 max<{args.gate_mm}mm → {verdict}")
print("RESULT:", verdict)

env.close()
app.close()

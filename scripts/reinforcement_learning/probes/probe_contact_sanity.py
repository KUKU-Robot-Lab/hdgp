"""접촉 센서가 실제로 힘을 보고하는가 — 물리로 확인한다.

fab_test1~4 에서 `contact/force_max` 가 2048 env 전 구간 **정확히 0.0000** 이었다.
원인은 접촉 필터가 루트 Xform(`/…/Object`)을 가리켜 PhysX 가
"GPU contact filter … is not supported" 경고와 함께 force_matrix_w 를 0 으로 준 것.
rigid body prim(`/…/Object/baseLink`)으로 고쳤다 — 이 probe 가 그 수정을 검증한다.

팔은 홈에 두고 **손가락만 완전히 닫아** 컵을 쥐게 한 뒤 힘을 읽는다.
컵은 손 안으로 옮겨 놓는다(팔 제어 없이 접촉만 격리).

    isaaclab.sh -p .../probe_contact_sanity.py
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=120)
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
zero = torch.zeros(args.num_envs, env.cfg.action_space, device=env.device)
env.step(zero)

# 컵을 손바닥 앞으로 옮긴다(팔은 홈 유지) — 접촉만 격리해서 본다.
palm = env.robot.data.body_pos_w[:, env.palm_idx]
tips = env.robot.data.body_pos_w[:, env._tip_t].mean(dim=1)
target = 0.5 * (palm + tips)                    # palm 과 손끝 사이
root = torch.zeros(args.num_envs, 13, device=env.device)
root[:, :3] = target
root[:, 3] = 1.0
env.object.write_root_state_to_sim(root)

print(f"\n컵을 손 안으로 이동: palm {palm[0].tolist()} → 컵 {target[0].tolist()}", flush=True)
print("손가락을 서서히 완전 폐합(a=+1)한다.\n", flush=True)

act = torch.zeros(args.num_envs, env.cfg.action_space, device=env.device)
for i in range(args.steps):
    frac = min(1.0, i / max(1, args.steps * 0.4))
    act[:, 6:] = frac                            # 손만 닫는다(팔 액션 0 = 홈 유지)
    env.step(act)
    if i in (0, 10, 30, 60, args.steps - 1):
        force, wrapped = env._contact()
        print(f"  step {i:3d} 폐합 {frac:4.2f} | 힘 최대 {force.max():.4f} N"
              f" · env별최대 평균 {force.max(dim=1).values.mean():.4f}"
              f" · >1N 손가락 {(force > 1.0).float().sum(dim=1).mean():.2f}"
              f" · >0.1N {(force > 0.1).float().sum(dim=1).mean():.2f}", flush=True)

# ── 센서 구성 · 손가락별/역할별 분리 확인 ──────────────────────────────
print("\n=== 센서 구성 ===")
n_sensor = 0
for f in env._fingers:
    r = env._sensors[f]
    n_sensor += len(r["tip"]) + len(r["wrap"])
    print(f"  {f:7s} tip {len(r['tip'])}개 {env.profile.finger_tip_bodies[f]}"
          f"  wrap {len(r['wrap'])}개 {env.profile.finger_wrap_bodies.get(f, ())}")
print(f"  총 ContactSensor {n_sensor}개 (scene 등록 {len([k for k in env.scene.sensors if k.startswith('contact_')])}개)")
_s0 = env._sensors[env._fingers[0]]["tip"][0]
print(f"  force_matrix_w shape = {tuple(_s0.data.force_matrix_w.shape)}  (N, body, filter, 3)")
print(f"  필터 = {list(env.cfg.object_contact_filter)}")

print("\n=== 손가락별 · 역할별 접촉력 [N] ===")
print(f"  {'손가락':8s} {'팁':>10s} {'감쌈(최대)':>12s} {'합':>10s}")
for f in env._fingers:
    r = env._sensors[f]
    tp = sum(x.data.force_matrix_w.view(env.num_envs, -1, 3).sum(1).norm(dim=-1) for x in r["tip"])
    wr = [x.data.force_matrix_w.view(env.num_envs, -1, 3).sum(1).norm(dim=-1) for x in r["wrap"]]
    wmax = torch.stack(wr, 0).max(0).values if wr else torch.zeros_like(tp)
    print(f"  {f:8s} {tp.mean():10.3f} {wmax.mean():12.3f} {(tp + sum(wr) if wr else tp).mean():10.3f}")

force, wrapped = env._contact()
print(f"\n  _contact() 손가락별 합계 : {[f'{v:.2f}' for v in force.mean(dim=0).tolist()]}")
print(f"  _contact() wrap 플래그   : {[f'{v:.2f}' for v in wrapped.mean(dim=0).tolist()]}")
print("\n" + "=" * 56)
if force.max() > 0.1:
    print(f"PASS — 접촉 센서가 힘을 보고한다 (최대 {force.max():.3f} N)")
    print(f"  손가락별 평균: {[f'{v:.2f}' for v in force.mean(dim=0).tolist()]}")
else:
    print("★FAIL — 손가락을 완전히 닫았는데도 힘이 0 이다.")
    print("  접촉 필터 prim 경로 · activate_contact_sensors · 콜라이더를 확인할 것.")
env.close()
app.close()

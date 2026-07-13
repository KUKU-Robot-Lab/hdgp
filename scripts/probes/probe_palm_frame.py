"""Phase 0 — palm 프레임 실측.

계획의 전제를 검증한다:
  (1) tesollo palm 로컬 축이 정말 +X=법선 / +Z=손가락인가
  (2) 현재 top-down pregrasp 에서 법선이 정말 수평(+Y)인가  ← "가짜 top-down" 확증
  (3) 좌우 규약이 동일한가 (left 의 C 를 가정하지 않고 실측)
  (4) reset 시 hand_to_object_err(MAX 거리)와 물체 침투 여유

이 수치가 예상과 다르면 이후 계획의 전제가 무너지므로 즉시 중단해야 한다.

사용:
  ./isaaclab.sh -p scripts/probes/probe_palm_frame.py --task open-tesol_r_grasp_v2-lstm
  ./isaaclab.sh -p scripts/probes/probe_palm_frame.py --task open-tesol_l_grasp_v2-lstm
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-tesol_r_grasp_v2-lstm")
parser.add_argument("--num_envs", type=int, default=32)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402

import sys
_OUT = open("/tmp/probe_palm_frame.txt", "w")
_orig_print = print


def print(*a, **kw):  # noqa: A001 — 결과를 파일로도 남긴다(Isaac 종료 시 stdout 유실)
    kw.pop("file", None)
    _orig_print(*a, **kw, flush=True)
    _orig_print(*a, **kw, file=_OUT, flush=True)


env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()

zero = torch.zeros(env.num_envs, env.cfg.num_actions, device=env.device)
for _ in range(int(env.cfg.settle_steps) + 3):
    env.step(zero)

palm_quat = env.robot.data.body_quat_w[:, env.palm_body_index]     # (N,4) wxyz
R = matrix_from_quat(palm_quat)                                    # (N,3,3)
normal = R[:, :, 0]     # +X_local (법선이라고 가정한 축)
finger = R[:, :, 2]     # +Z_local (손가락이라고 가정한 축)

pid = env.palm_pose_id
top = pid == 1


def vec(name, v, mask):
    m = v[mask].mean(dim=0)
    print("  %-22s (%+.3f, %+.3f, %+.3f)" % (name, m[0], m[1], m[2]))


print("\n" + "=" * 68)
print("Phase 0 — palm 프레임 실측 :", args.task)
print("  envs %d  |  top-down %d  side(cup) %d" % (
    env.num_envs, int(top.sum()), int((~top).sum())))
print("=" * 68)

print("\n[1] top-down env 의 palm 축 (world)")
if top.any():
    vec("+X_local (법선?)", normal, top)
    vec("+Z_local (손가락?)", finger, top)
    nz = normal[top][:, 2].abs().mean()
    fz = finger[top][:, 2].mean()
    print("\n  법선의 |z| 성분  = %.3f   → 1.0 이어야 진짜 top-down (손바닥이 아래)" % nz)
    print("  손가락의 z 성분  = %+.3f  → 0.0 이어야 진짜 top-down (손가락 수평)" % fz)
    if nz < 0.5 and fz < -0.5:
        print("\n  ❌ 가짜 top-down 확증: 법선이 수평, 손가락이 수직으로 꽂혀 있다.")
    elif nz > 0.9:
        print("\n  ✅ 진짜 top-down")
    else:
        print("\n  ⚠️  애매 — 수치를 직접 판단할 것")

print("\n[2] side(cup) env 의 palm 축 (world)")
if (~top).any():
    vec("+X_local (법선?)", normal, ~top)
    vec("+Z_local (손가락?)", finger, ~top)

print("\n[3] 거리")
obj = env.object_pos
palm_p = env.palm_center_pos
tips = env.fingertip_pos
d_palm = (palm_p - obj).norm(dim=-1)
d_tips = (tips - obj.unsqueeze(1)).norm(dim=-1)
maxd = torch.cat([d_palm.unsqueeze(1), d_tips], dim=1).max(dim=1).values
mind = torch.cat([d_palm.unsqueeze(1), d_tips], dim=1).min(dim=1).values
for label, m in (("top-down", top), ("side(cup)", ~top)):
    if not m.any():
        continue
    print("  %-10s palm~물체 %5.1fcm | MAX(palm,tips) %5.1fcm | MIN %5.1fcm" % (
        label, d_palm[m].mean() * 100, maxd[m].mean() * 100, mind[m].mean() * 100))
print("  hand_to_object = exp(-10 * MAX). 학습 실측 0.066 → MAX 27cm 에 해당")

print("\n[4] 물체 침투 여유 (MIN 거리 > 0 이어야 겹침 없음)")
print("  최소 MIN 거리 = %.1f cm" % (mind.min() * 100))

_OUT.close()
env.close()
app.close()

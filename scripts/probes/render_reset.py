"""에피소드 시작(reset) 자세를 PNG 로 렌더한다.

정책 없이 무행동으로 settle 까지만 진행 → pregrasp 자세 그대로를 여러 각도에서 촬영.

사용:
  ./isaaclab.sh -p scripts/probes/render_reset.py --task open-tesol_r_grasp_v2-lstm
  ./isaaclab.sh -p scripts/probes/render_reset.py --object cup        # cup 의 side 접근
  ./isaaclab.sh -p scripts/probes/render_reset.py --object small_8_cuboid
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-tesol_r_grasp_v2-lstm")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--object", type=str, default=None)
parser.add_argument("--out", type=str, default="/tmp/reset_shots")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
if args.object is not None:
    names = list(env_cfg.active_object_names)
    if args.object not in names:
        raise SystemExit(f"물체 '{args.object}' 없음")
    env_cfg.cup_cfg.spawn.assets_cfg = [env_cfg.cup_cfg.spawn.assets_cfg[names.index(args.object)]]

env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()

zero = torch.zeros(env.num_envs, env.cfg.num_actions, device=env.device)
for _ in range(int(env.cfg.settle_steps) + 3):
    env.step(zero)

obj = env.object_pos[0].cpu().numpy()
palm = env.palm_center_pos[0].cpu().numpy()
tips = env.fingertip_pos[0].cpu().numpy()
name = args.object or list(env.cfg.active_object_names)[int(env.object_idx[0].item())]
pose = "top-down" if int(env.palm_pose_id[0].item()) == 1 else "side"

os.makedirs(args.out, exist_ok=True)
import omni.replicator.core as rep  # noqa: E402

# 손 + 물체가 모두 들어오는 look-at 지점
center = 0.5 * (obj + palm)
R = 0.55   # 카메라 거리

views = {
    "1_front":  center + np.array([R * 0.85, -R * 0.50, R * 0.20]),   # 정면 비스듬
    "2_side":   center + np.array([0.05, -R * 0.95, R * 0.15]),        # 옆 (손바닥 방향 확인)
    "3_above":  center + np.array([R * 0.35, -R * 0.35, R * 0.85]),    # 위 (top-down 확인)
    "4_close":  center + np.array([R * 0.35, -R * 0.30, R * 0.10]),    # 근접
}
saved = []
for vname, eye in views.items():
    cam = rep.create.camera(
        position=tuple(float(v) for v in eye),
        look_at=tuple(float(v) for v in center),
    )
    rp = rep.create.render_product(cam, (1280, 800))
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach([rp])
    rep.orchestrator.step()          # 1프레임만 렌더
    data = annot.get_data()
    path = os.path.join(args.out, f"{vname}.png")
    Image.fromarray(np.asarray(data)[:, :, :3]).save(path)
    annot.detach()
    rp.destroy()
    saved.append(path)

tip_d = np.linalg.norm(tips - obj, axis=1)
print("\n" + "=" * 62, flush=True)
print("reset 자세 — %s" % args.task, flush=True)
print("  물체       : %s   (%s 접근)" % (name, pose), flush=True)
print("  물체 pos   : (%.3f, %.3f, %.3f)" % tuple(obj), flush=True)
print("  palm pos   : (%.3f, %.3f, %.3f)" % tuple(palm), flush=True)
print("  palm - 물체: Δz %+.1f cm,  거리 %.1f cm" % (
    (palm[2] - obj[2]) * 100, np.linalg.norm(palm - obj) * 100), flush=True)
print("  손끝~물체  : 최소 %.1f cm  최대 %.1f cm" % (tip_d.min() * 100, tip_d.max() * 100), flush=True)
print("  테이블 z   : 0.200  (물체 바닥이 그 위에 있어야 정상)", flush=True)
for p in saved:
    print("  저장: %s" % p, flush=True)
print("=" * 62, flush=True)

env.close()
app.close()

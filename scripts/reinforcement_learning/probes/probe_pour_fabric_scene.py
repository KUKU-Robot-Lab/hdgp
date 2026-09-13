"""pour_fabric 씬 검사 — env_v1 픽스처·컵·비드가 제대로 소환되는지 (프림 목록 + 위치 + 렌더).

    ./isaaclab.sh -p scripts/reinforcement_learning/probes/probe_pour_fabric_scene.py \
        --num_envs 2 --out_dir ~/rl_ws/our_source/pour_fabric_scene_0913 --enable_cameras
"""
from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-short_b_pour_fab")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--out_dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym      # noqa: E402
import numpy as np           # noqa: E402
import torch                 # noqa: E402
from PIL import Image        # noqa: E402
from isaaclab.sensors import TiledCamera, TiledCameraCfg   # noqa: E402
import isaaclab.sim as sim_utils   # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg   # noqa: E402
from pxr import Usd, UsdGeom   # noqa: E402

import openarm.tasks         # noqa: E402,F401
import openarm.agnostic.tasks.pour_fabric.config  # noqa: E402,F401
from openarm.agnostic.tasks.pour_fabric import pour_fabric_env as _pe   # noqa: E402

os.makedirs(args.out_dir, exist_ok=True)

# ---- 카메라: env_.* 에 하나씩(3인칭, 테이블 정면 위에서) --------------------------------------
_orig_setup_scene = _pe.PourFabricEnv._setup_scene


def _setup_scene_with_cam(self):
    _orig_setup_scene(self)
    self._probe_cam = TiledCamera(TiledCameraCfg(
        prim_path="/World/envs/env_.*/ProbeCam",
        offset=TiledCameraCfg.OffsetCfg(pos=(1.35, 0.0, 0.85), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=1.2,
                                         horizontal_aperture=20.955, clipping_range=(0.05, 6.0)),
        width=960, height=720))
    self.scene.sensors["probe_cam"] = self._probe_cam


_pe.PourFabricEnv._setup_scene = _setup_scene_with_cam

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg).unwrapped
# 카메라를 테이블 중심(0.35, 0, 0.30)을 보게 — world 규약: yaw 180° 로 −x 를 보고 살짝 내려봄
from isaaclab.utils.math import quat_from_euler_xyz  # noqa: E402
_q = quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.42]), torch.tensor([3.14159]))[0]
env._probe_cam.set_world_poses(
    positions=env.scene.env_origins + torch.tensor([1.35, 0.0, 0.85], device=env.device),
    orientations=_q.to(env.device).unsqueeze(0).expand(env.num_envs, 4), convention="world")

obs, _ = env.reset()
N, A, hold = env.num_envs, env.cfg.action_space, int(env.cfg.hold_steps)

# ---- 1. env_0 프림 목록 ----------------------------------------------------------------------
stage = env.sim.stage
prims = [p.GetPath().pathString for p in stage.Traverse()
         if p.GetPath().pathString.startswith("/World/envs/env_0/")]
top = sorted({p.split("/")[4] for p in prims if len(p.split("/")) > 4})
report = {"env0_children": top,
          "table_prims": [p for p in prims if "/Table" in p][:6],
          "n_beads": len([t for t in top if t.startswith("Bead")]),
          "n_table_prims": len([p for p in prims if "/Table" in p])}
# 테이블 상판 bbox(world) — env_0
bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
tp = stage.GetPrimAtPath("/World/envs/env_0/Table")
if tp.IsValid():
    r = bbox.ComputeWorldBound(tp).ComputeAlignedRange()
    report["table_bbox_env0"] = {"min": list(r.GetMin()), "max": list(r.GetMax())}
o0 = env.scene.env_origins[0]


def _loc(v):
    return [round(float(x), 4) for x in (v - o0).tolist()]


def snapshot(tag):
    cfg = env.cfg
    return {
        "tag": tag,
        "source_cup": _loc(env.source_cup.data.root_pos_w[0]),
        "receiver_cup": _loc(env.receiver_cup.data.root_pos_w[0]),
        "cup_rest_z_expected": round(float(cfg.table_surface_z) + float(cfg.object_origin_offset_z), 4),
        "src_palm": _loc(env.robot.data.body_pos_w[0, env.src.palm_idx]),
        "rcv_palm": _loc(env.robot.data.body_pos_w[0, env.rcv.palm_idx]),
        "beads_z_min_max": [round(float((env.beads.data.object_pos_w[0, :, 2] - o0[2]).min()), 4),
                            round(float((env.beads.data.object_pos_w[0, :, 2] - o0[2]).max()), 4)],
        "beads_in_source": round(float(env._prev_in_src[0]), 3),
        "src_spawn_center": list(env.src.profile.object_spawn_center),
        "rcv_spawn_center": list(env.rcv.profile.object_spawn_center),
    }


def save_frame(name):
    env.sim.render()
    env._probe_cam.update(dt=0.0)
    rgb = env._probe_cam.data.output["rgb"][0].cpu().numpy()
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    Image.fromarray(rgb.astype(np.uint8)).save(os.path.join(args.out_dir, name))


half = A // 2


def scripted(t):
    a = torch.zeros(N, A, device=env.device)
    u = t - hold
    if u < 0:
        return a
    for i, rig in enumerate((env.src, env.rcv)):
        a[:, i * half] = min(0.07 / float(rig.delta_hi[0]), 1.0)
    if u >= 100:
        a[:, 6:half] = 1.0
        a[:, half + 6:] = 1.0
    if u >= 250:
        for i, rig in enumerate((env.src, env.rcv)):
            a[:, i * half + 2] = min(0.25 / float(rig.delta_hi[2]), 1.0)
    if u >= 400:
        a[:, 5] = -1.0
    return a


snaps = []
frames = {0: "00_reset.png", hold: "01_after_hold.png", hold + 240: "02_grasped.png",
          hold + 390: "03_lifted.png", hold + 620: "04_pouring.png"}
for t in range(hold + 640):
    if t in frames:
        env.sim.step(render=True) if False else None
        snaps.append(snapshot(frames[t]))
        save_frame(frames[t])
    env.step(scripted(t))
snaps.append(snapshot("end"))
report["snapshots"] = snaps
with open(os.path.join(args.out_dir, "scene_report.json"), "w") as f:
    json.dump(report, f, indent=1, ensure_ascii=False)
print(json.dumps(report, indent=1, ensure_ascii=False))
env.close()
app.close()

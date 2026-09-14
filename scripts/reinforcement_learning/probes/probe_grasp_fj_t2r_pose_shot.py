"""grasp_fj_t2r 시작 자세 스냅샷 — 리셋 직후 자세를 여러 카메라·여러 env(컵 종류)로 PNG 로 남긴다(사용자 확인용).

팔은 액션 0(증분 0 = 유지), 손은 리셋 자세를 유지하는 액션으로 `--settle_steps` 만큼 굴린 뒤 찍는다 —
정책이 받는 첫 관측과 같은 상태다. 부팅 로그(시작 거리 가드·팔 속도·물체 뱅크)와 실측 palm 위치를 같이 남긴다.

    isaaclab.sh -p scripts/reinforcement_learning/probes/probe_grasp_fj_t2r_pose_shot.py \
        --task open-short_r_grasp_fj_t2r_reach-lstm-sapg --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-short_r_grasp_fj_t2r_reach-lstm-sapg")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--settle_steps", type=int, default=30)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True          # rgb_array 렌더에 필요
app = AppLauncher(args).app

import gymnasium as gym      # noqa: E402
import torch                 # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg   # noqa: E402

import openarm.tasks         # noqa: E402,F401
# ★tasks/__init__ 는 등록 모듈의 ImportError 를 조용히 삼킨다 — 여기서 명시 import 해 드러낸다.
import openarm.agnostic.tasks.grasp_fj_t2r.config  # noqa: E402,F401

#: (이름, env index, eye, lookat) — env 원점 기준 m. env_id % 8 = cup_family 종 순서(0 cup_big_s085 · 3 s130 · 4 shaker_closed).
SHOTS = (
    ("front", 0, (1.10, -0.80, 0.78), (0.20, -0.22, 0.35)),
    ("wide", 0, (1.70, -1.30, 1.05), (0.15, -0.20, 0.35)),
    ("side_right", 0, (0.25, -1.35, 0.55), (0.22, -0.20, 0.33)),
    ("top_oblique", 0, (-0.20, -0.30, 1.45), (0.26, -0.20, 0.20)),
    ("front_env3", 3, (1.10, -0.80, 0.78), (0.20, -0.22, 0.35)),
    ("front_env4", 4, (1.10, -0.80, 0.78), (0.20, -0.22, 0.35)),
)

os.makedirs(args.out, exist_ok=True)
cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
cfg.viewer.origin_type = "env"
cfg.viewer.env_index = SHOTS[0][1]
cfg.viewer.eye, cfg.viewer.lookat = SHOTS[0][2], SHOTS[0][3]
cfg.viewer.resolution = (1280, 720)
env = gym.make(args.task, cfg=cfg, render_mode="rgb_array").unwrapped
env.reset()
N, A, dev = env.num_envs, int(env.cfg.action_space), env.device
lo, hi = env._act_lo, env._act_hi
act = torch.zeros(N, A, device=dev)
act[:, 7:] = (2.0 * (env._hand_reset_q - lo) / (hi - lo).clamp(min=1e-6) - 1.0).clamp(-1.0, 1.0)
for _ in range(args.settle_steps):
    env.step(act)


def _save(img, path: str) -> str:
    """PNG 로 저장(PIL → cv2). 둘 다 없으면 .npy 로 남긴다(로컬에서 변환) — 저장 실패로 부팅 결과를 잃지 않게."""
    arr = img[..., :3] if img.ndim == 3 else img
    try:
        from PIL import Image
        Image.fromarray(arr).save(path)
        return path
    except ImportError:
        pass
    try:
        import cv2
        cv2.imwrite(path, arr[..., ::-1])
        return path
    except ImportError:
        import numpy as np
        npy = path[:-4] + ".npy"
        np.save(npy, arr)
        return npy


ctrl = env.viewport_camera_controller
saved = []
for name, idx, eye, lookat in SHOTS:
    ctrl.set_view_env_index(idx)
    ctrl.update_view_location(eye=eye, lookat=lookat)
    img = None
    for _ in range(6):                    # 카메라를 옮긴 뒤 몇 프레임 지나야 annotator 에 반영된다
        img = env.render()
    saved.append(_save(img, os.path.join(args.out, f"{name}.png")))

palm = (env.robot.data.body_pos_w[:, env.palm_idx] - env.scene.env_origins)
obj = (env.object.data.root_pos_w - env.scene.env_origins)
arm_q0 = env.robot.data.joint_pos[0, env._arm_ids_t]
summary = {
    "task": args.task, "settle_steps": args.settle_steps,
    "episode_steps": int(env.max_episode_length), "k_arm": float(env.cfg.k_arm),
    "arm_slew_rad_s": float(env.cfg.arm_slew_rad_s), "object_bank": env.cfg.object_bank,
    "arm_reset_override": list(env.cfg.arm_reset_joint_pos_override),
    "arm_q_env0": [round(float(v), 4) for v in arm_q0],
    "palm_env0": [round(float(v), 4) for v in palm[0]],
    "palm_to_cup_m": [round(float(v), 4) for v in (palm - obj).norm(dim=-1)],
    "hand_z_min": round(float(env._hand_z_min.min()), 4),
    "cup_per_env": [env._species_names[int(i)] for i in env._species_ids],
    "images": saved,
}
print("POSE_SHOT " + json.dumps(summary, ensure_ascii=False), flush=True)
with open(os.path.join(args.out, "summary.json"), "w") as f:
    json.dump(summary, f, indent=1, ensure_ascii=False)
env.close()
app.close()

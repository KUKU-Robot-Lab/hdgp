"""pour_fabric 부팅·무작위 롤아웃 스모크 (Phase 1 검증 게이트).

게이트: 부팅(fabric FK 정합 ×2 팔) · N 스텝 무작위 액션에서 NaN/runaway 0 ·
hold 종료 시 비드 in_source ≥ 0.9 · 보상 로더 경로 동작.

    ./isaaclab.sh -p scripts/reinforcement_learning/probes/probe_pour_fabric_boot.py \
        --task open-short_b_pour_fab --num_envs 8 --steps 300 [--reward_code_path f.py]
"""
from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-short_b_pour_fab")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--reward_code_path", default="")
parser.add_argument("--action_scale", type=float, default=1.0)
parser.add_argument("--zero_action", action="store_true")
parser.add_argument("--script", default="", choices=["", "grasp", "pour"],
                    help="스크립트 액션: grasp = 접근→손 닫기→들기 · pour = 그 뒤 소스 회전 슬롯 −1")
parser.add_argument("--approach_m", type=float, default=0.07, help="닫기 전 +x 접근 거리")
parser.add_argument("--lift_m", type=float, default=0.10, help="들기 높이")
parser.add_argument("--tilt_slot", type=int, default=3, help="pour 에서 −1 을 줄 소스 회전 슬롯(3|4|5)")
parser.add_argument("--out", default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym      # noqa: E402
import torch                 # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg   # noqa: E402

import openarm.tasks         # noqa: E402,F401
# ★tasks/__init__ 는 등록 모듈의 ImportError 를 조용히 삼킨다 — 여기서 명시 import 해 드러낸다.
import openarm.agnostic.tasks.pour_fabric.config  # noqa: E402,F401

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env_cfg.reward_code_path = args.reward_code_path
env = gym.make(args.task, cfg=env_cfg).unwrapped
obs, _ = env.reset()
N = env.num_envs
A = env.cfg.action_space
hold = int(env.cfg.hold_steps)
print(f"[boot] obs policy {tuple(obs['policy'].shape)} critic {tuple(obs['critic'].shape)} action {A}")

summary = {"nan_steps": 0, "runaway": 0.0, "drop": 0.0, "in_source_after_hold": None,
           "grasp_src_max": 0.0, "grasp_rcv_max": 0.0, "palm_err_src": [], "palm_err_rcv": [],
           "reward_mean": [], "terms": {}}
half = A // 2


def scripted(t: int) -> torch.Tensor:
    """hold 뒤: 0~100 +x 접근 · 100~250 손 닫기 · 250~400 z +10cm 들기 · pour: 400~ 회전 슬롯 −1."""
    a = torch.zeros(N, A, device=env.device)      # a=0 = 앵커(시작 palm)
    u = t - hold
    if u < 0:
        return a
    for i, rig in enumerate((env.src, env.rcv)):
        a[:, i * half + 0] = min(args.approach_m / float(rig.delta_hi[0]), 1.0)
    if u >= 100:
        a[:, 6:half] = 1.0
        a[:, half + 6:] = 1.0
    if u >= 250:
        for i, rig in enumerate((env.src, env.rcv)):
            a[:, i * half + 2] = min(args.lift_m / float(rig.delta_hi[2]), 1.0)   # (a≥0 → a·hi)
    if args.script == "pour" and u >= 400:
        a[:, args.tilt_slot] = -1.0
    return a


for t in range(args.steps):
    if args.script:
        a = scripted(t)
    elif args.zero_action:
        a = torch.zeros(N, A, device=env.device)
    else:
        a = (torch.rand(N, A, device=env.device) * 2 - 1) * args.action_scale
    obs, rew, term, trunc, extras = env.step(a)
    if not torch.isfinite(obs["policy"]).all() or not torch.isfinite(rew).all():
        summary["nan_steps"] += 1
    summary["runaway"] += float(extras.get("task/runaway_rate", 0.0))
    summary["drop"] += float(extras.get("done/drop", 0.0))
    if t == hold:
        summary["in_source_after_hold"] = float(extras["bead/in_source"])
    summary["grasp_src_max"] = max(summary["grasp_src_max"], float(extras["task/src_grasped"]))
    summary["grasp_rcv_max"] = max(summary["grasp_rcv_max"], float(extras["task/rcv_grasped"]))
    summary["palm_err_src"].append(float(extras["fabric/src_palm_err"]))
    summary["palm_err_rcv"].append(float(extras["fabric/rcv_palm_err"]))
    summary["reward_mean"].append(float(rew.mean()))
    for k, v in extras.items():
        if k.startswith("reward/"):
            summary["terms"].setdefault(k, []).append(float(v))
    if t % 50 == 0:
        print(f"[step {t:4d}] inS={float(extras['bead/in_source']):.3f} "
              f"gS={float(extras['task/src_grasped']):.2f} gR={float(extras['task/rcv_grasped']):.2f} "
              f"palm_err S/R={float(extras['fabric/src_palm_err'])*1000:.1f}/"
              f"{float(extras['fabric/rcv_palm_err'])*1000:.1f}mm "
              f"liftS={float(extras['task/src_cup_lift']):+.3f} liftR={float(extras['task/rcv_cup_lift']):+.3f} "
              f"tiltS={float(extras['task/src_tilt_deg']):.1f} rotErrS={float(extras['fabric/src_rot_err_deg']):.1f}° "
              f"rew={float(rew.mean()):+.3f}", flush=True)

summary["palm_err_src"] = {"mean_mm": 1000 * sum(summary["palm_err_src"]) / len(summary["palm_err_src"]),
                           "max_mm": 1000 * max(summary["palm_err_src"])}
summary["palm_err_rcv"] = {"mean_mm": 1000 * sum(summary["palm_err_rcv"]) / len(summary["palm_err_rcv"]),
                           "max_mm": 1000 * max(summary["palm_err_rcv"])}
summary["reward_mean"] = sum(summary["reward_mean"]) / len(summary["reward_mean"])
summary["terms"] = {k: sum(v) / len(v) for k, v in summary["terms"].items()}
gate = (summary["nan_steps"] == 0 and summary["runaway"] == 0.0
        and (summary["in_source_after_hold"] or 0.0) >= 0.9)
summary["GATE"] = "PASS" if gate else "FAIL"
print(json.dumps(summary, indent=1, ensure_ascii=False))
if args.out:
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
env.close()
app.close()

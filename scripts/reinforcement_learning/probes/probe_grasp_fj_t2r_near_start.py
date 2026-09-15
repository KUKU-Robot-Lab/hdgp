"""grasp_fj_t2r 가까운 출발(시작 상태 커리큘럼) 스모크 — 학습 없이 리셋이 컵 옆 IK 자세를 안전하게 심는지 본다.

★09.15 사용자 "시작 상태 커리큘럼 + env 고정 · 50 % · C자 사전파지"(reach leaf `near_start_*`).
  부팅 → 기본 손 자세 유지 액션으로 워밍업(공통 스텝 > near_start_after_common_steps) → env.reset() 으로 전 env 리셋 →
  같은 유지 액션으로 N 스텝. 가까운 출발 env 가
    · 비율 ≈ near_start_frac,
    · 첫 스텝에 C자 완료 조금 앞(손바닥면 4.5 cm · 컵 축 R+2.5 cm · 띠 0.8 H · 시작 방향)에 있고(컵 스폰 ±2 cm 포함),
    · ★09.16 사용자 "가까운 출발 = 접근 완료로 시작" — 첫 스텝에 접근 래치가 서 있고(먼 출발은 안 서 있고),
    · 유지하는 동안 종료·abnormal 없이, 손가락·엄지가 컵에 닿지 않고, 컵이 밀리지 않고, 팔이 목표를 따라가는지.
  먼 출발 env 는 여전히 0.38 m 쪽에서 시작하는지.

    isaaclab.sh -p scripts/reinforcement_learning/probes/probe_grasp_fj_t2r_near_start.py \
        --num_envs 64 --steps 90 [--out f.json]
"""
from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-short_r_grasp_fj_t2r_reach-lstm")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--warmup", type=int, default=6, help="리셋 전 유지 스텝 — 공통 스텝이 near_start_after_common_steps 를 넘게")
parser.add_argument("--steps", type=int, default=90)
parser.add_argument("--settle", type=int, default=5, help="팔 추종 오차를 재기 시작하는 스텝")
parser.add_argument("--touch_n", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=1234)
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
import openarm.agnostic.tasks.grasp_fj_t2r.config  # noqa: E402,F401
from openarm.agnostic.tasks.grasp_fj_t2r import grasp_gates as G   # noqa: E402
from openarm.agnostic.tasks.grasp_fj_t2r.stage_funnel import palm_band_gap   # noqa: E402

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
cfg.seed = args.seed
cfg.reward_code_path = ""
env = gym.make(args.task, cfg=cfg).unwrapped
N, A, dev = env.num_envs, int(env.cfg.action_space), env.device
if env._t2r_near_q is None:
    raise SystemExit(f"[probe] {args.task} 는 가까운 출발이 꺼져 있다(near_start_frac {env.cfg.near_start_frac})")
lo, hi = env._act_lo, env._act_hi
hold = torch.zeros(N, A, device=dev)
hold[:, 7:] = (2.0 * (env._hand_reset_q - lo) / (hi - lo).clamp(min=1e-6) - 1.0).clamp(-1.0, 1.0)

env.reset()
for _ in range(args.warmup):
    env.step(hold)
counter_at_reset = int(env.common_step_counter)
env.reset()
near = env._t2r_near.clone()
far = ~near
print(f"[probe] 리셋 공통 스텝 {counter_at_reset} · 가까운 출발 {int(near.sum())}/{N}", flush=True)


def geometry(ctx) -> dict:
    gap, along, height = G.c_pregrasp_geometry(ctx.palm_pos, ctx.palm_normal, ctx.palm_finger_dir,
                                               ctx.cup_pos, ctx.cup_axis, ctx.cup_radius)
    return {"plane_gap": gap, "along_offset": along, "height_frac": height / ctx.cup_half_height,
            "orient": G.hand_orientation(ctx.palm_normal, ctx.palm_finger_dir),
            "band_gap": palm_band_gap(ctx.palm_pos, ctx.cup_pos, ctx.cup_axis, ctx.cup_radius, ctx.cup_half_height)}


def span(x: torch.Tensor, m: torch.Tensor) -> list[float]:
    return [round(float(x[m].min()), 4), round(float(x[m].mean()), 4), round(float(x[m].max()), 4)] if bool(m.any()) else []


digit_max = torch.zeros(N, device=dev)
palm_max = torch.zeros(N, device=dev)
cup_disp_max = torch.zeros(N, device=dev)
arm_err_max = torch.zeros(N, device=dev)
done_any = torch.zeros(N, dtype=torch.bool, device=dev)
abnormal_any = torch.zeros(N, dtype=torch.bool, device=dev)
approach_any = torch.zeros(N, dtype=torch.bool, device=dev)
first = last = None
for t in range(args.steps):
    _, _, term, trunc, _ = env.step(hold)
    ctx = env._t2r_ctx
    geo = geometry(ctx)
    if t == 0:
        first = geo
        latched_first = ctx.approach_done.clone()
    last = geo
    live = ~done_any                     # 한 번 끝난 env 는 새 에피소드라 이후 값을 섞지 않는다
    digit_max = torch.where(live, torch.maximum(digit_max, ctx.link_cup_force.amax(dim=(1, 2))), digit_max)
    palm_max = torch.where(live, torch.maximum(palm_max, ctx.palm_cup_force), palm_max)
    disp = (ctx.cup_pos[:, :2] - ctx.cup_spawn_pos[:, :2]).norm(dim=-1)
    cup_disp_max = torch.where(live, torch.maximum(cup_disp_max, disp), cup_disp_max)
    if t >= args.settle:
        err = (env.robot.data.joint_pos[:, env._arm_ids_t] - env._arm_q_target).abs().amax(dim=-1)
        arm_err_max = torch.where(live, torch.maximum(arm_err_max, err), arm_err_max)
    abnormal_any |= live & env._abnormal
    approach_any |= live & ctx.approach_done
    done_any |= term | trunc

summary = {
    "task": args.task, "num_envs": N, "common_step_at_reset": counter_at_reset,
    "near_frac": round(float(near.float().mean()), 3), "near_start_frac_cfg": float(env.cfg.near_start_frac),
    "near_first_step": {k: span(v, near) for k, v in first.items()},
    "near_last_step": {k: span(v, near & ~done_any) for k, v in last.items()},
    "far_first_step_band_gap": span(first["band_gap"], far),
    "near_digit_force_max_N": round(float(digit_max[near].max()), 4) if bool(near.any()) else None,
    "near_palm_force_max_N": round(float(palm_max[near].max()), 4) if bool(near.any()) else None,
    "near_cup_disp_max_m": round(float(cup_disp_max[near].max()), 4) if bool(near.any()) else None,
    "near_arm_err_max_rad": round(float(arm_err_max[near].max()), 4) if bool(near.any()) else None,
    "far_arm_err_max_rad": round(float(arm_err_max[far].max()), 4) if bool(far.any()) else None,
    "near_done": int((done_any & near).sum()), "near_abnormal": int((abnormal_any & near).sum()),
    "near_approach_latched": int((approach_any & near).sum()),
    "near_latched_first_step": int((latched_first & near).sum()),
    "far_latched_first_step": int((latched_first & far).sum()),
    "species_near_counts": {nm: int((near & (env._species_ids == i)).sum()) for i, nm in enumerate(env._species_names)},
}
nf = summary["near_first_step"]
gate = (0.3 <= summary["near_frac"] <= 0.7
        # 손바닥면 4.5 cm ± 컵 스폰 2 cm — 손·컵 겹침 여유
        and bool(nf["plane_gap"]) and 0.021 <= nf["plane_gap"][0] and nf["plane_gap"][2] <= 0.07
        # ★09.16 가까운 출발은 접근 완료(래치)로 시작하고 먼 출발은 아니다
        and summary["near_latched_first_step"] == int(near.sum()) and summary["far_latched_first_step"] == 0
        and 0.0 <= nf["along_offset"][0] and nf["along_offset"][2] <= 0.05
        and 0.5 <= nf["height_frac"][0] and nf["height_frac"][2] <= 1.0
        and nf["orient"][0] >= 0.95
        and summary["far_first_step_band_gap"][0] > 0.2
        and summary["near_done"] == 0 and summary["near_abnormal"] == 0
        and summary["near_digit_force_max_N"] < args.touch_n
        and summary["near_cup_disp_max_m"] < 0.005
        and summary["near_arm_err_max_rad"] < 0.05)
summary["GATE"] = "PASS" if gate else "FAIL"
print(json.dumps(summary, indent=1, ensure_ascii=False))
if args.out:
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
env.close()
app.close()

"""grasp_fj_t2r 부팅·접촉 배선 스모크 — 학습 없이 생성 보상 경로와 **보상 전용** 컵 접촉 센서를 확인한다.

두 모드:
  wire   : 리셋 뒤 컵을 손 앞(검·중·약지 `_3`·`_4` 링크 중심 + 손바닥 법선 수평성분 × (R + margin))으로
           **수평으로만** 옮기고(높이는 테이블에 놓인 그대로), env 를 6 묶음으로 나눠 묶음 k(0..4)는
           손가락 k 만, 묶음 5 는 다섯 손가락을 전부 굽힌다. 팔 액션은 0(유지).
           → 검~새끼 한 손가락 묶음에서 **그 손가락 행만** 서고(엄지 행은 대향면이라 같이 설 수 있다)
             "전부" 묶음에서 엄지 행이 서야 손가락→행 매핑이 맞다. 필터 경로가 0개 매칭이거나 다중 body 를
             한 센서에 묶으면 `force_matrix_w` 가 무증상 0 이라, "힘이 0 이 아니다" 자체도 확인 대상이다.
  random : 무작위 액션 — 보상·항 유한성 · 리셋 직후 `ctx.prev_actions` 가 0 인지.

    isaaclab.sh -p scripts/reinforcement_learning/probes/probe_grasp_fj_t2r_boot.py \
        --mode wire --num_envs 12 --steps 150 [--reward_code_path <abs>] [--out f.json]
"""
from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-short_r_grasp_fj_t2r-lstm-sapg")
parser.add_argument("--num_envs", type=int, default=12)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--mode", default="wire", choices=["wire", "random"])
parser.add_argument("--reward_code_path", default="")
parser.add_argument("--settle_steps", type=int, default=5, help="wire: 컵을 옮기기 전 대기 스텝(≥1 — ctx 가 있어야 한다)")
parser.add_argument("--margin_m", type=float, default=0.02, help="wire: 링크 중심에서 컵 표면까지 여유")
parser.add_argument("--touch_n", type=float, default=0.1, help="접촉으로 셀 힘 [N]")
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

FINGERS = ("thumb", "index", "middle", "ring", "pinky")    # 프롬프트 계약 순서
N_GROUPS = len(FINGERS) + 1                                 # 손가락 하나씩 + 전부

if args.settle_steps < 1:
    raise SystemExit("--settle_steps 는 1 이상이어야 한다(첫 스텝 전에는 ctx 가 없다)")
cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
cfg.seed = args.seed
cfg.reward_code_path = args.reward_code_path
env = gym.make(args.task, cfg=cfg).unwrapped
env.reset()
N, A, dev = env.num_envs, int(env.cfg.action_space), env.device
if tuple(env._finger_names) != FINGERS:
    raise RuntimeError(f"손가락 순서가 프롬프트 계약과 다르다: {env._finger_names}")
names = [env.robot.data.joint_names[i] for i in env._syn_ids]
lo, hi = env._act_lo, env._act_hi
a_hold_hand = (2.0 * (env._hand_reset_q - lo) / (hi - lo).clamp(min=1e-6) - 1.0).clamp(-1.0, 1.0)
movable = (hi - lo) > 0.05
cols = {f: torch.tensor([j for j, nm in enumerate(names) if f"_{f}_" in nm and bool(movable[j])],
                        device=dev, dtype=torch.long) for f in FINGERS}
group = torch.arange(N, device=dev) % N_GROUPS
print(f"[probe] N {N} · action {A} · 손 관절 {len(names)} · 가동 열 "
      + " ".join(f"{f}:{len(c)}" for f, c in cols.items()), flush=True)


def place_cup_in_hand() -> dict:
    """컵을 손가락이 굽혀 닿는 자리로 옮긴다(수평만). 반환 = 기하 진단."""
    ctx = env._t2r_ctx
    mid = ctx.link_pos[:, 1:4, 0:2].reshape(N, -1, 3).mean(dim=1)
    n = ctx.palm_normal
    n_xy = n[:, :2] / n[:, :2].norm(dim=-1, keepdim=True).clamp(min=1e-6)
    new = ctx.cup_pos.clone()
    new[:, :2] = mid[:, :2] + n_xy * (ctx.cup_radius + args.margin_m).unsqueeze(-1)
    ids = torch.arange(N, device=dev)
    env.object.write_root_pose_to_sim(
        torch.cat([new + env.scene.env_origins, env.object.data.root_quat_w], dim=-1), env_ids=ids)
    env.object.write_root_velocity_to_sim(torch.zeros(N, 6, device=dev), env_ids=ids)
    return {"palm_normal_z_mean": float(n[:, 2].mean()),
            "palm_normal_z_absmax": float(n[:, 2].abs().max()),
            "links_minus_cup_z_mean_m": float((mid[:, 2] - ctx.cup_pos[:, 2]).mean()),
            "cup_moved_xy_mean_m": float((new[:, :2] - ctx.cup_pos[:, :2]).norm(dim=-1).mean())}


def action(t: int) -> torch.Tensor:
    if args.mode == "random":
        return torch.rand(N, A, device=dev) * 2.0 - 1.0
    a = torch.zeros(N, A, device=dev)
    a[:, 7:] = a_hold_hand
    if t > args.settle_steps:
        for k, f in enumerate(FINGERS):
            idx = ((group == k) | (group == N_GROUPS - 1)).nonzero(as_tuple=True)[0]
            if len(idx) and len(cols[f]):
                a[idx.unsqueeze(1), 7 + cols[f].unsqueeze(0)] = 1.0
    return a


# 프롬프트가 생성기에게 보여준 손 관절표(이름·순서·범위)가 부팅 실측과 같은가 — 표는 손으로 옮긴 값이다.
from openarm.agnostic.tasks.grasp_fj_t2r.t2r import prompts as P   # noqa: E402

_tbl = {nm: (a, b) for nm, a, b in P.HAND_JOINT_RANGES}
joint_table = {"order_ok": tuple(names) == tuple(P.HAND_JOINT_NAMES),
               "range_max_abs_err": (max(max(abs(float(lo[j]) - _tbl[nm][0]), abs(float(hi[j]) - _tbl[nm][1]))
                                         for j, nm in enumerate(names)) if set(names) == set(_tbl) else None)}
summary: dict = {"mode": args.mode, "reward_code_path": args.reward_code_path or "(zero)",
                 "nan_steps": 0, "resets": 0, "prev_actions_nonzero_after_reset": 0,
                 "reward_mean": [], "terms": {}, "contact_keys": [], "done_counts": {}}
force_max = torch.zeros(N, len(FINGERS), 3, device=dev)
palm_max = torch.zeros(N, device=dev)
done_prev = torch.zeros(N, dtype=torch.bool, device=dev)
for t in range(args.steps):
    if args.mode == "wire" and t == args.settle_steps:
        summary["place"] = place_cup_in_hand()
    _, rew, term, trunc, extras = env.step(action(t))
    ctx = env._t2r_ctx
    if not torch.isfinite(rew).all():
        summary["nan_steps"] += 1
    if bool(done_prev.any()):
        summary["prev_actions_nonzero_after_reset"] += int((ctx.prev_actions[done_prev].abs().amax(dim=-1) > 0).sum())
    done_prev = term | trunc
    summary["resets"] += int(done_prev.sum())
    if t > args.settle_steps:
        force_max = torch.maximum(force_max, ctx.link_cup_force)
        palm_max = torch.maximum(palm_max, ctx.palm_cup_force)
    summary["reward_mean"].append(float(rew.mean()))
    for k, v in extras.items():
        if k.startswith("reward/"):
            summary["terms"].setdefault(k, []).append(float(v))
        elif k.startswith("done/") and "qd_max" not in k and "beyond_j" not in k:
            # 종료 사유별 env 수(스텝 평균 × N 누적) — 리셋이 났을 때 무엇 때문인지 가른다.
            summary["done_counts"][k] = summary["done_counts"].get(k, 0.0) + float(v) * N
    if t % 25 == 0:
        print(f"[step {t:4d}] fingers_touching {float(extras.get('contact/fingers_touching', -1)):.2f} "
              f"palm {float(extras.get('contact/palm_touching', -1)):.2f} "
              f"force_mean {float(extras.get('contact/link_force_mean', -1)):.3f}N "
              f"tilt {float(extras.get('task/tilt_deg', -1)):.1f}° rew {float(rew.mean()):+.4f}", flush=True)

summary["contact_keys"] = sorted(k for k in extras if k.startswith("contact/"))
# 에피소드 퍼널(루프 틱 재료) — 마지막 스텝 값. *_ep 는 끝난 에피소드가 없으면 −1.
summary["stage"] = {k: round(float(v), 4) for k, v in extras.items() if k.startswith("stage/")}
summary["reward_mean"] = sum(summary["reward_mean"]) / max(len(summary["reward_mean"]), 1)
summary["terms"] = {k: sum(v) / len(v) for k, v in summary["terms"].items()}
gate = (summary["nan_steps"] == 0 and "reward/total" in summary["terms"]
        and "contact/fingers_touching" in summary["contact_keys"]
        and "stage/palm_cup_gap" in summary["stage"]
        and summary["prev_actions_nonzero_after_reset"] == 0)
summary["prompt_joint_table"] = joint_table
gate = gate and joint_table["order_ok"] and (joint_table["range_max_abs_err"] or 1.0) <= 2e-3

if args.mode == "wire":
    touch = force_max > args.touch_n
    per_group = {}
    for g in range(N_GROUPS):
        m = group == g
        label = FINGERS[g] if g < len(FINGERS) else "all"
        per_group[label] = {
            "finger_touch_frac": {f: round(float(touch[m, k].any(dim=-1).float().mean()), 2)
                                  for k, f in enumerate(FINGERS)},
            "finger_force_max_N": {f: round(float(force_max[m, k].max()), 3) for k, f in enumerate(FINGERS)},
            "palm_force_max_N": round(float(palm_max[m].max()), 3),
        }
    summary["per_group"] = per_group
    # ★매핑 판정(09.14 첫 실측으로 고침): 한 손가락만 굽혀도 **엄지 행**이 선다 — 굽힌 손가락이 컵을
    #   가만히 있는 엄지(대향면) 쪽으로 밀기 때문이다(검지 묶음 엄지 9.4 N · 새끼 묶음 59 N). 그래서
    #   "가장 센 행 = 굽힌 손가락" 은 틀린 판정이다. 네 손가락(검~새끼) 묶음은 **자기 행이 서고 다른 세
    #   손가락 행은 0** 이어야 하고, 엄지 행은 "전부" 묶음에서 서야 한다(엄지 단독 굽힘은 컵에 안 닿는다).
    fingers_rows = {}
    for k in range(1, len(FINGERS)):
        rows = touch[group == k].float().amax(dim=(0, 2)) > 0          # (5,) 이 묶음에서 한 번이라도 선 행
        fingers_rows[FINGERS[k]] = {
            "own_row": bool(rows[k]),
            "other_finger_rows": [FINGERS[j] for j in range(1, len(FINGERS)) if j != k and bool(rows[j])]}
    thumb_live = bool(touch[group == N_GROUPS - 1][:, 0].any())
    summary["mapping"] = {"fingers": fingers_rows, "thumb_row_live_in_all_group": thumb_live}
    summary["mapping_ok"] = thumb_live and all(v["own_row"] and not v["other_finger_rows"]
                                               for v in fingers_rows.values())
    gate = gate and summary["mapping_ok"]

summary["GATE"] = "PASS" if gate else "FAIL"
print(json.dumps(summary, indent=1, ensure_ascii=False))
if args.out:
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
env.close()
app.close()

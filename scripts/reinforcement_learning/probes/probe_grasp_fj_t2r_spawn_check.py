"""grasp_fj_t2r 컵 소환 점검 — 컵이 상판 위에 제대로 서는가, 시작 자세의 손·팔과 겹쳐 소환되는가.

★09.17 사용자 "테이블 위치에 정확히 소환되는지와, 손등에 겹침 소환이 되는지도 확인" — rand leaf(소환 x 0.10–0.40 · y −0.30–0.00).
팔 증분 0 · 손은 리셋 자세 유지로 `--settle` 스텝 굴리며 전체 리셋을 `--rounds` 번 반복한다(리셋마다 소환 재추첨).

판정(env 별, 리셋마다):
  · 겹침  — 첫 `--early` 스텝 안에 컵의 xy 가 정착 소환점에서 `--disp_mm` 넘게 밀리거나, 컵 속도가 `--speed` 를 넘거나,
            보상 전용 접촉 센서(손가락 마디 `_3`·`_4`·tip · 손바닥)의 컵 접촉력이 `--touch_n` 을 넘으면. 센서가 없는 링크(근위 마디·
            손목·팔)와의 겹침은 PhysX 밀어내기로 컵이 움직이는 것으로 잡는다.
  · 상판 이상 — 정착 뒤 컵 원점 z 가 (상판 + 종별 원점 오프셋) 에서 `--z_mm` 넘게 어긋나거나 기울기가 `--tilt_deg` 를 넘으면.

    ../IsaacLab/isaaclab.sh -p scripts/reinforcement_learning/probes/probe_grasp_fj_t2r_spawn_check.py \
        --task open-short_r_grasp_fj_t2r_rand-lstm --num_envs 1024 --rounds 4 --headless
"""
from __future__ import annotations

import argparse
import json
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-short_r_grasp_fj_t2r_rand-lstm")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--rounds", type=int, default=4, help="전체 리셋 횟수(리셋마다 소환 재추첨)")
parser.add_argument("--settle", type=int, default=60, help="리셋 뒤 정착 스텝")
parser.add_argument("--early", type=int, default=5, help="겹침을 보는 첫 스텝 수")
parser.add_argument("--disp_mm", type=float, default=5.0)
parser.add_argument("--speed", type=float, default=0.2, help="m/s")
parser.add_argument("--touch_n", type=float, default=0.5)
parser.add_argument("--z_mm", type=float, default=5.0)
parser.add_argument("--tilt_deg", type=float, default=3.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym      # noqa: E402
import torch                 # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg   # noqa: E402

import openarm.tasks         # noqa: E402,F401
import openarm.agnostic.tasks.grasp_fj_t2r.config  # noqa: E402,F401

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=cfg).unwrapped
N, A, dev = env.num_envs, int(env.cfg.action_space), env.device
lo, hi = env._act_lo, env._act_hi
act = torch.zeros(N, A, device=dev)
act[:, 7:] = (2.0 * (env._hand_reset_q - lo) / (hi - lo).clamp(min=1e-6) - 1.0).clamp(-1.0, 1.0)
table_z = float(env.cfg.table_surface_z)

rows = []
for rnd in range(args.rounds):
    env.reset()
    origins = env.scene.env_origins
    spawn = env.object_spawn_pos.clone()                       # 정착 소환점(env-local)
    species = env._species_ids.clone()
    max_disp = torch.zeros(N, device=dev)
    max_speed = torch.zeros(N, device=dev)
    max_touch = torch.zeros(N, device=dev)
    # ★1차 실행에서 밀림 수백 mm·속도 0.16 m/s 인 사례가 나왔다 — 첫 스텝들 안에 에피소드가 끝나 컵이 **재소환**된 것이다.
    #   재소환 뒤 위치를 원래 소환점과 비교하면 가짜 밀림이 된다 → 리셋을 따로 세고 그 뒤로는 밀림을 재지 않는다.
    early_reset = torch.zeros(N, dtype=torch.bool, device=dev)
    # 소환 순간 컵에 가장 가까운 로봇 몸체(원점 기준, 컵 높이 띠 안) — 겹침이 어느 몸체인지
    _bp = env.robot.data.body_pos_w - origins.unsqueeze(1)
    _top = spawn[:, 2] + env._obj_origin_off + 0.03
    _dxy = (_bp[:, :, :2] - spawn[:, None, :2]).norm(dim=-1)
    _dxy = torch.where(_bp[:, :, 2] <= _top.unsqueeze(1), _dxy, torch.full_like(_dxy, 9.0))
    near_d, near_i = _dxy.min(dim=1)
    for step in range(args.settle):
        env.step(act)
        if step < args.early:
            early_reset |= env.episode_length_buf <= 1
            pos = env.object.data.root_pos_w - origins
            _d = (pos[:, :2] - spawn[:, :2]).norm(dim=-1)
            max_disp = torch.maximum(max_disp, torch.where(early_reset, max_disp, _d))
            max_speed = torch.maximum(max_speed, env.object.data.root_lin_vel_w.norm(dim=-1))
            links_f, palm_f = env._link_cup_forces()
            max_touch = torch.maximum(max_touch, torch.maximum(links_f.flatten(1).amax(dim=1), palm_f.reshape(N, -1).amax(dim=1)))
    pos = env.object.data.root_pos_w - origins
    q = env.object.data.root_quat_w                             # (w, x, y, z)
    tilt = torch.rad2deg(torch.acos((1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)).clamp(-1.0, 1.0)))
    z_err = pos[:, 2] - (table_z + env._obj_origin_off)
    palm = env.robot.data.body_pos_w[:, env.palm_idx] - origins
    overlap = (max_disp * 1e3 > args.disp_mm) | (max_speed > args.speed) | (max_touch > args.touch_n) | early_reset
    bad_table = (z_err.abs() * 1e3 > args.z_mm) | (tilt > args.tilt_deg)
    for i in range(N):
        rows.append(dict(round=rnd, env=i, cup=env._species_names[int(species[i])],
                         sx=float(spawn[i, 0]), sy=float(spawn[i, 1]),
                         disp_mm=float(max_disp[i]) * 1e3, speed=float(max_speed[i]), touch_n=float(max_touch[i]),
                         z_err_mm=float(z_err[i]) * 1e3, tilt=float(tilt[i]),
                         palm_cup_xy=float((palm[i, :2] - spawn[i, :2]).norm()), early_reset=bool(early_reset[i]),
                         near_body=env.robot.data.body_names[int(near_i[i])], near_xy=float(near_d[i]),
                         overlap=bool(overlap[i]), bad_table=bool(bad_table[i])))

n = len(rows)
xs = [r["sx"] for r in rows]
ys = [r["sy"] for r in rows]
ov = [r for r in rows if r["overlap"]]
bt = [r for r in rows if r["bad_table"]]
print(f"SPAWN_CHECK 표본 {n} (env {N} × 리셋 {args.rounds}) · 소환 x [{min(xs):.3f}, {max(xs):.3f}] · y [{min(ys):.3f}, {max(ys):.3f}]", flush=True)
print(f"SPAWN_CHECK 겹침 {len(ov)} ({len(ov) / n:.2%}) · 상판 이상 {len(bt)} ({len(bt) / n:.2%})", flush=True)


def _stat(key):
    v = sorted(r[key] for r in rows)
    return f"{key} p50 {v[n // 2]:.2f} · p99 {v[int(n * 0.99)]:.2f} · max {v[-1]:.2f}"


for k in ("disp_mm", "speed", "touch_n", "z_err_mm", "tilt"):
    print("SPAWN_CHECK " + _stat(k), flush=True)
# 5 cm 칸 지도: 칸별 표본 수 / 겹침 / 상판 이상
grid = {}
for r in rows:
    key = (math.floor((r["sx"] - 0.10) / 0.05), math.floor((r["sy"] + 0.30) / 0.05))
    c = grid.setdefault(key, [0, 0, 0])
    c[0] += 1
    c[1] += r["overlap"]
    c[2] += r["bad_table"]
print("SPAWN_CHECK 지도(행 y 위→아래 0.00→−0.30, 열 x 0.10→0.40) 칸 = 겹침/상판이상/표본", flush=True)
for gy in range(5, -1, -1):
    line = []
    for gx in range(6):
        c = grid.get((gx, gy), [0, 0, 0])
        line.append(f"{c[1]:>3d}/{c[2]:<3d}/{c[0]:<4d}")
    print(f"SPAWN_CHECK y[{-0.30 + 0.05 * gy:+.2f},{-0.25 + 0.05 * gy:+.2f}) " + " ".join(line), flush=True)
from collections import Counter
for lbl, sel in (("겹침 y<-0.20", [r for r in ov if r["sy"] < -0.20]), ("겹침 y>=-0.20", [r for r in ov if r["sy"] >= -0.20])):
    print(f"SPAWN_CHECK {lbl} {len(sel)} · 첫 스텝 내 리셋 {sum(r['early_reset'] for r in sel)} · 최근접 몸체 "
          + str(Counter(r["near_body"] for r in sel).most_common(5)), flush=True)
worst = sorted(ov + bt, key=lambda r: (-r["disp_mm"], -abs(r["z_err_mm"])))[:12]
for r in worst:
    print("SPAWN_CHECK 사례 " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()},
                                          ensure_ascii=False), flush=True)
env.close()

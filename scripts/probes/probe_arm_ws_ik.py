"""팔 워크스페이스 — **제약 IK** 판정판 (관절공간 분석의 정공법).

균일 관절 샘플 FK(probe_arm_ws_fk)는 side-to-side 자세가 7-DOF 안의 얇은
매니폴드라 2M 표본 중 55개(0.003%)만 걸렸다 — 지도가 아니라 샘플링 실패.
여기서는 각 (x,y,z) 그리드 점에 자세 제약(법선 수평·롤 연직)을 손실로 걸고
fabric FK 를 **autograd 로 미분**해 경사법 IK 를 배치로 푼다. 관절 한계는
q = lo + σ(w)·(hi−lo) 재매개화로 항상 만족. K 회 무작위 재시작.

성공 판정: 위치 < pos_tol ∧ 법선수평 < normal_tol ∧ 롤연직 < roll_tol 인
해가 하나라도 존재. 순수 기구학(충돌·물리 없음).
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--iters", type=int, default=400)
parser.add_argument("--restarts", type=int, default=8)
parser.add_argument("--pos_tol_mm", type=float, default=15.0)
parser.add_argument("--normal_tol_deg", type=float, default=15.0)
parser.add_argument("--roll_tol_deg", type=float, default=20.0)
parser.add_argument("--azimuth_tol_deg", type=float, default=30.0,
                    help="법선 방위 허용각 — ★사용자 명세(08.26): side-to-side 는 "
                         "우팔 palm_x→+world_y / 좌팔 palm_x→−world_y (사람이 컵을 "
                         "바깥에서 감싸는 자세). 방위 자유가 아니다")
parser.add_argument("--xs", type=str, default="0.06,0.10,0.14,0.18,0.22,0.26,0.30,0.34")
parser.add_argument("--ys", type=str,
                    default="-0.10,-0.06,-0.02,0.02,0.06,0.10,0.14,0.18,0.22,0.26,0.30,0.34,0.38,0.42",
                    help="팔쪽 부호로 자동 변환: 값 v → (v if 좌팔 else −v)")
parser.add_argument("--zs", type=str, default="0.278,0.32")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import math                                    # noqa: E402
import gymnasium as gym                        # noqa: E402
import torch                                   # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import openarm.tasks                           # noqa: E402,F401

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env_cfg.episode_length_s = 1.0e9
env = gym.make(args.task, cfg=env_cfg).unwrapped
dev = env.device
env.reset()

is_left = "_l_" in args.task
n_arm = env.profile.num_arm_joints
jl = env.robot.data.soft_joint_pos_limits[0]
lo = jl[env._fab_t[:n_arm], 0].clone()
hi = jl[env._fab_t[:n_arm], 1].clone()
q_home = env.fabric_q[:1].clone()

xs = [float(v) for v in args.xs.split(",")]
ys = [(float(v) if is_left else -float(v)) for v in args.ys.split(",")]
zs = [float(v) for v in args.zs.split(",")]
pts = torch.tensor([(x, y, z) for z in zs for y in ys for x in xs], device=dev)
P, K = pts.shape[0], args.restarts
tgt = pts.repeat_interleave(K, dim=0)                     # (P·K, 3)
B = tgt.shape[0]
# ★fabric taskmap 은 배치가 num_envs 로 고정이다(입력 (B,J)를 줘도 num_envs 행만
#   반환 — 실측 4 vs 1792 로 즉사). env 를 IK 배치 크기로 부팅해야 한다.
if args.num_envs != B:
    raise SystemExit(
        f"[중단] --num_envs {args.num_envs} ≠ IK 배치 {B} (그리드 {P} × 재시작 {K}).\n"
        f"       다시 실행: --num_envs {B}")

# 미분 가능성 확인 — fabric taskmap 이 autograd 를 지원하는지 fail-loud.
_qt = q_home.repeat(2, 1).requires_grad_(True)
_o, _R = env._palm_frame(_qt)
try:
    _o.sum().backward()
    assert _qt.grad is not None and torch.isfinite(_qt.grad).all()
except Exception as e:
    raise SystemExit(f"[중단] fabric FK 가 autograd 불가: {e}")
print("[확인] fabric FK autograd 가능 — 경사법 IK 진행", flush=True)

g = torch.Generator(device="cpu").manual_seed(3)
w = torch.randn(B, n_arm, generator=g).to(dev) * 1.5     # σ(w) 초기 분산
w.requires_grad_(True)
opt = torch.optim.Adam([w], lr=0.05)
sin_n = math.sin(math.radians(args.normal_tol_deg))
cos_r = math.cos(math.radians(args.roll_tol_deg))
cos_a = math.cos(math.radians(args.azimuth_tol_deg))
y_sign = -1.0 if is_left else 1.0            # 좌: −y 방향, 우: +y 방향

q_full = q_home.repeat(B, 1)
for it in range(args.iters):
    opt.zero_grad()
    q_arm = lo + torch.sigmoid(w) * (hi - lo)
    q = torch.cat([q_arm, q_full[:, n_arm:]], dim=1)
    o, R = env._palm_frame(q)
    l_pos = ((o - tgt) ** 2).sum(-1)
    l_n = R[:, 2, 0] ** 2                                  # 법선 수평: (x̂·ẑ)²
    l_r = (1.0 - R[:, 2, 1].abs()) ** 2                    # 롤 연직: |ŷ·ẑ|→1
    l_a = (1.0 - y_sign * R[:, 1, 0]) ** 2                 # 법선 방위: x̂·ŷ → ±1
    loss = (l_pos + 0.3 * l_n + 0.3 * l_r + 0.3 * l_a).sum()
    loss.backward()
    opt.step()
    if it % 100 == 99:
        print(f"  iter {it+1}/{args.iters}  pos_rmse "
              f"{float(l_pos.detach().mean().sqrt())*1000:.1f}mm", flush=True)

with torch.no_grad():
    q_arm = lo + torch.sigmoid(w) * (hi - lo)
    q = torch.cat([q_arm, q_full[:, n_arm:]], dim=1)
    o, R = env._palm_frame(q)
    perr = (o - tgt).norm(dim=-1) * 1000.0
    ok = ((perr < args.pos_tol_mm)
          & (R[:, 2, 0].abs() < sin_n) & (R[:, 2, 1].abs() > cos_r)
          & (y_sign * R[:, 1, 0] > cos_a))
    ok_p = ok.view(P, K).any(dim=1)
    best = perr.view(P, K).min(dim=1).values

side = "좌" if is_left else "우"
print("\n" + "=" * 96)
print(f"제약 IK 워크스페이스 — {side}팔 · 그리드 {P}점 × 재시작 {K} · "
      f"성공 {int(ok_p.sum())}/{P}")
print(f"  성공 = 위치<{args.pos_tol_mm:.0f}mm ∧ 법선수평<{args.normal_tol_deg:.0f}° "
      f"∧ 롤연직<{args.roll_tol_deg:.0f}° 해 존재 · ✓=성공, 숫자=최선 위치오차mm")
print("=" * 96)
i = 0
for z in zs:
    print(f"\n  z={z:.3f}   x→   " + "    ".join(f"{x:.2f}" for x in xs))
    # ★라벨은 **실제 world y** — 입력 ys 는 팔쪽 부호로 변환되므로(우팔 = −v)
    #   변환값 그대로 찍는다. 처음에 입력값을 찍어 우팔 지도의 부호가 뒤집혔었다.
    for yv in ys:
        row = []
        for x in xs:
            row.append("  ✓ " if bool(ok_p[i]) else f"{min(float(best[i]), 999):4.0f}")
            i += 1
        print(f"    y={yv:+.2f}  " + "  ".join(row))
print("=" * 96 + "\n", flush=True)
env.close(); app.close()

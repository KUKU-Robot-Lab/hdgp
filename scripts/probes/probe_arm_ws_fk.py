"""팔 관절공간 FK 워크스페이스 — side-to-side 자세 필터의 xy 점유 지도.

08.26 사용자 지시: "joint 값들이 가능한 workspace 분석으로 풀어야지."
동역학 정착 프로브는 3중 오염(정착 미달·중력 처짐·손-테이블 간섭)으로 지도가
흔들렸다. 이 프로브는 **순수 기구학**이다: 팔 관절을 soft limit 안에서 균일
대량 샘플 → fabric FK 로 palm 프레임 → 자세 필터 통과 표본의 위치를 z 층별
xy 격자에 점유 표시. 물리·수렴·충돌이 아예 없다.

자세 필터(side-to-side): |cos(palm_x, ẑ)| < sin(normal_tol) — 법선 수평
                         |cos(palm_y, ẑ)| > cos(roll_tol)   — 롤 축 연직
주의: 자기충돌·테이블 간섭은 안 본다(순수 도달 가능성). 테이블 간섭은
손 반폭(±77mm 수직 스택)을 z 에 더해 따로 판단할 것.
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--samples", type=int, default=2_000_000)
parser.add_argument("--batch", type=int, default=8192)
parser.add_argument("--normal_tol_deg", type=float, default=15.0)
parser.add_argument("--roll_tol_deg", type=float, default=20.0)
parser.add_argument("--z_slices", type=str, default="0.268:0.288,0.30:0.34",
                    help="z 하한:상한 쉼표 구분 — 첫 층 기본 = 컵 원점 ±10mm")
parser.add_argument("--cell", type=float, default=0.02)
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

n_arm = env.profile.num_arm_joints
jl = env.robot.data.soft_joint_pos_limits[0]           # (J,2) 로봇 순서
lo = jl[env._fab_t[:n_arm], 0]
hi = jl[env._fab_t[:n_arm], 1]
q0 = env.fabric_q[:1].clone()                          # (1,J) 홈(손은 홈 고정)
sin_n = math.sin(math.radians(args.normal_tol_deg))
cos_r = math.cos(math.radians(args.roll_tol_deg))
slices = []
for tok in args.z_slices.split(","):
    a, b = tok.split(":")
    slices.append((float(a), float(b)))

# xy 격자
X0, X1, Y0, Y1 = 0.0, 0.55, -0.55, 0.55
nx = int(round((X1 - X0) / args.cell))
ny = int(round((Y1 - Y0) / args.cell))
occ = [torch.zeros(ny, nx, dtype=torch.long, device=dev) for _ in slices]
pass_orient = 0
total = 0

B = args.batch
q = q0.repeat(B, 1)
rounds = max(1, args.samples // B)
for _ in range(rounds):
    u = torch.rand(B, n_arm, device=dev)
    q[:, :n_arm] = lo + u * (hi - lo)
    o, R = env._palm_frame(q)                          # (B,3) env-local, (B,3,3)
    ok = (R[:, 2, 0].abs() < sin_n) & (R[:, 2, 1].abs() > cos_r)
    total += B
    pass_orient += int(ok.sum())
    if not bool(ok.any()):
        continue
    p = o[ok]
    for si, (za, zb) in enumerate(slices):
        m = (p[:, 2] >= za) & (p[:, 2] < zb)
        if not bool(m.any()):
            continue
        ix = ((p[m, 0] - X0) / args.cell).long().clamp(0, nx - 1)
        iy = ((p[m, 1] - Y0) / args.cell).long().clamp(0, ny - 1)
        occ[si].index_put_((iy, ix), torch.ones_like(ix), accumulate=True)

side = "좌" if "_l_" in args.task else "우"
print("\n" + "=" * 90)
print(f"팔 FK 워크스페이스(순수 기구학) — {side}팔 · 표본 {total:,} · "
      f"자세 필터 통과 {pass_orient:,} ({100.0*pass_orient/total:.2f}%)")
print(f"  필터: 법선수평<{args.normal_tol_deg:.0f}° ∧ 롤연직<{args.roll_tol_deg:.0f}° · "
      f"셀 {args.cell*100:.0f}cm · 표시 = 그 셀 표본수(·=1~2, o=3~9, O=10~49, █=50+)")
print("=" * 90)
for si, (za, zb) in enumerate(slices):
    g = occ[si].cpu()
    tot_s = int(g.sum())
    print(f"\n── z ∈ [{za:.3f}, {zb:.3f})  (표본 {tot_s:,}) ──")
    ys_lab = [Y0 + (i + 0.5) * args.cell for i in range(ny)]
    xs_lab = [X0 + (i + 0.5) * args.cell for i in range(nx)]
    # 점유 행만 출력
    rows = [iy for iy in range(ny) if int(g[iy].sum()) > 0]
    if not rows:
        print("  (자세 필터를 통과한 표본이 이 층에 없음)")
        continue
    xcols = [ix for ix in range(nx) if int(g[:, ix].sum()) > 0]
    print("        x→ " + " ".join(f"{xs_lab[ix]:.2f}"[1:] for ix in xcols))
    for iy in rows:
        line = []
        for ix in xcols:
            v = int(g[iy, ix])
            line.append(" · " if 1 <= v <= 2 else (" o " if v <= 9 else (" O " if v <= 49 else " █ ")) if v else "   ")
        print(f"  y={ys_lab[iy]:+.2f} " + " ".join(c.strip() or " " for c in line))
    # 범위 요약(표본 3+ 셀 기준)
    solid = (g >= 3)
    if bool(solid.any()):
        iys, ixs = torch.nonzero(solid, as_tuple=True)
        print(f"  ★범위(셀 표본≥3): x {xs_lab[int(ixs.min())]:.2f}~{xs_lab[int(ixs.max())]:.2f} · "
              f"y {ys_lab[int(iys.min())]:+.2f}~{ys_lab[int(iys.max())]:+.2f}")
print("=" * 90 + "\n", flush=True)
env.close(); app.close()

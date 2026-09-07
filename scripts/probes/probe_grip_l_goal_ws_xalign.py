"""이송 목표 영역 적합성 — **tcp_z ∥ world +x** 제약 IK 판 (사용자 지시 08.27).

질문: "goal 을 임의로 정하지 말고, tcp_z 가 world +x 를 향할 때 **실제로 도달 가능한**
  공간 분포를 재서 그 안으로 goal 영역을 정하자."
구 판(probe_grip_l_spawn_ws_ik.py)은 접근축이 **수평이기만** 하면 통과였다(방위각 자유).
여기서는 방위각까지 묶는다 — 파지·이송 내내 같은 자세를 유지해야 하기 때문이다.

`probe_arm_ws_ik.py`(agnostic 직접 env 전용, `env._palm_frame`)의 grip_l 어댑터.
질문(사용자 08.26): **현 컵 스폰 영역이, gripper base 가 도달할 수 있으면서
tcp_z(접근축)를 수평으로 둘 수 있는 영역 안인가.**

방법: 각 후보 컵 위치 c 에 대해 fabric palm taskmap FK 를 autograd 미분,
  손실 = |jaw_mid(q) − c|² + w·(접근축 world-z 성분)²
  jaw_mid(q) = palm_o(q) + R_palm(q)·d_local  (d_local 은 홈에서 실측 — 눈대중 금지)
  접근축 = R_palm(q)·a_local                  (a_local 도 홈에서 실측)
관절은 q = lo + σ(w)(hi−lo) 재매개화로 한계 안. K 회 무작위 재시작.
성공 = 위치오차 < pos_tol ∧ 접근축 기울기 < tilt_tol(= U_perp 만점 10°).
추가로 성공 해의 palm 위치가 **PALM_BOX(액션 박스) 안**인지도 판정한다 —
기구학이 돼도 박스가 못 덮으면 정책은 지령을 낼 수 없다.

사용(그리드 P × 재시작 K = num_envs 필수):
  ./isaaclab.sh -p scripts/probes/probe_grip_l_spawn_ws_ik.py --num_envs <P*K>
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-grip_l_grasp_sensor_fab")
parser.add_argument("--num_envs", type=int, default=672)
parser.add_argument("--iters", type=int, default=400)
parser.add_argument("--restarts", type=int, default=8)
parser.add_argument("--pos_tol_mm", type=float, default=15.0)
parser.add_argument("--tilt_weight", type=float, default=0.3,
                    help="자세 항 가중치. 0 이면 **위치만** 푸는 자세 무관 지도")
parser.add_argument("--tilt_tol_deg", type=float, default=15.0,
                    help="tcp_z 와 world +x 의 허용 사잇각")
parser.add_argument("--xs", type=str, default="0.28,0.31,0.34,0.37,0.40,0.43,0.46")
parser.add_argument("--ys", type=str, default="0.09,0.13,0.17,0.21,0.25,0.29")
parser.add_argument("--zs", type=str, default="0.280,0.307")
parser.add_argument("--dump_path", type=str, default="",
                    help="성공 해(q7·palm o·euler_zyx·달성 jaw_mid)를 npz 로 저장")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import math                                    # noqa: E402
import gymnasium as gym                        # noqa: E402
import torch                                   # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import openarm.gripper.left.grasp_sensor       # noqa: E402,F401
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P  # noqa: E402

xs = [float(v) for v in args.xs.split(",")]
ys = [float(v) for v in args.ys.split(",")]
zs = [float(v) for v in args.zs.split(",")]
pts_list = [(x, y, z) for z in zs for y in ys for x in xs]
NP, K = len(pts_list), args.restarts
B = NP * K
if args.num_envs != B:
    raise SystemExit(f"[중단] --num_envs {args.num_envs} ≠ 그리드 {NP} × 재시작 {K} = {B}")

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=B)
env_cfg.episode_length_s = 1.0e9          # ★프로브 리셋 오염 금지 (이 트랙 4회 피해)
for t in ("time_out", "object_dropping", "object_out_of_workspace", "object_tipped"):
    if hasattr(env_cfg.terminations, t):
        setattr(env_cfg.terminations, t, None)
env_cfg.curriculum = None
env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()
dev = env.device
act = env.action_manager.get_term("arm_action")
fab = act._fabric
robot = env.scene["robot"]

jl = robot.data.soft_joint_pos_limits[0]
lo = jl[act._arm_joint_ids, 0].clone()
hi = jl[act._arm_joint_ids, 1].clone()
q_home = act._q_home.unsqueeze(0)              # (1,7)

def palm_frame(q: torch.Tensor):
    """fabric palm taskmap → (원점 (N,3), 회전 (N,3,3) — 열 = palm x/y/z 축)."""
    pts, _ = fab.get_taskmap("palm")(q, None)
    if pts.shape[1] < 18:
        raise SystemExit("[중단] palm taskmap 이 1점 모드 — 축 보조점이 없다")
    o = pts[:, :3]
    ax = torch.stack([
        torch.nn.functional.normalize(pts[:, 3:6] - o, dim=1),
        torch.nn.functional.normalize(pts[:, 9:12] - o, dim=1),
        torch.nn.functional.normalize(pts[:, 15:18] - o, dim=1),
    ], dim=-1)
    return o, ax

# ── 홈에서 (a_local, d_local) 실측 — "눈대중으로 상수를 넣지 않는다" ──────
# 홈에 정착시킨 뒤 articulation 실측 턱 프레임을 palm 프레임으로 옮긴다.
zero = torch.zeros(B, env.action_manager.total_action_dim, device=dev)
zero[:, 6:] = 1.0                               # ★isaaclab Binary: a≥0 = 열림 (t53 스모크에서 부호 확인)
for _ in range(120):
    env.step(zero)
jaw_ids = [robot.body_names.index(b) for b in P.GRIPPER_FINGER_BODIES]
org = env.scene.env_origins
jp = robot.data.body_pos_w[:, jaw_ids, :]
a_axis_w = matrix_from_quat(robot.data.body_quat_w[:, jaw_ids[0], :])[:, :, 2]
jaw_mid_w = (jp + (a_axis_w * P.JAW_PAD_OFFSET).unsqueeze(1)).mean(dim=1) - org

o_h, R_h = palm_frame(act._fabric_q)
a_local = torch.einsum("nji,nj->ni", R_h, a_axis_w).mean(0)
a_local = a_local / a_local.norm()
d_local = torch.einsum("nji,nj->ni", R_h, jaw_mid_w - o_h)
d_std = d_local.std(0) * 1000.0
d_local = d_local.mean(0)
print(f"[캘리브] a_local {a_local.tolist()}  d_local(mm) "
      f"{[round(float(v)*1000,1) for v in d_local]}  분산 {[round(float(v),2) for v in d_std]}")
if float(d_std.max()) > 6.0:
    print("★경고: d_local 분산 > 6mm — fabric 자세의존 오차 범위 밖, 결과 해석 주의")

# ── 미분 가능성 fail-loud ──────────────────────────────────────────────
_qt = q_home.repeat(2, 1).requires_grad_(True)
_o, _ = palm_frame(_qt)
_o.sum().backward()
assert _qt.grad is not None and torch.isfinite(_qt.grad).all(), "palm taskmap autograd 불가"
print("[확인] fabric FK autograd 가능 — 경사법 IK 진행", flush=True)

tgt = torch.tensor(pts_list, device=dev).repeat_interleave(K, dim=0)   # (B,3)
g = torch.Generator(device="cpu").manual_seed(3)
w = torch.randn(B, 7, generator=g).to(dev) * 1.5
w.requires_grad_(True)
opt = torch.optim.Adam([w], lr=0.05)
for it in range(args.iters):
    opt.zero_grad()
    q = lo + torch.sigmoid(w) * (hi - lo)
    o, R = palm_frame(q)
    a_w = torch.einsum("nij,j->ni", R, a_local)
    jm = o + torch.einsum("nij,j->ni", R, d_local)
    l_pos = ((jm - tgt) ** 2).sum(-1)
    # ★tcp_z 를 world +x 로 정렬: y·z 성분을 동시에 0 으로 민다(= a_w → (1,0,0)).
    #   구 판은 z 성분만 봐서 방위각이 자유였다(옆에서 잡든 뒤에서 잡든 통과).
    l_tilt = a_w[:, 1] ** 2 + a_w[:, 2] ** 2
    (l_pos + args.tilt_weight * l_tilt).sum().backward()
    opt.step()
    if it % 100 == 99:
        print(f"  iter {it+1}/{args.iters}  pos_rmse "
              f"{float(l_pos.detach().mean().sqrt())*1000:.1f}mm", flush=True)

with torch.no_grad():
    q = lo + torch.sigmoid(w) * (hi - lo)
    o, R = palm_frame(q)
    a_w = torch.einsum("nij,j->ni", R, a_local)
    jm = o + torch.einsum("nij,j->ni", R, d_local)
    perr = (jm - tgt).norm(dim=-1) * 1000.0
    # +x 축과의 사잇각. 0° = tcp_z 가 정확히 world +x.
    tilt = torch.rad2deg(torch.acos(a_w[:, 0].clamp(-1.0, 1.0)))
    in_box = ((o - act._box_center).abs() <= act._box_half).all(dim=-1)
    ok = (perr < args.pos_tol_mm) & (tilt < args.tilt_tol_deg)
    okb = ok & in_box
    ok_p = ok.view(NP, K).any(dim=1)
    okb_p = okb.view(NP, K).any(dim=1)
    best_err = perr.view(NP, K).min(dim=1).values
    best_tilt = torch.where(ok.view(NP, K), tilt.view(NP, K),
                            torch.full_like(tilt.view(NP, K), 99.0)).min(dim=1).values

print("\n" + "=" * 100)
print(f"목표 영역 적합성 IK (tcp_z ∥ world +x) — 그리드 {NP}점 × 재시작 {K} · 성공 {int(ok_p.sum())}/{NP}"
      f" (박스 안 {int(okb_p.sum())}/{NP})")
print(f"  ✓=성공(+x정렬∧위치, palm 지령 박스 안) · b=성공이나 박스 밖 · 숫자=최선 위치오차 mm")
print(f"  현 스폰: x∈[{P.CUP_SPAWN_X_CENTER-P.CUP_SPAWN_X_RANGE:.2f},"
      f"{P.CUP_SPAWN_X_CENTER+P.CUP_SPAWN_X_RANGE:.2f}] · "
      f"y∈[{P.CUP_SPAWN_Y_CENTER-P.CUP_SPAWN_Y_RANGE:.2f},"
      f"{P.CUP_SPAWN_Y_CENTER+P.CUP_SPAWN_Y_RANGE:.2f}] · 컵원점 z {P.CUP_SPAWN_Z:.3f}")
print("=" * 100)
i = 0
for z in zs:
    print(f"\n  z={z:.3f}   x→   " + "     ".join(f"{x:.2f}" for x in xs))
    for yv in ys:
        row = []
        for x in xs:
            if bool(okb_p[i]):
                row.append("  ✓ ")
            elif bool(ok_p[i]):
                row.append("  b ")
            else:
                row.append(f"{min(float(best_err[i]), 999):4.0f}")
            i += 1
        print(f"    y={yv:+.2f}  " + "  ".join(row))
print("=" * 100 + "\n", flush=True)

# ── 해 덤프 (fab_test57 pre-grasp 리셋 주입용) ────────────────────────────
if args.dump_path:
    import numpy as np
    with torch.no_grad():
        # euler_zyx 추출: R = Rz(ez)·Ry(ey)·Rx(ex)
        ey_ = torch.asin((-R[:, 2, 0]).clamp(-1.0, 1.0))
        ez_ = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        ex_ = torch.atan2(R[:, 2, 1], R[:, 2, 2])
        np.savez(
            args.dump_path,
            q=q.cpu().numpy(), palm_o=o.cpu().numpy(),
            euler_zyx=torch.stack([ez_, ey_, ex_], -1).cpu().numpy(),
            jaw_mid=jm.cpu().numpy(), target=tgt.cpu().numpy(),
            perr_mm=perr.cpu().numpy(), tilt_deg=tilt.cpu().numpy(),
            ok=ok.cpu().numpy(),
        )
    print(f"[덤프] {args.dump_path}  (성공 {int(ok.sum())}/{B})", flush=True)

env.close(); app.close()

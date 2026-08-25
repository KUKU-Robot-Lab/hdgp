"""DG-5F-S 손 **관절공간** 자기충돌 지도 — PhysX 실제 접촉으로 판정한다.

왜 관절공간인가: 액션공간(손끝 15D) 샘플은 fabric IK 를 거치므로 fabric 의 편향이
섞이고, 겹침 판정도 fabric 의 **충돌 구 근사**(반경 9mm)를 쓴다. 관절을 직접 명령하고
PhysX 접촉력을 읽으면 실제 콜라이더 메시(convexDecomposition) 기준의 진짜 겹침이 나온다.

검출 방법: 컵·지면을 멀리 치우면 손 링크에 남는 접촉은 **자기충돌뿐**이다.
`ContactSensor.data.net_forces_w`(필터 없는 총 접촉력)가 0 이 아니면 겹친 것이다.
※`force_matrix_w` 는 컵으로 필터된 값이라 여기서는 쓸 수 없다.

조건 대비 — "외전을 잠그면 교차가 막히는가"가 이 프로브의 1 순위 질문이다:
    free  : 손 관절 전부 자유 (현재 tip 모드 상태)
    noabd : 외전 `_1` 을 홈에 고정 (KUKA 의 PCA 가 하는 일과 같은 성격)

사용:
    isaaclab.sh -p .../probe_hand_selfcollision_jointspace.py --mode free
    isaaclab.sh -p .../probe_hand_selfcollision_jointspace.py --mode noabd
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--rounds", type=int, default=16)
parser.add_argument("--settle", type=int, default=3,
                    help="관절 텔레포트 후 물리 스텝. 접촉 갱신에 1~3 스텝이면 족하다.")
parser.add_argument("--mode", choices=["free", "noabd"], default="free")
parser.add_argument("--force_thr", type=float, default=0.5,
                    help="[N] 이 값을 넘으면 접촉으로 센다(수치 잡음 배제).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch                                       # noqa: E402
import gymnasium as gym                            # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg     # noqa: E402

import openarm.tasks                               # noqa: E402,F401
from openarm.agnostic.tasks.grasp_lift_fabric.grasp_lift_fabric_env_cfg import (  # noqa: E402
    resolve_cfg,
)

cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
cfg.enable_self_collisions = True          # ★이 프로브의 전제
cfg.enable_gravity = False
cfg.episode_length_s = 1.0e6               # ★리셋 오염 차단(프로브 필수 규약)
resolve_cfg(cfg)
env = gym.make(args.task, cfg=cfg).unwrapped
env.reset()
dev, N = env.device, args.num_envs
robot = env.robot

# ---- 손 관절 인덱스와 한계 ---------------------------------------------------------
HAND = env._hand_t                                   # (M,) 로봇 articulation 인덱스
lim = robot.data.soft_joint_pos_limits[0, HAND]      # (M, 2)
lo, hi = lim[:, 0], lim[:, 1]
names = [robot.joint_names[int(i)] for i in HAND]
home = env._default_q[0, HAND].clone()

# 외전(`_1`) 마스크 — 프로필의 손가락 이름으로 찾는다(로봇 이름 하드코딩 금지).
ABD = torch.tensor([n.endswith("_1") for n in names], device=dev)
print(f"[probe] 손 관절 {len(names)} · 외전(_1) {int(ABD.sum())}개 · mode={args.mode}",
      flush=True)

# ---- 컵·환경을 멀리 치운다 ----------------------------------------------------------
# ★안 치우면 컵 접촉이 자기충돌로 오계상된다.
_far = torch.zeros(N, 13, device=dev)
_far[:, 0] = 50.0
_far[:, 3] = 1.0
env.object.write_root_state_to_sim(_far)

# ---- 취합 버퍼 ----------------------------------------------------------------------
FING = list(env.profile.fingers)
F = len(FING)
n_hit = 0
tot = 0
per_finger = torch.zeros(F, device=dev)
# 겹친 표본의 관절값을 모아 "어느 관절 구간에서" 를 본다
hit_q, free_q = [], []

for r in range(args.rounds):
    q = robot.data.default_joint_pos.clone()
    u = torch.rand(N, len(names), device=dev)
    qh = lo + u * (hi - lo)
    if args.mode == "noabd":
        qh[:, ABD] = home[ABD]                       # 외전을 홈에 고정
    q[:, HAND] = qh
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)
    for _ in range(args.settle):
        env.sim.step(render=False)
        env.scene.update(env.physics_dt)

    # ★net_forces_w = 필터 없는 총 접촉력. 컵을 치웠으므로 남는 건 자기충돌뿐이다.
    f_per = torch.zeros(N, F, device=dev)
    for fi, fname in enumerate(FING):
        roles = env._sensors[fname]
        acc = torch.zeros(N, device=dev)
        for role in ("tip", "wrap"):
            for s_ in roles[role]:
                acc = acc + s_.data.net_forces_w.view(N, -1, 3).norm(dim=-1).sum(-1)
        f_per[:, fi] = acc

    hit = (f_per > args.force_thr).any(dim=1)
    n_hit += int(hit.sum())
    tot += N
    per_finger += (f_per > args.force_thr).float().sum(0)
    if hit.any():
        hit_q.append(qh[hit].clone())
    if (~hit).any():
        free_q.append(qh[~hit].clone())
    print(f"  round {r+1}/{args.rounds}: 누적 자기충돌 {n_hit}/{tot} "
          f"({100*n_hit/tot:.1f}%)", flush=True)

frac = n_hit / tot
print("\n" + "=" * 90)
print(f"손 관절공간 자기충돌 지도 — mode={args.mode} · 표본 {tot} · 임계 {args.force_thr}N")
print("=" * 90)
print(f"  ★자기충돌 비율  {frac*100:.1f}%   ({n_hit}/{tot})")
print("\n  손가락별 접촉 표본 수:")
for fi, fname in enumerate(FING):
    c = float(per_finger[fi])
    print(f"     {fname:8s} {int(c):7d}  ({100*c/tot:5.1f}% of 표본)")

if hit_q and free_q:
    H = torch.cat(hit_q)
    G = torch.cat(free_q)
    print("\n  관절별 — 겹친 표본 평균 vs 안 겹친 표본 평균 [deg] (차이 큰 순 8개):")
    dh = (H.mean(0) - G.mean(0)) * 180.0 / 3.141592653589793
    order = dh.abs().argsort(descending=True)[:8]
    for i in order:
        i = int(i)
        print(f"     {names[i]:18s} 차이 {float(dh[i]):+7.1f}°  "
              f"(겹침 {float(H[:, i].mean())*57.3:+7.1f}° · "
              f"자유 {float(G[:, i].mean())*57.3:+7.1f}°)")

print("\n  판정:", "액션 절단 불필요 (repulsion 으로 충분)" if frac < 0.05
      else "감시 필요" if frac < 0.20
      else "★구조적 차단 검토 (외전 고정 / 액션 박스 축소)")
print("=" * 90)
app.close()

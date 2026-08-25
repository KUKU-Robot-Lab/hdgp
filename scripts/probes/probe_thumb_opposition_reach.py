"""엄지 대향 도달 probe — **엄지 손끝과 4 지 손끝이 얼마나 가까워질 수 있는가**를 잰다.

왜: kuka2~kuka4 전 런에서 좌·우 가리지 않고 정책이 엄지를 버리고 4 지만으로 파지한다
(엄지측 접촉력 0.00N vs 4 지측 6.47N). 대향 게이트가 영구히 0 이라 lift·tracking·
success 가 죽고, 그 신호 고갈이 σ 팽창·붕괴로 이어진다.
[[synergy-grip-port]]: **도달 가능 최대치를 probe 로 먼저 재라** — 보상만 고치면
tip_cyl 처럼 9,000ep 를 버린다.

★1 판 실패 기록(같은 회차): 컵을 초기 파지중심에 고정하고 step 했더니 팔이 액션
  박스 중앙으로 이동하며 손이 컵을 떠나 **전 조건 0N** 이 나왔다. 접촉을 1 순위
  지표로 삼으면 이렇게 배치 하나로 측정이 통째로 죽는다. 그래서 여기서는 컵을 빼고
  **손끝 쌍 거리**를 직접 잰다 — 물체 배치·마찰·접촉필터에 의존하지 않는 값이다.

조건(전부 액션 → fabric → 물리 파이프라인 통과):
    home    : 홈 자세
    oppose  : 엄지를 4 지 쪽 **끝**, 4 지를 엄지 쪽 **끝** = 박스가 허용하는 최대 근접
    center  : 모든 손끝을 파지중심으로 = 보상(approach)이 실제로 요구하는 자세

판정: oppose 최소거리가 컵 지름(35~60mm)보다 크면 **박스 안에서 대향 파지 불가**.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--settle", type=int, default=60)
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
cfg.episode_length_s = 1.0e6
cfg.enable_adr = False
cfg.enable_physics_dr = False
resolve_cfg(cfg)
env = gym.make(args.task, cfg=cfg).unwrapped
env.reset()
dev, N, A = env.device, args.num_envs, cfg.action_space

FING = list(env.profile.fingers)
IA, IB = env._grp_a.tolist(), env._grp_b.tolist()
LO, HI = env._tip_lo, env._tip_hi                     # (F,3) palm 상대 [m]
HOME = env._tip_home
GC = env._grasp_center_local                          # (3,)

print(f"\n[probe] 대향 그룹 A={[FING[i] for i in IA]} · B={[FING[i] for i in IB]}")
print("[probe] 액션 박스 z 범위[mm] — 엄지와 4 지가 **겹치는 구간이 있는가**:")
za_lo, za_hi = float(LO[IA, 2].min()) * 1000, float(HI[IA, 2].max()) * 1000
zb_lo, zb_hi = float(LO[IB, 2].min()) * 1000, float(HI[IB, 2].max()) * 1000
print(f"    엄지 z[{za_lo:7.1f},{za_hi:7.1f}]   4지 z[{zb_lo:7.1f},{zb_hi:7.1f}]")
gap = zb_lo - za_hi
print(f"    ★z 축 간격 {gap:+.1f}mm  ({'겹침 없음 = 대향 불가' if gap > 0 else '겹침 있음'})")
print(f"    파지중심 palm+({GC[0]*1000:.0f},{GC[1]*1000:.0f},{GC[2]*1000:.0f})mm "
      f"— 엄지 z 상한 {za_hi:.0f} / 4지 z 하한 {zb_lo:.0f} 사이에 있는가?", flush=True)


def _u_from_local(t: torch.Tensor) -> torch.Tensor:
    """palm 상대 목표(F,3) → 정규화 액션(F,3). 박스 밖은 clamp(=도달 불가의 일부)."""
    span = (HI - LO).clamp(min=1e-6)
    return (2.0 * (t - LO) / span - 1.0).clamp(-1.0, 1.0)


def build(mode: str) -> torch.Tensor:
    a = torch.zeros(N, A, device=dev)
    if mode == "home":
        u = _u_from_local(HOME)
    elif mode == "center":
        u = _u_from_local(GC[None, :].expand(len(FING), 3))
    elif mode == "oppose":
        # 엄지는 박스 안에서 4 지 쪽(z 최대), 4 지는 엄지 쪽(z 최소)으로 최대한.
        # x·y 는 파지중심에 맞춰 서로를 향하게 둔다.
        t = HOME.clone()
        t[:, 0] = GC[0]
        t[:, 1] = GC[1]
        t[IA, 2] = HI[IA, 2]
        t[IB, 2] = LO[IB, 2]
        u = _u_from_local(t.clamp(LO, HI))
    else:
        raise ValueError(mode)
    a[:, 6:] = u.reshape(-1)[None, :].expand(N, -1)
    return a


def run(mode: str) -> dict:
    env.reset()
    act = build(mode)
    for _ in range(args.settle):
        env.step(act)
    tips = env.robot.data.body_pos_w[:, env._tip_t]          # (N,F,3) world
    # 엄지 ↔ 4 지 각각의 거리
    d = (tips[:, IA][:, :, None, :] - tips[:, IB][:, None, :, :]).norm(dim=-1)  # (N,|A|,|B|)
    per_b = d.mean(dim=(0, 1))                                # 4 지별 평균
    # 지시한 목표에 실제로 도달했는가(액션이 무효인지 가리는 값)
    from isaaclab.utils.math import matrix_from_quat
    o = env.robot.data.body_pos_w[:, env._tcp_idx]
    R = matrix_from_quat(env.robot.data.body_quat_w[:, env._tcp_idx])
    tip_local = torch.einsum("bji,bkj->bki", R, tips - o[:, None, :])
    span = (HI - LO).clamp(min=1e-6)
    want = LO + 0.5 * (act[0, 6:].view(-1, 3) + 1.0) * span
    err = (tip_local - want[None]).norm(dim=-1).mean(0)        # (F,)
    return {"mode": mode, "dmin": float(d.min(dim=2).values.min(dim=1).values.mean()),
            "per_b": per_b, "err": err}


rows = [run(m) for m in ("home", "oppose", "center")]

print("\n" + "=" * 88)
print(f"엄지↔4지 손끝 거리 — task={args.task} · 표본 {N}")
print("=" * 88)
print(f"{'조건':<9}{'★최소거리[mm]':>15}" + "".join(f"{FING[i]:>10}" for i in IB))
for r in rows:
    print(f"{r['mode']:<9}{r['dmin']*1000:>15.1f}"
          + "".join(f"{v*1000:>10.1f}" for v in r["per_b"]))
print("\n지시↔실제 손끝 오차[mm] (크면 그 지시가 도달 불가라는 뜻):")
print(f"{'조건':<9}" + "".join(f"{f:>10}" for f in FING))
for r in rows:
    print(f"{r['mode']:<9}" + "".join(f"{v*1000:>10.1f}" for v in r["err"]))

opp = next(r for r in rows if r["mode"] == "oppose")["dmin"] * 1000
print("\n판정:")
print(f"  박스가 허용하는 **최대 근접** 엄지↔4지 거리 = {opp:.1f}mm")
if opp > 60.0:
    print("  ★★대향 파지 불가 — 컵 지름(35~60mm)보다 멀다. 손가락이 컵을 사이에 두고")
    print("    마주볼 수 없으므로 대향 게이트는 어떤 보상으로도 열리지 않는다.")
    print("    → 액션 박스(tip_action_span_frac / tip_workspace_quantile) 또는")
    print("      파지중심 정의를 고쳐야 한다. 보상 수정은 무효다.")
elif opp > 35.0:
    print("  ★가는 물체만 가능 — 큰 컵은 못 잡는다. 박스 확장 검토.")
else:
    print("  대향 가능 — 병목은 다른 곳이다(탐색/보상).")
print("=" * 88)
app.close()

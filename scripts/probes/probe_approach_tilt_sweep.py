"""P1 — 접근 기울기 τ 스윕: 어떤 자세로 접근해야 물체가 잡히는가?

## 왜 이 probe 가 학습보다 먼저인가

두 개의 독립된 병목이 확인됐다.
  (1) 회전 박스가 side 를 원천 차단한다. G 규약에서 palm 법선은
        normal = [cos(ex)·sin(ey),  -sin(ex),  cos(ex)·cos(ey)]
      이고 top-down(ex=180°)과 side(ex=270°)는 **ex 축 하나에서 90° 차이**다.
      현행 박스(ex∈[135,225], ey∈±45)의 도달 천장은
        tilt_max = acos(-cos135°·cos45°) = 60°
      LEFT 성공 실측이 54° + 회전 3축 전부 포화 — 천장에 밀착해 있었다.
  (2) finger_curl_reg 페널티가 per-finger 에서 envelope 을 처벌한다
      (굽힘 -0.49 vs h2o 이득 +0.28). 실측: LEFT 는 엄지를 -0.958 로 펴고 있다.

이 probe 는 **정책도 보상도 쓰지 않는다.** 스크립트로 palm 을 놓고 손을 강제로 닫아
물리적으로 잡히는지만 본다. 따라서 (2)와 무관하게 **(1) 기하만 순수 판정**한다.

## 판정 (go/no-go)

  GO   : 각 물체 클래스(특히 FLAT — 테이블에 붙은 납작한 것)에서 리프트 > 5cm 를 내는 τ 가 존재
         → 회전 박스 통합(ex 중심 225°)으로 학습 진행. 그 τ* 가 pregrasp 규칙의 데이터.
  NO-GO: FLAT 에서 어떤 τ 에서도 리프트 < 5cm
         → 접근 자세로는 못 푸는 문제. 학습 돌리기 전에 메커니즘(마찰·폐쇄궤적)으로 재정의.

## 측정

palm 을 물체 기준으로 (τ, ez) 자세에 놓는다:
    n̂        = 법선(τ=0 → (0,0,-1) top-down,  τ=90 → (0,±1,0) side)
    support  = Σ_j |(R_objᵀ n̂)_j| · half_j      # n̂ 방향 물체 반경
    palm_pos = obj − n̂ · (support + standoff)
그리고 손을 FULL_GRIP 까지 닫고 20cm 들어올려 object_height 를 잰다.

물체는 안착 높이 h = 2·support(obj, ẑ) 로 FLAT / MID / TALL 분류해 집계한다.

사용:
  ./isaaclab.sh -p scripts/probes/probe_approach_tilt_sweep.py --task open-tesol_r_grasp_v2-lstm
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-tesol_r_grasp_v2-lstm")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--standoff", type=float, default=0.06)
parser.add_argument("--repeats", type=int, default=2, help="τ 당 reset 반복(물체 표본 확대)")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import math  # noqa: E402
from collections import defaultdict  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402

_OUT = open("/tmp/probe_approach_tilt_sweep.txt", "w")
_p = print


def print(*a, **kw):  # noqa: A001
    _p(*a, **kw, flush=True)
    _p(*a, **kw, file=_OUT, flush=True)


env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()

n = env.num_envs
D = env.device
IS_LEFT = "_l_" in args.task
SIDE_SIGN = -1.0 if IS_LEFT else +1.0        # side 법선의 y 부호 (right=+Y, left=-Y)
TABLE_Z = 0.200
LIFT_OK = 0.05                                # 성공 판정: 5cm


def support(half: torch.Tensor, R: torch.Tensor, nhat: torch.Tensor) -> torch.Tensor:
    """n̂ 방향 물체 반경 = Σ_j |(Rᵀ n̂)_j| · half_j.  half (N,3), R (N,3,3), nhat (N,3)."""
    local = torch.bmm(R.transpose(1, 2), nhat.unsqueeze(-1)).squeeze(-1)   # (N,3)
    return (local.abs() * half).sum(dim=-1)


def normal_from(tau_deg: float, ez_deg: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(τ, ez) → G-euler(ez,ey,ex) 와 그때의 palm 법선.

    ex = 180° + SIDE_SIGN·τ  →  normal_z = -cos(τ), normal_y = SIDE_SIGN·sin(τ)
    즉 τ=0 이 순수 top-down, τ=90 이 순수 side.
    ez 는 그 접근의 방위각(수평성분 회전).
    """
    ex = math.radians(180.0 + SIDE_SIGN * tau_deg)
    ey = 0.0
    ez = math.radians(ez_deg)
    # normal = [cos(ex)·sin(ey), -sin(ex), cos(ex)·cos(ey)] 을 ez 로 수평회전
    nx0 = math.cos(ex) * math.sin(ey)
    ny0 = -math.sin(ex)
    nz = math.cos(ex) * math.cos(ey)
    nx = nx0 * math.cos(ez) - ny0 * math.sin(ez)
    ny = nx0 * math.sin(ez) + ny0 * math.cos(ez)
    euler = torch.tensor([ez, ey, ex], device=D).unsqueeze(0).expand(n, -1)
    nhat = torch.tensor([nx, ny, nz], device=D).unsqueeze(0).expand(n, -1)
    return euler, nhat


def trial(tau_deg: float, ez_deg: float):
    env.reset()
    zero = torch.zeros(n, env.cfg.num_actions, device=D)
    for _ in range(int(env.cfg.settle_steps) + 2):
        env.step(zero)

    obj0 = env.object_pos.clone()
    R_obj = matrix_from_quat(env.object_rot)
    half = env.object_half_extent[env.object_idx]                    # (N,3)
    oid = env.object_idx.clone()

    # 안착 높이 h = 2·support(obj, ẑ)  → 물체 클래스
    zhat = torch.tensor([0.0, 0.0, 1.0], device=D).unsqueeze(0).expand(n, -1)
    h_settled = 2.0 * support(half, R_obj, zhat)                     # (N,)

    euler, nhat = normal_from(tau_deg, ez_deg)
    sup = support(half, R_obj, nhat)                                 # (N,)
    palm_tgt = obj0 - nhat * (sup + args.standoff).unsqueeze(-1)     # (N,3)

    # 테이블 가드 — 손이 상판을 뚫으면 depenetration 폭주(전례 -4.9e7)
    clamped = palm_tgt[:, 2] < (TABLE_Z + 0.02)
    palm_tgt[:, 2] = palm_tgt[:, 2].clamp(min=TABLE_Z + 0.02)

    tgt = torch.cat([palm_tgt, euler], dim=1)                        # (N,6)
    tgt = torch.max(torch.min(tgt, env.palm_maxs_env), env.palm_mins_env)

    lo, hi = env.palm_mins_env, env.palm_maxs_env
    act = torch.zeros(n, env.cfg.num_actions, device=D)
    act[:, :6] = (2.0 * (tgt - lo) / (hi - lo + 1e-9) - 1.0).clamp(-1.0, 1.0)
    act[:, 6:11] = -1.0                                              # 손 개방으로 접근
    for _ in range(100):
        env.step(act)

    ik_err = (env.palm_center_pos - tgt[:, :3]).norm(dim=-1)         # 명령 대비 도달 오차

    act[:, 6:11] = 1.0                                               # per-finger 전 손가락 폐쇄
    for _ in range(140):
        env.step(act)

    tipd = (env.fingertip_pos - env.object_pos.unsqueeze(1)).norm(dim=-1).min(dim=1).values
    grip = (env.binary_contact_buf | env.middle_binary_contact_buf
            | env.distal_binary_contact_buf).sum(dim=-1).float()     # 물체 접촉만(필터 복원됨)

    tu = tgt.clone()
    tu[:, 2] = torch.clamp(tgt[:, 2] + 0.20, max=env.palm_maxs_env[:, 2])
    act[:, :6] = (2.0 * (tu - lo) / (hi - lo + 1e-9) - 1.0).clamp(-1.0, 1.0)
    for _ in range(140):
        env.step(act)

    lift = env.object_pos[:, 2] - obj0[:, 2]                         # (N,)
    return dict(lift=lift, grip=grip, tipd=tipd, ik=ik_err, h=h_settled,
                clamp=clamped.float(), oid=oid)


TAUS = (0, 15, 30, 45, 60, 75, 90)
EZS = (0.0,)          # 방위각은 1차 스윕에서 고정 (스폰 반경 ±0.06m 는 좁다)

print("=" * 96)
print("P1 — 접근 기울기 τ 스윕  (%s)" % args.task)
print("  τ=0 순수 top-down  ·  τ=90 순수 side  ·  현행 박스 도달 천장 = 60°")
print("  정책/보상 미개입. 스크립트로 palm 배치 → 손 강제 폐쇄 → 20cm 상승.")
print("  standoff = %.3f m,  성공 판정 = 리프트 > %.0fcm" % (args.standoff, LIFT_OK * 100))
print("=" * 96)

# 클래스 경계: 안착 높이 h
FLAT_MAX, MID_MAX = 0.06, 0.12
acc = defaultdict(list)          # (tau, cls) -> lift 들
extra = defaultdict(list)        # (tau, cls) -> (grip, tipd, ik, clamp)

for tau in TAUS:
    for ez in EZS:
        for _ in range(args.repeats):
            r = trial(float(tau), ez)
            h = r["h"]
            cls = torch.where(h < FLAT_MAX, 0, torch.where(h < MID_MAX, 1, 2))
            for c in (0, 1, 2):
                m = cls == c
                if not m.any():
                    continue
                acc[(tau, c)].append(r["lift"][m].cpu())
                extra[(tau, c)].append(torch.stack([
                    r["grip"][m], r["tipd"][m], r["ik"][m], r["clamp"][m]
                ], dim=1).cpu())

CLS = {0: "FLAT (h<6cm)", 1: "MID (6~12cm)", 2: "TALL (>12cm)"}
print("\n[리프트 cm — 클래스별 평균 / 성공률(>5cm)]")
print("  %-6s" % "τ" + "".join("%26s" % CLS[c] for c in (0, 1, 2)))
best = {0: (None, -9), 1: (None, -9), 2: (None, -9)}
for tau in TAUS:
    row = "  %-6d" % tau
    for c in (0, 1, 2):
        if (tau, c) not in acc:
            row += "%26s" % "-"
            continue
        lf = torch.cat(acc[(tau, c)])
        m = lf.mean().item() * 100
        sr = (lf > LIFT_OK).float().mean().item()
        if m > best[c][1]:
            best[c] = (tau, m)
        mark = "*" if sr > 0.1 else " "
        row += "%20s%s" % ("%6.1f cm  (%.0f%%)" % (m, sr * 100), mark)
    print(row)

print("\n[진단 — 클래스별 최적 τ 에서]")
print("  %-14s %6s %8s %10s %10s %8s" % ("클래스", "τ*", "리프트", "grip", "손끝~물체", "IK오차"))
for c in (0, 1, 2):
    tau = best[c][0]
    if tau is None or (tau, c) not in extra:
        continue
    e = torch.cat(extra[(tau, c)])
    lf = torch.cat(acc[(tau, c)])
    print("  %-14s %6d %7.1fcm %10.2f %10.3f %8.4f   (테이블클램프 %.0f%%)"
          % (CLS[c], tau, lf.mean() * 100, e[:, 0].mean(), e[:, 1].mean(),
             e[:, 2].mean(), e[:, 3].mean() * 100))

print("\n[판정]")
flat_best = best[0]
if flat_best[0] is not None and flat_best[1] > LIFT_OK * 100:
    print("  GO   — FLAT 물체가 τ=%d° 에서 %.1fcm 들린다." % flat_best)
    print("         → 회전 박스 통합(ex 중심 225°)으로 학습 진행. 이 τ* 가 pregrasp 규칙의 데이터.")
else:
    print("  NO-GO — FLAT 물체가 어떤 τ 에서도 %.0fcm 를 못 넘는다 (최고 %.1fcm @ τ=%s)."
          % (LIFT_OK * 100, flat_best[1], flat_best[0]))
    print("         → 접근 자세로는 못 푸는 문제. 학습 전에 메커니즘(마찰·폐쇄궤적·스쿠프)으로 재정의.")
print("  참고: 현행 박스는 τ ≤ 60° 만 도달 가능. τ*>60° 이면 박스 통합이 **필수**다.")

_OUT.close()
env.close()
app.close()

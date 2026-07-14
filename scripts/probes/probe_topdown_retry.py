"""올바른 시너지 부호로 top-down 을 재판정한다.

지금까지의 probe 는 전부 act[:, 6] = +1 을 "손 폐쇄"로 가정했다. **틀렸다.**
LEFT 정책이 실제로 물체를 드는 순간(89119 샘플, 평균 리프트 18.2cm)의 시너지 action 은

    [-1.000, +0.979, -0.997, -0.999, +0.975]      ← PC1 이 -1 이다

즉 부호가 정반대였다. 열 번의 probe 가 전부 이 오염 위에 서 있었고,
"top-down 파지는 5cm 를 못 넘는다"는 결론도 마찬가지다. 다시 잰다.

비교:
  SYN_SUCCESS  — LEFT 성공 조합 그대로
  SYN_PC1NEG   — PC1 = -1 만 (나머지 0)
  SYN_OLD      — 내가 쓰던 PC1 = +1 (오염된 baseline. 재현 확인용)
  LERP         — per-finger lerp (grasp_v1 방식, 형상 adaptive). 5손가락 동일 계수로 폐쇄.

각 조합 × palm 자세(top-down / LEFT 성공 자세) × 물체 상대위치.

사용:
  ./isaaclab.sh -p scripts/probes/probe_topdown_retry.py --task open-tesol_r_grasp_v2-lstm
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-tesol_r_grasp_v2-lstm")
parser.add_argument("--num_envs", type=int, default=128)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

_OUT = open("/tmp/probe_topdown_retry.txt", "w")
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
_MY = -1.0 if IS_LEFT else 1.0        # LEFT 성공값은 왼손 기준 — 오른손은 y 미러

OPEN = env.hand_open_pose.clone()
GRIP = env.hand_full_grip_pose.clone()
BASIS0 = env.hand_synergy_basis.clone()
ANCHOR0 = env.hand_synergy_anchor.clone()
MINS0 = env.hand_synergy_mins.clone()
MAXS0 = env.hand_synergy_maxs.clone()

# per-finger lerp 를 같은 progress 틀로 재현: basis 행 i = 손가락 i 의 (grip-open)
LERP_BASIS = torch.zeros(5, 20, device=D)
_d = GRIP - OPEN
for i in range(5):
    LERP_BASIS[i, 4 * i: 4 * i + 4] = _d[4 * i: 4 * i + 4]

SYN = {
    "SYN_SUCCESS(LEFT실측)": [-1.000, +0.979, -0.997, -0.999, +0.975],
    "SYN_PC1NEG(-1,0,0,0,0)": [-1.0, 0.0, 0.0, 0.0, 0.0],
    "SYN_OLD(내 오염 baseline)": [+1.0, 0.0, 0.0, 0.0, 0.0],
}


def run(syn_vec, palm_euler_deg, dxyz, lerp=False):
    """palm 을 물체 기준 dxyz + 지정 자세에 두고 → syn_vec 로 폐쇄 → 20cm 상승."""
    if lerp:
        env.hand_synergy_basis.copy_(LERP_BASIS)
        env.hand_synergy_anchor.copy_(OPEN)
        env.hand_synergy_mins.copy_(torch.zeros(5, device=D))
        env.hand_synergy_maxs.copy_(torch.ones(5, device=D))
    else:
        env.hand_synergy_basis.copy_(BASIS0)
        env.hand_synergy_anchor.copy_(ANCHOR0)
        env.hand_synergy_mins.copy_(MINS0)
        env.hand_synergy_maxs.copy_(MAXS0)

    env.reset()
    zero = torch.zeros(n, env.cfg.num_actions, device=D)
    for _ in range(int(env.cfg.settle_steps) + 2):
        env.step(zero)

    obj0 = env.object_pos.clone()
    tgt = torch.zeros(n, 6, device=D)
    tgt[:, 0] = obj0[:, 0] + dxyz[0]
    tgt[:, 1] = obj0[:, 1] + dxyz[1] * _MY
    tgt[:, 2] = obj0[:, 2] + dxyz[2]
    for k in range(3):
        tgt[:, 3 + k] = math.radians(palm_euler_deg[k])
    tgt = torch.max(torch.min(tgt, env.palm_maxs_env), env.palm_mins_env)

    lo, hi = env.palm_mins_env, env.palm_maxs_env
    act = torch.zeros(n, env.cfg.num_actions, device=D)
    act[:, :6] = (2.0 * (tgt - lo) / (hi - lo + 1e-9) - 1.0).clamp(-1.0, 1.0)
    act[:, 6:11] = -1.0 if not lerp else -1.0
    for _ in range(90):
        env.step(act)

    act[:, 6:11] = torch.tensor(syn_vec, device=D).unsqueeze(0)
    for _ in range(120):
        env.step(act)
    g = (env.binary_contact_buf | env.middle_binary_contact_buf
         | env.distal_binary_contact_buf).sum(dim=-1).float().mean()

    tu = tgt.clone()
    tu[:, 2] = torch.clamp(tgt[:, 2] + 0.20, max=env.palm_maxs_env[:, 2])
    act[:, :6] = (2.0 * (tu - lo) / (hi - lo + 1e-9) - 1.0).clamp(-1.0, 1.0)
    for _ in range(120):
        env.step(act)
    return g, (env.object_pos[:, 2] - obj0[:, 2]).mean() * 100


# palm 자세 2종 (G 규약 euler deg)
POSES = {
    "top-down(0,0,180)": ([0.0, 0.0, 180.0], (0.0, 0.0, 0.10)),
    "LEFT성공자세근사":   ([-40.0, -40.0, 140.0], (-0.013, 0.105, 0.015)),
}

print("=" * 92)
print("올바른 시너지 부호로 top-down 재판정 — %s" % args.task)
print("  LEFT 성공 시너지 = [-1.000, +0.979, -0.997, -0.999, +0.975]  (PC1 이 -1)")
print("  내 이전 probe 는 PC1 = +1 을 폐쇄로 썼다 — 부호가 정반대였다.")
print("=" * 92)

for pname, (peuler, pdxyz) in POSES.items():
    print("\n[palm 자세: %s]   물체기준 offset = (%+.3f, %+.3f, %+.3f)"
          % (pname, *pdxyz))
    print("  %-28s %12s %10s" % ("시너지 조합", "grip", "리프트cm"))
    for sname, sv in SYN.items():
        g, lf = run(sv, peuler, pdxyz, lerp=False)
        print("  %-28s %12.2f %10.1f%s" % (sname, g, lf, " *" if lf > 3 else ""))
    for a in (0.5, 1.0):
        g, lf = run([a] * 5, peuler, pdxyz, lerp=True)
        print("  %-28s %12.2f %10.1f%s" % ("LERP(5지 동일 %.1f)" % a, g, lf, " *" if lf > 3 else ""))

print("\n  * = 리프트 3cm 초과.")
print("  top-down 에서 SYN_SUCCESS 나 LERP 가 뜨면 → top-down 은 가능했고 내 부호가 틀렸던 것.")
print("  top-down 에서 전부 안 뜨고 LEFT성공자세에서만 뜨면 → 자세(회전)가 결정적이다.")

_OUT.close()
env.close()
app.close()

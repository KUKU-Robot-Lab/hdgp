"""손가락 굽힘 방향을 palm 로컬 프레임에서 측정한다.

손가락 굽힘은 palm 로컬 관계이므로 palm 이 어느 방향을 보든 로컬 이동 벡터는
같아야 한다. world 좌표로 재면 palm 자세에 따라 달라 보여 해석을 그르친다.

palm 로컬 축 (tesollo): +X = 손바닥 법선, +Z = 손가락 방향.
정상 굽힘이면 손끝이
    -Z_local (짧아짐) + **+X_local (법선 = 손바닥 쪽)**
으로 움직여야 물체를 감쌀 수 있다. +X 성분이 음수면 손등 쪽으로 젖혀지는 것이다.

side / top-down 두 자세에서 같은 값이 나와야 정상이다.

사용:
  ./isaaclab.sh -p scripts/probes/probe_curl_local.py --task open-tesol_r_grasp_v2-lstm
  ./isaaclab.sh -p scripts/probes/probe_curl_local.py --task open-tesol_l_grasp_v2-lstm
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-tesol_r_grasp_v2-lstm")
parser.add_argument("--num_envs", type=int, default=32)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402

_OUT = open("/tmp/probe_curl_local.txt", "w")
_p = print


def print(*a, **kw):  # noqa: A001
    _p(*a, **kw, flush=True)
    _p(*a, **kw, file=_OUT, flush=True)


env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()

n = env.num_envs
FING = ("thumb", "index", "middle", "ring", "pinky")


def palm_R():
    q = env.robot.data.body_quat_w[:, env.palm_body_index]
    return matrix_from_quat(q)                      # (n,3,3)  열 = palm 로컬축의 world 표현


def measure(pose_name, g_euler_deg, abduction):
    """지정한 palm 자세로 옮긴 뒤 손을 닫고, 손끝 이동을 palm 로컬로 변환해 본다."""
    env.reset()
    zero = torch.zeros(n, env.cfg.num_actions, device=env.device)
    for _ in range(int(env.cfg.settle_steps) + 2):
        env.step(zero)

    obj = env.object_pos.clone()
    tgt = torch.zeros(n, 6, device=env.device)
    tgt[:, :3] = obj
    tgt[:, 2] += 0.08                                # 물체 위 8cm (top-down 기준)
    for k in range(3):
        tgt[:, 3 + k] = math.radians(g_euler_deg[k])
    tgt = torch.max(torch.min(tgt, env.palm_maxs_env), env.palm_mins_env)

    lo, hi = env.palm_mins_env, env.palm_maxs_env
    act = torch.zeros(n, env.cfg.num_actions, device=env.device)
    act[:, :6] = (2.0 * (tgt - lo) / (hi - lo + 1e-9) - 1.0).clamp(-1.0, 1.0)
    act[:, 6:11] = -1.0                              # 손 개방
    act[:, 11:15] = abduction
    for _ in range(90):
        env.step(act)

    R0 = palm_R()
    p0 = env.palm_center_pos.clone()
    t0 = env.fingertip_pos.clone()

    act[:, 6:11] = 1.0                               # 손 폐쇄
    for _ in range(120):
        env.step(act)

    t1 = env.fingertip_pos.clone()
    p1 = env.palm_center_pos.clone()

    # 손끝 이동을 palm 로컬로: d_local = R0ᵀ · (tip1 - palm1) - R0ᵀ · (tip0 - palm0)
    rel0 = torch.bmm(R0.transpose(1, 2), (t0 - p0.unsqueeze(1)).transpose(1, 2)).transpose(1, 2)
    rel1 = torch.bmm(R0.transpose(1, 2), (t1 - p1.unsqueeze(1)).transpose(1, 2)).transpose(1, 2)
    d = (rel1 - rel0) * 100                          # cm, (n,5,3)

    print("\n[%s]  G-euler %s, abduction %+.1f" % (pose_name, g_euler_deg, abduction))
    print("  palm 로컬 이동 (dX=법선쪽 +, dY, dZ=손가락방향 +)")
    print("  %-8s %-30s %s" % ("손가락", "(dX, dY, dZ) cm", "판정"))
    for k in range(5):
        v = d[:, k, :].mean(dim=0)
        ok = "손바닥 쪽 ✓" if v[0] > 1.0 else ("손등 쪽 ✗" if v[0] < -1.0 else "평면 내")
        print("  %-8s (%+6.1f, %+6.1f, %+6.1f)          %s" % (FING[k], v[0], v[1], v[2], ok))
    return d


print("=" * 78)
print("손가락 굽힘 방향 — palm 로컬 프레임 (palm 자세와 무관해야 정상)")
print("  task:", args.task)
print("=" * 78)

# G 규약 euler: top-down (0,0,180) / side (0,0,-90 or +90)
import importlib.util
import sys
from pathlib import Path

_side = "right" if "_r_" in args.task else "left"
_pk = Path(__file__).resolve().parents[1] / ".." / "source" / "openarm" / "openarm" / "tesollo" / _side / "grasp_v2"
_spec = importlib.util.spec_from_file_location("_pr", _pk / f"grasp_{_side}_preset.py")
_pr = importlib.util.module_from_spec(_spec)
sys.modules["_pr"] = _pr
_spec.loader.exec_module(_pr)

measure("top-down", _pr.PREGRASP_G_EULER_TOPDOWN, 0.0)
measure("top-down (abduction +1)", _pr.PREGRASP_G_EULER_TOPDOWN, 1.0)
measure("side (cup 자세)", _pr.PREGRASP_G_EULER_SIDE, 0.0)

print("\n → 세 경우의 palm 로컬 이동이 같아야 정상이다.")
print("   다르면 palm 자세가 손가락 굽힘에 영향을 준다는 뜻 = 어딘가 좌표 버그.")
print("   dX 가 양수여야 손끝이 손바닥 쪽으로 말려 물체를 감쌀 수 있다.")

_OUT.close()
env.close()
app.close()

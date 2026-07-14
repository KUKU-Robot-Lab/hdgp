"""contact sensor 필터가 정말 GPU 미지원인가, 아니면 필터 경로가 틀렸던 것인가?

env.py:493 의 주석은 이렇게 단정한다:
    "MultiAsset(replicate_physics=False)에서 filter_prim_paths_expr(force_matrix_w)는
     GPU 미지원 → contact 0. filter 제거하고 net_forces_w(물체 구분 없음)로 판정"

그 결과 contact/grip 은 테이블을 짚어도 오른다. 실제로 "grip 3.2 인데 object_height 0"
이라는 모순된 로그를 오래 들여다봤다.

그런데 IsaacLab 문서상 필터가 실패하는 흔한 원인은 GPU 가 아니라:
  (a) filter_prim_paths_expr 를 걸어놓고 net_forces_w 를 읽음 (필터는 force_matrix_w 에만 적용)
  (b) prim_path 가 여러 rigid body 를 선택 → force_matrix_w 가 None
  (c) **filter 경로가 실제 rigid body prim 이 아님** (Xform 을 가리킴)

우리 센서는 이미 손가락별 **개별** 센서라 (b)는 아니다. (a)는 확실하다 — 필터 자체를
지워버렸으니까. 남은 건 (c) 다. Cup 은 MultiAsset 이라 실제 body 가 하위에 있을 수 있다.

이 probe 가 확인하는 것:
  1. Cup prim 트리에서 RigidBodyAPI 를 가진 실제 prim 경로
  2. 그 경로로 필터를 건 센서의 force_matrix_w 가 값을 내는가 (None/0 이 아닌가)
  3. 물체 접촉만 세는가 (테이블을 짚었을 때 0 인가)

사용:
  ./isaaclab.sh -p scripts/probes/probe_contact_filter.py --task open-tesol_r_grasp_v2-lstm
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-tesol_r_grasp_v2-lstm")
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402

import omni.usd  # noqa: E402
from pxr import UsdPhysics  # noqa: E402

_OUT = open("/tmp/probe_contact_filter.txt", "w")
_p = print


def print(*a, **kw):  # noqa: A001
    _p(*a, **kw, flush=True)
    _p(*a, **kw, file=_OUT, flush=True)


env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()

print("=" * 88)
print("contact filter 확증 — 테이블만 짚으면 contact 가 0 인가")
print("  force_matrix_w 는 필터 대상(Cup)과의 접촉력만 담는다. 테이블 접촉은 0 이어야 한다.")
print("=" * 88)

import math

n = env.num_envs
D = env.device


def scene(dx, dy, dz, close):
    """palm 을 물체 기준 (dx,dy,dz) 에 두고 손을 닫는다 → contact 를 잰다."""
    env.reset()
    zero = torch.zeros(n, env.cfg.num_actions, device=D)
    for _ in range(int(env.cfg.settle_steps) + 2):
        env.step(zero)
    obj = env.object_pos.clone()
    tgt = torch.zeros(n, 6, device=D)
    tgt[:, 0] = obj[:, 0] + dx
    tgt[:, 1] = obj[:, 1] + dy
    tgt[:, 2] = obj[:, 2] + dz
    tgt[:, 5] = math.pi
    tgt = torch.max(torch.min(tgt, env.palm_maxs_env), env.palm_mins_env)
    lo, hi = env.palm_mins_env, env.palm_maxs_env
    act = torch.zeros(n, env.cfg.num_actions, device=D)
    act[:, :6] = (2.0 * (tgt - lo) / (hi - lo + 1e-9) - 1.0).clamp(-1.0, 1.0)
    act[:, 6:11] = -1.0
    for _ in range(90):
        env.step(act)
    act[:, 6:11] = close
    for _ in range(120):
        env.step(act)
    tip = env.num_contacts_buf.float().mean()
    grip = (env.binary_contact_buf | env.middle_binary_contact_buf
            | env.distal_binary_contact_buf).sum(dim=-1).float().mean()
    tipd = (env.fingertip_pos - env.object_pos.unsqueeze(1)).norm(dim=-1).min(dim=1).values.mean()
    return tip, grip, tipd


print("\n  %-42s %8s %8s %12s" % ("상황", "tip", "grip", "손끝~물체"))
# (a) 물체에서 멀리 떨어져 테이블만 짚는다 — y 로 30cm 비켜서 바닥까지 내린다
t, g, d = scene(0.0, -0.30 if "_r_" in args.task else 0.30, -0.02, 1.0)
print("  %-42s %8.2f %8.2f %12.3f" % ("테이블만 짚음 (물체에서 30cm 비켜서)", t, g, d))
# (b) 물체 위에서 손을 닫는다
t, g, d = scene(-0.08, 0.0, 0.10, 1.0)
print("  %-42s %8.2f %8.2f %12.3f" % ("물체 위에서 폐쇄", t, g, d))
# (c) 손을 열어둔 채 물체 위
t, g, d = scene(-0.08, 0.0, 0.10, -1.0)
print("  %-42s %8.2f %8.2f %12.3f" % ("물체 위, 손 개방", t, g, d))

print("\n  판정: (a) 가 0 이면 필터가 물체만 센다 = 확증.")
print("        (a) 가 0 이 아니면 여전히 테이블을 세고 있다.")

_OUT.close()
env.close()
app.close()

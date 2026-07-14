"""153종 물체 전부에 rigid body 'baseLink' 가 있는가?

contact 필터를 /World/envs/env_.*/Cup/{cup_rigid_body_name} 로 걸었다. 그런데 grasp_v2 는
MultiAsset(replicate_physics=False)이라 **env 마다 다른 물체**가 소환된다. 어떤 물체의
rigid body 이름이 baseLink 가 아니면 그 env 는 필터가 비어 contact 가 영원히 0 이 되고,
학습이 조용히 오염된다.

env 별로 Cup 트리를 훑어 RigidBodyAPI prim 이름을 모으고, 전부 동일한지 확인한다.

사용:
  ./isaaclab.sh -p scripts/probes/probe_cup_bodies.py --task open-tesol_r_grasp_v2-lstm --num_envs 160
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-tesol_r_grasp_v2-lstm")
parser.add_argument("--num_envs", type=int, default=160)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

from collections import Counter  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import omni.usd  # noqa: E402
from pxr import UsdPhysics  # noqa: E402

_OUT = open("/tmp/probe_cup_bodies.txt", "w")
_p = print


def print(*a, **kw):  # noqa: A001
    _p(*a, **kw, flush=True)
    _p(*a, **kw, file=_OUT, flush=True)


env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()

stage = omni.usd.get_context().get_stage()
want = env.cfg.cup_rigid_body_name

names = Counter()
missing = []

for i in range(env.num_envs):
    root = stage.GetPrimAtPath(f"/World/envs/env_{i}/Cup")
    if not root.IsValid():
        missing.append((i, "Cup prim 없음"))
        continue
    found = []
    stack = [root]
    while stack:
        p = stack.pop()
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            found.append(p.GetName())
        stack.extend(p.GetChildren())
    for f in found:
        names[f] += 1
    if want not in found:
        missing.append((i, "rigid body %s (기대: %s)" % (found or "없음", want)))

print("=" * 84)
print("Cup rigid body 이름 일관성 — %s  (env %d 개)" % (args.task, env.num_envs))
print("  contact 필터 대상: /World/envs/env_*/Cup/%s" % want)
print("=" * 84)

print("\n[발견된 rigid body 이름 분포]")
for nm, c in names.most_common():
    mark = "  ← 필터 대상" if nm == want else ""
    print("  %-24s %5d env%s" % (nm, c, mark))

print("\n[필터가 비는 env]  %d 개" % len(missing))
for i, why in missing[:10]:
    print("  env_%-4d %s" % (i, why))
if len(missing) > 10:
    print("  ... 외 %d 개" % (len(missing) - 10))

# 실제로 필터가 값을 내는지 env 별로 확인 (force_matrix_w 가 None 이면 센서 자체가 깨진 것)
s0 = env._tip_sensors[0]
fm = s0.data.force_matrix_w
print("\n[force_matrix_w]  %s" % ("None!" if fm is None else tuple(fm.shape)))
if fm is not None:
    print("  shape (N, bodies, filters, 3) — filters 가 1 이어야 자기 env 의 Cup 만 본다.")

print("\n  판정: '필터가 비는 env' 가 0 이면 153종 전부 baseLink 를 갖는다 = 안전.")
print("        하나라도 있으면 그 env 는 contact 가 영원히 0 이라 학습이 오염된다.")

_OUT.close()
env.close()
app.close()

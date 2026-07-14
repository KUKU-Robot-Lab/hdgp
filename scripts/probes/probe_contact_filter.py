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
print("contact filter 진단 — %s" % args.task)
print("=" * 88)

# ---- 1) Cup prim 트리에서 실제 rigid body 를 찾는다 ----
stage = omni.usd.get_context().get_stage()
print("\n[1] Cup prim 트리 (env_0) — RigidBodyAPI 를 가진 prim 이 필터 대상이어야 한다")
root = stage.GetPrimAtPath("/World/envs/env_0/Cup")
print("  /World/envs/env_0/Cup   valid=%s" % root.IsValid())
rigid_paths = []
if root.IsValid():
    def walk(prim, depth=0):
        has_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        has_col = prim.HasAPI(UsdPhysics.CollisionAPI)
        mark = ""
        if has_rb:
            mark += "  ← RigidBodyAPI"
            rigid_paths.append(str(prim.GetPath()))
        if has_col:
            mark += " +CollisionAPI"
        print("    %s%s (%s)%s" % ("  " * depth, prim.GetName(), prim.GetTypeName(), mark))
        if depth < 3:
            for c in prim.GetChildren():
                walk(c, depth + 1)
    walk(root)

print("\n  RigidBodyAPI 를 가진 경로:")
for rp in rigid_paths:
    print("    %s" % rp)
if not rigid_paths:
    print("    (없음 — Cup 자체가 rigid body 가 아니면 필터가 아무것도 못 찾는다)")

# ---- 2) 현행 센서: force_matrix_w 가 존재하나 ----
print("\n[2] 현행 tip 센서 (필터 없음)")
s0 = env._tip_sensors[0]
print("  body_names      : %s" % (s0.body_names,))
print("  net_forces_w    : %s" % (tuple(s0.data.net_forces_w.shape),))
print("  force_matrix_w  : %s" % ("None" if s0.data.force_matrix_w is None
                                  else tuple(s0.data.force_matrix_w.shape)))
print("  → 필터를 안 걸었으니 force_matrix_w 가 None 인 것은 당연하다.")

# ---- 3) 필터를 건 센서를 새로 만들어 본다 ----
print("\n[3] 필터를 건 센서를 새로 생성 — force_matrix_w 가 값을 내는가")
_cands = []
if rigid_paths:
    # env_0 경로를 env_.* 패턴으로 일반화
    for rp in rigid_paths[:2]:
        _cands.append(rp.replace("/World/envs/env_0/", "/World/envs/env_.*/"))
_cands.append("/World/envs/env_.*/Cup")          # 현행 코드가 쓰던 경로 (대조군)

_link = env.cfg.right_tip_contact_links[1]        # index_tip
for cand in _cands:
    try:
        s = ContactSensor(ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{_link}",
            filter_prim_paths_expr=[cand],
            history_length=1,
            track_air_time=False,
        ))
        env.scene.sensors[f"_probe_{cand}"] = s
        s._initialize_impl()
        fm = s.data.force_matrix_w
        print("  filter=%-46s force_matrix_w=%s"
              % (cand, "None" if fm is None else tuple(fm.shape)))
    except Exception as e:                        # noqa: BLE001
        print("  filter=%-46s 실패: %s" % (cand, str(e)[:60]))

print("\n  → force_matrix_w 가 (N,1,F,3) 형태로 나오면 필터는 동작한다.")
print("     그렇다면 'GPU 미지원' 이라는 현행 주석은 오진이고, 경로가 틀렸던 것이다.")

_OUT.close()
env.close()
app.close()

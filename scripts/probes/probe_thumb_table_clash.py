"""top-down 접근에서 엄지가 테이블에 먼저 닿는가?

사용자 렌더 관찰: "테이블에 손가락이 먼저 닿게 됨(thumb_2 때문에). 그래서 물체에
palm 을 가까이 가는 게 아니라 테이블을 쓸고 있음."

palm 을 top-down 자세로 여러 높이에 놓고, HAND_APPROACH_POSE 에서 각 손끝의 world z 를
잰다. 엄지가 다른 손끝보다 아래(테이블 쪽)면 palm 이 물체까지 못 내려간다.

thumb_2(대향)와 thumb_3(PIP)를 각각 스윕해 어느 관절이 범인인지 분리한다.
thumb_3 = -0.5 는 주석상 side(컵) 접근 전용 튜닝이다.

사용:
  ./isaaclab.sh -p scripts/probes/probe_thumb_table_clash.py --task open-tesol_r_grasp_v2-lstm
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-tesol_r_grasp_v2-lstm")
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

_OUT = open("/tmp/probe_thumb_table_clash.txt", "w")
_p = print


def print(*a, **kw):  # noqa: A001
    _p(*a, **kw, flush=True)
    _p(*a, **kw, file=_OUT, flush=True)


env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()

n = env.num_envs
FING = ("thumb", "index", "middle", "ring", "pinky")
TABLE_Z = 0.200

# 손 자세는 시너지 anchor(hand_open_pose)를 직접 바꿔서 준다.
# env.step() 이 매 스텝 hand target 을 이 anchor 로 되쓰므로, set_joint_position_target 을
# 밖에서 덮어써봐야 다음 스텝에 지워진다(초판 probe 의 버그 — 스윕이 전혀 안 먹었다).
_OPEN_POSE_BACKUP = env.hand_open_pose.clone()


def place_palm_and_measure(palm_z_above_table: float, hand_q: torch.Tensor, steps: int = 90):
    """palm 을 테이블 위 지정 높이에 top-down 으로 놓고 손끝 world z 를 잰다.

    hand_q: (20,) 시너지 open anchor 로 쓸 손 자세.
    """
    env.hand_open_pose.copy_(hand_q)
    env.reset()
    zero = torch.zeros(n, env.cfg.num_actions, device=env.device)
    for _ in range(int(env.cfg.settle_steps) + 2):
        env.step(zero)

    obj = env.object_pos.clone()
    tgt = torch.zeros(n, 6, device=env.device)
    tgt[:, 0] = obj[:, 0]
    tgt[:, 1] = obj[:, 1]
    tgt[:, 2] = TABLE_Z + palm_z_above_table
    tgt[:, 5] = math.pi                       # G 규약 top-down
    tgt = torch.max(torch.min(tgt, env.palm_maxs_env), env.palm_mins_env)

    lo, hi = env.palm_mins_env, env.palm_maxs_env
    act = torch.zeros(n, env.cfg.num_actions, device=env.device)
    act[:, :6] = (2.0 * (tgt - lo) / (hi - lo + 1e-9) - 1.0).clamp(-1.0, 1.0)
    act[:, 6:11] = -1.0                        # 시너지 개방

    for _ in range(steps):
        env.step(act)

    tips = env.fingertip_pos.clone()           # (n,5,3) env-local
    palm = env.palm_center_pos.clone()         # (n,3)
    return palm, tips


base = _OPEN_POSE_BACKUP.clone()               # HAND_APPROACH_POSE

print("=" * 92)
print("top-down 접근에서 엄지가 테이블을 먼저 치는가 — %s" % args.task)
print("  테이블 상면 z = %.3f. 손끝 z 가 이보다 낮으면 테이블에 박힌다." % TABLE_Z)
print("=" * 92)

# ---- 1) 현행 HAND_APPROACH_POSE 그대로, palm 높이 스윕 ----
print("\n[1] 현행 HAND_APPROACH_POSE (thumb = [0, -1.57, -0.5, 0])")
print("    palm 을 테이블 위 h 에 두었을 때 각 손끝의 world z (테이블 침투는 ✗)")
print("    %-8s %-9s %s" % ("palm h", "palm z", "  ".join("%8s" % f for f in FING)))
for h in (0.20, 0.16, 0.12, 0.10, 0.08, 0.06):
    palm, tips = place_palm_and_measure(h, base)
    tz = tips[:, :, 2].mean(dim=0)
    row = "    %-8.2f %-9.3f" % (h, palm[:, 2].mean())
    for k in range(5):
        mark = "✗" if tz[k] < TABLE_Z else " "
        row += "  %7.3f%s" % (tz[k], mark)
    print(row)

print("\n    → 엄지 z 가 다른 손끝보다 낮으면, palm 을 내릴 때 엄지가 먼저 테이블에 걸린다.")

# ---- 2) thumb_2(대향) 스윕: 범인인가? ----
print("\n[2] thumb_2(대향) 스윕 — palm 을 테이블 위 0.10 에 고정")
print("    %-10s %-8s %s" % ("thumb_2", "palm z", "  ".join("%8s" % f for f in FING)))
for t2 in (0.0, -0.5, -1.0, -1.57, -2.0):
    q = base.clone(); q[1] = t2
    palm, tips = place_palm_and_measure(0.10, q)
    tz = tips[:, :, 2].mean(dim=0)
    row = "    %-10.2f %-8.3f" % (t2, palm[:, 2].mean())
    for k in range(5):
        mark = "✗" if tz[k] < TABLE_Z else " "
        row += "  %7.3f%s" % (tz[k], mark)
    print(row)

# ---- 3) thumb_3(PIP) 스윕: -0.5 는 side(컵) 전용 튜닝이다 ----
print("\n[3] thumb_3(PIP) 스윕 — thumb_2 = -1.57 고정, palm 테이블 위 0.10")
print("    %-10s %-8s %s" % ("thumb_3", "palm z", "  ".join("%8s" % f for f in FING)))
for t3 in (0.5, 0.0, -0.5, -1.0):
    q = base.clone(); q[2] = t3
    palm, tips = place_palm_and_measure(0.10, q)
    tz = tips[:, :, 2].mean(dim=0)
    row = "    %-10.2f %-8.3f" % (t3, palm[:, 2].mean())
    for k in range(5):
        mark = "✗" if tz[k] < TABLE_Z else " "
        row += "  %7.3f%s" % (tz[k], mark)
    print(row)

# ---- 4) thumb_1(abduction) 스윕 + 엄지 전체 중립화 ----
print("\n[4] thumb_1(벌림) 스윕 + 엄지 완전 중립(전부 0) — palm 테이블 위 0.06 요청")
print("    요청 palm z = %.3f. 실제 palm z 가 이보다 높으면 엄지가 테이블에 막힌 것이다." % (TABLE_Z + 0.06))
print("    %-22s %-8s %s" % ("thumb 자세", "palm z", "  ".join("%8s" % f for f in FING)))
_cases = [
    ("현행 [0,-1.57,-0.5,0]", base.clone()),
    ("thumb_1=+0.89(최대벌림)", None),
    ("엄지 전부 0", None),
    ("Allegro식 t1=0.5만", None),
]
q = base.clone(); q[0] = 0.8901179; _cases[1] = (_cases[1][0], q)
q = base.clone(); q[0:4] = 0.0;     _cases[2] = (_cases[2][0], q)
q = base.clone(); q[0:4] = 0.0; q[0] = 0.5; _cases[3] = (_cases[3][0], q)
for label, q in _cases:
    palm, tips = place_palm_and_measure(0.06, q)
    tz = tips[:, :, 2].mean(dim=0)
    row = "    %-22s %-8.3f" % (label, palm[:, 2].mean())
    for k in range(5):
        mark = "✗" if tz[k] < TABLE_Z else " "
        row += "  %7.3f%s" % (tz[k], mark)
    print(row)

print("\n  판정 기준:")
print("   - 엄지 z 가 index~pinky 보다 뚜렷이 낮다  → 엄지가 테이블에 먼저 닿는다 (사용자 관찰 확증)")
print("   - thumb_2 또는 thumb_3 를 바꿔 엄지 z 가 다른 손끝 수준으로 올라오면 그 관절이 범인이다")

_OUT.close()
env.close()
app.close()

"""손가락을 완전히 굽히면 물체를 잡을 수 있는가?

학습된 정책은 손을 굽히지 않는다(curl ≈ 0, grip 0.5, object_height 음수).
curl 페널티를 3배 낮췄는데도 안 굽힌다는 건 "굽혀도 이득이 없다" = 굽혀도 물체를
못 잡는다는 뜻일 수 있다. 정책 없이 직접 확인한다:

  palm 을 여러 높이(물체 위 N cm)에 고정하고 손가락 action 을 +1(완전 폐쇄)로 준 뒤
    - 5개 손끝이 물체에 얼마나 접근하는가
    - 실제 접촉(contact) 이 몇 개 잡히는가
    - 물체가 들리는가 (palm 을 위로 올렸을 때)

사용:
  ./isaaclab.sh -p scripts/probes/probe_close_hand.py --task open-tesol_r_grasp_v2-lstm
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-tesol_r_grasp_v2-lstm")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--close_steps", type=int, default=120, help="손 폐쇄 유지 스텝")
parser.add_argument("--lift_steps", type=int, default=120, help="이후 들어올리는 스텝")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

_OUT = open("/tmp/probe_close_hand.txt", "w")
_p = print


def print(*a, **kw):  # noqa: A001
    _p(*a, **kw, flush=True)
    _p(*a, **kw, file=_OUT, flush=True)


env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()

n = env.num_envs
zero = torch.zeros(n, env.cfg.num_actions, device=env.device)
for _ in range(int(env.cfg.settle_steps) + 2):     # 물체 낙하·안착
    env.step(zero)

obj0 = env.object_pos.clone()                       # env-local (이미 변환됨)

# env 별로 palm 을 물체 위 서로 다른 높이에 고정
heights = [0.14, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02, 0.00]
per = max(1, n // len(heights))
labels = [heights[min(i // per, len(heights) - 1)] for i in range(n)]
off = torch.tensor(labels, device=env.device)

tgt = torch.zeros(n, 6, device=env.device)
tgt[:, :3] = obj0
tgt[:, 2] += off
tgt[:, 5] = math.pi                                 # G 규약 top-down
tgt = torch.max(torch.min(tgt, env.palm_maxs_env), env.palm_mins_env)

lo, hi = env.palm_mins_env, env.palm_maxs_env
a_palm = (2.0 * (tgt - lo) / (hi - lo + 1e-9) - 1.0).clamp(-1.0, 1.0)

# 1단계: palm 을 그 높이로 옮기면서 손은 편 채
act = torch.zeros(n, env.cfg.num_actions, device=env.device)
act[:, :6] = a_palm
act[:, 6:11] = -1.0        # 시너지: 완전 개방
# abduction 5축: [thumb_1, thumb_2(대향), index_1, pinky_1, pinky_2]
# thumb_2 를 env 별로 스윕한다 — 이게 열려야 top-down 파지가 되는지 본다.
act[:, 11:16] = 0.0
_th2 = torch.tensor([-1.0, -0.5, 0.0, +0.5, +1.0], device=env.device)   # action 공간
act[:, 12] = _th2[torch.arange(n, device=env.device) % 5]
for _ in range(80):
    env.step(act)

palm_before = env.palm_center_pos.clone()
tips_open = env.fingertip_pos.clone()

# 2단계: 손가락 완전 폐쇄
act[:, 6:11] = 1.0         # 시너지: 완전 폐쇄 (thumb_2 는 위에서 준 값 유지)
for _ in range(args.close_steps):
    env.step(act)

tips_closed = env.fingertip_pos.clone()
obj_after_close = env.object_pos.clone()
grip = (
    env.binary_contact_buf | env.middle_binary_contact_buf | env.distal_binary_contact_buf
).sum(dim=-1).float()
tip_contact = env.num_contacts_buf.float()

# 3단계: palm 을 위로 올려 리프트 시도 (손은 폐쇄 유지)
tgt_lift = tgt.clone()
tgt_lift[:, 2] = torch.clamp(tgt[:, 2] + 0.20, max=env.palm_maxs_env[:, 2])
act[:, :6] = (2.0 * (tgt_lift - lo) / (hi - lo + 1e-9) - 1.0).clamp(-1.0, 1.0)
for _ in range(args.lift_steps):
    env.step(act)

obj_lifted = env.object_pos.clone()
lift = obj_lifted[:, 2] - obj0[:, 2]

print("\n" + "=" * 88)
print("손을 완전히 굽히면 물체를 잡을 수 있는가 — %s" % args.task)
print("  palm 을 물체 위 N cm 에 고정 → 손가락 완전 폐쇄 → palm 20cm 상승")
print("=" * 88)
import math as _m
_th2_deg = {}
for i in range(n):
    a = float(act[i, 12])
    lo_i = float(env.abduction_limits_min[1]); hi_i = float(env.abduction_limits_max[1])
    _th2_deg[i] = _m.degrees(0.5 * (a + 1.0) * (hi_i - lo_i) + lo_i)

print("\n  palm높이 × 엄지대향(thumb_2) → 리프트 (물체가 따라 올라온 높이 cm)")
print("  %-9s %s" % ("palm높이", "  ".join("%7.0f°" % _th2_deg[k] for k in range(5))))
seen = set()
for i in range(n):
    h = labels[i]
    if h in seen:
        continue
    seen.add(h)
    row = "  %-9.2f" % h
    for t in range(5):
        idx = [j for j in range(n) if labels[j] == h and j % 5 == t]
        if not idx:
            row += "  %7s" % "-"
            continue
        row += "  %+7.1f" % (lift[idx].mean() * 100)
    print(row)

print("\n  같은 표, grip (접촉 손가락 수)")
print("  %-9s %s" % ("palm높이", "  ".join("%7.0f°" % _th2_deg[k] for k in range(5))))
seen = set()
for i in range(n):
    h = labels[i]
    if h in seen:
        continue
    seen.add(h)
    row = "  %-9.2f" % h
    for t in range(5):
        idx = [j for j in range(n) if labels[j] == h and j % 5 == t]
        row += "  %7.2f" % (grip[idx].mean() if idx else float("nan"))
    print(row)

# ---- 손가락 굽힘 방향 실측 (핵심) ----
# top-down 에서 손을 굽히면 손끝이 world -Z(아래, 물체 쪽)로 말려야 물체를 감싼다.
# 위(+Z)나 옆으로 가면 물체를 놓친다.
delta = tips_closed - tips_open          # (n, 5, 3) 손끝 이동 벡터
print("\n[손가락 굽힘 방향] 폐쇄 시 손끝이 world 어디로 이동하는가")
print("  %-9s %-24s %-9s" % ("손가락", "이동 (dx, dy, dz) cm", "판정"))
_fnames = ("thumb", "index", "middle", "ring", "pinky")
for k in range(5):
    d = delta[:, k, :].mean(dim=0) * 100
    verdict = "아래(물체쪽) ✓" if d[2] < -2 else ("위(반대) ✗" if d[2] > 2 else "수평(못 감쌈)")
    print("  %-9s (%+6.1f, %+6.1f, %+6.1f)     %s" % (_fnames[k], d[0], d[1], d[2], verdict))
print("  → dz 가 음수여야 손끝이 아래로 말려 물체를 감싼다 (DEXTRAH Allegro 방식)")

print("\n  손끝~물체: 폐쇄 전후 최소 거리 (m). 폐쇄 후에도 크면 굽혀도 안 닿는다.")
print("  리프트: palm 을 20cm 올렸을 때 물체가 따라 올라온 높이. 0 이면 못 잡은 것.")
print("\n  → 어느 높이에서도 리프트가 0 이면 '굽혀도 못 잡는다'가 확증된다.")
print("  → 특정 높이에서 리프트가 되면 정책이 그 높이로 못 내려간 것이 문제다.")

_OUT.close()
env.close()
app.close()

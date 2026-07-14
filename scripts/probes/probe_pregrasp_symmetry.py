"""리셋 직후 pregrasp 상태가 좌우 정확한 거울상인가?

학습의 출발점은 pregrasp 이다. 여기가 좌우로 다르면 초기 궤적이 갈리고,
그 뒤 무엇을 맞춰놔도 좌우가 다른 흡인 영역으로 빨려간다.

지금까지 확인된 대칭: 보상 가중치·goal·spawn xy·종료 경계·fabric world·
손 관절 부호(_HAND_SIGN)·palm 도달성(좌우 1~2mm). 아직 안 잰 것이 이것이다.

**물체 기준 상대 좌표**로 재고, LEFT 의 y 부호를 뒤집어 RIGHT 와 겹쳐본다.
거울상이면 모든 성분이 일치해야 한다.

사용:
  ./isaaclab.sh -p scripts/probes/probe_pregrasp_symmetry.py --task open-tesol_r_grasp_v2-lstm
  ./isaaclab.sh -p scripts/probes/probe_pregrasp_symmetry.py --task open-tesol_l_grasp_v2-lstm
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-tesol_r_grasp_v2-lstm")
parser.add_argument("--num_envs", type=int, default=256)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402

_OUT = open("/tmp/probe_pregrasp_symmetry.txt", "w")
_p = print


def print(*a, **kw):  # noqa: A001
    _p(*a, **kw, flush=True)
    _p(*a, **kw, file=_OUT, flush=True)


env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()

n = env.num_envs
FING = ("thumb", "index", "middle", "ring", "pinky")
IS_LEFT = "_l_" in args.task
SIDE = "LEFT" if IS_LEFT else "RIGHT"

# settle 만 돌린다 — action 을 주지 않는다. 순수 pregrasp 상태.
zero = torch.zeros(n, env.cfg.num_actions, device=env.device)
for _ in range(int(env.cfg.settle_steps) + 2):
    env.step(zero)

obj = env.object_pos.clone()                     # (n,3) env-local, 안착 후
palm = env.palm_center_pos.clone()               # (n,3)
tips = env.fingertip_pos.clone()                 # (n,5,3)
R = matrix_from_quat(env.robot.data.body_quat_w[:, env.palm_body_index])   # (n,3,3)

# 물체 기준 상대 좌표
d_palm = palm - obj                               # (n,3)
d_tips = tips - obj.unsqueeze(1)                  # (n,5,3)

# palm 로컬축의 world 표현: col0 = +X(손바닥 법선), col2 = +Z(손가락 방향)
normal = R[:, :, 0]
finger = R[:, :, 2]

# LEFT 는 y 를 뒤집어 RIGHT 와 겹친다(거울상이면 일치해야 한다)
flip = torch.tensor([1.0, -1.0, 1.0], device=env.device) if IS_LEFT else torch.ones(3, device=env.device)

print("=" * 84)
print("pregrasp 대칭 검증 — %s (%s)" % (args.task, SIDE))
print("  물체 기준 상대 좌표. LEFT 는 y 부호를 뒤집어 RIGHT 와 같은 좌표계로 옮긴다.")
print("  → 거울상이면 아래 모든 값이 좌우 일치해야 한다.")
print("=" * 84)

print("\n[물체 위치] (env-local, 안착 후)")
print("  obj = (%+.4f, %+.4f, %+.4f)   [y-flip 후: %+.4f]"
      % (*obj.mean(dim=0), (obj.mean(dim=0) * flip)[1]))

print("\n[palm ← 물체]  (물체 기준 상대, y-flip 적용)")
v = (d_palm * flip).mean(dim=0)
print("  d_palm = (%+.4f, %+.4f, %+.4f)   |d| = %.4f" % (*v, d_palm.norm(dim=-1).mean()))

print("\n[palm 자세]  (y-flip 적용)")
vn = (normal * flip).mean(dim=0)
vf = (finger * flip).mean(dim=0)
print("  손바닥 법선 (+X) = (%+.3f, %+.3f, %+.3f)   ← top-down 이면 z ≈ -1" % (*vn,))
print("  손가락 방향 (+Z) = (%+.3f, %+.3f, %+.3f)" % (*vf,))

print("\n[손끝 ← 물체]  (물체 기준 상대, y-flip 적용)")
print("  %-8s %-28s %8s" % ("손가락", "(dx, dy, dz)", "거리"))
for k in range(5):
    v = (d_tips[:, k, :] * flip).mean(dim=0)
    dist = d_tips[:, k, :].norm(dim=-1).mean()
    print("  %-8s (%+.4f, %+.4f, %+.4f)   %8.4f" % (FING[k], v[0], v[1], v[2], dist))

print("\n[요약 — 좌우 비교용 한 줄]")
_dp = (d_palm * flip).mean(dim=0)
_mind = d_tips.norm(dim=-1).min(dim=1).values.mean()
_maxd = d_tips.norm(dim=-1).max(dim=1).values.mean()
print("  %s  d_palm=(%+.4f,%+.4f,%+.4f)  손끝~물체 min=%.4f max=%.4f  법선z=%+.3f"
      % (SIDE, _dp[0], _dp[1], _dp[2], _mind, _maxd, vn[2]))

_OUT.close()
env.close()
app.close()

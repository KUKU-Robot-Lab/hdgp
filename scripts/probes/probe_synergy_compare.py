"""손가락 협응 방식 3종 중 무엇이 top-down 에서 물체를 드는가?

현행 basis 는 Allegro PCA 를 tesollo 로 리타겟한 것이다(tesollo_hand_synergy.py
docstring). 그리고 명백히 깨져 있다:
  - pinky_1 열이 5축 전부 정확히 0.0  (Allegro 는 4지라 대응 관절이 없다)
  - PC3 는 coeff min(0.34) > max(-3.06)  (범위 역전)
  - PC5 는 범위 폭이 0.09  (사실상 죽은 축)
게다가 finger_action_utils.py:61 이 스스로 적어둔 한계가 있다 —
"PC1 하나가 20관절을 커플링 → 2지 최소해가 action 공간에서 표현 불가".

비교 대상:
  A) 현행       — Allegro 리타겟 PCA (baseline. offset 스윕에서 최대 리프트 2.3cm)
  B) tesollo PCA — 실제 tesollo 파지에서 역추론 (assets/demograsp_references/…_from_right.pt)
                   단 side 접근 성공분에서 뽑은 것이다.
  C) per-finger lerp — grasp_v1 방식(98% 실증). basis 없이 손가락별 open→grip 직선.
                   같은 progress 틀 안에서 basis 로 정확히 재현 가능하다.

각 방식에서 PC1(주 개폐축)을 스윕하고, palm x offset 두 지점에서 리프트를 잰다.

사용:
  ./isaaclab.sh -p scripts/probes/probe_synergy_compare.py --task open-tesol_r_grasp_v2-lstm
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
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

_OUT = open("/tmp/probe_synergy_compare.txt", "w")
_p = print


def print(*a, **kw):  # noqa: A001
    _p(*a, **kw, flush=True)
    _p(*a, **kw, file=_OUT, flush=True)


env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env = gym.make(args.task, cfg=env_cfg).unwrapped
env.reset()

n = env.num_envs
IS_LEFT = "_l_" in args.task
SIDE = "left" if IS_LEFT else "right"
D = env.device

# 현행 값 백업
BASIS0 = env.hand_synergy_basis.clone()
ANCHOR0 = env.hand_synergy_anchor.clone()
MINS0 = env.hand_synergy_mins.clone()
MAXS0 = env.hand_synergy_maxs.clone()
OPEN = env.hand_open_pose.clone()                     # (20,) HAND_APPROACH_POSE
GRIP = env.hand_grip_pose.clone() if hasattr(env, "hand_grip_pose") else None
if GRIP is None:                                      # 이름이 다르면 FULL_GRIP 을 찾는다
    GRIP = env.hand_full_grip_pose.clone()


def set_synergy(basis, anchor, mins, maxs):
    env.hand_synergy_basis.copy_(basis)
    env.hand_synergy_anchor.copy_(anchor)
    env.hand_synergy_mins.copy_(mins)
    env.hand_synergy_maxs.copy_(maxs)


# ---- B) tesollo 실측 PCA ----
_pca_path = (
    Path(__file__).resolve().parents[1].parent
    / "assets" / "demograsp_references" / f"tesollo_grasp_pca5_from_{SIDE}.pt"
)
_d = torch.load(_pca_path, map_location="cpu", weights_only=False)
_k = "basis_%s" % SIDE
B_basis = torch.as_tensor(_d[_k] if _k in _d else _d["basis_right"], dtype=torch.float32, device=D)
_km = "mean_%s" % SIDE
B_anchor = torch.as_tensor(_d[_km] if _km in _d else _d["mean_right"], dtype=torch.float32, device=D)
B_mins = torch.as_tensor(_d["coeff_mins_uncentered"], dtype=torch.float32, device=D)
B_maxs = torch.as_tensor(_d["coeff_maxs_uncentered"], dtype=torch.float32, device=D)

# ---- C) per-finger lerp: basis 행 i = 손가락 i 의 (grip - open) 성분만 ----
# q* = open + Σ a_i·(grip-open)|_finger_i  →  progress_j = a_i  (정확히 lerp)
C_basis = torch.zeros(5, 20, device=D)
_delta = GRIP - OPEN
for i in range(5):
    C_basis[i, 4 * i: 4 * i + 4] = _delta[4 * i: 4 * i + 4]
C_anchor = OPEN.clone()
C_mins = torch.zeros(5, device=D)
C_maxs = torch.ones(5, device=D)

MODES = [
    ("A 현행(Allegro리타겟)", BASIS0, ANCHOR0, MINS0, MAXS0),
    ("B tesollo실측PCA", B_basis, B_anchor, B_mins, B_maxs),
    ("C per-finger lerp", C_basis, C_anchor, C_mins, C_maxs),
]


def trial(dx: float, a1: float, all_fingers: bool, dz: float = 0.10):
    """palm 을 물체 위 (dx, 0, +0.10) 에 top-down 으로 두고 → 손 폐쇄 → 20cm 상승."""
    env.reset()
    zero = torch.zeros(n, env.cfg.num_actions, device=D)
    for _ in range(int(env.cfg.settle_steps) + 2):
        env.step(zero)

    obj0 = env.object_pos.clone()

    tgt = torch.zeros(n, 6, device=D)
    tgt[:, 0] = obj0[:, 0] + dx
    tgt[:, 1] = obj0[:, 1]
    tgt[:, 2] = obj0[:, 2] + dz
    tgt[:, 5] = math.pi
    tgt = torch.max(torch.min(tgt, env.palm_maxs_env), env.palm_mins_env)

    lo, hi = env.palm_mins_env, env.palm_maxs_env
    act = torch.zeros(n, env.cfg.num_actions, device=D)
    act[:, :6] = (2.0 * (tgt - lo) / (hi - lo + 1e-9) - 1.0).clamp(-1.0, 1.0)
    act[:, 6:11] = -1.0                      # 개방 상태로 접근
    for _ in range(90):
        env.step(act)

    # 폐쇄: lerp 는 5축 모두, PCA 는 PC1(주 개폐축)만
    if all_fingers:
        act[:, 6:11] = a1
    else:
        act[:, 6] = a1
    for _ in range(120):
        env.step(act)

    grip = (
        env.binary_contact_buf | env.middle_binary_contact_buf | env.distal_binary_contact_buf
    ).sum(dim=-1).float().mean()

    tgt_up = tgt.clone()
    tgt_up[:, 2] = torch.clamp(tgt[:, 2] + 0.20, max=env.palm_maxs_env[:, 2])
    act[:, :6] = (2.0 * (tgt_up - lo) / (hi - lo + 1e-9) - 1.0).clamp(-1.0, 1.0)
    for _ in range(120):
        env.step(act)

    lift = (env.object_pos[:, 2] - obj0[:, 2]).mean() * 100
    return grip, lift


print("=" * 90)
print("palm 높이(dz) × x offset(dx) 2D 스윕 — %s" % args.task)
print("  현행 basis. PC1=+0.5 로 폐쇄. 리프트 cm.")
print("  실측 pregrasp 는 dz=+0.106 — 아래 표에서 그 높이가 최적인지 본다.")
print("=" * 90)

set_synergy(BASIS0, ANCHOR0, MINS0, MAXS0)
DZS = (0.04, 0.06, 0.08, 0.10, 0.12)
DXS = (0.00, -0.04, -0.08, -0.12)

print("\n  %-8s %s" % ("dx \\ dz", "  ".join("%12s" % ("%.2f" % z) for z in DZS)))
for dx in DXS:
    row = "  %-8.2f" % dx
    for dz in DZS:
        g, lf = trial(dx, 0.5, False, dz)
        mark = "*" if lf > 3.0 else " "
        row += "  %6.1f(g%.1f)%s" % (lf, g, mark)
    print(row)

print("\n  * = 리프트 3cm 초과.  실측 pregrasp = (dx≈0, dz≈0.106)")

_OUT.close()
env.close()
app.close()

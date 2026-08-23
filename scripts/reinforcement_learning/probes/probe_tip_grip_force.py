"""손끝 IK 로 **파지력이 만들어지는가** — Notion 경고의 정량화.

경고(2026-08-23 Fabrics 손 제어 조사): "손끝 목표를 접촉면에 주면 오차 소멸로 파지력이
안 생길 수 있다(컵 안쪽으로 줘야 조인다)." 관절 PD 는 컵 표면 너머까지 관절을 밀어
힘을 만든다. 손끝 attractor 는 목표에 **도달하면 힘이 0** 이므로, 목표가 컵 표면이면
접촉만 하고 조이지 않는다.

그래서 재는 것은 하나다: **손끝 목표를 컵 안쪽으로 얼마나 넣어야 파지력이 나오는가**,
그리고 그 깊이가 **액션 박스 안에 있는가**(밖이면 정책이 도달할 수 없다).

절차(관절 모드와 같은 시나리오):
  리셋 → 손 열기 → 컵을 palm·손끝 중간에 **운동학적으로 고정**(중력 낙하로 접촉이
  사라지면 액추에이터 변조성과 파지 안정성이 섞인다) → 손끝 목표를 컵 축 주변
  반경 r 로 램프 → 접촉력·감쌈 측정. r 을 컵 반경부터 안쪽으로 스윕한다.

    isaaclab.sh -p .../probe_tip_grip_force.py --num_envs 32
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="open-bis_r_grasp_lift_fab")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--settle", type=int, default=70, help="지령 단계마다 정착 스텝")
parser.add_argument("--ramp", type=int, default=30)
parser.add_argument("--mode", default="tip", choices=["tip", "curl", "joint"],
                    help="tip=손끝을 컵 대향점으로 · curl=손끝을 손바닥 쪽으로 말기 "
                         "· joint=관절 PD(대조군)")
parser.add_argument("--gain", type=float, default=None, help="tip attractor conical_gain")
parser.add_argument("--span", type=float, default=None, help="tip_action_span_frac")
parser.add_argument("--radius", type=float, default=0.025, help="tip 모드 파지 반경[m]")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym      # noqa: E402
import torch                 # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg   # noqa: E402

import openarm.tasks         # noqa: E402,F401
from openarm.agnostic.tasks.grasp_lift_fabric.grasp_lift_fabric_env_cfg import (  # noqa: E402
    resolve_cfg,
)

env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
env_cfg.enable_self_collisions = True      # 손가락 상호관통이 있으면 접촉력이 허수다
env_cfg.enable_gravity = True
env_cfg.gravity_compensation = 1.0
env_cfg.use_tip_fabric = args.mode in ("tip", "curl")
if args.gain is not None:
    env_cfg.tip_attractor_gain = args.gain
if args.span is not None:
    env_cfg.tip_action_span_frac = args.span
resolve_cfg(env_cfg)
env = gym.make(args.task, cfg=env_cfg).unwrapped
N, A = args.num_envs, env.cfg.action_space
dev = env.device


def _hold_cup() -> torch.Tensor:
    """컵을 palm 과 손끝 중간에 두는 root state (운동학적 고정용)."""
    palm = env.robot.data.body_pos_w[:, env.palm_idx]
    tips = env.robot.data.body_pos_w[:, env._tip_t].mean(dim=1)
    hold = torch.zeros(N, 13, device=dev)
    hold[:, :3] = 0.5 * (palm + tips)
    hold[:, 3] = 1.0
    return hold


def _tip_action_for_radius(hold: torch.Tensor, radius: float) -> torch.Tensor:
    """손끝을 컵 축 주변 반경 `radius` 에 두는 액션을 역산한다.

    보상의 대향 파지점과 같은 기하: 컵 중심에서 palm→컵 방향의 **수직축** 위로
    ±radius. 엄지(그룹A)는 한쪽, 나머지는 반대쪽. radius 를 컵 반경보다 작게 주면
    손끝 목표가 컵 **안쪽**이 되어 attractor 가 계속 밀어붙인다.
    """
    # ★fabric FK 는 로봇 base(=env 원점) 기준이고 robot.data.body_pos_w 는 world 다.
    #   섞으면 env spacing 만큼 오차가 난다(첫 시도: 손끝오차 5.5m 로 드러났다).
    org = env.scene.env_origins
    obj = hold[:, :3] - org
    palm_o, R = env._palm_frame(env.fabric_q.detach())
    u = obj[:, :2] - palm_o[:, :2]
    u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    n = torch.stack([-u[:, 1], u[:, 0], torch.zeros_like(u[:, 0])], dim=-1)   # (N,3)
    # 부호는 현재 엄지 쪽으로 — 좌우 로봇에서 방향 가정 없이 대향만 강제한다.
    tips_w = env.robot.data.body_pos_w[:, env._tip_t] - org[:, None, :]
    a_idx = env._grp_a
    sgn = torch.sign(((tips_w[:, a_idx].mean(dim=1) - obj) * n).sum(dim=-1, keepdim=True))
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    n = n * sgn
    tgt_w = obj[:, None, :].repeat(1, tips_w.shape[1], 1)
    grp_a = set(env._grp_a.tolist())
    for k in range(tips_w.shape[1]):
        tgt_w[:, k] = obj + n * (radius if k in grp_a else -radius)
    # world → palm 상대 → 액션(구간별 선형의 역함수)
    rel = torch.einsum("bij,bkj->bki", R.transpose(1, 2), tgt_w - palm_o[:, None, :])
    d = rel - env._tip_home
    a = torch.where(d >= 0.0,
                    d / (env._tip_hi - env._tip_home).clamp(min=1e-6),
                    d / (env._tip_home - env._tip_lo).clamp(min=1e-6))
    return a.clamp(-1.0, 1.0).reshape(N, -1)


def _tip_action_curl(depth: float) -> torch.Tensor:
    """홈 손끝을 손끝 중심 쪽으로 `depth`[m] 당기는 액션 — 자연스러운 폐합.

    손끝 중심은 파지 중심에 가깝다. 각 손끝이 **자기 방향으로** 안쪽으로 들어가므로
    손가락이 말리고 중간·원위 마디가 물체에 닿는다 = 감쌈. 대향점 목표(mode=tip)는
    네 손가락을 한 점으로 보내 서로 막혀 한 손가락만 닿았다.
    """
    home = env._tip_home                                   # (T,3) palm 상대
    c = home.mean(dim=0, keepdim=True)
    d = c - home
    d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    want = home + d * depth
    delta = want - home
    a = torch.where(delta >= 0.0,
                    delta / (env._tip_hi - env._tip_home).clamp(min=1e-6),
                    delta / (env._tip_home - env._tip_lo).clamp(min=1e-6))
    return a.clamp(-1.0, 1.0).reshape(1, -1).repeat(N, 1)


def measure(level: float) -> tuple[float, float, float, float]:
    """level = tip 모드면 파지 반경[m], joint 모드면 폐합 지령(0~1)."""
    env.reset()
    act = torch.zeros(N, A, device=dev)
    act[:, 6:] = -1.0 if args.mode == "joint" else 0.0
    for _ in range(25):
        env.step(act)
    hold = _hold_cup()
    q_before = env.robot.data.joint_pos[:, env._hand_t].clone()

    if args.mode == "tip":
        goal_a = _tip_action_for_radius(hold, level)
        start_a = torch.zeros_like(goal_a)
    elif args.mode == "curl":
        goal_a = _tip_action_curl(level)
        start_a = torch.zeros_like(goal_a)
    else:
        goal_a = torch.full((N, A - 6), -1.0 + 2.0 * level, device=dev)
        start_a = torch.full_like(goal_a, -1.0)

    for i in range(args.ramp + args.settle):
        env.object.write_root_state_to_sim(hold)          # 매 스텝 고정
        w = min(1.0, (i + 1) / args.ramp)
        act[:, 6:] = start_a + (goal_a - start_a) * w
        env.step(act)
    env.object.write_root_state_to_sim(hold)

    # ★palm 추종오차 — tip attractor 가 palm attractor 를 이기면 팔이 손끝 목표를
    #   따라가 파지 자체가 성립하지 않는다(첫 측정 perr 683mm).
    palm_now = env.robot.data.body_pos_w[:, env.palm_idx] - env.scene.env_origins
    perr = float((env.palm_targets[:, :3] - palm_now).norm(dim=-1).mean()) * 1000.0
    dq_hand = (env.robot.data.joint_pos[:, env._hand_t] - q_before).abs().mean()
    f, wrapped, mid, dist, _, _ = env._contact()
    thr = float(env.cfg.contact_force_threshold)
    n_touch = (f > thr).float().sum(dim=1).mean()
    env_frac = ((mid > thr) | (dist > thr)).float()[:, env._env_f].mean()
    # 손끝이 목표에 얼마나 도달했는가 — 오차가 0 이면 힘도 0 이라는 것이 이 조사의 요지다
    tip_err = 0.0
    if args.mode in ("tip", "curl"):
        po, R = env._palm_frame(env.fabric_q.detach())
        tips_w = (env.robot.data.body_pos_w[:, env._tip_t]
                  - env.scene.env_origins[:, None, :])
        rel = torch.einsum("bij,bkj->bki", R.transpose(1, 2), tips_w - po[:, None, :])
        a_g = goal_a.view(N, -1, 3)
        want = env._tip_home + torch.where(
            a_g >= 0.0, a_g * (env._tip_hi - env._tip_home), a_g * (env._tip_home - env._tip_lo))
        tip_err = float((rel - want).norm(dim=-1).mean()) * 1000.0
    return (float(f.max()), float(f.sum(dim=1).mean()), float(n_touch),
            (float(env_frac), tip_err, perr, float(dq_hand)))


if args.mode == "tip":
    LEVELS = [args.radius]
    head = "반경[mm]"
elif args.mode == "curl":
    LEVELS = [0.020, 0.040, 0.060]
    head = "말기깊이[mm]"
else:
    LEVELS = [0.80, 0.85, 0.90, 0.95, 1.00]
    head = "폐합지령"

print(f"\n=== 파지력 스윕 · mode={args.mode} · {N}env ===", flush=True)
print(f"{head:>10s} {'Fmax[N]':>9s} {'F합[N]':>9s} {'접촉손가락':>10s} {'감쌈':>7s} "
      f"{'손끝오차[mm]':>12s} {'palm오차[mm]':>12s} {'Δ손[rad]':>10s}", flush=True)
for lv in LEVELS:
    fmax, fsum, ntouch, (ef, terr, dfab, dq) = measure(lv)
    label = f"{lv*1000:.0f}" if args.mode in ("tip", "curl") else f"{lv:.2f}"
    print(f"{label:>10s} {fmax:9.2f} {fsum:9.2f} {ntouch:10.2f} {ef:7.3f} {terr:12.1f} "
          f"{dfab:12.1f} {dq:10.4f}", flush=True)

env.close()
app.close()

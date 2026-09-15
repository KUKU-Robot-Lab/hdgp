"""pour_fabric_mimic 보상 입력 palm_axes 앞 3칸이 실제 손바닥 법선인지 sim 에서 확인 (09.15, 트랙 CLAUDE.md "보상 입력 손바닥 축 정정").

팔은 앵커에 두고 손만 편 상태 → 끝까지 닫는다. 손바닥 기준 손끝 이동을 palm 링크 열 0/1/2 에 투영 —
법선 열(ctx_palm_normal_col=2)에 큰 양수, 옛 열 0 에 작은 값이어야 한다. ctx 앞 3칸 == 열 2 도 대조.

    ./isaaclab.sh -p scripts/reinforcement_learning/probes/probe_pour_fabric_mimic_palm_normal.py \
        --num_envs 4 --reward_code_path <abs>/reward_gen/pour_bi_rh/iter_03/compute_reward.py
"""
import argparse, json
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(ap)
ap.add_argument("--task", default="open-rh_b_pour_fab_mimic"); ap.add_argument("--num_envs", type=int, default=4)
ap.add_argument("--reward_code_path", required=True)
args = ap.parse_args(); args.headless = True
app = AppLauncher(args).app
import gymnasium as gym, torch
from isaaclab_tasks.utils import parse_env_cfg
import openarm.tasks  # noqa
import openarm.agnostic.tasks.pour_fabric_mimic.config  # noqa
cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
cfg.reward_code_path = args.reward_code_path
env = gym.make(args.task, cfg=cfg).unwrapped
env.reset()
N, A = env.num_envs, env.cfg.action_space; half = A // 2; hold = int(env.cfg.hold_steps)
def snap():
    out = {}
    for tag, rig in (("src", env.src), ("rcv", env.rcv)):
        R = rig.palm_R(); out[tag] = dict(P=rig.palm_pos().clone(), R=R.clone(), T=rig.tips_pos().clone(),
                                          ctx=env._ctx_palm_axes(rig)[:, :3].clone(), clos=float(rig.closure().mean()))
    return out
a = torch.zeros(N, A, device=env.device)
for _ in range(hold + 30): env.step(a)
s0 = snap()
a[:, 6:half] = 1.0; a[:, half + 6:] = 1.0
for _ in range(150): env.step(a)
s1 = snap()
res = {}
for tag in ("src", "rcv"):
    R, P, T0, T1 = s0[tag]["R"], s0[tag]["P"], s0[tag]["T"], s1[tag]["T"]
    d = (T1 - T0) - (s1[tag]["P"] - P).unsqueeze(1)                  # 손바닥 기준 손끝 이동 (N,F,3)
    proj = lambda col: float((d * R[:, :, col].unsqueeze(1)).sum(-1).mean())
    off = T0.mean(1) - P
    res[tag] = {"ctx_eq_col2": float((s0[tag]["ctx"] - R[:, :, 2]).abs().max()),
                "closure_open_closed": (s0[tag]["clos"], s1[tag]["clos"]),
                "tip_move_mm_along_col0": 1000 * proj(0), "col1": 1000 * proj(1), "col2_new_normal": 1000 * proj(2),
                "open_tip_offset_mm_col0": 1000 * float((off * R[:, :, 0]).sum(-1).mean()),
                "open_tip_offset_mm_col2": 1000 * float((off * R[:, :, 2]).sum(-1).mean())}
print("[PALM_NORMAL]", json.dumps(res, indent=1))
app.close()

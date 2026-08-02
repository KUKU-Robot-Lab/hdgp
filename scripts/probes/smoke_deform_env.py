"""Gate B 스모크: deformable cup env 부팅 → reset → step. obs 차원(133)·NaN·패널각 확인.

articulation cup(패널 스프링)이 2048-env 아닌 소규모에서 실제로 스폰/리셋/스텝되고,
obs/action 차원이 rigid 태스크와 동일(133/action)한지, 파지 스텝에서 NaN이 안 나는지 검증.

사용:
  cd /home/user/rl_ws/hdgp
  /home/user/rl_ws/IsaacLab/isaaclab.sh -p scripts/probes/smoke_deform_env.py --num_envs 16
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str,
                    default="open-tesol_r_grasp_adapt_deform-lstm")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--stiffness", type=float, default=-1.0,
                    help=">0이면 패널 actuator stiffness override(변형 유도 검증용)")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    if args.stiffness > 0.0:
        cfg.cup_cfg.actuators["panels"].stiffness = args.stiffness
        print(f"[smoke] override panel stiffness -> {args.stiffness}", flush=True)
    env = gym.make(args.task, cfg=cfg).unwrapped
    obs_dim = int(env.cfg.num_observations)
    act_dim = int(env.cfg.num_actions)
    print(f"[smoke] cup_is_articulated={env.cfg.cup_is_articulated}", flush=True)
    print(f"[smoke] obs_space={obs_dim} act_space={act_dim}", flush=True)
    print(f"[smoke] cup type={type(env.cup).__name__} "
          f"num_joints={getattr(env.cup, 'num_joints', 'NA')}", flush=True)

    obs, _ = env.reset()
    policy_obs = obs["policy"] if isinstance(obs, dict) else obs
    print(f"[smoke] reset obs shape={tuple(policy_obs.shape)} "
          f"nan={bool(torch.isnan(policy_obs).any())}", flush=True)

    max_panel_deg = 0.0
    nan_seen = False
    for t in range(args.steps):
        act = torch.zeros((args.num_envs, act_dim), device=env.device)
        # 손가락을 닫는 방향으로 약하게 밀어 패널 접촉 유도(전 action +0.3)
        act[:] = 0.3
        obs, rew, term, trunc, info = env.step(act)
        policy_obs = obs["policy"] if isinstance(obs, dict) else obs
        if torch.isnan(policy_obs).any() or torch.isnan(rew).any():
            nan_seen = True
            print(f"[smoke] NaN at step {t}", flush=True)
            break
        q = env.cup.data.joint_pos  # (N, n_joints)
        max_panel_deg = max(max_panel_deg, float(q.abs().max()) * 180.0 / 3.14159265)

    # 접촉 필터 검증: tip force(force_matrix 합)가 계산되는지(다중-body 필터 정상)
    tip_f = env.contact_force_xyz_raw  # (N,5,3)
    tip_fmax = float(tip_f.norm(dim=-1).max())
    tip_contacts = int(env.binary_contact_buf.sum())
    print(f"[smoke] tip_force_max={tip_fmax:.4f} tip_contacts(all env)={tip_contacts} "
          f"nan={bool(torch.isnan(tip_f).any())}", flush=True)

    # Gate C 파이프 확인: radial(=deg)·reward/damage·buckle_rate·dose
    ex = env.extras
    print(f"[smoke] radial(deg)={float(ex.get('task/radial_compression', -1)):.3f} "
          f"reward/damage={float(ex.get('reward/damage', 0)):.4f} "
          f"buckle_rate={float(ex.get('task/buckle_rate', -1)):.3f} "
          f"dose={float(ex.get('task/damage_dose', -1)):.4f}", flush=True)
    print(f"[smoke] {args.steps} steps done. nan_seen={nan_seen} "
          f"max_panel_deg={max_panel_deg:.3f}", flush=True)
    verdict = ("PASS" if (not nan_seen and obs_dim == 133)
               else "FAIL")
    print(f"[smoke] GATE B smoke => {verdict}", flush=True)
    env.close()
    app.close()


if __name__ == "__main__":
    main()

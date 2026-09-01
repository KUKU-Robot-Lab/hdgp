#!/usr/bin/env python3
"""정책의 **관측 계약을 env 에서 직접 뽑는다** — 이름·차원·오프셋 + 표본 벡터.

배포용 obs 빌더는 학습 env 와 **한 칸도 어긋나면 안 된다.** 어긋나면 정책이 죽는 게
아니라 **조용히 이상하게 돈다** — 이 저장소 이력의 반복되는 사고다(154 vs 155,
홈 불일치, 게인 불일치). 그래서 손으로 옮겨 적지 않고 여기서 뽑는다.

  · ManagerBased (좌팔 grasp_sensor_v2) — `ObservationManager` 가 항 이름과 차원을 안다.
  · Direct       (우팔 grasp_s2r)       — `torch.cat` 리터럴을 소스에서 뽑는 쪽이 정확하다
                                          (`sim2real/scripts/obs_contract.py`). 여기서는
                                          **총 차원과 표본 벡터**만 확인한다.

산출물(json): 항별 이름·차원·오프셋 · 표본 obs · 그 순간의 로봇/물체 상태.
표본과 상태가 함께 있어야 배포 빌더를 **같은 입력으로** 대조할 수 있다.

    ../IsaacLab/isaaclab.sh -p scripts/probes/probe_obs_layout.py \\
        --task open-grip_l_grasp_sensor_v2-play --out /tmp/left_obs.json --headless
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", type=Path, default=None,
                    help="주면 그 런의 cfg 로 되씌운다 (계약은 런마다 다를 수 있다)")
parser.add_argument("--steps", type=int, default=8, help="표본을 뜨기 전 굴릴 스텝")
parser.add_argument("--num-envs", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--gui", action="store_true")

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "source" / "openarm"), str(_REPO / "scripts" / "tools")):
    sys.path.insert(0, _p)

from isaaclab.app import AppLauncher                              # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
args.headless = not args.gui
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym                                           # noqa: E402
import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402

import openarm  # noqa: E402,F401
import openarm.tasks  # noqa: E402,F401
from isaaclab_tasks.utils.hydra import hydra_task_config          # noqa: E402

from run_cfg_restore import restore_run_cfg_if_available          # noqa: E402


def _term_layout(env) -> list[dict]:
    """ManagerBased 면 항별 이름·차원, Direct 면 빈 목록."""
    om = getattr(env, "observation_manager", None)
    if om is None:
        return []
    names = om.active_terms.get("policy", [])
    dims = [int(np.prod(s)) for s in om.group_obs_term_dim.get("policy", [])]
    out, off = [], 0
    for name, dim in zip(names, dims):
        out.append({"name": name, "dim": dim, "offset": off})
        off += dim
    return out


def _robot_of(env):
    robot = getattr(env, "robot", None)
    return robot if robot is not None else env.scene["robot"]


@hydra_task_config(args.task, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    if args.checkpoint is not None:
        agent_cfg = restore_run_cfg_if_available(
            env_cfg, agent_cfg, resume_path=str(args.checkpoint),
            workspace_root=str(_REPO.parent))
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    for attr in ("enable_adr", "enable_success_adr"):
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, False)

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    obs, _ = env.reset()
    n_act = int(np.prod(env.action_space.shape[1:])) if env.action_space.shape else 0
    zero = torch.zeros(env.num_envs, n_act, device=env.device)
    for _ in range(args.steps):
        obs = env.step(zero)[0]

    policy = obs["policy"] if isinstance(obs, dict) else obs
    robot = _robot_of(env)
    layout = _term_layout(env)
    total = int(policy.shape[1])
    covered = sum(t["dim"] for t in layout)
    print(f"[계약] {args.task} · obs {total} 차원 · 항 {len(layout)}개 (합 {covered})")
    for t in layout:
        print(f"  {t['offset']:4d}..{t['offset']+t['dim']-1:<4d} {t['name']:32} {t['dim']:3d}")
    if layout and covered != total:
        print(f"  ⚠ 항 합계 {covered} ≠ 전체 {total} — 어딘가 빠졌다")
    if not layout:
        print("  (Direct env — 항 목록이 없다. obs_contract.py 로 소스에서 뽑을 것)")

    def _f(t):  # (env0 만)
        return [round(float(v), 6) for v in torch.as_tensor(t)[0].flatten()]

    payload = {
        "task": args.task,
        "obs_dim": total,
        "action_dim": n_act,
        "terms": layout,
        "sample_obs": _f(policy),
        "state": {
            "joint_names": list(robot.joint_names),
            "joint_pos": _f(robot.data.joint_pos),
            "joint_vel": _f(robot.data.joint_vel),
            "joint_pos_target": _f(robot.data.joint_pos_target),
            "body_names": list(robot.body_names),
            "body_pos_env_local": [
                round(float(v), 6)
                for v in (robot.data.body_pos_w[0] - env.scene.env_origins[0]).flatten()],
            "body_quat_wxyz": _f(robot.data.body_quat_w),
        },
    }
    for name in ("cup", "object", "left_target_cup"):
        obj = getattr(env, name, None)
        if obj is None:
            try:
                obj = env.scene[name]
            except (KeyError, TypeError):
                obj = None
        if obj is not None and hasattr(obj, "data"):
            payload["state"][f"{name}_pos_env_local"] = [
                round(float(v), 6)
                for v in (obj.data.root_pos_w[0] - env.scene.env_origins[0])]
            payload["state"][f"{name}_quat_wxyz"] = _f(obj.data.root_quat_w)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[저장] {args.out}")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code or 0)

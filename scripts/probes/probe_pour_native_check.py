#!/usr/bin/env python3
"""e1_pour1 체크포인트가 **자기 env(warm 텔레포트 리셋)** 에서 붓는지 — 분리 판정.

통합 러너에서 교차 0 이 나올 때, "체크포인트 품질" 과 "통합 결함" 을 가른다.
env.step 을 그대로 쓰므로 리셋·보상·판정 전부 네이티브다. 성공 집계는 env 가
스스로 하는 `_successful_episodes/_total_episodes` 를 읽는다.

★play 세션은 ADR 레벨 0 에서 시작한다(기본 난이도) — 여기서도 못 부으면
  체크포인트/실행 방식 문제, 여기서 잘 부으면 통합 러너 쪽 결함이다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", default="open-tesol_r_pour_sensor-play-lstm")
parser.add_argument("--checkpoint", type=Path,
                    default=Path("/home/user/rl_ws/sim2real/logs/policy/pour_e1/nn/e1_pour1_ep6000.pth"))
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=2600, help="정책 스텝 (에피소드 1200)")
parser.add_argument("--stochastic", action="store_true", help="mu 대신 샘플링")
parser.add_argument("--receiver-pos", default="",
                    help="받는컵 상수 위치 오버라이드 'x,y,z' — 통합 씬의 실측 받는점으로 "
                         "궤적을 다시 뽑을 때 (좌 REST 도착 실측: 0.265,0.045,0.296)")
parser.add_argument("--record-traj", type=Path, default=None,
                    help="upright 시작 성공 에피소드의 우팔 실측 관절 궤적을 npz 로 저장")
parser.add_argument("--save-obs", type=Path, default=None,
                    help="리셋 직후 첫 obs 표본을 npz 로 — 통합 러너 obs0 대조 기준")
_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "source" / "openarm"), str(_REPO / "scripts" / "tools")):
    sys.path.insert(0, _p)

from isaaclab.app import AppLauncher                              # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math                                                       # noqa: E402
import gymnasium as gym                                           # noqa: E402
import torch                                                      # noqa: E402

import openarm  # noqa: E402,F401
import openarm.tasks  # noqa: E402,F401
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config          # noqa: E402
from rl_games.common import env_configurations, vecenv            # noqa: E402
from rl_games.torch_runner import Runner                          # noqa: E402
from run_cfg_restore import restore_run_cfg_if_available          # noqa: E402


@hydra_task_config(args.task, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    agent_cfg = restore_run_cfg_if_available(
        env_cfg, agent_cfg, resume_path=str(args.checkpoint),
        workspace_root=str(_REPO.parent))
    if args.receiver_pos:
        _rp = tuple(float(v) for v in args.receiver_pos.split(","))
        env_cfg.left_target_cup_pos_env_local = _rp
        print(f"[오버라이드] 받는컵 상수 위치 → {_rp}", flush=True)
    env_cfg.scene.num_envs = args.num_envs
    env = gym.make(args.task, cfg=env_cfg)
    base = env.unwrapped
    device = agent_cfg["params"]["config"]["device"]
    wrapped = RlGamesVecEnvWrapper(
        env, device,
        agent_cfg["params"]["env"].get("clip_observations", math.inf),
        agent_cfg["params"]["env"].get("clip_actions", math.inf))
    vecenv.register("IsaacRlgWrapper",
                    lambda n, a, **kw: RlGamesGpuEnv(n, a, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: wrapped})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(args.checkpoint)
    agent_cfg["params"]["config"]["num_actors"] = base.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    player = runner.create_player()
    player.restore(str(args.checkpoint))
    player.reset()

    obs = wrapped.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    player.get_batch_size(obs, 1)
    if player.is_rnn:
        player.init_rnn()

    prev_cross = torch.zeros(base.num_envs, dtype=torch.long, device=base.device)
    done_cross: list[int] = []
    # 시작 유형(upright warm vs deep-tilt boot) — 리셋 다음 스텝의 src_up_z 로 판정
    start_type = ["?"] * base.num_envs
    pending_type = [True] * base.num_envs
    by_type: dict[str, list[int]] = {"upright": [], "tilted": []}
    # upright(warm) 시작의 **스텝1** obs 만 모은다 — 리셋 직후 obs 는 기하 버퍼가
    # 낡아(0) 무의미하고, deep-tilt 시작은 러너와 비교 대상이 아니다.
    upright_obs: list = []
    since_reset = torch.zeros(base.num_envs, dtype=torch.long, device=base.device)
    # 궤적 기록 — env 별 진행 버퍼, upright 시작 & done 시 교차>=15 면 채택
    traj_buf: dict[int, list] = {i: [] for i in range(base.num_envs)}
    picked = {}
    arm_ids = base.arm_dof_indices
    with torch.inference_mode():
        for step in range(args.steps):
            action = player.get_action(player.obs_to_torch(obs),
                                       is_deterministic=not args.stochastic)
            obs, _, dones, _ = wrapped.step(action)
            if isinstance(obs, dict):
                obs = obs["obs"]
            if player.is_rnn and player.states is not None:
                for s in player.states:
                    s[:, dones, :] = 0.0
            since_reset += 1
            for i in range(base.num_envs):
                if pending_type[i]:
                    _z = float(base._source_up_axis_w[i, 2])
                    if abs(_z) > 1e-6:
                        start_type[i] = "upright" if _z > 0.9 else "tilted"
                        pending_type[i] = False
                        if (args.save_obs is not None and start_type[i] == "upright"
                                and len(upright_obs) < 60):
                            upright_obs.append(
                                torch.as_tensor(obs)[i].detach().cpu().numpy())
            if args.record_traj is not None and not picked:
                q = base.robot.data.joint_pos[:, arm_ids]
                qd = base.robot.data.joint_vel[:, arm_ids]
                cz = base.cup.data.root_pos_w[:, 2] - base.scene.env_origins[:, 2]
                for i in range(base.num_envs):
                    traj_buf[i].append((q[i].detach().cpu().numpy().copy(),
                                        qd[i].detach().cpu().numpy().copy(),
                                        float(cz[i]),
                                        int(base._bead_cross_count[i])))
            d = dones.nonzero(as_tuple=False).flatten()
            for i in d.tolist():
                done_cross.append(int(prev_cross[i]))
                if start_type[i] != "?":
                    by_type[start_type[i]].append(int(prev_cross[i]))
                if (args.record_traj is not None and not picked
                        and start_type[i] == "upright" and int(prev_cross[i]) >= 15
                        and len(traj_buf[i]) > 50):
                    import numpy as _np
                    rows = traj_buf[i]
                    picked["env"] = i
                    picked["crossed"] = int(prev_cross[i])
                    args.record_traj.parent.mkdir(parents=True, exist_ok=True)
                    _np.savez_compressed(
                        args.record_traj,
                        arm_q=_np.stack([r[0] for r in rows]),
                        arm_qd=_np.stack([r[1] for r in rows]),
                        cup_z=_np.array([r[2] for r in rows], dtype=_np.float32),
                        crossed=_np.array([r[3] for r in rows], dtype=_np.int64),
                        meta_receiver=_np.array(
                            base.cfg.left_target_cup_pos_env_local, dtype=_np.float32),
                        meta_step_dt=_np.float32(base.step_dt),
                        meta_final_cross=_np.int64(int(prev_cross[i])))
                    print(f"[궤적] env{i} upright 성공(교차 {int(prev_cross[i])}) "
                          f"{len(rows)}스텝 → {args.record_traj}", flush=True)
                traj_buf[i] = []
                pending_type[i] = True
                since_reset[i] = 0
            prev_cross.copy_(base._bead_cross_count)
            if step % 200 == 0:
                print(f"  [{step:4d}] 교차 {base._bead_cross_count.tolist()} · "
                      f"에피소드 {int(base._total_episodes)} · "
                      f"성공 {int(base._successful_episodes)}", flush=True)

    n = max(int(base._total_episodes), 1)
    if args.save_obs is not None and upright_obs:
        import numpy as _np
        args.save_obs.parent.mkdir(parents=True, exist_ok=True)
        _np.savez_compressed(args.save_obs, obs=_np.stack(upright_obs))
        print(f"[표본] upright 시작 스텝1 obs {len(upright_obs)}개 → {args.save_obs}", flush=True)
    for k, v in by_type.items():
        if v:
            _ok = sum(1 for c in v if c >= 10)
            print(f"[유형] {k} 시작 {len(v)}회 · 교차≥10 {_ok}회 ({_ok / len(v):.2f}) · "
                  f"분포 {sorted(v)}", flush=True)
    print(f"\n[네이티브] 에피소드 {int(base._total_episodes)} · "
          f"성공 {int(base._successful_episodes)} ({base._successful_episodes / n:.2f}) · "
          f"pose 성공 {int(base._pose_successful_episodes)} · "
          f"종료시 교차 분포 {sorted(done_cross)}", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    import os
    os._exit(int(code or 0))

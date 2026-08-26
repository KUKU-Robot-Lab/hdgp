# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""리프트 궤적 실측 — "낚아채듯 든다"를 시간축 수치로 확정한다.

왜 필요한가: hier_test2 영상에서 사용자 관찰 — "컵을 위로 올리는 게 아니라
오른쪽에서 왼쪽으로 낚아채듯이 든 다음 제자리로 온다". TFEvents 는 에폭 평균이라
**시간축이 없다** — 언제 얼마나 옆으로 나갔다 돌아오는지를 못 본다.

목표 정의(env 실코드): goal = 컵 정착점 + (0,0,0.15). **수평 이동 요구 0**.
따라서 xy 이탈은 전부 순손실이고, 이 프로브는 그 이탈의 시간 프로파일을 잰다.

재는 것 (매 스텝, env 평균 + 최악 env):
  · 컵 위치 − 스폰점: Δx, Δy (수평 이탈 성분별), Δz (수직 상승)
  · xy_disp = √(Δx²+Δy²), 그리고 에피소드 내 **최대 xy 이탈과 그 시점**
  · tilt (컵 z축 vs world z)
  · d_goal, 게이트 μ/ν/ρ 진입 시점
  · 컵 속도 (수평/수직 분해) — "낚아챔"은 수평 속도 스파이크로 나타난다

출력: 스텝별 표(20스텝 간격) + 요약(최대 이탈·시점·복귀 시점·기울기 정점).

실행:
    PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/probes/probe_lift_trajectory.py \
        --checkpoint <ckpt.pth> --num_envs 256 --steps 600
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-sens_r_grasp_sensor-lstm")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--print_every", type=int, default=20)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab.envs import DirectRLEnvCfg  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import openarm.tasks  # noqa: E402,F401


def _unwrap(env):
    e = env.unwrapped
    while hasattr(e, "env"):
        e = e.env.unwrapped
    return e


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    # ★리셋 배제 — 궤적이 중간에 끊기면 시간축이 오염된다
    env_cfg.episode_length_s = 1.0e6
    agent_cfg["params"]["seed"] = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg)
    core = _unwrap(env)
    _e = agent_cfg["params"]["env"]
    clip_obs = float(_e.get("clip_observations", 1e6))
    clip_act = float(_e.get("clip_actions", 1e6))
    env = RlGamesVecEnvWrapper(env, agent_cfg["params"]["config"].get("device", "cuda:0"),
                               clip_obs, clip_act,
                               _e.get("obs_groups"), _e.get("concate_obs_groups", True))
    vecenv.register("IsaacRlgWrapper",
                    lambda cn, na, **kw: RlGamesGpuEnv(cn, na, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args_cli.checkpoint
    agent_cfg["params"]["config"]["num_actors"] = core.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(args_cli.checkpoint)

    torch.manual_seed(args_cli.seed)
    with torch.inference_mode():
        obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    agent.reset()
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    dev = core.device
    N = core.num_envs
    spawn = core.object_spawn_pos.clone()          # (N,3) env-local 정착점
    goal = core.goal_pos.clone()

    # 시계열 버퍼
    T = args_cli.steps
    tr = {k: torch.zeros(T, N, device=dev) for k in
          ("dx", "dy", "dz", "xy", "tilt", "dgoal", "vxy", "vz")}
    gate_first = {k: torch.full((N,), -1, device=dev, dtype=torch.long)
                  for k in ("mu", "nu", "rho")}

    up_w = torch.tensor([0.0, 0.0, 1.0], device=dev)
    prev_pos = None

    for t in range(T):
        with torch.inference_mode():
            obs_t = agent.obs_to_torch(obs)
            # 결정론 — 학습된 μ 의 궤적을 본다(σ 노이즈가 섞이면 낚아챔과 구분 안 됨)
            actions = agent.get_action(obs_t, is_deterministic=True)
            obs, _r, _d, _i = env.step(actions)
            if isinstance(obs, dict):
                obs = obs["obs"]

        pos = core._env_local(core.object.data.root_pos_w)
        quat = core.object.data.root_quat_w
        # 컵 +z 축 world 성분 → tilt
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        up_z = 1.0 - 2.0 * (x * x + y * y)
        tilt = torch.rad2deg(torch.acos(up_z.clamp(-1.0, 1.0)))

        d = pos - spawn
        tr["dx"][t] = d[:, 0]
        tr["dy"][t] = d[:, 1]
        tr["dz"][t] = d[:, 2]
        tr["xy"][t] = torch.norm(d[:, :2], dim=-1)
        tr["tilt"][t] = tilt
        tr["dgoal"][t] = torch.norm(pos - goal, dim=-1)
        if prev_pos is not None:
            v = (pos - prev_pos) / core.step_dt
            tr["vxy"][t] = torch.norm(v[:, :2], dim=-1)
            tr["vz"][t] = v[:, 2]
        prev_pos = pos.clone()

        # 게이트 최초 진입 시점 — per-env 로 보상 게이트와 같은 임계를 직접 계산
        # (extras 는 배치 평균이라 못 쓴다). μ 는 접촉이 필요해 여기선 ν/ρ 만.
        hd = d[:, 2]
        for gk, cond in (("nu", hd >= float(core.cfg.stage_gate_lift_m)),
                         ("rho", tr["dgoal"][t] < float(core.cfg.stage_gate_transfer_m))):
            fresh = (gate_first[gk] < 0) & cond
            gate_first[gk][fresh] = t

    # ── 출력 ──────────────────────────────────────────────────────────
    P = args_cli.print_every
    print("\n" + "=" * 100, flush=True)
    print(f"리프트 궤적 실측 — {os.path.basename(args_cli.checkpoint)}", flush=True)
    print(f"env {N} · {T} 스텝 · 결정론(σ=0) · goal = spawn + (0,0,0.15)", flush=True)
    print("-" * 100, flush=True)
    print(f"{'step':>5} {'Δx(mm)':>8} {'Δy(mm)':>8} {'Δz(mm)':>8} {'xy이탈':>8} "
          f"{'p95xy':>8} {'tilt°':>7} {'p95tilt':>8} {'d_goal':>8} {'v_xy':>7}", flush=True)
    for t in range(0, T, P):
        row = {k: tr[k][t] for k in tr}
        print(f"{t:>5} {row['dx'].mean()*1e3:>8.1f} {row['dy'].mean()*1e3:>8.1f} "
              f"{row['dz'].mean()*1e3:>8.1f} {row['xy'].mean()*1e3:>8.1f} "
              f"{row['xy'].quantile(0.95)*1e3:>8.1f} {row['tilt'].mean():>7.2f} "
              f"{row['tilt'].quantile(0.95):>8.2f} {row['dgoal'].mean()*1e3:>8.1f} "
              f"{row['vxy'].mean()*1e3:>7.1f}", flush=True)

    # ── 요약: env 별 최대 이탈·시점 ───────────────────────────────────
    xy_max, xy_arg = tr["xy"].max(dim=0)              # (N,)
    tilt_max, tilt_arg = tr["tilt"].max(dim=0)
    dz_end = tr["dz"][-40:].mean(dim=0)
    xy_end = tr["xy"][-40:].mean(dim=0)
    print("-" * 100, flush=True)
    print("[요약 — env 분포]", flush=True)
    print(f"  최대 xy 이탈    평균 {xy_max.mean()*1e3:6.1f}mm · p95 {xy_max.quantile(0.95)*1e3:6.1f}mm"
          f" · 발생 시점 중앙값 step {int(xy_arg.float().median())}", flush=True)
    print(f"  최대 tilt       평균 {tilt_max.mean():6.2f}° · p95 {tilt_max.quantile(0.95):6.2f}°"
          f" · 발생 시점 중앙값 step {int(tilt_arg.float().median())}", flush=True)
    print(f"  종단(마지막 40스텝)  Δz {dz_end.mean()*1e3:6.1f}mm · xy 이탈 {xy_end.mean()*1e3:6.1f}mm",
          flush=True)
    for gk, name in (("nu", "ν 리프트(h≥5cm)"), ("rho", "ρ 목표권(d<8cm)")):
        g = gate_first[gk]
        hit = g >= 0
        if hit.any():
            print(f"  {name:<18} 도달 {hit.float().mean()*100:5.1f}% · "
                  f"최초 진입 중앙값 step {int(g[hit].float().median())}", flush=True)
        else:
            print(f"  {name:<18} 도달 0%", flush=True)
    # 낚아챔 판정 재료: 리프트 구간(ν 진입 전후 30스텝)의 수평 속도 vs 종단
    nu_med = int(gate_first["nu"][gate_first["nu"] >= 0].float().median()) \
        if (gate_first["nu"] >= 0).any() else T // 3
    a, b = max(1, nu_med - 15), min(T, nu_med + 15)
    v_lift = tr["vxy"][a:b].mean()
    v_end = tr["vxy"][-40:].mean()
    print(f"  수평속도  리프트 창(step {a}-{b}) {v_lift*1e3:6.1f}mm/s · "
          f"종단 {v_end*1e3:6.1f}mm/s  → 비율 {float(v_lift/max(v_end,1e-9)):.1f}배", flush=True)
    print("=" * 100, flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

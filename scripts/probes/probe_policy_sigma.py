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

"""정책 액션 노이즈(σ)가 과제 수행을 파괴하는가 — 같은 체크포인트 결정론 vs 확률론.

왜 필요한가: `agnostic/grasp_sensor` 가 **작동하던 상태에서 두 번 붕괴**했다
(lstm_test14 ep1150, lstm_test15 ep35). 두 번째는 `lr_schedule: adaptive` 로
KL 이 전혀 안 튀었는데도(0.012~0.024) 같은 방식으로 무너졌다 → 원인은 LR
스케줄러가 아니다.

정점 체크포인트(ep1102, rew 8958.9)에서 직접 읽은 값:
    a2c_network.sigma  평균 0.946 · 최소 0.510 · 최대 1.273
    sigma_init = const 0 → 시작 σ = 1.0. 1,100 에폭 학습하고 0.946.

이 트랙의 액션은 **절대 palm pose** 라 σ≈0.95 면 실행 액션이 palm 박스 전체에
걸친 거의 무작위 표본이다. 그리고 ADR difficulty 가 전 구간 0.0000 이라
**obs 노이즈도 도메인 랜덤화도 없다** — 남은 확률원은 컵 스폰과 액션 표본뿐이다.

여기서 가르는 것:
  · 결정론(σ=0)에서 접근→파지→리프트→이송이 되면 → μ 는 멀쩡, 노이즈가 파괴 →
    처방은 σ 억제
  · 결정론에서도 엉망이면 → μ 자체가 안 배워진 것 → 보상 설계로 돌아가야 함

★같은 시드·같은 컵 스폰을 쓴다. 안 그러면 스폰 산포가 두 조건을 갈라 놓는다
  (probe_seqclose 가 정확히 그 함정에 빠져 거짓 단조성을 만든 전례).

실행:
    PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/probes/probe_policy_sigma.py \
        --task open-sens_r_grasp_sensor-lstm --num_envs 256 --steps 600 \
        --checkpoint <peak.pth>
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
parser.add_argument("--steps", type=int, default=600, help="조건당 롤아웃 스텝 수")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--seed", type=int, default=12345)
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


# 재는 지표. env 가 `_get_rewards` 에서 매 스텝 갱신하는 extras 키다.
STEP_KEYS = (
    "task/five_frac", "task/d_graspcenter", "task/object_height_delta",
    "task/object_tilt_deg", "task/upright_q", "task/perp_q", "task/orient_q",
    "task/goal_dist", "task/xy_disp", "task/grasp_q", "task/near_q",
    "task/syn_close", "reward/total",
)
# 에피소드 종료 시점에만 갱신되는 단계 성공률.
STAGE_KEYS = tuple(f"task/stage/{n}" for n in
                   ("approach", "grasp", "lift", "transfer", "stay"))


def _unwrap(env):
    """RlGamesVecEnvWrapper → DirectRLEnv 본체."""
    e = env.unwrapped
    while hasattr(e, "env"):
        e = e.env.unwrapped
    return e


def _rollout(agent, env, core, steps: int, deterministic: bool) -> dict[str, float]:
    """한 조건을 롤아웃하고 지표 평균을 낸다.

    ★`agent.reset()` + `init_rnn()` 을 조건마다 다시 부른다 — LSTM 은닉이 이전
      조건의 것을 물고 가면 두 번째 조건이 첫 번째에 오염된다.
    """
    torch.manual_seed(args_cli.seed)          # 컵 스폰을 두 조건에서 동일하게
    # ★reset 도 inference_mode 안에서 부른다. 앞 조건의 `env.step` 이 inference_mode
    #   안에서 돌면 env 버퍼가 inference tensor 가 되어, 밖에서 `_reset_idx` 가
    #   그것을 제자리 수정하려다 RuntimeError 로 죽는다(두 번째 조건에서 재현).
    with torch.inference_mode():
        obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    agent.reset()
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    acc: dict[str, list[float]] = {k: [] for k in STEP_KEYS}
    stage: dict[str, list[float]] = {k: [] for k in STAGE_KEYS}
    act_std: list[float] = []
    prev = None

    for _ in range(steps):
        with torch.inference_mode():
            obs_t = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_t, is_deterministic=deterministic)
            if prev is not None:
                act_std.append(float((actions - prev).abs().mean()))
            prev = actions.clone()
            obs, _rew, _done, _info = env.step(actions)
            if isinstance(obs, dict):
                obs = obs["obs"]
        ex = core.extras
        for k in STEP_KEYS:
            v = ex.get(k)
            if v is not None:
                acc[k].append(float(v))
        for k in STAGE_KEYS:
            v = ex.get(k)
            if v is not None:
                stage[k].append(float(v))

    out = {k: (sum(v) / len(v) if v else float("nan")) for k, v in acc.items()}
    out.update({k: (sum(v) / len(v) if v else float("nan")) for k, v in stage.items()})
    out["_action_step"] = sum(act_std) / len(act_std) if act_std else float("nan")
    return out


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg["params"]["seed"] = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg)
    core = _unwrap(env)
    # ★clip 은 반드시 agent 설정에서 읽는다. 하드코딩하면(1e6) 관측이 ±5 로 안 잘리고
    #   액션이 ±1 로 안 잘려 **정책이 학습과 다른 분포에서 돈다** — 실제로 그렇게 재서
    #   액션 스텝이 513,324 로 나왔다(정상은 ≤2).
    _e = agent_cfg["params"]["env"]
    clip_obs = float(_e.get("clip_observations", 1e6))
    clip_act = float(_e.get("clip_actions", 1e6))
    print(f"[probe] clip_observations={clip_obs} · clip_actions={clip_act}", flush=True)
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

    # 체크포인트가 들고 있는 σ 를 그대로 보고한다(추정이 아니라 실측).
    sig = None
    for k, v in agent.model.state_dict().items():
        if k.endswith("sigma"):
            sig = v.detach().float().exp()
    print("\n" + "=" * 74, flush=True)
    print(f"체크포인트: {os.path.basename(args_cli.checkpoint)}", flush=True)
    if sig is not None:
        print(f"정책 σ : 평균 {sig.mean():.4f} · 최소 {sig.min():.4f} · "
              f"최대 {sig.max():.4f}  (액션 {sig.numel()}D, 범위 [-1,1])", flush=True)
    print(f"env {core.num_envs} · 조건당 {args_cli.steps} 스텝 · seed {args_cli.seed}",
          flush=True)

    res = {}
    for name, det in (("결정론 σ=0", True), ("확률론 σ=0.95", False)):
        print(f"\n[{name}] 롤아웃...", flush=True)
        res[name] = _rollout(agent, env, core, args_cli.steps, det)

    names = list(res)
    print("\n" + "=" * 74, flush=True)
    print(f"{'지표':<26}{names[0]:>22}{names[1]:>22}", flush=True)
    print("-" * 74, flush=True)
    for k in STAGE_KEYS + STEP_KEYS + ("_action_step",):
        a, b = res[names[0]].get(k), res[names[1]].get(k)
        if a is None or b is None or (a != a and b != b):
            continue
        short = k.replace("task/", "").replace("reward/", "R:")
        print(f"{short:<26}{a:>22.4f}{b:>22.4f}", flush=True)
    print("=" * 74, flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

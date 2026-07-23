"""P3.1 — rl_games LSTM 정책 standalone 로딩 + action 재현 검증 (raw-app 라이브 루프용).

질문: isaaclab env 없이(dummy env 로 shape 만 제공) rl_games player 를 로드하고, 덤프된 obs 시퀀스를
먹여 action 이 isaaclab 롤아웃과 일치하는가(LSTM 상태 재현 포함). 통과하면 라이브 루프에서 정책 구동 가능.

방법: p3_dump_rollout.py 가 덤프한 env0 의 (actor_obs, action) 시퀀스 → player.get_action 재현 → 대조.
LSTM 은 per-env 독립이므로 init_rnn 후 같은 obs 순서면 같은 action 이 나와야 함.

실행: ./IsaacLab/isaaclab.sh -p hdgp/scripts/reinforcement_learning/probes/p3_1_policy_load.py \
        --checkpoint <lstm_test2 ckpt> --rollout hdgp/docs/eval/p3_rollout.npz
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--agent_yaml", type=str,
                    default="hdgp/log/rl_games/open-tesol/right/pour-v1/lstm_test2/params/agent.yaml")
parser.add_argument("--rollout", type=str, default="hdgp/docs/eval/p3_rollout.npz")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402
from isaaclab.utils.io import load_yaml  # noqa: E402

OBS_DIM = 55
ACT_DIM = 12
CRITIC_DIM = 144
DEV = "cuda:0"


class _DummyEnv:
    """rl_games player 가 obs/action shape 만 필요로 하므로 최소 구현."""
    def __init__(self, config_name, num_actors, **kw):
        self.num_agents = 1
        self.num_actors = num_actors
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,))
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,))
        self.state_space = gym.spaces.Box(-np.inf, np.inf, (CRITIC_DIM,))

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        return {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "state_space": self.state_space,
        }


def main():
    agent_cfg = load_yaml(os.path.abspath(args_cli.agent_yaml))
    agent_cfg["params"]["config"]["num_actors"] = 1
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = os.path.abspath(args_cli.checkpoint)

    vecenv.register("IsaacRlgWrapper",
                    lambda config_name, num_actors, **kw: _DummyEnv(config_name, num_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: _DummyEnv("rlgpu", 1, **kw)})

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(os.path.abspath(args_cli.checkpoint))
    agent.reset()
    if agent.is_rnn:
        agent.init_rnn()
    print(f"[P3.1] player 로드 OK | is_rnn={agent.is_rnn} | is_deterministic={agent.is_deterministic}", flush=True)

    roll = {k: v for k, v in np.load(os.path.abspath(args_cli.rollout)).items()}
    ref_obs = roll["actor_obs"]      # (T, 55)
    ref_act = roll["action"]         # (T, 12)
    T = ref_obs.shape[0]

    errs = []
    for t in range(T):
        obs_t = torch.tensor(ref_obs[t:t + 1], dtype=torch.float32, device=DEV)
        with torch.inference_mode():
            obs_t = agent.obs_to_torch(obs_t)
            act = agent.get_action(obs_t, is_deterministic=agent.is_deterministic)
        act = act.detach().cpu().numpy().reshape(-1)[:ACT_DIM]
        errs.append(np.abs(act - ref_act[t]).max())

    errs = np.array(errs)
    print(f"[P3.1] {T} 스텝 action 재현 | max|err|={errs.max():.3e} | mean|err|={errs.mean():.3e} | "
          f"worst step={int(np.argmax(errs))}", flush=True)
    tol = 1e-4
    ok = errs.max() < tol
    print(f"[P3.1] ===== 판정: {'PASS — 정책 standalone 로딩·LSTM 재현 정확, 라이브 루프 구동 가능' if ok else 'FAIL'} "
          f"(tol={tol:.0e}) =====", flush=True)
    if not ok:
        w = int(np.argmax(errs))
        print(f"[P3.1]   worst step {w}: ref={np.array2string(ref_act[w], precision=3)}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()

"""2단계 t2r 정책의 "놓기 게이트" 계측 — `ready = placed & resting` 가 뜨는가, 뜨면 손을 여는가.

배경. iter_00 학습에서 `t2r_reward/release` ≈ 0, `withdraw` == 0 이 250 epoch 내내 유지됐다.
생성된 보상은 `release = 1.5 · open_frac` 를 **`ready = placed & resting` 일 때만** 지급하고, 그 밖의
모든 상태에서 여는 것은 `premature_release` 로 벌한다. 따라서 두 가지 원인이 같은 로그를 만든다:

  ① 게이트가 너무 좁아 `ready` 자체가 거의 안 뜬다      → place_tolerance / resting_tol 문제
  ② `ready` 는 뜨는데 정책이 그래도 안 연다             → premature_release 가 여는 행동을 말려 죽였다

학습 로그는 `place/placed` 와 `place/resting` 을 각각의 평균으로만 남기므로 **논리곱을 알 수 없다**.
이 프로브는 그 논리곱과, `ready` 인 스텝에서의 `open_frac` 분포를 직접 잰다.

각 env 의 **첫 에피소드만** 집계한다(모두 같은 리셋에서 출발하므로 짧은 에피소드 편향이 없다 —
eval_iker.py 의 FirstEpisodeRecorder 와 같은 규약).

사용법:
    cd ~/rl_ws/hdgp-iker && TERM=xterm OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD/source/openarm \
        ../IsaacLab/_isaac_sim/python.sh scripts/iker/t2r2_ready_probe.py \
        --checkpoint <t2r2 .pth> --reward-code-path <reward_fn.py> --out <probe.json> --headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="2단계 t2r 놓기 게이트 계측 프로브.")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--reward-code-path", required=True, help="학습에 쓴 생성 보상 파일(같은 것을 써야 한다)")
parser.add_argument("--config-index", type=int, default=0)
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--no-noise", action="store_true", help="학습용 관측/액션 노이즈를 끈다")
parser.add_argument("--out", required=True, help="요약 JSON")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
sys.argv = [sys.argv[0]]
app = AppLauncher(args).app


def _hard_exit(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)
    print("PROBE FAILED", flush=True)
    os._exit(1)


sys.excepthook = _hard_exit

from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.agnostic.tasks.iker_shoe.config  # noqa: E402,F401  (gym id 등록)
from openarm.agnostic.modules.iker import run_files  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.iker_shoe_t2r_env_cfg import IkerShoeT2rEnvCfg  # noqa: E402
from openarm.agnostic.tasks.iker_shoe.policy_player import load_player, reset_player, step_player  # noqa: E402

TASK = "open-sens_l_iker_shoe_t2r"


class GateRecorder:
    """`_get_rewards` 를 감싸 매 스텝의 게이트 상태를 누적한다(첫 에피소드에 한해).

    `_get_dones` 가 먼저 돌아 `_t2r_last` 를 채우고, 그 다음 `_get_rewards` 가 불린다 — 그래서 여기서
    읽으면 이번 스텝의 판정값이 그대로 보인다. 리셋은 `_get_rewards` 뒤에 오므로 `_log_episode_end` 로
    "이 env 의 첫 에피소드가 끝났다"를 표시한다.
    """

    KEYS = ("ready", "placed", "resting", "still", "released", "ok")

    def __init__(self, u):
        n, dev = u.num_envs, u.device
        self.u = u
        self.live = torch.ones(n, dtype=torch.bool, device=dev)   # 첫 에피소드가 아직 안 끝난 env
        self.steps = torch.zeros(n, device=dev)                   # env 별 집계된 스텝 수
        self.counts = {key: torch.zeros(n, device=dev) for key in self.KEYS}
        self.open_sum = torch.zeros(n, device=dev)                # 전체 스텝의 open_frac 합
        self.open_max = torch.zeros(n, device=dev)                # 에피소드 중 최대 open_frac
        self.open_ready_sum = torch.zeros(n, device=dev)          # ready 인 스텝만의 open_frac 합
        self.stable_max = torch.zeros(n, device=dev)              # 연속 카운터 최고치
        self._get_rewards, self._log_episode_end = u._get_rewards, u._log_episode_end
        u._get_rewards, u._log_episode_end = self.get_rewards, self.log_episode_end

    def get_rewards(self):
        out = self._get_rewards()
        u, last = self.u, self.u._t2r_last
        live = self.live.float()
        placed, resting = last["placed"], last["resting"]
        ready = placed & resting
        # 생성 보상과 같은 정의: grip_norm -1(뱅크 파지) → 0, +1(프로필 개방) → 1
        open_frac = ((u._grip_scalar.reshape(-1) + 1.0) * 0.5).clamp(0.0, 1.0)
        masks = {"ready": ready, "placed": placed, "resting": resting,
                 "still": last["still"], "released": last["released"],
                 "ok": ready & last["still"] & last["released"]}
        for key, mask in masks.items():
            self.counts[key] += mask.float() * live
        self.steps += live
        self.open_sum += open_frac * live
        self.open_ready_sum += open_frac * ready.float() * live
        self.open_max = torch.maximum(self.open_max, open_frac * live)
        self.stable_max = torch.maximum(self.stable_max, last["stable_count"] * live)
        return out

    def log_episode_end(self, env_ids):
        # 최초 env.reset() 도 전 env 에 대해 이걸 부른다(그때 episode_length_buf 는 0). 그것을 "에피소드가
        # 끝났다" 로 세면 루프가 시작도 하기 전에 live 가 전부 False 가 되어 0 스텝으로 끝난다 —
        # eval_iker.py 의 FirstEpisodeRecorder 와 같은 길이 가드를 둔다.
        ended = env_ids[self.u.episode_length_buf[env_ids] > 0]
        self.live[ended] = False
        self._log_episode_end(env_ids)


def summarize(rec: GateRecorder) -> dict:
    steps = rec.steps.clamp(min=1.0)
    n_ready = rec.counts["ready"]

    def rate(key):
        """env 별 '그 조건이 참이었던 스텝 비율' 의 평균."""
        return round(float((rec.counts[key] / steps).mean()), 4)

    def any_env(key):
        """에피소드 중 한 번이라도 그 조건이 참이었던 env 의 비율."""
        return round(float((rec.counts[key] > 0).float().mean()), 4)

    ready_envs = n_ready > 0
    open_when_ready = rec.open_ready_sum[ready_envs] / n_ready[ready_envs] if bool(ready_envs.any()) else None
    return {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "reward_code_path": str(Path(args.reward_code_path).resolve()),
        "num_envs": rec.u.num_envs, "noise": not args.no_noise,
        "episode_steps_mean": round(float(rec.steps.mean()), 2),
        "step_rate": {key: rate(key) for key in GateRecorder.KEYS},
        "ever_true_env_frac": {key: any_env(key) for key in GateRecorder.KEYS},
        "open_frac": {
            "mean_all_steps": round(float((rec.open_sum / steps).mean()), 4),
            "max_per_env_mean": round(float(rec.open_max.mean()), 4),
            "max_per_env_p90": round(float(torch.quantile(rec.open_max, 0.9)), 4),
            "mean_on_ready_steps": round(float(open_when_ready.mean()), 4) if open_when_ready is not None else None,
        },
        "stable_count_max": {
            "mean": round(float(rec.stable_max.mean()), 3),
            "p90": round(float(torch.quantile(rec.stable_max, 0.9)), 3),
            "max": round(float(rec.stable_max.max()), 3),
            "target": int(rec.u.cfg.place.stable_steps),
            # 실제 성공률: 20스텝 연속을 끝까지 채운 env 의 비율. mean/p90 만으로는 "거의 다 온 것" 과
            # "끝낸 것" 이 구분되지 않아, 학습 로그의 sustained_success 와 직접 비교할 수가 없었다.
            "at_target_frac": round(float((rec.stable_max >= float(rec.u.cfg.place.stable_steps)).float().mean()), 4),
        },
        "gate_cfg": {
            "place_tolerance": float(rec.u.cfg.place.place_tolerance),
            "resting_tol": float(rec.u.cfg.place.resting_tol),
            "release_radius": float(rec.u.cfg.place.release_radius),
            "still_speed": float(rec.u.cfg.place.still_speed),
        },
    }


def make_env():
    cfg = IkerShoeT2rEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.config_index = args.config_index
    cfg.seed = args.seed
    cfg.add_noise = not args.no_noise
    cfg.reward_code_path = str(Path(args.reward_code_path).resolve())
    return gym.make(TASK, cfg=cfg)


def main() -> int:
    for path in (Path(args.checkpoint), Path(args.reward_code_path)):
        if not path.is_file():
            raise FileNotFoundError(path)
    env = make_env()
    u = env.unwrapped
    rec = GateRecorder(u)
    wrapped, agent = load_player(env, TASK, Path(args.checkpoint))
    with torch.inference_mode():
        obs = reset_player(wrapped, agent)
        steps = 0
        while bool(rec.live.any()) and steps < 3 * u.max_episode_length:
            obs, _ = step_player(wrapped, agent, obs)
            steps += 1
    if not bool((rec.steps > 0).any()):
        raise RuntimeError(f"{steps} 스텝 동안 집계된 에피소드가 없다")
    summary = summarize(rec)
    run_files.write_json(Path(args.out), {"schema": run_files.SCHEMA_VERSION, "summary": summary})
    print("PROBE " + json.dumps(summary), flush=True)
    return 0


os._exit(main())

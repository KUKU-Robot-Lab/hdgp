#!/usr/bin/env python3
"""pour lstm_test4 성공 에피소드의 양팔 궤적 추출 — S2R 는 재생으로 우회한다(사용자 방침).

★이 스크립트는 **worktree `hdgp_pour23`(커밋 9b43f40)** 에서만 돈다. 현재 hdgp 소스는
  NUM_ACTIONS 가 6 으로 갈려 이 체크포인트(action 15)를 재생할 수 없다.
★warm 캐시는 08.17 격리된 아카이브 본을 오버라이드로 쓴다(격리 해제 금지) — 이 런은
  구 자산(DG-5F) 시절이라 아카이브 캐시가 정합.
★기록은 매 스텝 env.step **직전**에 스냅샷 — 첫 스냅샷이 곧 에피소드 초기 상태다.
  done 스텝 뒤의 버퍼는 이미 리셋되어 있으므로(DirectRLEnv), 성공 판정은 내부 플래그가
  아니라 기록된 fill/spill 시계열로 후처리한다.

산출물(--out 디렉토리):
  pour_traj_env<i>_ep<j>.npz  선정 에피소드 전체 시계열(60 Hz)
  pour_init.npz               선정 에피소드 첫 프레임(양팔 palm/tcp pose + 전 관절 + 컵)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", default="open-tesol_b_pour_sensor-play-lstm")
parser.add_argument("--checkpoint", type=Path, default=Path(
    "/home/user/rl_ws/sim2real/logs/policy/pour_lstm_test4/nn/"
    "last_open-tesol_b_pour_sensor-lstm_ep_1300_rew__33892.785_.pth"))
parser.add_argument("--warm", type=Path, default=Path(
    "/home/user/rl_ws/archive/hdf5_2026-08-17_pre_dg5fs/_quarantined_from_hdgp_data/"
    "grasp_warm_tesollo.hdf5"))
parser.add_argument("--out", type=Path, default=Path(
    "/home/user/rl_ws/sim2real/logs/shadow/pour_traj"))
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--success_episodes", type=int, default=3, help="이만큼 성공을 모으면 종료")
parser.add_argument("--max_steps", type=int, default=6000, help="안전 상한(정책 스텝)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--gui", action="store_true")

# ★sys.path 는 AppLauncher **이전** — 안 그러면 옆 세션의 live hdgp 를 조용히 읽는다
#   (08.27 소스 오염 사고 재발 방지).
_WT = Path("/home/user/rl_ws/hdgp_pour23")
sys.path.insert(0, str(_WT / "source" / "openarm"))

from isaaclab.app import AppLauncher                              # noqa: E402
AppLauncher.add_app_launcher_args(parser)
# ★hydra_task_config 가 나머지 argv 를 자기 것으로 읽는다 — 우리 인자를 남기면
#   "unrecognized arguments" 로 죽는다(실측). IsaacLab 표준 패턴대로 잘라낸다.
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
args.headless = not args.gui

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math                                                        # noqa: E402
import numpy as np                                                 # noqa: E402
import torch                                                       # noqa: E402
import gymnasium as gym                                            # noqa: E402

import openarm.tasks                                               # noqa: E402,F401
import fabrics_sim                                                 # noqa: E402,F401
import openarm                                                     # noqa: E402
from isaaclab.envs import DirectRLEnvCfg                           # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config           # noqa: E402
from rl_games.common import env_configurations, vecenv             # noqa: E402
from rl_games.torch_runner import Runner                           # noqa: E402


def _assert_source_tree() -> None:
    """openarm/fabrics_sim 이 worktree 밖이면 즉시 죽는다 — 결과 오염 방지."""
    for mod in (openarm, fabrics_sim):
        p = Path(mod.__file__).resolve()
        print(f"[소스] {mod.__name__}: {p}")
        if _WT not in p.parents:
            raise RuntimeError(f"{mod.__name__} 이 worktree 밖에서 로드됨: {p}")


def _pose_of(asset, origins: torch.Tensor) -> np.ndarray:
    """(N,7) env-local pos(3)+quat wxyz(4)."""
    pos = (asset.data.root_pos_w - origins).cpu().numpy()
    quat = asset.data.root_quat_w.cpu().numpy()
    return np.concatenate([pos, quat], axis=-1).astype(np.float32)


@hydra_task_config(args.task, "rl_games_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    _assert_source_tree()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.warm.is_file():
        raise FileNotFoundError(f"warm 캐시 없음: {args.warm}")

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    agent_cfg["params"]["seed"] = args.seed
    env_cfg.warm_state_paths = (str(args.warm),)
    # 추출은 학습이 아니다 — 난이도 자동조절이 남아 있으면 시작 조건이 흔들린다.
    for attr in ("enable_adr", "enable_success_adr"):
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, False)
            print(f"[설정] {attr}=False")

    env = gym.make(args.task, cfg=env_cfg)
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_act = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_act)
    vecenv.register("IsaacRlgWrapper",
                    lambda cfg_name, n_actors, **kw: RlGamesGpuEnv(cfg_name, n_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: env})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(args.checkpoint)
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(str(args.checkpoint))
    agent.reset()

    base = env.unwrapped
    robot = base.robot
    origins = base.scene.env_origins
    jn = list(robot.joint_names)
    palm_i = robot.body_names.index("r_hl_palm")
    ltcp_i = robot.body_names.index("l_hl_gripper_base")
    fill_hi = float(base.cfg.success_target_fill_ratio)
    spill_hi = float(base.cfg.success_spill_max)
    step_dt = float(base.step_dt)
    print(f"[설정] 관절 {len(jn)} · step_dt {step_dt:.4f}s · 성공 = fill≥{fill_hi} ∧ spill≤{spill_hi}")

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    def _body_pose(idx: int) -> np.ndarray:
        pos = (robot.data.body_pos_w[:, idx] - origins).cpu().numpy()
        quat = robot.data.body_quat_w[:, idx].cpu().numpy()
        return np.concatenate([pos, quat], axis=-1).astype(np.float32)

    rec: dict[str, list] = {k: [] for k in (
        "q_meas", "qd_meas", "q_target", "palm_r", "tcp_l",
        "cup_src", "cup_recv", "action", "fill", "spill", "done")}

    n_done_success = 0
    step = 0
    with torch.inference_mode():
        while simulation_app.is_running() and step < args.max_steps:
            # ── env.step **직전** 스냅샷 = 시각 t 의 상태 ──────────────────────
            rec["q_meas"].append(robot.data.joint_pos.cpu().numpy().astype(np.float32))
            rec["qd_meas"].append(robot.data.joint_vel.cpu().numpy().astype(np.float32))
            rec["q_target"].append(robot.data.joint_pos_target.cpu().numpy().astype(np.float32))
            rec["palm_r"].append(_body_pose(palm_i))
            rec["tcp_l"].append(_body_pose(ltcp_i))
            rec["cup_src"].append(_pose_of(base.cup, origins))
            rec["cup_recv"].append(_pose_of(base.left_target_cup, origins))
            rec["fill"].append(base._bead_in_target_fraction.cpu().numpy().astype(np.float32))
            rec["spill"].append(base._spill_ratio.cpu().numpy().astype(np.float32))

            obs_t = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_t, is_deterministic=True)
            obs, _, dones, _ = env.step(actions)
            if isinstance(obs, dict):
                obs = obs["obs"]
            rec["action"].append(actions.cpu().numpy().astype(np.float32))
            d = dones.cpu().numpy().astype(bool)
            rec["done"].append(d)
            if agent.is_rnn and agent.states is not None:
                for s in agent.states:
                    s[:, dones, :] = 0.0
            step += 1

            if d.any():
                # 방금 끝난 에피소드들의 성공 여부를 기록에서 후처리로 센다(내부 버퍼는 이미 리셋).
                arr_fill = np.stack(rec["fill"])     # (T, N)
                arr_done = np.stack(rec["done"])
                for i in np.nonzero(d)[0]:
                    ends = np.nonzero(arr_done[:, i])[0]
                    s0 = 0 if len(ends) < 2 else ends[-2] + 1
                    seg_fill = arr_fill[s0:ends[-1] + 1, i]
                    seg_spill = np.stack(rec["spill"])[s0:ends[-1] + 1, i]
                    # ★env 와 같은 판정: **같은 스텝에서** fill≥임계 ∧ spill≤상한.
                    #   episode 전체 spill.max 에 상한을 걸면 warm 리셋 스파이크
                    #   spill=1.0 때문에 전부 실패로 오독한다(실측).
                    ok = bool(((seg_fill >= fill_hi) & (seg_spill <= spill_hi)).any())
                    n_done_success += int(ok)
                    print(f"[ep] env{i} steps {ends[-1]-s0+1} fill_max {seg_fill.max():.3f} "
                          f"spill_max {seg_spill.max():.3f} → {'✅성공' if ok else '실패'}"
                          f"  (누적 성공 {n_done_success}/{args.success_episodes})")
                if n_done_success >= args.success_episodes:
                    break

    # ── 후처리: 에피소드 분해 → 성공 중 peak 관절속도 최소 에피소드 선정 ──────────
    A = {k: np.stack(v) for k, v in rec.items()}     # 각 (T, N, ...)
    T, N = A["done"].shape
    cands = []
    for i in range(N):
        ends = np.nonzero(A["done"][:, i])[0]
        s0 = 0
        for e in ends:
            seg = slice(s0, e + 1)
            fill = A["fill"][seg, i]
            spill = A["spill"][seg, i]
            if ((fill >= fill_hi) & (spill <= spill_hi)).any():
                qd = np.abs(np.diff(A["q_target"][seg, i], axis=0)) / step_dt
                cands.append((float(qd.max()), i, s0, e))
            s0 = e + 1
    if not cands:
        print("[결론] ❌ 성공 에피소드 없음 — 추출 실패. 시드/에피소드 수를 늘려 재시도할 것.")
        return 1
    cands.sort()
    peak, i, s0, e = cands[0]
    seg = slice(s0, e + 1)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = dict(meta_joint_names=np.array(jn), meta_step_dt=np.float32(step_dt),
                meta_checkpoint=str(args.checkpoint), meta_commit="9b43f40",
                meta_success=np.array([fill_hi, spill_hi], np.float32))
    traj = args.out / f"pour_traj_env{i}_s{s0}_e{e}.npz"
    np.savez_compressed(traj, **{k: A[k][seg, i] for k in A}, **meta)
    init = args.out / "pour_init.npz"
    np.savez_compressed(
        init,
        q=A["q_meas"][s0, i], qd=A["qd_meas"][s0, i], q_target=A["q_target"][s0, i],
        palm_r=A["palm_r"][s0, i], tcp_l=A["tcp_l"][s0, i],
        cup_src=A["cup_src"][s0, i], cup_recv=A["cup_recv"][s0, i], **meta)
    n_steps = e + 1 - s0
    print(f"\n[선정] env{i} steps {n_steps} ({n_steps*step_dt:.1f}s) · peak |Δq_target|/dt "
          f"{peak:.3f} rad/s · 후보 {len(cands)}개 중 최소")
    print(f"[저장] {traj}\n[저장] {init}")
    print(f"[초기] palm_r {np.round(A['palm_r'][s0, i][:3], 4)} · tcp_l "
          f"{np.round(A['tcp_l'][s0, i][:3], 4)} · cup_src {np.round(A['cup_src'][s0, i][:3], 4)}")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code or 0)

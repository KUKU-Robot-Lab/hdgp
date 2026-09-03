"""v2 정책의 **성공 에피소드 하나**를 통째로 뽑는다 (S2R 이식용).

`probe_v2_env_outcomes.py` 와 **같은 셋업**(ADR 레벨 강제·결정론·같은 판정식)을 쓴다 —
결말 분류가 그쪽과 일치해야 뽑은 궤적이 대표성을 갖는다.

성공 정의는 학습과 동일하다: `success_ok`(도달+정지+직립+파지)가 `EPISODE_DWELL_STEPS`
연속. 성공한 env 중 **컵–목표 최저거리가 가장 작은 것** 하나를 고른다.

기록(스텝당):
  · 팔 관절 7 — 실측 `q`, 지령 `q_target`, 속도 `qd`
  · 그리퍼 관절 — 실측·지령, 게이트 상태
  · 정책 액션 원값 (palm 6 + gripper)
  · TCP 위치·자세 · 컵 위치·자세 · 목표 위치  (전부 **로봇 베이스 기준**)
  · 판정 플래그 — grasp_ok / 컵높이 / 컵–목표거리 / 성공

⚠ 좌표계: 위치는 전부 로봇 base link 원점 기준이다(ENV_POS = 원점과 일치).
⚠ dt = 0.02 s (decimation 2 · 50 Hz). `t` 열이 그 시간축이다.

실행:
  HDGP_V2_DWELL_END=1 ../IsaacLab/isaaclab.sh -p scripts/probes/probe_v2_extract_traj.py \
      --checkpoint log/checkpoints_keep/v2E29_band80_best.pth --adr_level 4 --out log/traj/e29
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default="open-grip_l_grasp_sensor_v2")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--adr_level", type=int, default=None)
parser.add_argument("--out", type=str, required=True, help="출력 접두어 (.csv/.json 이 붙는다)")
parser.add_argument("--stochastic", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import json  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from openarm.gripper.left.grasp_sensor_v2 import v2_preset as P  # noqa: E402
from openarm.gripper.left.grasp_sensor_v2 import v2_stages as S  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

ARM_JOINTS = [f"l_aj_{i}" for i in range(1, 8)]


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(args.task, "rl_games_cfg_entry_point")
    env = gym.make(args.task, cfg=env_cfg)
    raw = env.unwrapped

    if args.adr_level is not None:
        cm = raw.curriculum_manager
        names = list(cm.active_terms)
        if "adr" not in names:
            raise SystemExit("[traj] ADR 항이 없다 — DR 이 꺼져 있다")
        term = cm._term_cfgs[names.index("adr")].func
        term._level = int(args.adr_level)
        term._apply(raw)
        print(f"[traj] ADR 레벨 {args.adr_level} 강제", flush=True)

    inf = float("inf")
    wrapped = RlGamesVecEnvWrapper(
        env, args.device,
        agent_cfg["params"]["env"].get("clip_observations", inf),
        agent_cfg["params"]["env"].get("clip_actions", inf))
    vecenv.register("IsaacRlgWrapper", lambda n, k, **kw: RlGamesGpuEnv(n, k, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: wrapped})
    agent_cfg["params"]["config"]["env_info"] = {
        "observation_space": wrapped.observation_space,
        "action_space": wrapped.action_space, "agents": 1}
    hz = int(agent_cfg["params"]["config"].get("horizon_length", 24))
    agent_cfg["params"]["config"]["minibatch_size"] = args.num_envs * hz

    runner = Runner(); runner.load(agent_cfg)
    agent = runner.create_player(); agent.restore(args.checkpoint); agent.reset()

    N, dev = args.num_envs, args.device
    robot = raw.scene["robot"]; obj = raw.scene["object"]; eef = raw.scene["ee_frame"]
    robot_cfg = SceneEntityCfg("robot"); robot_cfg.resolve(raw.scene)
    jaw_cfg = SceneEntityCfg("robot", body_names=list(P.GRIPPER_FINGER_BODIES))
    jaw_cfg.resolve(raw.scene)
    obj_cfg = SceneEntityCfg("object"); obj_cfg.resolve(raw.scene)
    aidx = [robot.joint_names.index(n) for n in ARM_JOINTS]
    gidx = robot.joint_names.index(P.GRIPPER_DRIVE_JOINT)

    obs = wrapped.reset()
    obs = obs["obs"] if isinstance(obs, dict) else obs
    # ★rl_games 플레이어는 배치 크기를 obs 로부터 한 번 확정해야 한다. 안 부르면
    #   배치 1 로 굳어 (N·obs_dim) 을 한 샘플로 읽고 첫 층에서 shape 오류가 난다.
    agent.get_batch_size(obs, 1)
    if getattr(agent, "is_rnn", False):
        agent.init_rnn()

    rec = []                     # 스텝별 (N, D) 텐서
    done_lock = torch.zeros(N, dtype=torch.bool, device=dev)
    run_ok = torch.zeros(N, device=dev)
    best_run = torch.zeros(N, device=dev)
    min_dist = torch.full((N,), 9.9, device=dev)

    with torch.inference_mode():
        for t in range(args.steps):
            act = agent.get_action(obs, is_deterministic=not args.stochastic)
            obs, _, dones, _ = wrapped.step(act)
            obs = obs["obs"] if isinstance(obs, dict) else obs
            live = ~done_lock

            base_p, base_q = robot.data.root_pos_w, robot.data.root_quat_w
            cup_p, _ = subtract_frame_transforms(base_p, base_q, obj.data.root_pos_w)
            tcp_p, _ = subtract_frame_transforms(base_p, base_q,
                                                 eef.data.target_pos_w[:, 0, :])
            goal_p = S.goal_pos_w(raw, "object_pose", robot_cfg)
            goal_b, _ = subtract_frame_transforms(base_p, base_q, goal_p)
            dist = S.cup_goal_distance(raw, "object_pose", robot_cfg, obj_cfg)
            spd = torch.norm(obj.data.root_lin_vel_w, dim=1)
            cos = S._cup_upright_cos(raw, obj_cfg)
            rclose = S.stage_close(raw, jaw_cfg, obj_cfg)
            succ = (S.success_ok(dist, spd, cos, rclose) > 0.5).float()
            run_ok = torch.where(live & (succ > 0.5), run_ok + 1.0, torch.zeros_like(run_ok))
            best_run = torch.maximum(best_run, run_ok)
            min_dist = torch.where(live, torch.minimum(min_dist, dist), min_dist)

            rec.append(torch.cat([
                robot.data.joint_pos[:, aidx],                     # 0:7   q
                robot.data.joint_pos_target[:, aidx],              # 7:14  q_target
                robot.data.joint_vel[:, aidx],                     # 14:21 qd
                robot.data.joint_pos[:, gidx:gidx + 1],            # 21    grip q
                robot.data.joint_pos_target[:, gidx:gidx + 1],     # 22    grip target
                act.float(),                                       # 23:23+A
                tcp_p, eef.data.target_quat_w[:, 0, :],            # tcp pos/quat
                cup_p, obj.data.root_quat_w,                       # cup pos/quat
                goal_b,                                            # goal
                dist.unsqueeze(-1), spd.unsqueeze(-1),
                cos.unsqueeze(-1), rclose.unsqueeze(-1), succ.unsqueeze(-1),
                live.float().unsqueeze(-1),
            ], dim=-1))
            d = dones.bool() if torch.is_tensor(dones) else torch.as_tensor(dones, device=dev).bool()
            done_lock |= d.reshape(-1)

    traj = torch.stack(rec, dim=0)            # (T, N, D)
    ever = best_run >= 1.0
    ok = best_run >= float(P.EPISODE_DWELL_STEPS)
    n_ok, n_ever = int(ok.sum()), int(ever.sum())
    br = best_run.cpu()
    print(f"[traj] 한 번이라도 성공 {n_ever}/{N} · 연속 {P.EPISODE_DWELL_STEPS} 스텝 "
          f"{n_ok}/{N} · 최장 연속 스텝 max {int(br.max())} 중앙 {int(br.median())}",
          flush=True)
    print(f"[traj] 컵–목표 최저거리 (mm) min {float(min_dist.min())*1000:.1f} · "
          f"중앙 {float(min_dist.median())*1000:.1f}", flush=True)
    if n_ok == 0:
        # ★학습은 '연속 10 스텝'에서 에피소드를 끊지만, 결정론 롤아웃에서는 그만큼
        #   못 버티는 판이 많다. 그때는 **최장 연속 성공 구간**이 가장 긴 env 로 대신한다
        #   — "성공에 가장 가까운 궤적"이 S2R 이식에는 여전히 쓸모가 있다.
        if n_ever == 0:
            raise SystemExit("[traj] 성공 스텝이 하나도 없다 — 레벨을 낮추거나 체크포인트 확인")
        ok = ever
        n_ok = n_ever
        print("[traj] ⚠ 연속 기준 미달 — **최장 연속 성공** env 로 대체 선택", flush=True)
    # 우선순위: 연속 성공이 긴 것 → 그중 컵–목표가 가까운 것
    score = torch.where(ok, best_run * 1000.0 - min_dist * 1000.0,
                        torch.full_like(min_dist, -1e9))
    e = int(torch.argmax(score))
    live_col = traj.shape[-1] - 1
    T_live = int(traj[:, e, live_col].sum())
    print(f"[traj] 선택 env {e} · 에피소드 {T_live} 스텝 · 컵–목표 최저 "
          f"{float(min_dist[e]) * 1000:.1f} mm", flush=True)

    A = int(act.shape[-1])
    cols = ([f"q{i}" for i in range(1, 8)] + [f"qt{i}" for i in range(1, 8)]
            + [f"qd{i}" for i in range(1, 8)] + ["grip_q", "grip_target"]
            + [f"act{i}" for i in range(A)]
            + ["tcp_x", "tcp_y", "tcp_z", "tcp_qw", "tcp_qx", "tcp_qy", "tcp_qz"]
            + ["cup_x", "cup_y", "cup_z", "cup_qw", "cup_qx", "cup_qy", "cup_qz"]
            + ["goal_x", "goal_y", "goal_z"]
            + ["cup_goal_dist", "cup_speed", "cup_upright_cos", "r_close", "success", "live"])
    assert len(cols) == traj.shape[-1], f"열 이름 {len(cols)} ≠ 데이터 {traj.shape[-1]}"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    arr = traj[:T_live, e, :].cpu().numpy()
    with open(args.out + ".csv", "w", encoding="utf-8") as f:
        f.write("t," + ",".join(cols) + "\n")
        for i, row in enumerate(arr):
            f.write(f"{i * 0.02:.3f}," + ",".join(f"{v:.6f}" for v in row) + "\n")
    meta = {
        "checkpoint": args.checkpoint, "task": args.task,
        "adr_level": args.adr_level, "deterministic": not args.stochastic,
        "env_index": e, "steps": T_live, "dt": 0.02,
        "min_cup_goal_dist_mm": round(float(min_dist[e]) * 1000, 2),
        "success_envs": f"{n_ok}/{N}",
        "arm_joints": ARM_JOINTS, "gripper_joint": P.GRIPPER_DRIVE_JOINT,
        "frame": "로봇 base link 원점 (ENV_POS 와 일치)",
        "grasp_height_band_m": list(P.GRASP_HEIGHT_BAND),
        "table_surface_z": P.TABLE_SURFACE_Z,
        "columns": ["t"] + cols,
    }
    with open(args.out + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[traj] 저장 {args.out}.csv ({T_live} 행 × {len(cols)} 열) · {args.out}.json",
          flush=True)
    env.close()


main()
app.close()

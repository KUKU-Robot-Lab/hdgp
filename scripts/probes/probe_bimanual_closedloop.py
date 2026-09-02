#!/usr/bin/env python3
"""양팔 파지를 **실기 배포와 동형의 폐루프**로 한 씬에서 보인다.

실기 구조(사용자 확정 09.02): 정책이 (로봇 상태 + FD++ 컵 pose)를 관측하고,
액션이 로봇을 직접 제어한다. 여기서 '현실' = 통합 pour 씬. 정책 사슬은
`bimanual_chain.py` 가 각 학습 env 의 원본 코드로 세운다.

순서:  정착(preset=정책 홈) → [Enter] ① 좌팔 v2B25 폐루프 — shaker 를 접촉으로
       파지·리프트 → [Enter] ② 유지 → (물리 dt 100→120Hz 전환) → [Enter]
       ③ 우팔 E1 폐루프 — cup_big_s100 → [Enter] ④ 양팔 유지

미러/재생과 다른 점: 텔레포트·컵 root 고정·순간 부착이 **없다**. 초기 배치
1회 뒤에는 PD 와 접촉만 있다. 여기서 파지가 실패하면 실기도 실패한다는 뜻이다.

    ../IsaacLab/isaaclab.sh -p scripts/probes/probe_bimanual_closedloop.py \\
        --auto --render /tmp/bi_cl --verify           # 영상 + 재현 대조
    ../IsaacLab/isaaclab.sh -p scripts/probes/probe_bimanual_closedloop.py --gui
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_SR = Path("/home/user/rl_ws/sim2real")
parser.add_argument("--left-checkpoint", type=Path,
                    default=_SR / "logs/policy/left_v2B25/nn/v2B25_tip30_ep2150.pth")
parser.add_argument("--right-checkpoint", type=Path,
                    default=_SR / "logs/policy/right_g1/nn/g1_ep17000.pth",
                    help="우팔 grasp_s2r (09.02 사용자 지정 g1 — use_real_gains 런은 "
                         "HDGP_S2R_REAL_GAINS=1 로 실행)")
parser.add_argument("--left-stream", type=Path,
                    default=_SR / "logs/shadow/pour_entry/stream_left_v2b25.npz",
                    help="goal·컵 스폰·(--verify 시) 대조 기준")
parser.add_argument("--right-stream", type=Path,
                    default=_SR / "logs/shadow/pour_entry/stream_right_g1.npz")
parser.add_argument("--left-cup-json", type=Path, default=None,
                    help="FD++ capture JSON (base_link) — shaker 스폰을 실측으로 대체")
parser.add_argument("--right-cup-json", type=Path, default=None,
                    help="FD++ capture JSON — cup_big 스폰을 실측으로 대체")
parser.add_argument("--force-spawn", action="store_true",
                    help="실측 스폰이 학습 분포 상자 밖이어도 진행 (기본은 거부+재배치 안내)")
parser.add_argument("--left-steps", type=int, default=300)
parser.add_argument("--right-steps", type=int, default=420)
parser.add_argument("--stop-lift", type=float, default=0.08)
parser.add_argument("--lift-hold", type=int, default=40)
parser.add_argument("--settle", type=int, default=120)
parser.add_argument("--hold", type=int, default=120)
parser.add_argument("--final-hold", type=int, default=300)
parser.add_argument("--pour-checkpoint", type=Path,
                    default=_SR / "logs/policy/pour_e1/nn/e1_pour1_ep6500.pth",
                    help="e1_pour1 최종본 (ep6500 — 학습이 그 시점에 종료됨, md5 6e3366d5)")
parser.add_argument("--pour-steps", type=int, default=900)
parser.add_argument("--pour-mode", choices=("follow", "policy"), default="follow",
                    help="follow=네이티브 성공 에피소드의 관절 궤적 추종(사용자 지시 09.02) · "
                         "policy=폐루프 (현재 β=0 미해결)")
parser.add_argument("--pour-traj", type=Path,
                    default=_SR / "logs/shadow/pour_entry/pour_traj_receiver_live_ep6500.npz",
                    help="기본 = ep6500 · 실측 받는점(0.265,0.045,0.296) 기준 성공 궤적 20/20")
parser.add_argument("--skip-pour", action="store_true", help="파지 4국면까지만")
parser.add_argument("--receiver-up-contract", action="store_true",
                    help="pour obs 의 tgt_up 을 훈련 상수로 고정 — pour 훈련에서 받는컵"
                         " 자세는 kinematic 상수 계약이었다 (실컵 기울기 편차는 계약 밖)")
parser.add_argument("--diag-spoof-left-obs", action="store_true",
                    help="진단: pour obs 의 좌팔 18D 를 훈련 상수(REST·qd0)로 교체 — 좌팔 원인 분리")
parser.add_argument("--carry-vel", type=float, default=0.25,
                    help="전환 램프 관절속도 상한 [rad/s]")
parser.add_argument("--pour-entry-joints", default="0.512,0.414,-0.487,0.243,0.084,0.546,1.168",
                    help="⑤′ 우팔 pour 세팅 관절 7 — 기본 = E1 뱅크 mean (n=2107)")
parser.add_argument("--auto", action="store_true")
parser.add_argument("--verify", action="store_true",
                    help="기록 에피소드와 같은 스폰·goal 로 돌고 궤적 편차를 보고")
parser.add_argument("--render", type=Path, default=None)
parser.add_argument("--render-every", type=int, default=4)
parser.add_argument("--gui", action="store_true")

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "source" / "openarm"), str(_REPO / "scripts" / "tools"),
           str(_REPO / "scripts" / "probes"), str(_SR / "scripts")):
    sys.path.insert(0, _p)

from isaaclab.app import AppLauncher                              # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
args.headless = not args.gui
args.enable_cameras = args.render is not None
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym                                           # noqa: E402
import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402
import yaml                                                       # noqa: E402

import openarm  # noqa: E402,F401
import openarm.tasks  # noqa: E402,F401
from isaaclab.sim.utils import find_matching_prim_paths           # noqa: E402
from isaaclab_tasks.utils.parse_cfg import (                      # noqa: E402
    load_cfg_from_registry, parse_env_cfg)
from rl_games.torch_runner import Runner                          # noqa: E402
from run_cfg_restore import restore_run_cfg_if_available          # noqa: E402

from bimanual_chain import (                                      # noqa: E402
    LEFT9, LeftChain, RightChainShim, align_pour_cfg, make_show_env)
from openarm.tesollo.right.pour_sensor.pour_right_env import PourRightEnv  # noqa: E402

POUR_TASK = "open-tesol_r_pour_sensor-play-lstm"
LEFT_TASK = "open-grip_l_grasp_sensor_v2-play"
RIGHT_TASK = "open-sens_r_grasp_s2r-play-lstm"
DEVICE = "cuda:0"


def make_player(agent_cfg: dict, ckpt: Path, obs_dim: int, act_dim: int):
    """rl_games 플레이어를 env 없이 만든다 — 배포 노드가 하는 그대로."""
    cfg = agent_cfg["params"]
    cfg["config"]["env_info"] = {
        "observation_space": gym.spaces.Box(-np.inf, np.inf, (obs_dim,)),
        "action_space": gym.spaces.Box(-1.0, 1.0, (act_dim,)),
        "agents": 1,
    }
    cfg["config"]["num_actors"] = 1
    cfg["load_checkpoint"] = True
    cfg["load_path"] = str(ckpt)
    runner = Runner()
    runner.load(agent_cfg)
    player = runner.create_player()
    player.restore(str(ckpt))
    player.has_batch_dimension = True
    player.batch_size = 1
    return player


def main() -> int:
    # ── cfg 3벌: pour(씬) · 좌(사슬) · 우(사슬) — 전부 런 dump 로 복원 ────────
    left_cfg = parse_env_cfg(LEFT_TASK, device=DEVICE, num_envs=1)
    left_agent = load_cfg_from_registry(LEFT_TASK, "rl_games_cfg_entry_point")
    left_agent = restore_run_cfg_if_available(
        left_cfg, left_agent, resume_path=str(args.left_checkpoint),
        workspace_root=str(_REPO.parent))
    right_cfg = parse_env_cfg(RIGHT_TASK, device=DEVICE, num_envs=1)
    right_agent = load_cfg_from_registry(RIGHT_TASK, "rl_games_cfg_entry_point")
    right_agent = restore_run_cfg_if_available(
        right_cfg, right_agent, resume_path=str(args.right_checkpoint),
        workspace_root=str(_REPO.parent))
    right_cfg.finalize_after_overrides()

    zl = np.load(args.left_stream, allow_pickle=True)
    zr = np.load(args.right_stream, allow_pickle=True)
    left_goal = zl["goal"][0].astype(np.float32)              # 7D (root 프레임)
    right_goal = zr["goal"][0].astype(np.float32)             # 3D (env-local)
    left_spawn = zl["meta_cup_spawn"].astype(np.float32)
    right_spawn = zr["meta_cup_spawn"].astype(np.float32)

    # FD++ capture JSON 스폰 (Step 1) — base_link ≡ env-local (로봇 베이스가 원점).
    # 분포 상자 가드: 실물 배치가 학습 분포 밖이면 정책이 조용히 이상해진다 — 거부한다.
    from cup_pose_capture import load_capture, spawn_box_for_side, verdict  # noqa: PLC0415
    for _side, _jpath, _tag in (("left", args.left_cup_json, "shaker"),
                                ("right", args.right_cup_json, "cup_big")):
        if _jpath is None:
            continue
        _cp = load_capture(_jpath, expect_frame="base_link")
        _vd = verdict(_cp, spawn_box_for_side(_side))
        print(f"[FD++ 스폰] {_tag} ← {_jpath.name}: "
              f"{[round(v, 4) for v in _cp.position]} · "
              f"{'분포 안 ✅' if _vd.inside else '분포 밖 ❌'}", flush=True)
        if not _vd.inside and not args.force_spawn:
            raise SystemExit(
                f"[FD++ 스폰] {_tag} 이 학습 분포 밖이다:\n{_vd.describe()}\n"
                "실물 컵을 상자 안으로 재배치하거나 --force-spawn 으로 강행하라.")
        if _side == "left":
            left_spawn = np.array(_cp.position, dtype=np.float32)
        else:
            right_spawn = np.array(_cp.position, dtype=np.float32)
            right_goal = right_spawn + np.array([0.0, 0.0, 0.12], dtype=np.float32)
            print("[FD++ 스폰] 우 goal = 스폰 + (0,0,0.12) (E1/g1 liftonly 규약)", flush=True)

    left_yaml = yaml.unsafe_load(
        (args.left_checkpoint.parent.parent / "params" / "env.yaml").read_text())
    right_yaml = yaml.unsafe_load(
        (args.right_checkpoint.parent.parent / "params" / "env.yaml").read_text())
    pour_yaml = yaml.unsafe_load(
        (args.pour_checkpoint.parent.parent / "params" / "env.yaml").read_text()) \
        if not args.skip_pour else None

    pour_cfg = parse_env_cfg(POUR_TASK, device=DEVICE, num_envs=1)
    pour_agent = load_cfg_from_registry(POUR_TASK, "rl_games_cfg_entry_point")
    if not args.skip_pour:
        # ★e1_pour1 런 cfg 복원 — 씬 필드는 뒤의 align 이 다시 정리하고, 사슬이 읽는
        #   런타임 파라미터(리미터·게이트·보상 상수·freeze·구슬 수)가 학습본으로 잠긴다.
        pour_agent = restore_run_cfg_if_available(
            pour_cfg, pour_agent, resume_path=str(args.pour_checkpoint),
            workspace_root=str(_REPO.parent))
    lgrip = str(_REPO / "assets/robot/openarm_tesollo_sensor_rl_lgrip"
                / "openarm_tesollo_sensor_rl.usd")
    for line in align_pour_cfg(pour_cfg, left_scene=left_yaml["scene"],
                               right_actuators=right_yaml["robot_cfg"]["actuators"],
                               lgrip_usd=lgrip, left_spawn=left_spawn,
                               physics_dt=float(left_cfg.sim.dt)):
        print(f"[정합] {line}")

    # ── 씬 ──────────────────────────────────────────────────────────────────
    env = make_show_env(PourRightEnv)(cfg=pour_cfg, render_mode=None)
    env.reset()
    robot, scene = env.robot, env.scene
    if not find_matching_prim_paths("/World/envs/env_0/Cup/baseLink"):
        raise RuntimeError("컵 필터 프림이 없다 — 센서 force_matrix 가 무증상 0 이 된다")

    dt_box = [float(left_cfg.sim.dt)]                    # 국면별 물리 dt (좌 먼저)

    def T(a, shape=None):
        t = torch.tensor(np.asarray(a, dtype=np.float32), device=env.device)
        return t if shape is None else t.reshape(shape)

    def place(obj, pos_env_local):
        pose = torch.cat([scene.env_origins[0] + T(pos_env_local),
                          T([1.0, 0, 0, 0])]).unsqueeze(0)
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))

    place(env.cup, right_spawn)
    place(env.left_target_cup, left_spawn)
    if hasattr(env, "_hide_beads"):
        env._hide_beads(torch.arange(1, device=env.device))

    # 정책 홈 주입 — default_joint_pos 를 (좌 v2B25 · 우 E1) 로. 이것이 preset 이다.
    # ★★각 dump 는 **반대팔의 주차 자세도** 담고 있다 (우 dump 의 l_aj_2=-0.671 등).
    #   필터 없이 합치면 나중 소스가 앞 팔의 홈을 덮어써 TCP 가 11cm 어긋난다 —
    #   그런데 obs joint 항은 자기 default 상대라 0 으로 보여 **조용히** 틀린다
    #   (09.02 diagL2 실측, obs0 대조가 잡음). 각 소스는 자기 팔 접두사만 낸다.
    dq = robot.data.default_joint_pos
    injected = 0
    for prefix, src in (("l_", left_yaml["scene"]["robot"]["init_state"]["joint_pos"]),
                        ("r_", _right_home(right_cfg))):
        for pat, val in src.items():
            if not pat.startswith(prefix):
                continue
            ids, _ = robot.find_joints(pat)
            for i in ids:
                dq[:, i] = float(val)
                injected += 1
    print(f"[홈] 관절 {injected}개 주입 (좌 v2B25 · 우 E1)")
    robot.write_joint_state_to_sim(dq.clone(), torch.zeros_like(dq))
    robot.set_joint_position_target(dq.clone())

    # ── 우 사슬 (부팅 게이트 3종이 여기서 돈다) + 좌 사슬 ────────────────────
    right = RightChainShim(env, right_cfg, env.bi_finger_sensors, env.bi_palm_sensor)
    right.zero_obs_noise()
    right.goal_pos[:] = T(right_goal).unsqueeze(0)
    left = LeftChain(env, left_cfg,
                     step_dt=float(left_cfg.sim.dt) * int(left_cfg.decimation))
    left.reset()

    # ── 렌더 ────────────────────────────────────────────────────────────────
    shots = [0]

    def shot(tag: str) -> None:
        return

    if args.render is not None:
        import omni.replicator.core as rep  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        args.render.mkdir(parents=True, exist_ok=True)
        c = scene.env_origins[0].cpu().numpy() + np.array([0.34, 0.0, 0.32])
        cam = rep.create.camera(position=tuple(c + np.array([1.20, -0.70, 0.55])),
                                look_at=tuple(float(v) for v in c))
        rp = rep.create.render_product(cam, (1280, 800))
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach([rp])

        def shot(tag: str) -> None:  # noqa: F811
            env.sim.render()
            arr = np.asarray(annot.get_data())
            if arr.size:
                Image.fromarray(arr[:, :, :3]).save(
                    args.render / f"{shots[0]:04d}_{tag}.png")
                shots[0] += 1

    # ── 감시 (전 스텝) — 폭발·NaN 을 수치로 잡는다. 미러 사고의 재발 방지 ────
    worst = {"qd": 0.0, "cupv": 0.0}

    def watch() -> None:
        qd = float(robot.data.joint_vel.abs().max())
        cv = max(float(env.cup.data.root_lin_vel_w.norm(dim=-1).max()),
                 float(env.left_target_cup.data.root_lin_vel_w.norm(dim=-1).max()))
        worst["qd"] = max(worst["qd"], qd)
        worst["cupv"] = max(worst["cupv"], cv)
        if not torch.isfinite(robot.data.joint_pos).all():
            raise RuntimeError("관절 NaN — 물리 폭발")
        if qd > 12.0 or cv > 3.0:
            raise RuntimeError(f"폭발 감지: |q̇|max {qd:.1f} rad/s · 컵 {cv:.2f} m/s")

    def cups_z() -> str:
        s = float(env.cup.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
        r = float(env.left_target_cup.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
        return f"cup_big z {s:.3f} · shaker z {r:.3f}"

    def passive(tag: str, n: int) -> None:
        for f in range(n):
            scene.write_data_to_sim()
            env.sim.step(render=args.gui)
            scene.update(dt_box[0])
            watch()
            if f % args.render_every == 0:
                shot(tag)
        print(f"[{tag}] {n}스텝 · {cups_z()}")

    def gate(msg: str) -> None:
        print(f"\n▶ {msg}")
        if not args.auto:
            input("  [Enter] …")

    passive("0settle", args.settle)
    for _nm, _obj, _rec in (("cup_big", env.cup, right_spawn),
                            ("shaker", env.left_target_cup, left_spawn)):
        _z = float(_obj.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
        print(f"[정착] {_nm} z {_z:.4f} (기록 {float(_rec[2]):.4f} · "
              f"Δ {_z - float(_rec[2]):+.4f})", flush=True)
    # 앵커 스냅샷 — env 리셋 정착 스냅샷의 대응물 (anchor_mode=spawn 이 이걸 쓴다)
    right.object_spawn_pos[:] = (env.cup.data.root_pos_w - scene.env_origins)[0:1]

    # ── 플레이어 2벌 ────────────────────────────────────────────────────────
    from left_obs_builder import ACTOR_OBS_DIM as LEFT_OBS_DIM  # noqa: PLC0415
    from left_obs_builder import NUM_ACTIONS as LEFT_ACTS  # noqa: PLC0415
    player_l = make_player(left_agent, args.left_checkpoint, LEFT_OBS_DIM, LEFT_ACTS)
    player_r = make_player(right_agent, args.right_checkpoint,
                           int(right_cfg.observation_space),
                           int(right_cfg.action_space))

    trace: dict[str, list] = {k: [] for k in
                              ("l_palm", "l_armt", "r_palm", "r_armt", "r_latch")}

    # ── ① 좌팔 폐루프 ───────────────────────────────────────────────────────
    from bimanual_obs import left_actor_obs  # noqa: PLC0415
    q0_left = np.array([float(dq[0, robot.joint_names.index(n)]) for n in LEFT9])
    gate("① 좌팔(v2B25) 폐루프 — shaker 파지")
    la = np.zeros(LEFT_ACTS, dtype=np.float32)
    z0 = float(env.left_target_cup.data.root_pos_w[0, 2])
    streak, l_steps = 0, 0
    with torch.inference_mode():
        for step in range(args.left_steps):
            obs = left_actor_obs(env, left, goal7=left_goal, last_action=la,
                                 q_default=q0_left)
            if step < 8:
                # obs 결백 검사 — 초기 스텝의 세그먼트별 편차. 접촉 전 발산의 근원을
                # 짚는다(어긋난 세그먼트 = 조립 버그 또는 물리 부정합).
                from left_obs_builder import SEGMENTS as _LSEG  # noqa: PLC0415
                _rec0, _i, _tops = zl["obs"][step], 0, []
                for _nm, _dd in _LSEG:
                    _dm = float(np.abs(obs[_i:_i + _dd] - _rec0[_i:_i + _dd]).max())
                    if _dm > (0.02 if step == 0 else 0.05):
                        _tops.append(f"{_nm} {_dm:.3f}")
                    _i += _dd
                if _tops:
                    print(f"  [obs{step} Δ] " + " · ".join(_tops), flush=True)
            act = player_l.get_action(T(obs, (1, -1)), is_deterministic=True)
            act = act.reshape(1, LEFT_ACTS)   # raw — 텀이 내부 클램프(학습 동형)
            left.step_policy(act, render=args.gui)
            watch()
            la = act[0].detach().cpu().numpy()
            trace["l_palm"].append(
                left.arm.processed_actions[0].detach().cpu().numpy().copy())
            trace["l_armt"].append(robot.data.joint_pos_target[
                0, left.arm._arm_joint_ids].detach().cpu().numpy().copy())
            l_steps = step + 1
            if step % 20 == 0 or step < 24 and step % 2 == 0:
                sp = (env.left_target_cup.data.root_pos_w[0]
                      - scene.env_origins[0]).cpu().numpy()
                print(f"  [좌{step:3d}] gate {int(left.gate_open[0])} · shaker "
                      f"({sp[0]:.3f},{sp[1]:.3f},{sp[2]:.3f}) · palm_cmd "
                      f"{np.round(trace['l_palm'][-1][:3], 3).tolist()}", flush=True)
            if step % args.render_every == 0:
                shot("1left")
            dz = float(env.left_target_cup.data.root_pos_w[0, 2]) - z0
            streak = streak + 1 if dz > args.stop_lift else 0
            if streak >= args.lift_hold:
                print(f"[좌] step {step}: Δz {dz:+.3f} 를 {args.lift_hold}스텝 유지 — 파지 성립")
                break
        left.freeze_targets()

        gate("② 좌팔 유지")
        passive("2holdL", args.hold)

        # ── 물리 dt 전환 100 → 120 Hz (우 국면은 E1 학습 주기) ───────────────
        rdt = float(right_cfg.sim.dt)
        env.sim.set_simulation_dt(physics_dt=rdt)
        dt_box[0] = rdt
        right.physics_dt = rdt
        print(f"[dt] 물리 {1/rdt:.0f} Hz 로 전환 (우 국면)")

        # ── ③ 우팔 폐루프 ───────────────────────────────────────────────────
        gate("③ 우팔(E1) 폐루프 — cup_big_s100 파지")
        if player_r.is_rnn:
            player_r.init_rnn()
        obs_r = right.observe()
        z0r = float(env.cup.data.root_pos_w[0, 2])
        streak, r_steps = 0, 0
        for step in range(args.right_steps):
            act = player_r.get_action(obs_r.reshape(1, -1), is_deterministic=True)
            act = act.reshape(1, -1)          # raw — _pre_physics_step 이 클램프
            obs_r = right.step_policy(act, render=args.gui)
            watch()
            trace["r_palm"].append(right.palm_targets[0].detach().cpu().numpy().copy())
            trace["r_armt"].append(robot.data.joint_pos_target[
                0, right.arm_ids].detach().cpu().numpy().copy())
            trace["r_latch"].append(bool(right._latched[0]))
            r_steps = step + 1
            if step % args.render_every == 0:
                shot("3right")
            dz = float(env.cup.data.root_pos_w[0, 2]) - z0r
            streak = streak + 1 if dz > args.stop_lift else 0
            if streak >= args.lift_hold:
                print(f"[우] step {step}: Δz {dz:+.3f} 를 {args.lift_hold}스텝 유지 — 파지 성립")
                break
        right.freeze_targets()

        gate("④ 양팔 유지 — 두 컵")
        passive("4final", args.final_hold)
        # 파지 판정은 여기서 캐시 — pour 가 컵을 내리면 최종값으로는 못 잰다
        grasp_lz = float(env.left_target_cup.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
        grasp_rz = float(env.cup.data.root_pos_w[0, 2] - scene.env_origins[0, 2])

        pour_report = None
        if not args.skip_pour:
            # ── ⑤ 좌팔 전환 — pour 받는 자세(학습 REST)로, 쥔 채 관절 램프 ──────
            gate("⑤ 좌팔 전환 — 받는 자세로 (쥔 채)")
            rest7 = env.left_arm_zero_pos[0, :7].clone()      # 학습 상수 (덮어쓰기 전)
            lids = list(env.left_arm_dof_indices[:7])
            cur = robot.data.joint_pos_target[0, lids].clone()
            # ★2단 램프 — 한 번에 lerp 하면 경로가 낮게 스쳐 컵 바닥이 테이블면 아래로
            #   ~16mm 파고든다(09.02 사용자 관찰 + z 추적 실측). 1단: j1(베이스 요)만
            #   돌려 수평 이동 → 2단: 나머지 6관절로 자세 조정(목표 지점 위에서 수직).
            via = cur.clone()
            via[0] = rest7[0]
            steps = 0
            for seg_from, seg_to in ((cur, via), (via, rest7)):
                seg_n = max(int(float((seg_to - seg_from).abs().max())
                               / (args.carry_vel * dt_box[0])), 1)
                for f in range(seg_n):
                    a = (f + 1) / seg_n
                    robot.set_joint_position_target(
                        (seg_from * (1 - a) + seg_to * a).unsqueeze(0), joint_ids=lids)
                    scene.write_data_to_sim()
                    env.sim.step(render=args.gui)
                    scene.update(dt_box[0])
                    watch()
                    if steps % 40 == 0:
                        _sz = (env.left_target_cup.data.root_pos_w[0]
                               - scene.env_origins[0]).cpu().numpy()
                        print(f"  [전환{steps:3d}] shaker ({_sz[0]:.3f},{_sz[1]:.3f},"
                              f"{_sz[2]:.3f})", flush=True)
                    if steps % (args.render_every * 3) == 0:
                        shot("5carry")
                    steps += 1
            for f in range(90):                                # 정착
                scene.write_data_to_sim()
                env.sim.step(render=args.gui)
                scene.update(dt_box[0])
                watch()
                if f % args.render_every == 0:
                    shot("5carry")
            # 받는점 잔차 — **서보는 3방식 모두 발산해 폐기**(09.02: fabric v1 euler
            #   클램프 / v2 palm 프레임 불일치 / v3 자코비안 노이즈 지배). 대신 pour
            #   궤적을 실측 받는점 기준으로 재추출한다(probe_pour_native_check
            #   --receiver-pos) — 궤적이 실컵을 노리므로 좌팔은 REST 도착이면 충분하다.
            _ref_src = pour_cfg.left_target_cup_pos_env_local
            if args.pour_mode == "follow" and args.pour_traj.exists():
                _zt0 = np.load(args.pour_traj)
                if "meta_receiver" in _zt0:
                    _ref_src = tuple(float(v) for v in _zt0["meta_receiver"])
            ref = T(_ref_src)
            lz2 = float(env.left_target_cup.data.root_pos_w[0, 2]
                        - scene.env_origins[0, 2])
            res = float((ref - (env.left_target_cup.data.root_pos_w[0]
                                - scene.env_origins[0])).norm())
            print(f"[전환] {steps}+90스텝 · shaker z {lz2:.3f} · 받는점 잔차 "
                  f"{res * 1000:.0f}mm {'✅' if res < 0.05 and lz2 > 0.28 else '⚠'}",
                  flush=True)

            # ── ⑤′ 우팔 FK 세팅 램프 — pour-sensor 세팅 자세로 (사용자 지시) ─────
            # ★교차 0 의 판명 원인: E1 이 transfer 목표까지 이송한 진입 palm 이 뱅크
            #   분포 밖(y −2.6cm·z −3.4cm). 뱅크 mean 관절로 쥔 채 램프하면 컵도 뱅크
            #   mean 위치(0.363,−0.159,0.400) 부근으로 따라와 분포 정중앙에서 시작한다.
            rest9_orig = env.left_arm_zero_pos[0].clone()      # 스푸핑 진단용 (덮어쓰기 전)
            gate("⑤′ 우팔 세팅 — pour 진입 자세로 (쥔 채)")
            entry7 = T([float(v) for v in args.pour_entry_joints.split(",")])
            rids = list(right.arm_ids)
            cur_r = robot.data.joint_pos_target[0, rids].clone()
            right.freeze_targets()                     # 속도 FF 잔재 제거
            steps_r = max(int(float((entry7 - cur_r).abs().max())
                              / (args.carry_vel * dt_box[0])), 1)
            for f in range(steps_r):
                a = (f + 1) / steps_r
                robot.set_joint_position_target(
                    (cur_r * (1 - a) + entry7 * a).unsqueeze(0), joint_ids=rids)
                scene.write_data_to_sim()
                env.sim.step(render=args.gui)
                scene.update(dt_box[0])
                watch()
                if f % (args.render_every * 3) == 0:
                    shot("5entry")
            for f in range(90):
                scene.write_data_to_sim()
                env.sim.step(render=args.gui)
                scene.update(dt_box[0])
                watch()
                if f % args.render_every == 0:
                    shot("5entry")
            cpos = (env.cup.data.root_pos_w[0] - scene.env_origins[0]).cpu().numpy()
            print(f"[우세팅] {steps_r}+90스텝 · 컵 ({cpos[0]:.3f},{cpos[1]:.3f},{cpos[2]:.3f})"
                  f" (뱅크 mean 0.363,-0.159,0.400) · "
                  f"{'✅쥔 채' if cpos[2] > 0.33 else '❌이탈'}", flush=True)

            # ── ⑥ pour 초기화 — warm 텔레포트의 라이브 대응 ────────────────────
            gate("⑥ pour 초기화 (구슬 자동 소환 예약)")
            from bimanual_chain import (  # noqa: PLC0415
                disarm_receiver_pin, init_pour_from_live, refresh_receiver_buffer)
            disarm_receiver_pin(env)
            init_pour_from_live(env, right, pour_yaml["robot_cfg"]["actuators"])
            player_p = make_player(pour_agent, args.pour_checkpoint,
                                   int(pour_cfg.observation_space),
                                   int(pour_cfg.action_space))

            if args.pour_mode == "follow":
                # ── ⑦ pour 궤적 추종 — 네이티브 성공 에피소드의 실측 관절을 따라간다
                #   (사용자 지시 09.02). 실측 궤적은 동역학적으로 실현 가능하고,
                #   속도 FF 를 함께 주면 PD 지연이 사라진다(vel_ff 교훈 그대로).
                gate("⑦ pour — 성공 궤적 추종 (우팔)")
                ztr = np.load(args.pour_traj)
                tq = torch.tensor(ztr["arm_q"], dtype=torch.float32, device=env.device)
                tqd = torch.tensor(ztr["arm_qd"], dtype=torch.float32, device=env.device)
                print(f"[추종] 궤적 {tq.shape[0]}스텝 · 원본 교차 "
                      f"{int(ztr['meta_final_cross'])}/20", flush=True)
                rids2 = [robot.joint_names.index(f"r_aj_{i}") for i in range(1, 8)]
                cur2 = robot.data.joint_pos_target[0, rids2].clone()
                st = max(int(float((tq[0] - cur2).abs().max())
                             / (args.carry_vel * dt_box[0])), 1)
                for f in range(st + 60):                      # 궤적 시작점으로 램프+정착
                    a = min((f + 1) / st, 1.0)
                    robot.set_joint_position_target(
                        (cur2 * (1 - a) + tq[0] * a).unsqueeze(0), joint_ids=rids2)
                    scene.write_data_to_sim()
                    env.sim.step(render=args.gui)
                    scene.update(dt_box[0])
                    watch()
                    if f % (args.render_every * 3) == 0:
                        shot("6pour")
                # 구슬 주입 — env 자체 샘플러로 쥔 컵 안에 (원본 리셋과 같은 규약)
                cup_pose_now = torch.cat([env.cup.data.root_pos_w,
                                          env.cup.data.root_quat_w], dim=-1)
                bead_state = env._sample_bead_states_inside_cup(cup_pose_now)
                env.beads.write_object_state_to_sim(
                    bead_state, env_ids=torch.arange(1, device=env.device))
                env._beads_spawned[:] = True
                for f in range(60):                            # 구슬 정착
                    scene.write_data_to_sim()
                    env.sim.step(render=args.gui)
                    scene.update(dt_box[0])
                    watch()
                    if f % args.render_every == 0:
                        shot("6pour")
                print("[추종] 구슬 주입 완료 — 추종 시작", flush=True)
                for i in range(tq.shape[0]):
                    robot.set_joint_position_target(tq[i:i + 1], joint_ids=rids2)
                    robot.set_joint_velocity_target(tqd[i:i + 1], joint_ids=rids2)
                    for _ in range(int(pour_cfg.decimation)):
                        scene.write_data_to_sim()
                        env.sim.step(render=args.gui)
                        scene.update(dt_box[0])
                    refresh_receiver_buffer(env)
                    env._get_rewards()                         # 교차 계수 갱신
                    watch()
                    if i % 30 == 0:
                        _ins = int(env._bead_in_source[0].sum())
                        _int = int(env._bead_in_target[0].sum())
                        _bz = env.beads.data.object_pos_w[0, :, 2]
                        _floor = int((_bz < scene.env_origins[0, 2] + 0.25).sum())
                        print(f"  [추종{i:3d}] 교차 {int(env._bead_cross_count[0])}"
                              f"/{int(env.num_beads)} · 소스안 {_ins} · 받는안 {_int} · "
                              f"바닥 {_floor} · 소스컵 z "
                              f"{float(env.cup.data.root_pos_w[0, 2] - scene.env_origins[0, 2]):.3f}",
                              flush=True)
                    if i % args.render_every == 0:
                        shot("6pour")
                robot.set_joint_velocity_target(
                    torch.zeros(1, 7, device=env.device), joint_ids=rids2)
                # 받는컵 안 구슬 = 직접 계산 (env 카운터는 이 실행 경로에서 stale)
                _rc = env.left_target_cup.data.root_pos_w[0]
                _bp = env.beads.data.object_pos_w[0]
                _inb = int((((_bp[:, :2] - _rc[:2]).norm(dim=-1) < 0.043)
                            & (_bp[:, 2] > _rc[2] - 0.10)
                            & (_bp[:, 2] < _rc[2] + 0.12)).sum())
                pour_report = {"steps": int(tq.shape[0]),
                               "crossed": _inb,
                               "beads": int(env.num_beads),
                               "success": _inb >= 10}
                gate("⑧ 마무리 유지")
                passive("7end", args.hold)
            if args.pour_mode == "policy":
                    # ── ⑦ pour 폐루프 (e1_pour1) ────────────────────────────────────
                    gate("⑦ pour — 우팔이 왼손 shaker 에 붓는다")
                    if player_p.is_rnn:
                        player_p.init_rnn()
                    obs_p = env._get_observations()["policy"]
                    _nat_p = _SR / "logs/shadow/pour_entry/pour_obs0_native.npz"
                    if _nat_p.exists():
                        # obs0 결백 검사 — native 리셋 직후 표본의 min/max 범위 밖 세그먼트를 짚는다
                        _nat = np.load(_nat_p)["obs"]
                        _segs = (("arm_q", 7), ("arm_qd", 7), ("grasp_prog", 5), ("l_q", 9),
                                 ("l_qd", 9), ("pp_to_open", 3), ("pour_axis", 3),
                                 ("src_up", 3), ("tgt_up", 3), ("last_act", 6))
                        _mine = obs_p.reshape(-1).detach().cpu().numpy()
                        _i = 0
                        for _nm, _dd in _segs:
                            _lo = _nat[:, _i:_i + _dd].min(0) - 0.05
                            _hi = _nat[:, _i:_i + _dd].max(0) + 0.05
                            _out = float(np.maximum(_lo - _mine[_i:_i + _dd],
                                                    _mine[_i:_i + _dd] - _hi).max())
                            if _out > 0:
                                print(f"  [pour obs0 밖] {_nm}: 이탈 {_out:.3f} · "
                                      f"내 {np.round(_mine[_i:_i + _dd], 3).tolist()}", flush=True)
                            _i += _dd
                    succ_streak, p_steps = 0, 0
                    tgt_up_const = None
                    if args.receiver_up_contract:
                        # 훈련 받는컵 up = R(cfg quat)·ẑ — cfg 상수에서 직접 파생
                        from isaaclab.utils.math import quat_apply  # noqa: PLC0415
                        _q = T(pour_cfg.left_target_cup_quat_wxyz).unsqueeze(0)
                        tgt_up_const = quat_apply(_q, T([0.0, 0.0, 1.0]).unsqueeze(0))[0]
                        print(f"[pour] tgt_up 계약 상수 고정: "
                              f"{[round(float(v), 3) for v in tgt_up_const]}", flush=True)
                    for step in range(args.pour_steps):
                        if tgt_up_const is not None:
                            obs_p = obs_p.clone()
                            obs_p.view(-1)[46:49] = tgt_up_const
                        if args.diag_spoof_left_obs:
                            obs_p = obs_p.clone()
                            obs_p.view(-1)[19:28] = rest9_orig
                            obs_p.view(-1)[28:37] = 0.0
                        act = player_p.get_action(obs_p.reshape(1, -1), is_deterministic=True)
                        env._pre_physics_step(act.reshape(1, -1))
                        for _ in range(int(pour_cfg.decimation)):
                            env._apply_action()                # 컵 고정핀은 ⑥에서 무장해제됨
                            scene.write_data_to_sim()
                            env.sim.step(render=args.gui)
                            scene.update(dt_box[0])
                        refresh_receiver_buffer(env)           # obs 가 읽는 받는컵 버퍼 = 라이브
                        env.episode_length_buf += 1
                        env._get_rewards()                     # 상태·계수 갱신 (상태쓰기 없음 검증)
                        env._get_dones()
                        obs_p = env._get_observations()["policy"]
                        watch()
                        p_steps = step + 1
                        if step % 30 == 0:
                            _su = env._source_up_axis_w[0]
                            print(f"  [pour{step:3d}] mouth_xy {float(env._mouth_xy_distance[0]):.3f} · "
                                  f"gate {float(env._action_tilt_gate[0]):.2f} · "
                                  f"β {float(env._beta_cmd[0]):.2f} · "
                                  f"src_up_z {float(_su[2]):+.2f} · "
                                  f"교차 {int(env._bead_cross_count[0])}"
                                  f"/{int(env.num_beads)} · 성공 {bool(env.episode_success_buf[0])}"
                                  f" · 소스컵 z {float(env.cup.data.root_pos_w[0, 2] - scene.env_origins[0, 2]):.3f}",
                                  flush=True)
                        if step % args.render_every == 0:
                            shot("6pour")
                        if bool(env.episode_success_buf[0]):
                            succ_streak += 1
                            if succ_streak >= 60:
                                print(f"[pour] step {step}: 성공 판정 60스텝 유지 — 종료", flush=True)
                                break
                        else:
                            succ_streak = 0
                    robot.set_joint_velocity_target(
                        torch.zeros(1, len(env.arm_dof_indices), device=env.device),
                        joint_ids=list(env.arm_dof_indices))
                    pour_report = {
                        "steps": p_steps,
                        "crossed": int(env._bead_cross_count[0]),
                        "beads": int(env.num_beads),
                        "success": bool(env.episode_success_buf[0]),
                    }
                    gate("⑧ 마무리 유지")
                    passive("7end", args.hold)

    # ── 보고 ────────────────────────────────────────────────────────────────
    lz = float(env.left_target_cup.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    rz = float(env.cup.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    print(f"\n[결과] 좌 {l_steps}스텝 · 우 {r_steps}스텝 · {cups_z()}")
    print(f"[감시] |q̇|max {worst['qd']:.2f} rad/s · 컵속도 max {worst['cupv']:.2f} m/s "
          f"(임계 12 / 3 — 미러는 여기서 죽었다)")
    ok_l = grasp_lz > float(left_spawn[2]) + 0.05
    ok_r = grasp_rz > float(right_spawn[2]) + 0.05
    print(f"[판정] 좌 파지 {'✅' if ok_l else '❌'} (파지시 z {grasp_lz:.3f} · 최종 {lz:.3f}) · "
          f"우 파지 {'✅' if ok_r else '❌'} (파지시 z {grasp_rz:.3f} · 최종 {rz:.3f})")
    if pour_report is not None:
        pr = pour_report
        print(f"[판정] pour {'✅' if pr['success'] else '진행중/미성공'} — "
              f"{pr['steps']}스텝 · 구슬 교차 {pr['crossed']}/{pr['beads']} "
              f"(e1_pour1 최종본 ep6500)")

    if args.verify:
        _verify(trace, zl, zr)

    if args.render is not None:
        print(f"[렌더] {shots[0]}장 → {args.render}")
    if args.gui and not args.auto:
        print("\n창 유지 — Ctrl-C 로 닫는다")
        try:
            with torch.inference_mode():
                while simulation_app.is_running():
                    passive("live", 1)
        except KeyboardInterrupt:
            pass
    return 0 if (ok_l and ok_r) else 1


def _right_home(right_cfg) -> dict:
    jp = right_cfg.robot_cfg.init_state.joint_pos
    return dict(jp)


def _verify(trace: dict, zl, zr) -> None:
    """기록 에피소드 대비 편차 — 보고용(단정 아님: 접촉 카오스는 발산이 정상).

    잡으려는 것은 사슬 배선 오류다: 스케일·순서·앵커가 틀리면 **첫 스텝부터**
    수십 cm/rad 로 어긋난다. 물리 발산은 뒤로 갈수록 서서히 커진다 — 둘은
    시작 구간(초기 20스텝)의 편차로 구분된다.
    """
    def rms(a, b, n=None):
        n = n or min(len(a), len(b))
        if n == 0:
            return float("nan"), 0
        d = np.asarray(a[:n]) - np.asarray(b[:n])
        return float(np.sqrt((d ** 2).mean())), n

    print("\n[재현 대조] (같은 스폰·goal·노이즈0 — 초기 20스텝이 배선 판정)")
    for key, rec, name in (("l_palm", zl.get("palm_targets"), "좌 palm 지령 6D"),
                           ("l_armt", zl.get("arm_target"), "좌 팔 관절지령 7D"),
                           ("r_palm", zr.get("palm_targets"), "우 palm 지령 6D"),
                           ("r_armt", zr.get("arm_target"), "우 팔 관절지령 7D")):
        if rec is None or not trace[key]:
            continue
        e20, _ = rms(trace[key], rec, min(20, len(trace[key]), len(rec)))
        efull, n = rms(trace[key], rec)
        print(f"  {name}: 초기20 RMS {e20:.4f} · 전체({n}) RMS {efull:.4f}")
    if trace["r_latch"] and "latched" in zr:
        mine = next((i for i, v in enumerate(trace["r_latch"]) if v), -1)
        rec = next((i for i, v in enumerate(zr["latched"]) if v > 0.5), -1)
        print(f"  우 latch 프레임: 러너 {mine} vs 기록 {rec}")


if __name__ == "__main__":
    code = main()
    # ★simulation_app.close() 가 replicator attach 상태에서 행 — 프로세스가 좀비로
    #   남아 GPU 를 물고 결과 출력이 버퍼에 갇힌다(09.02 실측, 47분 좀비 2개).
    #   결과를 밀어내고 즉시 죽는다 — 프로세스 종료가 GPU 를 돌려준다.
    sys.stdout.flush()
    sys.stderr.flush()
    import os
    os._exit(int(code or 0))

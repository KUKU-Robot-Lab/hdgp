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
                    default=_SR / "logs/policy/right_e1/nn/e1_best.pth")
parser.add_argument("--left-stream", type=Path,
                    default=_SR / "logs/shadow/pour_entry/stream_left_v2b25.npz",
                    help="goal·컵 스폰·(--verify 시) 대조 기준")
parser.add_argument("--right-stream", type=Path,
                    default=_SR / "logs/shadow/pour_entry/stream_right_e1_v2.npz")
parser.add_argument("--left-steps", type=int, default=300)
parser.add_argument("--right-steps", type=int, default=420)
parser.add_argument("--stop-lift", type=float, default=0.08)
parser.add_argument("--lift-hold", type=int, default=40)
parser.add_argument("--settle", type=int, default=120)
parser.add_argument("--hold", type=int, default=120)
parser.add_argument("--final-hold", type=int, default=300)
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

    left_yaml = yaml.unsafe_load(
        (args.left_checkpoint.parent.parent / "params" / "env.yaml").read_text())

    pour_cfg = parse_env_cfg(POUR_TASK, device=DEVICE, num_envs=1)
    lgrip = str(_REPO / "assets/robot/openarm_tesollo_sensor_rl_lgrip"
                / "openarm_tesollo_sensor_rl.usd")
    for line in align_pour_cfg(pour_cfg, left_scene=left_yaml["scene"],
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

    # ── 보고 ────────────────────────────────────────────────────────────────
    lz = float(env.left_target_cup.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    rz = float(env.cup.data.root_pos_w[0, 2] - scene.env_origins[0, 2])
    print(f"\n[결과] 좌 {l_steps}스텝 · 우 {r_steps}스텝 · {cups_z()}")
    print(f"[감시] |q̇|max {worst['qd']:.2f} rad/s · 컵속도 max {worst['cupv']:.2f} m/s "
          f"(임계 12 / 3 — 미러는 여기서 죽었다)")
    ok_l, ok_r = lz > float(left_spawn[2]) + 0.05, rz > float(right_spawn[2]) + 0.05
    print(f"[판정] 좌 파지 {'✅' if ok_l else '❌'} (shaker z {lz:.3f}) · "
          f"우 파지 {'✅' if ok_r else '❌'} (cup_big z {rz:.3f})")

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

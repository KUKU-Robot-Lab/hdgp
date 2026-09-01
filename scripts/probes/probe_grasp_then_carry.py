#!/usr/bin/env python3
"""파지 정책을 굴려 **실제로 컵을 쥔 다음**, 붓기 시작 자세까지 옮기는 것을 찍는다.

**왜 정책을 굴리는가.** 자세를 손으로 합성해 텔레포트하고 컵을 붙이면 답이 안 나온다 —
파지 조임과 컵 고정이 양립하지 않아 수만 N 이 나거나 씬이 폭발한다(09.01 실측).
정책을 그대로 굴리면 **컵은 물리가 잡는다.** 그 뒤의 이송은 배포에서도 스크립트이므로
관절 램프로 간다.

두 단계다.

  ① **파지** — 정책이 `env.step` 으로 돈다. 컵은 진짜로 쥐어진다.
  ② **이송** — 정책을 놓고 `sim.step` 을 직접 밟으며 팔 관절을 목표로 램프한다.
     손 지령은 ①이 마지막으로 낸 값을 그대로 유지한다 — 놓으면 컵을 떨어뜨린다.

좌우는 태스크가 달라 한 판에 같이 못 돈다. 각각 돌린다.

    # 우팔 (E1, cup_big_s100)
    ../IsaacLab/isaaclab.sh -p scripts/probes/probe_grasp_then_carry.py \\
        --task open-sens_r_grasp_s2r-play-lstm --checkpoint <e1.pth> \\
        --goto " -0.0547,-0.1049,-0.1815,1.2056,0.2882,-0.7815,0.6346" --render <dir>

    # 좌팔 (v2B25, shaker) — ★HDGP_V2_VENDOR_GAINS=1 필수
    HDGP_V2_VENDOR_GAINS=1 ../IsaacLab/isaaclab.sh -p scripts/probes/probe_grasp_then_carry.py \\
        --task open-grip_l_grasp_sensor_v2-play --checkpoint <v2b25.pth> \\
        --goto " -0.315,-0.079,0.217,0.513,0.666,-0.729,-0.957" --render <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--agent", default="rl_games_cfg_entry_point")
parser.add_argument("--goto", default="", help="이송 목표 팔 7관절 (콤마). 비우면 이송 없이 유지만")
parser.add_argument("--from-rest", action="store_true",
                    help="차렷에서 시작해 정책 홈까지 램프하는 것을 먼저 찍는다")
parser.add_argument("--other-arm", default="",
                    help="반대 팔을 이 자세로 세워 둔다 (7관절). 정책과 무관하지만 그림에는 나온다")
parser.add_argument("--other-state", type=Path, default=None,
                    help="앞 판이 --save-state 로 남긴 json. 반대 팔을 그 자세로 세우고 "
                         "그 손에 컵까지 들려 준다 — 이래야 양팔이 각자 컵을 든 그림이 된다")
parser.add_argument("--other-cup-usd", default="",
                    help="반대 손에 들려 줄 컵 USD. 생략하면 shaker_closed_rl.usd")
parser.add_argument("--preset-settle", type=int, default=600,
                    help="preset 램프 뒤 정착을 기다릴 최대 스텝")
parser.add_argument("--preset-tol", type=float, default=0.01,
                    help="rad — 이보다 가까워지면 도착으로 본다")
parser.add_argument("--hold-steps", type=int, default=90,
                    help="정책이 끝난 뒤 그 자세를 유지하며 찍을 스텝 수")
parser.add_argument("--policy-steps", type=int, default=420, help="파지 단계 최대 스텝")
parser.add_argument("--stop-on-lift", type=float, default=0.0,
                    help="컵이 이만큼 올라가고 유지되면 정책을 멈춘다 [m]. "
                         "0 이면 안 멈춘다. ★안 쓰면 에피소드가 리셋되며 파지를 무한 반복한다")
parser.add_argument("--lift-hold", type=int, default=40,
                    help="--stop-on-lift 판정을 이만큼 연속 유지해야 멈춘다")
parser.add_argument("--carry-max-vel", type=float, default=0.5, help="rad/s")
parser.add_argument("--env-id", type=int, default=0, help="찍을 env")
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--render", type=Path, default=None, help="PNG 시퀀스 (GUI 로 볼 때는 생략)")
parser.add_argument("--keep-open", action="store_true",
                    help="끝난 뒤 창을 열어 둔다 (GUI 로 볼 때). Ctrl-C 로 닫는다")
parser.add_argument("--save-stream", type=Path, default=None,
                    help="정책 단계의 팔·손 관절 지령 스트림을 npz 로 남긴다 — 통합 씬 재생용")
parser.add_argument("--save-state", type=Path, default=None,
                    help="정책 종료 시점의 팔·손 상태와 지령을 json 으로 남긴다")
parser.add_argument("--render-every", type=int, default=4)
parser.add_argument("--gui", action="store_true")

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "source" / "openarm"), str(_REPO / "scripts" / "tools"),
           str(_REPO.parent / "sim2real" / "scripts")):
    sys.path.insert(0, _p)

from isaaclab.app import AppLauncher                              # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
args.headless = not args.gui
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import math                                                       # noqa: E402
import gymnasium as gym                                           # noqa: E402
import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402

import openarm  # noqa: E402,F401
import openarm.tasks  # noqa: E402,F401
from isaaclab.envs import DirectRLEnvCfg                          # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config          # noqa: E402
from rl_games.common import env_configurations, vecenv            # noqa: E402
from rl_games.torch_runner import Runner                          # noqa: E402

from run_cfg_restore import restore_run_cfg_if_available          # noqa: E402
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms  # noqa: E402
from transition_plan import ramp                                  # noqa: E402

#: 팔·손 관절을 이름으로 찾는다 — 자산마다 순서가 다르므로 위치로 자르면 어긋난다.
_ARM = {"r": [f"r_aj_{i}" for i in range(1, 8)], "l": [f"l_aj_{i}" for i in range(1, 8)]}
_HAND = {
    "r": [f"r_hj_{f}_{j}" for f in ("thumb", "index", "middle", "ring", "pinky")
          for j in range(1, 5)],
    "l": ["l_hj_gripper_1", "l_hj_gripper_2"],
}


#: 컵을 물릴 기준 body. 이 프레임에서 본 컵 자세를 저장해 다음 판에 그대로 들려 준다.
_HAND_BODY = {"r": "r_hl_palm", "l": "l_hl_gripper_base"}


def _ids(robot, names: list[str]) -> list[int]:
    missing = [n for n in names if n not in robot.joint_names]
    if missing:
        raise SystemExit(f"로봇에 없는 관절: {missing}")
    return [robot.joint_names.index(n) for n in names]


@hydra_task_config(args.task, args.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    agent_cfg = restore_run_cfg_if_available(
        env_cfg, agent_cfg, resume_path=str(args.checkpoint), workspace_root=str(_REPO.parent))
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    agent_cfg["params"]["seed"] = args.seed
    for attr in ("enable_adr", "enable_success_adr"):
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, False)

    # ★반대 손에 들려 줄 컵은 **env 생성 전에 씬 cfg 로** 넣어야 한다. 생성 뒤에
    #   RigidObject 를 만들면 물리가 이미 돌아 "Failed to create rigid body" 로 죽는다.
    other_state = None
    if args.other_state is not None:
        other_state = json.loads(args.other_state.read_text())
        from isaaclab.assets import RigidObjectCfg  # noqa: PLC0415
        import isaaclab.sim as sim_utils  # noqa: PLC0415
        usd = args.other_cup_usd or str(_REPO / "assets" / "cup" / "shaker_closed_rl.usd")
        env_cfg.scene.other_cup = RigidObjectCfg(
            prim_path="/World/envs/env_.*/OtherCup",
            spawn=sim_utils.UsdFileCfg(usd_path=usd),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.5)))
        print(f"[반대팔] {args.other_state.name} · 컵 {usd.split('/')[-1]} 을 씬에 추가")

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

    # DirectRLEnv 는 `base.robot`, ManagerBasedRLEnv 는 `base.scene["robot"]` 이다.
    robot = getattr(base, "robot", None)
    if robot is None:
        robot = base.scene["robot"]
    side = "l" if "l_aj_1" in robot.joint_names and args.task.startswith("open-grip") else "r"
    arm_ids, hand_ids = _ids(robot, _ARM[side]), _ids(robot, _HAND[side])
    dt = float(base.step_dt)
    def _seven(text: str, what: str) -> list[float] | None:
        if not text.strip():
            return None
        vals = [float(v) for v in text.split(",")]
        if len(vals) != 7:
            raise SystemExit(f"{what} 는 7개여야 한다: {len(vals)}개")
        return vals

    goal = _seven(args.goto, "--goto")
    if other_state is not None:
        other_side = "l" if side == "r" else "r"
        args.other_arm = ",".join(str(v) for v in other_state[f"{other_side}_aj"])
    other = _seven(args.other_arm, "--other-arm")
    other_ids = _ids(robot, _ARM["l" if side == "r" else "r"]) if other else None
    print(f"[설정] {side}팔 · 관절 {len(robot.joint_names)} · dt {dt:.4f}s "
          f"· 파지 {args.policy_steps}스텝 · 이송 {args.carry_max_vel} rad/s")

    # ── 렌더 ───────────────────────────────────────────────────────────────
    shots = [0]
    _rend = {}

    if args.render is None:
        print("[렌더] 생략 — GUI 로 본다")

        def shot(tag: str) -> None:
            return
    else:
        import omni.replicator.core as rep  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        args.render.mkdir(parents=True, exist_ok=True)
        origin = base.scene.env_origins[args.env_id].cpu().numpy()
        center = origin + np.array([0.33, -0.05 if side == "r" else 0.05, 0.33])
        cam = rep.create.camera(
            position=tuple(float(v) for v in center + np.array([1.05, -0.80, 0.50])),
            look_at=tuple(float(v) for v in center))
        rp = rep.create.render_product(cam, (1280, 800))
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach([rp])
        _rend["annot"], _rend["Image"] = annot, Image

        def shot(tag: str) -> None:
            base.sim.render()
            arr = np.asarray(_rend["annot"].get_data())
            if arr.size:
                _rend["Image"].fromarray(arr[:, :, :3]).save(
                    args.render / f"{shots[0]:04d}_{tag}.png")
                shots[0] += 1

    def cup_z() -> float:
        for name in ("cup", "object", "left_target_cup"):
            obj = getattr(base, name, None)
            if obj is None:
                try:
                    obj = base.scene[name]
                except (KeyError, TypeError):
                    obj = None
            if obj is not None and hasattr(obj, "data"):
                return float(obj.data.root_pos_w[args.env_id, 2]
                             - base.scene.env_origins[args.env_id, 2])
        return float("nan")

    _other_q = (torch.tensor([other], dtype=torch.float32, device=base.device)
                .repeat(base.num_envs, 1) if other_ids is not None else None)
    _other_placed = [False]

    # ★반대 팔의 **손**도 세운다. 팔만 세우면 그리퍼가 열린 채라 컵을 든 그림이 아니다.
    _other_hand = {}
    if other_state is not None:
        oside = "l" if side == "r" else "r"
        hkey = "l_hj_gripper" if oside == "l" else "r_hj"
        if hkey in other_state:
            _other_hand["ids"] = _ids(robot, _HAND[oside])
            _other_hand["q"] = torch.tensor(
                [other_state[hkey]], dtype=torch.float32, device=base.device
            ).repeat(base.num_envs, 1)
            _other_hand["t"] = torch.tensor(
                [other_state.get(hkey + "_target", other_state[hkey])],
                dtype=torch.float32, device=base.device).repeat(base.num_envs, 1)
            print(f"[반대팔] {oside}손도 파지 자세로 — 실측 {other_state[hkey]} "
                  f"지령 {other_state.get(hkey + '_target')}")

    def stand_other() -> None:
        """반대 팔을 지정 자세로 세운다. 정책과 무관하지만 그림에는 나온다.

        ★텔레포트(`write_joint_state_to_sim`)는 **처음 한 번만**. 매 스텝 부르면
          joint_acc 인덱싱에서 죽는다(실측).
        """
        if other_ids is None:
            return
        if not _other_placed[0]:
            robot.write_joint_state_to_sim(_other_q, torch.zeros_like(_other_q),
                                           joint_ids=other_ids)
            if _other_hand:
                robot.write_joint_state_to_sim(
                    _other_hand["q"], torch.zeros_like(_other_hand["q"]),
                    joint_ids=_other_hand["ids"])
            _other_placed[0] = True
        robot.set_joint_position_target(_other_q, joint_ids=other_ids)
        if _other_hand:
            robot.set_joint_position_target(_other_hand["t"], joint_ids=_other_hand["ids"])

    # ── 반대 손에 컵 들려 주기 ─────────────────────────────────────────────
    _other_cup = {}
    if other_state is not None and other_state.get("cup_in_hand"):
        _other_cup["obj"] = base.scene["other_cup"]
        _other_cup["body"] = robot.body_names.index(other_state["hand_body"])
        _other_cup["rel"] = torch.tensor(
            other_state["cup_in_hand"], dtype=torch.float32, device=base.device).unsqueeze(0)
        print(f"[반대팔] 컵을 {other_state['hand_body']} 에 들려 준다")

    def pin_other_cup() -> None:
        """반대 팔은 정지해 있으므로 컵을 그 손에 고정해도 정직하다 (pour 도 같은 방식)."""
        if not _other_cup:
            return
        bi = _other_cup["body"]
        pos, quat = combine_frame_transforms(
            robot.data.body_pos_w[0:1, bi], robot.data.body_quat_w[0:1, bi],
            _other_cup["rel"][:, :3], _other_cup["rel"][:, 3:])
        _other_cup["obj"].write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
        _other_cup["obj"].write_root_velocity_to_sim(torch.zeros(1, 6, device=base.device))

    # ── ① 파지 : 정책이 돈다 ───────────────────────────────────────────────
    obs = wrapped.reset()
    stand_other()
    pin_other_cup()

    if args.from_rest:
        # 차렷 → 정책 홈. 정책이 출발하는 자리를 **어떻게 가는지** 보여준다.
        home = robot.data.joint_pos[args.env_id, arm_ids].cpu().numpy().tolist()
        zeros = [0.0] * 7
        q0 = torch.zeros(base.num_envs, 7, device=base.device)
        robot.write_joint_state_to_sim(q0, q0, joint_ids=arm_ids)
        hand_now = robot.data.joint_pos[:, hand_ids].clone()
        print(f"[preset] 차렷 → 정책 홈 {[round(v,3) for v in home]}")
        for f, row in enumerate(ramp(zeros, home, max_vel=args.carry_max_vel, dt=dt)):
            q = torch.tensor(row, dtype=torch.float32, device=base.device).repeat(base.num_envs, 1)
            robot.set_joint_position_target(q, joint_ids=arm_ids)
            robot.set_joint_position_target(hand_now, joint_ids=hand_ids)
            stand_other()
            robot.write_data_to_sim()
            base.sim.step(render=False)
            robot.update(dt)
            if f % args.render_every == 0:
                shot(f"0preset_{f:04d}")
        # ★램프가 끝났다고 도착한 게 아니다. PD 는 지령을 뒤따라오므로 **정착을 기다려야**
        #   한다 — 안 기다리면 36° 뒤처진 자세에서 정책이 출발한다(09.01 실측).
        def _err() -> float:
            return float(np.abs(robot.data.joint_pos[args.env_id, arm_ids].cpu().numpy()
                                - np.array(home)).max())

        q_home = torch.tensor(home, dtype=torch.float32,
                              device=base.device).repeat(base.num_envs, 1)
        for k in range(args.preset_settle):
            robot.set_joint_position_target(q_home, joint_ids=arm_ids)
            robot.set_joint_position_target(hand_now, joint_ids=hand_ids)
            stand_other()
            robot.write_data_to_sim()
            base.sim.step(render=False)
            robot.update(dt)
            if k % args.render_every == 0:
                shot(f"0settle_{k:04d}")
            if _err() < args.preset_tol:
                break
        err = _err()
        print(f"[preset] 끝 · {shots[0]}장 · 정착 {k+1}스텝 · 도착오차 {err:.4f} rad "
              f"({np.degrees(err):.2f}°) {'✅' if err < args.preset_tol else '❌ 아직 멀다'}")
    if isinstance(obs, dict):
        obs = obs["obs"]
    player.get_batch_size(obs, 1)
    if player.is_rnn:
        player.init_rnn()
    z0 = cup_z()
    streak = [0]
    stream: dict[str, list] = {"arm_target": [], "hand_target": [], "cup_pos": []}
    with torch.inference_mode():
        for step in range(args.policy_steps):
            action = player.get_action(player.obs_to_torch(obs), is_deterministic=True)
            obs, _, dones, _ = wrapped.step(action)
            if isinstance(obs, dict):
                obs = obs["obs"]
            if player.is_rnn and player.states is not None:
                for s in player.states:
                    s[:, dones, :] = 0.0
            stand_other()
            pin_other_cup()
            if args.save_stream is not None:
                stream["arm_target"].append(
                    robot.data.joint_pos_target[args.env_id, arm_ids].cpu().numpy().copy())
                stream["hand_target"].append(
                    robot.data.joint_pos_target[args.env_id, hand_ids].cpu().numpy().copy())
                stream["cup_pos"].append(
                    np.array([cup_z()], dtype=np.float32))
            if args.stop_on_lift > 0:
                lifted_now = (cup_z() - z0) > args.stop_on_lift
                streak[0] = streak[0] + 1 if lifted_now else 0
                if streak[0] >= args.lift_hold:
                    print(f"[파지] step {step} 에서 멈춘다 — 컵 Δz "
                          f"{cup_z()-z0:+.3f} m 를 {args.lift_hold}스텝 유지")
                    if step % args.render_every == 0:
                        shot(f"1grasp_{step:04d}")
                    break
            if step % args.render_every == 0:
                shot(f"1grasp_{step:04d}")
            if step % 100 == 0:
                print(f"[파지] {step}/{args.policy_steps} · 컵 z {cup_z():.3f} "
                      f"(Δ{cup_z()-z0:+.3f} m)")
    lift = cup_z() - z0
    print(f"[파지] 끝 · 컵 Δz {lift:+.3f} m {'✅ 들었다' if lift > 0.03 else '❌ 못 들었다'}")

    # ── ② 이송 : 정책을 놓고 관절 램프 ─────────────────────────────────────
    #    ★손 지령은 ①이 마지막으로 낸 값을 그대로 쥔다. 놓으면 컵을 떨어뜨린다.
    hand_hold = robot.data.joint_pos_target[:, hand_ids].clone()
    start = robot.data.joint_pos[args.env_id, arm_ids].cpu().numpy().tolist()
    path = ramp(start, goal if goal else start, max_vel=args.carry_max_vel, dt=dt)
    if goal is None:
        path = np.array([start] * max(args.hold_steps, 1))
        print(f"[유지] --goto 없음 — 정책 종료 자세를 {len(path)}스텝 유지한다")
    span = float(np.abs(np.array(goal) - np.array(start)).max()) if goal else 0.0
    print(f"[{'이송' if goal else '유지'}] {path.shape[0]}프레임 "
          f"{(path.shape[0]-1)*dt:.1f}s · 최대 |Δq| {span:.3f} rad")
    with torch.inference_mode():
        for f, row in enumerate(path):
            q = torch.tensor(row, dtype=torch.float32, device=base.device).repeat(base.num_envs, 1)
            robot.set_joint_position_target(q, joint_ids=arm_ids)
            robot.set_joint_position_target(hand_hold, joint_ids=hand_ids)
            stand_other()
            pin_other_cup()
            robot.write_data_to_sim()
            base.sim.step(render=False)
            robot.update(dt)
            if f % args.render_every == 0:
                shot(f"2{'carry' if goal else 'hold'}_{f:04d}")
            if f % 40 == 0:
                print(f"[{'이송' if goal else '유지'}] {f}/{path.shape[0]} · 컵 z {cup_z():.3f} (Δ{cup_z()-z0:+.3f} m)")
    held = cup_z() - z0
    print(f"[{'이송' if goal else '유지'}] 끝 · 컵 Δz {held:+.3f} m "
          f"{'✅ 유지' if held > 0.03 else '❌ 놓쳤다'}")
    if args.render is not None:
        print(f"[렌더] {shots[0]}장 → {args.render}")

    def _cup_in_hand() -> list[float] | None:
        """손 프레임에서 본 컵 자세 (pos3 + quat wxyz). 다음 판에 그대로 들려 준다."""
        obj = None
        for name in ("cup", "object", "left_target_cup"):
            obj = getattr(base, name, None)
            if obj is None:
                try:
                    obj = base.scene[name]
                except (KeyError, TypeError):
                    obj = None
            if obj is not None and hasattr(obj, "data"):
                break
        if obj is None:
            return None
        bi = robot.body_names.index(_HAND_BODY[side])
        pos, quat = subtract_frame_transforms(
            robot.data.body_pos_w[args.env_id:args.env_id + 1, bi],
            robot.data.body_quat_w[args.env_id:args.env_id + 1, bi],
            obj.data.root_pos_w[args.env_id:args.env_id + 1],
            obj.data.root_quat_w[args.env_id:args.env_id + 1])
        return [round(float(v), 5) for v in torch.cat([pos[0], quat[0]])]

    if args.save_stream is not None:
        args.save_stream.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_stream,
            arm_target=np.stack(stream["arm_target"]),
            hand_target=np.stack(stream["hand_target"]),
            cup_z=np.stack(stream["cup_pos"]),
            meta_arm_names=np.array(_ARM[side]),
            meta_hand_names=np.array(_HAND[side]),
            meta_step_dt=np.float32(dt),
            meta_checkpoint=str(args.checkpoint),
            meta_task=str(args.task))
        print(f"[스트림] {len(stream['arm_target'])}프레임 → {args.save_stream}")

    if args.save_state is not None:
        args.save_state.parent.mkdir(parents=True, exist_ok=True)
        e = args.env_id
        r3 = lambda t: [round(float(v), 5) for v in t]  # noqa: E731
        args.save_state.write_text(json.dumps({
            "_설명": f"{side}팔 정책 종료 상태 — 실측(q)과 PD 지령(target). 지령이 파지력이다",
            "_task": args.task, "_checkpoint": str(args.checkpoint),
            "_cup_dz": round(float(held), 4),
            f"{side}_aj": r3(robot.data.joint_pos[e, arm_ids]),
            f"{side}_aj_target": r3(robot.data.joint_pos_target[e, arm_ids]),
            ("r_hj" if side == "r" else "l_hj_gripper"): r3(robot.data.joint_pos[e, hand_ids]),
            ("r_hj_target" if side == "r" else "l_hj_gripper_target"):
                r3(robot.data.joint_pos_target[e, hand_ids]),
            "cup_in_hand": _cup_in_hand(),
            "hand_body": _HAND_BODY[side],
        }, ensure_ascii=False, indent=2))
        print(f"[상태] {args.save_state}")

    if args.keep_open:
        print("\n★창을 열어 둔다 — 자세를 유지하며 계속 렌더한다. Ctrl-C 로 닫는다.")
        try:
          with torch.inference_mode():
            while simulation_app.is_running():
                robot.set_joint_position_target(
                    robot.data.joint_pos_target[:, arm_ids].clone(), joint_ids=arm_ids)
                robot.set_joint_position_target(hand_hold, joint_ids=hand_ids)
                stand_other()
                robot.write_data_to_sim()
                base.sim.step(render=True)
                robot.update(dt)
        except KeyboardInterrupt:
            print("닫는다")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code or 0)

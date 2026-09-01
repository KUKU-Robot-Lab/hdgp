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
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--agent", default="rl_games_cfg_entry_point")
parser.add_argument("--goto", required=True, help="이송 목표 팔 7관절 (콤마)")
parser.add_argument("--policy-steps", type=int, default=420, help="파지 단계 스텝 수")
parser.add_argument("--carry-max-vel", type=float, default=0.5, help="rad/s")
parser.add_argument("--env-id", type=int, default=0, help="찍을 env")
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--render", type=Path, required=True)
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
from transition_plan import ramp                                  # noqa: E402

#: 팔·손 관절을 이름으로 찾는다 — 자산마다 순서가 다르므로 위치로 자르면 어긋난다.
_ARM = {"r": [f"r_aj_{i}" for i in range(1, 8)], "l": [f"l_aj_{i}" for i in range(1, 8)]}
_HAND = {
    "r": [f"r_hj_{f}_{j}" for f in ("thumb", "index", "middle", "ring", "pinky")
          for j in range(1, 5)],
    "l": ["l_hj_gripper_1", "l_hj_gripper_2"],
}


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
    goal = [float(v) for v in args.goto.split(",")]
    if len(goal) != 7:
        raise SystemExit(f"--goto 는 7개여야 한다: {len(goal)}개")
    print(f"[설정] {side}팔 · 관절 {len(robot.joint_names)} · dt {dt:.4f}s "
          f"· 파지 {args.policy_steps}스텝 · 이송 {args.carry_max_vel} rad/s")

    # ── 렌더 ───────────────────────────────────────────────────────────────
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
    shots = [0]

    def shot(tag: str) -> None:
        base.sim.render()
        arr = np.asarray(annot.get_data())
        if arr.size:
            Image.fromarray(arr[:, :, :3]).save(args.render / f"{shots[0]:04d}_{tag}.png")
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

    # ── ① 파지 : 정책이 돈다 ───────────────────────────────────────────────
    obs = wrapped.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    player.get_batch_size(obs, 1)
    if player.is_rnn:
        player.init_rnn()
    z0 = cup_z()
    with torch.inference_mode():
        for step in range(args.policy_steps):
            action = player.get_action(player.obs_to_torch(obs), is_deterministic=True)
            obs, _, dones, _ = wrapped.step(action)
            if isinstance(obs, dict):
                obs = obs["obs"]
            if player.is_rnn and player.states is not None:
                for s in player.states:
                    s[:, dones, :] = 0.0
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
    path = ramp(start, goal, max_vel=args.carry_max_vel, dt=dt)
    print(f"[이송] {path.shape[0]}프레임 {(path.shape[0]-1)*dt:.1f}s · "
          f"최대 |Δq| {np.abs(np.array(goal)-np.array(start)).max():.3f} rad")
    for f, row in enumerate(path):
        q = torch.tensor(row, dtype=torch.float32, device=base.device).repeat(base.num_envs, 1)
        robot.set_joint_position_target(q, joint_ids=arm_ids)
        robot.set_joint_position_target(hand_hold, joint_ids=hand_ids)
        robot.write_data_to_sim()
        base.sim.step(render=False)
        robot.update(dt)
        if f % args.render_every == 0:
            shot(f"2carry_{f:04d}")
        if f % 40 == 0:
            print(f"[이송] {f}/{path.shape[0]} · 컵 z {cup_z():.3f} (Δ{cup_z()-z0:+.3f} m)")
    held = cup_z() - z0
    print(f"[이송] 끝 · 컵 Δz {held:+.3f} m {'✅ 유지' if held > 0.03 else '❌ 놓쳤다'}")
    print(f"[렌더] {shots[0]}장 → {args.render}")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code or 0)

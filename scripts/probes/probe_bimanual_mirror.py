#!/usr/bin/env python3
"""양팔 순차 파지를 **미러 방식**으로 한 씬에서 보인다 — 실기 배포와 같은 구조.

실기 좌팔 라이브의 실제 구조: 정책은 자기 env 에서 폐루프로 돌고, 로봇은 그
**실측 관절 스트림을 따라간다**(probe_sim_follower 가 sim 미러로 같은 일을 한다).
여기서는 그 미러 로봇 역할을 이 씬이 한다 — 각 정책이 자기 env 에서 만든 실측
스트림을 관절 상태로 그대로 재생한다. PD 재생이 아니므로 게인·effort·지연이 낄
자리가 없다(09.02: PD 재생은 이동 중 0.3~0.4 rad 지연 + 자산 기하 차로 파지 재현
실패).

컵: 각 정책 env 폐루프에서 파지가 **물리로 증명**된 시점의 (손 ← 컵) 상대자세
(`state_*_end.json` 의 `cup_in_hand`)로, 기록상 컵이 들리기 시작하는 프레임에 손에
물린다. 그 전에는 테이블에 놓여 있다.

순서:  정착 → [Enter] 좌팔(v2B25 실측) → shaker 부착·리프트 → [Enter] 유지
       → [Enter] 우팔(E1) → cup_big 부착·리프트 → [Enter] 양팔 유지
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", default="open-tesol_r_pour_sensor-play-lstm")
parser.add_argument("--left-stream", type=Path, required=True)
parser.add_argument("--left-env", type=int, default=0)
parser.add_argument("--left-frames", type=int, default=250)
parser.add_argument("--left-state", type=Path, required=True, help="state_left_end.json")
parser.add_argument("--right-stream", type=Path, required=True)
parser.add_argument("--right-state", type=Path, required=True, help="state_right_end.json")
parser.add_argument("--right-cup-xy", default="0.362,-0.16")
parser.add_argument("--repeat", type=int, default=2, help="스트림 1프레임당 sim 스텝 수 (감속)")
parser.add_argument("--settle", type=int, default=60)
parser.add_argument("--hold", type=int, default=110)
parser.add_argument("--final-hold", type=int, default=240)
parser.add_argument("--auto", action="store_true")
parser.add_argument("--render", type=Path, default=None)
parser.add_argument("--render-every", type=int, default=5)
parser.add_argument("--gui", action="store_true")

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "source" / "openarm"), str(_REPO / "scripts" / "tools")):
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

import openarm  # noqa: E402,F401
import openarm.tasks  # noqa: E402,F401
from isaaclab.utils.math import combine_frame_transforms          # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config          # noqa: E402

_ARM = {"l": [f"l_aj_{i}" for i in range(1, 8)], "r": [f"r_aj_{i}" for i in range(1, 8)]}
_GRIP_L = ["l_hj_gripper_1", "l_hj_gripper_2"]


def _lift_frame(z_series: np.ndarray, rise: float = 0.02) -> int:
    """기록에서 컵이 들리기 시작한 프레임 — 여기서 컵을 손에 물린다."""
    dz = z_series - z_series[0]
    up = np.nonzero(dz > rise)[0]
    return int(up[0]) if len(up) else len(z_series) - 1


@hydra_task_config(args.task, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    env_cfg.scene.num_envs = 1
    for attr in ("enable_adr", "enable_success_adr"):
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, False)
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.reset()
    robot = env.robot
    dt = float(env.step_dt)
    origin = env.scene.env_origins[0]

    # ── 스트림 (★실측 — PD 재생이 아니라 상태 미러) ────────────────────────
    zl = np.load(args.left_stream, allow_pickle=True)
    e = args.left_env
    L_arm = zl["arm_meas"][: args.left_frames, e]
    L_grip = zl["grip_meas"][: args.left_frames, e]
    L_cupz = zl["cup_pos"][: args.left_frames, e, 2]
    L_cup0 = zl["cup_pos"][0, e]
    L_attach = _lift_frame(L_cupz)
    zr = np.load(args.right_stream, allow_pickle=True)
    R_arm = zr["arm_target"]           # 우팔은 실측 기록이 없어 지령을 쓴다(자기 env 에서 잘 추종했음)
    R_hand = zr["hand_target"]
    R_hand_names = [str(x) for x in zr["meta_hand_names"]]
    R_cupz = zr["cup_z"][:, 0]
    R_attach = _lift_frame(R_cupz)
    rx, ry = (float(v) for v in args.right_cup_xy.split(","))
    print(f"[스트림] 좌 {L_arm.shape[0]}프레임(부착 {L_attach}) · "
          f"우 {R_arm.shape[0]}프레임(부착 {R_attach})")

    Lst = json.loads(args.left_state.read_text())
    Rst = json.loads(args.right_state.read_text())

    ids_la = [robot.joint_names.index(n) for n in _ARM["l"]]
    ids_lg = [robot.joint_names.index(n) for n in _GRIP_L]
    ids_ra = [robot.joint_names.index(n) for n in _ARM["r"]]
    ids_rh = [robot.joint_names.index(n) for n in R_hand_names]
    bi_l = robot.body_names.index(Lst["hand_body"])
    bi_r = robot.body_names.index(Rst["hand_body"])

    def T(a):
        return torch.tensor(np.asarray(a, dtype=np.float32)[None], device=env.device)

    def set_state(ids, arr):
        q = T(arr)
        robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=ids)
        robot.set_joint_position_target(q, joint_ids=ids)

    set_state(ids_la, L_arm[0])
    set_state(ids_lg, L_grip[0])
    set_state(ids_ra, R_arm[0])
    set_state(ids_rh, R_hand[0])

    def place(obj, pos):
        pose = torch.cat([origin + torch.tensor(pos, device=env.device),
                          torch.tensor([1.0, 0, 0, 0], device=env.device)]).unsqueeze(0)
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))

    place(env.cup, [rx, ry, float(R_cupz[0])])
    place(env.left_target_cup, [float(L_cup0[0]), float(L_cup0[1]), float(L_cup0[2])])
    if hasattr(env, "_hide_beads"):
        env._hide_beads(torch.arange(1, device=env.device))

    # ── 부착 ───────────────────────────────────────────────────────────────
    _held: dict[str, tuple] = {}

    from isaaclab.utils.math import subtract_frame_transforms  # noqa: PLC0415

    def attach(name: str, cup, bi: int):
        """부착 시점의 **현재** (손 ← 컵) 상대자세를 잡는다.

        ★기록 env 의 `cup_in_hand` 를 그대로 이식하면 안 된다 — 손 body 프레임 규약이
          자산마다 달라 컵이 손에서 17 cm 떠 버린다(09.02 실측). 컵은 기록의 자리에
          놓여 있고 미러는 기록의 파지 자세를 그대로 지나가므로, 그 프레임에서의
          실제 상대자세가 곧 in-hand 자세다.
        """
        rp, rq = subtract_frame_transforms(
            robot.data.body_pos_w[0:1, bi], robot.data.body_quat_w[0:1, bi],
            cup.data.root_pos_w[0:1], cup.data.root_quat_w[0:1])
        _held[name] = (cup, bi, rp.clone(), rq.clone())
        print(f"  [부착] {name} (그 프레임의 실제 상대자세)")

    def pin_held():
        for cup, bi, rp, rq in _held.values():
            pos, quat = combine_frame_transforms(
                robot.data.body_pos_w[0:1, bi], robot.data.body_quat_w[0:1, bi], rp, rq)
            cup.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
            cup.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))

    # ── 렌더 ───────────────────────────────────────────────────────────────
    shots = [0]

    def shot(tag: str):
        return

    if args.render is not None:
        import omni.replicator.core as rep  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        args.render.mkdir(parents=True, exist_ok=True)
        c = origin.cpu().numpy() + np.array([0.34, 0.0, 0.32])
        cam = rep.create.camera(position=tuple(c + np.array([1.20, -0.70, 0.55])),
                                look_at=tuple(float(v) for v in c))
        rp_ = rep.create.render_product(cam, (1280, 800))
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach([rp_])

        def shot(tag: str):  # noqa: F811
            env.sim.render()
            arr = np.asarray(annot.get_data())
            if arr.size:
                Image.fromarray(arr[:, :, :3]).save(args.render / f"{shots[0]:04d}_{tag}.png")
                shots[0] += 1

    def cups_z() -> str:
        s = float(env.cup.data.root_pos_w[0, 2] - origin[2])
        r = float(env.left_target_cup.data.root_pos_w[0, 2] - origin[2])
        return f"cup_big z {s:.3f} · shaker z {r:.3f}"

    def step_once(tag: str, frame: int):
        pin_held()
        robot.write_data_to_sim()
        env.sim.step(render=args.gui)
        robot.update(dt)
        if frame % args.render_every == 0:
            shot(tag)

    def hold(tag: str, n: int):
        for f in range(n):
            step_once(tag, f)
        print(f"[{tag}] {n}스텝 · {cups_z()}")

    def gate(msg: str):
        print(f"\n▶ {msg}")
        if not args.auto:
            input("  [Enter] …")

    def mirror(tag: str, frames_arm, frames_hand, ids_arm, ids_hand,
               attach_at: int, do_attach):
        f = 0
        for k in range(frames_arm.shape[0]):
            set_state(ids_arm, frames_arm[k])
            set_state(ids_hand, frames_hand[k])
            if k == attach_at:
                do_attach()
            for _ in range(args.repeat):
                step_once(tag, f)
                f += 1
        print(f"[{tag}] {f}스텝 · {cups_z()}")

    with torch.inference_mode():
        hold("0settle", args.settle)

        gate("① 좌팔(v2B25 실측 미러) — shaker 파지")
        mirror("1left", L_arm, L_grip, ids_la, ids_lg, L_attach,
               lambda: attach("shaker", env.left_target_cup, bi_l))

        gate("② 좌팔 유지")
        hold("2holdL", args.hold)

        gate("③ 우팔(E1) — cup_big 파지")
        mirror("3right", R_arm, R_hand, ids_ra, ids_rh, R_attach,
               lambda: attach("cup_big", env.cup, bi_r))

        gate("④ 양팔 유지 — 두 컵")
        hold("4final", args.final_hold)

    if args.render is not None:
        print(f"[렌더] {shots[0]}장 → {args.render}")
    if args.gui and not args.auto:
        print("\n창 유지 — Ctrl-C 로 닫는다")
        try:
            with torch.inference_mode():
                while simulation_app.is_running():
                    step_once("live", 1)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code or 0)

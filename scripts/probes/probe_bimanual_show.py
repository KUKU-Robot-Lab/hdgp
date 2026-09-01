#!/usr/bin/env python3
"""한 씬에서 양팔이 **순서대로** 각자 컵을 잡는 것을 보인다 (Enter 단계 전이).

씬은 pour env(`open-tesol_r_pour_sensor`)를 빌린다 — 양팔 + 붓는 컵(cup_big) +
받는 컵(shaker)이 이미 다 있다. env 는 씬으로만 쓰고 `env.step` 은 부르지 않는다.

각 단계의 팔 지령은 **그 정책이 자기 env 에서 폐루프로 만든 스트림**이다
(`probe_grasp_then_carry --save-stream`). 여기서는 그 지령을 재생하고 **컵은 물리가
잡는다** — 고정·텔레포트 없음. 폐루프 자체를 통합 씬에서 돌리는 것은 obs 빌더
(검증 완료)에 더해 fabric 액션 사슬이 필요해 다음 단계다.

순서:  정착 → [Enter] 좌팔 재생(shaker 파지) → [Enter] 유지
       → [Enter] 우팔 재생(cup_big 파지) → [Enter] 양팔 유지 (마무리 샷)

    ../IsaacLab/isaaclab.sh -p scripts/probes/probe_bimanual_show.py \\
        --left-stream <v2B25 npz> --right-stream <E1 npz> --gui        # Enter 로 전이
    ... --auto --render <dir> --headless                               # 영상 추출
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", default="open-tesol_r_pour_sensor-play-lstm")
parser.add_argument("--left-stream", type=Path, required=True,
                    help="좌팔 기록 npz (arm_target/grip_cmd/cup_pos, 8-env)")
parser.add_argument("--left-env", type=int, default=0)
parser.add_argument("--left-frames", type=int, default=250, help="좌팔 재생 프레임 수")
parser.add_argument("--right-stream", type=Path, required=True,
                    help="우팔 스트림 npz (arm_target/hand_target/cup_z)")
parser.add_argument("--right-cup-xy", default="0.362,-0.16",
                    help="붓는 컵을 놓을 xy (우팔 기록의 스폰 위치)")
parser.add_argument("--settle", type=int, default=90)
parser.add_argument("--hold", type=int, default=120, help="단계 사이 유지 스텝")
parser.add_argument("--final-hold", type=int, default=240)
parser.add_argument("--auto", action="store_true", help="Enter 없이 자동 전이 (영상 추출용)")
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
from isaaclab_tasks.utils.hydra import hydra_task_config          # noqa: E402

_ARM = {"l": [f"l_aj_{i}" for i in range(1, 8)], "r": [f"r_aj_{i}" for i in range(1, 8)]}
_GRIP_L = ["l_hj_gripper_1", "l_hj_gripper_2"]


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

    # ── 스트림 적재 ────────────────────────────────────────────────────────
    zl = np.load(args.left_stream, allow_pickle=True)
    e = args.left_env
    L_arm = zl["arm_target"][: args.left_frames, e]           # (T,7)
    L_grip = zl["grip_cmd"][: args.left_frames, e]            # (T,2)
    L_cup0 = zl["cup_pos"][0, e]                              # 기록 시작 시점의 shaker 위치
    zr = np.load(args.right_stream, allow_pickle=True)
    R_arm = zr["arm_target"]                                  # (T,7)
    R_hand = zr["hand_target"]                                # (T,20)
    R_hand_names = [str(x) for x in zr["meta_hand_names"]]
    R_cup_z0 = float(zr["cup_z"][0, 0])
    rx, ry = (float(v) for v in args.right_cup_xy.split(","))
    print(f"[스트림] 좌 {L_arm.shape[0]}프레임(env{e}) · 우 {R_arm.shape[0]}프레임")

    ids_la = [robot.joint_names.index(n) for n in _ARM["l"]]
    ids_lg = [robot.joint_names.index(n) for n in _GRIP_L]
    ids_ra = [robot.joint_names.index(n) for n in _ARM["r"]]
    ids_rh = [robot.joint_names.index(n) for n in R_hand_names]

    def T(a) -> torch.Tensor:
        return torch.tensor(np.asarray(a, dtype=np.float32)[None], device=env.device)

    # ── 초기 배치: 팔은 각 기록의 시작 지령으로, 컵은 각 기록의 시작 위치로 ──
    targets = {"la": T(L_arm[0]), "lg": T([0.044, 0.044]),
               "ra": T(R_arm[0]), "rh": T(R_hand[0])}
    for key, ids in (("la", ids_la), ("lg", ids_lg), ("ra", ids_ra), ("rh", ids_rh)):
        robot.write_joint_state_to_sim(targets[key], torch.zeros_like(targets[key]),
                                       joint_ids=ids)
        robot.set_joint_position_target(targets[key], joint_ids=ids)

    def place(obj, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        pose = torch.cat([origin + torch.tensor(pos, device=env.device),
                          torch.tensor(quat, device=env.device)]).unsqueeze(0)
        obj.write_root_pose_to_sim(pose)
        obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))

    place(env.cup, [rx, ry, R_cup_z0])                    # 붓는 컵 (우팔 기록의 스폰)
    place(env.left_target_cup, list(L_cup0))              # 받는 컵 (좌팔 기록의 스폰)
    if hasattr(env, "_hide_beads"):
        env._hide_beads(torch.arange(1, device=env.device))
        print("[설정] 비드 숨김")

    # ── 렌더 ───────────────────────────────────────────────────────────────
    shots = [0]

    def shot(tag: str) -> None:
        return

    if args.render is not None:
        import omni.replicator.core as rep  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        args.render.mkdir(parents=True, exist_ok=True)
        c = origin.cpu().numpy() + np.array([0.34, 0.0, 0.32])
        cam = rep.create.camera(position=tuple(c + np.array([1.20, -0.70, 0.55])),
                                look_at=tuple(float(v) for v in c))
        rp = rep.create.render_product(cam, (1280, 800))
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach([rp])

        def shot(tag: str) -> None:  # noqa: F811
            env.sim.render()
            arr = np.asarray(annot.get_data())
            if arr.size:
                Image.fromarray(arr[:, :, :3]).save(args.render / f"{shots[0]:04d}_{tag}.png")
                shots[0] += 1

    def cups_z() -> str:
        s = float(env.cup.data.root_pos_w[0, 2] - origin[2])
        r = float(env.left_target_cup.data.root_pos_w[0, 2] - origin[2])
        return f"cup_big z {s:.3f} · shaker z {r:.3f}"

    def step_once(tag: str, frame: int) -> None:
        for key, ids in (("la", ids_la), ("lg", ids_lg), ("ra", ids_ra), ("rh", ids_rh)):
            robot.set_joint_position_target(targets[key], joint_ids=ids)
        robot.write_data_to_sim()
        env.sim.step(render=args.gui)
        robot.update(dt)
        if frame % args.render_every == 0:
            shot(tag)

    def run(tag: str, n: int) -> None:
        for f in range(n):
            step_once(tag, f)
        print(f"[{tag}] {n}스텝 · {cups_z()}")

    def gate(msg: str) -> None:
        print(f"\n▶ {msg}")
        if not args.auto:
            input("  [Enter] 로 진행 …")

    with torch.inference_mode():
        run("0settle", args.settle)

        gate("① 좌팔 정책(v2B25) 재생 — shaker 파지")
        for f in range(L_arm.shape[0]):
            targets["la"], targets["lg"] = T(L_arm[f]), T(L_grip[f])
            step_once("1left", f)
        print(f"[좌팔] 끝 · {cups_z()}")

        gate("② 좌팔 유지 (파지 지령을 쥔 채)")
        run("2holdL", args.hold)

        gate("③ 우팔 정책(E1) 재생 — cup_big 파지")
        for f in range(R_arm.shape[0]):
            targets["ra"], targets["rh"] = T(R_arm[f]), T(R_hand[f])
            step_once("3right", f)
        print(f"[우팔] 끝 · {cups_z()}")

        gate("④ 양팔 유지 — 두 컵을 든 마무리")
        run("4final", args.final_hold)

    if args.render is not None:
        print(f"[렌더] {shots[0]}장 → {args.render}")
    if args.gui and not args.auto:
        print("\n창을 열어 둔다 — Ctrl-C 로 닫는다.")
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

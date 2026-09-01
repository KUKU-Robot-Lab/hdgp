#!/usr/bin/env python3
"""파지 종료 자세 → 붓기 시작 자세 **전환이 안전한가**를 sim 으로 판정한다.

실기에는 텔레포트가 없다. pour 는 sim 에서 warm 상태로 순간이동해 시작하지만, 실물은
파지가 끝난 자리에서 붓기 시작 자리까지 **실제로 지나가야** 한다. 그 사이에 몸통·
테이블·반대팔을 치는지 여기서 본다.

**판정은 힘의 크기가 아니라 "새로 닿았는가" 다.** 관통력은 물리량이 아니라서 solver
와 자세에 따라 마구 변한다. 시작 자세에서 이미 닿아 있던 것(파지한 컵, 손가락)을
기준선으로 빼고 **새로 닿은 몸통만** 센다.

**두 팔을 함께 본다.** 우팔만 보면 좌팔과 부딪는 것을 놓친다. `--order` 로 순서를
바꿔 시험할 수 있다 — 동시에 움직이는 것이 가장 위험한 경우다.

사용:
    ../IsaacLab/isaaclab.sh -p scripts/probes/probe_pour_entry_clear.py \\
        --start-json /path/start.json --goal-json /path/goal.json --headless

  json 형식 (없는 키는 건드리지 않는다):
      {"r_aj": [7개], "l_aj": [7개], "r_hj": [20개], "l_hj_gripper": [2개]}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", default="open-tesol_r_pour_sensor-play-lstm")
parser.add_argument("--start-json", type=Path, required=True)
parser.add_argument("--goal-json", type=Path, required=True)
parser.add_argument("--max-vel", type=float, default=0.5, help="rad/s — 램프 속도 상한")
parser.add_argument("--settle", type=int, default=60, help="시작 자세에서 기준선을 잴 스텝 수")
parser.add_argument("--contact-threshold", type=float, default=1.0, help="N")
parser.add_argument("--order", choices=("right_first", "left_first", "together"),
                    default="right_first",
                    help="두 팔을 움직이는 순서. together 가 가장 위험한 경우다.")
parser.add_argument("--out", type=Path, default=None, help="판정 json")
parser.add_argument("--gui", action="store_true")

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "source" / "openarm"))
sys.path.insert(0, str(_REPO / "scripts" / "tools"))
sys.path.insert(0, str(_REPO.parent / "sim2real" / "scripts"))   # transition_plan

from isaaclab.app import AppLauncher                              # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
args.headless = not args.gui
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym                                           # noqa: E402
import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402

import openarm  # noqa: E402,F401
from isaaclab.envs import DirectRLEnvCfg                          # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg      # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config          # noqa: E402

from transition_plan import (                                     # noqa: E402
    contact_set,
    describe_transition,
    new_contacts,
    ramp,
)

#: json 키 → 그 그룹의 관절 이름을 만드는 규칙.
_GROUPS = {
    "r_aj": [f"r_aj_{i}" for i in range(1, 8)],
    "l_aj": [f"l_aj_{i}" for i in range(1, 8)],
    "r_hj": [f"r_hj_{f}_{j}" for f in ("thumb", "index", "middle", "ring", "pinky")
             for j in range(1, 5)],
    "l_hj_gripper": ["l_hj_gripper_1", "l_hj_gripper_2"],
}


def _load_pose(path: Path) -> dict[str, list[float]]:
    raw = json.loads(path.read_text())
    out = {}
    for key, names in _GROUPS.items():
        if key not in raw:
            continue
        vals = [float(v) for v in raw[key]]
        if len(vals) != len(names):
            raise SystemExit(f"{path.name}: '{key}' 는 {len(names)}개여야 하는데 {len(vals)}개다")
        out[key] = vals
    if not out:
        raise SystemExit(f"{path.name}: 아는 관절 그룹이 하나도 없다 (가능한 키 {list(_GROUPS)})")
    return out


def _joint_ids(robot, names: list[str]) -> list[int]:
    missing = [n for n in names if n not in robot.joint_names]
    if missing:
        raise SystemExit(f"로봇에 없는 관절: {missing}")
    return [robot.joint_names.index(n) for n in names]


@hydra_task_config(args.task, "rl_games_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg, agent_cfg: dict):
    env_cfg.scene.num_envs = 1
    for attr in ("enable_adr", "enable_success_adr"):
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, False)

    env = gym.make(args.task, cfg=env_cfg)
    base = env.unwrapped
    robot = base.robot
    dt = float(base.step_dt)

    # 전신 접촉 센서를 env 생성 **뒤에** 붙인다 — 씬 cfg 에 넣으면 prim 을 못 찾는다.
    sensor = ContactSensor(ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=1, track_air_time=False))
    if not sensor.is_initialized:
        sensor._initialize_impl()
        sensor._is_initialized = True
    body_names = list(sensor.body_names)
    print(f"[설정] 접촉 감시 몸통 {len(body_names)}개 · step_dt {dt:.4f}s "
          f"· 임계 {args.contact_threshold} N · 램프 {args.max_vel} rad/s")

    start, goal = _load_pose(args.start_json), _load_pose(args.goal_json)
    env.reset()

    def force_norms() -> np.ndarray:
        sensor.update(dt)
        return sensor.data.net_forces_w[0].norm(dim=-1).cpu().numpy()

    def write_pose(pose: dict[str, list[float]]) -> None:
        for key, vals in pose.items():
            ids = _joint_ids(robot, _GROUPS[key])
            q = torch.tensor([vals], dtype=torch.float32, device=base.device)
            robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=ids)
            robot.set_joint_position_target(q, joint_ids=ids)

    def pin_left_cup() -> None:
        """★env 은 **매 스텝** 받는 컵을 고정 pose 로 다시 쓴다(`pour_right_env.py:1531`).

        이 프로브는 `env.step` 이 아니라 `sim.step` 을 직접 밟으므로 그 경로가 돌지
        않는다. 안 넣으면 컵이 흘러내려 엉뚱한 자리에서 그리퍼와 부딪히고, 그것을
        전환 충돌로 오독한다(09.01 실측: 컵과 11 cm 떨어진 곳에서 83.9 N).
        """
        cup = getattr(base, "left_target_cup", None)
        if cup is None:
            return
        pose = base._get_left_target_cup_fixed_pose()
        cup.write_root_pose_to_sim(pose)
        cup.write_root_velocity_to_sim(torch.zeros(base.num_envs, 6, device=base.device))

    def hold(steps: int) -> None:
        for _ in range(steps):
            pin_left_cup()
            robot.write_data_to_sim()
            base.sim.step(render=False)
            robot.update(dt)

    def body_z() -> dict[str, float]:
        z = (robot.data.body_pos_w[0, :, 2] - base.scene.env_origins[0, 2]).cpu().numpy()
        return {n: float(z[i]) for i, n in enumerate(robot.body_names)}

    # ── 기준선 ─────────────────────────────────────────────────────────────
    # ★시작 **과** 목표 양쪽에서 잰다. 목표에서 닿는 것은 구조다 — 받는 컵은 좌팔 rest
    #   자세의 FK 로 손 앞 5 cm 에 고정되므로, 좌팔이 목표에 도착하면 그리퍼가 **자기 컵**을
    #   만난다. 그것을 충돌로 세면 무엇을 해도 실패한다(09.01 실측: 142 N 오탐).
    write_pose(goal)
    hold(args.settle)
    at_goal = contact_set(body_names, force_norms(), threshold=args.contact_threshold)
    write_pose(start)
    hold(args.settle)
    at_start = contact_set(body_names, force_norms(), threshold=args.contact_threshold)
    baseline = at_start | at_goal
    baseline_z = body_z()
    print(f"[기준선] 시작에서 {len(at_start)}개 {sorted(at_start)}")
    print(f"[기준선] 목표에서 {len(at_goal)}개 {sorted(at_goal)}  → 합쳐 {len(baseline)}개를 뺀다")

    # ── 전환 ───────────────────────────────────────────────────────────────
    arm_keys = [k for k in ("r_aj", "l_aj") if k in start and k in goal]
    if args.order == "left_first":
        arm_keys = sorted(arm_keys, reverse=True)
    stages = [arm_keys] if args.order == "together" else [[k] for k in arm_keys]

    report, ok = {}, True
    for stage in stages:
        paths = {k: ramp(start[k], goal[k], max_vel=args.max_vel, dt=dt) for k in stage}
        n = max(p.shape[0] for p in paths.values())
        worst: dict[str, tuple[int, float]] = {}
        where: dict[str, list[float]] = {}
        min_z: dict[str, float] = {name: 1e9 for name in body_names}
        for f in range(n):
            for key, path in paths.items():
                row = path[min(f, path.shape[0] - 1)]
                ids = _joint_ids(robot, _GROUPS[key])
                robot.set_joint_position_target(
                    torch.tensor([row], dtype=torch.float32, device=base.device), joint_ids=ids)
            pin_left_cup()
            robot.write_data_to_sim()
            base.sim.step(render=False)
            robot.update(dt)
            forces = force_norms()
            for name in new_contacts(baseline,
                                     contact_set(body_names, forces, threshold=args.contact_threshold)):
                mag = float(forces[body_names.index(name)])
                if name not in worst or mag > worst[name][1]:
                    worst[name] = (f, mag)
                    # ★어디서 닿았는지 없이 "닿았다"만 알면 고칠 수가 없다.
                    where[name] = [round(float(v), 4) for v in
                                   (robot.data.body_pos_w[0, robot.body_names.index(name)]
                                    - base.scene.env_origins[0]).cpu().numpy()]
            for name, z in body_z().items():
                if name in min_z:
                    min_z[name] = min(min_z[name], z)
        label = "+".join(stage)
        table_z = float(getattr(base.cfg, "table_surface_z", 0.2))
        print("\n" + describe_transition(label, next(iter(paths.values())), dt=dt,
                                         worst=worst, min_z=min_z, table_z=table_z,
                                         baseline_z=baseline_z))
        report[label] = {"frames": int(n), "seconds": round((n - 1) * dt, 2),
                         "new_contacts": {k: {"frame": v[0], "N": round(v[1], 2),
                                              "pos_env_local": where.get(k)}
                                          for k, v in worst.items()},
                         "min_z": {k: round(v, 4) for k, v in sorted(min_z.items(),
                                                                    key=lambda kv: kv[1])[:5]}}
        ok = ok and not worst
        # 다음 구간은 이 구간이 끝난 자세에서 출발한다.
        for key, path in paths.items():
            start[key] = list(path[-1])

    from openarm.tesollo.right.pour_sensor.pour_right_preset import (  # noqa: PLC0415
        LEFT_TARGET_CUP_POS_ENV_LOCAL)
    report["reference"] = {
        "left_target_cup_pos_env_local": [round(float(v), 4) for v in LEFT_TARGET_CUP_POS_ENV_LOCAL],
        "table_surface_z": float(getattr(base.cfg, "table_surface_z", 0.2)),
    }
    report["verdict"] = "pass" if ok else "fail"
    report["order"] = args.order
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\n[저장] {args.out}")
    print(f"\n[판정] {'✅ 통과' if ok else '❌ 실패'} (순서 {args.order})")
    return 0 if ok else 1


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code or 0)

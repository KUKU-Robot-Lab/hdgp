"""좌 그리퍼 Fabrics 정책을 sim 에서 굴리며 **실기로 내보낼 것과 비교할 것**을 전부 남긴다.

이 파일이 하는 일은 기록뿐이다. env 도 액션 항도 건드리지 않고, 액션 항이 이미 갖고 있는
텐서를 읽기만 한다. 정책을 바꾸면 그림자 비교의 기준이 사라지기 때문이다.

무엇을 남기고 왜 남기는가 — "Fabrics IK 가 도는가"는 세 층으로 갈린다:

    L1  FK(fabric_q) vs 지령 palm pose   attractor 가 목표에 수렴하나   (여기서 나온다)
    L2  sim 물리 TCP  vs FK(fabric_q)     sim PD 가 fabric 해를 따라가나 (여기서 나온다)
    L3  실기 measured vs arm_target       실팔이 그 관절 목표를 따라가나 (재생 뒤 나온다)

★`fabric_q` 와 **중력 처짐 보상분(droop)** 을 따로 남긴다. 액션 항은 관절공간에서
  `target = fabric_q + droop` 을 지령하는데, droop 의 상한이 `effort/강성` 이고 그 강성이
  **sim 의 400** 이다. 실기 펌웨어는 70/60/10 이라 같은 보정량이 전혀 다른 뜻이 된다.
  합쳐서 남기면 실기 쪽에서 그 둘을 다시 가를 방법이 없다.

★리셋 오염 차단: `episode_length_s` 를 크게 잡는다. 이 태스크는 그 함정에 세 번 당했다
  (`probe_fab_action_mapping.py` docstring).

실행 (학습 시점 FABRICS 를 PYTHONPATH 로 지정하는 것이 중요하다 — 08.25 실측으로 트리가
다르면 IK 해가 최대 0.32 rad 갈리는 것을 확인했다):

    PYTHONPATH=<학습시점>/source/FABRICS/src \\
    ../IsaacLab/isaaclab.sh -p scripts/probes/probe_fab_shadow_record.py \\
        --checkpoint log/rl_games/open-grip/left/grasp-sensor-fab/fab_test16/nn/open-grip_l_grasp_sensor_fab.pth \\
        --steps 1500 --out logs/shadow/sim_fab_test16.npz
"""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="open-grip_l_grasp_sensor_fab-play")
parser.add_argument("--checkpoint", required=True, type=Path, help="rl_games .pth")
parser.add_argument("--steps", type=int, default=1500, help="기록할 env 스텝 수")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--gravity_comp", choices=["on", "off"], default="on",
                    help="액션 항의 처짐 보상. off 는 HDGP_GRAVITY_COMP=0 과 같다.")
parser.add_argument("--cup_pose", type=Path, default=None,
                    help="perception 이 준 컵 pose(json). 주면 리셋 직후 컵을 그 자리로 "
                         "옮긴다. env 는 건드리지 않는다 — 씬 객체에 직접 쓴다. "
                         "`sim2real/scripts/cup_pose_capture.py` 가 만든다.")
parser.add_argument("--keep_drop_termination", action="store_true",
                    help="컵 전도 종료를 살려 둔다. 기본은 끈다 — 리셋마다 팔이 홈으로 "
                         "텔레포트해 **연속 궤적이 아니게** 되고, 그림자 재생에서 그건 "
                         "이동이 아니라 도약이다. 실기에는 컵이 없으므로 컵 상태는 "
                         "이 측정의 대상이 아니다.")
parser.add_argument("--fabrics_src", type=Path, default=None,
                    help="쓸 fabrics_sim 소스 트리(.../source/FABRICS/src). "
                         "★PYTHONPATH 로는 안 된다 — `openarm.tasks` 가 저장소 사본을 "
                         "sys.path[0] 에 꽂아 덮어쓴다. 이 인자는 그보다 먼저 import 한다.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if args.gravity_comp == "off":
    os.environ["HDGP_GRAVITY_COMP"] = "0"     # ★preset import 전에 설정해야 한다

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym            # noqa: E402
import numpy as np                 # noqa: E402
import torch                       # noqa: E402

_HDGP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HDGP / "source/openarm"))
sys.path.insert(0, str(_HDGP / "scripts/tools"))

# ★fabrics_sim 은 `openarm.tasks` **보다 먼저** 확정한다. 그 모듈이 저장소 사본을
#   sys.path[0] 에 꽂기 때문에, 나중에 import 하면 어떤 PYTHONPATH 를 줘도 저장소 것이
#   이긴다. 08.25 실측: 두 트리는 같은 목표에서 관절 해가 최대 0.32 rad 갈린다 —
#   조용히 다른 IK 로 기록하면 그림자 비교가 통째로 무의미해진다.
if args.fabrics_src is not None:
    resolved_src = args.fabrics_src.resolve()
    if not (resolved_src / "fabrics_sim").is_dir():
        raise SystemExit(f"--fabrics_src 아래에 fabrics_sim 이 없다: {resolved_src}")
    sys.path.insert(0, str(resolved_src))
import fabrics_sim                                               # noqa: E402
_FABRICS_FILE = Path(fabrics_sim.__file__).resolve()
if args.fabrics_src is not None and not str(_FABRICS_FILE).startswith(str(resolved_src)):
    raise SystemExit(
        f"fabrics_sim 이 요청한 트리에서 오지 않았다:\n  요청 {resolved_src}\n"
        f"  실제 {_FABRICS_FILE}"
    )

import openarm.tasks                                             # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg                   # noqa: E402
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz   # noqa: E402
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P   # noqa: E402
from run_cfg_restore import restore_run_cfg_if_available         # noqa: E402


def build_policy(checkpoint: Path, agent_cfg: dict, env):
    """rl_games player 를 만들고 체크포인트를 얹는다."""
    from rl_games.torch_runner import Runner
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

    device = agent_cfg["params"]["config"]["device"]
    wrapped = RlGamesVecEnvWrapper(env, device, np.inf, np.inf, None, True)
    vecenv_name = "IsaacRlgWrapper"
    from rl_games.common import env_configurations, vecenv
    vecenv.register(vecenv_name, lambda name, num, **kw: RlGamesGpuEnv(name, num, **kw))
    env_configurations.register(vecenv_name, {
        "vecenv_type": vecenv_name, "env_creator": lambda **kw: wrapped})
    agent_cfg["params"]["config"]["env_name"] = vecenv_name
    agent_cfg["params"]["config"]["env_info"] = wrapped.get_env_info()

    runner = Runner()
    runner.load(agent_cfg)
    player = runner.create_player()
    player.restore(str(checkpoint))
    player.reset()
    if hasattr(player, "has_batch_dimension"):
        player.has_batch_dimension = True
    return player, wrapped


def main() -> int:
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)

    import yaml
    run_dir = args.checkpoint.parent.parent
    agent_cfg = yaml.safe_load((run_dir / "params/agent.yaml").read_text())
    agent_cfg = restore_run_cfg_if_available(
        env_cfg, agent_cfg, resume_path=str(args.checkpoint),
        workspace_root=str(_HDGP.parent),
    )
    agent_cfg["params"]["config"]["device"] = args.device
    agent_cfg["params"]["config"]["device_name"] = args.device
    # ★★복원 **뒤에** 다시 강제한다. `params/env.yaml` 은 학습의 num_envs(1024)를 담고
    #   있어서 `parse_env_cfg(num_envs=1)` 을 조용히 되돌린다. 순서를 바꾸면 1024 env 를
    #   돌리고도 눈치채지 못한다(367 MB npz 로 알아챘다).
    env_cfg.scene.num_envs = args.num_envs
    # ★리셋 오염 차단. 에피소드가 중간에 끊기면 지령·처짐 적분이 초기화돼 시계열이 갈린다.
    env_cfg.episode_length_s = 1e6
    if not args.keep_drop_termination and hasattr(env_cfg.terminations, "object_dropping"):
        env_cfg.terminations.object_dropping = None
        print("[REC] 컵 전도 종료를 껐다 — 팔 궤적을 끊기지 않게 한다", flush=True)

    env = gym.make(args.task, cfg=env_cfg).unwrapped
    player, wrapped = build_policy(args.checkpoint, agent_cfg, env)

    cup_pose = None
    if args.cup_pose is not None:
        sys.path.insert(0, str(_HDGP.parent / "sim2real/scripts"))
        from cup_pose_capture import load_capture, spawn_box_from_preset, verdict

        cup_pose = load_capture(args.cup_pose, expect_frame="base_link")
        report = verdict(cup_pose, spawn_box_from_preset(P))
        print("[CUP] " + report.describe().replace("\n", "\n[CUP] "), flush=True)

    robot = env.scene["robot"]
    # ★fabric 의 `palm_link` 은 그리퍼 **TCP** 인데(fabric URDF 주석) USD 에서는 그 프레임이
    #   강체로 병합돼 사라진다. base 바디를 그대로 쓰면 두 점이 구조적으로 80 mm 떨어져
    #   있어 L2 가 "추종오차 99 mm" 로 읽힌다 — 실제로는 오프셋이다. 회전시켜 더한다.
    tcp_offset = torch.tensor([0.0, 0.0, P.TCP_OFFSET_IN_BASE_Z], device=env.device)
    term = env.action_manager.get_term("arm_action")
    arm_ids = term._arm_joint_ids
    grip_ids = [robot.joint_names.index(n) for n in P.GRIPPER_JOINT_NAMES]
    base_body = robot.body_names.index(P.GRIPPER_BASE_BODY)
    fabric = term._fabric

    # ★액션 항의 palm 지령 표현은 트랙이 살아 있는 동안 바뀐다(08.25 실측: 7D xyz+quat(xyzw)
    #   → 6D xyz+euler_zyx). 어느 쪽이든 **위치 + wxyz 쿼터니언**으로 정규화해 남긴다.
    #   둘 다 없으면 조용히 넘어가지 않고 무엇이 있는지 적어 죽는다 — 기록이 반쯤 비면
    #   그림자 비교는 실패가 아니라 **틀린 결론**으로 나타난다.
    def palm_command():
        if hasattr(term, "_palm_target_xyz_q"):          # 구 규약: xyz + xyzw
            raw = term._palm_target_xyz_q
            xyzw = raw[:, 3:7]
            return raw[:, :3], torch.cat([xyzw[:, 3:4], xyzw[:, 0:3]], dim=-1)
        if hasattr(term, "_palm_pose_target"):           # 신 규약: xyz + euler_zyx
            raw = term._palm_pose_target
            return raw[:, :3], quat_from_euler_xyz(raw[:, 5], raw[:, 4], raw[:, 3])
        raise AttributeError(
            "액션 항에서 palm 지령을 못 찾았다. 가진 것: "
            + ", ".join(sorted(a for a in vars(term) if "palm" in a or "target" in a))
        )

    palm_command()   # 기록 전에 한 번 불러 API 를 확정한다
    print(f"[REC] task={args.task} ckpt={args.checkpoint.name}", flush=True)
    print(f"[REC] gravity_comp={args.gravity_comp} (P.GRAVITY_COMP_ENABLED={P.GRAVITY_COMP_ENABLED})",
          flush=True)
    print(f"[REC] fabrics_sim -> {_FABRICS_FILE}", flush=True)
    print(f"[REC] 팔 관절 {[robot.joint_names[i] for i in arm_ids]}", flush=True)
    print(f"[REC] 그리퍼 {[robot.joint_names[i] for i in grip_ids]}", flush=True)

    rec: dict[str, list] = {k: [] for k in (
        "action", "palm_cmd_pos", "palm_cmd_quat_wxyz", "fabric_q", "droop", "arm_target",
        "arm_meas", "arm_vel", "palm_fk_pos", "palm_fk_quat_wxyz", "tcp_pos", "tcp_quat_wxyz",
        "grip_cmd", "grip_meas", "gripper_gate", "cup_pos", "cmd_step_norm", "reward", "done")}

    def policy_obs(raw):
        """rl_games 래퍼는 obs 를 dict 로 준다 — actor 가 먹는 텐서만 꺼낸다."""
        return raw["obs"] if isinstance(raw, dict) else raw

    def place_cup() -> None:
        """리셋 뒤 컵을 인지가 준 자리로 옮긴다.

        env 를 고치지 않는 이유 두 가지: ①이 태스크 파일은 자매 세션이 지금 고치고 있다
        ②스폰은 이벤트가 무작위로 하는데, 그걸 바꾸면 학습 경로를 건드리게 된다.
        씬 객체에 직접 쓰면 둘 다 피하면서 같은 결과를 얻는다.
        """
        if cup_pose is None:
            return
        cup = env.scene["object"]
        pose = torch.tensor(
            [*cup_pose.position, *cup_pose.orientation_wxyz],
            device=env.device, dtype=torch.float32).unsqueeze(0).repeat(env.num_envs, 1)
        pose[:, :3] += env.scene.env_origins          # 씬 좌표는 world 다
        cup.write_root_pose_to_sim(pose)
        cup.write_root_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device))

    obs = policy_obs(wrapped.reset())
    place_cup()
    with torch.inference_mode():
        for step in range(args.steps):
            action = player.get_action(obs, is_deterministic=True)
            raw_obs, reward, dones, _ = wrapped.step(action)
            if cup_pose is not None and bool(dones.any()):
                # 리셋이 나면 이벤트가 컵을 다시 무작위로 놓는다 — 되돌린다.
                place_cup()
            obs = policy_obs(raw_obs)

            palm = fabric.get_palm_pose(term._fabric_q, "quaternion")     # xyz + xyzw
            body_quat = robot.data.body_quat_w[:, base_body]              # wxyz
            body_pos = (robot.data.body_pos_w[:, base_body] - env.scene.env_origins
                        + quat_apply(body_quat, tcp_offset.expand(env.num_envs, 3)))
            gate = env.obs_buf["policy"][:, -1] if isinstance(env.obs_buf, dict) else None

            rec["action"].append(action.detach().cpu().numpy().copy())
            cmd_pos, cmd_quat = palm_command()
            rec["palm_cmd_pos"].append(cmd_pos.detach().cpu().numpy().copy())
            rec["palm_cmd_quat_wxyz"].append(cmd_quat.detach().cpu().numpy().copy())
            rec["fabric_q"].append(term._fabric_q.detach().cpu().numpy().copy())
            rec["droop"].append(term._droop.detach().cpu().numpy().copy())
            rec["arm_target"].append(
                (term._fabric_q + term._droop).detach().cpu().numpy().copy()
                if P.GRAVITY_COMP_ENABLED else term._fabric_q.detach().cpu().numpy().copy())
            rec["arm_meas"].append(robot.data.joint_pos[:, arm_ids].detach().cpu().numpy().copy())
            rec["arm_vel"].append(robot.data.joint_vel[:, arm_ids].detach().cpu().numpy().copy())
            fk_xyzw = palm[:, 3:7]
            rec["palm_fk_pos"].append(palm[:, :3].detach().cpu().numpy().copy())
            rec["palm_fk_quat_wxyz"].append(
                torch.cat([fk_xyzw[:, 3:4], fk_xyzw[:, 0:3]], dim=-1).detach().cpu().numpy().copy())
            rec["tcp_pos"].append(body_pos.detach().cpu().numpy().copy())
            rec["tcp_quat_wxyz"].append(body_quat.detach().cpu().numpy().copy())
            rec["grip_cmd"].append(
                robot.data.joint_pos_target[:, grip_ids].detach().cpu().numpy().copy())
            rec["grip_meas"].append(robot.data.joint_pos[:, grip_ids].detach().cpu().numpy().copy())
            rec["gripper_gate"].append(
                gate.detach().cpu().numpy().copy() if gate is not None
                else np.full(env.num_envs, np.nan))
            rec["cup_pos"].append(
                (env.scene["object"].data.root_pos_w - env.scene.env_origins)
                .detach().cpu().numpy().copy())
            rec["cmd_step_norm"].append(term.cmd_step_norm.detach().cpu().numpy().copy())
            rec["reward"].append(reward.detach().cpu().numpy().copy())
            rec["done"].append(dones.detach().cpu().numpy().copy())

            if step % 200 == 0:
                err = float(np.linalg.norm(rec["palm_fk_pos"][-1][0] - rec["palm_cmd_pos"][-1][0]))
                print(f"[REC] {step:5d}/{args.steps}  L1 {err*1000:6.1f} mm  "
                      f"done {int(rec['done'][-1][0])}", flush=True)

    arrays = {k: np.stack(v) for k, v in rec.items()}
    arrays["meta_joint_names"] = np.array([robot.joint_names[i] for i in arm_ids])
    arrays["meta_grip_names"] = np.array([robot.joint_names[i] for i in grip_ids])
    arrays["meta_step_dt"] = np.array([env.step_dt])
    arrays["meta_gravity_comp"] = np.array([args.gravity_comp])
    arrays["meta_checkpoint"] = np.array([str(args.checkpoint)])
    arrays["meta_fabrics"] = np.array([str(_FABRICS_FILE)])
    # ★소스 신원. 이 트랙은 살아 있는 동안 태스크 파일이 바뀐다(08.25 실측: 기록 도중
    #   다른 세션이 preset·액션 항을 갈아 recorder 가 죽었다). 어떤 코드로 잰 기록인지
    #   파일 자신이 답할 수 있어야 사후에 해석이 가능하다.
    import hashlib
    task_dir = _HDGP / "source/openarm/openarm/gripper/left/grasp_sensor"
    digests = [f"{f.name}:{hashlib.sha256(f.read_bytes()).hexdigest()[:12]}"
               for f in sorted(task_dir.glob("*.py"))]
    arrays["meta_task_sha256"] = np.array(digests)
    arrays["meta_cup_pose_source"] = np.array(
        [cup_pose.source if cup_pose is not None else "env 무작위 스폰"])
    # ★fabric 을 **다른 인터프리터에서 다시 풀어 보려면** 이 다섯이 있어야 한다. 지금은
    #   전부 preset 상수라 소스만 바뀌면 조용히 달라진다(08.25 에 damping 20→10,
    #   fabric_dt step_dt/2→step_dt 로 바뀌었다). 기록이 스스로 답하게 한다.
    arrays["meta_fabric_dt"] = np.array([float(term._fabric_dt)])
    arrays["meta_fabric_decimation"] = np.array([int(P.FABRIC_DECIMATION)])
    arrays["meta_fabric_damping"] = np.array([float(term._damping[0, 0].item())])
    arrays["meta_fabric_vel_ff"] = np.array(
        [float(getattr(term, "_vel_ff_scale", float("nan")))])
    arrays["meta_home_q"] = np.array(term._q_home.detach().cpu().numpy())
    arrays["meta_fabric_robot_dir"] = np.array([str(P.FABRIC_ROBOT_DIR)])
    arrays["meta_fabric_world"] = np.array([str(P.FABRIC_WORLD_FILENAME)])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)

    l1 = np.linalg.norm(arrays["palm_fk_pos"] - arrays["palm_cmd_pos"], axis=-1)[:, 0] * 1000
    l2 = np.linalg.norm(arrays["tcp_pos"] - arrays["palm_fk_pos"], axis=-1)[:, 0] * 1000
    trk = np.abs(arrays["arm_meas"] - arrays["arm_target"])[:, 0].max(axis=-1)
    print(f"\n[REC] -> {args.out}  ({arrays['action'].shape[0]} 스텝)")
    print(f"[REC] L1 FK vs 지령      mean {l1.mean():7.2f}  p95 {np.percentile(l1,95):7.2f}  max {l1.max():7.2f}  mm")
    print(f"[REC] L2 물리TCP vs FK   mean {l2.mean():7.2f}  p95 {np.percentile(l2,95):7.2f}  max {l2.max():7.2f}  mm")
    print(f"[REC] sim 관절 추종오차  mean {trk.mean():7.4f}  max {trk.max():7.4f}  rad")
    print(f"[REC] 에피소드 종료 {int(arrays['done'].sum())}회")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
    raise SystemExit(code)

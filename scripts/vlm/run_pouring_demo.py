"""VLM pouring 파이프라인을 fabric 태스크 sim 에서 눈으로 확인하는 데모.

Qwen 은 스텁(고정 TaskSpecification)이다 — GPU 경계(모델 로딩) 밖에서
파이프라인 → 스킬 라우팅 → fabric 제어 배선을 검증한다.

동작: 규칙 기반 approach 스킬이 `TASK_SPACE_POSE`(물체 위 pregrasp 지점)를 내고,
`vlm.pouring.fabric_bridge` 가 이를 fabric 태스크의 절대 액션으로 인코딩해
Fabrics 가 팔을 보낸다. `--policy_run`(또는 `--policy_checkpoint`)을 주면
같은 태스크로 학습된 rl_games 체크포인트가 `grasp_lift` 스킬로 접속되어
approach 성공 후 정책이 파지·리프트를 이어받는다(플랜: approach → grasp_lift).
정책 obs 는 env 자신의 관측 벡터를 그대로 쓴다(학습과 동일 계약).

실행 (GUI 로 보려면 --headless 빼기):
    # approach 만 (정책 없음)
    PYTHONUNBUFFERED=1 ../IsaacLab/isaaclab.sh -p scripts/vlm/run_pouring_demo.py \
        --num_envs 2 --steps 600 --out outputs/vlm_demo --headless
    # 학습 정책 접속 (예: arm4090 open-bis/left test2)
    PYTHONUNBUFFERED=1 ../IsaacLab/isaaclab.sh -p scripts/vlm/run_pouring_demo.py \
        --task open-bis_l_grasp_lift_fab-play --policy_run test2 \
        --num_envs 2 --steps 720 --out outputs/vlm_demo --headless

카메라 프레임은 --out 아래 PNG 로 저장된다 — 이후 Qwen task-grounding 의
입력 이미지가 되는 바로 그 시점이다.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

_HDGP_ROOT = Path(__file__).resolve().parents[2]
_VLM_SRC = _HDGP_ROOT / "source/vlm"
if str(_VLM_SRC) not in sys.path:
    sys.path.insert(0, str(_VLM_SRC))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-bis_r_grasp_lift_fab-play")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=600, help="정책 스텝 수 (60Hz)")
parser.add_argument("--tick_interval", type=int, default=12,
                    help="HRL tick 주기(정책 스텝) — 12 = 5Hz")
parser.add_argument("--out", type=str, default="outputs/vlm_demo", help="프레임/로그 출력 디렉토리")
parser.add_argument("--frame_interval", type=int, default=60, help="카메라 저장 주기(스텝), 0 = 끔")
parser.add_argument("--approach_offset", type=float, nargs=3, default=(0.0, 0.0, 0.12),
                    help="물체 기준 pregrasp 오프셋 [m] (env-local)")
parser.add_argument("--approach_tol", type=float, default=0.03, help="approach 성공 반경 [m]")
parser.add_argument("--policy_run", type=str, default=None,
                    help="grasp_lift 로 접속할 rl_games 런 디렉토리 이름(글롭). "
                         "task id 에서 log/rl_games/<robot>/<side>/<task>/ 를 파생한다.")
parser.add_argument("--policy_checkpoint", type=str, default=None,
                    help="명시적 .pth 경로 (없으면 policy_run 의 최신 last_*.pth)")
parser.add_argument("--lift_height", type=float, default=0.08,
                    help="grasp_lift 성공 = 안착 기준선 대비 이만큼 상승 [m]")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = args.frame_interval > 0 or args.enable_cameras

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import openarm.tasks  # noqa: F401,E402  (gym id 등록)
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from vlm.pouring import (  # noqa: E402
    DeterministicHighLevelPolicy,
    PouringPipeline,
    SkillId,
    TaskSpecification,
)
from vlm.pouring.checkpoint_resolver import CheckpointResolver  # noqa: E402
from vlm.pouring.contracts import HOLD_MODES, ControlMode  # noqa: E402
from vlm.pouring.fabric_bridge import (  # noqa: E402
    PalmActionSpace,
    arm_channel_to_action,
    euler_zyx_to_quat_wxyz,
    hand_channel_to_action,
)
from vlm.pouring.isaac_state import FabricStateProvider  # noqa: E402
from vlm.pouring.rl_games_backend import RlGamesPolicyBackend  # noqa: E402
from vlm.pouring.skill_manager import SkillManager  # noqa: E402
from vlm.pouring.skill_registry import SkillRegistry  # noqa: E402
from vlm.pouring.skills import ApproachSkill, GraspLiftSkill  # noqa: E402

# hand_control 스위치가 생기기 전 학습된 런의 손 배선(정책 액션 = 관절 목표 → 직접 PD).
_LEGACY_HAND_CONTROL = "pd"

# 카메라: env_0 에만 붙인다(작업면 조망 — 이후 Qwen 입력 시점).
_CAM_EYE = (1.35, 0.55, 0.85)
_CAM_LOOKAT = (0.30, -0.10, 0.30)


class _EnvSceneView:
    """grasp_lift_fabric env 버퍼 → FabricSceneView (env-local, CPU 리스트)."""

    def __init__(self, env) -> None:
        self.env = env
        if len(env.hand_ids) != 20:
            raise RuntimeError(
                f"SemanticState 는 손 20관절을 요구한다 — profile={env.profile.name} "
                f"hand={len(env.hand_ids)}. 이 프로필은 아직 데모 미지원."
            )

    @property
    def num_envs(self) -> int:
        return self.env.num_envs

    def palm_pose_zyx(self):
        return self.env._palm_pose_6d().cpu().tolist()

    def object_pose(self):
        data = self.env.object.data
        pos = data.root_pos_w - self.env.scene.env_origins
        return torch.cat([pos, data.root_quat_w], dim=1).cpu().tolist()

    def object_velocity(self):
        return self.env.object.data.root_vel_w.cpu().tolist()

    def arm_joint_pos(self):
        return self.env.robot.data.joint_pos[:, self.env.arm_ids].cpu().tolist()

    def arm_joint_vel(self):
        return self.env.robot.data.joint_vel[:, self.env.arm_ids].cpu().tolist()

    def hand_joint_pos(self):
        return self.env.robot.data.joint_pos[:, self.env.hand_ids].cpu().tolist()

    def hand_joint_vel(self):
        return self.env.robot.data.joint_vel[:, self.env.hand_ids].cpu().tolist()


def _task_identity(task: str) -> tuple[str, str, str]:
    """gym id → (base_id, '<robot>/<side>', '<task-folder>') — train.py 로그 규약."""
    base = re.sub(r"(-play|-lstm)+$", "", task)
    match = re.match(r"^(open-[a-z0-9]+)_([lr])_(.+)$", base)
    if match is None:
        raise SystemExit(f"[vlm_demo] task id 규약 해석 실패: {task}")
    robot, side, task_part = match.groups()
    side_dir = "left" if side == "l" else "right"
    return base, f"{robot}/{side_dir}", task_part.replace("_", "-")


def _match_training_setup(cfg, env_yaml: Path) -> None:
    """학습 덤프의 **동역학·제어 경로** 플래그를 데모 env 에 반영한다.

    체크포인트는 자기가 학습된 env 에서만 의미가 있다. cfg 기본값을 그대로 쓰면
    정책이 본 적 없는 동역학·제어 배선에서 평가된다 — 실측 2건:
      · PLAY 기본은 gravity/self_collisions OFF 인데 test2 는 둘 다 ON 이었다.
      · 현재 기본 `hand_control="fabric"`(손을 Fabrics 가 소유) 인데 test2 는 그
        스위치가 생기기 **전**에 학습돼 손이 직접 PD("pd")였다. 그대로 돌리면
        같은 액션이 다른 손 자세가 되어 파지가 무너진다(리프트 31.5mm → 5.9mm).
    ★키가 **없으면** 그 스위치 이전 런이라는 뜻이라 구 배선 값을 쓴다.
    """
    text = env_yaml.read_text()
    for key in ("enable_gravity", "enable_self_collisions"):
        match = re.search(rf"^{key}: (true|false)$", text, re.M)
        if match is not None:
            setattr(cfg, key, match.group(1) == "true")
            print(f"[vlm_demo] 학습 정합: {key} = {match.group(1)}", flush=True)

    if hasattr(cfg, "hand_control"):
        match = re.search(r"^hand_control: (\w+)$", text, re.M)
        value = match.group(1) if match is not None else _LEGACY_HAND_CONTROL
        origin = "덤프" if match is not None else "덤프에 없음 → 구 배선"
        cfg.hand_control = value
        print(f"[vlm_demo] 학습 정합: hand_control = {value} ({origin})", flush=True)
    if hasattr(cfg, "use_tip_fabric"):
        match = re.search(r"^use_tip_fabric: (true|false)$", text, re.M)
        cfg.use_tip_fabric = match is not None and match.group(1) == "true"


def _resolve_policy(task: str):
    """--policy_run/--policy_checkpoint → PolicyArtifacts (없으면 None)."""
    if not (args.policy_run or args.policy_checkpoint):
        return None
    base_id, side_dir, folder = _task_identity(task)
    resolver = CheckpointResolver(_HDGP_ROOT, task_logs={base_id: (side_dir, folder)})
    checkpoint = Path(args.policy_checkpoint) if args.policy_checkpoint else None
    if checkpoint is None:
        run_root = _HDGP_ROOT / "log/rl_games" / side_dir / folder
        runs = sorted(path for path in run_root.glob(args.policy_run) if path.is_dir())
        if len(runs) != 1:
            raise SystemExit(f"[vlm_demo] 런 선택자가 정확히 1개여야 한다: {run_root}/{args.policy_run} → {runs}")
        ckpts = sorted((runs[0] / "nn").glob("*.pth"), key=lambda p: p.stat().st_mtime)
        if not ckpts:
            raise SystemExit(f"[vlm_demo] 체크포인트 없음: {runs[0]}/nn")
        checkpoint = ckpts[-1]
    return resolver.resolve(base_id, args.policy_run or "ignored", checkpoint=checkpoint)


def main() -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = _resolve_policy(args.task)
    _, side_dir, _ = _task_identity(args.task)
    controlled_side = side_dir.rsplit("/", 1)[-1]

    cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    if artifacts is not None:
        _match_training_setup(cfg, artifacts.env_yaml)
    # 데모 중 에피소드 timeout 리셋이 끼면 SkillManager 의 per-env 상태와 어긋난다.
    cfg.episode_length_s = max(cfg.episode_length_s, args.steps / 60.0 + 5.0)
    if args.frame_interval > 0:
        cfg.scene.vlm_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/vlm_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 20.0)),
        )

    env = gym.make(args.task, cfg=cfg).unwrapped
    env.reset()
    # ★reset 직후 버퍼는 stale 일 수 있다(반복 실측 함정) — 1스텝 굴린 뒤 읽는다.
    zero = torch.zeros(env.num_envs, env.cfg.action_space, device=env.device)
    latest_obs, *_ = env.step(zero)

    cam = env.scene.sensors.get("vlm_cam") if args.frame_interval > 0 else None
    if cam is not None:
        origin0 = env.scene.env_origins[0].cpu().tolist()
        eye = [[_CAM_EYE[i] + origin0[i] for i in range(3)]] * env.num_envs
        look = [[_CAM_LOOKAT[i] + origin0[i] for i in range(3)]] * env.num_envs
        cam.set_world_poses_from_view(
            torch.tensor(eye, device=env.device), torch.tensor(look, device=env.device)
        )

    # ---- vlm 파이프라인 조립 (Qwen 스텁: 고정 task) ------------------------------
    space = PalmActionSpace(
        home=tuple(env.home_palm[0].cpu().tolist()),
        low=tuple(env.palm_lo[0].cpu().tolist()),
        high=tuple(env.palm_hi[0].cpu().tolist()),
    )
    hand_dim = int(env.cfg.action_space) - 6
    approach_quat = euler_zyx_to_quat_wxyz(space.home[3:6])

    # 손 채널은 env 의 hand_control 모드 규약을 따른다:
    #   tip IK 모드  → HAND_TIP_TARGETS, a=0 = 손끝 홈(펴진 자세)
    #   관절/fabric  → HAND_JOINT_TARGETS, a=-1 = 완전 개방
    tip_mode = bool(getattr(env, "_tip_ik", False))
    hand_open = (0.0,) * hand_dim if tip_mode else (-1.0,) * hand_dim
    hand_open_mode = (
        ControlMode.HAND_TIP_TARGETS if tip_mode else ControlMode.HAND_JOINT_TARGETS
    )
    skills: list = [
        ApproachSkill(
            offset=tuple(args.approach_offset),
            orientation_wxyz=approach_quat,
            hand_mode=hand_open_mode,
            hand_targets=hand_open,
        ),
    ]
    plan = ("approach",)
    obs_holder = {"policy": latest_obs["policy"]}
    if artifacts is not None:
        def _policy_obs(env_ids, states):
            rows = obs_holder["policy"]
            return tuple(tuple(float(v) for v in rows[i].tolist()) for i in env_ids)

        backend = RlGamesPolicyBackend(
            artifacts, num_envs=env.num_envs, device=str(env.device),
        )
        grasp_skill = GraspLiftSkill(artifacts, observation_builder=_policy_obs, backend=backend)
        if grasp_skill.action_dim != int(env.cfg.action_space):
            raise SystemExit(
                f"[vlm_demo] 정책 action {grasp_skill.action_dim}D ≠ env {env.cfg.action_space}D — "
                "다른 태스크의 체크포인트다."
            )
        backend.load()   # 첫 tick 중간 스톨 방지 — 여기서 미리 로드
        skills.append(grasp_skill)
        plan = ("approach", "grasp_lift")
        print(f"[vlm_demo] 정책 접속: {artifacts.checkpoint} "
              f"(obs {grasp_skill.observation_dim}D / act {grasp_skill.action_dim}D)", flush=True)

    task = TaskSpecification(
        task="pour",
        source_id=f"cup_{controlled_side}",
        target_id="cup_other",
        nominal_plan=plan,
        allowed_skills=plan + ("recovery",),
    )
    registry = SkillRegistry(skills)
    manager = SkillManager(
        registry=registry,
        num_envs=env.num_envs,
        minimum_steps={SkillId.APPROACH: 2, SkillId.GRASP_LIFT: 5},
    )
    # grasp_lift 성공 기준선 = **안착한 물체 원점**(settle 후 실측) + lift_height.
    view = _EnvSceneView(env)
    baseline_z = float(np.mean([pose[2] for pose in view.object_pose()]))
    provider = FabricStateProvider(
        view,
        manager,
        controlled_side=controlled_side,
        approach_offset=tuple(args.approach_offset),
        approach_tolerance=args.approach_tol,
        grasp_lift_success_z=baseline_z + args.lift_height if artifacts is not None else None,
        grasp_lift_hold_ticks=3,   # 0.6s 유지 — 쳐올림 스파이크를 성공으로 안 센다
    )
    pipeline = PouringPipeline(
        task=task,
        state_provider=provider,
        high_level_policy=DeterministicHighLevelPolicy(),
        skill_manager=manager,
    )

    print(f"[vlm_demo] task={args.task} envs={env.num_envs} action={env.cfg.action_space} "
          f"hand_dim={hand_dim} tick={args.tick_interval}steps side={controlled_side} "
          f"plan={plan} baseline_z={baseline_z:.4f} "
          f"home={[round(v, 3) for v in space.home]}", flush=True)

    # ---- 루프: 저주파 HRL tick + 고주파 절대 액션 유지 ---------------------------
    actions = zero.clone()
    frames = 0
    for step in range(args.steps):
        if step % args.tick_interval == 0:
            result = pipeline.tick()
            palm_now = provider.view.palm_pose_zyx()
            for env_id, command in enumerate(result.commands):
                # 팔/손은 분리 채널 — 각자 독립적으로 갱신한다.
                # ★hold(NO_OP/SAFE_STOP)는 **해당 채널의 마지막 액션을 그대로 유지**.
                #   절대 액션이라 목표가 고정된다 — 매 tick 현재 pose 재명령은
                #   fabric 추종 오차가 래칫이 되어 홈까지 표류했었다(실측 180mm).
                #   파지 후 hold 에서 손 목표까지 유지되는 것도 이 방식뿐이다.
                if command.arm.control_mode not in HOLD_MODES:
                    arm_row = arm_channel_to_action(
                        command.arm, space, hold_pose=tuple(palm_now[env_id])
                    )
                    actions[env_id, :6] = torch.tensor(
                        arm_row, dtype=torch.float32, device=env.device)
                if command.hand.control_mode not in HOLD_MODES:
                    hand_row = hand_channel_to_action(command.hand, hand_dim=hand_dim)
                    actions[env_id, 6:] = torch.tensor(
                        hand_row, dtype=torch.float32, device=env.device)
            for record in result.transitions:
                if record.accepted and record.previous_skill is not record.accepted_skill:
                    print(f"[vlm_demo] step {step:4d} env {record.env_id}: "
                          f"{record.previous_skill.value} -> {record.accepted_skill.value} "
                          f"({record.reason})", flush=True)
        latest_obs, *_ = env.step(actions)
        obs_holder["policy"] = latest_obs["policy"]

        if cam is not None and args.frame_interval > 0 and step % args.frame_interval == 0:
            env.sim.render()
            cam.update(dt=0.0)
            rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
            from PIL import Image
            Image.fromarray(rgb).save(out_dir / f"frame_{step:04d}.png")
            frames += 1

    # ---- 요약 ------------------------------------------------------------------
    palm_final = provider.view.palm_pose_zyx()
    object_final = provider.view.object_pose()
    print("[vlm_demo] === 최종 상태 ===", flush=True)
    for env_id in range(env.num_envs):
        target = [object_final[env_id][axis] + args.approach_offset[axis] for axis in range(3)]
        dist = float(np.linalg.norm(np.array(palm_final[env_id][:3]) - np.array(target)))
        rise = (object_final[env_id][2] - baseline_z) * 1000
        print(f"  env {env_id}: skill={manager.current_skills[env_id].value} "
              f"palm-target {dist * 1000:.1f}mm object-rise {rise:+.1f}mm", flush=True)
    if frames:
        print(f"[vlm_demo] 카메라 프레임 {frames}장 → {out_dir}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

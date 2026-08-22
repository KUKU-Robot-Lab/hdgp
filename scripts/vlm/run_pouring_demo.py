"""VLM pouring 파이프라인을 fabric 태스크 sim 에서 눈으로 확인하는 데모.

Qwen 은 스텁(고정 TaskSpecification)이다 — GPU 경계(모델 로딩) 밖에서
파이프라인 → 스킬 라우팅 → fabric 제어 배선만 검증한다.

동작: 규칙 기반 approach 스킬이 `TASK_SPACE_POSE`(물체 위 pregrasp 지점)를 내고,
`vlm.pouring.fabric_bridge` 가 이를 grasp_lift_fabric 의 절대 액션으로 인코딩해
Fabrics 가 팔을 보낸다. palm 이 목표 반경 안에 들면 결정론적 HRL 이 DONE 으로
전이한다. 학습 스킬(grasp_lift 등)은 아직 붙이지 않는다(로드맵 5단계).

실행 (GUI 로 보려면 --headless 빼기):
    PYTHONUNBUFFERED=1 ../IsaacLab/isaaclab.sh -p scripts/vlm/run_pouring_demo.py \
        --num_envs 2 --steps 600 --out outputs/vlm_demo --headless

카메라 프레임은 --out 아래 PNG 로 저장된다 — 이후 Qwen task-grounding 의
입력 이미지가 되는 바로 그 시점이다.
"""

from __future__ import annotations

import argparse
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
from vlm.pouring.fabric_bridge import (  # noqa: E402
    HoldPoseLatch,
    PalmActionSpace,
    command_to_action,
    euler_zyx_to_quat_wxyz,
)
from vlm.pouring.isaac_state import FabricStateProvider  # noqa: E402
from vlm.pouring.skill_manager import SkillManager  # noqa: E402
from vlm.pouring.skill_registry import SkillRegistry  # noqa: E402
from vlm.pouring.skills import ApproachSkill  # noqa: E402

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


def main() -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
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
    env.step(zero)

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
    task = TaskSpecification(
        task="pour",
        source_id="cup_right",
        target_id="cup_left",
        nominal_plan=("approach",),
        allowed_skills=("approach", "recovery"),
    )
    registry = SkillRegistry([
        ApproachSkill(offset=tuple(args.approach_offset), orientation_wxyz=approach_quat),
    ])
    manager = SkillManager(
        registry=registry,
        num_envs=env.num_envs,
        minimum_steps={SkillId.APPROACH: 2},
    )
    provider = FabricStateProvider(
        _EnvSceneView(env),
        manager,
        approach_offset=tuple(args.approach_offset),
        approach_tolerance=args.approach_tol,
    )
    pipeline = PouringPipeline(
        task=task,
        state_provider=provider,
        high_level_policy=DeterministicHighLevelPolicy(),
        skill_manager=manager,
    )

    print(f"[vlm_demo] task={args.task} envs={env.num_envs} action={env.cfg.action_space} "
          f"hand_dim={hand_dim} tick={args.tick_interval}steps home={[round(v, 3) for v in space.home]}",
          flush=True)

    # ---- 루프: 저주파 HRL tick + 고주파 절대 액션 유지 ---------------------------
    actions = zero.clone()
    frames = 0
    # ★hold 는 진입 시점 pose 를 래치한다 — 매 tick 현재 pose 재명령은 fabric
    #   추종 오차가 래칫이 되어 홈까지 표류한다(실측 44 tick 에 180mm).
    hold_latch = HoldPoseLatch(env.num_envs)
    for step in range(args.steps):
        if step % args.tick_interval == 0:
            result = pipeline.tick()
            palm_now = provider.view.palm_pose_zyx()
            rows = [
                command_to_action(
                    command, space, hand_dim=hand_dim,
                    hold_pose=hold_latch.resolve(env_id, command, tuple(palm_now[env_id])),
                )
                for env_id, command in enumerate(result.commands)
            ]
            actions = torch.tensor(rows, dtype=torch.float32, device=env.device)
            for record in result.transitions:
                if record.accepted and record.previous_skill is not record.accepted_skill:
                    print(f"[vlm_demo] step {step:4d} env {record.env_id}: "
                          f"{record.previous_skill.value} -> {record.accepted_skill.value} "
                          f"({record.reason})", flush=True)
        env.step(actions)

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
        print(f"  env {env_id}: skill={manager.current_skills[env_id].value} "
              f"palm-target {dist * 1000:.1f}mm", flush=True)
    if frames:
        print(f"[vlm_demo] 카메라 프레임 {frames}장 → {out_dir}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

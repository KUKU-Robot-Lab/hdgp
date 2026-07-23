"""live_pour_fluid.py — 라이브 정책 실행 + PBD 유체 평가 (play.py 스타일).

record→replay 가 아니라 **정책을 라이브로 실행**하며 로봇이 물리 grasp 로 컵을 잡고 붓고,
컵 안에 PBD 유체. DexPour 식 라이브 유체 평가.

계획서: hdgp/docs/eval/live_policy_fluid_plan.md
핵심 제약: isaaclab SimulationContext(텐서 파이프라인)가 PBD 를 죽이므로, 텐서 없이
raw Isaac Sim(omni.timeline + app.update)에서 obs/action 파이프라인을 직접 구현한다.

=========================================================================
!!! 이 파일은 프레임워크 스캐폴드다 (2026-07-05, GPU 미검증). !!!
    각 섹션의 `# TODO[Pn]` 은 계획서 단계별로 다음 세션에서 채우고 GPU 로 검증한다.
    포팅 원본: source/openarm/openarm/tesollo/right/pour_v1/pour_right_env.py
=========================================================================
"""

from __future__ import annotations

import argparse

from isaacsim import SimulationApp  # noqa: E402  (SimulationApp 은 pxr/omni import 전 생성)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="라이브 pour 정책 + PBD 유체 평가.")
parser.add_argument("--task", type=str, required=True,
                    help="예: open-tesol_r_pour_v1-play-lstm (env cfg/obs 규격 참조용).")
parser.add_argument("--checkpoint", type=str, required=True, help="rl_games .pth 체크포인트.")
parser.add_argument("--episodes", type=int, default=8, help="평가 에피소드 수.")
parser.add_argument("--headless", action="store_true", default=False)
parser.add_argument("--particle_contact", type=float, default=0.008)
parser.add_argument("--fill_height", type=float, default=0.045)
parser.add_argument("--success_frac", type=float, default=0.5)
parser.add_argument("--report_out", type=str, default=None)
parser.add_argument("--capture_dir", type=str, default=None)
args = parser.parse_args()

app = SimulationApp({"headless": args.headless})

# --- SimulationApp 이후에만 pxr/omni/isaac 로드 가능 ---
import math  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402
import omni.timeline  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

# rl_games 정책 로딩 (record_pour_traj.py 패턴 재사용)
from rl_games.torch_runner import Runner  # noqa: E402,F401

_M = 100.0  # meter → cm (metersPerUnit=0.01, PBD 안정성)


# ===========================================================================
# 1. 씬 구성  (replay_pour_fluid.py 의 raw-app cm 씬을 재사용/이식)
# ===========================================================================
def build_scene():
    """raw Isaac Sim cm 씬 구성: physicsScene(GPU dynamics) + ground + PBD particle system
    + 컵 2개 + 로봇. replay_pour_fluid.py 의 _build_particle_system / _add_cylinder_cup /
    로봇 로드·드라이브 로직을 그대로 이식(cm 스케일, ×100 로봇).

    반환: dict(stage, source_cup, target_cup, robot_prim, drives, palm_prim, system_path, ...)

    # TODO[P1]: contact grasp 를 위해 컵을 kinematic 이 아닌 dynamic 으로.
    #           로봇 collision ENABLED (grasp 접촉 필요). ×100 스케일 contact 안정성 검증 필수.
    #           실패 시 계획서 대안 A(로봇 cm 재베이킹)/B(미터+PBD튜닝)/C(grip fixed-joint).
    """
    raise NotImplementedError("P1: replay_pour_fluid.py 씬 빌드 이식 + dynamic 컵/contact grasp")


# ===========================================================================
# 2. 상태 읽기 (텐서 파이프라인 X — USD/omni.physx 직접 쿼리로 PBD 보존)
# ===========================================================================
def read_joint_state(stage, joint_prims):
    """관절 pos/vel 읽기. UsdPhysics 조인트의 state:*:physics:position/velocity 속성.
    joint_prims: joint_name → prim(Revolute/Prismatic). 순서는 policy obs 규격과 일치해야 함.
    # TODO[P2]: 조인트별 state 속성 정확 확인 (angular=deg? rad? 단위 변환).
    """
    raise NotImplementedError("P2: USD 조인트 state 읽기")


def read_body_pose_cm(prim):
    """body(fingertip/palm) world 포즈. UsdGeom.Xformable.ComputeLocalToWorldTransform → (pos_cm, quat)."""
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    xf = Gf.Transform(m)
    t = xf.GetTranslation()
    q = xf.GetRotation().GetQuat()
    im = q.GetImaginary()
    return (np.array([t[0], t[1], t[2]], dtype=np.float64),
            np.array([q.GetReal(), im[0], im[1], im[2]], dtype=np.float64))


def read_cup_pose(rb_prim):
    """rigid body 컵 world 포즈 (dynamic 컵)."""
    return read_body_pose_cm(rb_prim)  # 필요 시 physx rigid body API 로 교체


def read_contact_forces(sensor_prims):
    """fingertip 접촉력 (obs tip_force_norm). PhysX contact report API.
    # TODO[P2]: env 의 fingertip ContactSensor(Cup-only) 대응. omni.physx contact report 구독.
    """
    raise NotImplementedError("P2: PhysX contact report → tip_force")


# ===========================================================================
# 3. Observation 조립 (actor 55D — pour_right_env._get_observations 이식)
#    포팅 원본: pour_right_env.py line ~1695 (_get_observations), ~1752 (actor_obs cat)
# ===========================================================================
# actor_obs 55D 레이아웃 (2026-07-05 확인, pour_v1 == pour_sensor):
#   arm_joint_pos(7) arm_joint_vel(7) finger_grasp_progress(5)
#   left_arm_joint_pos(9) left_arm_joint_vel(9)
#   pour_point_to_opening(3) source_pour_axis(3) source_up_axis(3) target_up_axis(3)
#   last_palm_actions(6)
NUM_OBSERVATIONS = 55
NUM_PALM_ACTION = 6


def compute_intermediate(source_cup_pose, target_cup_pose, cfg_axes):
    """컵 포즈 + body-frame 축 상수(preset)로 world 기하 계산.
    반환: source_pour_point_w, target_opening_w, source_pour_axis_w, source_up_axis_w, target_up_axis_w.
    preset 상수(pour_v1/pour_right_preset.py):
      SOURCE_CUP_POUR_POINT_POS_B=[0,0,0.100], SOURCE_CUP_POUR_AXIS_B=[1,0,0],
      SOURCE_CUP_UP_AXIS_B=[0,0,1], TARGET_CUP_UP_AXIS_B=[0,0,1], TARGET_CUP_OPENING_POS_B=[0,0,0.100]
    # TODO[P2]: pour_right_env 의 _compute_intermediate_values 정확 이식 (quat_apply 등).
    """
    raise NotImplementedError("P2: pour_point/axes world 기하 이식")


def finger_grasp_progress(finger_joint_pos, hand_open_pose, hand_grasp_pose):
    """손가락 grasp 진행도 5D. pour_right_env._finger_grasp_progress line ~1678 이식.
    (finger_joint_pos - open)/(grasp - open) clamp[0,1], 손가락별 4관절 평균.
    """
    raise NotImplementedError("P2: finger_grasp_progress 이식")


def assemble_actor_obs(state) -> np.ndarray:
    """55D actor obs 조립. record_pour_traj 로 기록한 궤적의 obs 와 일치하는지로 검증(P2)."""
    # obs = np.concatenate([arm_jp, arm_jv, fgp, left_jp, left_jv,
    #                       pour_point_to_opening, src_pour_axis, src_up_axis, tgt_up_axis, last_palm_act])
    # assert obs.shape[0] == NUM_OBSERVATIONS
    raise NotImplementedError("P2: 55D obs 조립 + 차원 검증")


# ===========================================================================
# 4. Action 파이프라인 (pour_right_env._pre_physics_step 이식)
#    포팅 원본: pour_right_env.py line ~1071 (_pre_physics_step). nullspace/gate/latch/clamp 237곳.
# ===========================================================================
def apply_action(action, state, drives):
    """정책 action(7D α) → arm/hand 관절 target → drive 적용.
    포함: palm pose delta → IK/nullspace, hand PCA/grasp, pour_ready latch, clamp.
    # TODO[P3]: _pre_physics_step 의 nullspace(α self-motion)·gate·latch·clamp 정확 이식.
    #           warmstart hold_steps(에피소드 초기 grasp pose 강제 유지)도.
    """
    raise NotImplementedError("P3: action 파이프라인 이식 (최난이도)")


# ===========================================================================
# 5. 정책 로딩 (rl_games)  — record_pour_traj.py 의 로딩 패턴 재사용
# ===========================================================================
def load_policy(task, checkpoint):
    """rl_games player 로드. LSTM 상태 관리 포함.
    # TODO[P3]: record_pour_traj.py 의 Runner/agent 로딩 + get_action + is_rnn 상태 리셋 이식.
    #           env cfg 없이 obs_dim=55, action_dim=7 로 네트워크만 복원.
    """
    raise NotImplementedError("P3: rl_games 정책 로딩 (env 없이 네트워크만)")


# ===========================================================================
# 6. 유체 이송률 측정  (replay_pour_fluid._fraction_in_cup 재사용)
# ===========================================================================
def fraction_in_target(particles_cm, target_pose_cm):
    """target 컵 로컬 프레임 내부 파티clle 비율 = η_ft. replay_pour_fluid._fraction_in_cup 이식."""
    raise NotImplementedError("P4: _fraction_in_cup 이식")


# ===========================================================================
# 7. 메인 루프
# ===========================================================================
def main():
    scene = build_scene()                       # P1
    policy = load_policy(args.task, args.checkpoint)  # P3
    tl = omni.timeline.get_timeline_interface()
    tl.play()

    results = []
    for ep in range(args.episodes):
        # reset_to_warmstart(scene)             # P3: grasp 된 상태로 초기화
        # spawn_fluid(scene)                    # P4: 컵 안 PBD 유체
        # obs, done = assemble_actor_obs(read_state(scene)), False
        # while not done:
        #     action = policy.get_action(obs)   # P3
        #     apply_action(action, state, scene["drives"])  # P3
        #     app.update()                      # 물리 1스텝 (텐서 X)
        #     state = read_state(scene)         # P2
        #     obs = assemble_actor_obs(state)   # P2
        #     done = check_done(state)          # P4
        # eta = fraction_in_target(read_particles(scene), target_pose)  # P4
        # results.append((ep, eta))
        raise NotImplementedError("P1~P4 완료 후 활성화")

    # report(results)  # P5

    import threading
    import time as _time
    threading.Thread(target=app.close, daemon=True).start()
    _time.sleep(3.0)
    os._exit(0)   # PBD teardown hang 회피


if __name__ == "__main__":
    main()

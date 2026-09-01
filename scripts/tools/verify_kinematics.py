#!/usr/bin/env python3
"""OpenArm 7-DOF 진짜 로봇 순방향 기구학(FK) & 오프라인 정밀 좌표 검증 스크립트

이 스크립트는 하드코딩된 오일러 변환이 아니라, 실제 OpenArm 7-DOF URDF 체인의
기구학 행렬(Forward Kinematics)을 0.1초 만에 직접 수치 계산하여 엔드이펙터 3D 위치,
손바닥 정면, 손가락 뻗음 축, 엄지 마운트축을 정밀 출력합니다.

사용법:
    PYTHONPATH=source/openarm:source/FABRICS/src /home/user/rl_ws/IsaacLab/_isaac_sim/python.sh scripts/tools/verify_kinematics.py
"""

import sys
import math
import torch
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

# Fabrics 7-DOF OpenArm URDF 순방향 기구학 엔진 임포트
from fabrics_sim.fabrics.openarm_tesollo_pose_fabric import OpenArmTeoslloPoseFabric

def analyze_7dof_arm_fk(q_arm_list: list[float], label: str = "입력 관절 포즈"):
    """OpenArm 7개 관절 각도 q[0..6]를 입력받아 진짜 7-DOF FK 3D 위치 및 월드 좌표축을 계산합니다."""
    device = 'cpu'
    num_envs = 1
    timestep = 1.0 / 60.0

    fabric = OpenArmTeoslloPoseFabric(
        batch_size=num_envs,
        device=device,
        timestep=timestep,
        graph_capturable=False,
        use_hand_fabric=False,
        robot_dir_name="openarm_tesollo_bi_s",
        robot_name="openarm_tesollo_bi_s",
    )

    q_arm = torch.tensor([q_arm_list], dtype=torch.float32, device=device)
    q_full = torch.zeros(1, 27, device=device)
    q_full[:, :7] = q_arm

    # 1. OpenArm URDF 7-DOF 순방향 기구학(FK) 수행
    palm_pose_6d = fabric.get_palm_pose(q_full, "euler_zyx")  # [x, y, z, roll, pitch, yaw]
    
    pos_3d = palm_pose_6d[0, :3].round(decimals=4).tolist()
    euler_3d_deg = [round(math.degrees(v), 2) for v in palm_pose_6d[0, 3:].tolist()]
    
    # 오일러 각도 ➔ 쿼터니언 변환
    roll, pitch, yaw = palm_pose_6d[0, 3], palm_pose_6d[0, 4], palm_pose_6d[0, 5]
    quat = quat_from_euler_xyz(roll.unsqueeze(0), pitch.unsqueeze(0), yaw.unsqueeze(0))

    # 2. 로컬 3축 3D 월드 투영
    palm_skin_local    = torch.tensor([[0.0,  1.0,  0.0]])  # +Y: 손바닥 피부 정면 (장풍)
    finger_four_local  = torch.tensor([[0.0,  0.0, -1.0]])  # -Z: 4손가락 뻗는 방향
    thumb_local        = torch.tensor([[1.0,  0.0,  0.0]])  # +X: 엄지손가락 마운트 방향
    back_of_hand_local = torch.tensor([[0.0, -1.0,  0.0]])  # -Y: 손등 방향

    palm_skin_world    = quat_apply(quat, palm_skin_local)[0].round(decimals=3).tolist()
    finger_four_world  = quat_apply(quat, finger_four_local)[0].round(decimals=3).tolist()
    thumb_world        = quat_apply(quat, thumb_local)[0].round(decimals=3).tolist()
    back_of_hand_world = quat_apply(quat, back_of_hand_local)[0].round(decimals=3).tolist()

    print("=" * 70)
    print(f"🤖 [OpenArm 7-DOF URDF FK 분석] {label}")
    print("=" * 70)
    print(f"  📥 입력 7개 팔 관절 각도 q[1..7] (rad): {q_arm_list}")
    print(f"  📍 계산된 손바닥 중심 3D 월드 위치 (X, Y, Z): {pos_3d} [m]")
    print(f"  🔄 계산된 오일러 회전 각도 (Roll, Pitch, Yaw): {euler_3d_deg}°")
    print("-" * 70)
    print(f"  ✋ 손바닥 피부 정면 (+Y_local, 장풍) ➔ 월드 방향: {palm_skin_world}")
    print(f"  🤚 손등 방향        (-Y_local)       ➔ 월드 방향: {back_of_hand_world}")
    print(f"  🖐️ 4손가락 뻗음     (-Z_local)       ➔ 월드 방향: {finger_four_world}")
    print(f"  👍 엄지 마운트      (+X_local)       ➔ 월드 방향: {thumb_world}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    # 1. 양팔 수직 차렷 포즈 [0, 0, 0, 0, 0, 0, 0] FK 계산
    analyze_7dof_arm_fk([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], label="양팔 수직 차렷 정지 자세")

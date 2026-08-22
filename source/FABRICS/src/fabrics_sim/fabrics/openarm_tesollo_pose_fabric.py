# OpenArm + Teosllo right arm pose fabric
# Based on kuka_allegro_pose_fabric.py
# 7 DOF OpenArm right arm + 20 DOF Teosllo right hand = 27 DOF total

import torch

from fabrics_sim.fabric_terms.attractor import Attractor
from fabrics_sim.fabric_terms.joint_limit_repulsion import JointLimitRepulsion
from fabrics_sim.fabric_terms.body_sphere_3d_repulsion import BodySphereRepulsion
from fabrics_sim.fabric_terms.body_sphere_3d_repulsion import BaseFabricRepulsion
from fabrics_sim.fabrics.fabric import BaseFabric
from fabrics_sim.taskmaps.identity import IdentityMap
from fabrics_sim.taskmaps.upper_joint_limit import UpperJointLimitMap
from fabrics_sim.taskmaps.lower_joint_limit import LowerJointLimitMap
from fabrics_sim.taskmaps.linear_taskmap import LinearMap
from fabrics_sim.energy.euclidean_energy import EuclideanEnergy
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotFrameOriginsTaskMap
from fabrics_sim.utils.path_utils import get_robot_urdf_path
from fabrics_sim.utils.rotation_utils import euler_to_matrix, matrix_to_euler
from fabrics_sim.utils.rotation_utils import quaternion_to_matrix, matrix_to_quaternion


class OpenArmTeoslloPoseFabric(BaseFabric):
    """
    Fabric for OpenArm right arm (7 DOF) + Teosllo right hand (20 DOF) = 27 DOF total.
    Action space: 6D palm pose + 5D hand PCA = 11D (same as DEXTRAH paper).

    Joint order in URDF (27 revolute joints):
      [0-6]  openarm_right_joint1~7     (arm)
      [7-10] rj_dg_1_1~4               (thumb)
      [11-14] rj_dg_2_1~4              (index)
      [15-18] rj_dg_3_1~4              (middle)
      [19-22] rj_dg_4_1~4              (ring)
      [23-26] rj_dg_5_1~4              (pinky)
    """

    def __init__(self, batch_size, device, timestep, graph_capturable=True, use_hand_fabric=True,
                 palm_position_only=False,
                 robot_dir_name="openarm_tesollo", robot_name="openarm_tesollo",
                 default_config_override=None, default_palm_euler_zyx=None,
                 fabric_params_filename=None):
        self._use_hand_fabric = use_hand_fabric
        # [새 구조] palm_position_only=True: palm_link origin 1점(position 3-DOF)만 attractor로
        #   고정하고 orientation은 자유(cspace nullspace가 결정). j6 leak 차단 → IK가 j5 roll을
        #   demo대로 실현. False(기본)=기존 7-point full 6-DOF(v5 대조군 유지).
        self._palm_position_only = palm_position_only
        # 좌팔 등 변형 URDF 지원: robot_dir_name/robot_name/default_config/palm 기본자세만
        # 바꾸면 재사용 가능 (좌측 URDF 는 링크/조인트 이름을 우측과 동일하게 유지).
        self._default_config_override = default_config_override
        self._default_palm_euler_zyx = default_palm_euler_zyx
        # params 파일도 변형이 바꿀 수 있게 열어둔다. 기본값은 기존 소비자(pour/grasp_v1/
        # grasp_sensor) 보호용으로 그대로 유지 — 인자를 안 주면 동작이 완전히 동일하다.
        # ⚠ joint_limits.acceleration 리스트 길이가 곧 cspace_dim 이다(fabric.py:155).
        #   관절 수가 다른 변형은 반드시 자기 params 파일을 줘야 한다.
        fabric_params_filename = fabric_params_filename or "openarm_tesollo_pose_params.yaml"
        super().__init__(device, batch_size, timestep, fabric_params_filename,
                         graph_capturable=graph_capturable)

        self.urdf_path = get_robot_urdf_path(robot_dir_name, robot_name)

        self.load_robot(robot_dir_name, robot_name, batch_size)

        # Default cspace config (27 DOF):
        # Arm: natural working pose for right OpenArm
        # Hand: slightly curled (Kuka-Allegro 방식, 0.5~0.75 구부러짐과 동일)
        #   thumb _2 음수=curl, 나머지 손가락 _2 양수=curl
        #   robot_start_joint_pos 및 curled_q와 일치
        default_config = torch.tensor([
            # OpenArm right arm joint1~7
            # j3: 0.0 → -0.6 (내회전 방향, pour tilt 유도: null-space가 j3를 음수 방향으로 당김)
            1.0,  -0.1,  -0.6,  0.5,  0.0,  0.0,  0.0,
            # Teosllo thumb (rj_dg_1_1~4):
            #   _1(X): 0.0 (neutral abduction)
            #   _2(Z): -1.0 (opposition curl, 한계 -π, 32%)
            #   _3,_4: 0.5 (distal flex; PC3 최대 시 _2=-1.5, _3=0.5+0.7=1.2)
            0.0,  -1.0,  0.5,  0.5,
            # Index (rj_dg_2_1~4):
            #   _1(X, 외전): 0.0  _2(Y, curl): 0.7(한계2.007, 35%)  _3,_4: 0.5
            0.0,   0.7,  0.5,  0.5,
            # Middle (rj_dg_3_1~4):
            #   _2(Y, curl): 0.7(한계1.955, 36%)
            0.0,   0.7,  0.5,  0.5,
            # Ring (rj_dg_4_1~4):
            #   _2(Y, curl): 0.7(한계1.902, 37%)
            0.0,   0.7,  0.5,  0.5,
            # Pinky (rj_dg_5_1~4):
            #   _1(Z, 굽힘관절!): 0.0 (사용자 확인: 파지 자세에서 _1=0)
            #   _2(X, 외전): 0.0  _3: 0.7  _4: 0.5
            0.0,   0.0,  0.7,  0.5,
        ], device=self.device)
        if self._default_config_override is not None:
            default_config = torch.as_tensor(
                self._default_config_override, device=self.device, dtype=default_config.dtype
            )
        self.default_config = default_config.unsqueeze(0).repeat(self.batch_size, 1)

        self._pca_matrix = None

        self.construct_fabric()

        # Allocate palm pose target tensor (b x 12): 3D origin + 9D rotation matrix
        self._palm_pose_target = torch.zeros(batch_size, 12, device=device)
        # Default palm orientation (euler_zyx): ez=pi/2, ey=0, ex=pi/2
        # -> palm +X aligns with world +Y, palm +Z aligns with world +X.
        # (좌팔은 미러: M R M = Rz(-ez) Ry(ey) Rx(-ex) → (-pi/2, 0, -pi/2))
        _palm_euler = self._default_palm_euler_zyx or (1.5708, 0.0, 1.5708)
        default_palm_euler = torch.tensor(list(_palm_euler), device=self.device).unsqueeze(0)
        default_palm_euler = default_palm_euler.repeat(self.batch_size, 1)
        self._palm_pose_target[:, 3:] = torch.transpose(
            euler_to_matrix(default_palm_euler), 1, 2
        ).reshape(self.batch_size, 9)
        self._native_palm_pose_target = None

        # Fingertip FK taskmap (sim2real용: rl_dg_*_tip 위치 계산)
        _tip_frames = [f"rl_dg_{i}_tip" for i in range(1, 6)]
        self._fingertip_taskmap = RobotFrameOriginsTaskMap(
            self.urdf_path, _tip_frames, batch_size, device
        )

    # ------------------------------------------------------------------
    # Fabric construction methods
    # ------------------------------------------------------------------

    def add_joint_limit_repulsion(self):
        joints = self.urdfpy_robot.joints
        upper_joint_limits = []
        lower_joint_limits = []
        for j in joints:
            if j.joint_type == 'revolute':
                upper_joint_limits.append(j.limit.upper)
                lower_joint_limits.append(j.limit.lower)

        # Upper limit repulsion
        taskmap_name = "upper_joint_limit"
        taskmap = UpperJointLimitMap(upper_joint_limits, self.batch_size, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)
        fabric = JointLimitRepulsion(True, self.fabric_params['joint_limit_repulsion'],
                                     self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, "joint_limit_repulsion", fabric)

        # Lower limit repulsion
        taskmap_name = "lower_joint_limit"
        taskmap = LowerJointLimitMap(lower_joint_limits, self.batch_size, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)
        fabric = JointLimitRepulsion(True, self.fabric_params['joint_limit_repulsion'],
                                     self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, "joint_limit_repulsion", fabric)

    def add_cspace_attractor(self, is_forcing):
        taskmap_name = "identity"
        taskmap = IdentityMap(self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)

        if not is_forcing:
            fabric_name = "cspace_attractor"
            fabric = Attractor(is_forcing, self.fabric_params['cspace_attractor'],
                               self.device, graph_capturable=self.graph_capturable)
            self.add_fabric(taskmap_name, fabric_name, fabric)
        else:
            fabric_name = "forcing_cspace_attractor"
            fabric = Attractor(is_forcing, self.fabric_params['forcing_cspace_attractor'],
                               self.device, graph_capturable=self.graph_capturable)

        self.add_fabric(taskmap_name, fabric_name, fabric)

    def add_hand_fabric(self):
        """
        PCA action space for Teosllo 20-DOF hand.
        Maps 5D PCA → 20D hand joint space.

        Joint order in hand block (indices 7-26):
          [0-3]  thumb  rj_dg_1_1~4
          [4-7]  index  rj_dg_2_1~4
          [8-11] middle rj_dg_3_1~4
          [12-15] ring  rj_dg_4_1~4
          [16-19] pinky rj_dg_5_1~4

        PC1: global proximal+middle flexion  (_2=0.38, _3=0.45 — Kuka 데이터 기반)
        PC2: global distal flexion           (_4=0.55 지배 — Kuka 데이터 기반)
        PC3: proximal펼침 + middle굽힘 diff  (_2 음수 + _3 양수 — Kuka 데이터 기반)
        PC4: thumb opposition 전담           (Kuka에서 thumb 담당 PC)
        PC5: spread + complex differential   (Kuka 데이터 기반)
        """
        # 5 x 20 PCA matrix  (rows = PCA components, cols = hand joints)
        # Col order: [th1,th2,th3,th4, idx1,idx2,idx3,idx4,
        #             mid1,mid2,mid3,mid4, rng1,rng2,rng3,rng4,
        #             pnk1,pnk2,pnk3,pnk4]
        #
        # ─── 설계 원칙 ──────────────────────────────────────────────────────────
        # Kuka Allegro PCA matrix (5×16) 를 Teosllo finger 순서에 맞게 재배열
        # Allegro joint 순서: index(0-3), middle(4-7), ring(8-11), thumb(12-15)
        # Teosllo 매핑:
        #   thumb (cols 0-3)  ← Allegro thumb  (cols 12-15)
        #   index (cols 4-7)  ← Allegro index  (cols 0-3)
        #   middle(cols 8-11) ← Allegro middle (cols 4-7)
        #   ring  (cols 12-15)← Allegro ring   (cols 8-11)
        #   pinky (cols 16-19)← zeros (cspace attractor GRASP_POSE로 고정)
        # ─────────────────────────────────────────────────────────────────────
        pca_matrix = torch.tensor([
            # PC1
            # thumb(Allegro 12-15), index(0-3), middle(4-7), ring(8-11), pinky(zeros)
            # ※ thumb col1 (rj_dg_1_2, Z축): Allegro thumb_joint_1 양수=curl,
            #   Teosllo rj_dg_1_2 음수=curl (같은 Z축이지만 curl 방향 반대) → 부호 반전
            [-1.4790e-02, -9.8163e-02,  4.3551e-02,  3.1699e-01,   # thumb ← Allegro thumb (col1 negated)
             -3.8872e-02,  3.7917e-01,  4.4703e-01,  7.1016e-03,   # index ← Allegro index
              2.1159e-03,  3.2014e-01,  4.4660e-01,  5.2108e-02,   # middle
              5.6869e-05,  2.9845e-01,  3.8575e-01,  7.5774e-03,   # ring
              0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],  # pinky: fixed

            # PC2
            [ 2.9753e-02, -2.6149e-02,  6.6994e-02,  1.8117e-01,   # thumb (col1 negated)
             -5.1148e-02, -1.3007e-01,  5.7727e-02,  5.7914e-01,   # index
              1.0156e-02, -1.8469e-01,  5.3809e-02,  5.4888e-01,   # middle
              1.3351e-04, -1.7747e-01,  2.7809e-02,  4.8187e-01,   # ring
              0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],  # pinky: fixed

            # PC3
            [ 2.3925e-03, -3.7238e-02, -1.0124e-01, -1.7442e-02,   # thumb (col1 negated)
             -5.7137e-02, -3.4707e-01,  3.3365e-01, -1.8029e-01,   # index
             -4.3560e-02, -4.7666e-01,  3.2517e-01, -1.5208e-01,   # middle
             -5.9691e-05, -4.5790e-01,  3.6536e-01, -1.3916e-01,   # ring
              0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],  # pinky: fixed

            # PC4 (thumb opposition PC)
            [ 2.2661e-01,  5.9911e-01,  7.0257e-01, -2.4525e-01,   # thumb (col1 negated: -0.599→+0.599)
              2.2795e-02, -3.4090e-02,  3.4366e-02, -2.6531e-02,   # index
              2.3471e-02,  4.6123e-02,  9.8059e-02, -1.2619e-03,   # middle
             -1.6452e-04, -1.3741e-02,  1.3813e-01,  2.8677e-02,   # ring
              0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],  # pinky: fixed

            # PC5
            [-4.7617e-01,  2.7734e-01, -2.3989e-01, -3.1222e-01,   # thumb (col1 negated: -0.277→+0.277)
             -4.4911e-02, -4.7156e-01,  9.3124e-02,  2.3135e-01,   # index
             -2.4607e-03,  9.5564e-02,  1.2470e-01,  3.6613e-02,   # middle
              1.3821e-04,  4.6072e-01,  9.9315e-02, -8.1080e-02,   # ring
              0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],  # pinky: fixed
        ], device=self.device)

        self._pca_matrix = torch.clone(pca_matrix.detach())

        # Pad with zeros for the 7 arm joints (arm joints not controlled via PCA)
        pca_matrix = torch.cat(
            [torch.zeros(pca_matrix.shape[0], 7, device=self.device), pca_matrix], dim=1
        )

        taskmap_name = "pca_hand"
        taskmap = LinearMap(pca_matrix, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)

        fabric = Attractor(True, self.fabric_params['hand_attractor'],
                           self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, "hand_attractor", fabric)

    def add_palm_points_attractor(self):
        """
        7-point palm frame attractor (origin + 6 axis points) for full 6-DOF palm control.
        """
        taskmap_name = "palm"
        if self._palm_position_only:
            # [새 구조] palm_link origin 1점만 → position 3-DOF. orientation task 없음.
            control_point_frames = ["palm_link"]
        else:
            control_point_frames = [
                "palm_link",
                "palm_x",  "palm_x_neg",
                "palm_y",  "palm_y_neg",
                "palm_z",  "palm_z_neg",
            ]
        taskmap = RobotFrameOriginsTaskMap(self.urdf_path, control_point_frames,
                                           self.batch_size, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)

        fabric = Attractor(True, self.fabric_params['palm_attractor'],
                           self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, "palm_attractor", fabric)

    def add_body_repulsion(self):
        collision_sphere_frames = self.fabric_params['body_repulsion']['collision_sphere_frames']
        self.collision_sphere_radii = self.fabric_params['body_repulsion']['collision_sphere_radii']

        assert len(collision_sphere_frames) == len(self.collision_sphere_radii), \
            "length of link names does not equal length of radii"

        collision_sphere_pairs = self.fabric_params['body_repulsion']['collision_sphere_pairs']

        collision_matrix = torch.zeros(
            len(collision_sphere_frames), len(collision_sphere_frames),
            dtype=int, device=self.device
        )

        if len(collision_sphere_pairs) == 0:
            collision_link_prefix_pairs = \
                self.fabric_params['body_repulsion']['collision_link_prefix_pairs']
            for prefix1, prefix2 in collision_link_prefix_pairs:
                frames_for_prefix1 = [s for s in collision_sphere_frames if prefix1 in s]
                frames_for_prefix2 = [s for s in collision_sphere_frames if prefix2 in s]
                for sphere1 in frames_for_prefix1:
                    for sphere2 in frames_for_prefix2:
                        collision_sphere_pairs.append([sphere1, sphere2])

        for sphere1, sphere2 in collision_sphere_pairs:
            collision_matrix[
                collision_sphere_frames.index(sphere1),
                collision_sphere_frames.index(sphere2)
            ] = 1

        taskmap_name = "body_points"
        taskmap = RobotFrameOriginsTaskMap(self.urdf_path, collision_sphere_frames,
                                           self.batch_size, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)

        sphere_radius = torch.tensor(self.collision_sphere_radii, device=self.device)
        sphere_radius = sphere_radius.repeat(self.batch_size, 1)

        fabric = BodySphereRepulsion(True, self.fabric_params['body_repulsion'],
                                     self.batch_size, sphere_radius, collision_matrix,
                                     self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, "repulsion", fabric)

        fabric_geom = BodySphereRepulsion(False, self.fabric_params['body_repulsion'],
                                          self.batch_size, sphere_radius, collision_matrix,
                                          self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, "geom_repulsion", fabric_geom)

        self.base_fabric_repulsion = BaseFabricRepulsion(
            self.fabric_params['body_repulsion'],
            self.batch_size,
            sphere_radius,
            collision_matrix,
            self.device,
        )

    def add_cspace_energy(self):
        taskmap_name = "identity"
        self.add_energy(
            taskmap_name, "euclidean",
            EuclideanEnergy(self.batch_size, self._num_joints, self.device)
        )

    def construct_fabric(self):
        self.add_joint_limit_repulsion()
        self.add_cspace_attractor(False)
        if self._use_hand_fabric:
            self.add_hand_fabric()
        self.add_palm_points_attractor()
        self.add_body_repulsion()
        self.add_cspace_energy()

    # ------------------------------------------------------------------
    # Runtime methods
    # ------------------------------------------------------------------

    def convert_transform_to_points(self):
        """Convert palm pose target (origin + rotation matrix) to 7×3D control points.

        palm_position_only=True면 origin 1점(3D)만 반환 (orientation 무시 = position attractor).
        """
        if self._palm_position_only:
            return self._palm_pose_target[:, :3].clone()  # (B, 3) origin only
        palm_transform = torch.zeros(self.batch_size, 4, 4, device=self.device)
        palm_transform[:, 3, 3] = 1.
        palm_transform[:, :3, :3] = torch.transpose(
            self._palm_pose_target[:, 3:].reshape(self.batch_size, 3, 3), 1, 2
        )
        palm_transform[:, :3, 3] = self._palm_pose_target[:, :3]

        def _axis_point(offset_xyz):
            p = torch.zeros(self.batch_size, 4, device=self.device)
            p[:, 3] = 1.
            p[:, 0] = offset_xyz[0]
            p[:, 1] = offset_xyz[1]
            p[:, 2] = offset_xyz[2]
            return p

        palm_targets = torch.zeros(self.batch_size, 7 * 3, device=self.device)
        # Origin
        palm_targets[:, :3] = self._palm_pose_target[:, :3]
        # ±x
        palm_targets[:, 3:6]   = torch.bmm(palm_transform, _axis_point([0.25, 0., 0.]).unsqueeze(2)).squeeze(2)[:, :3]
        palm_targets[:, 6:9]   = torch.bmm(palm_transform, _axis_point([-0.25, 0., 0.]).unsqueeze(2)).squeeze(2)[:, :3]
        # ±y
        palm_targets[:, 9:12]  = torch.bmm(palm_transform, _axis_point([0., 0.25, 0.]).unsqueeze(2)).squeeze(2)[:, :3]
        palm_targets[:, 12:15] = torch.bmm(palm_transform, _axis_point([0., -0.25, 0.]).unsqueeze(2)).squeeze(2)[:, :3]
        # ±z
        palm_targets[:, 15:18] = torch.bmm(palm_transform, _axis_point([0., 0., 0.25]).unsqueeze(2)).squeeze(2)[:, :3]
        palm_targets[:, 18:21] = torch.bmm(palm_transform, _axis_point([0., 0., -0.25]).unsqueeze(2)).squeeze(2)[:, :3]
        return palm_targets

    def get_sphere_radii(self):
        return self.collision_sphere_radii

    @property
    def collision_status(self):
        return self.base_fabric_repulsion.collision_status

    def get_palm_pose(self, cspace_position, orientation_convention):
        palm_points, _ = self.get_taskmap("palm")(cspace_position, None)
        palm_origin = palm_points[:, :3]
        if self._palm_position_only:
            # 1점 모드: orientation 정보 없음 → identity 반환 (호출측은 position만 사용).
            if orientation_convention == "quaternion":
                orientation = torch.zeros(self.batch_size, 4, device=self.device)
                orientation[:, 3] = 1.0  # qw=1 (xyzw)
            else:  # euler_zyx
                orientation = torch.zeros(self.batch_size, 3, device=self.device)
            return torch.cat([palm_origin, orientation], dim=-1)
        x_point = palm_points[:, 3:6]
        y_point = palm_points[:, 9:12]
        z_point = palm_points[:, 15:18]

        x_axis = torch.nn.functional.normalize(x_point - palm_origin, dim=1)
        y_axis = torch.nn.functional.normalize(y_point - palm_origin, dim=1)
        z_axis = torch.nn.functional.normalize(z_point - palm_origin, dim=1)

        rotation_matrix = torch.zeros(self.batch_size, 3, 3, device=self.device)
        rotation_matrix[:, :, 0] = x_axis
        rotation_matrix[:, :, 1] = y_axis
        rotation_matrix[:, :, 2] = z_axis

        if orientation_convention == "euler_zyx":
            orientation = matrix_to_euler(rotation_matrix)
        elif orientation_convention == "quaternion":
            orientation = matrix_to_quaternion(rotation_matrix)[:, [1, 2, 3, 0]]
        else:
            raise ValueError('orientation_convention must be "euler_zyx" or "quaternion"')

        return torch.cat([palm_origin, orientation], dim=-1)

    def get_fingertip_positions(self, cspace_position: torch.Tensor) -> torch.Tensor:
        """FK로 5개 손가락 끝 위치를 계산합니다 (sim2real 관측 구성용).

        Args:
            cspace_position: (B, 27) 관절 각도

        Returns:
            (B, 5, 3) 월드 프레임 기준 fingertip 위치
              [0] = rl_dg_1_tip (thumb)
              [1] = rl_dg_2_tip (index)
              [2] = rl_dg_3_tip (middle)
              [3] = rl_dg_4_tip (ring)
              [4] = rl_dg_5_tip (pinky)
        """
        tip_points, _ = self._fingertip_taskmap(cspace_position, None)
        return tip_points.view(cspace_position.shape[0], 5, 3)

    @property
    def pca_matrix(self):
        return self._pca_matrix

    @pca_matrix.setter
    def pca_matrix(self, pca_matrix):
        self._pca_matrix = pca_matrix

    def set_features(self, hand_target, palm_pose_target, orientation_convention,
                     batched_cspace_position, batched_cspace_velocity,
                     object_ids, object_indicator,
                     cspace_damping_gain=None):
        """
        Pass input features to fabric terms.

        Args:
            hand_target:              (B, 5)  PCA hand target
            palm_pose_target:         (B, 6)  [x,y,z, ez,ey,ex] with euler_zyx
                                   or (B, 7)  [x,y,z, qx,qy,qz,qw] with quaternion
            orientation_convention:   "euler_zyx" or "quaternion"
            batched_cspace_position:  (B, 27) current joint positions
            batched_cspace_velocity:  (B, 27) current joint velocities
            object_ids:               Warp array of object mesh IDs
            object_indicator:         Warp array indicating mesh presence
            cspace_damping_gain:      Optional damping gain scalar
        """
        if "pca_hand" in self.fabrics_features:
            self.fabrics_features["pca_hand"]["hand_attractor"] = hand_target
        self.fabrics_features["identity"]["cspace_attractor"] = self.default_config

        self._palm_pose_target[:, :3] = palm_pose_target[:, :3]

        if orientation_convention == "euler_zyx":
            assert palm_pose_target.shape[1] == 6, \
                "euler_zyx pose target must be (B, 6)"
            self._palm_pose_target[:, 3:] = torch.transpose(
                euler_to_matrix(palm_pose_target[:, 3:]), 1, 2
            ).reshape(self.batch_size, 9)
        elif orientation_convention == "quaternion":
            assert palm_pose_target.shape[1] == 7, \
                "quaternion pose target must be (B, 7)"
            self._palm_pose_target[:, 3:] = torch.transpose(
                quaternion_to_matrix(palm_pose_target[:, [6, 3, 4, 5]]), 1, 2
            ).reshape(self.batch_size, 9)
        elif orientation_convention == "matrix":
            # (B, 12) = [x,y,z, R_flat(9)]. CUDA Graph 캡처용 경로 — euler/quaternion 이
            # 쓰는 fancy indexing([:, [6,3,4,5]])·euler_to_matrix 없이 rotation matrix 를
            # slice+reshape 로만 다뤄 stream capture 중 금지 연산을 피한다. 저장 형식은
            # 기존 두 경로와 동일(transpose 후 9D flatten) → 결과 등가.
            assert palm_pose_target.shape[1] == 12, \
                "matrix pose target must be (B, 12)"
            self._palm_pose_target[:, 3:] = torch.transpose(
                palm_pose_target[:, 3:].reshape(self.batch_size, 3, 3), 1, 2
            ).reshape(self.batch_size, 9)
        else:
            raise ValueError('orientation_convention must be "euler_zyx", "quaternion", or "matrix"')

        palm_pose_target_points = self.convert_transform_to_points()

        if self._native_palm_pose_target is None:
            self._native_palm_pose_target = torch.clone(palm_pose_target_points)
        else:
            self._native_palm_pose_target.copy_(palm_pose_target_points)

        try:
            self.fabrics_features["palm"]["palm_attractor"] = self._native_palm_pose_target
            self.get_fabric_term("palm", "palm_attractor").damping_position = \
                self._native_palm_pose_target
        except Exception:
            raise ValueError('No task map "palm" or "palm_attractor"')

        # Compute body sphere positions and velocities
        body_point_pos, jac = self.get_taskmap("body_points")(batched_cspace_position, None)
        body_point_vel = torch.bmm(jac, batched_cspace_velocity.unsqueeze(2)).squeeze(2)

        self.base_fabric_repulsion.calculate_response(
            body_point_pos, body_point_vel, object_ids, object_indicator
        )

        self.fabrics_features["body_points"]["repulsion"] = self.base_fabric_repulsion
        self.fabrics_features["body_points"]["geom_repulsion"] = self.base_fabric_repulsion

        if cspace_damping_gain is not None:
            self.fabric_params['cspace_damping']['gain'] = cspace_damping_gain


# ---------------------------------------------------------------------------
# 좌팔 변형: openarm_tesollo_left URDF (bi USD 좌측과 FK 일치 검증됨)
# ---------------------------------------------------------------------------
# 좌측 URDF 는 scripts/assets_tools/generate_left_fabric_urdf.py 로 생성:
#   - 우측 fabric URDF 의 M-conjugation 미러 (링크/조인트 이름 동일 유지)
#   - axis/limits 는 bi USD(openarm_tesollo_bi_rl) 좌측 규약
# q 부호 매핑 (q_left = s * q_right):
#   arm  j1~j7:            [-1,-1,-1, 1,-1,-1,-1]
#   thumb  _1~_4:          [-1,-1,-1,-1]
#   index/middle/ring _1~_4: [-1, 1, 1, 1]
#   pinky  _1~_4:          [-1,-1, 1, 1]
_LEFT_DEFAULT_CONFIG = [
    # arm j1~j7 (우측 default 의 s 매핑)
    -1.0,  0.1,  0.6,  0.5,  0.0,  0.0,  0.0,
    # thumb: _2(Z) 대향 curl 은 좌측에서 +1.0
    0.0,  1.0,  -0.5,  -0.5,
    # index / middle / ring: curl(Y축)은 부호 유지
    0.0,  0.7,  0.5,  0.5,
    0.0,  0.7,  0.5,  0.5,
    0.0,  0.7,  0.5,  0.5,
    # pinky: _1(Z 굽힘) 부호 반전형이나 default 0
    0.0,  0.0,  0.7,  0.5,
]


class OpenArmTeoslloLeftPoseFabric(OpenArmTeoslloPoseFabric):
    """OpenArm 좌팔(7 DOF) + Teosllo 좌손(20 DOF) fabric.

    출력 q 는 openarm_tesollo_bi_rl.usd 좌측 관절(l_aj/l_hj)에 그대로 사용 가능
    (generate_left_fabric_urdf.py FK 교차검증 PASS).
    """

    def __init__(self, batch_size, device, timestep, graph_capturable=True,
                 use_hand_fabric=False, palm_position_only=False,
                 robot_dir_name="openarm_tesollo_left",
                 robot_name="openarm_tesollo_left"):
        # ★08.17 robot_dir_name/robot_name 패스스루 추가: DG-5FS 전용 URDF
        #   (openarm_tesollo_bi_s_left)를 쓰는 태스크가 기존 URDF 를 건드리지 않고
        #   선택할 수 있게 한다. 기본값은 구 DG-5F(pour 등 기존 소비자 보호).
        super().__init__(
            batch_size, device, timestep,
            graph_capturable=graph_capturable,
            use_hand_fabric=use_hand_fabric,
            palm_position_only=palm_position_only,
            robot_dir_name=robot_dir_name,
            robot_name=robot_name,
            default_config_override=_LEFT_DEFAULT_CONFIG,
            # 우측 기본 palm 자세 (ez,ey,ex)=(π/2,0,π/2) 의 미러 = (-π/2,0,-π/2)
            default_palm_euler_zyx=(-1.5708, 0.0, -1.5708),
        )


# ---------------------------------------------------------------------------
# 좌팔 2지 그리퍼 변형: openarm_tesollo_sensor_left_gripper URDF
# ---------------------------------------------------------------------------
# 대상 로봇: openarm_tesollo_sensor_rl 의 **왼팔**(7 DOF) + 2-DOF 프리즈매틱 그리퍼.
# URDF 는 scripts/assets_tools/generate_sensor_left_gripper_fabric_urdf.py 로 생성한다:
#   - 팔 7관절 origin/axis/limit = sensor_rl `l_aj_1..7` 실값 (미러 추정 아님, FK 0 오차 검증)
#   - 손 20관절은 **fixed 로 동결** → BaseFabric 이 revolute 만 세므로 cspace = 팔 7 DOF
#   - palm_link = 그리퍼 TCP, 축은 그리퍼 고유축 (+z 접근, +y jaw, +x 핑거 폭)
#   - 링크/조인트 이름은 우측(openarm_tesollo)과 동일 → fabric_params 프레임 리스트 재사용
#
# ⚠ 그리퍼 개폐(l_hj_gripper_1)는 이 fabric 이 제어하지 않는다. RL 액션이 직접 관절 목표를 준다.
#   Fabrics 는 팔 자세(IK)만 담당한다.
#
# cspace default = 그리퍼 태스크의 **홈 자세**(gripper/left/grasp_sensor preset 과 동일 값).
#   cspace attractor 는 이 자세로 당기므로, 파지 자세군 밖의 값을 넣으면 palm attractor 와
#   싸워 자세를 못 낸다. 실제로 처음에는 우팔 DG-5F 홈의 미러를 썼다가 Fabrics 가 파지
#   자세를 못 내고 jaw 가 28.5° 기울었다(Isaac 실측). 홈은 파지 해들의 관절공간 중심에서
#   물러난 자세로 다시 뽑았다(scripts/probes/probe_left_gripper_home.py).
#   ⚠ preset 의 LEFT_ARM_HOME_JOINT_POS 와 **항상 같이 바꿀 것**.
_GRIPPER_LEFT_DEFAULT_CONFIG = [
    -0.1569, -0.5984, +1.4065, +1.2005, +1.0895, -0.6695, +1.3563,
]


class OpenArmGripperLeftPoseFabric(OpenArmTeoslloPoseFabric):
    """OpenArm 좌팔(7 DOF) + 2지 그리퍼 fabric.

    출력 q(7,)는 openarm_tesollo_sensor_rl.usd 의 `l_aj_1..7` 에 그대로 사용 가능
    (generate_sensor_left_gripper_fabric_urdf.py FK 교차검증 PASS).
    """

    def __init__(self, batch_size, device, timestep, graph_capturable=True,
                 use_hand_fabric=False, palm_position_only=False,
                 robot_dir_name="openarm_tesollo_sensor_left_gripper",
                 robot_name="openarm_tesollo_sensor_left_gripper",
                 default_palm_euler_zyx=(0.0, 1.5708, 0.0),
                 fabric_params_filename="openarm_gripper_left_pose_params.yaml",
                 default_config_override=None):
        # 기본 palm 자세 (ez,ey,ex)=(0, π/2, 0) → R = Ry(90°):
        #   palm +z(접근축) = world +X,  palm +y(jaw) = world +Y,  palm +x(핑거 폭) = world -Z.
        #   즉 로봇 앞쪽으로 뻗어 컵의 좌우면을 수평 jaw 로 집는 **측면 파지** 기본자세.
        #   (우측 손의 (π/2,0,π/2) 와 달리 palm 프레임 정의 자체가 다르므로 미러값이 아니다.)
        if use_hand_fabric:
            raise ValueError(
                "2지 그리퍼에는 hand fabric(PCA 20관절)이 없다. use_hand_fabric=False 로 쓸 것."
            )
        super().__init__(
            batch_size, device, timestep,
            graph_capturable=graph_capturable,
            use_hand_fabric=False,
            palm_position_only=palm_position_only,
            robot_dir_name=robot_dir_name,
            robot_name=robot_name,
            # ★cspace rest(= attractor 가 팔을 당기는 기본 자세)는 **소비 태스크의 홈**과
            #   일치해야 한다. 내장값은 ABORTED 트랙 홈(j7=+1.356)인데, lift 트랙 홈은
            #   j7=−0.331 로 전혀 다르고 j7>0.7 은 l_al_5↔l_al_7 자기충돌 여유가 9 mm
            #   아래로 떨어지는 구간이다. 태스크가 자기 홈을 넘겨야 한다.
            default_config_override=(
                default_config_override
                if default_config_override is not None
                else _GRIPPER_LEFT_DEFAULT_CONFIG
            ),
            default_palm_euler_zyx=default_palm_euler_zyx,
            # ★팔 7 DOF 전용 params. 27 길이 accel/jerk 를 그대로 쓰면 첫 스텝에서
            #   "Number of joints does not match ..." assert 로 죽는다(실측).
            fabric_params_filename=fabric_params_filename,
        )

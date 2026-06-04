# OpenArm + RH56F1 right arm pose fabric
# Based on openarm_tesollo_pose_fabric.py
# 7 DOF OpenArm right arm + 6 actuated RH56F1 right hand = 13 DOF total
#
# 설계 메모 (계획 Phase 1):
#   - RH56F1 손은 underactuated 6 DOF (drive): thumb_1, thumb_2, index_1,
#     middle_1, ring_1, little_1. mimic 추종 관절은 fabrics URDF 에서 고정.
#   - 팔(7 DOF) 은 Tesollo 와 동일 → 팔/palm/충돌 구조 재사용.
#   - 기본적으로 env 는 use_hand_fabric=False 로 사용 (손은 직접 PD 제어).
#     add_hand_fabric 은 6D 직접(identity) 매핑으로 제공만 해 둔다.

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

NUM_ARM_DOF = 7
NUM_HAND_DOF = 6
NUM_DOF = NUM_ARM_DOF + NUM_HAND_DOF  # 13

# fingertip FK 프레임 (fabrics URDF 기준)
TIP_FRAMES = [
    "rh56f1_tip_thumb",
    "rh56f1_tip_index",
    "rh56f1_tip_middle",
    "rh56f1_tip_ring",
    "rh56f1_tip_little",
]


class OpenArmRh56f1PoseFabric(BaseFabric):
    """Fabric for OpenArm right arm (7 DOF) + RH56F1 right hand (6 DOF) = 13 DOF total.

    Joint order in fabrics URDF (13 revolute joints):
      [0-6]  openarm_right_joint1~7        (arm)
      [7]    rh56f1_right_right_thumb_1     (thumb abduction, 0~2.094)
      [8]    rh56f1_right_right_thumb_2     (thumb flex drive, 0~0.475)
      [9]    rh56f1_right_right_index_1     (index flex drive, 0~1.529)
      [10]   rh56f1_right_right_middle_1    (middle flex drive)
      [11]   rh56f1_right_right_ring_1      (ring flex drive)
      [12]   rh56f1_right_right_little_1    (little flex drive)
    """

    def __init__(self, batch_size, device, timestep, graph_capturable=True, use_hand_fabric=False):
        self._use_hand_fabric = use_hand_fabric
        fabric_params_filename = "openarm_rh56f1_pose_params.yaml"
        super().__init__(device, batch_size, timestep, fabric_params_filename,
                         graph_capturable=graph_capturable)

        robot_dir_name = "openarm_rh56f1"
        robot_name = "openarm_rh56f1"
        self.urdf_path = get_robot_urdf_path(robot_dir_name, robot_name)

        self.load_robot(robot_dir_name, robot_name, batch_size)

        # Default cspace config (13 DOF):
        #   arm: Tesollo 와 동일한 자연 작업 자세
        #   hand: 약한 grasp 자세 (drive 관절 기준)
        default_config = torch.tensor([
            # OpenArm right arm joint1~7
            1.0,  -0.1,  -0.6,  0.5,  0.0,  0.0,  0.0,
            # RH56F1 hand drive 6:
            #   thumb_1(abduction): 0.6 (opposition 방향)
            #   thumb_2(flex drive): 0.40
            #   index/middle/ring/little_1(flex): 0.90
            0.6,  0.40,  0.90,  0.90,  0.90,  0.90,
        ], device=self.device)
        self.default_config = default_config.unsqueeze(0).repeat(self.batch_size, 1)

        self._pca_matrix = None

        self.construct_fabric()

        # Palm pose target tensor (b x 12): 3D origin + 9D rotation matrix
        self._palm_pose_target = torch.zeros(batch_size, 12, device=device)
        default_palm_euler = torch.tensor([1.5708, 0.0, 1.5708], device=self.device).unsqueeze(0)
        default_palm_euler = default_palm_euler.repeat(self.batch_size, 1)
        self._palm_pose_target[:, 3:] = torch.transpose(
            euler_to_matrix(default_palm_euler), 1, 2
        ).reshape(self.batch_size, 9)
        self._native_palm_pose_target = None

        # Fingertip FK taskmap (sim2real 관측용)
        self._fingertip_taskmap = RobotFrameOriginsTaskMap(
            self.urdf_path, TIP_FRAMES, batch_size, device
        )

    # ------------------------------------------------------------------
    # Fabric construction
    # ------------------------------------------------------------------
    def add_joint_limit_repulsion(self):
        joints = self.urdfpy_robot.joints
        upper_joint_limits = []
        lower_joint_limits = []
        for j in joints:
            if j.joint_type == 'revolute':
                upper_joint_limits.append(j.limit.upper)
                lower_joint_limits.append(j.limit.lower)

        taskmap_name = "upper_joint_limit"
        taskmap = UpperJointLimitMap(upper_joint_limits, self.batch_size, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)
        fabric = JointLimitRepulsion(True, self.fabric_params['joint_limit_repulsion'],
                                     self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, "joint_limit_repulsion", fabric)

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
        """6D 직접 손 제어 (identity). env 가 use_hand_fabric=True 로 쓸 때만 활성.
        보통 env 는 손을 직접 PD 제어하므로 사용하지 않는다."""
        # 6x6 identity, 7 arm 컬럼은 0 패딩 → (6, 13)
        hand_map = torch.cat(
            [torch.zeros(NUM_HAND_DOF, NUM_ARM_DOF, device=self.device),
             torch.eye(NUM_HAND_DOF, device=self.device)], dim=1
        )
        self._pca_matrix = torch.clone(hand_map.detach())
        taskmap_name = "pca_hand"
        taskmap = LinearMap(hand_map, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)
        fabric = Attractor(True, self.fabric_params['hand_attractor'],
                           self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, "hand_attractor", fabric)

    def add_palm_points_attractor(self):
        taskmap_name = "palm"
        control_point_frames = [
            "palm_link",
            "palm_x", "palm_x_neg",
            "palm_y", "palm_y_neg",
            "palm_z", "palm_z_neg",
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
    # Runtime (Tesollo 와 동일)
    # ------------------------------------------------------------------
    def convert_transform_to_points(self):
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
        palm_targets[:, :3] = self._palm_pose_target[:, :3]
        palm_targets[:, 3:6]   = torch.bmm(palm_transform, _axis_point([0.25, 0., 0.]).unsqueeze(2)).squeeze(2)[:, :3]
        palm_targets[:, 6:9]   = torch.bmm(palm_transform, _axis_point([-0.25, 0., 0.]).unsqueeze(2)).squeeze(2)[:, :3]
        palm_targets[:, 9:12]  = torch.bmm(palm_transform, _axis_point([0., 0.25, 0.]).unsqueeze(2)).squeeze(2)[:, :3]
        palm_targets[:, 12:15] = torch.bmm(palm_transform, _axis_point([0., -0.25, 0.]).unsqueeze(2)).squeeze(2)[:, :3]
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
        """FK로 5개 손가락 끝 위치 계산 → (B, 5, 3).
          [0]=thumb [1]=index [2]=middle [3]=ring [4]=little
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
        """Pass input features to fabric terms.

        Args:
            hand_target:              (B, 6)  직접 손 target (use_hand_fabric=True 일 때만)
            palm_pose_target:         (B, 6) euler_zyx  또는 (B, 7) quaternion
            batched_cspace_position:  (B, 13) current joint positions
            batched_cspace_velocity:  (B, 13) current joint velocities
        """
        if "pca_hand" in self.fabrics_features:
            self.fabrics_features["pca_hand"]["hand_attractor"] = hand_target
        self.fabrics_features["identity"]["cspace_attractor"] = self.default_config

        self._palm_pose_target[:, :3] = palm_pose_target[:, :3]

        if orientation_convention == "euler_zyx":
            assert palm_pose_target.shape[1] == 6, "euler_zyx pose target must be (B, 6)"
            self._palm_pose_target[:, 3:] = torch.transpose(
                euler_to_matrix(palm_pose_target[:, 3:]), 1, 2
            ).reshape(self.batch_size, 9)
        elif orientation_convention == "quaternion":
            assert palm_pose_target.shape[1] == 7, "quaternion pose target must be (B, 7)"
            self._palm_pose_target[:, 3:] = torch.transpose(
                quaternion_to_matrix(palm_pose_target[:, [6, 3, 4, 5]]), 1, 2
            ).reshape(self.batch_size, 9)
        else:
            raise ValueError('orientation_convention must be "euler_zyx" or "quaternion"')

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

        body_point_pos, jac = self.get_taskmap("body_points")(batched_cspace_position, None)
        body_point_vel = torch.bmm(jac, batched_cspace_velocity.unsqueeze(2)).squeeze(2)

        self.base_fabric_repulsion.calculate_response(
            body_point_pos, body_point_vel, object_ids, object_indicator
        )

        self.fabrics_features["body_points"]["repulsion"] = self.base_fabric_repulsion
        self.fabrics_features["body_points"]["geom_repulsion"] = self.base_fabric_repulsion

        if cspace_damping_gain is not None:
            self.fabric_params['cspace_damping']['gain'] = cspace_damping_gain

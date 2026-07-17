# OpenArm + RH56F1 bimanual pose fabric (_rl 소스 단일 기준)
# Based on openarm_tesollo_pose_fabric.py
# 양팔: (arm 7 + hand drive 6) x 2 = 26 DOF total
#
# 설계 메모 (Phase 2 — 양팔 인프라):
#   - cspace 순서 = [r_arm 7, r_hand 6, l_arm 7, l_hand 6] (URDF revolute 정의순).
#   - RH56F1 손은 underactuated 6 DOF (drive): thumb_1, thumb_2, index_1,
#     middle_1, ring_1, pinky_1. mimic 추종 관절은 fabrics URDF 에서 고정.
#   - palm IK(control point)는 지금은 오른손(r_hl_palm_sensor)만 활성.
#     왼팔/왼손은 cspace_attractor 로 default_config 중립만 유지("인프라만" —
#     왼손 능동 palm 제어는 이후 phase 에서 add_palm_points_attractor(l) 로 확장).
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
NUM_SIDE_DOF = NUM_ARM_DOF + NUM_HAND_DOF  # 13 (한 팔)
NUM_DOF = 2 * NUM_SIDE_DOF  # 26 (양팔)

# cspace 슬라이스(26 DOF 중): [0:7]=r_arm, [7:13]=r_hand, [13:20]=l_arm, [20:26]=l_hand
R_ARM_SLICE = slice(0, NUM_ARM_DOF)
R_HAND_SLICE = slice(NUM_ARM_DOF, NUM_SIDE_DOF)
L_ARM_SLICE = slice(NUM_SIDE_DOF, NUM_SIDE_DOF + NUM_ARM_DOF)          # [13:20]
L_HAND_SLICE = slice(NUM_SIDE_DOF + NUM_ARM_DOF, NUM_DOF)             # [20:26]

# fingertip FK 프레임 (_rl URDF 기준). obs 전용.
TIP_FRAMES = [
    "r_hl_thumb_tip",
    "r_hl_index_tip",
    "r_hl_middle_tip",
    "r_hl_ring_tip",
    "r_hl_pinky_tip",
]
LEFT_TIP_FRAMES = [
    "l_hl_thumb_tip",
    "l_hl_index_tip",
    "l_hl_middle_tip",
    "l_hl_ring_tip",
    "l_hl_pinky_tip",
]

# grasp_v2 (DEXTRAH 물체파지 이식) 용 오른손 PCA5 basis.
# inspire grasp 시연 → RH56F1 6-drive remap → PCA5 (scripts/pca/compute_rh56f1_grasp_pca.py).
# rows = PC1~5 방향(6-drive 공간). PC1(97.9%)=엄지+4손가락 조율 닫힘=firm envelope 시너지.
# kuka_allegro_pose_fabric 의 pca_matrix(5x16) 와 동일 역할: LinearMap 이 cspace q 를
# 이 5개 방향으로 투영(mean 미차감, uncentered) → 5D PCA 공간 attractor.
# 출처: assets/demograsp_references/rh56f1_grasp_pca5.pt (단일 진실원=위 스크립트).
NUM_HAND_PCA = 5
RH56F1_HAND_PCA_MATRIX = [
    [ 5.055892e-01,  1.252147e-01,  4.233449e-01,  4.358092e-01,  4.029488e-01,  4.440524e-01],
    [ 8.196450e-01, -7.233454e-02, -2.688281e-01, -4.071321e-01,  3.556392e-02, -2.892402e-01],
    [-1.990037e-01, -2.065245e-01, -4.532336e-01, -3.136647e-01,  6.543947e-01,  4.309368e-01],
    [ 1.482936e-01,  1.309842e-01, -2.630515e-01, -1.082689e-01, -6.144237e-01,  7.088141e-01],
    [ 1.037749e-01, -5.117328e-01, -4.972013e-01,  6.815110e-01, -9.081748e-02, -8.629034e-02],
]


class OpenArmRh56f1PoseFabric(BaseFabric):
    """Bimanual fabric: (OpenArm 7 + RH56F1 hand drive 6) x 2 = 26 DOF total.

    Joint order in fabrics URDF (26 revolute joints, _rl 정의순):
      [0-6]   r_aj_1~7                 (right arm)
      [7]     r_hj_thumb_1             (thumb abduction)
      [8]     r_hj_thumb_2             (thumb flex drive)
      [9-12]  r_hj_{index,middle,ring,pinky}_1  (right hand flex drives)
      [13-19] l_aj_1~7                 (left arm)
      [20]    l_hj_thumb_1
      [21]    l_hj_thumb_2
      [22-25] l_hj_{index,middle,ring,pinky}_1  (left hand flex drives)
    """

    def __init__(self, batch_size, device, timestep, graph_capturable=True,
                 use_hand_fabric=False, hand_mode="direct", side="right"):
        # hand_mode: "direct" = 6D identity 직접 제어(기존, grasp_v1),
        #            "pca"    = 5D PCA action(grasp_v2 물체파지, DEXTRAH 방식).
        # side: "right" = 오른팔 palm IK 활성(왼팔 rest), "left" = 좌우 미러.
        assert hand_mode in ("direct", "pca"), f"invalid hand_mode: {hand_mode}"
        assert side in ("right", "left"), f"invalid side: {side}"
        self._use_hand_fabric = use_hand_fabric
        self._hand_mode = hand_mode
        self._side = side
        # side 별 제어점/슬라이스/FK 프레임(오른손 기본, 왼손 미러). construct_fabric 전에 설정.
        if side == "right":
            self._palm_frame = "r_hl_palm_sensor"
            self._arm_slice, self._hand_slice = R_ARM_SLICE, R_HAND_SLICE
            self._tip_frames = TIP_FRAMES
        else:
            self._palm_frame = "l_hl_palm_sensor"
            self._arm_slice, self._hand_slice = L_ARM_SLICE, L_HAND_SLICE
            self._tip_frames = LEFT_TIP_FRAMES
        fabric_params_filename = "openarm_rh56f1_pose_params.yaml"
        super().__init__(device, batch_size, timestep, fabric_params_filename,
                         graph_capturable=graph_capturable)

        robot_dir_name = "openarm_rh56f1"
        robot_name = "openarm_rh56f1"
        self.urdf_path = get_robot_urdf_path(robot_dir_name, robot_name)

        self.load_robot(robot_dir_name, robot_name, batch_size)

        # Default cspace config (26 DOF): [r_arm7, r_hand6, l_arm7, l_hand6]
        #   활성 팔(side): 자연 작업 자세 + 약한 grasp / 비활성 팔: rest + open.
        #   비활성 팔은 능동 IK 없이 이 cspace target 으로만 유지된다("인프라만").
        # 미러 규칙(arm sign [-1,-1,-1,1,-1,-1,-1]): 좌팔 grasp/rest = 우팔값 부호 매핑.
        _ARM_GRASP_R = [ 1.0, -0.1, -0.6,  0.5,  0.0,  0.0,  0.0]  # 우팔 자연 작업 자세
        _ARM_REST_R  = [ 0.315,  0.290, -0.400, 0.513, -0.666, 0.729, 0.957]  # 우팔 rest(좌rest 미러)
        _ARM_GRASP_L = [-1.0,  0.1,  0.6,  0.5,  0.0,  0.0,  0.0]  # 좌팔 grasp(우grasp 미러)
        _ARM_REST_L  = [-0.315, -0.290, 0.400, 0.513, 0.666, -0.729, -0.957]  # 좌팔 rest(grasp_v1 정합)
        _HAND_GRASP  = [0.6, 0.40, 0.90, 0.90, 0.90, 0.90]  # drive 6: thumb_1 abd, thumb_2 flex, 4지 flex
        _HAND_OPEN   = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if side == "right":
            _cfg = _ARM_GRASP_R + _HAND_GRASP + _ARM_REST_L + _HAND_OPEN
        else:
            _cfg = _ARM_REST_R + _HAND_OPEN + _ARM_GRASP_L + _HAND_GRASP
        default_config = torch.tensor(_cfg, device=self.device)
        self.default_config = default_config.unsqueeze(0).repeat(self.batch_size, 1)

        self._pca_matrix = None

        self.construct_fabric()

        # Palm pose target tensor (b x 12): 3D origin + 9D rotation matrix
        # NOTE: control point 를 r_hl_palm_sensor 로 옮긴 뒤로 이 euler 는 palm_sensor 프레임
        # 기준의 초기 placeholder 다. 실제 target 은 env 가 reset/step 에서 palm_sensor pose 로
        # 직접 공급(set_features)하므로 이 초기값은 첫 set_features 전까지만 유효.
        self._palm_pose_target = torch.zeros(batch_size, 12, device=device)
        default_palm_euler = torch.tensor([1.5708, 0.0, 1.5708], device=self.device).unsqueeze(0)
        default_palm_euler = default_palm_euler.repeat(self.batch_size, 1)
        self._palm_pose_target[:, 3:] = torch.transpose(
            euler_to_matrix(default_palm_euler), 1, 2
        ).reshape(self.batch_size, 9)
        self._native_palm_pose_target = None

        # Fingertip FK taskmap (sim2real 관측용). side 별 tip 프레임.
        self._fingertip_taskmap = RobotFrameOriginsTaskMap(
            self.urdf_path, self._tip_frames, batch_size, device
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
        """오른손 fabric taskmap "pca_hand". env 가 use_hand_fabric=True 로 쓸 때만 활성.

        hand_mode="direct": (6, 26) identity — 6D 직접 오른손 제어(grasp_v1 계열, 미사용 기본).
        hand_mode="pca":    (5, 26) PCA — 5D PCA action(grasp_v2 물체파지, DEXTRAH 방식).
                            오른손 drive 6열에 PCA basis(5x6), 나머지(팔·왼쪽) 0 패딩.
        어느 모드든 taskmap 이름은 "pca_hand", attractor 는 "hand_attractor" 로 동일 —
        set_features(hand_target) 가 hand_target(direct=B×6 / pca=B×5)을 그 attractor 로 넣는다.
        """
        if self._hand_mode == "pca":
            # (5, 26): kuka_allegro 와 동일 — PCA basis 를 오른손 drive 컬럼에 배치, 팔은 0 으로 소거.
            pca_basis = torch.tensor(RH56F1_HAND_PCA_MATRIX, device=self.device)  # (5, 6)
            hand_map = torch.zeros(NUM_HAND_PCA, NUM_DOF, device=self.device)
            hand_map[:, self._hand_slice] = pca_basis
        else:
            # (6, 26): 오른손 drive 컬럼 [7:13] 만 eye, 나머지(팔·왼쪽) 0 패딩.
            hand_map = torch.zeros(NUM_HAND_DOF, NUM_DOF, device=self.device)
            hand_map[:, self._hand_slice] = torch.eye(NUM_HAND_DOF, device=self.device)
        self._pca_matrix = torch.clone(hand_map.detach())
        taskmap_name = "pca_hand"
        taskmap = LinearMap(hand_map, self.device)
        self.add_taskmap(taskmap_name, taskmap, graph_capturable=self.graph_capturable)
        fabric = Attractor(True, self.fabric_params['hand_attractor'],
                           self.device, graph_capturable=self.graph_capturable)
        self.add_fabric(taskmap_name, "hand_attractor", fabric)

    def add_palm_points_attractor(self):
        taskmap_name = "palm"
        # 실제 손바닥 센서 링크 r_hl_palm_sensor + 그 로컬 축점 6개를 IK control point 로.
        # (Tesollo palm_link 가상프레임은 실 palm_sensor 와 위치 3.4cm·자세 90° 어긋나 제거됨.)
        # 정책 6D pose = palm_sensor pose 직접(env 의 offset 변환 소멸).
        _s = "r" if self._side == "right" else "l"
        control_point_frames = [
            self._palm_frame,
            f"ps_{_s}_x", f"ps_{_s}_x_neg",
            f"ps_{_s}_y", f"ps_{_s}_y_neg",
            f"ps_{_s}_z", f"ps_{_s}_z_neg",
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
            hand_target:              use_hand_fabric=True 일 때만. hand_mode="direct"=(B,6) 직접
                                      drive target / hand_mode="pca"=(B,5) PCA action target
            palm_pose_target:         (B, 6) euler_zyx  또는 (B, 7) quaternion (오른손 palm)
            batched_cspace_position:  (B, 26) current joint positions [r_arm,r_hand,l_arm,l_hand]
            batched_cspace_velocity:  (B, 26) current joint velocities
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


class OpenArmRh56f1LeftPoseFabric(OpenArmRh56f1PoseFabric):
    """좌팔 미러: 동일 bi-arm URDF(openarm_rh56f1)에서 왼손 palm IK 활성.

    base 클래스를 side="left" 로 인스턴스화 — palm 제어점 l_hl_palm_sensor + ps_l_*,
    hand slice [20:26], FK tip = l_hl_*_tip, default_config 좌활성/우rest.
    tesollo OpenArmTeoslloLeftPoseFabric 과 동일 역할(단 별도 미러 URDF 불필요 —
    rh56f1 bi-arm URDF 에 양팔·양 palm 축점이 이미 존재).
    """

    def __init__(self, batch_size, device, timestep, graph_capturable=True,
                 use_hand_fabric=False, hand_mode="direct"):
        super().__init__(batch_size, device, timestep,
                         graph_capturable=graph_capturable,
                         use_hand_fabric=use_hand_fabric, hand_mode=hand_mode,
                         side="left")

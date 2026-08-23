# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.                          
                                                                                                     
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual                           
# property and proprietary rights in and to this material, related                                   
# documentation and any modifications thereto. Any use, reproduction,                                
# disclosure or distribution of this material and related documentation                              
# without an express license agreement from NVIDIA CORPORATION or                                    
# its affiliates is strictly prohibited.

"""
Implements a map to a 3D point on the robot body.
"""

import os
import torch

import warp as wp
import warp.torch

from fabrics_sim.prod.kinematics import Kinematics
from fabrics_sim.taskmaps.maps_base import BaseMap

# Define PyTorch autograd op to wrap foward kinematics
# function.
class RobotKinematics(torch.autograd.Function):

    @staticmethod
    def forward(ctx, joint_q, robot_kinematics):

        # Hold onto recording of kernel launches.
        ctx.tape = wp.Tape()

        # Hold onto inputs and outputs
        ctx.joint_q = wp.torch.from_torch(joint_q)
        ctx.robot_kinematics = robot_kinematics
        
        with ctx.tape:
            ctx.robot_kinematics.eval(ctx.joint_q, jacobians=True)
            #ctx.robot_kinematics.eval(ctx.joint_q, batch_qd=ctx.joint_q, velocities=True, jacobians=True)
        
        return (wp.torch.to_torch(ctx.robot_kinematics.batch_link_transforms),
                wp.torch.to_torch(ctx.robot_kinematics.batch_link_jacobians))

    @staticmethod
    def backward(ctx, adj_link_transforms, adj_jacobians):

        # Map incoming Torch grads to our output variables
        grads = { ctx.robot_kinematics.batch_link_transforms:
                      wp.torch.from_torch(adj_link_transforms, dtype=wp.transform),
                  ctx.robot_kinematics.batch_link_jacobians:
                      wp.torch.from_torch(adj_jacobians, dtype=wp.vec3) }

        # Calculate gradients
        ctx.tape.zero()
        ctx.tape.backward(grads=grads)

        # Return adjoint w.r.t. inputs
        return (wp.torch.to_torch(ctx.tape.gradients[ctx.joint_q]),
                None,
                None)

class RobotFrameOriginsTaskMap(BaseMap):
    def __init__(self, urdf_path, link_names, batch_size, device):
        """
        Constructor for building the desired robot taskmap.
        -----------------------------------------
        :param urdf_path: str, robot URDF filepath
        :param link_names: list of link names (str) of the robot to build the taskmap
        :param batch_size: int, size of the batch of robots
        :param device: type str that sets the cuda device for the fabric
        """
        super().__init__(device)

        # Allocate for robot kinemtics, the relevant link indices, and the batch size.
        self.urdf_path = urdf_path
        self.robot_kinematics = None
        self.link_names = link_names
        self.link_indices = None
        self.batch_size = batch_size

        self.init_robot_kinematics(self.batch_size)

    def init_robot_kinematics(self, batch_size):
        # Create the robot kinematics object that wraps several Warp kernels for computing
        # forward kinematics
        multithreading = False
        self.robot_kinematics = Kinematics(self.urdf_path, batch_size, multithreading,
                                           device=self.device)

        self.link_indices =  []
        for link_name in self.link_names:
            self.link_indices.append(self.robot_kinematics.get_link_index(link_name))
        self.link_indices = torch.tensor(self.link_indices, device=self.device)

        self.batch_size = batch_size

    def forward_position(self, q, features):
        # Check if the batch size matches the batch size of the incoming q. If not,
        # then re-initialize the robots kinematics.
#        if self.batch_size != q.shape[0]:
#            self.init_robot_kinematics(q.shape[0])

        # Calculate the link transforms and their origin Jacobians.
        link_transforms, jacobians = RobotKinematics.apply(q, self.robot_kinematics)

        # Pull out the position of the origins and stack them across all desired frames.
        x = link_transforms[:, self.link_indices, :3].reshape((self.batch_size,
                                                               len(self.link_indices) * 3))

        # Pull out the Jacobians and stack them for the desired frames.
        # jacobian is of shape (batch_size, num_links, root_dim, 3)
        # so we transpose the last two dimensions to get a
        # jacobian of shape (batch_size, num_links, 3, root_dim)
        # and then reshape it to (batch_size, num_links * 3, root_dim)
        jacobian = jacobians[:, self.link_indices, :, :].transpose(2,3).reshape(
                        self.batch_size, len(self.link_indices) * 3, q.shape[1])

        return (x, jacobian)






class SubchainFrameOriginsTaskMap(RobotFrameOriginsTaskMap):
    """프레임 원점 taskmap 인데 **지정한 관절 구간만** 움직이게 한다.

    왜 필요한가: fabric 의 항들은 모두 같은 cspace 에 힘을 준다. 손끝 attractor 의
    Jacobian 은 팔 열까지 포함하므로, 손끝 목표가 손만으로 도달 불가하면 fabric 은
    **팔을 움직여서라도** 손끝을 목표로 보낸다. palm attractor 와 손끝 attractor 가
    같은 팔 관절을 두고 싸우고, 게인이 큰 쪽이 이긴다.

    실측(OpenArm+Tesollo, 손끝 목표를 컵 표면에): 손끝 게인 400 / palm 게인 80 일 때
    palm 추종오차가 **580mm** 까지 벌어져 파지 자체가 성립하지 않았다. 게인을 palm
    이하(40~80)로 낮추면 palm 은 3.6mm 로 돌아오지만 이번엔 손이 3.3° 밖에 안 움직인다.
    게인으로는 못 푸는 딜레마다 — 팔과 손은 **별개 제어 대상**이기 때문이다.

    Jacobian 의 팔 열을 0 으로 두면 이 항이 만드는 가속이 손 관절에만 분배된다.
    팔은 palm attractor 가, 손은 손끝 attractor 가 담당하는 원래 의도대로 돌아간다.
    도달 불가한 손끝 목표는 오차로 남을 뿐 팔을 끌고 가지 않는다.
    """

    def __init__(self, urdf_path, link_names, batch_size, device, joint_slice):
        """joint_slice: 이 taskmap 이 움직여도 되는 관절 구간(slice). 그 밖은 마스킹된다."""
        super().__init__(urdf_path, link_names, batch_size, device)
        self._joint_slice = joint_slice

    def forward_position(self, q, features):
        x, jacobian = super().forward_position(q, features)
        mask = torch.zeros_like(jacobian)
        mask[:, :, self._joint_slice] = 1.0
        return (x, jacobian * mask)

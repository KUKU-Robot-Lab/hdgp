# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""상수 정의: 5g_pour_right_v3

v3: Fluid Particle System (PhysX 5 PBD) + Cost-Benefit Tradeoff Reward + Warmstart v8

v2 대비 변경:
  - 액체 표현: Rigid Bead → PBD Fluid Particle (200개, 5mm, 1g)
  - 리워드: 복합 shaping → R = w_A * A - w_T * T - w_E * E + 보조 shaping
  - Warmstart: v7 (106D) → v8 (107D, bead_mass_normalized=1.0 고정)

Action (11D):
  [0:6]  6D palm pose (x,y,z,ez,ey,ex) → Fabrics IK → arm 7 DOF
  [6:11] 5D per-finger lerp (freeze_grasp=True → 항상 grasp_hold)

Actor Observation (101D) - v2와 동일:
  arm_joint_pos:            7
  arm_joint_vel:            7
  finger_joint_pos:        20
  finger_joint_vel:        20
  right_cup_pos_rel_palm:   3
  right_cup_quat:           4
  left_cup_pos_rel_palm:    3
  left_cup_quat:            4
  pour_point_to_opening:    3
  source_pour_axis:         3
  source_up_axis:           3
  target_up_axis:           3
  transport_summary:        5
  fingertip_contact_binary: 5
  last_actions:            11
  Total:                  101

Critic Extra (42D) - v2와 동일 구조 (bead→fluid centroid로 내용 교체):
  left_arm_joint_pos:       9
  left_arm_joint_vel:       9
  distal_contact_binary:    5
  distal_contact_norm:      5
  cup_height_delta:         1
  fluid_centroid_rel_source: 3  (bead_pos_rel_source_cup → fluid centroid)
  fluid_centroid_rel_target: 3  (bead_pos_rel_target_cup → fluid centroid)
  mouth_distance:            1
  mouth_xy_distance:         1
  mouth_z_clearance:         1
  source_up_dot_world:       1
  directional_tilt_cos:      1
  mouth_alignment_cos:       1
  fluid_accuracy:            1  (bead_cross_fraction → fluid in target fraction)
  Total:                    42

Critic Total: 101 + 42 = 143D

Warmstart v8 obs (107D):
  기존 106D + bead_mass_normalized 1D (항상 1.0 = max 하중)

Episode (6s @ 60Hz = 360 steps):
  Pour phase (0~359): Fabrics arm policy + frozen hand

Reward: Cost-Benefit Tradeoff
  R = w_A * A(t) - w_T * T(t) - w_E * E(t)
      + w_transport * transport + w_tilt * tilt - w_spill * spill
      + w_force_balance * force_balance + w_success * success
      - w_action_rate * action_rate
"""

import math

from .grasp_right_preset import (
    RIGHT_ARM_JOINT_NAMES,
    RIGHT_HAND_JOINT_NAMES,
    palm_pose_mins,
    palm_pose_maxs,
    RIGHT_ARM_START_POSE,
)

# ---------------------------------------------------------------------------
# Robot dimensions
# ---------------------------------------------------------------------------
NUM_ARM_DOF    = len(RIGHT_ARM_JOINT_NAMES)    # 7
NUM_HAND_DOF   = len(RIGHT_HAND_JOINT_NAMES)   # 20
NUM_ROBOT_DOF  = NUM_ARM_DOF + NUM_HAND_DOF     # 27
NUM_FINGERTIPS = 5
NUM_DISTAL_SENSORS = 5
NUM_MIDDLE_SENSORS = 5

# ---------------------------------------------------------------------------
# Action / Observation space
# ---------------------------------------------------------------------------
NUM_PALM_ACTION   = 6
NUM_FINGER_ACTION = 5
NUM_ACTIONS = NUM_PALM_ACTION + NUM_FINGER_ACTION  # 11

NUM_OBSERVATIONS         = 101   # Actor: sim2real (v2와 동일)
NUM_CRITIC_EXTRAS        = 42
NUM_CRITIC_OBSERVATIONS  = NUM_OBSERVATIONS + NUM_CRITIC_EXTRAS  # 143

# v8 warmstart: 기존 106D + bead_mass_normalized 1D
NUM_LEGACY_WARMSTART_OBS = 107

# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------
POUR_EPISODE_STEPS = 360    # 6s @ 60Hz
EPISODE_STEPS      = POUR_EPISODE_STEPS

# ---------------------------------------------------------------------------
# Contact
# ---------------------------------------------------------------------------
CONTACT_FORCE_THRESHOLD  = 0.1    # N
CONTACT_FORCE_MAX        = 10.0   # N
MIN_CONTACTS_FOR_SUCCESS = 2

# ---------------------------------------------------------------------------
# Fabrics
# ---------------------------------------------------------------------------
PREGRASP_FABRICS_STEPS = 60

# ---------------------------------------------------------------------------
# Cup geometry
# ---------------------------------------------------------------------------
CUP_RADIUS_APPROX = 0.045  # m

# ---------------------------------------------------------------------------
# Fluid Particle (PhysX 5 PBD)
# ---------------------------------------------------------------------------
PARTICLES_PER_ENV  = 200         # 에피소드당 유체 입자 수 (4×5×10 grid)
PARTICLE_DIAMETER  = 0.005       # 5mm diameter
PARTICLE_RADIUS    = PARTICLE_DIAMETER / 2.0
PARTICLE_MASS      = 0.001       # 1g per particle → total 200g per cup

# PBD 물리 파라미터 (Viscosity: 학습 안정성 위해 다소 높게 → 이후 curriculum으로 낮춤)
PARTICLE_VISCOSITY       = 0.3
PARTICLE_SURFACE_TENSION = 0.4
PARTICLE_COHESION        = 0.2
PARTICLE_ADHESION        = 0.05

# Particle system PhysX paths
PARTICLE_SYSTEM_PATH  = "/World/FluidSystem"
PARTICLE_SET_TEMPLATE = "/World/envs/env_0/FluidParticles"
PARTICLE_SET_PATTERN  = "/World/envs/env_*/FluidParticles"

# ---------------------------------------------------------------------------
# Aliases (v2 호환)
# ---------------------------------------------------------------------------
ARM_START_POSE      = RIGHT_ARM_START_POSE
PALM_POSE_MINS_FUNC = palm_pose_mins
PALM_POSE_MAXS_FUNC = palm_pose_maxs

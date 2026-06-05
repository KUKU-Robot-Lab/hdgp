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

import os

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, RigidObjectCollectionCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UrdfFileCfg
from isaaclab.utils import configclass

from .. import mdp
from ..grasp_env_cfg import GraspEnvCfg

TASK_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UNIDEX_OBJECT_CODE = "sem/Car-669043a8ce40d9d78781f76a6db4ab62"
UNIDEX_SCALES = ["006", "008", "010", "012", "015"]

UNIDEX_ASSET_DIR = os.environ.get(
    "UNIDEXGRASP_ASSET_DIR",
    os.path.join(TASK_ROOT_DIR, "..", "..", "assets", "unidexgrasp_assets"),
)
UNIDEX_MESH_DIR = os.path.join(UNIDEX_ASSET_DIR, "meshdatav3_scaled", UNIDEX_OBJECT_CODE, "coacd")
UNIDEX_FEAT_DIR = os.path.join(UNIDEX_ASSET_DIR, "meshdatav3_pc_feat", UNIDEX_OBJECT_CODE)
UNIDEX_POSEDATA_PATH = os.environ.get(
    "UNIDEXGRASP_POSEDATA", os.path.join(UNIDEX_ASSET_DIR, "datasetv4.1_posedata.npy")
)

UNIDEX_OBJECT_NAMES = [f"car_{scale}" for scale in UNIDEX_SCALES]
UNIDEX_URDF_PATHS = [os.path.join(UNIDEX_MESH_DIR, f"coacd_{scale}.urdf") for scale in UNIDEX_SCALES]
UNIDEX_PC_FEAT_PATHS = [os.path.join(UNIDEX_FEAT_DIR, f"pc_feat_{scale}.npy") for scale in UNIDEX_SCALES]
UNIDEX_SCALE_VALUES = [float(scale) / 100.0 for scale in UNIDEX_SCALES]


@configclass
class OpenArmGraspEnvCfg(GraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{TASK_ROOT_DIR}/../../usds/openarm_bimanual/openarm_tesollo_t2.usd",
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=5.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=[0.5, -0.3, 0.02],
                rot=[0.707, 0, 0, 0.707],
                joint_pos={
                    "openarm_right_joint1": 0.0,
                    "openarm_right_joint2": 0.0,
                    "openarm_right_joint3": 0.0,
                    "openarm_right_joint4": 0.0,
                    "openarm_right_joint5": 0.0,
                    "openarm_right_joint6": 0.0,
                    "openarm_right_joint7": 0.0,
                    "rj_dg_.*": 0.0,
                },
            ),
            actuators={
                "openarm_arm": ImplicitActuatorCfg(
                    joint_names_expr=["openarm_right_joint[1-7]"],
                    stiffness=400.0,
                    damping=80.0,
                ),
                "openarm_gripper": ImplicitActuatorCfg(
                    joint_names_expr=["rj_dg_.*"],
                    stiffness=2e3,
                    damping=1e2,
                ),
            },
            soft_joint_pos_limit_factor=1.0,
        )

        rigid_objects = {}
        for name, urdf_path in zip(UNIDEX_OBJECT_NAMES, UNIDEX_URDF_PATHS):
            rigid_objects[name] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Object_{name}",
                init_state=RigidObjectCfg.InitialStateCfg(pos=[0.4, 0.0, 0.055], rot=[1, 0, 0, 0]),
                spawn=UrdfFileCfg(
                    asset_path=urdf_path,
                    fix_base=False,
                    joint_drive=None,
                    rigid_props=RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_angular_velocity=1000.0,
                        max_linear_velocity=1000.0,
                        max_depenetration_velocity=5.0,
                        disable_gravity=False,
                    ),
                ),
            )

        self.scene.objects = RigidObjectCollectionCfg(rigid_objects=rigid_objects)

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "openarm_right_joint1",
                "openarm_right_joint2",
                "openarm_right_joint3",
                "openarm_right_joint4",
                "openarm_right_joint5",
                "openarm_right_joint6",
                "openarm_right_joint7",
            ],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.hand_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["rj_dg_.*"],
            scale=0.5,
            use_default_offset=True,
        )

                self.observations.policy.object.params["left_eef_link_name"] = None
        self.observations.policy.object.params["right_eef_link_name"] = "tesollo_right_rl_dg_1_4"
        self.rewards.object_distance.params["left_eef_link_name"] = None
        self.rewards.object_distance.params["right_eef_link_name"] = "tesollo_right_rl_dg_1_4"
        self.rewards.object_contact.params["left_eef_link_name"] = None
        self.rewards.object_contact.params["right_eef_link_name"] = "tesollo_right_rl_dg_1_4"
        self.rewards.grasp_success.params["left_eef_link_name"] = None
        self.rewards.grasp_success.params["right_eef_link_name"] = "tesollo_right_rl_dg_1_4"

        self.events.load_pc_feat.params = {
            "pc_feat_paths": UNIDEX_PC_FEAT_PATHS,
            "object_names": UNIDEX_OBJECT_NAMES,
        }
        self.events.load_grasp_prior.params = {
            "posedata_path": UNIDEX_POSEDATA_PATH,
            "object_code": UNIDEX_OBJECT_CODE,
            "scale_values": UNIDEX_SCALE_VALUES,
        }


@configclass
class OpenArmGraspEnvCfg_PLAY(OpenArmGraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

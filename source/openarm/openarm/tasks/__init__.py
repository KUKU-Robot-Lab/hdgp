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

##
# Register Gym environments.
##

import importlib
import sys
from pathlib import Path

from isaaclab_tasks.utils import import_packages

# Prefer vendored FABRICS under this repository to avoid external path dependency.
_SRC_ROOT = Path(__file__).resolve().parents[3]
_VENDORED_FABRICS_SRC = _SRC_ROOT / "FABRICS" / "src"
if _VENDORED_FABRICS_SRC.exists():
    vendored_path = str(_VENDORED_FABRICS_SRC)
    if vendored_path not in sys.path:
        sys.path.insert(0, vendored_path)

# Register SkillBlender custom policies with RSL-RL if available.
try:
    from sbm.rl import register_rsl_rl

    register_rsl_rl()
except ImportError:
    pass

# Register SkillBlender custom networks with rl_games if available.
try:
    from sbm.rl import register_rl_games_dualhead

    register_rl_games_dualhead()
except ImportError:
    pass

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", ".mdp"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)

# Explicitly import the new 'approach' task config to ensure registration
import openarm.tasks.manager_based.openarm_manipulation.pipeline.hand.both.approach.config
import openarm.tasks.manager_based.openarm_manipulation.pipeline.hand.both.bi_pouring_v1.config

# bimanual/reach
import openarm.tasks.manager_based.openarm_manipulation.pipeline.gripper.both.reach.config
# bimanual/grasp,grasp2g
import openarm.tasks.manager_based.openarm_manipulation.pipeline.hand.both.grasp.config
import openarm.tasks.manager_based.openarm_manipulation.pipeline.gripper.both.grasp_2g.config
import openarm.tasks.manager_based.openarm_manipulation.pipeline.gripper.both.grasp_2g.grasp_2g_env_cfg

# Explicit right-hand pipeline task imports (numeric module segments require importlib).
for _mod in [
    "openarm.tasks.manager_based.openarm_manipulation.pipeline.hand.right.5g_grasp_right_v1.config",
    "openarm.tasks.manager_based.openarm_manipulation.pipeline.hand.right.5g_grasp_right_v2.config",
    "openarm.tasks.manager_based.openarm_manipulation.pipeline.hand.right.5g_grasp_right_v3.config",
    "openarm.tasks.manager_based.openarm_manipulation.pipeline.hand.right.5g_grasp_right_v4.config",
    "openarm.tasks.manager_based.openarm_manipulation.pipeline.hand.right.5g_grasp_right_v5.config",
    "openarm.tasks.manager_based.openarm_manipulation.pipeline.hand.right.5g_grasp_right_v6.config",
    "openarm.tasks.manager_based.openarm_manipulation.pipeline.hand.right.5g_lift_right_v1.config",
    "openarm.tasks.manager_based.openarm_manipulation.pipeline.hand.right.5g_grasp_right_v7.config",
]:
    try:
        importlib.import_module(_mod)
    except (ModuleNotFoundError, ImportError):
        pass

# pipeline/gripper/left/2g_grasp_left_v1
# NOTE: module segment starts with a digit, so standard `import ...` syntax is invalid.
for _mod in [
    "openarm.tasks.manager_based.openarm_manipulation.pipeline.gripper.left.2g_grasp_left_v1.config",
    "openarm.tasks.manager_based.openarm_manipulation.pipeline.gripper.right.2g_grasp_right_v1.config",
    "openarm.tasks.manager_based.openarm_manipulation.pipeline.gripper.both.2g_pouring_v1.config",
]:
    try:
        importlib.import_module(_mod)
    except (ModuleNotFoundError, ImportError):
        pass

# pipeline/hand/*: auto import all task config modules
_TASKS_ROOT = Path(__file__).resolve().parent
_HAND_ROOT = _TASKS_ROOT / "manager_based" / "openarm_manipulation" / "pipeline" / "hand"
for cfg_init in sorted(_HAND_ROOT.glob("**/config/__init__.py")):
    module_path = cfg_init.parent.as_posix()
    marker = "openarm/tasks/"
    if marker not in module_path:
        continue
    rel_module = module_path.split(marker, 1)[1].replace("/", ".")
    module_name = f"openarm.tasks.{rel_module}"
    try:
        importlib.import_module(module_name)
    except (ModuleNotFoundError, ImportError):
        pass

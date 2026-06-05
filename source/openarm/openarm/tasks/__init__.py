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

# Prefer vendored FABRICS under this repository to avoid external path dependency.
_SRC_ROOT = Path(__file__).resolve().parents[3]
_VENDORED_FABRICS_SRC = _SRC_ROOT / "FABRICS" / "src"
if _VENDORED_FABRICS_SRC.exists():
    vendored_path = str(_VENDORED_FABRICS_SRC)
    if vendored_path not in sys.path:
        sys.path.insert(0, vendored_path)

# Register SkillBlender custom policies/networks if available.
try:
    from sbm.rl import register_rsl_rl
    register_rsl_rl()
except ImportError:
    pass

try:
    from sbm.rl import register_rl_games_dualhead
    register_rl_games_dualhead()
except ImportError:
    pass

# Auto-discover all task configs: <robot>/<side>/<task>/config/__init__.py
_TASKS_ROOT = Path(__file__).resolve().parent.parent
for _cfg_init in sorted(_TASKS_ROOT.glob("*/*/*/config/__init__.py")):
    _module_path = _cfg_init.parent.as_posix()
    if "openarm/openarm/" not in _module_path:
        continue
    _rel = _module_path.split("openarm/openarm/", 1)[1].replace("/", ".")
    try:
        importlib.import_module(f"openarm.{_rel}")
    except (ModuleNotFoundError, ImportError):
        pass

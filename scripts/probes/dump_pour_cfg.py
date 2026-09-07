"""env 의 **실효 cfg** 를 YAML 로 덤프한다 — grasp_s2r(생산자) 런 dump 와 대조용.

왜 필요한가. warm 뱅크는 생산자의 물리·제어 조건 아래서만 재현된다. 소스를 눈으로
비교하면 "cfg 에 안 적어서 조용한 기본값이 들어간 항목"(sim 마찰 0.5 등)을 놓친다.
실효 객체를 덤프해야 그런 항목까지 드러난다.

실행:
    ../IsaacLab/isaaclab.sh -p scripts/probes/dump_pour_cfg.py --out /tmp/pour_env.yaml
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="open-tesol_r_pour_sensor-play-lstm")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
args.headless = True
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import os  # noqa: E402
from pathlib import Path  # noqa: E402

_HDGP = Path(__file__).resolve().parents[2]
for _p in (str(_HDGP / "scripts/tools"), str(_HDGP / "source/openarm")):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
for _n in [m for m in sys.modules if m == "openarm" or m.startswith("openarm.")]:
    del sys.modules[_n]

import isaaclab_tasks  # noqa: F401,E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import openarm.tasks  # noqa: F401,E402

cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
dump_yaml(args.out, cfg)
print(f"덤프 완료: {args.out}", flush=True)
app.close()

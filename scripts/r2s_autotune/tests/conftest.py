"""r2s_autotune 순수 파이썬 테스트. Isaac Lab 없이 돈다.

replay_env.py / *_replay.py / make_synthetic_track.py는 isaaclab을 import하므로
여기서 다루지 않는다. 그 부분의 검증은 GPU에서 recovery test로 한다.
"""

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

CONFIG_DIR = SCRIPTS_DIR / "r2s_autotune" / "configs"


@pytest.fixture(params=["bi_rh56f1.yaml", "tesollo_sensor.yaml"])
def config_path(request) -> Path:
    return CONFIG_DIR / request.param

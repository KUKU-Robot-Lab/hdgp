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


# 목록을 손으로 적으면 새 부위별 config가 검증 밖에 남는다 — 디렉토리를 그대로 훑는다.
ALL_CONFIGS = sorted(p.name for p in CONFIG_DIR.glob("*.yaml"))


@pytest.fixture(params=ALL_CONFIGS)
def config_path(request) -> Path:
    return CONFIG_DIR / request.param

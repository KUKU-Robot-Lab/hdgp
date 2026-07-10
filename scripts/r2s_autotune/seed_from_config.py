"""Config 기본값에서 seed calibration을 만든다.

실물 identification 데이터가 아직 없을 때의 출발점이다.
real 데이터가 생기면 teleop repo의 real2sim_actuator_calibration.py 산출물이 seed가 된다.
"""

from __future__ import annotations

from r2s_autotune.calibration_io import Calibration, GroupCalibration
from r2s_autotune.config import AutotuneConfig


def seed_from_config(config: AutotuneConfig, source_dataset: str = "") -> Calibration:
    groups = {
        name: GroupCalibration(
            stiffness=group.stiffness,
            damping=group.damping,
            joint_friction=group.joint_friction,
        )
        for name, group in config.groups.items()
    }
    return Calibration(
        robot_asset=config.asset,
        source_dataset=source_dataset,
        groups=groups,
    )

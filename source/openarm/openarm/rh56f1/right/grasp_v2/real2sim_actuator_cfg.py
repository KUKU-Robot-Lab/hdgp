"""Real2Sim actuator calibration loading for grasp_right_v10."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class Real2SimActuatorGroup:
    stiffness: float | None = None
    damping: float | None = None
    effort_limit: float | None = None
    velocity_limit: float | None = None
    joint_friction: float | None = None
    delay_steps: int = 0


def _optional_float(values: Mapping[str, Any], key: str) -> float | None:
    if key not in values or values[key] is None:
        return None
    return float(values[key])


def load_real2sim_calibration(path: str | Path | None) -> dict[str, Real2SimActuatorGroup]:
    if path is None or str(path) == "":
        return {}
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Real2Sim calibration file does not exist: {path}")
    payload = json.loads(path.read_text())
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported Real2Sim calibration schema_version")
    groups = payload.get("groups", {})
    if not isinstance(groups, dict):
        raise ValueError("Real2Sim calibration 'groups' must be an object")

    result: dict[str, Real2SimActuatorGroup] = {}
    for name, values in groups.items():
        if not isinstance(values, dict):
            raise ValueError(f"Real2Sim calibration group must be an object: {name}")
        result[str(name)] = Real2SimActuatorGroup(
            stiffness=_optional_float(values, "stiffness"),
            damping=_optional_float(values, "damping"),
            effort_limit=_optional_float(values, "effort_limit"),
            velocity_limit=_optional_float(values, "velocity_limit"),
            joint_friction=_optional_float(values, "joint_friction"),
            delay_steps=int(values.get("delay_steps", 0) or 0),
        )
    return result


def get_actuator_params(
    group_name: str,
    calibration: Mapping[str, Real2SimActuatorGroup],
    default_stiffness: float,
    default_damping: float,
) -> dict[str, float]:
    group = calibration.get(group_name)
    if group is None:
        return {"stiffness": default_stiffness, "damping": default_damping}

    params: dict[str, float] = {
        "stiffness": default_stiffness if group.stiffness is None else group.stiffness,
        "damping": default_damping if group.damping is None else group.damping,
    }
    if group.effort_limit is not None:
        params["effort_limit_sim"] = group.effort_limit
    if group.velocity_limit is not None:
        params["velocity_limit_sim"] = group.velocity_limit
    if group.joint_friction is not None:
        params["friction"] = group.joint_friction
    return params

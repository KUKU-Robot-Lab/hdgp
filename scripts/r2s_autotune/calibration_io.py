"""Calibration JSON (schema_version 1) 읽기/쓰기.

schema는 hdgp의 기존 loader가 정한다:
  source/openarm/openarm/*/right/*/real2sim_actuator_cfg.py
새 loader를 만들지 않는다. autotune 결과를 그 schema에 맞춰 내보낼 뿐이다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 1

# 주의: hdgp의 get_actuator_params()는 delay_steps를 읽지 않는다. JSON에 기록되지만
# Isaac Lab에 전달되지 않는 죽은 필드다. 지금은 error 항에만 쓴다.


@dataclass(frozen=True)
class GroupCalibration:
    stiffness: float
    damping: float
    joint_friction: float = 0.0
    delay_steps: int = 0

    def scaled(self, stiffness: float, damping: float, friction: float) -> "GroupCalibration":
        return GroupCalibration(
            stiffness=self.stiffness * stiffness,
            damping=self.damping * damping,
            joint_friction=self.joint_friction * friction,
            delay_steps=self.delay_steps,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "stiffness": float(self.stiffness),
            "damping": float(self.damping),
            "joint_friction": float(self.joint_friction),
            "delay_steps": int(self.delay_steps),
        }


@dataclass(frozen=True)
class Calibration:
    robot_asset: str
    source_dataset: str
    groups: Mapping[str, GroupCalibration]
    fit_error: Mapping[str, float] | None = None

    def with_groups(self, groups: Mapping[str, GroupCalibration]) -> "Calibration":
        return Calibration(
            robot_asset=self.robot_asset,
            source_dataset=self.source_dataset,
            groups=dict(groups),
            fit_error=self.fit_error,
        )


def load_calibration(path: str | Path) -> Calibration:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"calibration JSON not found: {path}")
    payload = json.loads(path.read_text())

    version = int(payload.get("schema_version", 0))
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported calibration schema_version: {version}")

    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, dict) or not raw_groups:
        raise ValueError("calibration 'groups' must be a non-empty object")

    groups = {
        name: GroupCalibration(
            stiffness=float(body["stiffness"]),
            damping=float(body["damping"]),
            joint_friction=float(body.get("joint_friction", 0.0)),
            delay_steps=int(body.get("delay_steps", 0) or 0),
        )
        for name, body in raw_groups.items()
    }
    return Calibration(
        robot_asset=str(payload.get("robot_asset", "")),
        source_dataset=str(payload.get("source_dataset", "")),
        groups=groups,
        fit_error=payload.get("fit_error"),
    )


def write_calibration(
    path: str | Path,
    calibration: Calibration,
    extra: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "robot_asset": calibration.robot_asset,
        "source_dataset": calibration.source_dataset,
        "groups": {name: g.to_json() for name, g in calibration.groups.items()},
    }
    if calibration.fit_error:
        payload["fit_error"] = dict(calibration.fit_error)
    if extra:
        payload.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

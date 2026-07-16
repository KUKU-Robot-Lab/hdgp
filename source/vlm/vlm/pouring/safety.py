from __future__ import annotations

import math

from .contracts import ControlMode, SkillCommand


class SafetySupervisor:
    """Final command gate independent of the selected policy."""

    @staticmethod
    def validate(command: SkillCommand) -> SkillCommand:
        if not all(math.isfinite(float(value)) for value in command.values):
            return SkillCommand(ControlMode.SAFE_STOP, (), "safety_supervisor")
        return command

from __future__ import annotations

import math

from .contracts import SkillCommand


class SafetySupervisor:
    """Final command gate independent of the selected policy.

    A non-finite value on either channel invalidates the whole command —
    a corrupt hand command next to a healthy arm command is still a corrupt
    skill output, and freezing one channel while trusting the other would
    blend a broken source into the robot.
    """

    @staticmethod
    def validate(command: SkillCommand) -> SkillCommand:
        for channel in (command.arm, command.hand):
            if not all(math.isfinite(float(value)) for value in channel.values):
                return SkillCommand.safe_stop("safety_supervisor")
        return command

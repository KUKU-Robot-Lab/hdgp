from __future__ import annotations

from ..contracts import SkillId
from ..execution import ReferencedPolicySkill


class GraspLiftSkill(ReferencedPolicySkill):
    """Adapter for the existing 106D-observation, 11D-action grasp policy."""

    skill_id = SkillId.GRASP_LIFT
    observation_dim = 106
    action_dim = 11

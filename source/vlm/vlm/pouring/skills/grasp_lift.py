from __future__ import annotations

from ..contracts import SkillId
from ..execution import ReferencedPolicySkill


class GraspLiftSkill(ReferencedPolicySkill):
    """Adapter for an existing grasp policy.

    Dimensions come from the referenced run's `params/env.yaml` (legacy
    grasp_v1 was 106D/11D; retrained tracks differ) or an explicit override.
    """

    skill_id = SkillId.GRASP_LIFT

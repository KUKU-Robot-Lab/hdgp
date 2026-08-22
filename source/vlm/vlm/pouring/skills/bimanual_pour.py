from __future__ import annotations

from ..contracts import SkillId
from ..execution import ReferencedPolicySkill


class BimanualPourSkill(ReferencedPolicySkill):
    """Adapter for an existing pour policy.

    Dimensions come from the referenced run's `params/env.yaml` (the bimanual
    both/pour-v1 track is 51D/15D; the legacy right-arm pour_v1 was 55D/12D)
    or an explicit override.
    """

    skill_id = SkillId.BIMANUAL_POUR

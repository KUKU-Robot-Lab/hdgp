from __future__ import annotations

from ..contracts import SkillId
from ..execution import ReferencedPolicySkill


class BimanualPourSkill(ReferencedPolicySkill):
    """Adapter for the existing 55D-observation, 12D-action pour policy."""

    skill_id = SkillId.BIMANUAL_POUR
    observation_dim = 55
    action_dim = 12

from __future__ import annotations

from pathlib import Path


ENV_PATH = Path(
    "/home/user/rl_ws/hdgp/source/openarm/openarm/tesollo/right/grasp_v10_3/"
    "grasp_right_env.py"
)


def test_env_logs_categorized_tensorboard_tags_without_raw_legacy_prefixes():
    source = ENV_PATH.read_text(encoding="utf-8")

    required_metrics = {
        'self.extras["reward/palm"]',
        'self.extras["reward/thumb_pose_anchor"]',
        'self.extras["object_stat/success_rate"]',
        'self.extras["object_stat/obj_z"]',
        'self.extras["hand_force/f_thumb"]',
        'self.extras["hand_force/f_ratio_delta"]',
        'self.extras["hand_contact/middle_contacts"]',
        'self.extras["hand_action/lift/finger_action_abs_mean"]',
        'self.extras["hand_joint/rj_1/thumb_anchor_error"]',
        'self.extras["hand_joint/rj_1/thumb_tip_direction_cos"]',
        'self.extras[f"mass_bin/{_tag}/sr"]',
        'self.extras[f"mass_bin/{_tag}/contacts"]',
        'self.extras[f"mass_bin/{_tag}/lift"]',
        'self.extras[f"mass_bin/{_tag}/full_contact"]',
        'self.extras[f"mass_bin/{_tag}/f_ratio"]',
        'self.extras[f"mass_bin/{_tag}/adaptive_grip"]',
        'self.extras[f"mass_bin/{_tag}/multi_phalanx"]',
    }

    for metric in required_metrics:
        assert metric in source

    forbidden_fragments = {
        'self.extras["r_',
        'self.extras["stat_',
        'self.extras["f_thumb"]',
        'self.extras["f_ratio"]',
        'self.extras["thumb_anchor_error"]',
        'self.extras[f"bin_',
    }

    for fragment in forbidden_fragments:
        assert fragment not in source

from __future__ import annotations

from pathlib import Path


ENV_PATH = Path(
    "/home/user/rl_ws/hdgp/source/openarm/openarm/tasks/manager_based/"
    "openarm_manipulation/pipeline/hand/right/5g_grasp_right_v10/grasp_right_env.py"
)


def test_env_logs_per_bin_success_lift_contacts_and_full_contact_metrics():
    source = ENV_PATH.read_text(encoding="utf-8")

    required_metrics = {
        'self.extras[f"bin_{_tag}_sr"]',
        'self.extras[f"bin_{_tag}_contacts"]',
        'self.extras[f"bin_{_tag}_lift"]',
        'self.extras[f"bin_{_tag}_full_contact"]',
        'self.extras[f"bin_{_tag}_f_ratio"]',
        'self.extras[f"bin_{_tag}_adaptive_grip"]',
        'self.extras[f"bin_{_tag}_multi_phalanx"]',
    }

    for metric in required_metrics:
        assert metric in source

from __future__ import annotations

from pathlib import Path


ENV_PATH = Path(
    "/home/user/rl_ws/hdgp/source/openarm/openarm/tesollo/right/grasp_v10_3/"
    "grasp_right_env.py"
)


def test_env_logs_categorized_tensorboard_tags_without_raw_legacy_prefixes():
    source = ENV_PATH.read_text(encoding="utf-8")

    required_metrics = {
        'self.extras["object_stat/obj_z"]',
        'self.extras["debug/tesollo/control/raw_palm_action_norm"]',
        'self.extras["debug/tesollo/control/ema_palm_action_norm"]',
        'self.extras["debug/tesollo/control/palm_target_position_error"]',
        '"phase/stabilize"',
        '"reward/stabilize"',
        '"reward/stability"',
        '"task/stabilize_success_now"',
        '"task/success_rate"',
    }

    for metric in required_metrics:
        assert metric in source

    forbidden_fragments = {'"phase/transport"', '"reward/transport_', '"task/transport_'}

    for fragment in forbidden_fragments:
        assert fragment not in source

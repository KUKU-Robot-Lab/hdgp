from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "parse_grasp_right_v10_tb.py"


def load_module():
    if not SCRIPT_PATH.is_file():
        pytest.fail(f"analysis script missing: {SCRIPT_PATH}")

    spec = importlib.util.spec_from_file_location("parse_grasp_right_v10_tb", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_checkpoint_filename():
    module = load_module()

    parsed = module.parse_checkpoint_filename(
        "last_5g_grasp_right-v10_ep_1150_rew_-10554.675.pth"
    )

    assert parsed == {"episode": 1150, "reward": pytest.approx(-10554.675)}


def test_build_window_summary_splits_dataframe_into_four_ranges():
    module = load_module()

    df = pd.DataFrame(
        {
            "rewards": [1.0, 2.0, 3.0, 4.0],
            "stat_success_rate": [0.1, 0.2, 0.3, 0.4],
        },
        index=pd.Index([100, 200, 300, 400], name="iter"),
    )

    windows = [(1, 100), (101, 200), (201, 300), (301, 400)]
    summary = module.build_window_summary(
        df,
        metrics=["rewards", "stat_success_rate"],
        windows=windows,
    )

    assert list(summary["window"]) == ["idx 1-100", "idx 101-200", "idx 201-300", "idx 301-400"]
    assert list(summary["rewards"]) == [1.0, 2.0, 3.0, 4.0]
    assert list(summary["stat_success_rate"]) == [0.1, 0.2, 0.3, 0.4]


def test_render_markdown_includes_missing_metrics_section():
    module = load_module()

    report = module.render_markdown_report(
        report_date="2026-04-13",
        run_a_name="test7",
        run_b_name="test8",
        run_a_dir=Path("/tmp/test7"),
        run_b_dir=Path("/tmp/test8"),
        config_summary={"seed_a": 17, "seed_b": 42, "key_diffs": ["seed only"]},
        checkpoint_summary=pd.DataFrame(
            [
                {"run": "test7", "peak_reward": 10.0, "worst_reward": -1.0, "final_reward": 8.0},
                {"run": "test8", "peak_reward": 11.0, "worst_reward": -2.0, "final_reward": 7.0},
            ]
        ),
        window_tables={"core_metrics": pd.DataFrame([{"window": "idx 1-100", "rewards": 1.0}])},
        final_comparison=pd.DataFrame([{"metric": "rewards", "test7": 1.0, "test8": 2.0}]),
        findings=[
            "test8 keeps lower final reward than test7.",
            "Both runs differ only by seed.",
        ],
        missing_metrics=["bin_20b_sr", "bin_30b_sr"],
    )

    assert "# 5g_grasp_right_v10 test7/test8" in report
    assert "bin_20b_sr" in report
    assert "Both runs differ only by seed." in report


def test_bin_metrics_whitelist_matches_reduced_logging():
    module = load_module()

    expected_metrics = {
        "bin_0b_sr",
        "bin_10b_sr",
        "bin_20b_sr",
        "bin_30b_sr",
        "bin_0b_f_ratio",
        "bin_10b_f_ratio",
        "bin_20b_f_ratio",
        "bin_30b_f_ratio",
        "bin_0b_contacts",
        "bin_10b_contacts",
        "bin_20b_contacts",
        "bin_30b_contacts",
        "bin_0b_lift",
        "bin_10b_lift",
        "bin_20b_lift",
        "bin_30b_lift",
        "bin_0b_adaptive_grip",
        "bin_10b_adaptive_grip",
        "bin_20b_adaptive_grip",
        "bin_30b_adaptive_grip",
        "bin_0b_full_contact",
        "bin_10b_full_contact",
        "bin_20b_full_contact",
        "bin_30b_full_contact",
        "bin_0b_multi_phalanx",
        "bin_10b_multi_phalanx",
        "bin_20b_multi_phalanx",
        "bin_30b_multi_phalanx",
    }

    assert expected_metrics.issubset(set(module.BIN_METRICS))
    assert "bin_0b_slip" not in module.BIN_METRICS
    assert "bin_10b_f_smooth" not in module.BIN_METRICS


def test_core_metrics_include_thumb_and_shape_rewards():
    module = load_module()

    expected_metrics = {
        "r_thumb_pose_anchor",
        "r_thumb_slide_penalty",
        "r_grasp_shape_consistency",
        "thumb_anchor_error",
        "thumb_downward_delta",
        "grasp_shape_error",
    }

    assert expected_metrics.issubset(set(module.CORE_METRICS))

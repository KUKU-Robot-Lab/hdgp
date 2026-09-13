#!/usr/bin/env python3
"""Human baseline interaction for one IKER configuration (design spec §6). No Isaac needed.

Reads <run-dir>/keypoints.json and assets/iker_shoe/shoe_meta.json, carries the public IKER target
relation onto the snapshot's other shoe, refits it to the moving shoe, gates it like a VLM result
and writes <run-dir>/interaction_human.json. Exit code 0 only when the gate passes.

Usage:
    cd ~/rl_ws/hdgp && PYTHONPATH=source/openarm python3 scripts/iker/human_baseline.py \
        --run-dir iker_runs/shoe_place/config_00
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from openarm.agnostic.modules.iker import baseline, gate, run_files
from openarm.agnostic.modules.iker.interaction import Interaction
from openarm.agnostic.modules.iker.rotations import quat_wxyz_to_matrix
from openarm.agnostic.tasks.iker_shoe import layout

LEGACY_OTHER_SHOE = "object_1"


def human_interaction(doc: dict, meta: dict) -> tuple[Interaction, gate.GateReport, baseline.HumanTarget]:
    supports = layout.supports()
    movables = {obj.name: obj for obj in run_files.movable_objects(doc, meta, supports)}
    legacy = meta["legacy"]
    legacy_other = legacy["object_pose"][LEGACY_OTHER_SHOE]
    other = doc["objects"][layout.OTHER_SHOE]
    relation = baseline.legacy_relation_targets(
        legacy_other_pos=np.add(legacy_other["xyz"], [0.0, 0.0, legacy["z_lift"]]),
        legacy_other_rpy=legacy_other["rpy"],
        legacy_targets=legacy["target_position"],
        other_pos=other["position"],
        other_rotation=quat_wxyz_to_matrix(other["quat_wxyz"]),
    )
    moving = movables[layout.MOVING_SHOE]
    target = baseline.human_target(
        relation,
        moving.local_offsets,
        moving.hull_local,
        start_rotation=quat_wxyz_to_matrix(doc["objects"][layout.MOVING_SHOE]["quat_wxyz"]),
        long_axis=int(meta["objects"][layout.MOVING_SHOE]["horizontal_axes"][0]),
        supports=supports,
    )
    interaction = Interaction(
        object_name=layout.MOVING_SHOE,
        keypoint_ids=moving.keypoint_ids,
        grasp_mode=True,
        coordinates={kid: tuple(point) for kid, point in zip(moving.keypoint_ids, target.keypoints.tolist())},
        done=False,
    )
    report = gate.evaluate_gate(interaction, tuple(movables.values()), run_files.static_keypoint_ids(doc), layout.gate_config())
    return interaction, report, target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the human baseline interaction for one configuration.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--meta", type=Path, default=layout.SHOE_META_PATH)
    args = parser.parse_args(argv)
    run_dir = args.run_dir if args.run_dir.is_absolute() else layout.HDGP_ROOT / args.run_dir
    doc = run_files.read_json(run_dir / "keypoints.json")
    meta = run_files.load_shoe_meta(args.meta)
    interaction, report, target = human_interaction(doc, meta)
    extra = {
        "relation_residual_m": target.relation_residual_m.tolist(),
        "width_pair_swapped": target.width_pair_swapped,
        "support": target.support_name,
    }
    out = run_files.write_json(
        run_dir / "interaction_human.json", run_files.interaction_document("human", interaction, report, extra=extra)
    )
    print(f"gate passed={report.passed} failures={list(report.failures)}")
    print(f"targets {np.round(report.target_keypoints, 4).tolist()} -> {out}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

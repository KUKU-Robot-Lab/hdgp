"""IKER run artifacts (design spec §5-6).

``assets/iker_shoe/shoe_meta.json``    keypoint offsets, rest pose and hull of each shoe, legacy relation
``<run>/keypoints.json``               one head-camera snapshot: camera, object poses, labelled keypoints
``<run>/interaction_<source>.json``    gated target keypoints from the human baseline or a VLM response
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .gate import GateReport, MovableObject, Support
from .interaction import Interaction
from .projection import CameraPose
from .scene_image import AnnotatedSnapshot

SCHEMA_VERSION = 1
INTERACTION_SOURCES = ("human", "vlm")
META_OBJECT_KEYS = (
    "source", "rest_rotation", "rest_quat_wxyz", "horizontal_axes", "keypoint_offsets", "rest_height", "hull_local",
)


def read_json(path) -> dict:
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"missing IKER artifact: {file}")
    doc = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"{file}: expected a JSON object with schema {SCHEMA_VERSION}")
    return doc


def write_json(path, doc: Mapping) -> Path:
    """Write atomically. NaN and infinity are rejected so a broken value never reaches a reader."""
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=1, allow_nan=False) + "\n"
    tmp = file.with_name(file.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(file)
    return file


def load_shoe_meta(path) -> dict:
    meta = read_json(path)
    objects = meta.get("objects")
    if not isinstance(objects, dict) or not objects:
        raise ValueError(f"{path}: 'objects' is missing")
    for name, obj in objects.items():
        missing = [key for key in META_OBJECT_KEYS if key not in obj]
        if missing:
            raise ValueError(f"{path}: object {name!r} lacks {missing}")
    if "legacy" not in meta:
        raise ValueError(f"{path}: 'legacy' is missing")
    return meta


def keypoints_document(
    *,
    scene_config: Mapping,
    camera: CameraPose,
    intrinsic,
    snapshot: AnnotatedSnapshot,
    image_size: Sequence[int],
    object_poses: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    head: Mapping,
    checks: Mapping,
) -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "scene_config": dict(scene_config),
        "image_size": [int(v) for v in image_size],
        "camera": {
            "position": np.asarray(camera.position, dtype=float).tolist(),
            "rotation": np.asarray(camera.rotation, dtype=float).tolist(),
            "intrinsic": np.asarray(intrinsic, dtype=float).tolist(),
            "derotation_deg": float(snapshot.derotation_deg),
        },
        "axes": {name: [float(v) for v in direction] for name, direction in snapshot.axes.items()},
        "head": dict(head),
        "objects": {
            name: {"position": [float(v) for v in pos], "quat_wxyz": [float(v) for v in quat]}
            for name, (pos, quat) in object_poses.items()
        },
        "keypoints": [asdict(record) for record in snapshot.records],
        "checks": {**dict(checks), "min_margin_px": snapshot.min_margin_px, "min_gap_px": snapshot.min_gap_px},
    }


def vlm_keypoints(doc: Mapping) -> dict[int, tuple[float, float, float]]:
    """The ``keypoint_coordinates`` input: every keypoint drawn on the image, by label."""
    return {int(r["label"]): tuple(float(v) for v in r["world"]) for r in doc["keypoints"] if r["kept"]}


def static_keypoint_ids(doc: Mapping) -> tuple[int, ...]:
    return tuple(int(r["label"]) for r in doc["keypoints"] if r["is_static"] and r["kept"])


def movable_objects(doc: Mapping, meta: Mapping, supports: Sequence[Support]) -> tuple[MovableObject, ...]:
    movables = []
    for name, obj in meta["objects"].items():
        records = sorted((r for r in doc["keypoints"] if r["object_name"] == name), key=lambda r: r["label"])
        offsets = np.asarray(obj["keypoint_offsets"], dtype=float)
        if len(records) != len(offsets):
            raise ValueError(f"{name}: snapshot has {len(records)} keypoints, shoe meta has {len(offsets)}")
        if not all(r["kept"] for r in records):
            raise ValueError(f"{name}: a keypoint was removed by the overlap filter")
        position = np.asarray(doc["objects"][name]["position"], dtype=float)
        support = next((s for s in supports if s.contains_xy(position[0], position[1])), None)
        if support is None:
            raise ValueError(f"{name}: start position {position.tolist()} is above no support")
        movables.append(
            MovableObject(
                name=name,
                keypoint_ids=tuple(int(r["label"]) for r in records),
                local_offsets=offsets,
                hull_local=np.asarray(obj["hull_local"], dtype=float),
                init_keypoints=np.asarray([r["world"] for r in records], dtype=float),
                start_support_z=support.top_z,
            )
        )
    return tuple(movables)


def gate_document(report: GateReport) -> dict:
    return {
        "passed": report.passed,
        "failures": list(report.failures),
        "object": report.object_name,
        "metrics": dict(report.metrics),
    }


def interaction_document(
    source: str,
    interaction: Interaction,
    report: GateReport,
    snapshot_name: str = "snapshot.png",
    extra: Mapping | None = None,
) -> dict:
    if source not in INTERACTION_SOURCES:
        raise ValueError(f"source must be one of {INTERACTION_SOURCES}, got {source!r}")
    doc = {
        "schema": SCHEMA_VERSION,
        "source": source,
        "snapshot": snapshot_name,
        "object": report.object_name,
        "interaction": {
            "object_to_interact": interaction.object_name,
            "keypoint_ids": list(interaction.keypoint_ids),
            "grasp_mode": interaction.grasp_mode,
        },
        "target_keypoints": [list(point) for point in report.target_keypoints],
        "gate": gate_document(report),
    }
    if extra:
        doc["extra"] = dict(extra)
    return doc

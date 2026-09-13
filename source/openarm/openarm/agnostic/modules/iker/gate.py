"""Mechanical gate for a VLM interaction (design spec §6, G1-G7).

The gate never edits the prompt. It checks that the targets describe a pose the policy can reach:
G1 the code ran and is not ``done``; G2 the ids belong to one movable object; G3 its targets are
finite; G4 the best rigid fit of the object's keypoints meets the evaluation (5 cm) and training
(normalized 0.10) success bounds; G5 that pose does not sink into its support; G6 its centre lies in
the arm's palm box; G7 a push does not raise the object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .interaction import DEFAULT_TIMEOUT_S, Interaction, InteractionError, extract_code_block, run_interaction


@dataclass(frozen=True)
class Support:
    name: str
    top_z: float
    x_range: tuple[float, float] = (-math.inf, math.inf)
    y_range: tuple[float, float] = (-math.inf, math.inf)

    def contains_xy(self, x: float, y: float) -> bool:
        return self.x_range[0] <= x <= self.x_range[1] and self.y_range[0] <= y <= self.y_range[1]


@dataclass(frozen=True)
class MovableObject:
    name: str
    keypoint_ids: tuple[int, ...]
    local_offsets: np.ndarray  # (K, 3); row i belongs to keypoint_ids[i]
    hull_local: np.ndarray  # (M, 3) convex-hull vertices in the object frame
    init_keypoints: np.ndarray  # (K, 3) world, at the snapshot
    start_support_z: float


@dataclass(frozen=True)
class GateConfig:
    supports: tuple[Support, ...]  # the first support whose rectangle holds the fitted centre is used
    palm_box_min: tuple[float, float, float]
    palm_box_max: tuple[float, float, float]
    eval_success_m: float = 0.05
    train_success_norm: float = 0.10
    penetration_tol_m: float = 0.01
    push_lift_tol_m: float = 0.02
    min_travel_m: float = 1e-3


@dataclass(frozen=True)
class GateReport:
    passed: bool
    failures: tuple[str, ...]
    metrics: Mapping[str, object]
    object_name: str | None = None
    target_keypoints: tuple[tuple[float, float, float], ...] = ()


def kabsch(source, target) -> tuple[np.ndarray, np.ndarray]:
    """Proper rotation ``R`` and translation ``t`` minimising ``sum |R source_i + t - target_i|^2``."""
    p = np.asarray(source, dtype=float)
    q = np.asarray(target, dtype=float)
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 3 or len(p) < 3:
        raise ValueError("source and target must both be (N, 3) with N >= 3")
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))):
        raise ValueError("kabsch inputs contain non-finite values")
    p_center, q_center = p.mean(axis=0), q.mean(axis=0)
    u, _, vt = np.linalg.svd((p - p_center).T @ (q - q_center))
    d = 1.0 if np.linalg.det(vt.T @ u.T) >= 0.0 else -1.0
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rotation, q_center - rotation @ p_center


def gate_response(
    response: str,
    keypoints: Mapping[int, Sequence[float]],
    movables: Sequence[MovableObject],
    static_ids: Sequence[int],
    cfg: GateConfig,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> tuple[Interaction | None, GateReport]:
    """Extract, run and gate one VLM response; a response that cannot run fails G1."""
    try:
        interaction = run_interaction(extract_code_block(response), keypoints, timeout_s)
    except InteractionError as exc:
        return None, GateReport(passed=False, failures=(f"G1: {exc}",), metrics={})
    return interaction, evaluate_gate(interaction, movables, static_ids, cfg)


def evaluate_gate(
    interaction: Interaction, movables: Sequence[MovableObject], static_ids: Sequence[int], cfg: GateConfig
) -> GateReport:
    if interaction.done:
        return _fail("G1: done=True, but a single-step task needs an interaction")
    owner = {kid: obj for obj in movables for kid in obj.keypoint_ids}
    ids = list(interaction.keypoint_ids)
    unknown = [kid for kid in ids if kid not in owner and kid not in set(static_ids)]
    if unknown:
        return _fail(f"G2: unknown keypoint ids {unknown}")
    on_static = [kid for kid in ids if kid not in owner]
    if on_static:
        return _fail(f"G2: keypoint ids {on_static} belong to a static object")
    names = sorted({owner[kid].name for kid in ids})
    if len(names) != 1:
        return _fail(f"G2: keypoint ids span objects {names}")
    obj = owner[ids[0]]
    missing = [kid for kid in obj.keypoint_ids if kid not in interaction.coordinates]
    if missing:
        return _fail(f"G3: no target coordinates for keypoints {missing} of {obj.name}", obj.name)
    targets = np.array([interaction.coordinates[kid] for kid in obj.keypoint_ids], dtype=float)
    if not np.all(np.isfinite(targets)):
        return _fail(f"G3: non-finite target coordinates for {obj.name}", obj.name)
    return _check_fit(interaction, obj, targets, cfg)


def _check_fit(interaction: Interaction, obj: MovableObject, targets: np.ndarray, cfg: GateConfig) -> GateReport:
    rotation, center = kabsch(obj.local_offsets, targets)
    residual = np.linalg.norm(obj.local_offsets @ rotation.T + center - targets, axis=1)
    travel = np.linalg.norm(targets - obj.init_keypoints, axis=1)
    hull_min_z = float((obj.hull_local @ rotation.T + center)[:, 2].min())
    support = next((s for s in cfg.supports if s.contains_xy(center[0], center[1])), None)
    in_box = bool(np.all(center >= np.asarray(cfg.palm_box_min)) and np.all(center <= np.asarray(cfg.palm_box_max)))
    mean_distance = float(residual.mean())
    moves = bool(np.all(travel > cfg.min_travel_m))
    normalized = float((residual / travel).mean()) if moves else None

    failures = []
    if not mean_distance <= cfg.eval_success_m:
        failures.append(f"G4: mean keypoint distance at the best rigid fit {mean_distance:.4f} m > {cfg.eval_success_m} m")
    if normalized is None:
        failures.append(f"G4: target equals the start for some keypoints (travel {np.round(travel, 4).tolist()} m)")
    elif not normalized <= cfg.train_success_norm:
        failures.append(f"G4: normalized error at the best rigid fit {normalized:.4f} > {cfg.train_success_norm}")
    if support is None:
        failures.append(f"G5: fitted centre {np.round(center, 4).tolist()} is above no support")
    elif not hull_min_z >= support.top_z - cfg.penetration_tol_m:
        failures.append(f"G5: fitted pose sinks {support.top_z - hull_min_z:.4f} m into {support.name}")
    if not in_box:
        failures.append(f"G6: fitted centre {np.round(center, 4).tolist()} is outside the palm box")
    if not interaction.grasp_mode and not hull_min_z <= obj.start_support_z + cfg.push_lift_tol_m:
        failures.append(f"G7: push mode cannot raise {obj.name} to {hull_min_z:.4f} m (start support {obj.start_support_z} m)")

    metrics = {
        "fitted_center": center.tolist(),
        "fitted_rotation": rotation.tolist(),
        "residual_m": residual.tolist(),
        "mean_distance_m": mean_distance,
        "travel_m": travel.tolist(),
        "normalized_error": normalized,
        "hull_min_z": hull_min_z,
        "support": support.name if support else None,
        "support_top_z": support.top_z if support else None,
        "in_palm_box": in_box,
        "grasp_mode": interaction.grasp_mode,
    }
    return GateReport(
        passed=not failures,
        failures=tuple(failures),
        metrics=metrics,
        object_name=obj.name,
        target_keypoints=tuple(tuple(float(c) for c in row) for row in targets),
    )


def _fail(message: str, object_name: str | None = None) -> GateReport:
    return GateReport(passed=False, failures=(message,), metrics={}, object_name=object_name)

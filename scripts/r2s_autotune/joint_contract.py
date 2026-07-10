"""Joint name contract between URDF manifest, Isaac articulation, and real logs.

R2S MSE가 의미를 가지려면 다섯 곳이 같은 관절을 같은 이름으로 가리켜야 한다:
URDF manifest / Isaac DOF / FABRICS / teleop HDF5 / calibration group regex.

이 모듈은 canonical 이름(r_aj_*, r_hj_*, l_aj_*, l_hj_*)을 유일한 기준으로 삼고,
legacy 이름(rj_dg_*, openarm_right_joint*)은 manifest 매핑으로 정규화해 받아들인다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml


class JointContractError(ValueError):
    """관절 이름 계약 위반. 조용히 넘어가면 MSE가 무의미해지므로 즉시 실패한다."""


@dataclass(frozen=True)
class Manifest:
    """generate_rl_urdf.py 산출 manifest의 읽기 전용 뷰."""

    path: Path
    control_joints: tuple[str, ...]
    movable_joints: tuple[str, ...]
    source_to_canonical: Mapping[str, str]

    def canonical_index(self, joint: str) -> int:
        return self.movable_joints.index(joint)


def load_manifest(path: str | Path) -> Manifest:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    raw = yaml.safe_load(path.read_text())

    for key in ("control_joint_order", "kinematic_joint_order", "fixed_joint_order"):
        if key not in raw:
            raise JointContractError(f"manifest missing '{key}': {path}")

    fixed = frozenset(raw["fixed_joint_order"])
    movable = tuple(j for j in raw["kinematic_joint_order"] if j not in fixed)
    control = tuple(raw["control_joint_order"])

    missing = [j for j in control if j not in movable]
    if missing:
        raise JointContractError(f"control joints absent from movable set: {missing}")

    return Manifest(
        path=path,
        control_joints=control,
        movable_joints=movable,
        source_to_canonical=dict(raw.get("source_to_canonical_joints", {})),
    )


def normalize_joint_names(names: Sequence[str], manifest: Manifest) -> tuple[str, ...]:
    """legacy 또는 canonical 이름 목록을 canonical로 통일한다.

    이미 canonical인 이름은 그대로 통과시킨다. 어느 쪽에도 속하지 않으면 실패한다.
    """
    movable = frozenset(manifest.movable_joints)
    resolved: list[str] = []
    unknown: list[str] = []

    for name in names:
        if name in movable:
            resolved.append(name)
            continue
        mapped = manifest.source_to_canonical.get(name)
        if mapped is not None and mapped in movable:
            resolved.append(mapped)
            continue
        unknown.append(name)

    if unknown:
        raise JointContractError(
            f"joint names resolve to neither canonical nor manifest mapping: {unknown}"
        )
    return tuple(resolved)


def resolve_group_joints(
    joint_names_expr: Sequence[str],
    movable_joints: Sequence[str],
) -> tuple[str, ...]:
    """Isaac Lab의 joint_names_expr와 동일하게 fullmatch로 관절을 고른다.

    Isaac Lab은 부분 일치가 아니라 완전 일치를 쓴다. search()로 흉내내면
    r_hj_thumb_1 이 r_hj_thumb_1x 같은 이름까지 잡아 계약이 어긋난다.
    """
    matched: list[str] = []
    for expr in joint_names_expr:
        pattern = re.compile(expr)
        hits = [j for j in movable_joints if pattern.fullmatch(j)]
        if not hits:
            raise JointContractError(f"joint_names_expr matches nothing: {expr!r}")
        matched.extend(hits)

    duplicates = _duplicates(matched)
    if duplicates:
        raise JointContractError(f"joint_names_expr overlap within group: {duplicates}")
    return tuple(matched)


@dataclass(frozen=True)
class CoverageReport:
    group_joints: Mapping[str, tuple[str, ...]]
    uncovered: tuple[str, ...]
    overlapping: tuple[str, ...]

    @property
    def is_complete(self) -> bool:
        return not self.uncovered and not self.overlapping


def verify_group_coverage(
    groups: Mapping[str, Sequence[str]],
    manifest: Manifest,
) -> CoverageReport:
    """actuator group들이 movable joint 전체를 중복 없이 덮는지 확인한다.

    groups: group_name -> joint_names_expr
    """
    group_joints = {
        name: resolve_group_joints(exprs, manifest.movable_joints)
        for name, exprs in groups.items()
    }

    claimed = [j for joints in group_joints.values() for j in joints]
    uncovered = tuple(j for j in manifest.movable_joints if j not in frozenset(claimed))

    return CoverageReport(
        group_joints=group_joints,
        uncovered=uncovered,
        overlapping=tuple(_duplicates(claimed)),
    )


def _duplicates(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    dupes: list[str] = []
    for item in items:
        if item in seen and item not in dupes:
            dupes.append(item)
        seen.add(item)
    return dupes

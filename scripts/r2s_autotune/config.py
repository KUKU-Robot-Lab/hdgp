"""Asset별 autotune 설정 로딩과 검증.

설정의 group 이름과 joint_names_expr는 RL env_cfg의 actuator 블록을 그대로 옮긴 것이다.
가이드 문서의 표가 아니라 env_cfg가 source of truth다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from r2s_autotune.joint_contract import JointContractError, Manifest, load_manifest
from r2s_autotune.paths import resolve_hdgp_path


@dataclass(frozen=True)
class ScaleRange:
    low: float
    high: float

    def __post_init__(self) -> None:
        if not 0.0 < self.low <= self.high:
            raise ValueError(f"invalid scale range: [{self.low}, {self.high}]")


@dataclass(frozen=True)
class GroupConfig:
    name: str
    joint_names_expr: tuple[str, ...]
    stiffness: float
    damping: float
    joint_friction: float
    stiffness_scale: ScaleRange | None
    damping_scale: ScaleRange | None
    friction_scale: ScaleRange | None

    @property
    def is_tunable(self) -> bool:
        return self.stiffness_scale is not None


@dataclass(frozen=True)
class RealTrackConfig:
    hdf5: Path
    demo_key: str
    command_dataset: str
    measured_dataset: str
    velocity_dataset: str | None
    dt: float


@dataclass(frozen=True)
class ErrorWeights:
    q: float = 1.0
    dq: float = 0.05
    delay: float = 0.01


@dataclass(frozen=True)
class AutotuneConfig:
    asset: str
    usd_path: Path
    manifest: Manifest
    groups: Mapping[str, GroupConfig]
    tune_groups: tuple[str, ...]
    population_size: int
    random_seed: int
    real_track: RealTrackConfig | None
    error_weights: ErrorWeights

    def tunable(self) -> tuple[GroupConfig, ...]:
        return tuple(self.groups[name] for name in self.tune_groups)


def _scale(raw: Mapping[str, Any], key: str) -> ScaleRange | None:
    if key not in raw or raw[key] is None:
        return None
    low, high = raw[key]
    return ScaleRange(float(low), float(high))


def _group(name: str, raw: Mapping[str, Any]) -> GroupConfig:
    exprs = raw.get("joint_names_expr")
    if not exprs:
        raise ValueError(f"group '{name}' missing joint_names_expr")
    defaults = raw.get("defaults", {})
    search = raw.get("search", {})
    return GroupConfig(
        name=name,
        joint_names_expr=tuple(exprs),
        stiffness=float(defaults.get("stiffness", 0.0)),
        damping=float(defaults.get("damping", 0.0)),
        joint_friction=float(defaults.get("joint_friction", 0.0)),
        stiffness_scale=_scale(search, "stiffness_scale"),
        damping_scale=_scale(search, "damping_scale"),
        friction_scale=_scale(search, "friction_scale"),
    )


def _real_track(raw: Mapping[str, Any] | None) -> RealTrackConfig | None:
    if not raw:
        return None
    return RealTrackConfig(
        hdf5=resolve_hdgp_path(raw["hdf5"]),
        demo_key=raw.get("demo_key", "demo_0"),
        command_dataset=raw["command_dataset"],
        measured_dataset=raw["measured_dataset"],
        velocity_dataset=raw.get("velocity_dataset"),
        dt=float(raw.get("dt", 0.01)),
    )


def load_config(path: str | Path) -> AutotuneConfig:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"autotune config not found: {path}")
    raw = yaml.safe_load(path.read_text())

    manifest = load_manifest(resolve_hdgp_path(raw["manifest"]))
    groups = {name: _group(name, body) for name, body in raw["groups"].items()}

    tune_groups = tuple(raw.get("tune_groups", []))
    for name in tune_groups:
        if name not in groups:
            raise ValueError(f"tune_groups references unknown group: {name}")
        if not groups[name].is_tunable:
            raise ValueError(f"tune group '{name}' has no search.stiffness_scale")

    usd_path = resolve_hdgp_path(raw["usd_path"])
    if not usd_path.is_file():
        raise FileNotFoundError(f"usd_path does not exist: {usd_path}")

    population_size = int(raw.get("population_size", 128))
    if population_size < 1:
        raise ValueError("population_size must be >= 1")

    config = AutotuneConfig(
        asset=raw["asset"],
        usd_path=usd_path,
        manifest=manifest,
        groups=groups,
        tune_groups=tune_groups,
        population_size=population_size,
        random_seed=int(raw.get("random_seed", 7)),
        real_track=_real_track(raw.get("real_track")),
        error_weights=ErrorWeights(**raw.get("error_weights", {})),
    )
    _verify(config)
    return config


def _verify(config: AutotuneConfig) -> None:
    """movable joint를 중복 없이 덮는지 확인. 어긋나면 calibration이 조용히 무시된다."""
    from r2s_autotune.joint_contract import verify_group_coverage

    report = verify_group_coverage(
        {name: g.joint_names_expr for name, g in config.groups.items()},
        config.manifest,
    )
    if not report.is_complete:
        raise JointContractError(
            f"{config.asset}: actuator group coverage incomplete "
            f"(uncovered={report.uncovered}, overlapping={report.overlapping})"
        )

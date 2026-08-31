#!/usr/bin/env python3
"""pour 궤적 HDF5(schema v2) 의 스키마·검증·왕복 IO. **Isaac 무의존.**

왜 별도 모듈인가 — s2r 재생의 진실원천은 정책이 아니라 **이 파일**이다. 파일이
조용히 어긋나면 "정책이 낸 궤적"과 "로봇이 받은 궤적"을 다른 채로 비교하게 된다.
그래서 뽑는 쪽(Isaac 필요)과 검사하는 쪽(여기)을 갈라, 검사는 시뮬레이터 없이
테스트로 잠근다.

v1(`probes/record_pour_traj.py` 산출물) 과의 차이:
  - `joint_pos` 만이 아니라 **`joint_pos_target`(지령)** 을 따로 담는다. JTC 로 나가는
    것은 측정이 아니라 지령이다.
  - `joint_vel` · `action` · 좌우 **EE 포즈** · Fabrics **지령 포즈**를 담는다.
  - 지령 포즈는 태스크마다 있고 없다: `right/pour_sensor` 는 왼팔이 정지라
    `left_ee_cmd_pose` 가 없다(→ missing 선언). `both/pour_sensor` 는 둘 다 있다.
  - ★`cup_in_hand_pose` — source cup 을 우팔 palm 기준으로 본 상대자세. grasp 정책이
    만드는 파지는 pour warm bank 의 파지와 다르므로, 관절 재생만으로는 붓는 지점이
    어긋난다. 그 어긋남을 재는 채널이다.
  - 없는 채널은 **`missing_channels` 에 선언**한다. 조용히 비면 재생 소스로 오인된다.

포즈 표기: `[x, y, z, qw, qx, qy, qz]`, robot-base-local, meter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import os
from pathlib import Path
import re

import h5py
import numpy as np

SCHEMA_VERSION = 2

POSE_DIM = 7          # pos(3) + quat wxyz(4)
QUAT_SLICE = slice(3, 7)
QUAT_NORM_TOL = 1e-3  # 붓기는 자세가 전부다. 이보다 느슨하면 림 밖으로 나간다.
BEAD_FRAC_RANGE = (0.0, 1.0)

_UNKNOWN_INT = -1     # seed·decimation 미상
_SUCCESS_UNKNOWN = -1

#: 관절 전체 폭(=len(joint_names))을 갖는 채널.
JOINT_CHANNELS = ("joint_pos", "joint_vel", "joint_pos_target")
#: 액션 폭(태스크마다 12 또는 15)을 갖는 채널.
ACTION_CHANNELS = ("action",)
#: POSE_DIM 폭을 갖는 채널.
POSE_CHANNELS = (
    "right_palm_pose",
    "left_ee_pose",
    "right_palm_cmd_pose",
    "left_ee_cmd_pose",
    "source_cup_pose",
    "target_cup_pose",
    "cup_in_hand_pose",
)
ALL_CHANNELS = JOINT_CHANNELS + ACTION_CHANNELS + POSE_CHANNELS

_GROUP_PATTERNS = {
    "right_arm": re.compile(r"^r_aj_\d+$"),
    "left_arm": re.compile(r"^l_aj_\d+$"),
    "left_gripper": re.compile(r"^l_hj_gripper_\d+$"),
    "right_hand": re.compile(r"^r_hj_(?!gripper_)"),
}
#: 그룹에 안 잡혔는데 이 패턴에 걸리면 = 자산이 바뀌었는데 그룹 정의가 안 따라온 것.
_ACTUATED_HINT = re.compile(r"_(aj|hj)_")


# ---------------------------------------------------------------------------
# 관절 그룹
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JointGroups:
    """관절명을 팔/손 그룹으로 가른 결과. **순서는 `joint_names` 순서를 보존한다.**

    그룹은 곧 열 인덱서라, 이름 순서가 배열 열 순서와 어긋나면 좌팔 지령이 우팔로 간다.
    """

    right_arm: tuple[str, ...]
    left_arm: tuple[str, ...]
    right_hand: tuple[str, ...]
    left_gripper: tuple[str, ...]
    other: tuple[str, ...]


def split_joint_groups(joint_names: tuple[str, ...] | list[str]) -> JointGroups:
    """a1(`openarm_tesollo_sensor_rl`) 관절명 규약으로 그룹을 가른다.

    규약 밖 관절(예: 07.29 USD 교체로 붙은 `head_j_*`)은 `other` 로 간다.
    기록기가 preset 상수로 명시 전달할 수 있으면 그쪽이 우선이고, 이 함수는 편의용이다.
    """
    buckets: dict[str, list[str]] = {k: [] for k in (*_GROUP_PATTERNS, "other")}
    for name in joint_names:
        for group, pattern in _GROUP_PATTERNS.items():
            if pattern.search(name):
                buckets[group].append(name)
                break
        else:
            buckets["other"].append(name)
    return JointGroups(**{k: tuple(v) for k, v in buckets.items()})


def column_index(joint_names: tuple[str, ...], group: tuple[str, ...]) -> tuple[int, ...]:
    """그룹 관절명 → `joint_pos` 열 인덱스. 없는 이름은 이름을 대고 거부한다."""
    lookup = {name: i for i, name in enumerate(joint_names)}
    missing = [n for n in group if n not in lookup]
    if missing:
        raise KeyError(f"joint_names 에 없는 관절: {missing}")
    return tuple(lookup[n] for n in group)


# ---------------------------------------------------------------------------
# 자료형
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrajMeta:
    """궤적 파일 한 개의 출처·좌표계·관절 배치. 전부 값 타입(불변)."""

    task_id: str
    checkpoint: str
    checkpoint_sha256: str
    git_commit: str
    robot_usd: str
    env_yaml_sha256: str
    dt: float
    decimation: int
    num_beads: int
    env_origin: tuple[float, ...]
    robot_root: tuple[float, ...]
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    right_arm_joint_names: tuple[str, ...]
    right_hand_joint_names: tuple[str, ...]
    left_arm_joint_names: tuple[str, ...]
    left_gripper_joint_names: tuple[str, ...]
    right_palm_body: str
    left_ee_body: str
    recorded_at: str
    missing_channels: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        # ndarray/list 로 들어와도 값 비교가 되는 불변 형태로 정규화한다.
        for name in ("env_origin", "robot_root"):
            object.__setattr__(self, name, tuple(float(x) for x in getattr(self, name)))
        for name in (
            "joint_names", "body_names", "missing_channels",
            "right_arm_joint_names", "right_hand_joint_names",
            "left_arm_joint_names", "left_gripper_joint_names",
        ):
            object.__setattr__(self, name, tuple(str(x) for x in getattr(self, name)))

    @property
    def present_channels(self) -> tuple[str, ...]:
        missing = set(self.missing_channels)
        return tuple(c for c in ALL_CHANNELS if c not in missing)


@dataclass(frozen=True)
class Episode:
    """에피소드 하나. `arrays` 는 채널명 → (T, W) 배열."""

    arrays: dict[str, np.ndarray]
    bead_frac: float
    bead_spill: float
    seed: int = _UNKNOWN_INT
    success: bool | None = None
    extras: dict[str, float] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        first = next(iter(self.arrays.values()))
        return int(first.shape[0])


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------

def validate(meta: TrajMeta, episodes: tuple[Episode, ...]) -> tuple[str, ...]:
    """계약 위반을 **전부** 모아 돌려준다. 빈 튜플 = 통과. 예외를 던지지 않는다."""
    problems: list[str] = []
    problems += _validate_meta(meta)
    if not episodes:
        problems.append("에피소드가 하나도 없다")
    for i, ep in enumerate(episodes):
        problems += [f"ep_{i:03d}: {p}" for p in _validate_episode(meta, ep)]
    return tuple(problems)


def _validate_meta(meta: TrajMeta) -> list[str]:
    problems: list[str] = []
    if meta.schema_version != SCHEMA_VERSION:
        problems.append(f"schema_version {meta.schema_version} != {SCHEMA_VERSION}")
    if not (meta.dt > 0.0):
        problems.append(f"dt 가 양수가 아니다: {meta.dt}")
    if len(meta.env_origin) != 3:
        problems.append(f"env_origin 은 3차원이어야 한다: {len(meta.env_origin)}")
    if len(meta.robot_root) != POSE_DIM:
        problems.append(f"robot_root 는 {POSE_DIM}차원이어야 한다: {len(meta.robot_root)}")
    unknown = [c for c in meta.missing_channels if c not in ALL_CHANNELS]
    if unknown:
        problems.append(f"모르는 missing_channels: {unknown}")
    problems += _validate_joint_groups(meta)
    problems += _validate_bodies(meta)
    return problems


def _validate_joint_groups(meta: TrajMeta) -> list[str]:
    problems: list[str] = []
    known = set(meta.joint_names)
    groups = {
        "right_arm_joint_names": meta.right_arm_joint_names,
        "right_hand_joint_names": meta.right_hand_joint_names,
        "left_arm_joint_names": meta.left_arm_joint_names,
        "left_gripper_joint_names": meta.left_gripper_joint_names,
    }
    seen: dict[str, str] = {}
    for label, names in groups.items():
        outside = [n for n in names if n not in known]
        if outside:
            problems.append(f"{label} 이 joint_names 밖을 가리킨다: {outside}")
        for n in names:
            if n in seen:
                problems.append(f"관절 {n} 이 {seen[n]} 와 {label} 에 겹쳐 있다")
            else:
                seen[n] = label

    uncovered = [n for n in meta.joint_names if n not in seen and _ACTUATED_HINT.search(n)]
    if uncovered:
        problems.append(f"어느 그룹에도 안 잡힌 구동 관절이 있다(자산 변경 의심): {uncovered}")
    return problems


def _validate_bodies(meta: TrajMeta) -> list[str]:
    if not meta.body_names:
        return []
    problems = []
    for label, body in (("right_palm_body", meta.right_palm_body),
                        ("left_ee_body", meta.left_ee_body)):
        if body and body not in meta.body_names:
            problems.append(f"{label}='{body}' 가 body_names 에 없다")
    return problems


def _validate_episode(meta: TrajMeta, ep: Episode) -> list[str]:
    problems: list[str] = []
    missing = set(meta.missing_channels)

    for channel in ALL_CHANNELS:
        if channel in missing and channel in ep.arrays:
            problems.append(f"{channel}: missing 선언과 실제 존재가 모순")
        if channel not in missing and channel not in ep.arrays:
            problems.append(f"{channel}: 채널이 없는데 missing_channels 에도 없다")

    present = [c for c in ALL_CHANNELS if c in ep.arrays and c not in missing]
    if not present:
        return problems + ["담긴 채널이 하나도 없다"]

    problems += _validate_shapes(meta, ep, present)
    problems += _validate_values(ep, present)

    lo, hi = BEAD_FRAC_RANGE
    for label, value in (("bead_frac", ep.bead_frac), ("bead_spill", ep.bead_spill)):
        if not (lo <= float(value) <= hi):
            problems.append(f"{label} 이 [{lo}, {hi}] 밖이다: {value}")
    return problems


def _validate_shapes(meta: TrajMeta, ep: Episode, present: list[str]) -> list[str]:
    problems: list[str] = []
    n_joints = len(meta.joint_names)
    steps = {c: int(ep.arrays[c].shape[0]) for c in present}
    reference = steps[present[0]]
    for channel, n in steps.items():
        if n != reference:
            problems.append(f"{channel}: 길이 {n} 이 {present[0]} 의 {reference} 와 다르다")

    for channel in present:
        array = ep.arrays[channel]
        if array.ndim != 2:
            problems.append(f"{channel}: 2차원 (T, W) 이어야 한다 — {array.shape}")
            continue
        width = int(array.shape[1])
        if channel in JOINT_CHANNELS and width != n_joints:
            problems.append(f"{channel}: 폭 {width} != joint_names {n_joints}")
        if channel in POSE_CHANNELS and width != POSE_DIM:
            problems.append(f"{channel}: 폭 {width} != POSE_DIM {POSE_DIM}")
    return problems


def _validate_values(ep: Episode, present: list[str]) -> list[str]:
    problems: list[str] = []
    for channel in present:
        array = ep.arrays[channel]
        if not np.all(np.isfinite(array)):
            problems.append(f"{channel}: 유한하지 않은 값이 있다 (NaN/Inf)")
            continue
        if channel in POSE_CHANNELS and array.ndim == 2 and array.shape[1] == POSE_DIM:
            norms = np.linalg.norm(array[:, QUAT_SLICE], axis=1)
            worst = float(np.max(np.abs(norms - 1.0)))
            if worst > QUAT_NORM_TOL:
                problems.append(f"{channel}: quat 이 정규화되지 않았다 (최대 편차 {worst:.2e})")
    return problems


# ---------------------------------------------------------------------------
# 초기 단면
# ---------------------------------------------------------------------------

#: 초기 단면에서 갈라 담는 관절 그룹 → 접두어.
_INIT_GROUPS = {
    "right_arm_q": "right_arm_joint_names",
    "right_hand_q": "right_hand_joint_names",
    "left_arm_q": "left_arm_joint_names",
    "left_gripper_q": "left_gripper_joint_names",
}


def init_state(meta: TrajMeta, ep: Episode) -> dict[str, np.ndarray]:
    """t=0 단면 — pour 초기 세팅.

    **측정(`*_q`)과 지령(`*_q_cmd`)을 뭉개지 않는다.** 둘은 쓰임이 다르다:
      - `*_q`     = 재생 시작 전에 실기가 **도달해 있어야 할 자세** (warm state).
      - `*_q_cmd` = 재생의 **첫 지령**. 도달 자세와 같지 않다(정책이 첫 스텝에 이미 움직인다).
    파킹 목표로 `*_q_cmd` 를 쓰면 첫 프레임만큼 앞선 자세로 이송하게 된다.
    """
    out: dict[str, np.ndarray] = {}
    for source_channel, suffix in (("joint_pos", ""), ("joint_pos_target", "_cmd")):
        source = ep.arrays.get(source_channel)
        if source is None:
            continue
        for label, attr in _INIT_GROUPS.items():
            names = getattr(meta, attr)
            columns = list(column_index(meta.joint_names, names))
            out[f"{label}{suffix}"] = source[0, columns].copy()
    for channel in POSE_CHANNELS:
        if channel in ep.arrays:
            out[channel] = ep.arrays[channel][0].copy()
    return out


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def write_traj(path: str | Path, meta: TrajMeta, episodes: tuple[Episode, ...]) -> Path:
    """검증을 **통과한 것만** 디스크에 남긴다. 실패하면 파일을 만들지 않는다."""
    problems = validate(meta, episodes)
    if problems:
        raise ValueError("궤적이 계약을 어긴다:\n  " + "\n  ".join(problems))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    try:
        with h5py.File(tmp, "w") as f:
            _write_meta(f, meta)
            for i, ep in enumerate(episodes):
                _write_episode(f.create_group(f"ep_{i:03d}"), meta, ep)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def _write_meta(f: h5py.File, meta: TrajMeta) -> None:
    text = h5py.string_dtype(encoding="utf-8")
    for key in ("task_id", "checkpoint", "checkpoint_sha256", "git_commit",
                "robot_usd", "env_yaml_sha256", "right_palm_body", "left_ee_body",
                "recorded_at"):
        f.attrs[key] = getattr(meta, key)
    f.attrs["schema_version"] = meta.schema_version
    f.attrs["dt"] = meta.dt
    f.attrs["decimation"] = meta.decimation
    f.attrs["num_beads"] = meta.num_beads
    f.attrs["env_origin"] = np.asarray(meta.env_origin, dtype=np.float64)
    f.attrs["robot_root"] = np.asarray(meta.robot_root, dtype=np.float64)
    for key in ("joint_names", "body_names", "missing_channels",
                "right_arm_joint_names", "right_hand_joint_names",
                "left_arm_joint_names", "left_gripper_joint_names"):
        f.attrs[key] = np.array(getattr(meta, key), dtype=text)


def _write_episode(g: h5py.Group, meta: TrajMeta, ep: Episode) -> None:
    for channel, array in ep.arrays.items():
        g.create_dataset(channel, data=array, compression="gzip")
    g.attrs["bead_frac"] = float(ep.bead_frac)
    g.attrs["bead_spill"] = float(ep.bead_spill)
    g.attrs["seed"] = int(ep.seed)
    g.attrs["success"] = _SUCCESS_UNKNOWN if ep.success is None else int(ep.success)
    g.attrs["n_steps"] = ep.n_steps
    for key, value in ep.extras.items():
        g.attrs[f"extra_{key}"] = float(value)

    init = g.create_group("init")
    for key, value in init_state(meta, ep).items():
        init.attrs[key] = np.asarray(value, dtype=np.float64)


def read_traj(path: str | Path) -> tuple[TrajMeta, tuple[Episode, ...]]:
    """v2 파일을 그대로 되읽는다. 스키마가 다르면 이름을 대고 거부한다."""
    with h5py.File(Path(path), "r") as f:
        version = int(f.attrs.get("schema_version", -1))
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: schema_version {version} — 이 모듈은 v{SCHEMA_VERSION} 만 읽는다"
                " (v1 은 migrate_v1 을 쓸 것)"
            )
        meta = _read_meta(f)
        episodes = tuple(
            _read_episode(f[key]) for key in sorted(f.keys()) if key.startswith("ep_")
        )
    return meta, episodes


def _read_meta(f: h5py.File) -> TrajMeta:
    def s(key: str) -> str:
        return _as_str(f.attrs[key])

    def names(key: str) -> tuple[str, ...]:
        return tuple(_as_str(x) for x in f.attrs[key])

    return TrajMeta(
        task_id=s("task_id"),
        checkpoint=s("checkpoint"),
        checkpoint_sha256=s("checkpoint_sha256"),
        git_commit=s("git_commit"),
        robot_usd=s("robot_usd"),
        env_yaml_sha256=s("env_yaml_sha256"),
        dt=float(f.attrs["dt"]),
        decimation=int(f.attrs["decimation"]),
        num_beads=int(f.attrs["num_beads"]),
        env_origin=tuple(np.asarray(f.attrs["env_origin"]).tolist()),
        robot_root=tuple(np.asarray(f.attrs["robot_root"]).tolist()),
        joint_names=names("joint_names"),
        body_names=names("body_names"),
        right_arm_joint_names=names("right_arm_joint_names"),
        right_hand_joint_names=names("right_hand_joint_names"),
        left_arm_joint_names=names("left_arm_joint_names"),
        left_gripper_joint_names=names("left_gripper_joint_names"),
        right_palm_body=s("right_palm_body"),
        left_ee_body=s("left_ee_body"),
        recorded_at=s("recorded_at"),
        missing_channels=names("missing_channels"),
        schema_version=int(f.attrs["schema_version"]),
    )


def _read_episode(g: h5py.Group) -> Episode:
    arrays = {k: g[k][:] for k in g.keys() if k in ALL_CHANNELS}
    success_raw = int(g.attrs.get("success", _SUCCESS_UNKNOWN))
    extras = {
        k[len("extra_"):]: float(v) for k, v in g.attrs.items() if k.startswith("extra_")
    }
    return Episode(
        arrays=arrays,
        bead_frac=float(g.attrs["bead_frac"]),
        bead_spill=float(g.attrs["bead_spill"]),
        seed=int(g.attrs.get("seed", _UNKNOWN_INT)),
        success=None if success_raw == _SUCCESS_UNKNOWN else bool(success_raw),
        extras=extras,
    )


def _as_str(value) -> str:
    return value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)


# ---------------------------------------------------------------------------
# v1 승격
# ---------------------------------------------------------------------------

#: v1(`probes/record_pour_traj.py`) 이 담은 것 → v2 채널명.
_V1_RENAME = {
    "joint_pos": "joint_pos",
    "source_pose": "source_cup_pose",
    "target_pose": "target_cup_pose",
}
_V1_LEFT_EE_CANDIDATES = ("l_hl_gripper_base", "openarm_left_hand")


def migrate_v1(path: str | Path) -> tuple[TrajMeta, tuple[Episode, ...]]:
    """구 스키마 파일을 v2 로 승격한다. **값은 만지지 않는다.**

    v1 이 안 담은 채널은 채워 넣지 않고 `missing_channels` 에 **선언**한다. NaN 으로
    메우면 검증은 통과하면서 재생 소스로는 못 쓰는 파일이 조용히 생긴다.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        joint_names = tuple(_as_str(x) for x in f.attrs["joint_names"])
        body_names = tuple(_as_str(x) for x in f.attrs.get("body_names", []))
        groups = split_joint_groups(joint_names)
        meta = TrajMeta(
            task_id=_as_str(f.attrs.get("task", "")),
            checkpoint=_as_str(f.attrs.get("checkpoint", "")),
            checkpoint_sha256="",
            git_commit="",
            robot_usd=_as_str(f.attrs.get("robot_usd", "")),
            env_yaml_sha256="",
            dt=float(f.attrs["dt"]),
            decimation=_UNKNOWN_INT,
            num_beads=int(f.attrs.get("num_beads", 0)),
            env_origin=tuple(np.asarray(f.attrs["env_origin"]).tolist()),
            robot_root=tuple(np.asarray(f.attrs["robot_root"]).tolist()),
            joint_names=joint_names,
            body_names=body_names,
            right_arm_joint_names=groups.right_arm,
            right_hand_joint_names=groups.right_hand,
            left_arm_joint_names=groups.left_arm,
            left_gripper_joint_names=groups.left_gripper,
            right_palm_body="r_hl_palm" if "r_hl_palm" in body_names else "",
            left_ee_body=next((b for b in _V1_LEFT_EE_CANDIDATES if b in body_names), ""),
            recorded_at=datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            missing_channels=tuple(c for c in ALL_CHANNELS if c not in _V1_RENAME.values()),
        )
        episodes = tuple(
            _migrate_v1_episode(f[key]) for key in sorted(f.keys()) if key.startswith("ep_")
        )
    return meta, episodes


def _migrate_v1_episode(g: h5py.Group) -> Episode:
    arrays = {new: g[old][:] for old, new in _V1_RENAME.items() if old in g}
    return Episode(
        arrays=arrays,
        bead_frac=float(g.attrs["bead_frac"]),
        bead_spill=float(g.attrs["bead_spill"]),
        seed=_UNKNOWN_INT,
        success=None,          # v1 은 성공 판정을 안 남겼다. 지어내지 않는다.
    )

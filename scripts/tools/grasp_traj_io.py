#!/usr/bin/env python3
"""grasp_s2r 궤적 HDF5(schema v1) 의 스키마·검증·왕복 IO. **Isaac 무의존.**

왜 pour 판(`pour_traj_io.py`)을 재사용하지 않는가 — 그 스키마는 `bead_frac`·
`num_beads`·`source/target_cup_pose` 가 **필수**라 grasp 에는 지어낼 값밖에 못 넣는다.
지어낸 값은 회귀 픽스처를 오염시킨다. 규약(포즈 표기·지령/측정 분리·결손 선언)은
그대로 따르고 채널만 이 태스크의 것으로 둔다.

이 파일을 읽는 쪽은 **Isaac 이 없는 실기 재생기**다. 그래서 h5py/numpy 만 쓴다.

포즈 표기: `[x, y, z, qw, qx, qy, qz]`, **env 원점 기준**(이 자산에서는 robot base), meter.

지령 vs 측정 — 실기로 나가는 것은 측정이 아니라 **지령**이다:
  · `*_q`      = `robot.data.joint_pos`        (측정, 스텝 **전**)
  · `*_q_cmd`  = `robot.data.joint_pos_target` (지령, 스텝 **후**)
재생기는 `*_q_cmd` 를 JTC 에 넣고 `*_q` 는 추종 오차 대조용으로만 쓴다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

SCHEMA_VERSION = 1

POSE_DIM = 7          # pos(3) + quat wxyz(4)
PALM_CMD_DIM = 6      # pos(3) + euler(3) — fabric palm 지령 규약
QUAT_SLICE = slice(3, 7)
QUAT_NORM_TOL = 1e-3

#: 팔 폭(= len(arm_joint_names)) 채널.
ARM_CHANNELS = ("arm_q", "arm_qd", "arm_q_cmd")
#: 손 폭(= len(hand_joint_names)) 채널.
HAND_CHANNELS = ("hand_q", "hand_qd", "hand_q_cmd")
#: POSE_DIM 폭 채널.
POSE_CHANNELS = ("palm_pose", "object_pose")
#: PALM_CMD_DIM 폭 채널.
PALM_CMD_CHANNELS = ("palm_cmd",)
#: 액션 폭 채널.
ACTION_CHANNELS = ("action",)
#: 스텝당 스칼라 채널.
SCALAR_CHANNELS = ("success", "stay_run", "goal_dist", "obj_height_delta")

ALL_CHANNELS = (
    ARM_CHANNELS + HAND_CHANNELS + POSE_CHANNELS
    + PALM_CMD_CHANNELS + ACTION_CHANNELS + SCALAR_CHANNELS
)


# ---------------------------------------------------------------------------
# 자료형
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GraspTrajMeta:
    """궤적 파일 한 개의 출처·좌표계·관절 배치. 전부 값 타입(불변)."""

    task_id: str
    checkpoint: str
    checkpoint_sha256: str
    git_commit: str
    robot_usd: str
    env_yaml_sha256: str
    dt: float
    decimation: int
    #: 이 파일이 대표하는 컵 소환 중심 (env-local x, y). 파일 이름이 아니라 여기가 진실.
    spawn_center_xy: tuple[float, float]
    #: 소환 xy 균등 반범위. 0 이면 중심 고정(결정론).
    spawn_range: float
    #: 물체 종 id 와 스케일 — "컵 1.0" 이 실제로 무엇이었는지 남긴다.
    object_species: str
    object_scale: float
    #: 목표 = 정착점 + 이 오프셋.
    goal_offset_xyz: tuple[float, float, float]
    env_origin: tuple[float, ...]
    robot_root: tuple[float, ...]
    arm_joint_names: tuple[str, ...]
    hand_joint_names: tuple[str, ...]
    palm_body: str
    recorded_at: str
    #: 이 태스크에 존재하지 않는 채널. 조용히 비면 재생 소스로 오인된다.
    missing_channels: tuple[str, ...] = ()

    @property
    def num_arm_joints(self) -> int:
        return len(self.arm_joint_names)

    @property
    def num_hand_joints(self) -> int:
        return len(self.hand_joint_names)


@dataclass(frozen=True)
class GraspEpisode:
    """에피소드 하나. `arrays` 의 모든 값은 (T, W) 또는 (T,) 이고 T 가 같아야 한다."""

    arrays: dict[str, np.ndarray]
    #: 성공 판정이 참이었던 스텝 수 / 마지막 연속 성공 구간 길이.
    success_steps: int
    success_tail: int
    goal_dist_final: float
    lift_max: float
    env_index: int
    seed: int

    @property
    def n_steps(self) -> int:
        return int(len(next(iter(self.arrays.values()))))

    @property
    def success(self) -> bool:
        return self.success_tail > 0


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------

def _expected_width(channel: str, meta: GraspTrajMeta, action_dim: int) -> int | None:
    """채널의 기대 폭. None = 스칼라(1D)."""
    if channel in ARM_CHANNELS:
        return meta.num_arm_joints
    if channel in HAND_CHANNELS:
        return meta.num_hand_joints
    if channel in POSE_CHANNELS:
        return POSE_DIM
    if channel in PALM_CMD_CHANNELS:
        return PALM_CMD_DIM
    if channel in ACTION_CHANNELS:
        return action_dim
    return None


def validate(meta: GraspTrajMeta, episodes: tuple[GraspEpisode, ...]) -> list[str]:
    """계약 위반을 **전부** 모아 돌려준다. 빈 리스트 = 통과.

    한 개만 던지고 멈추면 고치고 다시 돌리기를 반복하게 되므로 모아서 낸다.
    """
    problems: list[str] = []
    if not episodes:
        return ["에피소드가 하나도 없다"]
    if meta.dt <= 0.0:
        problems.append(f"dt 가 양수가 아니다: {meta.dt}")
    if meta.num_arm_joints == 0 or meta.num_hand_joints == 0:
        problems.append("arm/hand joint_names 가 비었다")

    expected = set(ALL_CHANNELS) - set(meta.missing_channels)
    action_dim = int(episodes[0].arrays["action"].shape[1]) if "action" in episodes[0].arrays else 0

    for idx, ep in enumerate(episodes):
        tag = f"ep{idx}"
        missing = sorted(expected - set(ep.arrays))
        if missing:
            problems.append(f"{tag}: 채널 누락 {missing}")
        extra = sorted(set(ep.arrays) - set(ALL_CHANNELS))
        if extra:
            problems.append(f"{tag}: 스키마에 없는 채널 {extra}")

        lengths = {k: int(v.shape[0]) for k, v in ep.arrays.items()}
        if len(set(lengths.values())) > 1:
            problems.append(f"{tag}: 채널 길이가 다르다 {lengths}")
        if ep.n_steps < 2:
            problems.append(f"{tag}: 길이 {ep.n_steps} — 궤적이라 부를 수 없다")

        for name, arr in ep.arrays.items():
            width = _expected_width(name, meta, action_dim)
            if width is None:
                if arr.ndim != 1:
                    problems.append(f"{tag}: {name} 는 스칼라여야 한다 (shape {arr.shape})")
                continue
            if arr.ndim != 2 or arr.shape[1] != width:
                problems.append(
                    f"{tag}: {name} shape {arr.shape} — 기대 (T, {width})")
            if not np.isfinite(arr).all():
                problems.append(f"{tag}: {name} 에 NaN/Inf 가 있다")

        for name in POSE_CHANNELS:
            arr = ep.arrays.get(name)
            if arr is None or arr.ndim != 2 or arr.shape[1] != POSE_DIM:
                continue
            norm = np.linalg.norm(arr[:, QUAT_SLICE], axis=1)
            if float(np.abs(norm - 1.0).max()) > QUAT_NORM_TOL:
                problems.append(
                    f"{tag}: {name} quat 이 정규화되지 않았다 "
                    f"(최대 편차 {float(np.abs(norm - 1.0).max()):.2e})")
    return problems


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _encode(value):
    """h5 attr 로 넣을 수 있는 형태로. 문자열 튜플은 가변길이 str 배열."""
    if isinstance(value, (tuple, list)):
        if value and isinstance(value[0], str):
            return np.array(list(value), dtype=h5py.string_dtype())
        return np.asarray(value, dtype=np.float64) if value else np.zeros(0)
    return value


def write_traj(path: str | Path, meta: GraspTrajMeta,
               episodes: tuple[GraspEpisode, ...]) -> Path:
    """검증 후 저장. 계약을 어기면 **쓰지 않고** 거부한다."""
    problems = validate(meta, episodes)
    if problems:
        raise ValueError("궤적이 스키마 계약을 어긴다:\n  " + "\n  ".join(problems))

    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as handle:
        handle.attrs["schema_version"] = SCHEMA_VERSION
        for key, value in asdict(meta).items():
            handle.attrs[key] = _encode(value)
        group = handle.create_group("episodes")
        for idx, ep in enumerate(episodes):
            sub = group.create_group(f"ep_{idx:03d}")
            sub.attrs["success_steps"] = ep.success_steps
            sub.attrs["success_tail"] = ep.success_tail
            sub.attrs["goal_dist_final"] = ep.goal_dist_final
            sub.attrs["lift_max"] = ep.lift_max
            sub.attrs["env_index"] = ep.env_index
            sub.attrs["seed"] = ep.seed
            sub.attrs["n_steps"] = ep.n_steps
            for name, arr in ep.arrays.items():
                sub.create_dataset(name, data=np.asarray(arr, dtype=np.float32),
                                   compression="gzip", compression_opts=4)
    return out


def _decode_str_tuple(value) -> tuple[str, ...]:
    return tuple(v.decode() if isinstance(v, bytes) else str(v) for v in value)


def read_traj(path: str | Path) -> tuple[GraspTrajMeta, tuple[GraspEpisode, ...]]:
    """저장본을 그대로 되읽는다. 왕복이 깨지면 여기서 드러난다."""
    src = Path(path).expanduser().resolve()
    with h5py.File(src, "r") as handle:
        version = int(handle.attrs["schema_version"])
        if version != SCHEMA_VERSION:
            raise ValueError(f"schema_version {version} — 이 모듈은 {SCHEMA_VERSION} 만 읽는다")
        attrs = handle.attrs
        meta = GraspTrajMeta(
            task_id=str(attrs["task_id"]),
            checkpoint=str(attrs["checkpoint"]),
            checkpoint_sha256=str(attrs["checkpoint_sha256"]),
            git_commit=str(attrs["git_commit"]),
            robot_usd=str(attrs["robot_usd"]),
            env_yaml_sha256=str(attrs["env_yaml_sha256"]),
            dt=float(attrs["dt"]),
            decimation=int(attrs["decimation"]),
            spawn_center_xy=tuple(float(v) for v in attrs["spawn_center_xy"]),
            spawn_range=float(attrs["spawn_range"]),
            object_species=str(attrs["object_species"]),
            object_scale=float(attrs["object_scale"]),
            goal_offset_xyz=tuple(float(v) for v in attrs["goal_offset_xyz"]),
            env_origin=tuple(float(v) for v in attrs["env_origin"]),
            robot_root=tuple(float(v) for v in attrs["robot_root"]),
            arm_joint_names=_decode_str_tuple(attrs["arm_joint_names"]),
            hand_joint_names=_decode_str_tuple(attrs["hand_joint_names"]),
            palm_body=str(attrs["palm_body"]),
            recorded_at=str(attrs["recorded_at"]),
            missing_channels=_decode_str_tuple(attrs["missing_channels"])
            if len(attrs["missing_channels"]) else (),
        )
        episodes = []
        for name in sorted(handle["episodes"]):
            sub = handle["episodes"][name]
            episodes.append(GraspEpisode(
                arrays={k: np.asarray(sub[k]) for k in sub},
                success_steps=int(sub.attrs["success_steps"]),
                success_tail=int(sub.attrs["success_tail"]),
                goal_dist_final=float(sub.attrs["goal_dist_final"]),
                lift_max=float(sub.attrs["lift_max"]),
                env_index=int(sub.attrs["env_index"]),
                seed=int(sub.attrs["seed"]),
            ))
    return meta, tuple(episodes)


# ---------------------------------------------------------------------------
# 실기 재생용 CSV
# ---------------------------------------------------------------------------

def write_command_csv(path: str | Path, meta: GraspTrajMeta,
                      episode: GraspEpisode) -> Path:
    """에피소드 하나의 **지령** 궤적을 CSV 로. 실기 재생기가 이 파일만 읽으면 된다.

    열: `t` + 팔 관절 지령 + 손 관절 지령. 이름을 헤더에 그대로 적어 열 순서를
    손으로 맞출 필요가 없게 한다(순서 착오가 좌우/손가락 뒤바뀜의 단골 원인이다).
    """
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    names = ("t",) + meta.arm_joint_names + meta.hand_joint_names
    stamps = (np.arange(episode.n_steps, dtype=np.float64) * meta.dt).reshape(-1, 1)
    table = np.concatenate(
        [stamps, episode.arrays["arm_q_cmd"], episode.arrays["hand_q_cmd"]], axis=1)
    header = ",".join(names)
    np.savetxt(out, table, delimiter=",", header=header, comments="", fmt="%.6f")
    return out

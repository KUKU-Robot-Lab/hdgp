"""pour_traj_io — pour 궤적 HDF5 스키마의 계약 테스트 (Isaac 무의존).

여기서 지키는 것:
  ① **왕복 무손실** — 뽑은 궤적이 저장/적재를 거치며 조용히 바뀌면, 실기가 재생한
     궤적과 sim 이 낸 궤적을 다른 채로 비교하게 된다.
  ② **선언되지 않은 결손을 거부** — 채널이 없으면 없다고 메타에 적혀 있어야 한다.
     `joint_pos_target` 이 슬그머니 빠진 파일을 재생 소스로 쓰면 지령이 아니라 측정을
     JTC 로 보내게 된다.
  ③ **관절 그룹이 joint_names 의 부분집합이고 서로 겹치지 않음** — 그룹이 곧 열
     인덱서다. 겹치거나 벗어나면 좌팔 지령이 우팔로 간다.
  ④ **quat 정규화·유한값** — 붓기는 자세가 전부라 quat 이 조금만 어긋나도 붓는
     지점이 컵 밖으로 나간다.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import pour_traj_io as PT  # noqa: E402


# ---------------------------------------------------------------------------
# 픽스처 — a1 자산(openarm_tesollo_sensor_rl)의 관절 배치를 축약 없이 흉내낸다.
# ---------------------------------------------------------------------------

_FINGERS = ("index", "middle", "pinky", "ring", "thumb")

# articulation 순서(관절번호-major, 좌우 교차)를 그대로 흉내낸다 — 실제 기록과 같은 함정.
JOINT_NAMES = (
    tuple(n for i in range(1, 8) for n in (f"l_aj_{i}", f"r_aj_{i}"))
    + ("l_hj_gripper_1", "l_hj_gripper_2")
    + tuple(f"r_hj_{f}_{j}" for j in range(1, 5) for f in _FINGERS)
    + ("head_j_pan", "head_j_tilt")          # 07.29 USD 교체로 붙은 관절
)
BODY_NAMES = ("body_root", "r_hl_palm", "openarm_left_hand")


def _unit_quat_series(t: int) -> np.ndarray:
    """정규화된 wxyz quat 시계열 — 상수가 아니라 실제로 도는 값."""
    ang = np.linspace(0.0, 1.0, t, dtype=np.float64)
    return np.stack(
        [np.cos(ang / 2), np.sin(ang / 2), np.zeros(t), np.zeros(t)], axis=1
    ).astype(np.float32)


def _pose_series(t: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(t, 3)).astype(np.float32)
    return np.concatenate([pos, _unit_quat_series(t)], axis=1)


def make_meta(**over) -> PT.TrajMeta:
    groups = PT.split_joint_groups(JOINT_NAMES)
    base = dict(
        task_id="open-tesol_r_pour_sensor-play-lstm",
        checkpoint="last_open-tesol_r_pour_v1-lstm_ep_10000_rew_47074.97.pth",
        checkpoint_sha256="0bc1a5bd" + "0" * 56,
        git_commit="deadbeef",
        robot_usd="/x/openarm_tesollo_sensor_rl.usd",
        env_yaml_sha256="34766ddf" + "0" * 56,
        dt=1.0 / 60.0,
        decimation=2,
        num_beads=20,
        env_origin=np.zeros(3, dtype=np.float32),
        robot_root=np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32),
        joint_names=JOINT_NAMES,
        body_names=BODY_NAMES,
        right_arm_joint_names=groups.right_arm,
        right_hand_joint_names=groups.right_hand,
        left_arm_joint_names=groups.left_arm,
        left_gripper_joint_names=groups.left_gripper,
        right_palm_body="r_hl_palm",
        left_ee_body="openarm_left_hand",
        recorded_at="2026-08-31T12:00:00",
        missing_channels=(),
    )
    base.update(over)
    return PT.TrajMeta(**base)


def make_episode(t: int = 25, j: int = len(JOINT_NAMES), a: int = 12, seed: int = 0,
                 drop: tuple[str, ...] = ()) -> PT.Episode:
    rng = np.random.default_rng(seed)
    arrays = {
        "joint_pos": rng.normal(size=(t, j)).astype(np.float32),
        "joint_vel": rng.normal(size=(t, j)).astype(np.float32),
        "joint_pos_target": rng.normal(size=(t, j)).astype(np.float32),
        "action": rng.normal(size=(t, a)).astype(np.float32),
        "right_palm_pose": _pose_series(t, seed + 1),
        "left_ee_pose": _pose_series(t, seed + 2),
        "right_palm_cmd_pose": _pose_series(t, seed + 3),
        "left_ee_cmd_pose": _pose_series(t, seed + 7),
        "source_cup_pose": _pose_series(t, seed + 4),
        "target_cup_pose": _pose_series(t, seed + 5),
        "cup_in_hand_pose": _pose_series(t, seed + 6),
    }
    for k in drop:
        arrays.pop(k)
    return PT.Episode(arrays=arrays, bead_frac=1.0, bead_spill=0.0, seed=42, success=True)


# ---------------------------------------------------------------------------
# split_joint_groups
# ---------------------------------------------------------------------------

def test_split_joint_groups_covers_a1_layout():
    g = PT.split_joint_groups(JOINT_NAMES)
    assert g.right_arm == tuple(f"r_aj_{i}" for i in range(1, 8))
    assert g.left_arm == tuple(f"l_aj_{i}" for i in range(1, 8))
    assert g.left_gripper == ("l_hj_gripper_1", "l_hj_gripper_2")
    assert len(g.right_hand) == 20
    # head 는 어느 팔 그룹에도 안 들어간다 (obs/action 은 이름 기반이라 차원 불변).
    assert g.other == ("head_j_pan", "head_j_tilt")


def test_split_joint_groups_preserves_joint_names_order():
    """그룹은 곧 열 인덱서다. 이름 순서가 joint_names 순서와 어긋나면 열이 뒤바뀐다."""
    g = PT.split_joint_groups(JOINT_NAMES)
    for names in (g.right_arm, g.left_arm, g.right_hand, g.left_gripper):
        idx = [JOINT_NAMES.index(n) for n in names]
        assert idx == sorted(idx)


def test_column_index_maps_group_to_columns():
    g = PT.split_joint_groups(JOINT_NAMES)
    idx = PT.column_index(JOINT_NAMES, g.right_arm)
    assert [JOINT_NAMES[i] for i in idx] == list(g.right_arm)


# ---------------------------------------------------------------------------
# 왕복
# ---------------------------------------------------------------------------

def test_roundtrip_is_lossless(tmp_path: Path):
    meta, eps = make_meta(), (make_episode(seed=1), make_episode(t=31, seed=2))
    path = tmp_path / "traj.hdf5"
    PT.write_traj(path, meta, eps)
    r_meta, r_eps = PT.read_traj(path)

    assert r_meta == meta
    assert len(r_eps) == len(eps)
    for got, want in zip(r_eps, eps):
        assert set(got.arrays) == set(want.arrays)
        for k in want.arrays:
            np.testing.assert_array_equal(got.arrays[k], want.arrays[k])
        assert (got.bead_frac, got.bead_spill, got.seed, got.success) == (
            want.bead_frac, want.bead_spill, want.seed, want.success)


def test_write_rejects_invalid_payload(tmp_path: Path):
    """검증을 통과 못 한 궤적은 애초에 디스크에 남지 않는다."""
    bad = make_episode()
    bad.arrays["joint_vel"] = bad.arrays["joint_vel"][:-1]
    with pytest.raises(ValueError, match="joint_vel"):
        PT.write_traj(tmp_path / "bad.hdf5", make_meta(), (bad,))
    assert not (tmp_path / "bad.hdf5").exists()


# ---------------------------------------------------------------------------
# init 단면
# ---------------------------------------------------------------------------

def test_init_state_matches_first_frame():
    meta, ep = make_meta(), make_episode(seed=7)
    init = PT.init_state(meta, ep)
    np.testing.assert_array_equal(init["right_palm_pose"], ep.arrays["right_palm_pose"][0])
    np.testing.assert_array_equal(init["cup_in_hand_pose"], ep.arrays["cup_in_hand_pose"][0])
    cols = list(PT.column_index(JOINT_NAMES, meta.right_arm_joint_names))
    np.testing.assert_array_equal(init["right_arm_q"], ep.arrays["joint_pos"][0, cols])
    assert init["right_hand_q"].shape == (20,)
    assert init["left_gripper_q"].shape == (2,)


def test_init_state_separates_measured_from_commanded():
    """파킹 목표는 **측정** warm state 다. 첫 지령으로 파킹하면 한 프레임 앞선다."""
    meta, ep = make_meta(), make_episode(seed=7)
    init = PT.init_state(meta, ep)
    cols = list(PT.column_index(JOINT_NAMES, meta.right_arm_joint_names))
    np.testing.assert_array_equal(init["right_arm_q"], ep.arrays["joint_pos"][0, cols])
    np.testing.assert_array_equal(init["right_arm_q_cmd"], ep.arrays["joint_pos_target"][0, cols])
    assert not np.array_equal(init["right_arm_q"], init["right_arm_q_cmd"])


def test_init_state_omits_commanded_when_channel_missing():
    """v1 승격본처럼 지령이 없으면 `*_q_cmd` 를 지어내지 않는다."""
    ep = make_episode(seed=7, drop=("joint_pos_target",))
    meta = make_meta(missing_channels=("joint_pos_target",))
    init = PT.init_state(meta, ep)
    assert "right_arm_q" in init
    assert not any(k.endswith("_q_cmd") for k in init)


def test_init_group_survives_roundtrip(tmp_path: Path):
    meta, ep = make_meta(), make_episode(seed=8)
    path = tmp_path / "traj.hdf5"
    PT.write_traj(path, meta, (ep,))
    import h5py

    with h5py.File(path, "r") as f:
        stored = f["ep_000/init"].attrs["right_arm_q"]
    np.testing.assert_allclose(stored, PT.init_state(meta, ep)["right_arm_q"])


# ---------------------------------------------------------------------------
# validate — 거부해야 할 것들
# ---------------------------------------------------------------------------

def test_validate_accepts_wellformed():
    assert PT.validate(make_meta(), (make_episode(),)) == ()


def test_validate_rejects_length_mismatch():
    ep = make_episode()
    ep.arrays["action"] = ep.arrays["action"][:-3]
    problems = PT.validate(make_meta(), (ep,))
    assert any("action" in p and "길이" in p for p in problems)


def test_validate_rejects_wrong_joint_width():
    ep = make_episode(j=len(JOINT_NAMES) - 1)
    problems = PT.validate(make_meta(), (ep,))
    assert any("joint_pos" in p for p in problems)


def test_validate_rejects_nan():
    ep = make_episode()
    ep.arrays["joint_pos_target"][3, 2] = np.nan
    problems = PT.validate(make_meta(), (ep,))
    assert any("joint_pos_target" in p and "유한" in p for p in problems)


def test_validate_rejects_unnormalized_quat():
    ep = make_episode()
    ep.arrays["source_cup_pose"][5, 3:] *= 2.0
    problems = PT.validate(make_meta(), (ep,))
    assert any("source_cup_pose" in p and "quat" in p for p in problems)


def test_validate_rejects_undeclared_missing_channel():
    ep = make_episode(drop=("joint_pos_target",))
    problems = PT.validate(make_meta(), (ep,))
    assert any("joint_pos_target" in p and "없" in p for p in problems)


def test_validate_accepts_declared_missing_channel():
    ep = make_episode(drop=("joint_pos_target", "action"))
    meta = make_meta(missing_channels=("joint_pos_target", "action"))
    assert PT.validate(meta, (ep,)) == ()


def test_validate_rejects_declared_missing_but_present():
    """모순된 메타는 조용히 넘기면 안 된다 — 어느 쪽을 믿을지 알 수 없어진다."""
    meta = make_meta(missing_channels=("action",))
    problems = PT.validate(meta, (make_episode(),))
    assert any("action" in p and "모순" in p for p in problems)


def test_validate_rejects_group_not_subset():
    meta = make_meta(right_arm_joint_names=("r_aj_1", "r_aj_99"))
    problems = PT.validate(meta, (make_episode(),))
    assert any("r_aj_99" in p for p in problems)


def test_validate_rejects_group_overlap():
    meta = make_meta(left_gripper_joint_names=("l_hj_gripper_1", "l_aj_1"))
    problems = PT.validate(meta, (make_episode(),))
    assert any("겹" in p for p in problems)


def test_validate_rejects_nonpositive_dt():
    problems = PT.validate(make_meta(dt=0.0), (make_episode(),))
    assert any("dt" in p for p in problems)


def test_validate_rejects_bead_frac_out_of_range():
    ep = replace(make_episode(), bead_frac=1.4)
    problems = PT.validate(make_meta(), (ep,))
    assert any("bead_frac" in p for p in problems)


def test_validate_rejects_empty_episode_list():
    assert any("에피소드" in p for p in PT.validate(make_meta(), ()))


# ---------------------------------------------------------------------------
# v1 마이그레이션 — 실제 아카이브 산출물로 검증
# ---------------------------------------------------------------------------

_ARCHIVE_V1 = Path(
    "/home/user/rl_ws/archive/260802/log/rl_games/open-tesol/right/pour-v1/"
    "lstm_test2/pour_traj.hdf5"
)


@pytest.mark.skipif(not _ARCHIVE_V1.is_file(), reason="아카이브 v1 궤적 없음")
def test_migrate_v1_from_archive(tmp_path: Path):
    meta, eps = PT.migrate_v1(_ARCHIVE_V1)

    assert meta.schema_version == PT.SCHEMA_VERSION
    assert len(eps) == 8
    # v1 이 안 담은 채널은 **선언**되어야 한다. 조용히 비면 재생 소스로 오인된다.
    assert "joint_pos_target" in meta.missing_channels
    assert "cup_in_hand_pose" in meta.missing_channels
    assert "left_ee_cmd_pose" in meta.missing_channels
    assert "source_cup_pose" not in meta.missing_channels
    # 승격 결과가 스스로의 계약을 통과한다.
    assert PT.validate(meta, eps) == ()
    # 그리고 저장·재적재까지 된다.
    out = tmp_path / "v2.hdf5"
    PT.write_traj(out, meta, eps)
    r_meta, r_eps = PT.read_traj(out)
    assert r_meta == meta
    np.testing.assert_array_equal(r_eps[0].arrays["joint_pos"], eps[0].arrays["joint_pos"])


@pytest.mark.skipif(not _ARCHIVE_V1.is_file(), reason="아카이브 v1 궤적 없음")
def test_migrate_v1_preserves_recorded_values():
    """승격이 값을 만지지 않는다 — 원본 hdf5 와 바이트 단위로 같아야 한다."""
    import h5py

    _, eps = PT.migrate_v1(_ARCHIVE_V1)
    with h5py.File(_ARCHIVE_V1, "r") as f:
        np.testing.assert_array_equal(eps[0].arrays["joint_pos"], f["ep_000/joint_pos"][:])
        np.testing.assert_array_equal(
            eps[0].arrays["source_cup_pose"], f["ep_000/source_pose"][:]
        )
        assert eps[0].bead_frac == pytest.approx(f["ep_000"].attrs["bead_frac"])

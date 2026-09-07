"""pour_traj_report — 리포트가 HDF5 와 **어긋나지 않는지**만 지킨다.

리포트는 사람이 읽으려고 만드는 파생물이다. 파생물이 원본과 갈리면 리포트를 믿고
잘못 판단하게 되므로, 여기서는 서식이 아니라 **값의 일치**와 **결손 채널의 정직한 표시**를
테스트한다.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import pour_traj_io as PT  # noqa: E402
import pour_traj_report as PR  # noqa: E402
from test_pour_traj_io import JOINT_NAMES, make_episode, make_meta  # noqa: E402


def test_tilt_deg_upright_is_zero():
    quat = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(PR.tilt_deg(quat), [0.0], atol=1e-5)


def test_tilt_deg_90_about_x():
    s = 2.0 ** -0.5
    quat = np.array([[s, s, 0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(PR.tilt_deg(quat), [90.0], atol=1e-4)


def test_episode_rows_carry_source_values():
    meta, eps = make_meta(), (make_episode(t=30, seed=3), make_episode(t=41, seed=4))
    rows = PR.episode_rows(meta, eps)
    assert [r["steps"] for r in rows] == [30, 41]
    assert rows[0]["seconds"] == pytest.approx(30 * meta.dt)
    assert rows[0]["bead"] == eps[0].bead_frac


def test_speed_stats_match_the_arrays():
    meta, ep = make_meta(), make_episode(seed=5)
    stats = PR.joint_speed_stats(meta, (ep,))
    assert len(stats) == len(JOINT_NAMES)
    peak = np.abs(ep.arrays["joint_vel"]).max(axis=0)
    assert stats[0]["peak"] == pytest.approx(peak[0])
    assert stats[0]["joint"] == JOINT_NAMES[0]


def test_speed_stats_absent_when_channel_declared_missing():
    meta = make_meta(missing_channels=("joint_vel",))
    ep = make_episode(seed=5, drop=("joint_vel",))
    assert PR.joint_speed_stats(meta, (ep,)) == []


def test_group_ranges_flag_a_frozen_group():
    """좌팔이 정지인 트랙에서 이동폭 0 이 그대로 드러나야 한다."""
    meta, ep = make_meta(), make_episode(seed=6)
    cols = list(PT.column_index(JOINT_NAMES, meta.left_arm_joint_names))
    ep.arrays["joint_pos"][:, cols] = 0.25
    rows = {r["group"]: r for r in PR.group_ranges(meta, (ep,))}
    assert rows["좌팔"]["max_span_rad"] == pytest.approx(0.0)
    assert rows["우팔"]["max_span_rad"] > 0.0


def test_render_includes_source_and_init_values():
    meta, ep = make_meta(), make_episode(seed=7)
    text = PR.render_markdown(meta, (ep,), Path("traj.hdf5"))
    assert meta.task_id in text
    assert meta.checkpoint in text
    # 초기 우팔 첫 관절값이 본문에 그대로 있어야 한다 (파생물이 원본과 갈리지 않는다).
    q0 = PT.init_state(meta, ep)["right_arm_q"][0]
    assert f"{q0:.4f}" in text


def test_render_marks_missing_channels_honestly():
    meta = make_meta(missing_channels=("joint_vel", "cup_in_hand_pose"))
    ep = make_episode(seed=8, drop=("joint_vel", "cup_in_hand_pose"))
    text = PR.render_markdown(meta, (ep,), Path("traj.hdf5"))
    assert "`joint_vel`" in text and "`cup_in_hand_pose`" in text
    assert "관절 속도 (전 에피소드" not in text      # 없는 채널로 표를 지어내지 않는다
    assert "| — |" in text or "—" in text


def test_render_survives_v1_style_degraded_file():
    """승격본처럼 채널이 대거 빠져도 죽지 않고 렌더된다."""
    dropped = ("joint_vel", "joint_pos_target", "action", "right_palm_pose",
               "left_ee_pose", "right_palm_cmd_pose", "left_ee_cmd_pose", "cup_in_hand_pose")
    meta = make_meta(missing_channels=dropped)
    ep = make_episode(seed=9, drop=dropped)
    text = PR.render_markdown(meta, (ep,), Path("v1.hdf5"))
    assert "## 에피소드" in text and "## 관절 순서" in text

"""pour_traj_capture — env 에서 채널을 뽑는 어댑터의 계약 테스트 (Isaac 무의존).

여기서 지키는 것:
  ① **quat 규약** — `palm_pose_targets` 는 sim 안에서 **xyzw** 인데 파일 규약은 wxyz 다.
     한 번 뒤집히면 붓는 자세가 통째로 틀리는데, 값은 여전히 정규화돼 있어서 검증을
     통과한다. 그래서 여기서 잠근다.
  ② **palm_ee ≠ r_hl_palm** — 제어·관측 기준은 `r_hl_palm` 이 아니라 로컬 오프셋
     (0.028, 0, 0.04) 만큼 앞선 palm_ee 다. body 포즈를 그대로 쓰면 손바닥 두께만큼 어긋난다.
  ③ **cup_in_hand_pose 가 진짜 상대자세** — 팔이 어디 있든 컵을 손에 같은 자세로 쥐면
     같은 값이 나와야 한다. 이 불변량이 깨지면 grasp 결과와 pour 초기의 비교가 무의미해진다.
  ④ **왼팔 지령 채널의 있고 없음이 태스크로 결정** — right 는 왼팔이 정지라 지령이 없다.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import pour_traj_capture as PC  # noqa: E402
import pour_traj_io as PT  # noqa: E402


# ---------------------------------------------------------------------------
# quat 유틸
# ---------------------------------------------------------------------------

def _q(w, x, y, z) -> torch.Tensor:
    return torch.tensor([[w, x, y, z]], dtype=torch.float64)


def test_quat_mul_identity():
    q = _q(0.5, 0.5, 0.5, 0.5)
    ident = _q(1.0, 0.0, 0.0, 0.0)
    torch.testing.assert_close(PC.quat_mul(q, ident), q)
    torch.testing.assert_close(PC.quat_mul(ident, q), q)


def test_quat_conj_inverts():
    q = _q(0.5, 0.5, 0.5, 0.5)
    torch.testing.assert_close(PC.quat_mul(q, PC.quat_conj(q)), _q(1.0, 0.0, 0.0, 0.0))


def test_quat_rotate_inv_undoes_rotation():
    """z축 90° 회전의 역회전은 x축 벡터를 -y 로 보낸다."""
    s = 2.0 ** -0.5
    q = _q(s, 0.0, 0.0, s)                      # +90° about z
    v = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(
        PC.quat_rotate_inv(q, v), torch.tensor([[0.0, -1.0, 0.0]], dtype=torch.float64),
        atol=1e-12, rtol=0,
    )


def test_xyzw_to_wxyz_moves_scalar_to_front():
    out = PC.xyzw_to_wxyz(torch.tensor([[0.1, 0.2, 0.3, 0.9]], dtype=torch.float64))
    torch.testing.assert_close(out, torch.tensor([[0.9, 0.1, 0.2, 0.3]], dtype=torch.float64))


# ---------------------------------------------------------------------------
# relative_pose
# ---------------------------------------------------------------------------

def test_relative_pose_of_identical_frames_is_identity():
    p = torch.tensor([[0.3, -0.2, 0.5]], dtype=torch.float64)
    q = _q(0.5, 0.5, 0.5, 0.5)
    rel = PC.relative_pose(p, q, p, q)
    torch.testing.assert_close(rel[:, :3], torch.zeros(1, 3, dtype=torch.float64))
    torch.testing.assert_close(rel[:, 3:], _q(1.0, 0.0, 0.0, 0.0))


def test_relative_pose_is_invariant_to_rigid_motion():
    """팔이 어디 있든, 컵을 손에 같은 자세로 쥐면 cup_in_hand 는 같아야 한다."""
    s = 2.0 ** -0.5
    parent_p = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64)
    parent_q = _q(1.0, 0.0, 0.0, 0.0)
    child_p = torch.tensor([[0.15, 0.2, 0.35]], dtype=torch.float64)
    child_q = _q(s, s, 0.0, 0.0)
    before = PC.relative_pose(parent_p, parent_q, child_p, child_q)

    # 두 프레임을 함께 z축 90° 돌리고 평행이동 — 상대자세는 불변이어야 한다.
    motion = _q(s, 0.0, 0.0, s)
    shift = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float64)
    after = PC.relative_pose(
        PC.quat_rotate(motion, parent_p) + shift, PC.quat_mul(motion, parent_q),
        PC.quat_rotate(motion, child_p) + shift, PC.quat_mul(motion, child_q),
    )
    torch.testing.assert_close(after, before, atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# 태스크 레이아웃
# ---------------------------------------------------------------------------

def test_layout_for_right_has_no_left_command():
    layout = PC.layout_for("open-tesol_r_pour_sensor-play-lstm")
    assert layout.left_active is False
    assert "left_ee_cmd_pose" in layout.missing_channels


def test_layout_for_both_has_left_command():
    layout = PC.layout_for("open-tesol_b_pour_sensor-lstm")
    assert layout.left_active is True
    assert layout.missing_channels == ()


def test_layout_for_accepts_pre_rename_task_id():
    """`right/pour_v1` → `right/pour_sensor` 개명(4aed79a). 구 id 체크포인트가 아직 정본이다."""
    assert PC.layout_for("open-tesol_r_pour_v1-play-lstm") is PC.layout_for(
        "open-tesol_r_pour_sensor-play-lstm")


def test_layout_for_rejects_unknown_task():
    with pytest.raises(KeyError, match="pour"):
        PC.layout_for("open-tesol_r_grasp_v1")


def test_layout_missing_channels_are_known_channels():
    for task in ("open-tesol_r_pour_sensor", "open-tesol_b_pour_sensor"):
        for channel in PC.layout_for(task).missing_channels:
            assert channel in PT.ALL_CHANNELS


# ---------------------------------------------------------------------------
# 캡처 — stub env
# ---------------------------------------------------------------------------

_FINGERS = ("index", "middle", "pinky", "ring", "thumb")
JOINT_NAMES = (
    [n for i in range(1, 8) for n in (f"l_aj_{i}", f"r_aj_{i}")]
    + ["l_hj_gripper_1", "l_hj_gripper_2"]
    + [f"r_hj_{f}_{j}" for j in range(1, 5) for f in _FINGERS]
)
BODY_NAMES = ["body_root", "r_hl_palm", "l_hl_gripper_base"]

PALM_POS_W = [1.0, 2.0, 3.0]
PALM_QUAT_W = [1.0, 0.0, 0.0, 0.0]           # 무회전 — 오프셋이 그대로 더해진다
PALM_EE_OFFSET = [0.028, 0.0, 0.04]
ENV_ORIGIN = [10.0, 0.0, 0.0]
CUP_POS_W = [1.5, 2.0, 3.2]


def _rep(row, n):
    return torch.tensor([list(row)], dtype=torch.float32).repeat(n, 1)


class _Data:
    def __init__(self, n, n_joints, n_bodies):
        self.joint_pos = torch.arange(n_joints, dtype=torch.float32).reshape(1, -1).repeat(n, 1)
        self.joint_vel = self.joint_pos * 0.1
        self.joint_pos_target = self.joint_pos + 0.5
        self.body_pos_w = torch.zeros(n, n_bodies, 3)
        self.body_quat_w = torch.zeros(n, n_bodies, 4)
        self.body_quat_w[:, :, 0] = 1.0
        self.body_names = BODY_NAMES
        self.joint_names = JOINT_NAMES


class _Rigid:
    def __init__(self, pos, n, quat=(1.0, 0.0, 0.0, 0.0)):
        self.data = type("D", (), {})()
        self.data.root_pos_w = _rep(pos, n)
        self.data.root_quat_w = _rep(quat, n)


class _Robot:
    def __init__(self, n):
        self.data = _Data(n, len(JOINT_NAMES), len(BODY_NAMES))
        self.data.body_pos_w[:, BODY_NAMES.index("r_hl_palm")] = torch.tensor(PALM_POS_W)
        self.data.body_pos_w[:, BODY_NAMES.index("l_hl_gripper_base")] = torch.tensor([0.5, -1.0, 2.0])
        self.joint_names = JOINT_NAMES


class _StubEnv:
    """pour env 가 노출하는 계약만 흉내낸다."""

    def __init__(self, left_active: bool, n: int = 1):
        self.num_envs = n
        self.device = "cpu"
        self.robot = _Robot(n)
        self.scene = type("S", (), {})()
        self.scene.env_origins = _rep(ENV_ORIGIN, n)
        self.cup = _Rigid(CUP_POS_W, n)
        self.left_target_cup = _Rigid([0.6, -1.0, 2.1], n)
        self.palm_body_index = BODY_NAMES.index("r_hl_palm")
        self._palm_ee_offset_local = torch.tensor(PALM_EE_OFFSET)
        # 지령 quat 은 sim 안에서 xyzw 다 (x=0.1, y=0.2, z=0.3, w=0.9 정규화)
        raw = torch.tensor([0.1, 0.2, 0.3, 0.9])
        raw = raw / raw.norm()
        self.palm_pose_targets = torch.cat([_rep([0.4, 0.5, 0.6], n), _rep(raw.tolist(), n)], dim=1)
        self._last_done_bead = torch.full((n,), 0.85)
        self._last_done_spill = torch.full((n,), 0.05)
        if left_active:
            self.left_tcp_target_pos_b = _rep([0.7, -0.8, 0.9], n)
            self.left_tcp_fixed_quat_b = _rep([1.0, 0.0, 0.0, 0.0], n)


def _capture(left_active: bool, n: int = 1) -> PC.PourTrajCapture:
    task = "open-tesol_b_pour_sensor" if left_active else "open-tesol_r_pour_sensor"
    return PC.PourTrajCapture(_StubEnv(left_active, n), PC.layout_for(task))


def test_state_frame_has_exactly_the_measured_channels():
    frame = _capture(False).state_frame()
    assert set(frame) == set(PC.STATE_CHANNELS)


def test_command_frame_channels_follow_layout():
    assert set(_capture(False).command_frame()) == {"joint_pos_target", "right_palm_cmd_pose"}
    assert set(_capture(True).command_frame()) == {
        "joint_pos_target", "right_palm_cmd_pose", "left_ee_cmd_pose"}


def test_right_palm_pose_uses_palm_ee_not_body_origin():
    frame = _capture(False).state_frame()
    expected = np.array(PALM_POS_W) + np.array(PALM_EE_OFFSET) - np.array(ENV_ORIGIN)
    np.testing.assert_allclose(frame["right_palm_pose"][0, :3], expected, atol=1e-6)
    np.testing.assert_allclose(frame["right_palm_pose"][0, 3:], PALM_QUAT_W, atol=1e-6)


def test_poses_are_env_local():
    """world 원점이 아니라 env 원점 기준이어야 한다 — 다중 env 로 늘려도 값이 안 흔들린다."""
    frame = _capture(False).state_frame()
    np.testing.assert_allclose(
        frame["source_cup_pose"][0, :3], np.array(CUP_POS_W) - np.array(ENV_ORIGIN), atol=1e-6)


def test_command_quat_is_converted_to_wxyz():
    frame = _capture(False).command_frame()
    quat = frame["right_palm_cmd_pose"][0, 3:]
    raw = np.array([0.1, 0.2, 0.3, 0.9]) / np.linalg.norm([0.1, 0.2, 0.3, 0.9])
    np.testing.assert_allclose(quat, np.array([raw[3], raw[0], raw[1], raw[2]]), atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(quat), 1.0, atol=1e-6)


def test_cup_in_hand_pose_is_cup_relative_to_palm_ee():
    frame = _capture(False).state_frame()
    expected = np.array(CUP_POS_W) - (np.array(PALM_POS_W) + np.array(PALM_EE_OFFSET))
    np.testing.assert_allclose(frame["cup_in_hand_pose"][0, :3], expected, atol=1e-6)


def test_all_frames_are_float32_and_correct_shape():
    n = 5
    cap = _capture(True, n=n)
    for frame in (cap.state_frame(), cap.command_frame()):
        for name, value in frame.items():
            assert value.dtype == np.float32, name
            width = len(JOINT_NAMES) if name in PT.JOINT_CHANNELS else PT.POSE_DIM
            assert value.shape == (n, width), (name, value.shape)


def test_capture_is_batched_over_envs():
    """1 env 로 묶으면 스텝 오버헤드가 지배해 수집이 env 수만큼 느려진다."""
    n = 7
    cap = _capture(False, n=n)
    assert cap.num_envs == n
    frame = cap.state_frame()
    assert frame["joint_pos"].shape[0] == n
    # 모든 env 가 같은 stub 상태라 값도 같아야 한다 (축을 잘못 접으면 깨진다).
    np.testing.assert_allclose(frame["right_palm_pose"][0], frame["right_palm_pose"][-1])


def test_outcome_reads_done_metrics_per_env():
    bead, spill = _capture(False, n=3).outcome()
    assert bead.shape == (3,) and spill.shape == (3,)
    assert bead[0] == pytest.approx(0.85)
    assert spill[0] == pytest.approx(0.05)


def test_capture_rejects_env_missing_contract():
    env = _StubEnv(False)
    del env.palm_pose_targets
    with pytest.raises(AttributeError, match="palm_pose_targets"):
        PC.PourTrajCapture(env, PC.layout_for("open-tesol_r_pour_sensor"))


def test_joint_groups_come_from_env_joint_names():
    cap = _capture(False)
    assert cap.groups.right_arm == tuple(f"r_aj_{i}" for i in range(1, 8))
    assert len(cap.groups.right_hand) == 20

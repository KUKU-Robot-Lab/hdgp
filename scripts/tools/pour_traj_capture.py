#!/usr/bin/env python3
"""pour env → 궤적 채널 어댑터. **torch 만 쓰고 Isaac 을 import 하지 않는다.**

기록기(`probes/record_pour_traj_s2r.py`)는 Isaac 을 띄워야 돌지만, "무엇을 어디서
읽어 어떤 규약으로 적는가" 는 시뮬레이터 없이 잠글 수 있다. 그 부분만 여기 두고
stub env 로 테스트한다.

여기서 조용히 넘어가면 안 되는 것 셋:

1. **quat 규약이 두 개다.** `robot.data.*_quat_w` 는 wxyz, `palm_pose_targets` 는
   **xyzw**(env 내부에서 `[:, [3,0,1,2]]` 로 되돌린다). 파일 규약은 wxyz 하나뿐이라
   지령만 뒤집어 적는다. 한 번 틀리면 값은 정규화된 채로 자세만 통째로 어긋난다.
2. **제어 기준은 `r_hl_palm` 이 아니라 palm_ee** 다. URDF `palm_link_to_ee` 고정조인트
   origin 만큼(로컬 0.028, 0, 0.04) 앞서 있다. body 원점을 그대로 쓰면 손바닥 두께만큼
   어긋난 궤적을 실기에 보낸다.
3. **왼팔 지령은 태스크마다 있고 없다.** `right/pour_sensor` 는 왼팔이 rest 고정이라
   지령 채널 자체가 없다. 없는 것은 지어내지 않고 `missing_channels` 로 **선언**한다.

포즈는 전부 `[x, y, z, qw, qx, qy, qz]`, **env 원점 기준**(= 이 자산에서는 robot base).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pour_traj_io import JointGroups, split_joint_groups

#: 스텝 **전**에 찍는 측정 채널.
STATE_CHANNELS = (
    "joint_pos",
    "joint_vel",
    "right_palm_pose",
    "left_ee_pose",
    "source_cup_pose",
    "target_cup_pose",
    "cup_in_hand_pose",
)
#: 스텝 **후**에 찍는 지령 채널 (왼팔 지령은 레이아웃에 따라 빠진다).
COMMAND_CHANNELS = ("joint_pos_target", "right_palm_cmd_pose", "left_ee_cmd_pose")

RIGHT_PALM_BODY = "r_hl_palm"
LEFT_EE_BODY = "l_hl_gripper_base"

#: env 가 이걸 노출하지 않으면 계약이 깨진 것이다. 이름을 대고 거부한다.
_REQUIRED_ATTRS = (
    "robot", "scene", "cup", "left_target_cup",
    "palm_body_index", "_palm_ee_offset_local", "palm_pose_targets",
    "_last_done_bead", "_last_done_spill",
)
_REQUIRED_LEFT_ATTRS = ("left_tcp_target_pos_b", "left_tcp_fixed_quat_b")


# ---------------------------------------------------------------------------
# quat (wxyz)
# ---------------------------------------------------------------------------

def quat_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_rotate_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_rotate(quat_conj(q), v)


def xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    return q[..., [3, 0, 1, 2]]


def relative_pose(
    parent_pos: torch.Tensor,
    parent_quat: torch.Tensor,
    child_pos: torch.Tensor,
    child_quat: torch.Tensor,
) -> torch.Tensor:
    """child 를 parent 프레임에서 본 포즈 (…, 7). 강체운동에 불변이다."""
    pos = quat_rotate_inv(parent_quat, child_pos - parent_pos)
    quat = quat_mul(quat_conj(parent_quat), child_quat)
    return torch.cat([pos, quat], dim=-1)


# ---------------------------------------------------------------------------
# 태스크 레이아웃
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskLayout:
    """태스크마다 **다른 것만** 담는다."""

    name: str
    left_active: bool
    num_actions: int
    #: gym task id 에서 찾을 부분문자열. 개명 이력이 있어 한 레이아웃에 여러 개가 붙는다.
    match: tuple[str, ...] = ()

    @property
    def missing_channels(self) -> tuple[str, ...]:
        return () if self.left_active else ("left_ee_cmd_pose",)


_LAYOUTS = (
    # right/pour_sensor: palm 6 + nullspace 1 + hand 5. 왼팔은 rest 고정.
    #   ★`r_pour_v1` 은 같은 태스크의 개명 전 이름이다(커밋 4aed79a, 전 파일 R100 이동).
    #     2026-07 이전 체크포인트는 그 id 로 저장돼 있어 재생 시 그대로 쓰인다.
    TaskLayout(name="r_pour_sensor", left_active=False, num_actions=12,
               match=("r_pour_sensor", "r_pour_v1")),
    # both/pour_sensor: 위 12 + 왼팔 TCP 위치 delta 3 (DiffIK).
    TaskLayout(name="b_pour_sensor", left_active=True, num_actions=15,
               match=("b_pour_sensor",)),
)


def layout_for(task_id: str) -> TaskLayout:
    """gym task id → 레이아웃. 모르는 태스크는 이름을 대고 거부한다."""
    for layout in _LAYOUTS:
        if any(key in task_id for key in layout.match):
            return layout
    known = ", ".join(k for layout in _LAYOUTS for k in layout.match)
    raise KeyError(f"pour 궤적 레이아웃을 모르는 태스크: {task_id} (아는 것: {known})")


# ---------------------------------------------------------------------------
# 캡처
# ---------------------------------------------------------------------------

class PourTrajCapture:
    """env 전 환경에서 프레임 단위 채널을 **배치로** 뽑는다.

    ★단일 환경으로 묶지 않는다. IsaacLab 은 스텝당 고정 오버헤드가 지배적이라 1 env 나
      64 env 나 스텝 시간이 거의 같다(GPU 활용률 3% 실측). 이 태스크는 학습을
      `num_envs=2048` · `replicate_physics=True` 로 돌렸으므로 다중 env 가 오히려 원래
      구성이다. 에피소드 경계는 env 마다 다른 시점에 오므로 **기록기가 env 별 버퍼로** 갈라
      담는다 — 그 대가로 수집 처리량이 env 수만큼 는다.
    """

    def __init__(self, env, layout: TaskLayout):
        self._require_contract(env, layout)
        self.env = env
        self.layout = layout
        self.num_envs = int(env.num_envs)
        self.joint_names = tuple(env.robot.data.joint_names)
        self.body_names = tuple(env.robot.data.body_names)
        self.groups: JointGroups = split_joint_groups(self.joint_names)
        self.palm_index = int(env.palm_body_index)
        if self.palm_index < 0:
            raise AttributeError(f"env 에 {RIGHT_PALM_BODY} body 가 없다")
        self.left_ee_index = self._body_index(LEFT_EE_BODY)

    @staticmethod
    def _require_contract(env, layout: TaskLayout) -> None:
        needed = _REQUIRED_ATTRS + (_REQUIRED_LEFT_ATTRS if layout.left_active else ())
        missing = [name for name in needed if not hasattr(env, name)]
        if missing:
            raise AttributeError(
                f"env 가 궤적 기록 계약을 안 지킨다 — 없는 속성: {missing}"
            )

    def _body_index(self, name: str) -> int:
        if name not in self.body_names:
            raise AttributeError(f"env 에 {name} body 가 없다 (자산이 바뀌었는가)")
        return self.body_names.index(name)

    # -- 좌표 ---------------------------------------------------------------

    @property
    def _origin(self) -> torch.Tensor:
        return self.env.scene.env_origins

    def _body_pose(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.env.robot.data
        return data.body_pos_w[:, index], data.body_quat_w[:, index]

    def _palm_ee_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """palm_ee = r_hl_palm + R·offset. 회전은 palm_link 와 공유한다."""
        pos, quat = self._body_pose(self.palm_index)
        offset = self.env._palm_ee_offset_local.reshape(1, 3).to(pos.dtype).expand_as(pos)
        return pos + quat_rotate(quat, offset), quat

    @staticmethod
    def _root_pose_w(body) -> tuple[torch.Tensor, torch.Tensor]:
        return body.data.root_pos_w, body.data.root_quat_w

    # -- 프레임 -------------------------------------------------------------

    def state_frame(self) -> dict[str, np.ndarray]:
        """스텝 **전** 측정 상태, env 별 (N, W). env.step 이 done env 를 내부에서 reset 하므로
        반드시 앞에서 찍는다."""
        data = self.env.robot.data
        palm_pos_w, palm_quat = self._palm_ee_pose_w()
        left_pos_w, left_quat = self._body_pose(self.left_ee_index)
        cup_pos_w, cup_quat = self._root_pose_w(self.env.cup)
        target_pos_w, target_quat = self._root_pose_w(self.env.left_target_cup)
        origin = self._origin

        frame = {
            "joint_pos": data.joint_pos,
            "joint_vel": data.joint_vel,
            "right_palm_pose": torch.cat([palm_pos_w - origin, palm_quat], dim=-1),
            "left_ee_pose": torch.cat([left_pos_w - origin, left_quat], dim=-1),
            "source_cup_pose": torch.cat([cup_pos_w - origin, cup_quat], dim=-1),
            "target_cup_pose": torch.cat([target_pos_w - origin, target_quat], dim=-1),
            "cup_in_hand_pose": relative_pose(palm_pos_w, palm_quat, cup_pos_w, cup_quat),
        }
        return {k: _to_np(v) for k, v in frame.items()}

    def command_frame(self) -> dict[str, np.ndarray]:
        """스텝 **후** 지령, env 별 (N, W). 이 스텝의 액션이 만든 목표값이다."""
        env = self.env
        target = env.palm_pose_targets
        frame = {
            "joint_pos_target": env.robot.data.joint_pos_target,
            # ★ sim 은 xyzw, 파일은 wxyz.
            "right_palm_cmd_pose": torch.cat(
                [target[:, :3], xyzw_to_wxyz(target[:, 3:7])], dim=-1
            ),
        }
        if self.layout.left_active:
            frame["left_ee_cmd_pose"] = torch.cat(
                [env.left_tcp_target_pos_b, env.left_tcp_fixed_quat_b], dim=-1
            )
        return {k: _to_np(v) for k, v in frame.items()}

    def outcome(self) -> tuple[np.ndarray, np.ndarray]:
        """done 시점의 env 별 (bead_frac, spill_ratio). 각각 (N,)."""
        return _to_np(self.env._last_done_bead), _to_np(self.env._last_done_spill)


def _to_np(t: torch.Tensor) -> np.ndarray:
    """텐서를 float32 numpy 사본으로. 프레임 채널은 (N, W), outcome 은 (N,)."""
    return t.detach().float().cpu().numpy().copy()

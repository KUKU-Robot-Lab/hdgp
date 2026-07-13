# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""유틸리티: 5g_grasp_right_v1"""

import torch


@torch.jit.script
def scale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """[-1, 1] 정규화 액션을 [lower, upper] 범위로 스케일."""
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def tensor_clamp(t: torch.Tensor, min_t: torch.Tensor, max_t: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.min(t, max_t), min_t)


def compute_joint7_lift_wait_target(
    actual_arm: torch.Tensor,
    *,
    joint7_delta: float,
    joint7_min: float,
    joint7_max: float,
) -> torch.Tensor:
    """Keep the grasp arm pose and move only right joint7 into lift-wait."""
    target = actual_arm.clone()
    target[:, 6] = (target[:, 6] + float(joint7_delta)).clamp(
        min=float(joint7_min),
        max=float(joint7_max),
    )
    return target


def compute_lift_readiness(
    num_contacts: torch.Tensor,
    is_grasp_phase: torch.Tensor,
    previous_hold_count: torch.Tensor,
    previous_latched: torch.Tensor,
    min_contacts: int,
    hold_steps: int,
    num_envelope_fingers: torch.Tensor | None = None,
    min_envelope_fingers: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """접촉(손끝) + 인벨롭(tip&mid) + hold 로 lift 진입 판정. '감싸 잡으면 리프트' 래치.

    step-기반(LIFT_START_STEP) 트리거를 대체. 한번 래치되면 유지된다.
    인벨롭 게이트: num_envelope_fingers(손끝&중간마디 동시 접촉 손가락 수)가
    min_envelope_fingers 이상이어야 latch 진입 → 손끝만으로 일찍 리프트하는 것을 차단.
    """
    lift_contact_now = num_contacts >= int(min_contacts)
    if num_envelope_fingers is not None and int(min_envelope_fingers) > 0:
        lift_contact_now = lift_contact_now & (
            num_envelope_fingers >= int(min_envelope_fingers)
        )
    next_hold_count = torch.where(
        lift_contact_now & is_grasp_phase,
        previous_hold_count + 1,
        torch.where(
            previous_latched,
            previous_hold_count,
            torch.zeros_like(previous_hold_count),
        ),
    )
    lift_contact_ready_now = next_hold_count >= int(hold_steps)
    next_latched = previous_latched | lift_contact_ready_now
    return next_hold_count, lift_contact_ready_now, next_latched


def to_torch(x, dtype=torch.float, device: str = "cuda:0", requires_grad: bool = False) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def euler_zyx_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    """(n,3) [ez,ey,ex] (rad) → (n,3,3) R = Rz·Ry·Rx (fabrics rotation_utils 와 동일 규약)."""
    ez, ey, ex = euler[:, 0], euler[:, 1], euler[:, 2]
    cz, sz = torch.cos(ez), torch.sin(ez)
    cy, sy = torch.cos(ey), torch.sin(ey)
    cx, sx = torch.cos(ex), torch.sin(ex)
    R = torch.empty(euler.shape[0], 3, 3, dtype=euler.dtype, device=euler.device)
    R[:, 0, 0] = cz * cy
    R[:, 0, 1] = cz * sy * sx - sz * cx
    R[:, 0, 2] = cz * sy * cx + sz * sx
    R[:, 1, 0] = sz * cy
    R[:, 1, 1] = sz * sy * sx + cz * cx
    R[:, 1, 2] = sz * sy * cx - cz * sx
    R[:, 2, 0] = -sy
    R[:, 2, 1] = cy * sx
    R[:, 2, 2] = cy * cx
    return R


def matrix_to_quat_xyzw(R: torch.Tensor) -> torch.Tensor:
    """(n,3,3) → (n,4) [qx,qy,qz,qw]. fabric set_features("quaternion") 이 받는 레이아웃."""
    m = R
    t = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    q = torch.empty(m.shape[0], 4, dtype=m.dtype, device=m.device)

    # trace 가 작을 때의 수치 불안정을 피하려 최대 성분 기준 분기 (Shepperd's method)
    c0 = t > 0.0
    c1 = (~c0) & (m[:, 0, 0] >= m[:, 1, 1]) & (m[:, 0, 0] >= m[:, 2, 2])
    c2 = (~c0) & (~c1) & (m[:, 1, 1] >= m[:, 2, 2])
    c3 = ~(c0 | c1 | c2)

    s = torch.sqrt(torch.clamp(1.0 + t, min=1e-12)) * 2.0
    q[c0, 3] = 0.25 * s[c0]
    q[c0, 0] = (m[c0, 2, 1] - m[c0, 1, 2]) / s[c0]
    q[c0, 1] = (m[c0, 0, 2] - m[c0, 2, 0]) / s[c0]
    q[c0, 2] = (m[c0, 1, 0] - m[c0, 0, 1]) / s[c0]

    s1 = torch.sqrt(torch.clamp(1.0 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], min=1e-12)) * 2.0
    q[c1, 3] = (m[c1, 2, 1] - m[c1, 1, 2]) / s1[c1]
    q[c1, 0] = 0.25 * s1[c1]
    q[c1, 1] = (m[c1, 0, 1] + m[c1, 1, 0]) / s1[c1]
    q[c1, 2] = (m[c1, 0, 2] + m[c1, 2, 0]) / s1[c1]

    s2 = torch.sqrt(torch.clamp(1.0 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2], min=1e-12)) * 2.0
    q[c2, 3] = (m[c2, 0, 2] - m[c2, 2, 0]) / s2[c2]
    q[c2, 0] = (m[c2, 0, 1] + m[c2, 1, 0]) / s2[c2]
    q[c2, 1] = 0.25 * s2[c2]
    q[c2, 2] = (m[c2, 1, 2] + m[c2, 2, 1]) / s2[c2]

    s3 = torch.sqrt(torch.clamp(1.0 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2], min=1e-12)) * 2.0
    q[c3, 3] = (m[c3, 1, 0] - m[c3, 0, 1]) / s3[c3]
    q[c3, 0] = (m[c3, 0, 2] + m[c3, 2, 0]) / s3[c3]
    q[c3, 1] = (m[c3, 1, 2] + m[c3, 2, 1]) / s3[c3]
    q[c3, 2] = 0.25 * s3[c3]

    return q / q.norm(dim=-1, keepdim=True)


def g_pose_to_fabric_quat(pose6: torch.Tensor, frame_rot: torch.Tensor) -> torch.Tensor:
    """G 규약 palm pose (n,6)[x,y,z,ez,ey,ex] → fabric quaternion pose (n,7).

    R_world_P = R_cmd(G-euler) · Cᵀ   (C = frame_rot, palm 프레임 → G 프레임)

    tesollo palm 로컬축(+X=법선, +Z=손가락)이 Allegro 규약(+X=손가락, ±Z=법선)과
    90° 어긋나 있어, G 규약에서 명령해야 ey=0 에서도 손바닥을 아래로 돌릴 수 있다.
    최종 행렬은 euler 로 표현하면 특이점(ey=±90)에 걸리므로 quaternion 으로 넘긴다.
    """
    R_cmd = euler_zyx_to_matrix(pose6[:, 3:6])                       # (n,3,3)
    R_world_p = R_cmd @ frame_rot.transpose(0, 1).unsqueeze(0)       # (n,3,3)
    quat = matrix_to_quat_xyzw(R_world_p)                            # (n,4) xyzw
    return torch.cat([pose6[:, :3], quat], dim=-1)                   # (n,7)


def compute_rotated_half_z(
    half_extent: torch.Tensor,
    rot_matrix: torch.Tensor,
) -> torch.Tensor:
    """회전 후 물체의 z half-extent = Σ_j |R[2,j]|·half[j].

    half_extent: (n,3) 로컬 half-extent
    rot_matrix:  (n,3,3) spawn 회전
    직립이면 half[2] 그대로, 누우면 커진다 → palm 이 자동으로 그만큼 올라간다.
    """
    return (rot_matrix[:, 2, :].abs() * half_extent).sum(dim=1)


def compute_palm_pose_id(
    object_idx: torch.Tensor,
    side_object_idx: torch.Tensor,
) -> torch.Tensor:
    """접근 자세 분기: side 물체면 0, 그 외 전부 1(top-down).

    object_idx:      (n,)  각 env 의 활성 물체 인덱스
    side_object_idx: (k,)  side 접근을 유지할 물체 인덱스 (cup 등)

    물체 이름으로 고정 분기한다 — 물체 높이 규칙은 ADR 회전이 커지면 납작한 원통이
    누우면서 분기에서 빠져 스스로 꺼진다(lstm_test2: topdown_frac 0.025→0.0015).
    """
    is_side = (object_idx.unsqueeze(1) == side_object_idx.unsqueeze(0)).any(dim=1)
    return torch.where(
        is_side,
        torch.zeros_like(object_idx),
        torch.ones_like(object_idx),
    )


def compute_abduction_targets(
    abduction_action: torch.Tensor,
    limits_min: torch.Tensor,
    limits_max: torch.Tensor,
) -> torch.Tensor:
    """abduction action (n,4) ∈ [-1,1] → 절대 관절 목표 (n,4).

    범위는 URDF limit 의 한쪽 절반이라 손이 안쪽으로만 모인다 — 자기충돌 검사가
    꺼져 있으므로(enabled_self_collisions=False) 이 범위가 유일한 방어선이다.
    """
    return scale(abduction_action.clamp(-1.0, 1.0), limits_min, limits_max)

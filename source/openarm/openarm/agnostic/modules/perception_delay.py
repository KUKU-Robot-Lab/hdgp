"""지연 큐 + 코히런트 자세 노이즈 (순수 torch, isaaclab 금지).

grasp_kp/grasp_fj 의 obs·action·object 지연(DESIGN.md §4·§8)을 한 곳에 둔다.
SimToolReal 방식: 길이 L 의 링버퍼에 매 스텝 push 하고, env 별로 인덱스를
**매 스텝 재추첨**해 0..L-1 step 지연된 값을 돌려준다.

쿼터니언은 wxyz. `keypoint_goal.py` 와 동시에 작성되므로 여기서 필요한
quat_mul/quat_from_angle_axis 는 비공개로 자체 구현한다(isaaclab.utils.math 규약 동일).
"""

from __future__ import annotations

import math

import torch

_EPS = 1e-8


# =============================================================================
# 비공개 쿼터니언 헬퍼 (wxyz)
# =============================================================================
def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton 곱 a⊗b, 둘 다 (N,4) wxyz."""
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """angle (N,) rad · axis (N,3) 단위벡터 → (N,4) wxyz."""
    half = 0.5 * angle
    xyz = axis * torch.sin(half).unsqueeze(-1)
    return torch.cat((torch.cos(half).unsqueeze(-1), xyz), dim=-1)


def _check_shape(name: str, t: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tuple(t.shape) != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {tuple(t.shape)}")


# =============================================================================
# DelayQueue
# =============================================================================
class DelayQueue:
    """env 별 무작위 지연(0..L-1 step) 링버퍼.

    buf (N, L, dim). L=max_delay ≥ 1 이며 L=1 은 지연 없음(항상 방금 push 한 값).
    """

    def __init__(self, num_envs: int, max_delay: int, dim: int, device) -> None:
        if int(max_delay) < 1:
            raise ValueError(f"max_delay must be ≥ 1, got {max_delay}")
        if int(dim) < 1 or int(num_envs) < 1:
            raise ValueError(f"num_envs/dim must be ≥ 1, got {num_envs}/{dim}")
        self.num_envs = int(num_envs)
        self.max_delay = int(max_delay)
        self.dim = int(dim)
        self.device = device
        self.buf = torch.zeros(self.num_envs, self.max_delay, self.dim, device=device)
        self._arange = torch.arange(self.num_envs, device=device)

    def reset(self, ids) -> None:
        self.buf[ids] = 0.0

    def push(self, values: torch.Tensor, flush: torch.Tensor) -> torch.Tensor:
        """values (N,dim) 를 넣고 env 별 무작위 지연값 (N,dim) 을 돌려준다.

        flush 가 True 인 env 는 전 슬롯을 values 로 채운다(리셋 직후 옛 값 누출 방지).
        """
        _check_shape("values", values, (self.num_envs, self.dim))
        _check_shape("flush", flush, (self.num_envs,))
        if flush.dtype != torch.bool:
            raise TypeError(f"flush must be bool, got {flush.dtype}")
        # 왜 flush 를 roll 앞에 두나: roll 뒤 [:,0]=values 까지 하면 전 슬롯이 values 로 통일된다.
        self.buf[flush] = values[flush].unsqueeze(1).to(self.buf.dtype)
        self.buf = torch.roll(self.buf, shifts=1, dims=1)
        self.buf[:, 0] = values
        idx = torch.randint(0, self.max_delay, (self.num_envs,), device=self.buf.device)
        return self.buf[self._arange, idx]


# =============================================================================
# 자세 노이즈
# =============================================================================
def perturb_quat(quat_wxyz: torch.Tensor, max_deg: float) -> torch.Tensor:
    """무작위 축 · U(−max_deg, max_deg) 각도로 회전 교란(world 프레임 pre-multiply). max_deg==0 → 그대로."""
    if quat_wxyz.ndim != 2 or quat_wxyz.shape[-1] != 4:
        raise ValueError(f"quat_wxyz: expected (N,4), got {tuple(quat_wxyz.shape)}")
    if float(max_deg) < 0.0:
        raise ValueError(f"max_deg must be ≥ 0, got {max_deg}")
    if float(max_deg) == 0.0:
        return quat_wxyz
    n = quat_wxyz.shape[0]
    dev = quat_wxyz.device
    axis = torch.randn(n, 3, device=dev)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    max_rad = math.radians(float(max_deg))
    angle = (torch.rand(n, device=dev) * 2.0 - 1.0) * max_rad
    out = _quat_mul(_quat_from_angle_axis(angle, axis), quat_wxyz)
    # 왜 재정규화: 곱셈 오차 누적으로 단위 노름이 흔들리면 하류 quat_apply 가 스케일을 먹는다.
    return out / out.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def noisy_pose(
    pos: torch.Tensor, quat: torch.Tensor, xyz_std: float, rot_deg: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """위치 가우시안(xyz_std) + 자세 perturb_quat(rot_deg). 코히런트 파생용 자세 하나를 낸다."""
    if pos.ndim != 2 or pos.shape[-1] != 3:
        raise ValueError(f"pos: expected (N,3), got {tuple(pos.shape)}")
    if pos.shape[0] != quat.shape[0]:
        raise ValueError(f"pos/quat batch mismatch: {pos.shape[0]} vs {quat.shape[0]}")
    if float(xyz_std) < 0.0:
        raise ValueError(f"xyz_std must be ≥ 0, got {xyz_std}")
    new_pos = pos + torch.randn_like(pos) * float(xyz_std) if float(xyz_std) > 0.0 else pos
    return new_pos, perturb_quat(quat, rot_deg)

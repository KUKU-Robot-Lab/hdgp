#!/usr/bin/env python3
"""여자유도 baseline별 도달 가능 tilt 측정 (순수 FK, Isaac/GPU 불필요).

배경 — 논문 Table I의 NS_naive 0.0%는 강한 주장이라, "학습이 덜 된 것 아니냐"는
반론에 대해 **운동학적 실현 불가**임을 보여야 한다. `pour_right_constants.py:113`의
설계 주석이 그 근거를 이미 서술한다:

    deep pour는 j4=1.87(팔꿈치 up)+j5=-1.22(롤)로 j6를 거의 안 쓰고 달성.
    robot_start(j4=0.60)로 nullspace를 풀면 j6가 포화되어 tilt 막힘.

이 스크립트는 그 주석을 정식 측정으로 승격한다. `pour_right_env.py:1546~1566`의
nullspace 구성을 그대로 재현한다 — baseline이 j1~j4를 앵커하고, 정책의 α가
`NULLSPACE_OFFSET_ARM` 축을 따라 이동시키며, j5~j7은 IK/Fabric이 자유롭게 쓴다.

측정 두 가지:
  1) 도달 가능한 최대 cup tilt (offset-free: 회전만으로 결정되므로 컵 기하 불필요)
  2) palm 이동 예산 δ에 대한 max tilt(δ) — 컵이 palm에 강체 고정이므로 palm 변위는
     "조준점을 얼마나 희생해야 하는가"의 대리 지표다. 정밀 조준과 deep tilt의
     상충을 정량화한다.

사용:
    python3 scripts/probes/probe_nullspace_tilt_reachability.py
    python3 scripts/probes/probe_nullspace_tilt_reachability.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_HDGP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HDGP_ROOT / "scripts" / "tools"))

from openarm_fk import ARM_JOINTS, T_PALM_CENTER, T_PALM_LINK, make_T  # noqa: E402

# --- pour_sensor 상수 (source of truth = pour_right_{constants,preset}.py) -------
ARM_START_POSE = [0.5, 0.1, 0.4, 0.60, -0.2, 0.0, 0.0]
DEMO_POUR_ARM_POSE = [0.216, 0.633, -0.371, 1.868, -1.217, 0.038, 0.604]

# 정책 α가 움직이는 self-motion 축. 학습 설정은 `nullspace_offset_mode=true_nullspace`
# (NS_demo_s42/params/env.yaml:200 확인) → palm pose를 보존하는 elbow-swivel 축을 쓴다.
# ★ j4 성분이 0 — α는 팔꿈치 높이(j4)를 바꿀 수 없다. 그래서 baseline의 j4가
#   도달 가능한 tilt를 구조적으로 결정한다(= 이 측정의 요지).
N_DEMO_NULLSPACE_OFFSET = [-0.2321, -0.4811, 0.5291, 0.0000, -0.3976, 0.1821, 0.4935]
# (대조군) 구 설정 축 = demo − robot_start. α가 baseline을 demo까지 끌 수 있어 두 baseline이
# 같은 영역을 덮는다 → 구분력이 없다. `--offset-mode demo_minus_start`로 확인 가능.
NULLSPACE_OFFSET_ARM = [d - s for d, s in zip(DEMO_POUR_ARM_POSE, ARM_START_POSE)]

# openarm_tesollo.urdf <limit lower/upper> (j1~j7)
ARM_JOINT_LIMITS = np.array(
    [
        [-1.3962629, 3.4906588],
        [-0.1745327, 3.3161253],
        [-1.5707959, 1.5707959],
        [0.0, 2.4434607],
        [-1.5707959, 1.5707959],
        [-0.7853980, 0.7853980],  # j6 — ±45°뿐. 포화 병목의 주역
        [-1.5707959, 1.5707959],
    ]
)

# deep tilt 임계: 완전 이송에 필요한 기울기 (source 컵이 뒤집히기 시작하는 각)
DEEP_TILT_THRESHOLD_DEG = 100.0

# 실제 붓기가 일어나는 tilt 구간. 이 밖(직립~얕은 기울기)은 붓기 자세가 아니라
# approach 자세라 조준점 p*를 요구하는 것 자체가 무의미하다 → 비교 대상에서 제외.
POUR_REGIME_DEG = (60.0, 130.0)

# 정밀 조준 예산: 이 이상 palm이 움직이면 주둥이가 opening 조준을 벗어난다고 본다.
# receiver opening 반경 ~4.1 cm 기준의 보수적 값들.
PALM_BUDGETS_M = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 1e9)


# ---------------------------------------------------------------------------
# FK — openarm_fk.ARM_JOINTS를 벡터화 (배치 (N,7) → palm 회전/위치)
# ---------------------------------------------------------------------------
def _joint_transforms_batch(q: np.ndarray) -> np.ndarray:
    """(N,7) 관절각 → (N,4,4) link7 변환. openarm_fk.arm_fk_raw의 벡터화 등가물."""
    n = q.shape[0]
    T = np.tile(np.eye(4), (n, 1, 1))
    for i, (xyz, rpy, axis) in enumerate(ARM_JOINTS):
        T_origin = make_T(xyz, rpy)
        axis_v = np.asarray(axis, dtype=float)
        # Rodrigues를 배치로 (rot_axis의 배치판)
        c = np.cos(q[:, i])
        s = np.sin(q[:, i])
        K = np.array(
            [
                [0.0, -axis_v[2], axis_v[1]],
                [axis_v[2], 0.0, -axis_v[0]],
                [-axis_v[1], axis_v[0], 0.0],
            ]
        )
        R = np.eye(3)[None] + s[:, None, None] * K[None] + (1.0 - c)[:, None, None] * (K @ K)[None]
        T_rot = np.tile(np.eye(4), (n, 1, 1))
        T_rot[:, :3, :3] = R
        T = T @ T_origin[None] @ T_rot
    return T


def palm_frame_batch(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(N,7) → (palm_rot (N,3,3), palm_center_pos (N,3)). 보정 offset은 상수라 생략
    (tilt는 회전만, 변위는 차분만 쓰므로 상수 offset에 불변)."""
    T_palm = _joint_transforms_batch(q) @ T_PALM_LINK[None]
    palm_center = (T_palm @ T_PALM_CENTER[None])[:, :3, 3]
    return T_palm[:, :3, :3], palm_center


def _cup_up_in_palm() -> np.ndarray:
    """start 자세에서 컵은 직립(world +Z)이고 파지 중 palm에 강체 고정 → palm 좌표계 표현."""
    R_start, _ = palm_frame_batch(np.asarray([ARM_START_POSE], dtype=float))
    return R_start[0].T @ np.array([0.0, 0.0, 1.0])


def tilt_deg(q: np.ndarray, u_palm: np.ndarray) -> np.ndarray:
    """(N,7) → world +Z 대비 컵 up축 기울기(deg). 0=직립, 180=완전 뒤집힘."""
    R, _ = palm_frame_batch(q)
    up_world = R @ u_palm  # (N,3)
    cos = np.clip(up_world[:, 2] / np.linalg.norm(up_world, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


# ---------------------------------------------------------------------------
# 스윕
# ---------------------------------------------------------------------------
def palm_jacobian(q: np.ndarray) -> np.ndarray:
    """(7,) → (6,7) palm 6D Jacobian (유한차분). 상단 3행=위치, 하단 3행=회전(world rotvec)."""
    eps = 1e-5
    R0, p0 = palm_frame_batch(q[None])
    R0, p0 = R0[0], p0[0]
    J = np.zeros((6, 7))
    for i in range(7):
        dq = q.copy()
        dq[i] += eps
        R1, p1 = palm_frame_batch(dq[None])
        J[:3, i] = (p1[0] - p0) / eps
        dR = R1[0] @ R0.T
        J[3:, i] = _rotmat_to_rotvec(dR) / eps
    return J


def _rotmat_to_rotvec(R: np.ndarray) -> np.ndarray:
    """회전행렬 → axis-angle 벡터 (작은 각에서 안정)."""
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = float(np.arccos(cos))
    if angle < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (angle / (2.0 * np.sin(angle)))


def _rotvec_to_rotmat(v: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(v))
    if angle < 1e-12:
        return np.eye(3)
    a = v / angle
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def solve_ik(
    q_seed: np.ndarray,
    p_target: np.ndarray,
    R_target: np.ndarray,
    *,
    q_null: np.ndarray | None = None,
    null_gain: float = 0.5,
    iters: int = 300,
    damping: float = 0.05,
    step: float = 0.5,
) -> tuple[np.ndarray, float, float]:
    """palm 6D task를 푸는 DLS IK + **여자유도 cspace 인력**(env와 동일 구조).

    `pour_right_env.py:1546~1566`의 Fabric default_config가 하는 일의 축약판이다:
    palm task는 1순위로 만족시키고, 남는 자유도는 `q_null`(=여자유도 baseline) 쪽으로
    끌어당긴다. null-space 항이 없으면 IK가 seed와 무관한 전역해로 수렴해
    baseline의 효과가 사라진다 — 그래서 이 항이 측정의 본질이다.

    반환: (q, 위치오차 m, 회전오차 deg).
    """
    q = q_seed.copy()
    eye7 = np.eye(7)
    for _ in range(iters):
        R, p = palm_frame_batch(q[None])
        e_pos = p_target - p[0]
        e_rot = _rotmat_to_rotvec(R_target @ R[0].T)
        e = np.concatenate([e_pos, e_rot])
        J = palm_jacobian(q)
        JJt = J @ J.T + (damping**2) * np.eye(6)
        Jpinv = J.T @ np.linalg.inv(JJt)
        dq = Jpinv @ e
        if q_null is not None:
            # palm task를 건드리지 않는 성분만 남겨 baseline으로 유도 (N = I − J⁺J)
            dq = dq + (eye7 - Jpinv @ J) @ (null_gain * (q_null - q))
        q = np.clip(q + step * dq, ARM_JOINT_LIMITS[:, 0], ARM_JOINT_LIMITS[:, 1])
    R, p = palm_frame_batch(q[None])
    err_pos = float(np.linalg.norm(p_target - p[0]))
    err_rot = float(np.degrees(np.linalg.norm(_rotmat_to_rotvec(R_target @ R[0].T))))
    return q, err_pos, err_rot


def tilt_family(u_palm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """조준점을 고정한 채 tilt만 키우는 palm 자세 족 R(θ)를 만든다.

    조준점 p* = demo pour 자세의 palm 위치(= receiver 위 실제 붓기 위치).
    R_upright = 그 위치에서 컵이 직립인 palm 자세. tilt 축 a = 직립→demo 회전축.
    R(θ) = Rot(a, θ) · R_upright  →  θ가 곧 컵 기울기.
    """
    R_demo, p_demo = palm_frame_batch(np.asarray([DEMO_POUR_ARM_POSE], dtype=float))
    R_demo, p_star = R_demo[0], p_demo[0]

    u_now = R_demo @ u_palm
    u_now /= np.linalg.norm(u_now)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(u_now, z)
    axis /= np.linalg.norm(axis)
    ang = float(np.arccos(np.clip(u_now @ z, -1.0, 1.0)))
    R_upright = _rotvec_to_rotmat(axis * ang) @ R_demo
    # 직립 → demo 로 가는 회전축(= tilt 축). 부호가 반대이므로 −axis.
    return p_star, R_upright, -axis


def _wrist_grid(n5: int, n6: int, n7: int) -> np.ndarray:
    """j5·j6·j7을 관절 한계 안에서 격자 스윕 → (M,3)."""
    axes = [
        np.linspace(ARM_JOINT_LIMITS[j, 0], ARM_JOINT_LIMITS[j, 1], n)
        for j, n in ((4, n5), (5, n6), (6, n7))
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=-1)


def sweep_baseline(
    baseline: list[float],
    u_palm: np.ndarray,
    *,
    n_alpha: int,
    grid: np.ndarray,
    offset_vec: list[float],
) -> dict:
    """baseline이 j1~j4를 앵커한 상태에서 도달 가능한 tilt를 측정.

    env 재현(`pour_right_env.py:1546~1566`): j1~4 = baseline[:4] + α·offset[:4]
    (α∈[-1,1] = 정책 self-motion action), j5~7은 IK/Fabric이 자유롭게 사용.
    전부 관절 한계로 clamp (env와 동일).
    """
    offset = np.asarray(offset_vec, dtype=float)
    base = np.asarray(baseline, dtype=float)
    alphas = np.linspace(-1.0, 1.0, n_alpha)

    # 기준(조준) 자세 = α=0, 손목은 baseline 값 → 여기서의 palm 위치가 변위 원점
    p_ref = palm_frame_batch(base[None])[1][0]

    best_tilt = -1.0
    best_q = base.copy()
    tilt_all = []
    disp_all = []

    for a in alphas:
        q = np.tile(base, (grid.shape[0], 1))
        q[:, :4] = np.clip(
            base[:4] + a * offset[:4], ARM_JOINT_LIMITS[:4, 0], ARM_JOINT_LIMITS[:4, 1]
        )
        q[:, 4:7] = grid
        t = tilt_deg(q, u_palm)
        _, p = palm_frame_batch(q)
        d = np.linalg.norm(p - p_ref[None], axis=1)
        tilt_all.append(t)
        disp_all.append(d)
        k = int(np.argmax(t))
        if t[k] > best_tilt:
            best_tilt = float(t[k])
            best_q = q[k].copy()

    tilt_all = np.concatenate(tilt_all)
    disp_all = np.concatenate(disp_all)

    # palm 이동 예산별 최대 tilt
    budget_curve = {}
    for b in PALM_BUDGETS_M:
        m = disp_all <= b
        budget_curve[b] = float(tilt_all[m].max()) if m.any() else float("nan")

    # 최대 tilt 지점에서 어떤 관절이 한계에 붙었는가 (포화 진단)
    at_limit = {
        f"j{i + 1}": bool(
            abs(best_q[i] - ARM_JOINT_LIMITS[i, 0]) < 1e-3
            or abs(best_q[i] - ARM_JOINT_LIMITS[i, 1]) < 1e-3
        )
        for i in range(7)
    }

    return {
        "max_tilt_deg": best_tilt,
        "argmax_q": best_q.tolist(),
        "at_joint_limit": at_limit,
        "budget_curve_deg": {f"{b:.3g}": v for b, v in budget_curve.items()},
        "reaches_deep_tilt": bool(best_tilt >= DEEP_TILT_THRESHOLD_DEG),
    }


def gain_sensitivity(
    p_star: np.ndarray,
    R_upright: np.ndarray,
    axis: np.ndarray,
    *,
    gains: tuple[float, ...] = (0.1, 0.2, 0.35, 0.5, 0.8),
) -> dict:
    """null-space 인력 이득에 결론이 좌우되지 않음을 확인 (임계값/튜닝 인공물 배제)."""
    out = {}
    for g in gains:
        per = {}
        for name, base in (("robot_start", ARM_START_POSE), ("demo", DEMO_POUR_ARM_POSE)):
            seed = np.asarray(base, dtype=float)
            errs = []
            for th in range(
                int(POUR_REGIME_DEG[0]), int(POUR_REGIME_DEG[1]) + 1, 10
            ):
                R_t = _rotvec_to_rotmat(axis * math.radians(th)) @ R_upright
                _, ep, _ = solve_ik(seed, p_star, R_t, q_null=seed, null_gain=g)
                errs.append(ep * 1000.0)
            per[name] = float(np.mean(errs))
        per["ratio"] = per["robot_start"] / max(per["demo"], 1e-9)
        out[f"{g:.2f}"] = per
    return out


def sweep_ik_from_baseline(
    baseline: list[float],
    u_palm: np.ndarray,
    p_star: np.ndarray,
    R_upright: np.ndarray,
    axis: np.ndarray,
    thetas_deg: np.ndarray,
    *,
    pos_tol: float,
    rot_tol_deg: float,
) -> dict:
    """조준점 p*를 고정한 채 tilt θ를 키우며, baseline을 seed로 한 IK가 어디서 깨지는지 측정.

    이것이 NS_demo vs NS_naive의 실제 구조 차이다 — 도달 가능성이 아니라,
    **여자유도가 baseline으로 끌리는 상태에서 palm task를 어디까지 만족시키는가**.
    baseline은 seed이자 null-space 인력의 목표(env의 default_config)로 함께 쓴다.
    """
    seed = np.asarray(baseline, dtype=float)
    rows = []
    max_ok = 0.0
    for th in thetas_deg:
        R_t = _rotvec_to_rotmat(axis * math.radians(float(th))) @ R_upright
        q, ep, er = solve_ik(seed, p_star, R_t, q_null=seed)
        ok = bool(ep <= pos_tol and er <= rot_tol_deg)
        sat = [
            f"j{i + 1}"
            for i in range(7)
            if abs(q[i] - ARM_JOINT_LIMITS[i, 0]) < 1e-3
            or abs(q[i] - ARM_JOINT_LIMITS[i, 1]) < 1e-3
        ]
        rows.append(
            {
                "theta_deg": float(th),
                "solved": ok,
                "err_pos_m": ep,
                "err_rot_deg": er,
                "achieved_tilt_deg": float(tilt_deg(q[None], u_palm)[0]),
                "saturated": sat,
            }
        )
        if ok:
            max_ok = float(th)
    return {"max_solved_tilt_deg": max_ok, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-alpha", type=int, default=21, help="α(정책 self-motion) 스윕 해상도")
    ap.add_argument("--n5", type=int, default=61)
    ap.add_argument("--n6", type=int, default=33)
    ap.add_argument("--n7", type=int, default=33)
    ap.add_argument("--json", type=Path, default=None, help="결과 JSON 저장 경로")
    ap.add_argument(
        "--offset-mode",
        choices=("true_nullspace", "demo_minus_start"),
        default="true_nullspace",
        help="α self-motion 축. 학습 설정은 true_nullspace (env.yaml 확인값)",
    )
    ap.add_argument("--pos-tol", type=float, default=0.005, help="IK 수렴 위치 허용오차 (m)")
    ap.add_argument("--rot-tol", type=float, default=3.0, help="IK 수렴 회전 허용오차 (deg)")
    args = ap.parse_args()
    offset_vec = (
        N_DEMO_NULLSPACE_OFFSET
        if args.offset_mode == "true_nullspace"
        else NULLSPACE_OFFSET_ARM
    )

    u_palm = _cup_up_in_palm()

    # --- 모델 검증: demo 자세에서 실제로 deep tilt가 나와야 한다 ------------------
    t_start = float(tilt_deg(np.asarray([ARM_START_POSE], dtype=float), u_palm)[0])
    t_demo = float(tilt_deg(np.asarray([DEMO_POUR_ARM_POSE], dtype=float), u_palm)[0])
    print("=" * 68)
    print(" FK 모델 검증 (컵 up축은 start 자세 직립 기준으로 캘리브)")
    print("=" * 68)
    print(f"  ARM_START_POSE       tilt = {t_start:6.1f}°  (0°이어야 정상 — 정의상)")
    print(f"  DEMO_POUR_ARM_POSE   tilt = {t_demo:6.1f}°  (deep tilt ~110° 기대)")
    if t_demo < DEEP_TILT_THRESHOLD_DEG:
        print("  ⚠️ demo 자세가 deep tilt 임계 미달 — FK 모델/상수 확인 필요")

    grid = _wrist_grid(args.n5, args.n6, args.n7)
    print(f"\n  α 축: {args.offset_mode}  (j4 성분 = {offset_vec[3]:+.4f})")
    print(f"  스윕 크기: α {args.n_alpha} × 손목격자 {grid.shape[0]} = "
          f"{args.n_alpha * grid.shape[0]:,} configs / baseline")

    results = {}
    for name, base in (("robot_start", ARM_START_POSE), ("demo", DEMO_POUR_ARM_POSE)):
        results[name] = sweep_baseline(
            base, u_palm, n_alpha=args.n_alpha, grid=grid, offset_vec=offset_vec
        )

    print("\n" + "=" * 68)
    print(" 여자유도 baseline별 도달 가능 tilt")
    print("=" * 68)
    for name, r in results.items():
        flag = "✅ deep tilt 도달" if r["reaches_deep_tilt"] else "❌ deep tilt 불가"
        sat = [j for j, v in r["at_joint_limit"].items() if v]
        print(f"\n  [{name}]  max tilt = {r['max_tilt_deg']:6.1f}°   {flag}")
        print(f"     최대 tilt 지점에서 한계 포화 관절: {sat if sat else '없음'}")
        print("     palm 이동 예산별 max tilt:")
        for b, v in r["budget_curve_deg"].items():
            label = "무제한" if float(b) > 1.0 else f"≤{float(b) * 100:.0f} cm"
            print(f"        {label:>8s} : {v:6.1f}°")

    d = results["demo"]["max_tilt_deg"] - results["robot_start"]["max_tilt_deg"]
    print(f"\n  격차(자유손목 최대 tilt): demo − robot_start = {d:+.1f}°")
    print("  ⚠️ 손목을 자유롭게 두면 두 baseline 모두 deep tilt에 도달한다 —")
    print("     즉 '도달 불가'는 성립하지 않는다. 아래 IK 측정이 실제 구조 차이다.")

    # === 측정 2 (핵심): 조준점 고정 + baseline seed IK ==========================
    p_star, R_upright, axis = tilt_family(u_palm)
    thetas = np.arange(0.0, 141.0, 5.0)
    print("\n" + "=" * 68)
    print(" ★ 조준점 고정 IK — baseline을 seed로 어느 tilt까지 해가 유지되는가")
    print("=" * 68)
    print(f"   조준점 p* = demo pour 자세의 palm 위치 {np.round(p_star, 4).tolist()}")
    print(f"   허용오차: 위치 ≤{args.pos_tol * 1000:.0f} mm, 회전 ≤{args.rot_tol:.0f}°")

    ik = {}
    for name, base in (("robot_start", ARM_START_POSE), ("demo", DEMO_POUR_ARM_POSE)):
        ik[name] = sweep_ik_from_baseline(
            base, u_palm, p_star, R_upright, axis, thetas,
            pos_tol=args.pos_tol, rot_tol_deg=args.rot_tol,
        )

    # ★ 헤드라인은 임계값 통과 여부가 아니라 **잔여 조준오차**(등급형)로 보고한다.
    #   임계값 기반 "몇 도까지 되는가"는 tol 설정에 따라 0↔130°로 튀는 인공물이 된다.
    lo, hi = POUR_REGIME_DEG
    print(f"\n   붓기 구간 θ∈[{lo:.0f}°, {hi:.0f}°]의 잔여 조준오차 (회전은 양쪽 다 ≈0°):")
    print(f"      {'θ':>6} | {'robot_start':>12} | {'demo':>10}")
    summary = {"robot_start": [], "demo": []}
    for th in np.arange(lo, hi + 1, 10.0):
        vals = {}
        for name in ("robot_start", "demo"):
            row = next(r for r in ik[name]["rows"] if abs(r["theta_deg"] - th) < 1e-6)
            vals[name] = row["err_pos_m"] * 1000.0
            summary[name].append(vals[name])
        print(f"      {th:5.0f}° | {vals['robot_start']:9.2f} mm | {vals['demo']:7.2f} mm")
    m_rs = float(np.mean(summary["robot_start"]))
    m_de = float(np.mean(summary["demo"]))
    print(f"      {'평균':>5} | {m_rs:9.2f} mm | {m_de:7.2f} mm   → "
          f"⭐ robot_start가 {m_rs / max(m_de, 1e-9):.1f}배 부정확")

    print("\n   null-space 이득 민감도 (결론이 튜닝 인공물이 아님을 확인):")
    gs = gain_sensitivity(p_star, R_upright, axis)
    print(f"      {'gain':>6} | {'robot_start':>12} | {'demo':>8} | {'비율':>6}")
    for g, v in gs.items():
        print(f"      {g:>6} | {v['robot_start']:9.2f} mm | {v['demo']:5.2f} mm | "
              f"{v['ratio']:5.2f}x")

    payload = {
        "offset_mode": args.offset_mode,
        "model_check": {"start_tilt_deg": t_start, "demo_tilt_deg": t_demo},
        "deep_tilt_threshold_deg": DEEP_TILT_THRESHOLD_DEG,
        "pour_regime_deg": list(POUR_REGIME_DEG),
        "headline_mean_aim_err_mm": {"robot_start": m_rs, "demo": m_de,
                                     "ratio": m_rs / max(m_de, 1e-9)},
        "gain_sensitivity": gs,
        "free_wrist_reachability": results,
        "aim_locked_ik": ik,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  JSON 저장: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

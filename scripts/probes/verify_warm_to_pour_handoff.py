#!/usr/bin/env python3
"""[both/pour_v1] warm 뱅크 → pour 제어 **인계 규약**을 정적으로 검사한다 (Isaac 불필요).

pour 리셋은 warm 의 palm pose 를 **자기 workspace 로 클램프**한다. 클램프가 실제로
걸리면 palm 목표가 팔의 실제 자세와 분리되고, 그 상태로 hold 를 빠져나오면 제어가
어긋난 채 학습이 시작된다(`PourWarmStateBank` 도 같은 위험을 경고한다).

기존 경고는 "뱅크 메타의 workspace 값이 pour 와 다르다"만 본다. 값이 달라도 뱅크의
**실제 pose 가 pour 박스 안**이면 클램프는 일어나지 않는다. 반대로 값이 같아도
바깥 표본이 있으면 걸린다. 그래서 여기서는 저장된 pose 를 직접 잰다.

검사 항목
  1) palm 위치가 pour palm_mins/maxs 박스 안인가 (축별 위반율·최대 초과량)
  2) palm 회전이 max_pose_angle 범위 안인가
  3) 팔 관절이 한계 안인가 (URDF 한계는 여기서 못 읽으므로 분포만 보고)
  4) 컵 z 가 pour 의 낙하 판정(obj_fallen_z)보다 위인가

사용:
    python3 scripts/probes/verify_warm_to_pour_handoff.py
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import numpy as np

_HDGP_ROOT = Path(__file__).resolve().parents[2]

# pour_v1 preset 의 palm_pose_mins/maxs 와 **같은 값**을 둔다.
# (Isaac import 없이 돌리기 위해 복제한다 — 바꿀 때 양쪽을 같이 볼 것.)
POUR_MAX_POSE_ANGLE = 45.0
POUR_POS_MIN = (-0.30, -0.55, 0.10)
POUR_POS_MAX = (0.65, 0.25, 0.68)
POUR_OBJ_FALLEN_Z = 0.20


def _rot_bounds(max_angle: float):
    d = math.pi / 180.0
    lo = ((90.0 - max_angle) * d, (0.0 - max_angle) * d, (90.0 - max_angle) * d)
    hi = ((90.0 + max_angle) * d, (0.0 + max_angle) * d, (90.0 + max_angle) * d)
    return lo, hi


def _check(path: Path, side: str, args, check_palm: bool) -> bool:
    """check_palm=False 면 palm pose 검사를 건너뛴다.

    ★좌팔(receiver)이 그렇다. pour_v1 은 **좌 뱅크의 palm pose 를 쓰지 않는다** —
      그 값은 grasp 쪽 `palm_center_pos`(손바닥 중심)라 pour 의 IK 제어점 `l_hl_palm`
      과 약 4.9cm 어긋난다. 그래서 프레임을 추측하는 대신 리셋 후 첫 스텝에 실제
      왼손 pose 를 캡처해 per-env rest 로 쓴다(pour_right_env `_left_tcp_rest_env`).
      좌팔 회전이 우팔 박스를 "위반"하는 것은 미러라서 당연하며 제어와 무관하다.
    """
    ok = True
    with h5py.File(path, "r") as f:
        g = f["warm_states"]
        prov = {k: v for k, v in f.attrs.items() if str(k).startswith("prov/")}
        palm7 = np.asarray(g["palm_pose_quat_xyzw"])      # (N,7) [x,y,z,qx,qy,qz,qw]
        euler6 = np.asarray(g["palm_pose_euler_zyx"])     # (N,6) [x,y,z,ez,ey,ex]
        arm = np.asarray(g["arm_joint_pos"])              # (N,7)
        cup = np.asarray(g["cup_pos_local"])              # (N,3)
    n = len(palm7)
    print(f"\n=== {side}  n={n}  {path.name} ===")
    ck = prov.get("prov/checkpoint")
    print(f"  ckpt: {Path(str(ck)).name if ck else '⚠ provenance 없음 (출처 불명 뱅크)'}")

    if not check_palm:
        print("  ·  palm pose 검사 생략 — pour 는 좌 뱅크의 palm pose 를 쓰지 않는다")
        print(f"  ·  arm q 범위: " + " ".join(
            f"j{k+1} [{arm[:, k].min():+.2f},{arm[:, k].max():+.2f}]" for k in range(arm.shape[1])))
        n_low = int((cup[:, 2] < POUR_OBJ_FALLEN_Z).sum())
        flag = "OK " if n_low == 0 else "!! "
        print(f"  {flag}컵 z: [{cup[:, 2].min():.3f}, {cup[:, 2].max():.3f}] "
              f"vs obj_fallen_z {POUR_OBJ_FALLEN_Z}  즉시낙하 {n_low}/{n}")
        return n_low == 0

    # 1) palm 위치 박스
    pos = palm7[:, :3]
    for i, ax in enumerate("xyz"):
        lo, hi = POUR_POS_MIN[i], POUR_POS_MAX[i]
        below, above = pos[:, i] < lo, pos[:, i] > hi
        n_bad = int(below.sum() + above.sum())
        worst = 0.0
        if n_bad:
            worst = max(
                float((lo - pos[below, i]).max()) if below.any() else 0.0,
                float((pos[above, i] - hi).max()) if above.any() else 0.0,
            )
            ok = False
        flag = "OK " if n_bad == 0 else "!! "
        print(f"  {flag}palm {ax}: [{pos[:, i].min():+.3f}, {pos[:, i].max():+.3f}] "
              f"vs pour [{lo:+.2f}, {hi:+.2f}]  클램프 {n_bad}/{n}"
              + (f"  최대초과 {worst*1000:.1f}mm" if n_bad else ""))

    # 2) palm 회전
    lo_r, hi_r = _rot_bounds(POUR_MAX_POSE_ANGLE)
    for i, name in enumerate(("ez", "ey", "ex")):
        v = euler6[:, 3 + i]
        n_bad = int(((v < lo_r[i]) | (v > hi_r[i])).sum())
        if n_bad:
            ok = False
        flag = "OK " if n_bad == 0 else "!! "
        print(f"  {flag}palm {name}: [{math.degrees(v.min()):+.1f}°, {math.degrees(v.max()):+.1f}°] "
              f"vs [{math.degrees(lo_r[i]):+.1f}°, {math.degrees(hi_r[i]):+.1f}°]  위반 {n_bad}/{n}")

    # 3) 팔 관절 분포 (한계는 URDF 에 있어 여기선 분포만)
    print(f"  ·  arm q 범위: " + " ".join(
        f"j{k+1} [{arm[:, k].min():+.2f},{arm[:, k].max():+.2f}]" for k in range(arm.shape[1])))

    # 4) 컵 낙하 판정
    n_low = int((cup[:, 2] < POUR_OBJ_FALLEN_Z).sum())
    if n_low:
        ok = False
    flag = "OK " if n_low == 0 else "!! "
    print(f"  {flag}컵 z: [{cup[:, 2].min():.3f}, {cup[:, 2].max():.3f}] "
          f"vs obj_fallen_z {POUR_OBJ_FALLEN_Z}  즉시낙하 {n_low}/{n}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--right", default=str(_HDGP_ROOT / "data/grasp_warm_tesollo_right.hdf5"))
    ap.add_argument("--left", default=str(_HDGP_ROOT / "data/grasp_warm_tesollo_left.hdf5"))
    args = ap.parse_args()

    ok = True
    for path, side, check_palm in (
        (Path(args.right), "우팔 source", True),
        (Path(args.left), "좌팔 receiver", False),
    ):
        if not path.is_file():
            print(f"!! 뱅크 없음: {path}")
            ok = False
            continue
        ok &= _check(path, side, args, check_palm)

    print("\n" + ("=" * 60))
    if ok:
        print("PASS — warm pose 가 pour 제어 박스 안에 있다 (리셋 클램프 없음).")
    else:
        print("FAIL — 클램프/위반이 있다. 그대로 학습하면 palm 목표가 실제 팔 자세와")
        print("       어긋난 채 시작한다. pour workspace 를 넓히거나 수집 조건을 볼 것.")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

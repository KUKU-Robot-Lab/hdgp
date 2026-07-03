"""warmstart hdf5 의 palm_pose 를 palm_sensor 프레임으로 재계산(FK 마이그레이션).

배경: grasp_warm_*.hdf5 의 palm_pose_euler_zyx / palm_pose_quat_xyzw 는 구 grasp
palm 규약(palm_link 기반, ex 중심 ~90°, +3.4cm offset)으로 수집됐다. 이후 grasp/pour
fabric 이 r_hl_palm_sensor 를 직접 제어(ex 중심 180°, offset 0)하도록 정합됐으므로,
저장된 palm_pose 를 그대로 warmstart 제어 base 로 쓰면 ~90°/3.4cm 어긋난다.

이 스크립트는 **유효한 관절 데이터(arm_joint_pos/hand_joint_pos)만** 사용해 현재 fabric
의 get_palm_pose(= r_hl_palm_sensor FK)로 palm_pose 를 재계산하여 두 데이터셋만 덮어쓴다.
나머지(관절·컵·비드·mimic)는 보존한다. grasp 재학습 불필요.

서버(warp) 실행:
  python scripts/tools/migrate_warm_palm_to_sensor.py --hdf5 data/grasp_warm_rh56f1.hdf5
  (--dry-run 으로 재계산 통계만 확인 가능)
"""

import argparse
import os
import shutil
import sys

import h5py
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, *[os.path.dirname(_HERE)]):
    _v = os.path.join(_p, "..", "source", "FABRICS", "src")
    if os.path.isdir(_v):
        sys.path.insert(0, os.path.abspath(_v))
        break

from fabrics_sim.fabrics.openarm_rh56f1_pose_fabric import (
    OpenArmRh56f1PoseFabric,
    NUM_ARM_DOF,
    NUM_HAND_DOF,
    NUM_ROBOT_DOF,
    NUM_DOF,
)
from fabrics_sim.utils.utils import initialize_warp

_GROUP = "warm_states"


def main():
    ap = argparse.ArgumentParser(description="warm hdf5 palm_pose → palm_sensor FK 재계산.")
    ap.add_argument("--hdf5", required=True)
    ap.add_argument("--device_int", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="덮어쓰지 않고 통계만 출력")
    args = ap.parse_args()

    device = f"cuda:{args.device_int}"
    initialize_warp(str(args.device_int))

    with h5py.File(args.hdf5, "r") as f:
        g = f[_GROUP]
        arm = torch.tensor(np.asarray(g["arm_joint_pos"]), dtype=torch.float32, device=device)
        hand = torch.tensor(np.asarray(g["hand_joint_pos"]), dtype=torch.float32, device=device)
        old_euler = np.asarray(g["palm_pose_euler_zyx"])
    n = arm.shape[0]
    assert arm.shape[1] == NUM_ARM_DOF and hand.shape[1] == NUM_HAND_DOF, \
        f"관절 차원 불일치 arm{arm.shape} hand{hand.shape}"
    print(f"[load] N={n} states from {args.hdf5}")

    fabric = OpenArmRh56f1PoseFabric(n, device, timestep=1.0 / 60.0, graph_capturable=False)
    assert fabric.num_joints == NUM_DOF == 26, fabric.num_joints

    # cspace(26): 우측[0:13]=arm+hand, 좌측[13:26]=default 중립(우측 palm FK 에 무영향).
    cspace = fabric.default_config.clone()
    cspace[:, :NUM_ARM_DOF] = arm
    cspace[:, NUM_ARM_DOF:NUM_ROBOT_DOF] = hand

    with torch.no_grad():
        new_euler = fabric.get_palm_pose(cspace, "euler_zyx").detach().cpu().numpy()   # (N,6)
        new_quat = fabric.get_palm_pose(cspace, "quaternion").detach().cpu().numpy()   # (N,7) pos+xyzw

    # 통계
    def _deg(a):
        return np.round(np.degrees(a), 1)
    print("[euler ex(=idx5) rad] old mean=%.3f  new mean=%.3f  Δmean=%.3f (%.1f°)" % (
        old_euler[:, 5].mean(), new_euler[:, 5].mean(),
        new_euler[:, 5].mean() - old_euler[:, 5].mean(),
        np.degrees(new_euler[:, 5].mean() - old_euler[:, 5].mean()),
    ))
    print("[pos] old xyz range:", np.round(old_euler[:, :3].min(0), 3), "→", np.round(old_euler[:, :3].max(0), 3))
    print("[pos] new xyz range:", np.round(new_euler[:, :3].min(0), 3), "→", np.round(new_euler[:, :3].max(0), 3))
    print("[pos] |Δ| mean=%.4fm max=%.4fm" % (
        np.linalg.norm(new_euler[:, :3] - old_euler[:, :3], axis=1).mean(),
        np.linalg.norm(new_euler[:, :3] - old_euler[:, :3], axis=1).max(),
    ))
    print("[sanity] new euler[:3] sample:\n", np.round(new_euler[:3], 4))

    if args.dry_run:
        print("[dry-run] 파일 미변경.")
        return

    bak = args.hdf5 + ".pre_palmsensor_bak"
    if not os.path.exists(bak):
        shutil.copy2(args.hdf5, bak)
        print(f"[backup] {bak}")

    with h5py.File(args.hdf5, "r+") as f:
        g = f[_GROUP]
        g["palm_pose_euler_zyx"][...] = new_euler.astype(np.float32)
        g["palm_pose_quat_xyzw"][...] = new_quat.astype(np.float32)
        f.attrs["meta/palm_frame"] = "r_hl_palm_sensor (FK migrated)"
    print(f"[OK] palm_pose_euler_zyx / palm_pose_quat_xyzw → palm_sensor 프레임으로 재계산 완료.")


if __name__ == "__main__":
    main()

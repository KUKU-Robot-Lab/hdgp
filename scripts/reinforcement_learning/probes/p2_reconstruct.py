"""P2 Level-1 재구성 검증 (순수 numpy, GPU 불필요).

p2_dump_ref.py 가 덤프한 raw 상태로 actor obs 55D 를 **독립 재구현**해 덤프된 env obs 와 비교.
geometry(_compute_intermediate_values 서브셋) + obs 조립(_get_observations actor 경로)을 numpy 로
포팅한 것이 정확한지 element-wise 검증. 통과하면 P0 §1/§2 스펙이 라이브 루프에 이식 가능.

실행: python3 hdgp/scripts/reinforcement_learning/probes/p2_reconstruct.py --ref hdgp/docs/eval/p2_ref.npz
"""

import argparse
import os

import numpy as np

NUM_FINGERTIPS = 5
NUM_PALM_ACTION = 6


def quat_apply(q_wxyz, v):
    """wxyz 쿼터니언으로 벡터 회전 (batch). q:(N,4), v:(N,3) 또는 (3,)."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    v = np.broadcast_to(np.asarray(v, dtype=np.float64), q[:, 1:].shape)
    w = q[:, 0:1]
    u = q[:, 1:4]
    uv = np.cross(u, v)
    uuv = np.cross(u, uv)
    return v + 2.0 * (w * uv + uuv)


def finger_grasp_progress(finger_pos, open_pose, grasp_pose):
    """env._finger_grasp_progress numpy 포팅. finger_pos:(N,20)."""
    delta = grasp_pose - open_pose            # (20,)
    valid = np.abs(delta) > 1e-6
    denom = np.where(valid, delta, np.ones_like(delta))
    progress_20 = np.clip((finger_pos - open_pose[None, :]) / denom[None, :], 0.0, 1.0)
    progress_20 = progress_20 * valid[None, :].astype(progress_20.dtype)
    valid_counts = np.clip(valid.reshape(NUM_FINGERTIPS, 4).sum(-1), 1, None).astype(np.float64)
    return progress_20.reshape(-1, NUM_FINGERTIPS, 4).sum(-1) / valid_counts[None, :]


def compute_geometry(d):
    """_compute_intermediate_values 의 obs 관련 geometry 서브셋 numpy 재현."""
    cup_pos = d["cup_pos_w"]; cup_quat = d["cup_quat_w"]
    lcup_pos = d["left_cup_pos_w"]; lcup_quat = d["left_cup_quat_w"]
    n = cup_pos.shape[0]

    rim_center = cup_pos + quat_apply(cup_quat, d["source_cup_pour_point_pos_b"])
    target_opening = lcup_pos + quat_apply(lcup_quat, d["target_cup_opening_pos_b"])
    source_pour_axis = quat_apply(cup_quat, d["source_cup_pour_axis_b"])
    source_up_axis = quat_apply(cup_quat, d["source_cup_up_axis_b"])
    target_up_axis = quat_apply(lcup_quat, d["target_cup_up_axis_b"])
    cup_up = source_up_axis

    world_down = np.zeros((n, 3)); world_down[:, 2] = -1.0
    dot = np.sum(world_down * cup_up, axis=-1, keepdims=True)
    grav_perp = world_down - dot * cup_up
    grav_perp_hat = grav_perp / np.clip(np.linalg.norm(grav_perp, axis=-1, keepdims=True), 1e-6, None)
    perp_xy_mag = np.linalg.norm(grav_perp_hat[:, :2], axis=-1, keepdims=True)

    pour_dir_xy = target_opening[:, :2] - rim_center[:, :2]
    static_dir = pour_dir_xy / np.clip(np.linalg.norm(pour_dir_xy, axis=-1, keepdims=True), 1e-6, None)
    dynamic_dir = grav_perp_hat[:, :2] / np.clip(perp_xy_mag, 1e-6, None)
    su_dot = np.clip(cup_up[:, 2], -1.0, 1.0)
    tilt_amt = np.clip((1.0 - su_dot) / 2.0, 0.0, 1.0)
    lo = float(d["pour_point_dyn_lo"]); hi = float(d["pour_point_dyn_hi"])
    dyn_t = np.clip((tilt_amt - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    dyn_w = (dyn_t * dyn_t * (3.0 - 2.0 * dyn_t))[:, None]
    blended = (1.0 - dyn_w) * static_dir + dyn_w * dynamic_dir
    pour_dir_hat = blended / np.clip(np.linalg.norm(blended, axis=-1, keepdims=True), 1e-6, None)
    R = float(d["source_outer_radius"])
    pp_xy = rim_center[:, :2] + R * perp_xy_mag * pour_dir_hat
    pp_z = (rim_center[:, 2] + R * grav_perp_hat[:, 2])[:, None]
    source_pour_point = np.concatenate([pp_xy, pp_z], axis=-1)

    return dict(source_pour_point_w=source_pour_point, target_opening_w=target_opening,
                source_pour_axis_w=source_pour_axis, source_up_axis_w=source_up_axis,
                target_up_axis_w=target_up_axis)


def assemble_actor_obs(d, geo):
    fgp = finger_grasp_progress(d["finger_joint_pos"], d["hand_open_pose"], d["hand_grasp_pose"])
    pour_point_to_opening = geo["target_opening_w"] - geo["source_pour_point_w"]
    n = d["arm_joint_pos"].shape[0]
    last_palm = np.zeros((n, NUM_PALM_ACTION))   # reset 시 actions=0
    return np.concatenate([
        d["arm_joint_pos"],          # 7
        d["arm_joint_vel"],          # 7
        fgp,                         # 5
        d["left_arm_joint_pos"],     # 9
        d["left_arm_joint_vel"],     # 9
        pour_point_to_opening,       # 3
        geo["source_pour_axis_w"],   # 3
        geo["source_up_axis_w"],     # 3
        geo["target_up_axis_w"],     # 3
        last_palm,                   # 6
    ], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=str,
                    default=os.path.join(os.path.dirname(__file__), "..", "..", "docs", "eval", "p2_ref.npz"))
    a = ap.parse_args()
    d = {k: v for k, v in np.load(os.path.abspath(a.ref)).items()}
    ref = d["actor_obs"]
    n = ref.shape[0]

    # 1) geometry 교차검증
    geo = compute_geometry(d)
    print("[P2-rec] --- geometry _w 교차검증 (재구현 vs 덤프) ---")
    for k in ("source_pour_point_w", "target_opening_w", "source_pour_axis_w",
              "source_up_axis_w", "target_up_axis_w"):
        err = np.abs(geo[k] - d[k]).max()
        print(f"[P2-rec]   {k:22s} max|err|={err:.3e}")

    # 2) actor obs 55D 조립 검증
    rec = assemble_actor_obs(d, geo)
    if rec.shape != ref.shape:
        print(f"[P2-rec] SHAPE MISMATCH rec{rec.shape} vs ref{ref.shape}"); return
    err = np.abs(rec - ref)
    per_env_max = err.max(axis=1)
    worst_idx = int(np.argmax(per_env_max))
    print(f"[P2-rec] --- actor obs 55D 검증 ({n} envs) ---")
    print(f"[P2-rec]   전체 max|err|={err.max():.3e} | mean|err|={err.mean():.3e}")
    # 채널별 최대 오차 상위
    ch_err = err.max(axis=0)
    seg = [("arm_pos",0,7),("arm_vel",7,14),("fgp",14,19),("larm_pos",19,28),
           ("larm_vel",28,37),("pp2open",37,40),("src_pour_ax",40,43),
           ("src_up_ax",43,46),("tgt_up_ax",46,49),("last_palm",49,55)]
    print("[P2-rec]   채널별 max|err|:")
    for name, s, e in seg:
        print(f"[P2-rec]     {name:14s} [{s:2d}:{e:2d}] {ch_err[s:e].max():.3e}")
    tol = 1e-4
    ok = err.max() < tol
    print(f"[P2-rec] ===== 판정: {'PASS' if ok else 'FAIL'} (tol={tol:.0e}, worst env {worst_idx} "
          f"max|err|={per_env_max[worst_idx]:.3e}) =====")
    if not ok:
        wi = worst_idx
        bad = np.argsort(err[wi])[::-1][:6]
        print(f"[P2-rec]   worst env {wi} 최대오차 채널 idx={bad.tolist()}")
        for b in bad:
            print(f"[P2-rec]     idx {b:2d}: rec={rec[wi,b]:+.5f} ref={ref[wi,b]:+.5f} err={err[wi,b]:.3e}")


if __name__ == "__main__":
    main()

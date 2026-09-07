"""RH56F1 손바닥 법선·케이지 기하 **실측** — 유도가 아니라 측정한다.

## 왜

`palm_frame_remap`(태스크 표준축: 열0=손바닥 법선 · 열1=가로 · 열2=손가락 방향)을
09.02 에 URDF rpy 계산으로 **유도**해서 넣었다. 그 뒤 학습 영상에서 엄지가 컵 안으로
들어가는 이상 접근이 관찰됐다(사용자). 유도가 틀렸으면 접근 기하 전체가 틀어진다.

## 정의(측정 가능한 것만)

- **손바닥 법선** = 손을 닫을 때 **4지 손끝이 움직이는 방향**. 손가락은 손바닥 앞면
  쪽으로 말리므로 그 방향이 곧 법선이다. (tesollo 로 검산하면 +x 가 나와야 한다 —
  그 프로필은 법선이 국소 +x 라고 선언돼 있다.)
- **손가락 방향** = 손바닥 원점 → 4지 손끝 평균.
- **케이지** = 엄지 끝과 4지 끝 평균의 중점(태스크가 쓰는 정의 그대로).

전부 `palm_body` 의 **원 국소 프레임**(리맵 적용 전)으로 환산해 출력한다.

## 사용

    cd hdgp && ~/rl_ws/IsaacLab/isaaclab.sh -p scripts/probes/probe_ua_palm_frame.py \
        --task open-rh_r_grasp_ua-play-lstm
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="open-rh_r_grasp_ua-play-lstm")
parser.add_argument("--settle", type=int, default=250)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
_app = AppLauncher(args_cli).app

import dataclasses  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.agnostic.tasks.grasp_ua.config  # noqa: E402,F401
from isaaclab.utils.math import matrix_from_quat  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from openarm.agnostic.tasks.grasp_ua import robot_profiles as _rp  # noqa: E402


def _fmt(v):
    return "[" + ", ".join(f"{float(x):+.3f}" for x in v) + "]"


def main() -> int:
    cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=2)
    cfg.scene.num_envs = 2
    cfg.enable_events = False
    # ★물체 없이 손만 본다 — 접촉이 손가락 궤적을 왜곡하면 법선 측정이 오염된다.
    cfg.object_bank = "single_cup"
    name = cfg.profile_name
    prof = _rp.PROFILES[name]
    env = gym.make(args_cli.task, cfg=cfg)
    u = env.unwrapped
    env.reset()

    def snap():
        """palm 국소 프레임에서의 손끝 좌표 (thumb, 4지 평균, 케이지)."""
        R = matrix_from_quat(u.robot.data.body_quat_w[:, u.palm_idx])[0]   # 원 국소축
        p = u.robot.data.body_pos_w[0, u.palm_idx]
        tips = u.robot.data.body_pos_w[0, u._tip_ids_t]
        loc = torch.einsum("ji,nj->ni", R, tips - p.unsqueeze(0))          # Rᵀ(x−p)
        a = int(u._group_a0)
        others = [i for i in range(loc.shape[0]) if i != a]
        return loc, loc[a], loc[others].mean(dim=0)

    loc0, th0, fg0 = snap()
    # 손만 완전 폐쇄 (팔은 그대로)
    u._syn_close[:] = 1.0
    u._syn_target[:] = torch.lerp(
        u._syn_open.unsqueeze(0), u._syn_grip.unsqueeze(0), u._syn_close
    ).clamp(u._syn_lo.unsqueeze(0), u._syn_hi.unsqueeze(0))
    for _ in range(args_cli.settle):
        u.robot.set_joint_position_target(u._syn_target, joint_ids=u._syn_ids)
        u._apply_mimic_targets()
        u.scene.write_data_to_sim()
        u.sim.step(render=False)
        u.scene.update(u.physics_dt)
    loc1, th1, fg1 = snap()

    names = [n.split("_hl_")[-1] for n in prof.fingertip_bodies]
    print("\n" + "=" * 76, flush=True)
    print(f"[palm-frame] 프로필 {name} · palm_body = {prof.palm_body}", flush=True)
    print(f"[palm-frame] 선언된 palm_frame_remap = {prof.palm_frame_remap}", flush=True)
    print("\n[palm-frame] 손끝 위치(palm **원** 국소축) open → close", flush=True)
    for i, n in enumerate(names):
        d = loc1[i] - loc0[i]
        print(f"[palm-frame]   {n:12s} {_fmt(loc0[i])} → {_fmt(loc1[i])}  Δ={_fmt(d)} "
              f"|Δ|={float(d.norm()) * 1000:5.1f}mm", flush=True)

    # ★★폐쇄 이동에는 "손가락이 안쪽으로 말리며 palm 에 가까워지는" 성분이 크게
    #   섞인다(측정: 그 성분이 −0.87). 법선은 **손가락 방향에 수직인 성분**이다 —
    #   빼지 않으면 법선이 손가락 축 쪽으로 끌려가 엉뚱한 값이 나온다.
    f_hat = fg0 / fg0.norm().clamp(min=1e-9)
    d4 = (fg1 - fg0)
    d_perp = d4 - (d4 @ f_hat) * f_hat
    n_hat = d_perp / d_perp.norm().clamp(min=1e-9)
    print(f"\n[palm-frame] 폐쇄 이동 원벡터 = {_fmt(d4 / d4.norm())} "
          f"· 손가락축 성분 {float(d4 @ f_hat / d4.norm()):+.3f} (빼고 계산)", flush=True)
    print(f"\n[palm-frame] ★손바닥 법선(4지 폐쇄 이동방향) = {_fmt(n_hat)}", flush=True)
    print(f"[palm-frame]  손가락 방향(palm→4지끝)        = {_fmt(f_hat)}", flush=True)
    print(f"[palm-frame]  엄지 끝(open)                  = {_fmt(th0)}", flush=True)
    print(f"[palm-frame]  4지 끝 평균(open)              = {_fmt(fg0)}", flush=True)
    cage0 = 0.5 * (th0 + fg0)
    r0 = 0.5 * float((th0 - fg0).norm())
    print(f"[palm-frame]  케이지 중심(open)              = {_fmt(cage0)} "
          f"· 반경 {r0 * 1000:.0f}mm", flush=True)
    print(f"[palm-frame]  대향축(엄지끝→4지끝)           = "
          f"{_fmt((fg0 - th0) / (fg0 - th0).norm())}", flush=True)

    # 태스크 표준(열0=법선, 열2=손가락) 을 만족하는 리맵 제안
    z = f_hat
    y = torch.linalg.cross(z, n_hat)
    M = torch.stack([n_hat, y, z], dim=1)          # 열 = 표준축(국소 좌표)
    print("\n[palm-frame] ★측정에서 나온 palm_frame_remap (행 우선):", flush=True)
    for r in range(3):
        print("[palm-frame]      (" + ", ".join(f"{float(M[r, c]):+.4f}" for c in range(3)) + "),",
              flush=True)
    if prof.palm_frame_remap:
        D = torch.tensor(prof.palm_frame_remap, device=M.device, dtype=M.dtype)
        ang = torch.rad2deg(torch.arccos(
            ((torch.trace(D.T @ M) - 1.0) / 2.0).clamp(-1.0, 1.0)))
        print(f"[palm-frame] 선언값과의 회전 차이 = {float(ang):.1f}°", flush=True)
        for c, nm in enumerate(("법선", "가로", "손가락")):
            dot = float((D[:, c] * M[:, c]).sum())
            print(f"[palm-frame]   {nm:4s} 축 일치도 {dot:+.3f} "
                  f"({'OK' if dot > 0.95 else '★불일치'})", flush=True)
    # ★태스크 표준축(열0 법선 · 열1 가로 · 열2 손가락) 성분으로 다시 본다 —
    #   보상이 쓰는 좌표계가 이것이라 여기서 봐야 접근 설계가 보인다.
    def _task(v):
        return torch.stack([v @ M[:, 0], v @ M[:, 1], v @ M[:, 2]])
    print("\n[palm-frame] ★태스크 표준축 성분 (법선, 가로, 손가락)", flush=True)
    print(f"[palm-frame]   엄지 끝      = {_fmt(_task(th0))}", flush=True)
    print(f"[palm-frame]   4지 끝 평균  = {_fmt(_task(fg0))}", flush=True)
    print(f"[palm-frame]   케이지 중심  = {_fmt(_task(cage0))}", flush=True)
    _op = (fg0 - th0) / (fg0 - th0).norm()
    print(f"[palm-frame]   대향축       = {_fmt(_task(_op))}", flush=True)
    print("[palm-frame]   → 보상은 `palm_normal_dist`(법선 성분)를 0 으로 몰지만,"
          " 컵이 들어가야 할 케이지는 법선으로 "
          f"{float(_task(cage0)[0]) * 1000:+.0f}mm 떨어져 있다.", flush=True)
    print("=" * 76, flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    import traceback
    try:
        _rc = main()
    except BaseException:
        traceback.print_exc()
        _rc = 3
    _app.close()
    raise SystemExit(_rc)

#!/usr/bin/env python3
"""손목 진동·충격 실측 — E2(r2s 정합 게인)의 "테이블을 쓸고 팔이 튄다" 원인 규명.

배경 (09.01)
------------
E2(r2s 정합 게인)와 E1(KUKA 기본 게인)의 학습 지표가 이렇게 갈렸다(최근 25%):

    contact/force_max_prelatch   E2 77.5 N   vs  E1 10.8 N   (7.2배)
    done/arm_qd_max              E2 4.25     vs  E1 1.74     (2.4배)
    task/cup_disp                E2 0.091 m  vs  E1 0.014 m  (6.5배)
    fabric/palm_err_mean         E2 0.047    vs  E1 0.078    (E2 가 **더 정확**)

`palm_err_mean` 이 E2 가 더 작으므로 "팔이 지령을 못 따라간다"는 설명은 기각된다.
남은 가설은 **감쇠**다 — r2s 게인은 손목 kd 를 26~62배 낮춘다(j6 15→0.580,
j7 15→0.242). R2S 문서 실측 ζ 도 j5/j6/j7 = 0.071/0.012/0.069 로 부족감쇠이고
"j6 은 2.1 Hz 에서 5.4배 공진"이라 기록돼 있다.

    가설: 무감쇠 손목이 울린다 → 테이블을 세게 친다 → 컵을 밀어낸다
          → 밀린 자리에서 잡으니 목표(원래 정착 위치 기준)에 수평으로 못 간다
    (학습 지표 뒷받침: `goal_dxy` 와 `cup_disp` 가 정확히 같은 값 0.0906)

무엇을 재는가
-------------
정책을 쓰지 않는다. **스크립트 지령**으로 palm 을 내려 테이블/컵에 접촉시키고
스텝마다 기록한다 — 게인만 바꾼 통제 실험이라 인과가 깨끗하다.

    q(팔 7) · fabric_q(팔 7) · qd(팔 7) · 접촉력 최대 · palm z · 컵 xy

판정
----
  · **진동이면**: 접촉 뒤 관절 오차/속도가 **부호를 바꾸며 반복**하고 FFT 에
    2 Hz 대 피크가 선다. 충격력도 반복해서 뜬다.
  · **단발 변형이면**: 접촉 순간 한 번 튀고 그대로 눌린 채 유지된다(부호 불변).
  이 둘은 처방이 다르다 — 진동이면 감쇠(kd)를, 변형이면 강성(kp)을 봐야 한다.

사용
----
    python scripts/probes/probe_s2r_wrist_ring.py --out /tmp/ring_kuka.npz
    HDGP_S2R_REAL_GAINS=1 python scripts/probes/probe_s2r_wrist_ring.py --out /tmp/ring_r2s.npz

★게인 분기는 `robot_profiles.py` 가 **import 시점**에 환경변수로 정하므로
  한 프로세스에서 두 게인을 못 돈다. 두 번 실행해 npz 를 비교한다.
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--settle", type=int, default=120, help="a=0 정착 스텝")
parser.add_argument("--press", type=int, default=240, help="아래로 누르는 스텝")
parser.add_argument("--release", type=int, default=120, help="지령 복귀 후 관찰 스텝")
parser.add_argument("--envs", type=int, default=4)
parser.add_argument("--out", default="/tmp/ring.npz")
parser.add_argument("--friction", action="store_true",
                    help="★r2s 가 0 으로 지운 **실측 관절마찰**을 되돌린다. "
                         "R2S 문서는 'sim friction 은 안 먹는다'고 기록했지만 그 검증은 "
                         "접촉 없는 여진 재생이었다 — 마찰은 속도 부호에만 반응하므로 "
                         "접촉 운동에서는 기여가 다를 수 있다. 이 플래그가 그걸 시험한다.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from openarm.agnostic.tasks.grasp_s2r.grasp_s2r_env import GraspS2REnv  # noqa: E402
from openarm.agnostic.tasks.grasp_s2r.grasp_s2r_env_cfg import (  # noqa: E402
    GraspS2RTesolloRightEnvCfg,
)

_REAL = os.environ.get("HDGP_S2R_REAL_GAINS") == "1"
_TAG = "r2s" if _REAL else "kuka"


def main() -> None:
    cfg = GraspS2RTesolloRightEnvCfg()
    cfg.scene.num_envs = int(args.envs)
    cfg.object_bank = "single_cup"
    cfg.enable_events = False           # 마찰 DR 을 빼 게인만 남긴다
    cfg.enable_adr = False
    cfg.enable_self_collisions = False
    cfg.episode_length_s = 10_000.0     # 프로브 중 리셋 방지(관례)
    # ★마찰 복원 — robot_cfg 는 finalize 가 만들므로 그 **뒤에** 덮는다.
    #   값은 r2s 가 0 으로 지우기 전의 실측 동정치(커밋 bfe82c5 의 삭제분).
    if args.friction:
        cfg.finalize_after_overrides()
        _FR = {"1": 0.643, "2": 0.631, "3": 1.343, "4": 0.178,
               "5": 0.701, "6": 1.387, "7": 0.698}
        _hit = []
        for name, a in cfg.robot_cfg.actuators.items():
            if "arm" not in name:
                continue
            _j = name.rsplit("_j", 1)[-1] if "_j" in name else None
            if _j in _FR:                       # right_arm_j5 처럼 관절별로 나뉜 경우
                a.friction = _FR[_j]; _hit.append(f"{name}={_FR[_j]}")
            else:                               # right_arm_proximal 처럼 묶인 경우
                a.friction = sum(_FR[k] for k in "1234") / 4.0
                _hit.append(f"{name}={a.friction:.3f}(평균)")
        print(f"[RING] 마찰 복원: {' · '.join(_hit)}", flush=True)
    env = GraspS2REnv(cfg, render_mode=None)
    u = env.unwrapped
    dev = u.device
    n = u.num_envs
    arm = u._arm_ids_t

    act = torch.zeros(n, u.cfg.action_space, device=dev)
    rec = {k: [] for k in ("q", "fq", "qd", "force", "palm_z", "cup_xy", "phase")}

    def step(a, phase):
        env.step(a)
        rec["q"].append(u.robot.data.joint_pos[:, arm].cpu().numpy().copy())
        rec["fq"].append(u.fabric_q[:, :7].detach().cpu().numpy().copy())
        rec["qd"].append(u.robot.data.joint_vel[:, arm].cpu().numpy().copy())
        _m, _d = u._contact_forces_split()
        rec["force"].append(float(torch.maximum(_m.max(), _d.max())))
        _p = u._env_local(u.robot.data.body_pos_w[:, u.palm_idx])
        rec["palm_z"].append(_p[:, 2].cpu().numpy().copy())
        _c = u._env_local(u.object.data.root_pos_w)
        rec["cup_xy"].append(_c[:, :2].cpu().numpy().copy())
        rec["phase"].append(phase)

    for _ in range(args.settle):
        step(act, 0)
    # ★z 만 아래로. palm 액션은 앵커 기준 델타라 −1 이 델타 박스 하한이다.
    down = act.clone()
    down[:, 2] = -1.0
    for _ in range(args.press):
        step(down, 1)
    for _ in range(args.release):
        step(act, 2)

    out = {k: np.asarray(v) for k, v in rec.items()}
    out["gain_tag"] = np.array([_TAG])
    np.savez_compressed(args.out, **out)

    q, fq, qd = out["q"], out["fq"], out["qd"]
    err = fq - q                                    # (T, N, 7)
    ph = out["phase"]
    press = ph == 1
    print(f"\n[RING] 게인={_TAG}  스텝={len(ph)}  → {args.out}", flush=True)
    print(f"  접촉력 최대        {out['force'].max():9.2f} N "
          f"(누르는 구간 평균 {out['force'][press].mean():.2f})", flush=True)
    print(f"  관절오차 최대      {np.abs(err).max():9.4f} rad", flush=True)
    print(f"  관절속도 최대      {np.abs(qd).max():9.4f} rad/s", flush=True)
    print(f"  palm z 최저        {out['palm_z'].min():9.4f} m", flush=True)
    print(f"  컵 xy 이동         {np.linalg.norm(out['cup_xy'][-1] - out['cup_xy'][0], axis=-1).mean():9.4f} m",
          flush=True)
    # ★진동 판정 — 누르는 구간에서 관절오차의 **부호 변화 횟수**. 단발 변형이면 0 에 가깝다.
    for j in range(7):
        e = err[press][:, 0, j]
        if len(e) < 4:
            continue
        flips = int(np.sum(np.diff(np.sign(e - e.mean())) != 0))
        print(f"    j{j+1}: 오차 범위 {e.min():+.4f}~{e.max():+.4f} rad · "
              f"평균교차 {flips}회 · qd 최대 {np.abs(qd[press][:, 0, j]).max():.3f} rad/s",
              flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

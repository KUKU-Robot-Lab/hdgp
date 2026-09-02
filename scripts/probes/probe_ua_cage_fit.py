"""컵이 엄지-손가락 사이(케이지)에 **물리적으로 들어가는가** — 종별 실측.

## 왜

RH56F1 은 엄지가 손바닥 앞으로 튀어나온 대향형이고, 09.02 palm-frame 실측에서
법선 방향 엄지-손가락 간극이 **80mm** 로 나왔다(엄지 끝 +80mm · 4지 끝 0mm).
`shaker_family` 는 지름 70.4~96.8mm 라, 큰 종은 그 사이로 못 들어간다.
학습에서 `wrap_frac` 이 세 런 내내 0.002 대에 붙어 있고 사용자 관찰로 엄지가
컵 rim **안으로** 들어가는 접근이 나온 것이 그 귀결로 보인다.

정책 능력과 기하를 가르기 위해, **컵을 케이지 중심에 강제로 놓고** 손만 닫는다.
  · 감싸진다  → 기하는 충분하다. 정책이 거기까지 못 가는 것(보상·탐색 문제)
  · 안 감싸진다 → 컵이 케이지에 비해 크다. 스케일을 줄여야 한다

## 사용

    cd hdgp && ~/rl_ws/IsaacLab/isaaclab.sh -p scripts/probes/probe_ua_cage_fit.py
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="open-rh_r_grasp_ua-play-lstm")
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--scales", default="",
                    help="쉼표 구분 추가 스케일 시험(예: 0.5,0.6,0.7). 비우면 뱅크 기본 8종")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.agnostic.tasks.grasp_ua.config  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> int:
    cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=8)
    cfg.scene.num_envs = 8
    cfg.enable_events = False
    env = gym.make(args_cli.task, cfg=cfg)
    u = env.unwrapped
    env.reset()

    # ---- 케이지 중심(월드) 계산 — 태스크가 쓰는 정의 그대로 -------------------------
    palm = u.robot.data.body_pos_w[:, u.palm_idx]
    cage_w = palm + torch.einsum("nij,j->ni", u._palm_ee_R(), u._cage_offset_palm)
    r_cage = float(u._r_cage)

    # ---- ★★손가락별·링크별 접촉 분해 (Phase 1 측정) --------------------------------
    #   집계값(중간 N / 원위 M)만으로는 지표를 못 정한다. "어느 손가락의 어느 링크가
    #   닿는가"를 봐야 이 손의 감쌈을 정의할 수 있다.
    #   컵은 **매 스텝 고정**한다 — 케이지에 놓으면 겹쳐서 튕겨나가 판정이 안 된다
    #   (실측 175~284mm 이동).
    from openarm.agnostic.modules import object_bank as _ob
    _specs = _ob.get(cfg.object_bank).specs
    names = [sp.id for sp in _specs]
    fingers = list(u.profile.finger_sensor_bodies.keys())
    thr = float(cfg.contact_force_threshold)
    q_hold = u.robot.data.joint_pos.clone()

    def freeze_at(t):
        st = u.object.data.root_state_w.clone()
        st[:, 0:3] = t
        st[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=u.device)
        st[:, 7:13] = 0.0
        u.object.write_root_state_to_sim(st)

    def close_to(frac):
        for i2 in range(args_cli.steps):
            f = frac * min(1.0, i2 / max(1, args_cli.steps - 80))
            u._syn_close[:] = f
            u._syn_target[:] = torch.lerp(
                u._syn_open.unsqueeze(0), u._syn_grip.unsqueeze(0), u._syn_close
            ).clamp(u._syn_lo.unsqueeze(0), u._syn_hi.unsqueeze(0))
            u.robot.set_joint_position_target(q_hold[:, u._arm_ids_t], joint_ids=u.arm_ids)
            u.robot.set_joint_position_target(u._syn_target, joint_ids=u._syn_ids)
            u._apply_mimic_targets()
            freeze_at(cage_w)
            u._contact_step_reset()
            u.scene.write_data_to_sim()
            u.sim.step(render=False)
            u.scene.update(u.physics_dt)
        u._contact_step_reset()

    close_to(0.0)
    mid0, dist0 = u._contact_forces_split(); tip0 = u._tip_contact_forces()
    palm0 = u._palm_contact_force()
    close_to(1.0)
    mid1, dist1 = u._contact_forces_split(); tip1 = u._tip_contact_forces()
    palm1 = u._palm_contact_force()

    print("\n" + "=" * 96, flush=True)
    print(f"[cage-fit] 컵 고정 · 케이지 반경 {r_cage*1000:.1f}mm · 손가락별 링크 접촉 분해",
          flush=True)
    print(f"[cage-fit] 링크 규약(4지): 중간={u.profile.finger_sensor_bodies[fingers[1]][0]} "
          f"원위={u.profile.finger_sensor_bodies[fingers[1]][1]} "
          f"팁={u.profile.finger_sensor_bodies[fingers[1]][2]}", flush=True)
    for i2 in range(u.num_envs):
        sp = _specs[i2 % len(_specs)]
        dia = 2 * sp.base_grasp_radius * sp.scale[0] * 1000
        openhit = int((mid0[i2] > thr).sum() + (dist0[i2] > thr).sum()
                      + (tip0[i2] > thr).sum() + int(palm0[i2] > thr))
        print(f"[cage-fit] {names[i2 % len(names)]:14s} 지름 {dia:5.1f}mm · 열린손 접촉 "
              f"{openhit:2d} {'★자리없음' if openhit else '자리있음  '}", flush=True)
        row = "[cage-fit]    닫은손 "
        for k, fg in enumerate(fingers):
            m = "M" if mid1[i2, k] > thr else "."
            d = "D" if dist1[i2, k] > thr else "."
            t = "T" if tip1[i2, k] > thr else "."
            row += f" {fg[:6]}:{m}{d}{t}"
        row += f"  palm:{'P' if palm1[i2] > thr else '.'}"
        print(row, flush=True)
    print("\n[cage-fit] (M=중간 D=원위 T=팁 P=손바닥)", flush=True)
    print("=" * 96, flush=True)
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

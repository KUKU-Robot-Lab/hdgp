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
parser.add_argument("--aim", default="cage", choices=["cage", "thumb_index"],
                    help="컵을 고정할 겨냥점. cage=현행(엄지 vs 4지평균) · "
                         "thumb_index=엄지팁·검지팁 중점")
parser.add_argument("--close", type=float, default=1.0,
                    help="★판정 폐쇄율. 1.0(완전폐쇄)은 **파지 자세가 아니다** — "
                         "09.03 실측에서 엄지↔검지 간극이 0.65 의 14.1mm 에서 1.00 의 "
                         "34.6mm 로 **다시 벌어진다**(두 손끝이 서로를 지나쳐 엇갈린다).")
parser.add_argument("--family", default="shaker", choices=["shaker", "cup"],
                    help="--scales 로 만들 임시 뱅크의 자산 계열")
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
    from openarm.agnostic.modules import object_bank as _ob

    # ★`--scales` — 뱅크에 없는 크기를 시험한다. 09.03 에 이 플래그가 **선언만 되고
    #   구현이 없어** 조용히 기본 뱅크를 다시 잰 사고가 있었다. 뱅크 밖 대역을
    #   재려면 임시 뱅크를 **env 생성 전에** 등록해야 한다(스폰 cfg 가 그때 굳는다).
    _n = 8
    if args_cli.scales.strip():
        _sc = [float(x) for x in args_cli.scales.split(",") if x.strip()]
        _ob.BANKS["_probe_scan"] = _ob.ObjectBank(
            name="_probe_scan",
            specs=tuple((_ob._cup if args_cli.family == "cup" else _ob._shaker)(x)
                        for x in _sc),
            note="probe_ua_cage_fit --scales 임시 뱅크(비영속)")
        _n = len(_sc)

    cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=_n)
    cfg.scene.num_envs = _n
    if args_cli.scales.strip():
        cfg.object_bank = "_probe_scan"
    cfg.enable_events = False
    env = gym.make(args_cli.task, cfg=cfg)
    u = env.unwrapped
    env.reset()

    # ---- 케이지 중심(월드) 계산 — 태스크가 쓰는 정의 그대로 -------------------------
    palm = u.robot.data.body_pos_w[:, u.palm_idx]
    _off = u._cage_offset_palm
    if args_cli.aim == "thumb_index":
        # ★★엄지–검지 겨냥. 현행 케이지는 `midpoint(엄지팁, 4지팁 평균)` 이라 중지·약지·
        #   소지가 중점을 **측방으로 17mm** 끌어당긴다. 09.03 실측에서 작은 물체에 실제로
        #   닿는 조합은 엄지+검지뿐이었고, 무는 구간(폐쇄 0.2~0.5)의 엄지–검지 중점
        #   측방 좌표는 33~35mm 로 거의 일정한데 케이지는 14.0mm 였다.
        _f = list(u.profile.finger_sensor_bodies.keys())
        _ti, _ii = _f.index("thumb"), _f.index("index")
        _R0 = u._palm_ee_R()[0]
        _tips = u.robot.data.body_pos_w[0, u._tip_ids_t]
        _o0 = u.robot.data.body_pos_w[0, u.palm_idx]
        _L = (_tips - _o0) @ _R0
        _off = (_L[_ti] + _L[_ii]) * 0.5
        print(f"[cage-fit] 겨냥 = 엄지–검지 중점(palm) "
              f"{[round(float(x) * 1000, 1) for x in _off]} mm "
              f"· 현행 케이지 {[round(float(x) * 1000, 1) for x in u._cage_offset_palm]} mm",
              flush=True)
    cage_w = palm + torch.einsum("nij,j->ni", u._palm_ee_R(), _off)
    r_cage = float(u._r_cage)

    # ---- ★★손가락별·링크별 접촉 분해 (Phase 1 측정) --------------------------------
    #   집계값(중간 N / 원위 M)만으로는 지표를 못 정한다. "어느 손가락의 어느 링크가
    #   닿는가"를 봐야 이 손의 감쌈을 정의할 수 있다.
    #   컵은 **매 스텝 고정**한다 — 케이지에 놓으면 겹쳐서 튕겨나가 판정이 안 된다
    #   (실측 175~284mm 이동).
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
    close_to(float(args_cli.close))
    mid1, dist1 = u._contact_forces_split(); tip1 = u._tip_contact_forces()
    palm1 = u._palm_contact_force()

    print("\n" + "=" * 96, flush=True)
    print(f"[cage-fit] 컵 고정 · 겨냥={args_cli.aim} · 판정폐쇄={args_cli.close:.2f} · "
          f"케이지 반경 {r_cage*1000:.1f}mm · 손가락별 링크 접촉 분해", flush=True)
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
        row = f"[cage-fit]    폐쇄{args_cli.close:.2f} "
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

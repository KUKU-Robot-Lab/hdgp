"""물체를 **어디에 놓아야** 이 손이 실제로 무는가 — palm 프레임 격자 전수.

## 왜

09.03 에 겨냥점 후보를 둘(현행 케이지 / 엄지–검지 중점) 놓고 A/B 했더니 **둘 다
접촉 0** 이었다. 즉 후보를 손으로 고르는 방식이 실패했다. 원인 후보가 셋인데
서로 얽혀 있어 이론으로 못 가른다:

  ① 케이지는 **열린 손** 손끝으로 정의되는데 손가락은 닫히며 다른 점으로 모인다
  ② `cage_ctr_dist` 는 물체 **원점**을, `finger_closure`·`palm_to_cup` 는
     **원점+30mm(grasp_center)** 를 쓴다 — 같은 보상 안에 기준점이 둘이다
  ③ `object_grasp_z_offset` 은 고정 30mm 라 물체를 스케일해도 안 따라간다

그래서 후보를 고르지 않고 **격자로 전수 측정**한다. env 마다 다른 배치 오프셋을
주고 같은 물체를 물려, 접촉이 실제로 나는 영역을 지도로 얻는다. 나온 영역의
중심이 곧 케이지가 가리켜야 할 점이다.

## 사용

    cd hdgp && ~/rl_ws/IsaacLab/isaaclab.sh -p scripts/probes/probe_ua_place_scan.py \
        --family cup --scale 0.58 --close 0.35
"""

from __future__ import annotations

import argparse
import itertools

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="open-rh_r_grasp_ua-play-lstm")
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--family", default="cup", choices=["shaker", "cup"])
parser.add_argument("--scale", type=float, default=0.58)
parser.add_argument("--close", type=float, default=0.60)
parser.add_argument("--hold_steps", type=int, default=150,
                    help="★해제 후 중력 낙하 관찰 스텝. 접촉 링크 수는 임계값 근처에서 "
                         "켜졌다 꺼지는 약한 대리지표다(4mm 이동에 5링크→1링크). "
                         "'잡히는가'는 놓아보면 바로 답이 나온다.")
parser.add_argument("--hold_drop_mm", type=float, default=15.0,
                    help="이 이하로 떨어지면 파지 성립")
parser.add_argument("--normal", default="25,40,55,70", help="palm 법선 축 [mm]")
parser.add_argument("--lateral", default="8,20,32,44", help="palm 측방 축 [mm]")
parser.add_argument("--finger", default="30,45,60,75", help="손가락방향 축 [mm]")
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

    grid = [tuple(float(v) / 1000.0 for v in c) for c in itertools.product(
        [float(x) for x in args_cli.normal.split(",")],
        [float(x) for x in args_cli.lateral.split(",")],
        [float(x) for x in args_cli.finger.split(",")])]
    n = len(grid)

    # ★단일 물체 뱅크 — env 마다 **물체는 같고 배치만** 달라야 격자가 성립한다.
    mk = _ob._cup if args_cli.family == "cup" else _ob._shaker
    _ob.BANKS["_place_scan"] = _ob.ObjectBank(
        name="_place_scan", specs=(mk(args_cli.scale),),
        note="probe_ua_place_scan 임시 뱅크(비영속)")

    cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=n)
    cfg.scene.num_envs = n
    cfg.object_bank = "_place_scan"
    cfg.enable_events = False
    env = gym.make(args_cli.task, cfg=cfg)
    u = env.unwrapped
    env.reset()

    fingers = list(u.profile.finger_sensor_bodies.keys())
    thr = float(cfg.contact_force_threshold)
    q_hold = u.robot.data.joint_pos.clone()
    spec = _ob.get("_place_scan").specs[0]
    dia = 2.0 * spec.base_grasp_radius * spec.scale[0] * 1000.0

    off = torch.tensor(grid, device=u.device)                       # (n,3) palm 프레임
    palm = u.robot.data.body_pos_w[:, u.palm_idx]
    target = palm + torch.einsum("nij,nj->ni", u._palm_ee_R(), off)  # (n,3) 월드

    def freeze() -> None:
        st = u.object.data.root_state_w.clone()
        st[:, 0:3] = target
        st[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=u.device)
        st[:, 7:13] = 0.0
        u.object.write_root_state_to_sim(st)

    def close_to(frac: float) -> None:
        for i in range(args_cli.steps):
            f = frac * min(1.0, i / max(1, args_cli.steps - 70))
            u._syn_close[:] = f
            u._syn_target[:] = torch.lerp(
                u._syn_open.unsqueeze(0), u._syn_grip.unsqueeze(0), u._syn_close
            ).clamp(u._syn_lo.unsqueeze(0), u._syn_hi.unsqueeze(0))
            u.robot.set_joint_position_target(q_hold[:, u._arm_ids_t], joint_ids=u.arm_ids)
            u.robot.set_joint_position_target(u._syn_target, joint_ids=u._syn_ids)
            u._apply_mimic_targets()
            freeze()
            u._contact_step_reset()
            u.scene.write_data_to_sim()
            u.sim.step(render=False)
            u.scene.update(u.physics_dt)
        u._contact_step_reset()

    close_to(0.0)
    m0, d0 = u._contact_forces_split(); t0 = u._tip_contact_forces()
    close_to(float(args_cli.close))
    m1, d1 = u._contact_forces_split(); t1 = u._tip_contact_forces()
    p1 = u._palm_contact_force()

    # ---- ★해제 후 낙하 — 진짜 판정 --------------------------------------------------
    #   고정한 채 재는 접촉 링크는 스치는 접촉을 세므로 배치 4mm 변화에 5→1 로 튄다.
    #   손을 그대로 둔 채 컵만 놓아 중력에 맡기면 "잡혔는가"가 이진으로 나온다.
    z_before = u.object.data.root_pos_w[:, 2].clone()
    for _ in range(int(args_cli.hold_steps)):
        u.robot.set_joint_position_target(q_hold[:, u._arm_ids_t], joint_ids=u.arm_ids)
        u.robot.set_joint_position_target(u._syn_target, joint_ids=u._syn_ids)
        u._apply_mimic_targets()
        u.scene.write_data_to_sim()
        u.sim.step(render=False)
        u.scene.update(u.physics_dt)
    drop_mm = (z_before - u.object.data.root_pos_w[:, 2]) * 1000.0
    # ★★"안 떨어졌다"의 함정 — 컵이 **테이블에 서 있으면** 손이 안 잡아도 안 떨어진다.
    #   바닥 여유 = 컵 바닥 z − 테이블 상면. 이게 0 근처면 파지가 아니라 지지다.
    _table_top = float(getattr(u.cfg, "table_surface_z", 0.20))
    _bottom = (u.object.data.root_pos_w[:, 2]
               - float(spec.base_origin_offset_z) * float(spec.scale[2]))
    clear_mm = (_bottom - _table_top) * 1000.0

    print("\n" + "=" * 96, flush=True)
    print(f"[place] 물체 {spec.id} 지름 {dia:.1f}mm · 판정폐쇄 {args_cli.close:.2f} · "
          f"격자 {n}점 (palm 프레임 mm)", flush=True)
    print(f"[place] 참고 — 현행 케이지 오프셋 "
          f"{[round(float(x) * 1000, 1) for x in u._cage_offset_palm]} mm", flush=True)
    rows = []
    for i in range(n):
        openhit = int((m0[i] > thr).sum() + (d0[i] > thr).sum() + (t0[i] > thr).sum())
        links = int((m1[i] > thr).sum() + (d1[i] > thr).sum() + (t1[i] > thr).sum())
        thumb_on = bool((m1[i, 0] > thr) | (d1[i, 0] > thr) | (t1[i, 0] > thr))
        four_on = bool(((m1[i, 1:] > thr) | (d1[i, 1:] > thr) | (t1[i, 1:] > thr)).any())
        pat = " ".join(
            f"{fingers[k][:5]}:"
            f"{'M' if m1[i, k] > thr else '.'}"
            f"{'D' if d1[i, k] > thr else '.'}"
            f"{'T' if t1[i, k] > thr else '.'}" for k in range(len(fingers)))
        dm, cl = float(drop_mm[i]), float(clear_mm[i])
        held = (openhit == 0) and (abs(dm) <= float(args_cli.hold_drop_mm)) \
            and cl > 5.0 and links > 0
        rows.append((held, links, thumb_on and four_on, openhit, grid[i], pat, dm, cl))
    rows.sort(key=lambda r: (not r[0], -r[1], abs(r[6])))
    print("[place] 상위 14 — 정렬 기준 = **파지 성립 후 낙하량**", flush=True)
    for held, links, opp, openhit, g, pat, dm, cl in rows[:14]:
        print(f"[place]  ({g[0]*1000:5.0f},{g[1]*1000:5.0f},{g[2]*1000:5.0f})mm "
              f"{'★잡힘' if held else '  놓침'} 낙하{dm:7.1f}mm 바닥여유{cl:6.1f}mm "
              f"링크{links:2d} {'대향O' if opp else '대향X'} "
              f"{'열린손충돌' if openhit else '          '} {pat}", flush=True)
    ok = [r for r in rows if r[0]]
    print(f"[place] ★파지 성립 {len(ok)}/{n} · 그중 대향 성립 "
          f"{sum(1 for r in ok if r[2])}", flush=True)
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

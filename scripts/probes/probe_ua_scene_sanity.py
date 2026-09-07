"""★장면 검증 — 컵이 테이블 위에 가만히 서 있는가.

09.03: 스캔 프로브를 여덟 번 돌리는 동안 여덟 번 다 무효였고, 마지막 무효 원인은
**컵이 테이블을 통과해 바닥으로 떨어져 있었던 것**이었다(밀림 222.7mm 가 36개 env
전부 동일 = 손이 민 게 아니라 낙하 거리). 손을 움직이기 전에 장면부터 재야 한다.

손을 전혀 건드리지 않고 컵만 놓고 본다:
  · 스폰 z → 정착 z (얼마나 떨어지는가)
  · 정착 후 바닥 여유 = 컵 바닥 − 테이블 상면 (0 이면 정상 착지)
  · 기울기 (서 있는가)
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="open-rh_r_grasp_ua-play-lstm")
parser.add_argument("--family", default="cup", choices=["shaker", "cup", "default"])
parser.add_argument("--scales", default="0.46,0.58,0.70,0.85,1.00")
parser.add_argument("--steps", type=int, default=300)
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

    sc = [float(x) for x in args_cli.scales.split(",")]
    n = len(sc)
    cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=n)
    cfg.scene.num_envs = n
    cfg.enable_events = False
    cfg.enable_adr = False
    cfg.episode_length_s = 120.0
    if args_cli.family != "default":
        mk = _ob._cup if args_cli.family == "cup" else _ob._shaker
        _ob.BANKS["_scene_scan"] = _ob.ObjectBank(
            name="_scene_scan", specs=tuple(mk(x) for x in sc),
            note="probe_ua_scene_sanity 임시 뱅크(비영속)")
        cfg.object_bank = "_scene_scan"
    env = gym.make(args_cli.task, cfg=cfg)
    u = env.unwrapped
    env.reset()

    specs = _ob.get(cfg.object_bank).specs
    z0 = u.object.data.root_pos_w[:, 2].clone()
    q_hold = u.robot.data.joint_pos.clone()
    for _ in range(args_cli.steps):
        u.robot.set_joint_position_target(q_hold, joint_ids=None)
        u.scene.write_data_to_sim()
        u.sim.step(render=False)
        u.scene.update(u.physics_dt)
    z1 = u.object.data.root_pos_w[:, 2]

    from isaaclab.utils.math import quat_apply
    up = quat_apply(u.object.data.root_quat_w,
                    torch.tensor([0.0, 0.0, 1.0], device=u.device).expand(n, 3))
    tilt = torch.rad2deg(torch.acos(up[:, 2].clamp(-1.0, 1.0)))
    top = float(cfg.table_surface_z)

    print("\n" + "=" * 92, flush=True)
    print(f"[scene] {args_cli.family} · 테이블 상면 z {top:.3f} · "
          f"cfg.object_origin_offset_z {float(cfg.object_origin_offset_z):.4f}", flush=True)
    ok = 0
    for i in range(n):
        sp = specs[i % len(specs)]
        off = float(sp.base_origin_offset_z) * float(sp.scale[2])
        bottom = float(z1[i]) - off
        drop = (float(z0[i]) - float(z1[i])) * 1000.0
        clear = (bottom - top) * 1000.0
        good = abs(clear) < 6.0 and float(tilt[i]) < 5.0
        ok += int(good)
        print(f"[scene] {sp.id:16s} 스폰z {float(z0[i]):.4f} → 정착z {float(z1[i]):.4f} "
              f"(낙하 {drop:7.1f}mm) · 원점오프셋 {off*1000:5.1f}mm · "
              f"바닥여유 {clear:8.1f}mm · 기울 {float(tilt[i]):5.1f}° "
              f"{'OK' if good else '★비정상'}", flush=True)
    print(f"[scene] 정상 {ok}/{n} — 비정상이면 이 자산·스케일로는 어떤 파지 측정도 무의미하다",
          flush=True)
    print("=" * 92, flush=True)
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

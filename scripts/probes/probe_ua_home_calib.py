"""grasp_ua 홈/워크스페이스 캘리브레이션 — **엔드이펙터마다 다시 재야 하는 값**을 뽑는다.

## 왜 EE 마다 다시 재는가

`init_joint_pos` 의 팔 관절값은 "팔 홈"이 아니라 **palm 포즈**를 정의한다. 같은 팔이라도
손이 바뀌면 palm 프레임의 위치·축이 달라져 같은 관절값이 전혀 다른 자세를 낸다.
09.02 실측(RH56F1): 자매 트랙 팔 홈(0.5, 0.1, 0.4, 0.60, −0.2, 0, 0)이
`r_hl_palm_sensor` 기준으로 `ez −9.6° · ey −62.7° · ex −99.1°` 를 내는데, 의도한
side-grasp pregrasp 는 `ez 180° · ey 0° · ex 90°` 다. 부팅 게이트가 이걸 잡아
**환경 생성 자체를 막는다**(조용히 엉뚱한 자세로 학습되는 것보다 낫다).

## 무엇을 뽑는가

1. 목표 palm 포즈를 fabric 으로 추종시켜 **수렴한 팔 관절값** → `init_joint_pos`
2. 그 자세에서의 **케이지 중심·반경** → `object_spawn_center` 제안
   (★스폰을 케이지 x 에 정렬하되 y 는 케이지 반경보다 크게 띄운다 — 안 그러면 리셋
     순간 컵이 홈 케이지 **안**에 들어가 손가락 메시가 컵을 관통한다)
3. 액션 델타 박스(앵커 ± `palm_delta_xyz`) 격자 **도달성** → `palm_box_verified` 근거

## 사용

    cd hdgp && ~/rl_ws/IsaacLab/isaaclab.sh -p scripts/probes/probe_ua_home_calib.py \
        --task open-rh_r_grasp_ua-play-lstm --pos 0.43,-0.275,0.325 --rot 180,0,90

★프로필 파일은 안 건드린다. 출력값을 사람이 보고 옮겨 적는다(자동 반영 금지 —
  홈은 체크포인트에 딸린 값이라 조용히 바뀌면 재생이 깨진다).
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="open-rh_r_grasp_ua-play-lstm")
parser.add_argument("--pos", default="", help="목표 palm 위치 x,y,z [m]. 비우면 현재 홈 위치.")
parser.add_argument("--rot", default="", help="목표 palm 자세 ez,ey,ex [deg]. 비우면 프로필 중심.")
parser.add_argument("--settle", type=int, default=400, help="수렴 스텝")
parser.add_argument("--reach_settle", type=int, default=120, help="격자 점당 스텝")
parser.add_argument("--skip_reach", action="store_true")
parser.add_argument("--scan", default="",
                    help="위치 후보를 ';' 로 나열해 도달성만 훑는다. "
                         "예: 0.40,-0.30,0.45;0.45,-0.30,0.42")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
_app = AppLauncher(args_cli).app

import dataclasses  # noqa: E402
import itertools  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.agnostic.tasks.grasp_ua.config  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from openarm.agnostic.tasks.grasp_ua import robot_profiles as _rp  # noqa: E402

_D = math.pi / 180.0


def _relax(cfg):
    """캘리브레이션 **전용** 완화 — 재려는 값이 게이트라서, 그 게이트를 열고 들어간다.

    ★프로필 파일이 아니라 런타임 딕셔너리 항목만 교체한다. 완화 **전** 원본을 돌려줘
      제안값을 원래 박스와 대조할 수 있게 한다.
    """
    orig = _rp.PROFILES[cfg.profile_name]
    cfg.enable_events = False        # 자산 shape 회계와 무관하게 돌아야 한다
    _rp.PROFILES[cfg.profile_name] = dataclasses.replace(
        orig,
        palm_box_min=(-2.0, -2.0, -1.0), palm_box_max=(2.0, 2.0, 2.0),
        palm_rot_half_deg=179.0, palm_rot_center_deg=(0.0, 0.0, 0.0),
    )
    return orig


def _drive(u, target: torch.Tensor, steps: int) -> None:
    """fabric attractor 로 palm 을 목표까지 끌고 간다(정책 없이).

    ★손을 **open 자세로 고정**한다. `_syn_target` 은 `_setup_synergy` 시점의 실측
      관절값이라 홈이 아직 안 써진 상태를 담고 있다 — 그대로 두면 엄지가 대향
      자세(1.57)로 안 벌어진 채 돌아 **케이지가 실제와 다르게 나온다**
      (09.02 실측: 같은 palm 자세인데 케이지 z 가 42mm 어긋났다).
    """
    u._syn_target[:] = u._syn_open.unsqueeze(0)
    u._syn_close[:] = 0.0
    u.palm_targets[:] = target.unsqueeze(0)
    for _ in range(steps):
        u._step_fabric()
        for _ in range(int(u.cfg.decimation)):
            u._apply_action()
            u.scene.write_data_to_sim()
            u.sim.step(render=False)
            u.scene.update(u.physics_dt)


def _wrap(a: torch.Tensor) -> torch.Tensor:
    """각도차를 (−π, π] 로 접는다. ★안 접으면 ez 180° 와 −180° 가 360° 차이로 읽혀
    수렴한 자세를 실패로 오판한다(09.02 실측: rot 오차 357° = 사실상 2.6°)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _pose_err(u, target: torch.Tensor) -> tuple[float, float]:
    got = u._palm_pose_6d()[0]
    dp = float((got[:3] - target[:3]).norm())
    dr = float(_wrap(got[3:] - target[3:]).abs().max())
    return dp, dr


def main() -> int:
    cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
    cfg.scene.num_envs = 1
    orig = _relax(cfg)
    env = gym.make(args_cli.task, cfg=cfg)
    u = env.unwrapped
    dev = u.device

    home = u._home_palm.clone()
    pos = ([float(v) for v in args_cli.pos.split(",")] if args_cli.pos
           else home[:3].tolist())
    rot = ([float(v) * _D for v in args_cli.rot.split(",")] if args_cli.rot
           else [v * _D for v in orig.palm_rot_center_deg])
    target = torch.tensor(pos + rot, device=dev, dtype=torch.float32)

    print("\n" + "=" * 78, flush=True)
    print(f"[home-calib] 프로필 = {orig.name}", flush=True)
    print(f"[home-calib] 완화 전 홈  = {[round(float(v), 4) for v in home]}", flush=True)
    print(f"[home-calib] 목표 palm   = {[round(float(v), 4) for v in target]}"
          f"  (자세 {[round(v / _D, 1) for v in rot]}°)", flush=True)

    if args_cli.scan:
        print("[home-calib] ---- 위치 후보 스캔 (자세 고정) ----", flush=True)
        best = None
        for cand in args_cli.scan.split(";"):
            xyz = [float(v) for v in cand.split(",")]
            t = torch.tensor(xyz + rot, device=dev, dtype=torch.float32)
            _drive(u, t, args_cli.settle)
            e, r = _pose_err(u, t)
            g = u._palm_pose_6d()[0]
            _t = (u.robot.data.body_pos_w[:, u._tip_ids_t]
                  - u.scene.env_origins[:, None, :])[0]
            _ai = int(u._group_a_idx[0])
            _oi = [i for i in range(len(u.tip_ids)) if i != _ai]
            _cg = 0.5 * (_t[_ai] + _t[_oi].mean(dim=0))
            _rc = 0.5 * float((_t[_ai] - _t[_oi].mean(dim=0)).norm())
            print(f"[home-calib]   {cand:22s} pos {e * 1000:6.1f}mm rot {r / _D:5.1f}° "
                  f"| 케이지 {[round(float(v), 3) for v in _cg]} r{_rc * 1000:.0f}mm", flush=True)
            if best is None or (e, r) < best[0]:
                best = ((e, r), cand)
        print(f"[home-calib] 최선 = {best[1]} (pos {best[0][0] * 1000:.1f}mm "
              f"rot {best[0][1] / _D:.1f}°)", flush=True)
        env.close()
        return 0

    for _k in range(4):
        _drive(u, target, max(1, args_cli.settle // 4))
        _dp, _dr = _pose_err(u, target)
        print(f"[home-calib]   수렴 {(_k + 1) * args_cli.settle // 4:4d}스텝 "
              f"pos {_dp * 1000:7.2f}mm · rot {_dr / _D:6.2f}°", flush=True)
    dp, dr = _pose_err(u, target)
    _got = u._palm_pose_6d()[0]
    print(f"[home-calib] 도달 palm  = {[round(float(v), 4) for v in _got]} "
          f"(자세 {[round(float(v) / _D, 1) for v in _got[3:]]}°)", flush=True)
    print(f"[home-calib] 축별 오차   = "
          f"xyz {[round(float(_got[i] - target[i]) * 1000, 1) for i in range(3)]}mm · "
          f"ezyx {[round(float(_wrap(_got[3 + i] - target[3 + i])) / _D, 1) for i in range(3)]}°",
          flush=True)
    q_arm = u.robot.data.joint_pos[0, u._arm_ids_t]
    names = [u.robot.data.joint_names[i] for i in u.arm_ids]
    ok_pose = dp <= 0.005 and dr <= 2.0 * _D
    print(f"[home-calib] {'OK   ' if ok_pose else '★FAIL'} 수렴 오차 "
          f"pos {dp * 1000:.2f}mm · rot {dr / _D:.2f}°  (허용 5mm / 2°)", flush=True)
    print("[home-calib] ---- init_joint_pos 에 옮겨 적을 값 ----", flush=True)
    print("        " + " ".join(f'"{n}": {float(v):+.4f},' for n, v in zip(names, q_arm)),
          flush=True)

    # ---- 케이지 (스폰 중심 제안) ------------------------------------------------
    tips = (u.robot.data.body_pos_w[:, u._tip_ids_t]
            - u.scene.env_origins[:, None, :])[0]
    _a = int(u._group_a_idx[0])
    _others = [i for i in range(len(u.tip_ids)) if i != _a]
    cage = 0.5 * (tips[_a] + tips[_others].mean(dim=0))
    r_cage = 0.5 * float((tips[_a] - tips[_others].mean(dim=0)).norm())
    _sep = round(r_cage + 0.03, 3)          # 케이지 반경 + 여유 30mm
    print(f"[home-calib] 케이지 중심 = {[round(float(v), 4) for v in cage]} "
          f"· 반경 {r_cage * 1000:.0f}mm", flush=True)
    print(f"[home-calib] ---- object_spawn_center 제안 ----", flush=True)
    print(f"        ({float(cage[0]):.3f}, {float(cage[1]) + _sep:.3f})   "
          f"# x=케이지 정렬 · y=케이지에서 {_sep * 1000:.0f}mm(반경+30) 띄움", flush=True)
    print(f"        ★y 를 반경({r_cage * 1000:.0f}mm)보다 크게 띄우는 이유: 컵이 홈 케이지 "
          f"안에서 시작하면 리셋 순간 손가락 메시가 컵을 관통한다", flush=True)

    # ---- 도달성 격자 -------------------------------------------------------------
    if not args_cli.skip_reach:
        d = [float(v) for v in cfg.palm_delta_xyz]
        worst, fails = 0.0, []
        pts = list(itertools.product(*[(-1, 0, 1)] * 3))
        for k, sgn in enumerate(pts):
            t = target.clone()
            for i in range(3):
                t[i] = target[i] + sgn[i] * d[i]
            # ★매 점마다 중심으로 되돌린다. 도달 불가점으로 **순간 이동**시키면 fabric 이
            #   발산하고(09.02 실측 648mm) 그 상태가 다음 점으로 이월돼 전부 실패로 읽힌다.
            #   학습에서는 `palm_cmd_rate_limit_m` 이 이런 점프를 애초에 막는다.
            _drive(u, target, max(40, args_cli.reach_settle // 2))
            _drive(u, t, args_cli.reach_settle)
            e, _ = _pose_err(u, t)
            worst = max(worst, e)
            if e > 0.01:
                fails.append((tuple(sgn), round(e * 1000, 1)))
            print(f"[home-calib]   격자 {k + 1:2d}/27 offset={sgn} 오차 {e * 1000:6.1f}mm",
                  flush=True)
        ok_reach = not fails
        print(f"[home-calib] {'OK   ' if ok_reach else '★FAIL'} 델타박스 도달성 "
              f"±{d} m · 최악 {worst * 1000:.1f}mm (허용 10mm)", flush=True)
        if fails:
            print(f"[home-calib]   미달 {len(fails)}점: {fails}", flush=True)
    else:
        ok_reach = False
        print("[home-calib] (도달성 격자 생략)", flush=True)

    print("=" * 78, flush=True)
    env.close()
    return 0 if (ok_pose and ok_reach) else 1


if __name__ == "__main__":
    import traceback
    try:
        _rc = main()
    except BaseException:
        traceback.print_exc()
        _rc = 3
    _app.close()
    raise SystemExit(_rc)

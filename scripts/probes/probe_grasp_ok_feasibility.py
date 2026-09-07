#!/usr/bin/env python3
"""`grasp_ok` 를 참으로 만드는 **액션이 존재하는가**를 직접 탐색한다.

왜
--
t30 은 790 epoch 동안 아래 다섯 항이 **정확히 0** 이었다:
    lifting_object · grasp_pose · grip_closure_when_enclosed
    object_goal_tracking · settled_at_goal
다섯은 전부 `grasp_ok` 하나에 걸려 있고, **그리퍼 하드 게이트도 같은 술어**다
(거짓이면 그리퍼가 강제로 열린다). 즉 과제 전체가 이 술어 하나를 목으로 지난다.

"느리게 배우는 중"과 "도달 불가능"은 학습 곡선으로 못 가른다 — 둘 다 0 으로 보인다.
그래서 학습이 아니라 **탐색으로 상한을 잰다.** 정책이 못 찾는 것과 존재하지 않는 것은
다른 문제이고, 후자면 보상·학습기를 아무리 고쳐도 소용없다.

방법
----
6D palm 액션에 대해 CEM(cross-entropy method). 라운드마다 env 마다 다른 액션을 뿌리고,
지령을 **고정한 채 정착**시킨 뒤 `grasp_ok` 의 세 성분을 잰다. 상위 elite 로 평균·분산을
갱신한다. 학습이 아니라 **직접 최적화**이므로, 여기서도 못 만들면 정책은 더 못 만든다.

⚠ 성분을 따로 본다 — 실패해도 **어느 조건에서 막혔는지** 알아야 고칠 수 있다:
    lateral  (턱 축에 수직한 컵 축까지 거리)     < GRASP_GATE_LATERAL_OK
    along    (턱 개구 방향 어긋남)                < GRASP_GATE_ALONG_OK
    axis_t   (턱 중점의 컵 축 좌표)               ∈ CUP_GRASP_BAND_AXIS
⚠ 리셋 오염 차단: episode_length_s 무한 · 종료 항 제거 · ADR 제거.

사용:
  TERM=xterm PYTHONUNBUFFERED=1 PYTHONPATH=<hdgp>/source/openarm \\
    ~/rl_ws/IsaacLab/isaaclab.sh -p scripts/probes/probe_grasp_ok_feasibility.py --num_envs 512
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--rounds", type=int, default=18)
parser.add_argument("--settle", type=int, default=40, help="지령 고정 후 정착 스텝")
parser.add_argument("--elite", type=float, default=0.1)
parser.add_argument("--sigma0", type=float, default=0.7)
parser.add_argument("--restitution", type=float, default=None,
                    help="컵·로봇 반발계수 오버라이드. 현재 kuka 정합값은 1.0, t16 은 0 이었다.")
parser.add_argument("--depenetration", type=float, default=None,
                    help="max_depenetration_velocity 오버라이드. 현재 1000, t16 은 5 였다.")
parser.add_argument("--rot_scale", type=float, default=1.0,
                    help="회전 액션 범위 배율. 1.0 = 현재(±45°). t16 은 축각 ≤30° 였다.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import openarm.gripper.left.grasp_sensor  # noqa: F401,E402
from openarm.gripper.left.grasp_sensor import grasp_left_preset as P  # noqa: E402
from openarm.gripper.left.grasp_sensor import grasp_left_rewards as R  # noqa: E402

TASK = "open-grip_l_grasp_sensor_fab"


def components(env, jaw_cfg):
    """`grasp_ok` 의 세 성분을 그대로 계산한다 (보상 코드와 **같은 자**)."""
    p_l, p_r, u, mid, cup_pt, axis_t = R._jaw_frame(
        env, P.JAW_PAD_OFFSET, jaw_cfg, SceneEntityCfg("object"))
    d = cup_pt - mid
    along = (d * u).sum(-1).abs()
    lateral = (d - u * (d * u).sum(-1, keepdim=True)).norm(dim=-1)
    return lateral, along, axis_t


def main() -> None:
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.episode_length_s = 1.0e9
    for t in ("time_out", "object_dropping", "object_out_of_workspace"):
        setattr(cfg.terminations, t, None)
    cfg.curriculum.adr = None
    cfg.events.reset_object_position.params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}   # 컵 고정 — 상한을 재는 것이다
    if args.restitution is not None:
        r = (args.restitution, args.restitution)
        cfg.events.cup_physics_material.params["restitution_range"] = r
        cfg.events.robot_physics_material.params["restitution_range"] = r
    if args.depenetration is not None:
        for _sp in (cfg.scene.robot.spawn, cfg.scene.object.spawn):
            if getattr(_sp, "rigid_props", None) is not None:
                _sp.rigid_props.max_depenetration_velocity = args.depenetration
    env = gym.make(TASK, cfg=cfg).unwrapped
    env.reset()
    dev, n = env.device, env.num_envs
    jaw_cfg = SceneEntityCfg("robot", body_names=list(P.GRIPPER_FINGER_BODIES))
    jaw_cfg.resolve(env.scene)

    lat_ok, along_ok = P.GRASP_GATE_LATERAL_OK, P.GRASP_GATE_ALONG_OK
    band = P.CUP_GRASP_BAND_AXIS
    print(f"목표: lateral < {lat_ok*1000:.0f} mm · along < {along_ok*1000:.0f} mm · "
          f"axis_t ∈ [{band[0]*1000:.0f}, {band[1]*1000:.0f}] mm")
    print(f"회전 범위 ×{args.rot_scale} (= ±{P.PALM_MAX_POSE_ANGLE*args.rot_scale*57.2958:.1f}°)\n")

    mu = torch.zeros(6, device=dev)
    sigma = torch.full((6,), args.sigma0, device=dev)
    n_elite = max(4, int(n * args.elite))
    best = None

    obj = env.scene["object"]
    spawn = obj.data.default_root_state.clone()
    spawn[:, :3] += env.scene.env_origins

    for r in range(args.rounds):
        # ★★라운드마다 컵을 **다시 세운다.** 종료를 껐기 때문에 한 번 넘어지면 그 뒤 판정이
        #   전부 쓰러진 컵 기준이 된다(`_jaw_frame` 은 컵 자기 z축을 쓴다). 1 차 실행에서
        #   이걸 빠뜨려 성공률 0.8% 를 얻었는데, 그 수는 신뢰할 수 없다.
        obj.write_root_pose_to_sim(spawn[:, :7])
        obj.write_root_velocity_to_sim(torch.zeros_like(spawn[:, 7:]))
        a = torch.zeros(n, env.action_manager.total_action_dim, device=dev)
        samp = (mu + sigma * torch.randn(n, 6, device=dev)).clamp(-1.0, 1.0)
        samp[:, 3:6] *= args.rot_scale
        a[:, :6] = samp
        a[:, 6:] = -1.0                       # 그리퍼는 게이트가 강제로 연다
        for _ in range(args.settle):
            env.step(a)
        lateral, along, axis_t = components(env, jaw_cfg)
        # 컵이 서 있는 상태에서만 유효한 판정이다 — 넘어졌으면 기하가 통째로 다르다.
        upright = R._cup_upright_cos(env, SceneEntityCfg("object"))
        # 비용 = 세 조건의 위반량 합 (전부 m 단위로 맞춘다)
        in_band_pen = (band[0] - axis_t).clamp(min=0.0) + (axis_t - band[1]).clamp(min=0.0)
        cost = ((lateral - lat_ok).clamp(min=0.0)
                + (along - along_ok).clamp(min=0.0) + in_band_pen)
        ok = (lateral < lat_ok) & (along < along_ok) & (axis_t > band[0]) & (axis_t < band[1])
        idx = torch.topk(-cost, n_elite).indices
        mu = samp[idx].mean(dim=0)
        sigma = samp[idx].std(dim=0).clamp(min=0.03)
        j = int(torch.argmin(cost))
        if best is None or float(cost[j]) < best[0]:
            best = (float(cost[j]), float(lateral[j]), float(along[j]), float(axis_t[j]),
                    samp[j].tolist())
        print(f"라운드{r+1:3d}  grasp_ok {int(ok.sum()):4d}/{n}  "
              f"컵 서있음 {int((upright > 0.9).sum()):4d}/{n}  "
              f"best cost {float(cost[j])*1000:7.1f} mm | "
              f"lat {float(lateral[j])*1000:6.1f}  along {float(along[j])*1000:6.1f}  "
              f"axis_t {float(axis_t[j])*1000:7.1f}")

    c, lat, alo, ax, act = best
    print("\n=== 결론 ===")
    print(f"최적 액션 {['%.3f' % v for v in act]}")
    print(f"  lateral {lat*1000:6.1f} mm  (목표 < {lat_ok*1000:.0f})   "
          f"{'OK' if lat < lat_ok else '★실패'}")
    print(f"  along   {alo*1000:6.1f} mm  (목표 < {along_ok*1000:.0f})   "
          f"{'OK' if alo < along_ok else '★실패'}")
    print(f"  axis_t  {ax*1000:6.1f} mm  (목표 {band[0]*1000:.0f}~{band[1]*1000:.0f})  "
          f"{'OK' if band[0] < ax < band[1] else '★실패'}")
    if c <= 0.0:
        print("→ **도달 가능**. 액션 공간에 해가 있다 = 학습/보상 문제다.")
    else:
        print(f"→ **도달 불가능** (잔여 위반 {c*1000:.1f} mm). 직접 최적화도 못 만든다 —")
        print("   보상이나 학습기를 고쳐도 소용없다. 액션 공간·기하부터 고쳐야 한다.")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

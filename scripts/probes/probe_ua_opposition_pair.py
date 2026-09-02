"""이 손의 **대향쌍**이 무엇이고, 그 틈이 케이지 중심과 얼마나 어긋나는가.

## 왜

`_cage_offset_palm` 은 `midpoint(엄지팁, 나머지 4지팁 평균)` 이다 — 엄지 대 4지
전체의 파워그립 모델이다. 그런데 09.03 케이지 고정 실측에서 작은 종에 실제로
닿는 조합은 **엄지+검지**뿐이었다(s055/s058/s061/s067 전부 `thumb + index`).
4지 평균은 중지·약지·소지 쪽으로 끌려가므로, 케이지 중심은 엄지–검지 틈이 아닌
다른 점을 가리킨다. 정책은 `cage_ctr_dist` 를 19mm 까지 줄였는데도 0.71 로 닫아
접촉 0 이었다 — **시킨 대로 했는데 그 지점이 잡히는 자리가 아니다**는 가설이다.

여기서 재는 것:
  · 개방/부분폐쇄/완전폐쇄에서 손끝 5개의 palm 프레임 좌표
  · 엄지↔각 손가락 간극과 그 중점
  · 현재 케이지 중심 대비 **엄지–검지 중점의 오프셋**
  · 폐쇄가 진행될 때 각 쌍의 간극이 어떻게 닫히는가(어느 쌍이 실제로 물체를 문다)

## 사용

    cd hdgp && ~/rl_ws/IsaacLab/isaaclab.sh -p scripts/probes/probe_ua_opposition_pair.py
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="open-rh_r_grasp_ua-play-lstm")
parser.add_argument("--steps", type=int, default=260, help="각 폐쇄 단계 정착 스텝")
parser.add_argument("--fracs", default="0.0,0.35,0.7,1.0", help="시험할 폐쇄 비율")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import openarm.agnostic.tasks.grasp_ua.config  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> int:
    cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=2)
    cfg.scene.num_envs = 2
    cfg.enable_events = False
    env = gym.make(args_cli.task, cfg=cfg)
    u = env.unwrapped
    env.reset()

    fingers = list(u.profile.finger_sensor_bodies.keys())
    q_hold = u.robot.data.joint_pos.clone()

    def settle(frac: float) -> None:
        for i in range(args_cli.steps):
            f = frac * min(1.0, i / max(1, args_cli.steps - 60))
            u._syn_close[:] = f
            u._syn_target[:] = torch.lerp(
                u._syn_open.unsqueeze(0), u._syn_grip.unsqueeze(0), u._syn_close
            ).clamp(u._syn_lo.unsqueeze(0), u._syn_hi.unsqueeze(0))
            u.robot.set_joint_position_target(q_hold[:, u._arm_ids_t], joint_ids=u.arm_ids)
            u.robot.set_joint_position_target(u._syn_target, joint_ids=u._syn_ids)
            u._apply_mimic_targets()
            u.scene.write_data_to_sim()
            u.sim.step(render=False)
            u.scene.update(u.physics_dt)

    def palm_local() -> torch.Tensor:
        """손끝 5개를 palm 프레임으로 — 열0=법선 열1=측방 열2=손가락방향."""
        R = u._palm_ee_R()[0]                                    # (3,3)
        p = u.robot.data.body_pos_w[0, u._tip_ids_t]             # (F,3)
        o = u.robot.data.body_pos_w[0, u.palm_idx]               # (3,)
        return (p - o) @ R                                       # (F,3)

    print(f"\n[opp] 손가락 순서: {fingers}")
    print(f"[opp] 현재 케이지 오프셋(palm) = "
          f"{[round(float(x) * 1000, 1) for x in u._cage_offset_palm]} mm · "
          f"반경 {float(u._r_cage) * 1000:.1f}mm")
    print("[opp] 좌표 규약: (법선, 측방, 손가락방향) mm\n")

    ti = fingers.index("thumb")
    for frac in [float(x) for x in args_cli.fracs.split(",")]:
        env.reset()
        q_hold = u.robot.data.joint_pos.clone()
        settle(frac)
        L = palm_local()
        print(f"── 폐쇄 {frac:.2f} " + "─" * 62)
        for k, f in enumerate(fingers):
            print(f"   {f:7s} tip = ({L[k, 0] * 1000:7.1f},{L[k, 1] * 1000:7.1f},"
                  f"{L[k, 2] * 1000:7.1f})")
        # 엄지↔각 손가락
        best = None
        for k, f in enumerate(fingers):
            if k == ti:
                continue
            gap = float((L[k] - L[ti]).norm()) * 1000.0
            mid = (L[k] + L[ti]) * 0.5
            d_cage = float((mid - u._cage_offset_palm).norm()) * 1000.0
            print(f"   엄지↔{f:7s} 간극 {gap:6.1f}mm · 중점 "
                  f"({mid[0] * 1000:6.1f},{mid[1] * 1000:6.1f},{mid[2] * 1000:6.1f}) · "
                  f"케이지중심과 {d_cage:5.1f}mm 어긋남")
            if best is None or gap < best[1]:
                best = (f, gap)
        # 현행 케이지 정의(엄지 vs 4지 평균)
        others = torch.stack([L[k] for k in range(len(fingers)) if k != ti]).mean(dim=0)
        mid4 = (others + L[ti]) * 0.5
        print(f"   [현행] 엄지↔4지평균 간극 {float((others - L[ti]).norm()) * 1000:6.1f}mm · "
              f"중점 ({mid4[0] * 1000:6.1f},{mid4[1] * 1000:6.1f},{mid4[2] * 1000:6.1f})")
        print(f"   → 가장 좁은 쌍: 엄지↔{best[0]} ({best[1]:.1f}mm)\n")

    env.close()
    return 0


if __name__ == "__main__":
    try:
        main()
    finally:
        _app.close()

#!/usr/bin/env python3
"""손 최하단 ↔ palm 원점 수직 오프셋 실측 — 테이블 바닥 여유(s2r 안전) 설정용.

왜 재는가 (09.01)
-----------------
사용자 관찰: "컵을 테이블 밑바닥까지 쓸 정도로 내려간다. 실기가 안 망가지려면
손이 적어도 1 cm 는 올라가야 한다."

그런데 palm 지령 상자 하한(`palm_box_min` z = 0.20 = 테이블 상면)은 **걸리지 않는다** —
학습 실측 `fabric/palm_z_min` 이 E1 0.2915 · E2 0.2769 로 이미 그보다 훨씬 위다.
즉 바닥을 쓰는 것은 **palm 이 아니라 손끝**이고, 상자 하한을 얼마로 올려야 하는지는
**palm 원점 ↔ 손 최하단**의 수직 거리를 알아야 정해진다.

    필요한 palm 하한 = 테이블 상면 + 여유(1cm) + (palm 원점 − 손 최하단)

무엇을 재는가
-------------
파지 자세(grip)와 개방 자세(open) 양쪽에서, palm 원점 대비 **모든 손 링크**의
최저 z 를 잰다. 손가락이 굽으면 최하단이 바뀌므로 두 자세를 다 봐야 한다.
★손끝(tip)만 보면 안 된다 — 굽힌 손에서는 중간마디가 더 아래일 수 있다.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--settle", type=int, default=90)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

from openarm.agnostic.tasks.grasp_s2r.grasp_s2r_env import GraspS2REnv  # noqa: E402
from openarm.agnostic.tasks.grasp_s2r.grasp_s2r_env_cfg import (  # noqa: E402
    GraspS2RTesolloRightEnvCfg,
)


def main() -> None:
    cfg = GraspS2RTesolloRightEnvCfg()
    cfg.scene.num_envs = 2
    cfg.object_bank = "single_cup"
    cfg.enable_events = False
    cfg.enable_adr = False
    cfg.enable_self_collisions = False
    cfg.episode_length_s = 10_000.0
    env = GraspS2REnv(cfg, render_mode=None)
    u = env.unwrapped
    dev = u.device
    p = u.profile

    # 손에 속한 **모든 body** 를 이름으로 모은다(팁만 보면 굽힌 손에서 틀린다).
    names = u.robot.data.body_names
    side = p.hand_joint_names[0].split("_hj_")[0]        # 예: "r"
    hand_ids = [i for i, nm in enumerate(names)
                if nm.startswith(f"{side}_hl_") and "palm" not in nm]
    print(f"[FLOOR] 손 링크 {len(hand_ids)}개 (palm 제외)", flush=True)

    act = torch.zeros(u.num_envs, u.cfg.action_space, device=dev)

    def measure(tag, a):
        for _ in range(args.settle):
            env.step(a)
        palm_z = u.robot.data.body_pos_w[:, u.palm_idx, 2]
        hz = u.robot.data.body_pos_w[:, hand_ids, 2]
        low = hz.min(dim=1)
        off = (palm_z - low.values).max()
        j = hand_ids[int(low.indices[0])]
        print(f"[FLOOR] {tag}: palm 원점 − 손최하단 = **{float(off)*100:.2f} cm** "
              f"(최하단 링크 {names[j]})", flush=True)
        return float(off)

    off_open = measure("개방 자세(a_hand=-1)", act)
    grip = act.clone()
    grip[:, 6:] = 1.0                      # 손 채널 전부 폐쇄
    off_grip = measure("파지 자세(a_hand=+1)", grip)

    worst = max(off_open, off_grip)
    tbl = float(cfg.table_surface_z)
    print("\n" + "=" * 66, flush=True)
    print(f"  테이블 상면            {tbl:.4f} m", flush=True)
    print(f"  palm−손최하단 최대     {worst:.4f} m", flush=True)
    for margin in (0.005, 0.010, 0.020):
        print(f"  여유 {margin*100:>4.1f} cm 확보 → palm_box_min z = "
              f"**{tbl + margin + worst:.4f}**", flush=True)
    print(f"  ※ 참고: 학습 실측 palm 최저 E1 0.2915 · E2 0.2769", flush=True)
    print("=" * 66, flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

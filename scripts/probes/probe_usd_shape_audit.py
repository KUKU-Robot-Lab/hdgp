"""로봇 USD 의 **shape 회계**를 감사한다 — 링크별 PhysX 충돌 shape 수 + 좌우 대칭.

## 왜 필요한가

PhysX 에서 shape = 충돌 도형 한 조각이다. `convex_decomposition` 은 메시 하나를
볼록 덩어리 여러 개로 쪼개고, 그 하나하나가 shape 다(대부분 링크가 16개인 것은
분해 상한에 걸려서다).

IsaacLab 의 `randomize_rigid_body_material` 은 "링크마다 다른 마찰"을 넣으려고
**링크별 shape 수를 세서 인덱스를 만든다.** 그때 두 숫자를 대조한다:

    root_physx_view.max_shapes          # 아티큘레이션 전체 shape 수
    Σ create_rigid_body_view(link).max_shapes   # 링크에 귀속된 shape 수

둘이 다르면 term 생성이 `ValueError` 로 죽는데, ★그 예외를 EventManager 콜백이
**삼켜서** term 이 클래스인 채 남고 첫 리셋에서야
`randomize_rigid_body_material.__init__() got an unexpected keyword argument
'asset_cfg'` 라는 **전혀 무관해 보이는 TypeError** 로 터진다. 원인이 안 보이는
실패라 이 감사가 있다.

★좌우 대칭도 같이 본다. 미러 링크의 분해 hull 수가 다르면 좌우 물리가 달라지고,
  실제로 09.02 RH56F1 에서 `l_al_3=14` vs `r_al_3=15` 가 나왔다.

## 사용

    cd hdgp && ~/rl_ws/IsaacLab/isaaclab.sh -p scripts/probes/probe_usd_shape_audit.py \
        --task open-rh_r_grasp_ua-play-lstm

합격 조건 2가지 — 자산을 고친 뒤 이 둘이 모두 OK 여야 한다:
  1. 전체 = 링크별 합
  2. 비대칭 0건

## 09.02 RH56F1 실측 (미해결)

    링크 76개 · 전체 460 · 링크별 합 459 · 차이 1
    ★FAIL 회계 일치
    ★FAIL 좌우 대칭 — l_al_3=14 vs r_al_3=15 (차 +1)
    분해 상한(16)에 안 걸린 링크: l_al_3=14 · r_al_3=15

★두 실패가 같은 링크를 가리킨다. 팔꿈치 링크 `al_3` 만 분해가 상한(16) 아래에서
  수렴하는데, 좌우 미러 메시가 **다른 hull 수**로 쪼개졌다. 그 1개 차이가 회계
  불일치 1개와 정확히 같다. 나머지 링크는 전부 16(포화) 또는 1(단순 도형)이라
  좌우가 자동으로 같다.
  → `tools/build_usd.py` 쪽 처방 후보: (a) 좌우가 **같은 cooked 콜라이더**를 쓰고
    미러는 인스턴스 변환으로만 주기 (b) `al_3` 의 maxConvexHulls 를 양쪽이 포화하도록
    낮추기 (c) 그 링크만 convexHull 로.
  shape 0 인 링크 4개(body_root · head_cam_view · {l,r}_hl_palm_1)는 좌우 대칭이고
  매니페스트상 의도된 프레임/시각 전용이라 원인이 아니다.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="open-rh_r_grasp_ua-play-lstm")
parser.add_argument("--num_envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
_app = AppLauncher(args_cli).app

import dataclasses  # noqa: E402

import gymnasium as gym  # noqa: E402

import openarm.agnostic.tasks.grasp_ua.config  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from openarm.agnostic.tasks.grasp_ua import robot_profiles as _rp  # noqa: E402


def _relax(cfg) -> None:
    """감사 **전용** 완화 — 자산을 보려는데 태스크 캘리브레이션이 먼저 막지 않도록.

    ★프로필 파일은 안 건드린다(런타임 딕셔너리 항목만 교체). 감사 대상은 자산이고
      홈/워크스페이스는 태스크 값이라, 둘을 엮으면 "자산을 못 보는" 상태가 된다.
    ★`enable_events=False` 도 같은 이유다 — 재질 랜덤화가 바로 이 shape 불일치로
      죽으므로 켜 두면 감사 자체가 못 돈다(닭-달걀).
    """
    cfg.enable_events = False
    name = cfg.profile_name
    _rp.PROFILES[name] = dataclasses.replace(
        _rp.PROFILES[name],
        palm_box_min=(-2.0, -2.0, -1.0), palm_box_max=(2.0, 2.0, 2.0),
        palm_rot_half_deg=179.0, palm_rot_center_deg=(0.0, 0.0, 0.0),
    )


def main() -> int:
    cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    cfg.scene.num_envs = args_cli.num_envs
    _relax(cfg)
    env = gym.make(args_cli.task, cfg=cfg)
    robot = env.unwrapped.robot
    view = robot.root_physx_view

    per = []
    for link_path in view.link_paths[0]:
        lv = robot._physics_sim_view.create_rigid_body_view(link_path)
        per.append((link_path.split("/")[-1], int(lv.max_shapes)))
    counts = dict(per)
    total, summed = int(view.max_shapes), sum(c for _, c in per)

    print(f"\n[shape-audit] 자산 = {cfg.robot_cfg.spawn.usd_path}", flush=True)
    print(f"[shape-audit] 링크 {len(per)}개 · 전체 {total} · 링크별 합 {summed} "
          f"· 차이 {total - summed}")
    zero = [n for n, c in per if c == 0]
    if zero:
        print(f"[shape-audit] shape 0 인 링크({len(zero)}): {', '.join(zero)}")

    asym = []
    for name, c in per:
        if not name.startswith("l_"):
            continue
        mate = "r_" + name[2:]
        if mate not in counts:
            asym.append(f"{name}={c} → 짝 {mate} 없음")
        elif counts[mate] != c:
            asym.append(f"{name}={c} vs {mate}={counts[mate]} (차 {counts[mate] - c:+d})")

    ok_total = total == summed
    ok_sym = not asym
    print(f"[shape-audit] {'OK   ' if ok_total else '★FAIL'} 회계 일치", flush=True)
    print(f"[shape-audit] {'OK   ' if ok_sym else '★FAIL'} 좌우 대칭"
          + ("" if ok_sym else " — " + " · ".join(asym)))
    _odd = [f"{n}={c}" for n, c in per if c not in (0, 1, 16)]
    print("[shape-audit] 분해 상한(16)에 안 걸린 링크: "
          + (" · ".join(_odd) if _odd else "(없음)"), flush=True)
    env.close()
    return 0 if (ok_total and ok_sym) else 1


if __name__ == "__main__":
    import traceback
    try:
        _rc = main()
    except BaseException:
        traceback.print_exc()
        _rc = 3
    _app.close()
    raise SystemExit(_rc)

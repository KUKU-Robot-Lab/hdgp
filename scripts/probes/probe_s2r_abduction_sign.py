#!/usr/bin/env python3
"""고정 관절의 가동 방향 실측 — grasp_s2r 20관절 풀 제어 준비.

배경 (08.31)
------------
손 제어를 **20관절 풀 제어**로 재정의하기로 했다(사용자 확정). 그런데 프로필
자세표가 7 관절을 `open_pose == grip_pose` 로 고정해 두어 가동범위가 없다:

    thumb_1 · thumb_2 · index_1 · middle_1 · ring_1 · pinky_1 · pinky_2

이 중 외전축의 **부호를 추측으로 넣으면 안 된다** — 엄지 대향축을 반대로 잡으면
엄지가 손가락 반대편으로 벌어지고, 그건 학습 지표로는 안 보이는 조용한 오류다
(08.30 per_finger 슬롯맵이 엄지 `_2` 를 누락해 R 계열 전체가 무효가 된 전례).

무엇을 재는가
-------------
각 대상 관절을 URDF 하한/상한으로 **하나씩** 보내고(나머지는 open 자세 고정),
palm 프레임에서 손끝 위치를 읽어 두 지표를 뽑는다:

  1. `d_thumb`  — 엄지 팁과 4지 팁 중심의 거리. **작을수록 대향(마주봄)**.
  2. `spread`   — 4지 팁들의 상호 평균거리. **작을수록 모임(gather)**.

그래서 각 관절에 대해 "어느 극단이 파지 자세(grip)인가"가 부호와 함께 나온다:
  · 엄지 외전(`thumb_1`) → `d_thumb` 를 줄이는 쪽이 grip
  · 4지 외전(`index_1`·`middle_1`·`ring_1`·`pinky_2`) → `spread` 를 줄이는 쪽이 grip
  · 굴곡축(`pinky_1`) → 팁이 손바닥으로 다가가는 쪽(= `d_palm` 감소)이 grip

사용
----
  RUN_LABEL=probe_abd python scripts/probes/probe_s2r_abduction_sign.py
  # 옵션: --settle 60 (극단 도달까지 스텝 수) --joints thumb_1,index_1
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--settle", type=int, default=90,
                    help="각 극단으로 보낸 뒤 정착까지 스텝 수")
parser.add_argument("--joints", default="",
                    help="쉼표 목록으로 대상 한정(기본: 자세표가 고정한 7 관절 전부)")
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
    cfg.scene.num_envs = 1
    cfg.object_bank = "single_cup"
    cfg.enable_events = False
    cfg.enable_adr = False
    # ★학습과 동일 조건 — 자기충돌 ON 이면 외전이 이웃 손가락에 막혀 "안 움직인다"가
    #   관절 특성으로 오독된다(08.29 다물체 폭발의 그 설정).
    cfg.enable_self_collisions = False
    cfg.episode_length_s = 10_000.0      # 프로브 중 리셋 오염 방지(관례)
    env = GraspS2REnv(cfg, render_mode=None)
    u = env.unwrapped
    robot = u.robot
    p = u.profile
    dev = u.device

    # ---- 대상 관절: 자세표가 open==grip 으로 고정한 것들 --------------------------
    names = list(p.hand_joint_names)
    op = torch.tensor(p.hand_open_pose, device=dev, dtype=torch.float32)
    gp = torch.tensor(p.hand_grip_pose, device=dev, dtype=torch.float32)
    fixed = [names[i] for i in range(len(names)) if float(op[i]) == float(gp[i])]
    if args.joints:
        want = {s.strip() for s in args.joints.split(",") if s.strip()}
        fixed = [n for n in fixed if n.rsplit("_", 2)[-2] + "_" + n.rsplit("_", 1)[-1] in want
                 or n in want or n.endswith(tuple(want))]
    print(f"[probe] 자세표 고정 관절 {len(fixed)}개: {[n.split('hj_')[-1] for n in fixed]}",
          flush=True)

    # 관절 인덱스와 한계
    j_idx = {n: robot.find_joints(n)[0][0] for n in names}
    lim = robot.data.joint_limits[0]          # (J, 2)
    # ★sim(USD) 한계를 먼저 찍는다 — URDF 와 다르면 "관절이 안 움직인다"의 진짜 원인이다
    #   (자산 드리프트 계열). 게인(k=5.0·effort 1.5)만으로는 무부하 미도달을 설명 못 한다.
    print("\n[probe] sim 관절한계 (USD 실측):", flush=True)
    for n in names:
        ji = j_idx[n]
        print(f"    {n.split('hj_')[-1]:<10} [{float(lim[ji, 0]):+.4f}, "
              f"{float(lim[ji, 1]):+.4f}]  범위 {float(lim[ji, 1] - lim[ji, 0]):.4f}",
              flush=True)

    # ★★손 관절 기입은 **`_syn_ids`** 로 한다 — `_hand_ids_t` 는 정규식(joint-major)
    #   순서라 프로필(finger-major) 순서 벡터를 그대로 넣으면 다른 관절로 간다.
    #   env 본체는 `_syn_ids = [jn.index(nm) for nm in hand_joint_names]` 로 이름
    #   해석해 정합을 지킨다(08.31 probe 오배선 자책 — 저장소 함정 "fabric 관절순서" 계열).
    hand_w = torch.tensor(u._syn_ids, device=dev, dtype=torch.long)

    # 손끝·palm 바디 인덱스 (env 가 이미 만들어 둔 것을 쓴다 — 이름 리터럴 금지)
    tip_ids = u._tip_ids_t                     # (5,) thumb..pinky 순
    palm_id = u.palm_idx
    thumb_k = 0                                # profile 의 손가락 순서 = 엄지 우선

    def _measure() -> tuple[float, float, float]:
        """(d_thumb, spread, d_palm_pinky) — palm 기준 기하."""
        bp = robot.data.body_pos_w[0]
        palm = bp[palm_id]
        tips = bp[tip_ids] - palm                       # (5,3) palm 기준
        th = tips[thumb_k]
        others = torch.cat([tips[:thumb_k], tips[thumb_k + 1:]], dim=0)   # (4,3)
        d_thumb = float((th - others.mean(dim=0)).norm())
        # 4지 상호 평균거리
        dif = others.unsqueeze(0) - others.unsqueeze(1)
        n4 = others.shape[0]
        spread = float(dif.norm(dim=-1).sum() / (n4 * (n4 - 1)))
        d_palm_pinky = float(others[-1].norm())
        return d_thumb, spread, d_palm_pinky

    def _kine(target_q: torch.Tensor) -> torch.Tensor:
        """관절을 **직접 기입**하고 한 스텝만 굴려 바디 자세를 갱신한다.

        ★★동역학 정착으로 재면 안 된다 — 손 PD 는 `stiffness 5.0 · effort_limit_sim 1.5`
          라 유지 가능한 정적 오차가 0.30 rad 뿐이고, 실측상 open 자세조차 런마다
          207 vs 217 mm 로 흔들린다. 기하 질문은 기하로 답한다(저장소 규약:
          "도달성 판정은 동역학 정착이 아니라 제약 IK/기하").
        """
        full = robot.data.joint_pos[0].clone()
        full[hand_w] = target_q
        robot.write_joint_state_to_sim(
            full.unsqueeze(0), torch.zeros_like(full).unsqueeze(0))
        u.sim.step(render=False)
        robot.update(u.sim.get_physics_dt())
        return robot.data.joint_pos[0, hand_w].clone()

    def _hold(target_q: torch.Tensor, steps: int) -> torch.Tensor:
        """손 관절을 목표로 보내고 정착시킨다. **달성 관절값을 반환**한다.

        ★추종 검증 없이 기하만 읽으면 "관절이 안 움직였다"를 "그 축은 무의미하다"로
          오독한다 — 첫 판이 정확히 그렇게 실패했다(pinky_1 60° 지령에 손끝 0.4mm).
        """
        for _ in range(steps):
            robot.set_joint_position_target(
                target_q.unsqueeze(0), joint_ids=hand_w)
            robot.write_data_to_sim()
            u.sim.step(render=False)
            robot.update(u.sim.get_physics_dt())
        return robot.data.joint_pos[0, hand_w].clone()

    # 기준선: open 자세
    base = op.clone()
    _kine(base)
    b_th, b_sp, b_pp = _measure()
    print(f"[probe] 기준선(open) d_thumb={b_th * 1000:.1f}mm "
          f"spread={b_sp * 1000:.1f}mm d_palm_pinky={b_pp * 1000:.1f}mm", flush=True)

    print("\n관절            극단      d_thumb    spread   d_palm    판정", flush=True)
    print("-" * 74, flush=True)
    rec: dict[str, str] = {}
    for nm in fixed:
        ji = j_idx[nm]
        lo, hi = float(lim[ji, 0]), float(lim[ji, 1])
        out, track = {}, {}
        for tag, val in (("lower", lo), ("upper", hi)):
            _kine(base)                        # ★매번 기준선으로 복귀 — 순차 드리프트 차단
            q = base.clone()
            q[names.index(nm)] = val
            ach = _kine(q)
            got = float(ach[names.index(nm)])
            track[tag] = abs(got - val)
            out[tag] = _measure()
            d_th, sp, dp = out[tag]
            flag = "" if track[tag] < 0.05 else f"  ⚠추종실패 실제={got:+.3f}"
            print(f"  {nm.split('hj_')[-1]:<12} {tag:<7} {val:+.3f} "
                  f"{d_th * 1000:8.1f} {sp * 1000:8.1f} {dp * 1000:8.1f}{flag}", flush=True)
        if max(track.values()) >= 0.05:
            print("      → 판정 보류: 지령을 못 따라갔다(고정/막힘/게인 부족)", flush=True)
            rec[nm.split("hj_")[-1]] = "판정 보류 — 추종 실패"
            continue
        # 판정 — 어느 극단이 "모으는" 쪽인가
        short = nm.split("hj_")[-1]
        if short.startswith("thumb"):
            key, i = "d_thumb", 0
        elif short == "pinky_1":
            key, i = "d_palm", 2
        else:
            key, i = "spread", 1
        lo_v, hi_v = out["lower"][i], out["upper"][i]
        grip_tag = "lower" if lo_v < hi_v else "upper"
        grip_val = lo if grip_tag == "lower" else hi
        delta = abs(lo_v - hi_v) * 1000.0
        rec[short] = f"open 0.0 → grip {grip_val:+.3f}  ({key} {delta:.1f}mm 차)"
        print(f"      → grip = {grip_tag} ({grip_val:+.3f}) · {key} 차 {delta:.1f}mm",
              flush=True)

    print("\n=== 제안 자세표 (open → grip) ===", flush=True)
    for k, v in rec.items():
        print(f"  {k:<10} {v}", flush=True)
    print("★차이가 1mm 미만이면 그 축은 손끝 기하에 무의미 — 고정 유지를 권한다.",
          flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

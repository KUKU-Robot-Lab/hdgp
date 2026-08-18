#!/usr/bin/env python3
"""[both/pour_v1] 양손 warm state 공존 게이트 — zero-action.

무엇을 재는가
-------------
`both/pour_v1` 은 **좌/우 grasp_v1 에서 각각 수집한 warm state 를 동시에 적용**해
"양손이 각자 컵을 쥔 상태"로 에피소드를 시작한다. 그런데 두 뱅크는 서로를 모른다 —
좌 warm 은 오른팔이 REST 인 씬에서, 우 warm 은 왼팔이 REST 인 씬에서 수집된다.
즉 **두 자세의 조합은 한 번도 함께 검증된 적이 없다.**

이 probe 는 정책을 끄고(action=0) N 스텝 굴려, 학습을 쏟기 전에 다음을 확정한다.

  1. 우컵(source) 유지 — 파지가 풀리지 않는가
  2. 좌컵(receiver) 유지 — 물리 파지로 바뀐 컵을 놓치지 않는가
  3. source 비드 유지 — 수집 시 채운 비드가 리셋 직후 쏟아지지 않는가
  4. 양팔 최소거리 — 두 팔이 서로 파고들지 않는가
  5. 손 토크 포화율 — 게인을 grasp_v1 값(k=5)으로 내린 뒤에도 파지가 성립하는가

★ 4번 주의: 로봇 articulation 은 `enabled_self_collisions=False` 다. 즉 팔끼리
  **물리 접촉이 아예 생성되지 않는다** → 접촉력으로는 간섭을 볼 수 없다. 그래서
  좌/우 body 위치의 **최소 거리**를 대리 지표로 쓴다. 값이 0 에 가까우면 물리적으로는
  겹쳐 있어도 조용히 통과하고 있다는 뜻이므로, 낮으면 실패로 본다.

게이트 기준 (미달이면 학습 금지)
--------------------------------
  우컵 유지 ≥ 0.90 · 좌컵 유지 ≥ 0.90 · 비드 유지 ≥ 0.90
  양팔 최소거리 ≥ 0.02 m · 손 토크 포화율 < 0.50

사용
----
  isaaclab.sh -p scripts/probes/probe_bimanual_warm_coexist.py \\
      --num_envs 64 --steps 200

  # 뱅크 경로를 직접 지정 (기본은 cfg 값)
  isaaclab.sh -p scripts/probes/probe_bimanual_warm_coexist.py --num_envs 64 \\
      "env.warm_state_paths=['/abs/right.hdf5']" \\
      "env.left_warm_state_paths=['/abs/left.hdf5']"

  # 왼팔 고정 단계(1단계 학습 설정)로 재기
  isaaclab.sh -p scripts/probes/probe_bimanual_warm_coexist.py --num_envs 64 \\
      env.receiver_control_mode=frozen
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--task", type=str, default="open-tesol_b_pour_v1-lstm")
parser.add_argument(
    "--allow_no_warm",
    action="store_true",
    help="warm 뱅크가 없어도 진행(기본: 거부). 뱅크 없이 재면 게이트 의미가 없다 — "
         "FK 고정배치 degrade 경로를 재는 것일 뿐이다.",
)
parser.add_argument("--gate_cup_retain", type=float, default=0.90)
parser.add_argument("--gate_bead_retain", type=float, default=0.90)
# zero-action 으로 steps 를 완주한 env 비율. 팔이 자세를 유지하는지 본다.
parser.add_argument("--gate_survive", type=float, default=0.50)
parser.add_argument("--gate_arm_gap_m", type=float, default=0.02)
parser.add_argument("--gate_sat_frac", type=float, default=0.50)
parser.add_argument(
    "--gap_all_bodies",
    action="store_true",
    help="양팔 최소거리를 **모든** 팔 body 로 잰다(기본: 원위부만). 기본이 원위부인 이유는 "
         "어깨 마운트(al_0~2)가 몸통에서 구조적으로 붙어 있어 min 을 지배하고, 그러면 "
         "정작 보고 싶은 손 주변 간섭이 가려지기 때문이다.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys

sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym
import torch

import openarm.tesollo  # noqa: F401  (gym 등록)
from isaaclab_tasks.utils.hydra import hydra_task_config


def _effort_limits(robot) -> torch.Tensor | None:
    """관절별 effort limit (num_joints,). 접근 경로가 버전마다 달라 순차 시도."""
    for getter in (
        lambda: robot.root_physx_view.get_dof_max_forces()[0],
        lambda: robot.data.joint_effort_limits[0],
        lambda: robot.data.default_joint_limits.new_tensor([]),  # 실패 유도
    ):
        try:
            v = getter()
            if v is not None and v.numel() == robot.num_joints:
                return v.to(robot.device).abs()
        except Exception:
            continue
    return None


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    dev = uenv.device
    n = uenv.num_envs

    # ---- 사전 조건: 좌/우 warm 뱅크가 실제로 붙었는가 ----------------------
    env.reset()
    right_n = int(getattr(uenv, "_warmstart_cache_count", 0))
    left_n = int(getattr(uenv, "_left_warm_count", 0))
    bead_restored = getattr(uenv, "_warmstart_bead_state", None) is not None
    print(
        f"\n[probe] warm 뱅크: right={right_n} states / left={left_n} states / "
        f"비드복원={'예' if bead_restored else '아니오(hold-end 소환)'}",
        flush=True,
    )
    if (right_n == 0 or left_n == 0) and not args_cli.allow_no_warm:
        print(
            "[probe][ABORT] 좌/우 warm 뱅크가 모두 필요하다. 뱅크 없이 재면 FK 고정배치\n"
            "  degrade 경로를 재는 것이라 공존 게이트가 성립하지 않는다.\n"
            "  먼저 수집:  collect_grasp_v1_warm_states.py --robot tesollo_right --with_beads\n"
            "             collect_grasp_v1_warm_states.py --robot tesollo_left\n"
            "  (의도적으로 뱅크 없이 보려면 --allow_no_warm)",
            flush=True,
        )
        env.close()
        return

    # ---- body 인덱스: 좌/우 팔+손 (자기충돌 대리 지표용) --------------------
    names = list(uenv.robot.data.body_names)

    def _arm_bodies(prefix_arm: str, prefix_hand: str) -> list[int]:
        out = []
        for i, b in enumerate(names):
            if b.startswith(prefix_hand):
                out.append(i)
            elif b.startswith(prefix_arm):
                # 근위부(어깨~상완) 제외: 몸통에 붙어 있어 좌우 거리가 항상 작다.
                if args_cli.gap_all_bodies or not any(
                    b.startswith(f"{prefix_arm}{k}") for k in ("0", "1", "2", "3")
                ):
                    out.append(i)
        return out

    r_idx = _arm_bodies("r_al_", "r_hl_")
    l_idx = _arm_bodies("l_al_", "l_hl_")
    scope = "전체" if args_cli.gap_all_bodies else "원위부(팔 4~7 + 손)"
    print(
        f"[probe] 간격 측정 body[{scope}]: 우 {len(r_idx)}개 / 좌 {len(l_idx)}개", flush=True
    )
    if not r_idx or not l_idx:
        print(f"[probe][WARN] 팔 body 를 못 찾았다 — 최소거리 측정 생략. names={names[:8]}…")

    # ---- 토크 포화 기준 ----------------------------------------------------
    eff = _effort_limits(uenv.robot)
    hand_dofs = list(uenv.hand_dof_indices) + list(uenv.left_hand_dof_indices)
    if eff is None:
        print("[probe][WARN] effort limit 을 못 읽었다 — 포화율 측정 생략.", flush=True)

    zero = torch.zeros(n, uenv.cfg.action_space, device=dev)

    # ---- 기준선은 **1 스텝 굴린 뒤** 캡처한다 ------------------------------
    # ★2026-08-18 버그: 리셋 직후 `root_pos_w` / `body_pos_w` 는 write_*_to_sim 이
    #   아직 반영되지 않은 stale 값이다(실측: reset 직후 object_pos 가 정확히 0.000).
    #   그 값으로 기준선을 잡으면 실제 grip 거리 변화가 60mm 인데 **212mm** 로 보여
    #   파지 실패로 오판한다(E0-3 우컵 유지 0.141 = 이 오판이었다).
    #   1 스텝 뒤 값은 안정적이다(실측: step1→step2 변화 1.4mm).
    # inference_mode 로 감싸면 fabric 의 in-place 갱신이 막힌다(RuntimeError).
    env.step(zero)
    # ★좌표계: `object_pos` 는 env-local(root_pos_w − env_origins), `palm_center_pos` 도
    #   env-local 이다. 구 코드는 `cup.data.root_pos_w`(world) 에서 env-local 을 빼서
    #   env 가 흩어진 거리만큼 오차가 섞였다(실측 7610mm). 반드시 같은 프레임을 쓸 것.
    r_grip0 = torch.norm(uenv.object_pos - uenv.palm_center_pos, dim=-1)
    _l_hand0 = uenv.robot.data.body_pos_w[:, uenv._left_hand_body_index]
    l_grip0 = torch.norm(uenv.left_target_cup.data.root_pos_w - _l_hand0, dim=-1)
    l_z0 = uenv.left_target_cup.data.root_pos_w[:, 2].clone()
    print(
        f"[probe] 기준선(1스텝 후)  우 grip {r_grip0.mean()*1000:.1f}mm  "
        f"좌 grip {l_grip0.mean()*1000:.1f}mm  좌컵 z {l_z0.mean():.3f}m",
        flush=True,
    )

    # ---- "첫 에피소드 동안"만 값을 얼린다 ---------------------------------
    #   env.step() 은 종료된 env 를 자동 리셋한다. 리셋 후 값을 섞으면 게이트가 무의미하다.
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    first_term = torch.full((n,), -1, dtype=torch.long, device=dev)
    f_bead = torch.zeros(n, device=dev)
    f_rdrift = torch.zeros(n, device=dev)
    f_ldrift = torch.zeros(n, device=dev)
    f_lz = torch.zeros(n, device=dev)
    f_gap = torch.full((n,), float("inf"), device=dev)
    f_pair_r = torch.zeros(n, dtype=torch.long, device=dev)
    f_pair_l = torch.zeros(n, dtype=torch.long, device=dev)
    f_sat = torch.zeros(n, device=dev)
    f_contacts = torch.zeros(n, device=dev)
    done_reasons: dict[str, int] = {}

    for t in range(args_cli.steps):
        # --- 스텝 전 상태를 살아있는 env 에 대해 갱신 (리셋 오염 방지) ---
        # 프레임 통일: 둘 다 env-local (위 기준선 주석 참조)
        r_drift = (
            torch.norm(uenv.object_pos - uenv.palm_center_pos, dim=-1) - r_grip0
        ).abs()
        l_pos = uenv.left_target_cup.data.root_pos_w
        l_hand = uenv.robot.data.body_pos_w[:, uenv._left_hand_body_index]
        # ★env 내부 `_left_cup_ref_dist/_ref_z` 는 리셋 시 stale body_pos_w 로 잡힐 수
        #   있다(env 주석도 그 가능성을 인정한다). 게이트는 probe 자체 기준선을 쓴다.
        l_drift = (torch.norm(l_pos - l_hand, dim=-1) - l_grip0).abs()
        l_zdrop = l_z0 - l_pos[:, 2]

        if r_idx and l_idx:
            rb = uenv.robot.data.body_pos_w[:, r_idx]            # (n,R,3)
            lb = uenv.robot.data.body_pos_w[:, l_idx]            # (n,L,3)
            D = torch.cdist(rb, lb)                              # (n,R,L)
            flat = D.flatten(1)
            gap, amin = flat.min(dim=1)                           # (n,), (n,)
            # 어느 쌍이 최소인지 기록 — 숫자만 보면 해석이 안 된다(어깨인지 손인지).
            pair_r = (amin // len(l_idx)).long()
            pair_l = (amin % len(l_idx)).long()
        else:
            gap = torch.full((n,), float("inf"), device=dev)
            pair_r = pair_l = torch.zeros(n, dtype=torch.long, device=dev)

        if eff is not None:
            tau = uenv.robot.data.applied_torque[:, hand_dofs].abs()
            lim = eff[hand_dofs].clamp(min=1e-6).unsqueeze(0)
            sat = (tau >= 0.98 * lim).float().mean(dim=-1)
        else:
            sat = torch.zeros(n, device=dev)

        m = alive
        f_bead = torch.where(m, uenv._bead_in_source_fraction, f_bead)
        f_rdrift = torch.where(m, r_drift, f_rdrift)
        f_ldrift = torch.where(m, l_drift, f_ldrift)
        f_lz = torch.where(m, l_zdrop, f_lz)
        _new_min = m & (gap < f_gap)
        f_gap = torch.where(_new_min, gap, f_gap)                  # 최악(최소) 유지
        f_pair_r = torch.where(_new_min, pair_r, f_pair_r)
        f_pair_l = torch.where(_new_min, pair_l, f_pair_l)
        f_sat = torch.where(m, torch.maximum(f_sat, sat), f_sat)   # 최악(최대) 유지
        f_contacts = torch.where(m, uenv.num_contacts_buf.float(), f_contacts)

        _, _, term, trunc, _ = env.step(zero)
        done = (term | trunc).to(dev).bool()
        newly = alive & done
        if bool(newly.any()):
            first_term[newly] = t
            alive = alive & ~newly
            for k, v in uenv.extras.items():
                if k.startswith("done/"):
                    done_reasons[k] = done_reasons.get(k, 0) + int(
                        round(float(v) * n)
                    )

        if (t + 1) % 50 == 0:
            print(
                f"[probe] step {t+1:4d}  alive={int(alive.sum()):3d}/{n}  "
                f"bead={f_bead.mean():.3f}  r_drift={f_rdrift.mean()*1000:.1f}mm  "
                f"l_drift={f_ldrift.mean()*1000:.1f}mm  gap_min={f_gap.min()*1000:.1f}mm  "
                f"sat_max={f_sat.max():.2f}  contacts={f_contacts.mean():.2f}",
                flush=True,
            )

    # ---- 판정 -------------------------------------------------------------
    # "유지" 정의: 리셋 기준선 대비 드리프트가 낙하 임계 미만이면 유지로 본다
    #   (임계는 env cfg 와 동일 값을 써서 판정 규약을 일치시킨다).
    d_thr = float(uenv.cfg.left_cup_drop_dist_m)
    z_thr = float(uenv.cfg.left_cup_drop_z_m)
    r_ok = (f_rdrift < d_thr)
    l_ok = (f_ldrift < d_thr) & (f_lz < z_thr)
    r_retain = float(r_ok.float().mean())
    l_retain = float(l_ok.float().mean())
    bead_retain = float(f_bead.mean())
    gap_min = float(f_gap.min())
    sat_max = float(f_sat.max())
    sat_mean = float(f_sat.mean())
    never_term = int((first_term < 0).sum())

    print("\n" + "=" * 72, flush=True)
    print(f"[probe] 양손 warm 공존 게이트  (num_envs={n}, steps={args_cli.steps})", flush=True)
    print("=" * 72, flush=True)
    print(
        f"  우컵(source) 파지유지   {r_retain:.3f}   "
        f"(grip 거리 변화 < {d_thr*100:.0f}cm)",
        flush=True,
    )
    print(
        f"  좌컵(receiver) 파지유지 {l_retain:.3f}   "
        f"(grip 거리 변화 < {d_thr*100:.0f}cm & 낙하 < {z_thr*100:.0f}cm)",
        flush=True,
    )
    print(f"  source 비드 유지율      {bead_retain:.3f}", flush=True)
    _w = int(torch.argmin(f_gap))
    _pn = (f"{names[r_idx[int(f_pair_r[_w])]]} ↔ {names[l_idx[int(f_pair_l[_w])]]}"
           if r_idx and l_idx else "—")
    print(f"  양팔 최소거리(최악)     {gap_min*1000:.1f} mm  [{_pn}, env{_w}]", flush=True)
    print(f"                          ※self-collision off → 거리 대리지표. 쌍 이름을 꼭 볼 것 —", flush=True)
    print(f"                            근위부가 min 을 지배하면 손 간섭이 가려진다.", flush=True)
    print(f"  손 토크 포화율          max={sat_max:.3f}  mean={sat_mean:.3f}", flush=True)
    print(f"  손 접촉 수(평균)        {float(f_contacts.mean()):.2f}", flush=True)
    survive = never_term / max(n, 1)
    print(f"  ★zero-action 생존율     {survive:.3f}  ({never_term}/{n}, {args_cli.steps} 스텝)", flush=True)
    print(
        "     (구 해설 삭제: 'zero-action 이라 out_x 로 밀린다' 는 fabric 상태 동기화 전\n"
        "      이야기다. 지금은 out_x=0 이므로 사망은 파지/접촉 쪽 문제로 읽어야 한다.)",
        flush=True,
    )
    if done_reasons:
        top = sorted(done_reasons.items(), key=lambda kv: -kv[1])[:6]
        print("  종료 사유(누적 추정)    " + ", ".join(f"{k.split('/')[-1]}={v}" for k, v in top), flush=True)

    # ★지표가 **의미를 갖는 조건**을 먼저 판정한다. 조건이 깨진 지표를 PASS/FAIL 로 찍으면
    #   오독을 부른다(예: warm 뱅크가 없으면 오른손은 애초에 컵을 안 쥐므로 "우컵 유지 0.125"
    #   는 실패가 아니라 **측정 불가**다).
    grasp_live = float(f_contacts.mean()) >= 1.0
    cup_valid = (right_n > 0 and left_n > 0) and grasp_live
    na_reason = []
    if right_n == 0 or left_n == 0:
        na_reason.append("warm 뱅크 부재(FK 고정배치 degrade)")
    if not grasp_live:
        na_reason.append(f"손 접촉 {float(f_contacts.mean()):.2f}개 — 파지 미성립")

    checks = [
        ("우컵 유지", cup_valid, r_retain >= args_cli.gate_cup_retain,
         f"{r_retain:.3f} ≥ {args_cli.gate_cup_retain}"),
        ("좌컵 유지", cup_valid, l_retain >= args_cli.gate_cup_retain,
         f"{l_retain:.3f} ≥ {args_cli.gate_cup_retain}"),
        ("비드 유지", right_n > 0, bead_retain >= args_cli.gate_bead_retain,
         f"{bead_retain:.3f} ≥ {args_cli.gate_bead_retain}"),
        # ★2026-08-18 추가. 기존 5지표는 전부 **리셋 직후 품질**만 봐서, 팔이 자세를
        #   유지하지 못해 전 env 가 out_x 로 죽는 조건을 5/5 로 통과시켰다(E1 학습 불가).
        #   에피소드가 끝까지 살아야 접근·붓기 보상이 발화할 기회가 생긴다.
        ("생존율", True, survive >= args_cli.gate_survive,
         f"{survive:.3f} ≥ {args_cli.gate_survive}"),
        ("양팔 간격", right_n > 0 and left_n > 0, gap_min >= args_cli.gate_arm_gap_m,
         f"{gap_min*1000:.1f}mm ≥ {args_cli.gate_arm_gap_m*1000:.0f}mm"),
        # 포화율은 접촉이 있어야 의미가 있다 — 허공에서 0.25 가 나와도 파지 능력과 무관하다.
        ("토크 포화", grasp_live, sat_max < args_cli.gate_sat_frac,
         f"{sat_max:.3f} < {args_cli.gate_sat_frac}"),
    ]
    print("-" * 72, flush=True)
    for label, valid, ok, detail in checks:
        tag = ("PASS" if ok else "FAIL") if valid else " N/A"
        print(f"  [{tag}] {label:10s} {detail}", flush=True)
    if na_reason:
        print(f"\n  ※ N/A 사유: {' / '.join(na_reason)}", flush=True)
    evaluated = [(l, ok) for l, valid, ok, _ in checks if valid]
    passed = bool(evaluated) and all(ok for _, ok in evaluated)
    print("=" * 72, flush=True)
    if not evaluated:
        print(
            "[probe] 판정: 측정 불가 — 게이트가 성립하지 않는다.\n"
            "  좌/우 warm 뱅크를 먼저 수집할 것 (이 실행은 코드 경로 점검용일 뿐이다).",
            flush=True,
        )
    elif len(evaluated) < len(checks):
        print(
            f"[probe] 판정: 부분 측정 ({len(evaluated)}/{len(checks)} 항목) — "
            f"{'통과' if passed else '미달'}. **게이트로 인정하지 말 것.**",
            flush=True,
        )
    else:
        print(f"[probe] 판정: {'PASS — 학습 진행 가능' if passed else 'FAIL — 학습 금지'}", flush=True)
    if not passed and len(evaluated) == len(checks):
        print(
            "\n[probe] 미달 항목별 대응:\n"
            "  · 좌/우컵 유지 미달 → 손 게인(현재 grasp_v1 값 5.0/2.0)이 파지를 못 버티는 경우다.\n"
            "      먼저 비드 채운 수집(--with_beads)이 실제로 적용됐는지 확인(위 '비드복원' 줄).\n"
            "  · 비드 유지 미달   → 리셋 직후 쏟아진다. 컵 자세(warm cup_quat)와 비드 소환 offset 확인.\n"
            "  · 양팔 간격 미달   → 좌/우 독립 샘플링 가정이 깨졌다. 검증된 페어만 쓰도록\n"
            "      `_sample_left_warm()` 을 페어 제한으로 바꿔야 한다.\n"
            "  · 토크 포화        → k=5 에서도 포화면 effort limit 자체가 병목이다(URDF 7.5N·m).\n"
            "      게인을 더 낮추기보다 파지 기하(감쌈 깊이)를 봐야 한다.",
            flush=True,
        )
    env.close()


main()
simulation_app.close()

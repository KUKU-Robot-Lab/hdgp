#!/usr/bin/env python3
"""grasp_v1 외란 강건성 평가 — 물체별 파지 붕괴 임계를 측정한다.

배경(2026-08-16):
  grasp_v1 의 목표는 "컵을 인벨롭 그립으로 잡고, 형상·질량·마찰·외란이 바뀌어도 그 파지력을
  잃지 않는 것"이다. 그런데 지금까지 이걸 판정할 **평가 프로토콜이 없었다** — 학습 곡선
  (success_rate 등)만 봤고, 그 지표들은 "래치가 걸렸는가"에 가까워 감쌈이나 외란 내성을
  재지 않는다. pour 통과율을 대리 게이트로 쓰다가 하네스 버그로 무효 판정을 반복했다.

무엇을 재는가:
  각 에피소드에서 정책이 파지·리프트한 뒤(래치 성립), **외란 배율을 단계적으로 올리며**
  파지가 언제 깨지는지를 물체별로 측정한다. 학습 시 ADR 만렙(wrench 15 / rot 12)을 1.0 배로
  두고 0 → max_scale 까지 계단식으로 올린다.

  붕괴 판정(하나라도 만족):
    · 컵 낙하(object_pos.z < obj_fallen_z) 또는 워크스페이스 이탈
    · 컵 기울기 > cup_tipping_max_deg
    · 접촉 손가락 수 < min_grip_fingers (파지 상실)

  물체별로 기록:
    break_scale  : 붕괴 시점의 외란 배율(끝까지 버티면 max_scale, 클수록 강건)
    wrap_at_break: 붕괴 직전 감쌈 깊이
    survived     : 최대 배율까지 버틴 비율

사용:
  IsaacLab/isaaclab.sh -p scripts/eval_grasp/eval_grasp_robustness.py \\
      --checkpoint log/rl_games/.../nn/<ckpt>.pth --num_envs 512

주의:
  · 정책이 래치에 도달하지 못한 env 는 집계에서 제외한다(외란 내성이 아니라 파지 실패이므로
    별도 지표 grasp_fail_rate 로 보고). 섞으면 "강건해 보이는데 실은 안 잡은" 오판이 난다.
  · ADR 은 끄고(--disable_adr 동등) 외란 배율만 이 스크립트가 직접 제어한다 —
    학습 커리큘럼과 평가 난이도를 분리해야 런 간 비교가 성립한다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default="open-tesol_r_grasp_v1-play-lstm")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--settle_steps", type=int, default=260,
                    help="외란 시작 전 파지·리프트 성립을 기다리는 스텝")
parser.add_argument("--stage_steps", type=int, default=60, help="배율 단계당 관찰 스텝")
parser.add_argument("--stages", type=int, default=8, help="배율 단계 수")
parser.add_argument("--max_scale", type=float, default=2.0,
                    help="최대 외란 배율(1.0 = 학습 ADR 만렙과 동일)")
parser.add_argument("--hand_stiffness", type=float, default=0.0,
                    help="0 이면 학습 게인 유지. >0 이면 파지 성립 **후** 손 게인을 이 값으로 "
                         "교체하고 외란을 인가한다 — 기하는 학습 영역에서 만들고 힘 용량만 "
                         "바꿔 비교하기 위함(S2 능력 곡선).")
parser.add_argument("--hand_damping", type=float, default=-1.0,
                    help="<0 이면 감쇠비 보존(kd = 60·√(k/400)). S4 에서 재도출 예정.")
parser.add_argument("--mass_scale", type=float, default=1.0,
                    help="파지 성립 후 물체 질량 배율. ADR mass DR(0.5~4.0) 범위 실측용.")
parser.add_argument("--out", type=str, default="")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys  # noqa: E402

sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

import openarm.tesollo  # noqa: F401,E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    # 평가 난이도는 이 스크립트가 직접 제어한다 — ADR 커리큘럼은 끈다.
    env_cfg.enable_adr = False
    # 외란은 매 스텝 우리가 배율을 바꿔 넣으므로 env 자체 트리거 주기는 짧게.
    env_cfg.wrench_trigger_every = 15

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped

    clip_obs = agent_cfg["params"]["env"].get("clip_observations", float("inf"))
    clip_act = agent_cfg["params"]["env"].get("clip_actions", float("inf"))
    env = RlGamesVecEnvWrapper(env, uenv.device, clip_obs, clip_act)

    agent_cfg["params"]["config"]["env_info"] = env.get_env_info()
    runner = Runner()
    runner.load(agent_cfg)
    player = runner.create_player()
    player.restore(args_cli.checkpoint)
    player.reset()

    obs = env.reset()
    if player.is_rnn:
        _ = player.get_batch_size(obs, 1)
        player.init_rnn()

    n = uenv.num_envs
    dev = uenv.device
    n_obj = len(uenv._object_names)

    def step(scale: float):
        """외란 배율을 강제한 채 한 스텝. env 의 ADR 조회를 우회해 cfg 값을 직접 스케일."""
        uenv.cfg.wrench_max_accel = float(_BASE_WRENCH * scale)
        uenv.cfg.hold_rotation_perturb_max_accel = float(_BASE_ROT * scale)
        with torch.no_grad():
            act = player.get_action(player.obs_to_torch(step.obs), is_deterministic=True)
        step.obs, _, _, _ = env.step(act)

    _BASE_WRENCH = float(uenv.cfg.wrench_max_accel)
    _BASE_ROT = float(uenv.cfg.hold_rotation_perturb_max_accel)
    step.obs = obs

    # ---- 1단계: 외란 0 으로 파지·리프트 성립을 기다린다 ----
    for _ in range(args_cli.settle_steps):
        step(0.0)

    eligible = uenv.lift_ready_latched_buf.clone()          # 래치 성립 = 평가 대상
    grasp_fail = (~eligible).float().mean()
    print(f"[eval] 파지 성립 {int(eligible.sum())}/{n} "
          f"(grasp_fail_rate={float(grasp_fail):.3f})", flush=True)

    # ---- 1.5단계: 손 게인 / 물체 질량 교체 (S2 능력 곡선) ----
    # 기하는 학습 게인에서 만들고 그 뒤 힘 용량만 바꾼다. 처음부터 낮은 게인으로 돌리면
    # 파지 형상 자체가 달라져 "힘이 모자란 것"과 "잘못 잡은 것"이 섞인다.
    if args_cli.hand_stiffness > 0.0:
        k = float(args_cli.hand_stiffness)
        kd = float(args_cli.hand_damping) if args_cli.hand_damping >= 0.0 else 60.0 * (k / 400.0) ** 0.5
        hd = uenv.hand_dof_indices
        stiff = uenv.robot.data.joint_stiffness.clone()
        damp = uenv.robot.data.joint_damping.clone()
        stiff[:, hd] = k
        damp[:, hd] = kd
        uenv.robot.write_joint_stiffness_to_sim(stiff)
        uenv.robot.write_joint_damping_to_sim(damp)
        # ★write_joint_*_to_sim 은 actuator 모델을 갱신하지 않는다(IsaacLab articulation.py:624).
        #   장부를 안 맞추면 applied_torque 등 진단값이 옛 게인으로 계산된다.
        for _name, _a in uenv.robot.actuators.items():
            if _name.startswith("tesollo_hand"):
                _a.stiffness[:] = k
                _a.damping[:] = kd
        print(f"[eval] 손 게인 교체: stiffness={k:.2f} damping={kd:.2f}", flush=True)

    if args_cli.mass_scale != 1.0:
        masses = uenv.cup.root_physx_view.get_masses()
        uenv.cup.root_physx_view.set_masses(masses * float(args_cli.mass_scale),
                                            torch.arange(n))
        print(f"[eval] 물체 질량 ×{args_cli.mass_scale:.2f} "
              f"(평균 {float(masses.mean())*args_cli.mass_scale*1000:.0f} g)", flush=True)

    if args_cli.hand_stiffness > 0.0 or args_cli.mass_scale != 1.0:
        for _ in range(60):                                  # 새 물성에서 정착
            step(0.0)
        still = uenv.lift_ready_latched_buf & eligible
        print(f"[eval] 교체 정착 후 파지 유지 {int(still.sum())}/{int(eligible.sum())}",
              flush=True)

    broken = torch.zeros(n, dtype=torch.bool, device=dev)
    break_scale = torch.full((n,), float(args_cli.max_scale), device=dev)
    wrap_at_break = torch.zeros(n, device=dev)

    def _alive() -> torch.Tensor:
        """파지 유지 여부 — 낙하·이탈·전복·그립 상실 중 어느 것도 아니어야 한다."""
        p = uenv.object_pos
        out = (
            (p[:, 0] < uenv.cfg.obj_out_x_min) | (p[:, 0] > uenv.cfg.obj_out_x_max)
            | (p[:, 1] < uenv.cfg.obj_out_y_min) | (p[:, 1] > uenv.cfg.obj_out_y_max)
            | (p[:, 2] < uenv.cfg.obj_fallen_z)
        )
        grip = (
            uenv.binary_contact_buf
            | uenv.middle_binary_contact_buf
            | uenv.distal_binary_contact_buf
        ).sum(dim=-1)
        lost = grip < int(uenv.cfg.lift_start_min_grip_fingers)
        return ~(out | lost)

    # ---- 2단계: 배율 계단 상승 ----
    for s in range(1, args_cli.stages + 1):
        scale = args_cli.max_scale * s / args_cli.stages
        for _ in range(args_cli.stage_steps):
            step(scale)
            now_broken = (~_alive()) & eligible & (~broken)
            if now_broken.any():
                break_scale = torch.where(now_broken, torch.full_like(break_scale, scale),
                                          break_scale)
                wrap_at_break = torch.where(now_broken, uenv.wrap_frac_buf, wrap_at_break)
                broken |= now_broken
        n_alive = int((eligible & ~broken).sum())
        print(f"[eval] scale={scale:.2f} (wrench {_BASE_WRENCH*scale:.1f} / "
              f"rot {_BASE_ROT*scale:.1f}) 생존 {n_alive}/{int(eligible.sum())}", flush=True)

    # ---- 집계: 물체별 ----
    print("\n[eval] === 물체별 강건성 ===", flush=True)
    print(f"{'물체':22s} {'평가N':>6s} {'파지실패':>8s} {'break_scale':>12s} "
          f"{'끝까지생존':>10s} {'wrap@break':>11s}", flush=True)
    rows = []
    for i, name in enumerate(uenv._object_names):
        m_all = uenv.object_idx == i
        m = m_all & eligible
        cnt = int(m.sum())
        if cnt == 0:
            continue
        surv = float(((~broken) & m).float().sum() / cnt)
        bs = float(break_scale[m].mean())
        wb = float(wrap_at_break[m & broken].mean()) if int((m & broken).sum()) else float("nan")
        gf = float((~eligible & m_all).float().sum() / max(int(m_all.sum()), 1))
        rows.append(dict(object=name, n=cnt, grasp_fail=gf, break_scale=bs,
                         survived=surv, wrap_at_break=wb))
        print(f"{name:22s} {cnt:>6d} {gf:>8.3f} {bs:>12.3f} {surv:>10.3f} {wb:>11.4f}",
              flush=True)

    overall = dict(
        checkpoint=args_cli.checkpoint,
        base_wrench=_BASE_WRENCH, base_rot=_BASE_ROT, max_scale=args_cli.max_scale,
        grasp_fail_rate=float(grasp_fail),
        mean_break_scale=float(break_scale[eligible].mean()),
        survived_rate=float(((~broken) & eligible).float().sum() / max(int(eligible.sum()), 1)),
        per_object=rows,
    )
    print(f"\n[eval] 종합: 파지실패 {overall['grasp_fail_rate']:.3f} | "
          f"평균 break_scale {overall['mean_break_scale']:.3f} | "
          f"최대배율 생존 {overall['survived_rate']:.3f}", flush=True)

    if args_cli.out:
        Path(args_cli.out).write_text(json.dumps(overall, indent=2, ensure_ascii=False))
        print(f"[eval] saved → {args_cli.out}", flush=True)


main()
simulation_app.close()

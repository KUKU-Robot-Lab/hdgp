# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# SPDX-License-Identifier: Apache-2.0

"""Scripted deep-tilt feasibility probe (pour_v5).

목적 (analysis.md lstm_test1 §4 부트스트랩 설계 사전 검증):
  "안정된 warmstart 직립 grasp에서 출발해, env 기존 rim-pivot + Fabrics IK 경로로
   palm을 깊게(90~135°) 기울일 때 IK가 추종하고 grasp가 유지되는가?"

정책/체크포인트를 쓰지 않고, scripted action(tilt_toward 램프)만 env에 주입한다.
approach 단계를 격리하기 위해 tilt action gate를 강제로 열어(=1) IK·grasp 추종 능력만 본다.

판정 지표(매 스텝 출력):
  tilt_deg(actual)     : acos(_source_up_dot_world). 90~135°까지 오르면 IK 도달 OK.
  cup_rel_drift_deg    : grasp slip. < ~20° 유지면 컵이 손을 따라옴(grasp OK).
  palm_rot_err_deg     : IK 추종 오차. 작게 유지면 명령 palm 자세 도달.
  palm_clamp_viol_xy/z : rim-pivot palm 위치가 박스에 잘리는 양(>0이면 박스가 깊은 tilt를 제약).
  grasp_contacts       : 파지 접점 수(0으로 떨어지면 컵 놓침).
  bead_in_source/target: 텔레포트 아닌 연속 tilt에서 컵 내부 비드 거동.

실행 예:
  ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/probe_tilt_ramp.py \
      --task open-tesol_r_pour_v5-lstm --num_envs 16 --headless \
      --steps 400 --ramp_steps 60
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Scripted deep-tilt feasibility probe for pour_v5.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments.")
parser.add_argument("--task", type=str, default="open-tesol_r_pour_v5-lstm", help="Train task id (warmstart 캐시 보유).")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point", help="Agent cfg entry point (cfg 로드용, 학습 안 함).")
parser.add_argument("--seed", type=int, default=0, help="Seed.")
parser.add_argument("--steps", type=int, default=400, help="총 시뮬 스텝 수.")
parser.add_argument("--ramp_steps", type=int, default=60, help="hold 종료 후 tilt 0→max 램프 스텝 수.")
parser.add_argument("--tilt_max", type=float, default=1.0, help="tilt_toward 최대 명령(|action|, 보통 1.0).")
parser.add_argument("--print_every", type=int, default=10, help="몇 스텝마다 출력할지.")
parser.add_argument(
    "--no_force_gate",
    action="store_true",
    default=False,
    help="설정 시 tilt gate를 그대로 둔다(기본은 gate를 강제로 열어 approach 격리).",
)
parser.add_argument(
    "--freeze_grasp_hand",
    action="store_true",
    default=False,
    help="grasp 손 관절을 고정(손 컨트롤러 노이즈 제거, IK/팔 feasibility 격리).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os

import gymnasium as gym
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


def _force_local_openarm_path() -> str:
    """play.py와 동일: openarm 임포트를 로컬 hdgp/source/openarm으로 강제."""
    hdgp_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    openarm_src = os.path.join(hdgp_root, "source", "openarm")
    openarm_pkg = os.path.join(openarm_src, "openarm")
    if not os.path.isdir(openarm_pkg):
        raise RuntimeError(f"Local openarm package not found: {openarm_pkg}")
    if openarm_src in sys.path:
        sys.path.remove(openarm_src)
    sys.path.insert(0, openarm_src)
    return os.path.abspath(openarm_pkg)


_EXPECTED_OPENARM_DIR = _force_local_openarm_path()
import openarm  # noqa: E402

if not os.path.abspath(getattr(openarm, "__file__", "")).startswith(_EXPECTED_OPENARM_DIR + os.sep):
    raise RuntimeError(
        f"openarm resolved to unexpected location: {getattr(openarm, '__file__', None)} "
        f"(expected under {_EXPECTED_OPENARM_DIR})"
    )
import openarm.tasks  # noqa: F401,E402


# palm action 레이아웃: [x, y, z, spin, tilt_toward, tilt_ortho] + nullspace α
_TILT_TOWARD_IDX = 4


def _unwrap_reset(ret):
    return ret[0] if isinstance(ret, tuple) else ret


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    # ---- probe용 env_cfg 오버라이드 ----
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed

    # ADR/시각화 비활성 (클린 probe)
    for attr in (
        "enable_noise_adr",
        "enable_spill_adr",
        "enable_success_adr",
        "enable_bead_count_adr",
    ):
        if hasattr(env_cfg, attr):
            setattr(env_cfg, attr, False)
    if hasattr(env_cfg, "enable_visual_markers"):
        env_cfg.enable_visual_markers = False

    # tilt action gate 강제 개방: approach 단계를 격리하고 IK/grasp 추종만 본다.
    # (gate = clamp((far - mouth_xy)/den), 기본 far=0.25 → grasp 위치에선 gate≈0이라 tilt가 죽음)
    if not args_cli.no_force_gate:
        env_cfg.tilt_action_gate_xy_far = 100.0
        env_cfg.tilt_action_gate_xy_near = 0.0

    if args_cli.freeze_grasp_hand and hasattr(env_cfg, "freeze_grasp_hand_during_episode"):
        env_cfg.freeze_grasp_hand_during_episode = True

    # ---- env 생성 (rl-games 래퍼/체크포인트 없음, raw IsaacLab env) ----
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    e = env.unwrapped
    device = e.device
    n = e.num_envs
    num_actions = int(e.cfg.num_actions)
    hold = int(getattr(e.cfg, "episode_hold_steps", 120))

    print("=" * 96)
    print(f"[probe] task={args_cli.task} num_envs={n} num_actions={num_actions} "
          f"hold={hold} ramp={args_cli.ramp_steps} tilt_max={args_cli.tilt_max} "
          f"force_gate={not args_cli.no_force_gate} freeze_hand={args_cli.freeze_grasp_hand}")
    ws = int(getattr(e, "_warmstart_cache_count", 0))
    print(f"[probe] warmstart_cache_count={ws}")
    if ws <= 0:
        # 캐시가 비면 reset이 손 벌린 pregrasp(파지 안 됨)로 떨어져 tilt probe가 무의미.
        print("[probe] ABORT: warmstart 캐시가 비어 있어 grasped 시작 상태를 만들 수 없습니다.")
        print("        grasp 체크포인트 HDF5(5g_grasp_right_v7_2 warm bank) 경로/환경변수를 확인하세요.")
        env.close()
        return
    print("=" * 96)
    hdr = (f"{'step':>5} {'t_loc':>5} {'tilt_mean':>9} {'tilt_max':>8} "
           f"{'drift':>7} {'ik_err':>7} {'clmp_xy':>8} {'clmp_z':>7} "
           f"{'grasp':>6} {'rot_gate':>8} {'bead_src':>8} {'bead_tgt':>8} {'gate':>6}")
    print(hdr)

    # 롤아웃 전체에서 도달한 최대 tilt와 그 시점의 grasp/clamp 상태 추적
    best = {"tilt": -1.0, "drift": 0.0, "ik": 0.0, "clmp_xy": 0.0, "grasp": 0.0, "bead_tgt": 0.0}

    _unwrap_reset(env.reset())

    with torch.inference_mode():
        for step in range(args_cli.steps):
            # 각 env의 자기 episode step 기준으로 tilt 램프 (reset 시 자동 재시작)
            local_t = e.episode_length_buf.float()
            frac = ((local_t - hold) / max(args_cli.ramp_steps, 1)).clamp(0.0, 1.0)

            act = torch.zeros(n, num_actions, device=device)
            # tilt_toward: 학습 raw_action이 음수 방향으로 기울었으므로 음수 램프
            act[:, _TILT_TOWARD_IDX] = -float(args_cli.tilt_max) * frac

            env.step(act)

            # ---- 진단 지표 읽기 ----
            tilt_deg = torch.rad2deg(torch.acos(e._source_up_dot_world.clamp(-1.0, 1.0)))
            drift = e._cup_rel_drift_deg
            ik_err = e._palm_target_rot_error_deg
            clmp_xy = e._palm_clamp_viol_xy
            clmp_z = e._palm_clamp_viol_z
            grasp = e.binary_contact_buf.float().sum(dim=-1)  # 접점 수 (NUM_FINGERTIPS 중)
            rot_gate = e._internal_rot_gate
            bead_src = e._bead_in_source_fraction
            bead_tgt = e._bead_in_target_fraction
            tilt_gate = e._action_tilt_gate

            # 가장 깊게 기운 env 기준으로 best 갱신 (해당 env의 grasp/clamp 동시 기록)
            tmax, tidx = tilt_deg.max(dim=0)
            if tmax.item() > best["tilt"]:
                best.update(
                    tilt=tmax.item(),
                    drift=drift[tidx].item(),
                    ik=ik_err[tidx].item(),
                    clmp_xy=clmp_xy[tidx].item(),
                    grasp=grasp[tidx].item(),
                    bead_tgt=bead_tgt[tidx].item(),
                )

            if step % args_cli.print_every == 0 or step == args_cli.steps - 1:
                print(
                    f"{step:>5d} {local_t.mean().item():>5.0f} "
                    f"{tilt_deg.mean().item():>9.1f} {tmax.item():>8.1f} "
                    f"{drift.mean().item():>7.1f} {ik_err.mean().item():>7.1f} "
                    f"{clmp_xy.mean().item():>8.4f} {clmp_z.mean().item():>7.4f} "
                    f"{grasp.mean().item():>6.2f} {rot_gate.mean().item():>8.3f} "
                    f"{bead_src.mean().item():>8.3f} {bead_tgt.mean().item():>8.3f} "
                    f"{tilt_gate.mean().item():>6.2f}"
                )

    # ---- 판정 요약 ----
    print("=" * 96)
    print("[probe] === 판정 요약 (가장 깊게 기운 env 기준) ===")
    print(f"  max tilt 도달      : {best['tilt']:.1f}°   (목표: 90~135° 도달 시 IK 추종 OK)")
    print(f"  그 시점 grasp slip : {best['drift']:.1f}°   (< ~20°면 grasp 유지)")
    print(f"  그 시점 IK 오차    : {best['ik']:.1f}°")
    print(f"  그 시점 clamp_xy   : {best['clmp_xy']:.4f} m (>0이면 박스가 rim-pivot 해를 자름)")
    print(f"  그 시점 grasp 접점 : {best['grasp']:.2f}   (0이면 컵 놓침)")
    print(f"  그 시점 bead_tgt   : {best['bead_tgt']:.3f}")
    verdict = (
        "PASS (deep tilt 추종+grasp 유지 → 부트스트랩 설계 진행 가능)"
        if best["tilt"] >= 90.0 and best["drift"] < 20.0 and best["grasp"] >= 2.0
        else "FAIL/CHECK (deep tilt 미도달 또는 grasp 붕괴 → 부트스트랩 방향 재검토)"
    )
    print(f"  >>> {verdict}")
    print("=" * 96)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

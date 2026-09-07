#!/usr/bin/env python3
"""grasp_lift_fabric 성공 종료 상태 → pour_fabric warm 뱅크(HDF5) 수집기.

**설계 원칙** (pour_fabric/warm_bank.py 의 두 조건):
  1. 배포할 그 정책(grasp_lift_fabric 체크포인트)의 성공 상태에서 수집한다.
     구 collector(collect_grasp_v1_warm_states.py — 구 트랙 diffIK 규약)는 쓰지 않는다.
  2. 물리 상태만 저장한다(관절+컵 pose). slew 지령·prev_action 은 pour env 가
     리셋에서 측정량으로 재구성한다. 수집 물리 플래그는 meta 에 기록되어
     pour 로드 시 hard-fail 대조된다.

★비드는 수집하지 않는다 — pour env 가 리셋에서 검증된 배치(bead_offsets_in_cup)로
  스폰하고 hold(120스텝, 2s)가 정착을 담당한다. hold-정착이 부족하다고 실측되면
  그때 --with_beads 변형을 추가한다(pour_v1 방식).

★env 코드는 수정하지 않는다(grasp 트랙은 학습 중) — 성공 판정·상태 캡처를
  이 스크립트가 env 버퍼에서 직접 읽는다.

사용 (Isaac 환경):
    ./isaaclab.sh -p scripts/warm_states/collect_pour_fab_warm_states.py \\
        --pair bis --bank src --checkpoint <pth> [--count 2048] [--num_envs 256]
    --bank src → source(우) 파지 / --bank rcv → receiver(좌) 파지
    --latest   → log/rl_games/open-<short>/<side>/grasp-lift-fab/*/nn 최신 ep 체크포인트
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pair", default="bis")
parser.add_argument("--bank", choices=["src", "rcv"], required=True)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--latest", action="store_true",
                    help="log 에서 최신 grasp-lift-fab 체크포인트 자동 선택")
parser.add_argument("--count", type=int, default=2048)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--out", default=None)
parser.add_argument("--success_hold", type=int, default=10,
                    help="성공(goal+gate)이 이만큼 유지된 시점의 상태를 캡처")
parser.add_argument("--max_steps", type=int, default=20000)
parser.add_argument("--no_gravity", action="store_true",
                    help="(비권장) 중력 OFF 수집 — meta 에 기록되어 pour 기본과 안 맞는다")
parser.add_argument("--no_self_collisions", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import math  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

import openarm.tasks  # noqa: F401,E402
from openarm.agnostic.tasks.grasp_lift_fabric import config as _gcfg  # noqa: E402
from openarm.agnostic.tasks.grasp_lift_fabric import (  # noqa: E402
    grasp_lift_fabric_env_cfg as _gec,
)
from openarm.agnostic.tasks.pour_fabric import bimanual as _bm  # noqa: E402
from openarm.agnostic.tasks.pour_fabric.warm_bank import save_bank  # noqa: E402

_HDGP = Path(__file__).resolve().parents[2]


def _latest_checkpoint(profile) -> Path:
    side = "right" if profile.side == "r" else "left"
    root = _HDGP / "log/rl_games" / f"open-{profile.asset.short}" / side / "grasp-lift-fab"
    cands = sorted(root.glob("*/nn/*ep_*.pth"),
                   key=lambda q: q.stat().st_mtime, reverse=True)
    if not cands:
        raise SystemExit(f"체크포인트 없음: {root}/*/nn/*ep_*.pth")
    return cands[0]


def main() -> None:
    pair = _bm.get_pair(args.pair)
    profile = pair.source if args.bank == "src" else pair.receiver
    side = "r" if profile.side == "r" else "l"
    task = f"open-{profile.asset.short}_{side}_grasp_lift_fab-play"

    ckpt = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(profile)
    if not ckpt.is_file():
        raise SystemExit(f"체크포인트가 없다: {ckpt}")

    out = Path(args.out) if args.out else (
        _HDGP / "data" / f"pour_fab_warm_{pair.name}_{args.bank}.hdf5")
    if out.is_file():
        raise SystemExit(
            f"목적지가 이미 있다: {out}\n  옮기거나 지운 뒤 재실행할 것"
            " (구 뱅크를 새 수집으로 착각하는 사고 방지 — 2026-08-18 재발 방지).")

    # ---- env cfg: 물리 플래그를 pour 기본(ON/ON)과 맞춘다 -------------------------
    cfg_cls = getattr(_gcfg, f"GraspLiftFabric_{profile.name}_PLAY_Cfg")
    env_cfg = cfg_cls()
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.enable_gravity = not args.no_gravity
    env_cfg.enable_self_collisions = not args.no_self_collisions
    env_cfg.enable_adr = False
    _gec.resolve_cfg(env_cfg)
    print(f"[collect] task={task} profile={profile.name} ckpt={ckpt}\n"
          f"[collect] gravity={env_cfg.enable_gravity} "
          f"self_collisions={env_cfg.enable_self_collisions} "
          f"num_envs={env_cfg.scene.num_envs} → out={out}", flush=True)

    # ---- rl_games player (collect_warm_states.py 의 로딩 패턴) --------------------
    agent_yaml = (Path(_gcfg.__file__).parent / "agents" / "rl_games_ppo_cfg.yaml")
    agent_cfg = yaml.safe_load(agent_yaml.read_text())
    rl_device = agent_cfg["params"]["config"].get("device", "cuda:0")
    clip_obs = agent_cfg["params"].get("env", {}).get("clip_observations", math.inf)
    clip_act = agent_cfg["params"].get("env", {}).get("clip_actions", math.inf)

    env = gym.make(task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_act, None, True)
    vecenv.register("IsaacRlgWrapper",
                    lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                          "env_creator": lambda **kw: env})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(ckpt)
    agent_cfg["params"]["config"]["num_actors"] = uenv.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    player = runner.create_player()
    player.restore(str(ckpt))
    player.reset()

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = player.get_batch_size(obs, 1)
    if player.is_rnn:
        player.init_rnn()

    # ---- 캡처 루프 ---------------------------------------------------------------
    N = uenv.num_envs
    dev = uenv.device
    thr = float(env_cfg.contact_force_threshold)
    joint_idx = torch.cat([uenv._arm_t, uenv._hand_t])
    joint_names = tuple(uenv.robot.joint_names[j] for j in joint_idx.tolist())

    streak = torch.zeros(N, dtype=torch.long, device=dev)
    captured_ep = torch.zeros(N, dtype=torch.bool, device=dev)   # 에피소드당 1회
    rows_q, rows_qt, rows_cup, rows_nc = [], [], [], []
    steps = 0

    while len(rows_q) < args.count and steps < args.max_steps:
        with torch.inference_mode():
            obs = player.obs_to_torch(obs)
            actions = player.get_action(obs, is_deterministic=True)
            obs, _, dones, _ = env.step(actions)
            if player.is_rnn and player.states is not None and len(dones) > 0:
                for s in player.states:
                    s[:, dones, :] = 0.0
        steps += 1
        done_mask = dones.to(dev).bool() if torch.is_tensor(dones) else torch.zeros(N, dtype=torch.bool, device=dev)
        captured_ep &= ~done_mask
        streak[done_mask] = 0

        # ★grasp env 의 `_contact()` 반환 폭은 트랙 개편마다 늘었다
        #   (08.23: (tot, wrapped) → (tot, strict, mid, dist)). 첫 원소만 쓰되
        #   위치 언패킹을 하지 않는다 — 하면 트랙이 바뀔 때 조용히 죽는다.
        contact = uenv._contact()[0]
        gate = ((contact[:, uenv._grp_a] > thr).any(dim=-1)
                & (contact[:, uenv._grp_b] > thr).any(dim=-1))
        ok = uenv._goal_reached_now & gate
        streak = torch.where(ok, streak + 1, torch.zeros_like(streak))
        cap = (streak == int(args.success_hold)) & (~captured_ep)
        if bool(cap.any()):
            ids = cap.nonzero(as_tuple=False).squeeze(-1)
            q = uenv.robot.data.joint_pos[ids][:, joint_idx]
            # ★지령 목표 — 파지력 = kp·(target−q). 측정치만 저장하면 복원 시 파지가 풀린다.
            qt = uenv.robot.data.joint_pos_target[ids][:, joint_idx]
            cup_pos = uenv.object.data.root_pos_w[ids] - uenv.scene.env_origins[ids]
            cup_quat = uenv.object.data.root_quat_w[ids]
            nc = (contact[ids] > thr).sum(dim=-1)
            rows_q.append(q.cpu().numpy())
            rows_qt.append(qt.cpu().numpy())
            rows_cup.append(torch.cat([cup_pos, cup_quat], dim=1).cpu().numpy())
            rows_nc.append(nc.cpu().numpy())
            captured_ep |= cap
            total = sum(r.shape[0] for r in rows_q)
            print(f"[collect] step={steps} +{len(ids)} → {total}/{args.count}", flush=True)

    total = sum(r.shape[0] for r in rows_q) if rows_q else 0
    if total < args.count:
        raise SystemExit(
            f"[collect] ABORT: {steps} steps 에 {total}/{args.count} — 정책 성공률이 "
            "부족하거나 체크포인트가 잘못됐다. 뱅크를 저장하지 않는다.")

    try:
        git_hash = subprocess.check_output(
            ["git", "-C", str(_HDGP), "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        git_hash = "unknown"

    save_bank(
        out,
        joint_names=joint_names,
        joint_pos=np.concatenate(rows_q)[: args.count],
        joint_target=np.concatenate(rows_qt)[: args.count],
        cup_pose=np.concatenate(rows_cup)[: args.count],
        num_contacts=np.concatenate(rows_nc)[: args.count],
        bead_state=None,
        meta=dict(
            robot_usd=profile.asset.usd_relpath,
            profile=profile.name,
            checkpoint=str(ckpt),
            git_hash=git_hash,
            enable_gravity=bool(env_cfg.enable_gravity),
            enable_self_collisions=bool(env_cfg.enable_self_collisions),
            success_hold=int(args.success_hold),
        ),
    )
    print(f"[collect] DONE: {args.count}개 → {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()

#!/usr/bin/env python3
"""warm bank 전수 pour-물리 안정성 검증 → 안정 상태만 남긴 hdf5 생성.

각 warm 상태를 pour env(zero-action, freeze hand)에 순차 리셋해 bead 유지 여부를
판정한다. 원인 불문 "pour 물리에서 유지되는 파지"만 남기는 실용 필터 —
수집(grasp_v1)과 소비(pour)의 물리/파지스타일 차이로 인한 리셋 유실을 차단한다.

★08.16 수정: --src 가 pour env 의 warm 뱅크까지 결정한다(아래 _apply_src_to_cfg).
이전에는 --src 가 **마지막 hdf5 기록에만** 쓰이고 pour env 는 cfg 하드코딩 경로
(pour_right_env_cfg.warm_state_paths = data/grasp_warm_tesollo.hdf5)를 로드했다.
→ 새로 수집한 파일을 --src 로 줘도 실제로는 구 캐시를 측정했고, 08.15~16 의 pour 통과율
   보고(9~15%, 99.1%, 99.3%)가 전부 무효가 됐다. 크기가 우연히 맞으면 잘못된 keep 인덱스를
   조용히 기록하는 더 나쁜 실패도 가능했다.

사용:
  isaaclab.sh -p scripts/warm_states/validate_warm_states_in_pour.py \
      --num_envs 512 --rounds 4 --steps 300 \
      --src /abs/path/grasp_warm_NEW.hdf5 --out /abs/path/grasp_warm_NEW_valid.hdf5

  DR 설정과 동일 scale_set 로 검증하려면(spec 매칭 경로 동일화):
      --spec_filter cfg "env.source_cup_scale_set=[0.85,1.0,1.15,1.3]" \
      "env.source_warm_spec_map=[0,1,2,3]"

⚠️함정 3종(08.16 조사 확인):
 ① warm_state_bank._resolve_paths 는 경로가 없으면 **같은 basename 의 다른 파일로 조용히
    대체**한다(datasets/ 후보 탐색). 반드시 절대경로를 쓰고, 실행 로그의
    "loaded N warmstart states from disk (<path>)" 가 --src 와 같은지 눈으로 확인할 것.
 ② 뱅크 로더는 object_spawn_z 불일치 시 hard-raise(tolerance 1e-4, pour 기대 0.297).
 ③ warm_validation_sequential=True 와 spec 풀이 동시 활성이면 커서가 풀별이라
    풀 밖 인덱스는 영영 안 뽑힌다 → unseen 이 구조적으로 커진다(--rounds 상향 필요).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--rounds", type=int, default=4, help="전수 커버를 위한 리셋 라운드 수")
parser.add_argument("--steps", type=int, default=300, help="라운드당 관찰 스텝 (hold 120 + 안정 관찰)")
parser.add_argument("--keep_thresh", type=float, default=0.9, help="bead_in_source PASS 임계")
parser.add_argument("--src", type=str, default="data/grasp_warm_tesollo.hdf5")
parser.add_argument("--out", type=str, default="data/grasp_warm_tesollo_valid.hdf5")
parser.add_argument("--task", type=str, default="open-tesol_b_pour_sensor-lstm")
parser.add_argument(
    "--spec_filter",
    choices=("none", "cfg"),
    default="none",
    help=(
        "none(기본): warm_object_spec_filter 를 비워 --src 파일 전체를 뱅크로 쓴다 "
        "(bank == 원본 순서 전제를 만족 → keep 인덱스가 원본과 정합). "
        "cfg: pour cfg 값(기본 (1,)=cup_big_s100)을 그대로 적용 — grasp_v1 태깅 캐시에서는 "
        "뱅크가 1/8 로 줄어드니 --src 와 개수가 달라지는 것이 정상이다."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys

sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym
import h5py
import numpy as np
import torch

import openarm.tesollo  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


def _apply_src_to_cfg(env_cfg) -> tuple[str, tuple[int, ...]]:
    """--src 를 pour env 의 warm 뱅크 경로로 강제한다(★이 스크립트의 핵심 계약).

    이게 없으면 env 는 cfg 하드코딩 경로를 로드하고, --src 는 마지막 파일 기록에만 쓰여
    "수집한 파일이 아닌 다른 파일을 측정"하게 된다(08.16 무효 측정 사고).
    절대경로로 정규화해 _resolve_paths 의 basename 대체 폴백도 함께 차단한다.

    Returns:
        (src 절대경로, 실제 적용될 spec 필터). 필터가 비면 () — 전수 검증.
    """
    src_abs = str(Path(args_cli.src).expanduser().resolve())
    if not Path(src_abs).is_file():
        raise FileNotFoundError(f"--src 파일이 없다: {src_abs}")
    env_cfg.warm_state_paths = (src_abs,)
    if args_cli.spec_filter == "none":
        env_cfg.warm_object_spec_filter = ()
    spec_filter = tuple(getattr(env_cfg, "warm_object_spec_filter", ()) or ())
    print(
        f"[validate] warm bank ← {src_abs}\n"
        f"[validate] spec_filter={args_cli.spec_filter} → {spec_filter or '(전수)'}",
        flush=True,
    )
    return src_abs, spec_filter


def _bank_to_orig_indices(src_group, spec_filter: tuple[int, ...]) -> "np.ndarray | None":
    """뱅크 인덱스 → 원본 hdf5 인덱스 매핑.

    로더(warm_state_bank._resolve/filter)는 `np.isin(object_spec_idx, filter)` 로 부분집합을
    만들되, **태깅이 없는 구캐시(전부 -1)에서는 필터를 통째로 무시**한다. 그 분기를 여기서
    그대로 재현해야 keep 인덱스를 원본에 정확히 되돌릴 수 있다.

    Returns:
        필터가 실제로 적용되면 원본 인덱스 배열, 아니면 None(뱅크 == 원본).
    """
    if not spec_filter:
        return None
    if "object_spec_idx" not in src_group:
        return None                      # 태깅 없는 구캐시 → 로더가 필터 무시
    spec_idx = np.asarray(src_group["object_spec_idx"])
    if not (spec_idx >= 0).any():
        return None                      # 전부 -1 → 로더가 필터 무시
    return np.nonzero(np.isin(spec_idx, np.asarray(spec_filter, dtype=spec_idx.dtype)))[0]


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.warm_validation_sequential = True
    src_abs, spec_filter = _apply_src_to_cfg(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    zero = torch.zeros(args_cli.num_envs, uenv.cfg.action_space, device=uenv.device)

    n_bank = int(uenv._warmstart_cache_count)
    passed = np.zeros(n_bank, dtype=bool)
    failed = np.zeros(n_bank, dtype=bool)

    for rnd in range(args_cli.rounds):
        env.reset()
        picks0 = uenv._last_warm_pick.clone()          # 라운드 시작 pick 스냅샷
        early_fail = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=uenv.device)
        for t in range(args_cli.steps):
            _, _, terminated, truncated, _ = env.step(zero)
            done = (terminated | truncated).to(uenv.device)
            early_fail |= done.bool()                  # 관찰 중 조기종료 = FAIL (유실/파국)
        keep = uenv._bead_in_source_fraction >= args_cli.keep_thresh
        ok = keep & (~early_fail)
        p = picks0.cpu().numpy()
        okn = ok.cpu().numpy()
        for i in range(args_cli.num_envs):
            if p[i] < 0:
                continue
            if okn[i]:
                passed[p[i]] = True
            else:
                failed[p[i]] = True
        print(f"[validate] round {rnd+1}/{args_cli.rounds}: pass_now={int(okn.sum())}/{args_cli.num_envs} "
              f"| bank pass={int(passed.sum())} fail={int(failed.sum())} "
              f"seen={int((passed | failed).sum())}/{n_bank}", flush=True)

    keep_mask = passed & (~failed)                     # 한 번이라도 FAIL이면 제외 (보수)
    unseen = ~(passed | failed)
    print(f"[validate] 최종: keep={int(keep_mask.sum())} fail={int(failed.sum())} "
          f"unseen={int(unseen.sum())} / {n_bank}", flush=True)

    # keep 인덱스(뱅크 공간) → 원본 hdf5 인덱스로 되돌려 저장한다.
    # spec 필터가 걸리면 뱅크는 원본의 부분집합이므로 매핑이 필수(없이 저장하면 조용히
    # 엉뚱한 상태를 남긴다 — bank!=src raise 보다 나쁜 실패).
    src = h5py.File(src_abs, "r")
    g = src["warm_states"]
    n_src = g["arm_joint_pos"].shape[0]
    orig_idx_map = _bank_to_orig_indices(g, spec_filter)
    expected_bank = n_src if orig_idx_map is None else int(orig_idx_map.size)
    if expected_bank != n_bank:
        raise RuntimeError(
            f"bank({n_bank}) != 예상({expected_bank}) — 뱅크 구성이 스크립트 예상과 다르다.\n"
            f"  src = {src_abs}  (원본 {n_src}개, spec_filter={spec_filter or '(전수)'})\n"
            f"  원인 후보 ①env.source_cup_scale_set 이 켜져 spec 풀 매칭 모드로 전환됨"
            f"(그 경우 로더가 필터를 무시한다) ②로더가 다른 파일을 집음.\n"
            f"  ※실행 로그의 'loaded N warmstart states from disk (<path>)' 경로가 위 src 와 "
            f"같은지 확인할 것."
        )
    _keep_bank = np.nonzero(keep_mask)[0]
    idx = _keep_bank if orig_idx_map is None else orig_idx_map[_keep_bank]
    _seen = int((passed | failed).sum())
    print(
        f"[validate] 통과율 = {int(keep_mask.sum())}/{_seen} "
        f"({100.0 * keep_mask.sum() / max(_seen, 1):.1f}%) — unseen {int(unseen.sum())} 제외",
        flush=True,
    )
    with h5py.File(args_cli.out, "w") as dst:
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        dst.attrs["count"] = len(idx)
        dst.attrs["meta/pour_validated"] = 1.0
        gout = dst.create_group("warm_states")
        for key in g.keys():
            gout.create_dataset(key, data=np.asarray(g[key])[idx])
    print(f"[validate] saved {len(idx)} states → {args_cli.out}", flush=True)


main()
simulation_app.close()

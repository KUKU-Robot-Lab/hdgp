#!/usr/bin/env python3
"""학습된 정책의 **실제** obs/state/action 차원을 런 산출물에서 읽는다 (GPU 불필요).

왜 소스를 믿으면 안 되는가
--------------------------
업그레이드된 네 태스크 어디에도 `*_constants.py` 나 `NUM_*` 선언이 없다. 차원은
`__post_init__` / `resolve_cfg` 가 **프로필로부터 파생**한다:

    agnostic/grasp_sensor       tesollo_right 23/114/121  ·  gripper_left 7/48/55
    agnostic/grasp_lift_fabric  bis_right     19/121/127  ·  프로필 7종마다 다름
    agnostic/pour_fabric        bis            9/210/229
    gripper/left/grasp_sensor   joint 8/36 · ik 7/35 · fab 7/35

같은 태스크라도 프로필이 다르면 계약이 다르다. 배포·평가 코드가 소스에서 차원을
추정하면 **조용히 틀린다.** 런이 남긴 것만이 사실이다.

진실원천 두 가지
----------------
1. `params/env.yaml` — env 가 선언한 차원. state_space 까지 알 수 있다.
   ⚠ ManagerBased 런(`open-grip_l_grasp_sensor*`)은 이 키를 **아예 덤프하지 않는다**(실측).
2. `nn/*.pth` — 신경망이 실제로 가진 모양. 첫 actor 층 입력 = obs, mu 헤드 출력 = action.
   ManagerBased 런의 유일한 경로다. CPU 로드(`map_location="cpu"`)라 GPU 를 쓰지 않는다.

`--verify` 는 둘을 **대조**한다. 어긋나면 그 체크포인트를 그 env 로 재생할 수 없다는
뜻이고, 태스크가 업그레이드되는 국면에서 정확히 그런 일이 생긴다.

사용:
    python3 scripts/tools/policy_contract.py                    # 전 런 표
    python3 scripts/tools/policy_contract.py --task grasp-sensor
    python3 scripts/tools/policy_contract.py --run <run_dir> --verify
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

HDGP_ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = HDGP_ROOT / "log/rl_games"
_VLM_SRC = HDGP_ROOT / "source/vlm"

if str(_VLM_SRC) not in sys.path:
    sys.path.insert(0, str(_VLM_SRC))


class ContractError(RuntimeError):
    """계약을 확정하지 못했다. **추정값으로 대신하지 않는다.**"""


# ---------------------------------------------------------------------------
# 체크포인트 판독
# ---------------------------------------------------------------------------
_ACTOR_FIRST_LAYER = "a2c_network.actor_mlp.0.weight"
_MU_HEAD = "a2c_network.mu.weight"
_OBS_NORM = "running_mean_std.running_mean"


def dims_from_state_dict(state_dict) -> tuple[int, int]:
    """rl_games state_dict → (obs_dim, action_dim).

    obs 는 첫 actor 층의 입력폭에서, action 은 mu 헤드의 출력폭에서 읽는다. LSTM 이
    끼어도 이 두 층은 그대로라 변형에 강하다. 둘 중 하나라도 없으면 예외 —
    0 이나 None 으로 채우면 조용히 틀린 계약이 흘러다닌다.
    """
    def _shape(key):
        value = state_dict.get(key)
        return tuple(value.shape) if value is not None and hasattr(value, "shape") else None

    actor = _shape(_ACTOR_FIRST_LAYER) or (None,)
    obs = actor[1] if len(actor) == 2 else None
    if obs is None:
        norm = _shape(_OBS_NORM)
        obs = norm[0] if norm and len(norm) == 1 else None

    mu = _shape(_MU_HEAD)
    act = mu[0] if mu and len(mu) == 2 else None

    if obs is None or act is None:
        raise ContractError(
            f"체크포인트에서 차원을 못 읽었다 (obs={obs}, action={act}). "
            f"기대 키: {_ACTOR_FIRST_LAYER} / {_MU_HEAD}"
        )
    return int(obs), int(act)


def read_checkpoint_contract(path: Path) -> tuple[int, int]:
    import torch

    # 우리가 만든 산출물만 읽는다. rl_games 체크포인트는 텐서 외 객체를 품어
    # weights_only=True 로는 열리지 않는다.
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = blob.get("model", blob) if isinstance(blob, dict) else blob
    if not isinstance(state_dict, dict):
        raise ContractError(f"체크포인트 구조를 모르겠다: {path}")
    return dims_from_state_dict(state_dict)


def pick_checkpoint(run_dir: Path) -> Path | None:
    """best(태스크명.pth) 우선, 없으면 가장 최근 last_*.pth."""
    nn_dir = run_dir / "nn"
    if not nn_dir.is_dir():
        return None
    plain = sorted(p for p in nn_dir.glob("*.pth") if not p.name.startswith("last_"))
    if plain:
        return plain[0]
    lasts = sorted(nn_dir.glob("last_*.pth"), key=lambda p: p.stat().st_mtime)
    return lasts[-1] if lasts else None


# ---------------------------------------------------------------------------
# 런 해석
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunContract:
    run_dir: Path
    observation_dim: int
    action_dim: int
    state_dim: int | None
    source: str                    # "env.yaml" | "checkpoint"
    checkpoint: Path | None
    mismatch: str | None = None    # --verify 에서 두 출처가 어긋난 경우


def _env_yaml_dims(run_dir: Path):
    from vlm.pouring.checkpoint_resolver import read_policy_contract

    env_yaml = run_dir / "params/env.yaml"
    if not env_yaml.is_file():
        return None
    try:
        return read_policy_contract(env_yaml)
    except ValueError:
        # ManagerBased 런은 이 키를 덤프하지 않는다 — 결손이지 오류가 아니다.
        return None


def resolve_run(run_dir: Path, *, verify: bool = False) -> RunContract:
    run_dir = Path(run_dir)
    declared = _env_yaml_dims(run_dir)
    checkpoint = pick_checkpoint(run_dir)

    if declared is not None:
        mismatch = None
        if verify and checkpoint is not None:
            ck_obs, ck_act = read_checkpoint_contract(checkpoint)
            if (ck_obs, ck_act) != (declared.observation_dim, declared.action_dim):
                mismatch = (
                    f"env.yaml obs={declared.observation_dim} act={declared.action_dim} "
                    f"vs 체크포인트 obs={ck_obs} act={ck_act}"
                )
        return RunContract(
            run_dir=run_dir, observation_dim=declared.observation_dim,
            action_dim=declared.action_dim, state_dim=declared.state_dim,
            source="env.yaml", checkpoint=checkpoint, mismatch=mismatch,
        )

    if checkpoint is None:
        raise ContractError(
            f"{run_dir}: env.yaml 에 차원이 없고 체크포인트도 없다 — 계약을 확정할 수 없다"
        )
    obs, act = read_checkpoint_contract(checkpoint)
    return RunContract(
        run_dir=run_dir, observation_dim=obs, action_dim=act, state_dim=None,
        source="checkpoint", checkpoint=checkpoint,
    )


def discover_runs(log_root: Path = LOG_ROOT):
    """`log/rl_games/<robot>/<side>/<task>/<run>/` 을 훑는다."""
    if not log_root.is_dir():
        return []
    # 런 디렉터리 = `nn/` 또는 `params/` 를 **직접** 담고 있는 디렉터리.
    # (한 단계 더 올라가면 태스크 디렉터리가 잡혀 전부 "계약 불가" 가 된다.)
    seen = {d.parent for d in log_root.rglob("nn") if d.is_dir()} | {
        d.parent for d in log_root.rglob("params") if d.is_dir()
    }
    return sorted(d for d in seen if d.is_dir())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None, help="런 디렉터리 하나만 본다")
    ap.add_argument("--task", default=None, help="경로에 이 문자열이 든 런만")
    ap.add_argument("--verify", action="store_true",
                    help="env.yaml 과 체크포인트를 대조(체크포인트 로드 — CPU)")
    args = ap.parse_args()

    runs = [Path(args.run)] if args.run else discover_runs()
    if args.task:
        runs = [r for r in runs if args.task in str(r)]
    if not runs:
        print("해당하는 런이 없다")
        return 1

    print(f"{'런':66}{'출처':>11}{'obs':>6}{'state':>7}{'act':>5}  비고")
    print("-" * 112)
    bad = 0
    for run in runs:
        try:
            rc = resolve_run(run, verify=args.verify)
        except ContractError as exc:
            bad += 1
            rel = run.relative_to(HDGP_ROOT) if run.is_absolute() else run
            print(f"{str(rel):66}{'—':>11}{'—':>6}{'—':>7}{'—':>5}  계약 불가: {exc}")
            continue
        if rc.mismatch:
            bad += 1
        rel = rc.run_dir.relative_to(HDGP_ROOT) if rc.run_dir.is_absolute() else rc.run_dir
        note = rc.mismatch or ("" if rc.checkpoint else "체크포인트 없음")
        print(f"{str(rel):66}{rc.source:>11}{rc.observation_dim:>6}"
              f"{str(rc.state_dim or '—'):>7}{rc.action_dim:>5}  {note}")
    print("-" * 112)
    print(f"런 {len(runs)}개 — 계약 확정 {len(runs) - bad} / 문제 {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

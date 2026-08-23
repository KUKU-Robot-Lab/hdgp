"""grasp_v1 sim2real 평가 하네스 — 그리드 스윕/단일/인터랙티브.

설계: docs/superpowers/specs/2026-08-04-grasp-v1-s2r-eval-sp1-design.md
사용 예는 스펙 §3. GPU 실행은 사용자 게이트(Task 8 스모크 체크리스트).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import select
import subprocess
import sys
import time
from pathlib import Path

# hdgp 루트를 sys.path에 추가 (scripts.eval_s2r 패키지 import용)
_HDGP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _HDGP_ROOT not in sys.path:
    sys.path.insert(0, _HDGP_ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="grasp_v1 sim2real 평가 하네스 (SP1)")
parser.add_argument("--robot", required=True, choices=["left", "right"])
# ★--robot 은 자산/좌우 규약을, --task 는 실행할 gym id 를 정한다. 원래는 TASK_BY_ROBOT
#   두 개가 전부라 grasp_v1 말고는 이 하네스를 못 썼다. 기본값은 그대로 두어 기존
#   호출부의 거동을 바꾸지 않는다.
parser.add_argument("--task", default=None,
                    help="평가할 gym id (기본: --robot 에 대응하는 grasp_v1 play-lstm)")
parser.add_argument("--checkpoint", required=True, help="정책 .pth (prefix-glob 허용)")
parser.add_argument("--pose_source", default="state_frozen",
                    choices=["live", "state_frozen", "camera_frozen"])
parser.add_argument("--poses", type=str, default=None,
                    help="camera_frozen 전용: FoundationPose 결과 poses.json 경로")
parser.add_argument("--frames_meta", type=str, default=None,
                    help="camera_frozen 전용: render_pass meta.json 경로")
parser.add_argument("--grid_x", nargs=2, type=float, metavar=("MIN", "MAX"))
parser.add_argument("--grid_y", nargs=2, type=float, metavar=("MIN", "MAX"))
parser.add_argument("--grid_nx", type=int)
parser.add_argument("--grid_ny", type=int)
parser.add_argument("--grid_repeats", type=int, default=8)
parser.add_argument("--object_x", type=float)
parser.add_argument("--object_y", type=float)
parser.add_argument("--object_z", type=float, default=float("nan"))
parser.add_argument("--episodes_per_env", type=int, default=3)
parser.add_argument("--out", type=str, default=None)
parser.add_argument("--render", action="store_true", help="단일 모드 GUI 재생")
parser.add_argument("--real-time", action="store_true")
parser.add_argument("--interactive", action="store_true", help="상주 대화 세션(GUI)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if hydra_args:
    parser.error(f"알 수 없는 인자: {hydra_args}")


def _validate_mode(a) -> str:
    grid_args = (a.grid_x, a.grid_y, a.grid_nx, a.grid_ny)
    has_grid = any(v is not None for v in grid_args)
    has_single = a.object_x is not None or a.object_y is not None
    if a.interactive:
        if has_grid or has_single:
            parser.error("--interactive 는 그리드/단일 인자와 함께 쓸 수 없습니다")
        return "interactive"
    if has_grid and has_single:
        parser.error("그리드 인자와 --object_x/y 는 상호배타입니다")
    if has_grid:
        if not all(v is not None for v in grid_args):
            parser.error("그리드 모드는 --grid_x/--grid_y/--grid_nx/--grid_ny 전부 필요")
        if a.render:
            parser.error("--render 는 단일 모드(--object_x/y) 전용입니다")
        return "grid"
    if has_single:
        if a.object_x is None or a.object_y is None:
            parser.error("단일 모드는 --object_x 와 --object_y 둘 다 필요")
        return "single"
    parser.error("모드를 지정하세요: 그리드 인자 | --object_x/y | --interactive")


def _validate_pose_source_args(a, mode: str) -> None:
    """--poses/--frames_meta 는 camera_frozen 전용 — 부팅 전 조기 오용 차단(M4).

    camera_frozen은 그리드 스윕 전용이다(스펙 §4.3/§5) — pose 파일이 Pass 1 렌더 그리드와
    env 배치를 1:1로 정렬하는 계약이라, 단일/인터랙티브 모드(num_envs가 그리드와 무관하게
    1)로 실행하면 provider 버퍼([grid_num_envs,3])와 실제 num_envs(1)가 어긋난다. 이걸
    여기서 막지 않으면 무거운 Isaac 기동 이후에야 IndexError/torch.cat RuntimeError로
    터진다 — 그러니 부팅 전에 명시적으로 거부한다.
    """
    if a.pose_source == "camera_frozen":
        if a.poses is None or a.frames_meta is None:
            parser.error("--pose_source camera_frozen 은 --poses 와 --frames_meta 둘 다 필요합니다")
        if mode != "grid":
            parser.error("--pose_source camera_frozen 은 그리드 모드 전용입니다 (Pass 1 렌더와 env 정렬이 전제)")
    elif a.poses is not None or a.frames_meta is not None:
        parser.error("--poses/--frames_meta 는 --pose_source camera_frozen 전용입니다")


def _validate_checkpoint_early(checkpoint: str) -> None:
    """Isaac 기동 전 간단 존재 확인 (M4).

    --checkpoint 는 play.py와 동일하게 prefix-glob(확장자 없는 절단 경로, 예: `ep_15000`)도
    허용하는데, 그 해석(retrieve_file_path·glob 매칭)은 Isaac 기동 후에만 쓸 수 있는
    헬퍼(_resolve_checkpoint_path)가 필요하다. 그래서 여기서는 '완전한 .pth 경로인데
    파일이 없는' 명백한 오타만 조기 차단하고, 확장자 없는 prefix는 그대로 통과시켜
    post-launch 해석에 맡긴다.
    """
    if "*" in checkpoint or not checkpoint.endswith(".pth"):
        return
    candidate = Path(checkpoint).expanduser()
    if candidate.is_file():
        return
    if not candidate.is_absolute() and (Path(_HDGP_ROOT) / candidate).is_file():
        return
    parser.error(f"--checkpoint 경로를 찾을 수 없습니다: {checkpoint}")


MODE = _validate_mode(args_cli)

# ---- Isaac 기동 전 fail-fast (M4): 무거운 AppLauncher(GPU/렌더러 초기화) 뒤에서 인자 오류로
#      실패하면 기동 시간을 낭비한다 — 순수 파이썬으로 검증 가능한 것만 여기서 미리 걸러낸다.
if MODE == "grid":
    from scripts.eval_s2r.grid import GridSpec  # 모듈 최상단이 torch 미의존이라 기동 전 import 안전
    try:
        GridSpec(args_cli.grid_x[0], args_cli.grid_x[1], args_cli.grid_nx,
                 args_cli.grid_y[0], args_cli.grid_y[1], args_cli.grid_ny, args_cli.grid_repeats)
    except ValueError as e:
        parser.error(str(e))
_validate_checkpoint_early(args_cli.checkpoint)
_validate_pose_source_args(args_cli, MODE)

if MODE == "interactive":
    args_cli.real_time = True  # 인터랙티브 세션은 기본 실시간 재생(스펙 §4.6)
if MODE == "interactive" or args_cli.render:
    args_cli.headless = False  # GUI 강제
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- Isaac 기동 후 import (play.py 상단 순서와 동일, play.py:115-183 패턴 복제) ----
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml as _yaml  # noqa: E402

from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.io import load_yaml  # noqa: E402
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg  # noqa: E402

import openarm.tasks  # noqa: F401,E402

from scripts.eval_s2r.console import SessionState, apply_command, parse_command  # noqa: E402
from scripts.eval_s2r.grid import (  # noqa: E402
    GridSpec, build_cells, build_spawn_tensor, env_to_cell, single_spawn_tensor,
)
from scripts.eval_s2r.providers import make_provider  # noqa: E402
from scripts.eval_s2r.report import (  # noqa: E402
    EpisodeResult, aggregate, write_csv, write_heatmap, write_summary,
)

TASK_BY_ROBOT = {
    "left": "open-tesol_l_grasp_v1-play-lstm",
    "right": "open-tesol_r_grasp_v1-play-lstm",
}


def resolve_task(args) -> str:
    """--task 우선, 없으면 --robot 기본값. 기본값은 grasp_v1 전용이었다."""
    return args.task or TASK_BY_ROBOT[args.robot]
# 학습 스폰 분포(cfg 기본): 중심 ±(xy_range + ADR max). 벗어나면 경고만 (분포외 측정이 목적).
TRAIN_RANGE_WARN = 0.08 + 0.06
# 인터랙티브 STAGED 대기 위치(스펙 §4.6): obj_out_x_max 밖 원거리라 물리에 안 걸림.
STAGE_AWAY = (1.0, 1.0, float("nan"))


# ============================================================================
# play.py:144-164 패턴 복제 — logged env.yaml 전용 yaml loader.
# EventCfg의 SceneEntityCfg 기본값 slice(None)이 `!!python/object/apply:builtins.slice`
# 태그로 덤프되는데 FullLoader가 거부하므로 해당 태그만 명시적으로 복원한다.
# ============================================================================
class _RunCfgYamlLoader(_yaml.FullLoader):
    pass


_RunCfgYamlLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice",
    lambda loader, node: slice(*loader.construct_sequence(node, deep=True)),
)


def _load_run_yaml(path: str):
    try:
        return load_yaml(path)
    except _yaml.constructor.ConstructorError:
        with open(path) as f:
            return _yaml.load(f, Loader=_RunCfgYamlLoader)


# play.py:267-307 패턴 복제 — 체크포인트 prefix-glob 해석.
def _checkpoint_sort_key(path: Path) -> tuple[float, int, float]:
    """Prefer higher reward, then later epoch, then newer mtime for prefix matches."""
    match = re.search(r"_ep_(\d+)_rew_([-+]?\d+(?:\.\d+)?)", path.name)
    if match:
        return (float(match.group(2)), int(match.group(1)), path.stat().st_mtime)
    return (float("-inf"), -1, path.stat().st_mtime)


def _resolve_checkpoint_path(checkpoint: str) -> str:
    """Resolve checkpoint paths and common truncated-prefix CLI input."""
    try:
        return retrieve_file_path(checkpoint)
    except FileNotFoundError:
        pass

    candidate = Path(checkpoint).expanduser()
    search_candidates = [candidate]
    if not candidate.is_absolute():
        # play.py는 Path(__file__).resolve().parents[3]으로 sbm_root를 구하는데,
        # 이 파일은 scripts/eval_s2r/ 아래(1단계 얕음)라 이미 계산된 _HDGP_ROOT를 재사용한다.
        search_candidates.append((Path(_HDGP_ROOT) / candidate).resolve())

    prefix_matches: list[Path] = []
    for candidate in search_candidates:
        if candidate.is_file():
            return str(candidate)
        if candidate.parent.is_dir():
            prefix_matches.extend(sorted(candidate.parent.glob(candidate.name + "*.pth")))

    if prefix_matches:
        unique_matches = sorted(set(prefix_matches), key=_checkpoint_sort_key, reverse=True)
        if len(unique_matches) > 1:
            preview = "\n".join(f"  - {path}" for path in unique_matches[:10])
            raise FileNotFoundError(
                f"Multiple checkpoint files match prefix: {checkpoint}\n"
                f"Use the full checkpoint path. Top matches:\n{preview}"
            )
        resolved = unique_matches[0]
        print(f"[INFO] Parsed unique checkpoint prefix: {checkpoint} -> {resolved}")
        return str(resolved)

    raise FileNotFoundError(f"Unable to find the checkpoint file or prefix: {checkpoint}")


# play.py:310-391 패턴 복제 — 체크포인트 옆 params/env.yaml·agent.yaml 복원.
def _rebase_logged_paths(value, *, workspace_root: str):
    """Map absolute paths from another machine's logged cfg onto this workspace."""
    if isinstance(value, dict):
        return {k: _rebase_logged_paths(v, workspace_root=workspace_root) for k, v in value.items()}
    if isinstance(value, list):
        return [_rebase_logged_paths(v, workspace_root=workspace_root) for v in value]
    if isinstance(value, tuple):
        return tuple(_rebase_logged_paths(v, workspace_root=workspace_root) for v in value)
    if not isinstance(value, str) or os.path.exists(value):
        return value

    marker = "/rl_ws/"
    if marker not in value:
        return value
    rel = value.split(marker, 1)[1]
    candidate = os.path.join(workspace_root, rel)
    return candidate if os.path.exists(candidate) else value


def _apply_logged_env_cfg(target, logged: dict) -> None:
    """Recursively copy logged env config values onto the Hydra config object."""
    if not isinstance(logged, dict):
        return
    for key, value in logged.items():
        if key == "func":
            continue
        if isinstance(target, dict):
            if key not in target:
                continue
            current = target[key]
        elif hasattr(target, key):
            current = getattr(target, key)
        else:
            continue
        if callable(current):
            continue

        if isinstance(value, dict) and (isinstance(current, dict) or hasattr(current, "__dict__")):
            _apply_logged_env_cfg(current, value)
        elif isinstance(value, list) and isinstance(current, list) and any(
            isinstance(item, dict) for item in value
        ):
            # 설정 객체 리스트(spawn.assets_cfg 등): 통째로 교체하면 configclass가
            # dict로 바뀌어 asset_cfg.func 접근이 깨진다 → 같은 인덱스끼리 재귀 복원.
            for cur_item, val_item in zip(current, value):
                if isinstance(val_item, dict) and (
                    isinstance(cur_item, dict) or hasattr(cur_item, "__dict__")
                ):
                    _apply_logged_env_cfg(cur_item, val_item)
        else:
            try:
                if isinstance(target, dict):
                    target[key] = value
                else:
                    setattr(target, key, value)
            except Exception:
                pass


def _restore_run_cfg_if_available(env_cfg, agent_cfg: dict, *, resume_path: str, workspace_root: str) -> dict:
    """Use params saved next to the checkpoint so playback matches training."""
    run_dir = os.path.dirname(os.path.dirname(resume_path))
    params_dir = os.path.join(run_dir, "params")
    env_yaml = os.path.join(params_dir, "env.yaml")
    agent_yaml = os.path.join(params_dir, "agent.yaml")

    if os.path.exists(env_yaml):
        logged_env = _rebase_logged_paths(_load_run_yaml(env_yaml), workspace_root=workspace_root)
        _apply_logged_env_cfg(env_cfg, logged_env)
        print(f"[INFO] Restored playback env cfg from: {env_yaml}")
    else:
        print(f"[WARN] Run env cfg not found; using current source cfg: {env_yaml}")

    if os.path.exists(agent_yaml):
        logged_agent = _rebase_logged_paths(_load_run_yaml(agent_yaml), workspace_root=workspace_root)
        if isinstance(logged_agent, dict) and "params" in logged_agent:
            print(f"[INFO] Restored playback agent cfg from: {agent_yaml}")
            return logged_agent
        print(f"[WARN] Ignoring malformed run agent cfg: {agent_yaml}")
    else:
        print(f"[WARN] Run agent cfg not found; using current source cfg: {agent_yaml}")
    return agent_cfg


def _build_grid_and_num_envs(a, mode):
    if mode == "grid":
        spec = GridSpec(a.grid_x[0], a.grid_x[1], a.grid_nx,
                        a.grid_y[0], a.grid_y[1], a.grid_ny, a.grid_repeats)
        if spec.repeats % 8 != 0:
            print(f"[WARN] repeats={spec.repeats} 가 8의 배수가 아님 — 셀마다 물체(8종 % 배정) 구성이 달라짐")
        cells = build_cells(spec)
        return spec, cells, len(cells) * spec.repeats
    if mode == "single":
        return None, [(a.object_x, a.object_y)], 1
    return None, [], 1  # interactive


def _snapshot_metrics(ge) -> dict:
    """스텝 실행 '전' 지표 스냅샷.

    env.step()이 done env를 내부에서 즉시 자동 reset하므로(episode_success_buf 등이
    0으로 되돌아감), done을 감지한 시점엔 이미 해당 에피소드의 최종 상태를 잃는다.
    그래서 매 스텝 step() 호출 '직전'에 스냅샷을 찍어두고, done이면 그 스냅샷으로
    지표를 기록한다(정확히는 한 스텝 전 상태 — 알려진 근사, Task 8 GPU 스모크에서
    play.py --eval_episodes 결과와 ±0.05 이내 일치하는지로 검증한다).
    """
    return {
        "success": ge.episode_success_buf.clone(),
        "object_pos": ge.object_pos.clone(),
        "object_init_pos": ge.object_init_pos.clone(),
        "binary_contact": ge.binary_contact_buf.clone(),
        "obj_idx": ge.multi_object_idx_onehot.argmax(dim=-1).clone(),
    }


def _snapshot_to_result(snap: dict, env_idx: int, cell_idx: int) -> EpisodeResult:
    obj_pos = snap["object_pos"][env_idx]
    obj_init = snap["object_init_pos"][env_idx]
    lift_h = float((obj_pos[2] - obj_init[2]).item())
    disp = float(torch.norm(obj_pos[:2] - obj_init[:2]).item())
    contact_row = snap["binary_contact"][env_idx].float()
    grip = float(contact_row.sum().item())
    finger_contacts = tuple(contact_row.tolist())  # (thumb,index,middle,ring,pinky)
    obj_idx = int(snap["obj_idx"][env_idx].item())
    invalid = not bool(torch.isfinite(obj_pos).all().item())
    success = bool(snap["success"][env_idx].item())
    return EpisodeResult(cell_idx=cell_idx, success=success, lifted=lift_h > 0.05,
                         grip_count=grip, displacement=disp, obj_idx=obj_idx, invalid=invalid,
                         finger_contacts=finger_contacts)


# 지각(카메라/FoundationPose) 실패 env가 NaN pose를 내놓을 때 물리 스텝을 오염시키지 않기 위한
# 안전한 더미 좌표(작업공간 밖 far-away, z만 임의 유한값) — 이 env의 기록 결과는 perception_fail로
# 이미 합성돼 버려지므로 값 자체의 의미는 없고, "유한하다"는 것만 중요하다.
_PERCEPTION_FAIL_DUMMY_POS = (1.0, 1.0, 0.3)


def _sanitize_override_for_failed_envs(override: torch.Tensor | None, provider):
    """provider.failed_envs(있으면)의 행을 NaN → 안전한 유한 더미로 치환.

    CameraFileProvider.get_override는 실패 env 행을 NaN으로 반환한다(provider 계약 유지 —
    실패 마커는 provider 레벨에서 NaN 그대로 둔다). 그 NaN이 obs에 실려 정책 forward에
    들어가면 액션이 NaN이 돼 env.step() 배치 전체(건강한 env 포함)가 깨질 수 있어, eval
    루프 쪽에서만 스텝 직전에 치환한다. 실패 env 결과는 batch loop에서 이미 discard되므로
    이 치환이 기록되는 결과를 오염시키지 않는다. live/state_frozen처럼 failed_envs 속성이
    없는 provider는 원본을 그대로 통과시킨다.
    """
    if override is None:
        return None
    failed_envs = getattr(provider, "failed_envs", None)
    if not failed_envs:
        return override
    sanitized = override.clone()
    dummy = torch.tensor(_PERCEPTION_FAIL_DUMMY_POS, dtype=sanitized.dtype, device=sanitized.device)
    for env_id in failed_envs:
        sanitized[env_id] = dummy
    return sanitized


def _eval_step(env, ge, agent, obs, provider):
    """스텝 1회 실행 + 지표 스냅샷 + LSTM done 리셋 + provider 재캡처.

    play.py:684-710(스텝 루프 뼈대), :1053-1055(LSTM done 리셋) 패턴 복제.
    Task 7(인터랙티브)이 그대로 재사용하는 단위 함수 — num_envs=1이면 이 함수
    하나가 "한 스텝"이고, 아래 _run_one_episode가 이걸 감아 "한 에피소드"로 만든다.

    반환: (새 obs, dones[N] bool, done 이전 스냅샷 dict)
    """
    snap = _snapshot_metrics(ge)
    # NaN 치환은 여기서 한 번만 — override가 None(live)이면 그대로 통과.
    ge.eval_cup_pos_override = _sanitize_override_for_failed_envs(provider.get_override(ge), provider)
    with torch.inference_mode():
        actions = agent.get_action(agent.obs_to_torch(obs), is_deterministic=True)
        obs, _, dones, _ = env.step(actions)
        if isinstance(obs, dict):
            obs = obs["obs"]
        # LSTM done 리셋은 play.py처럼 inference_mode "안"에서 수행해야 한다 —
        # get_action이 갱신한 agent.states는 inference 텐서라 밖에서의 in-place가 금지됨.
        if agent.is_rnn and agent.states is not None and len(dones) > 0:
            for s in agent.states:
                s[:, dones, :] = 0.0
    if bool(dones.any()):
        done_ids = dones.nonzero(as_tuple=True)[0]
        provider.on_reset(ge, done_ids)
    return obs, dones, snap


def _run_one_episode(env, ge, agent, obs, provider, env_idx: int = 0,
                     cell_idx: int = 0, real_time: bool = False,
                     record_traj: list | None = None,
                     stop_before_done: bool = False):
    """env_idx 하나가 done 될 때까지 _eval_step 반복 — Task 7 인터랙티브(num_envs=1) 재사용.

    num_envs=1 세션 전용(env_idx 기본 0). real_time=True 면 배치 루프(play.py:1211-1213
    패턴, main()의 배치 루프와 동일 로직)와 마찬가지로 스텝마다 실측 소요시간을 step_dt에
    맞춰 sleep한다 — 인터랙티브 GUI 세션을 벽시계 속도로 재생하기 위함.

    매 반복 진입 시 simulation_app.is_running()을 재확인한다(배치 루프의
    `while ... and simulation_app.is_running()` 패턴과 동일한 이유) — GUI 창이 done 이전에
    닫히면 이미 종료된 시뮬레이션을 계속 step()하는 크래시/행 대신, 마지막으로 확보한
    스냅샷(있으면)을 invalid=True로 표시해 즉시 반환한다. 중단된 에피소드는 유효 표본이
    아니므로 절대 invalid=False 로 집계돼선 안 된다.
    stop_before_done=True(인터랙티브 back2ini 지원): env.step()의 done 프레임은 내부
    auto-reset으로 장면을 순간이동시키므로, 타임아웃 2스텝 전에 조기 정지해 "잡은 상태"
    장면을 보존한다. 지표는 정지 시점의 즉석 스냅샷(성공 래치는 누적 OR라 그대로 유효).
    단 물체 낙하/이탈 등으로 done이 먼저 오면 장면은 이미 리셋됨 → scene_intact=False.

    반환: (새 obs, EpisodeResult(cell_idx=cell_idx), scene_intact: bool)
    — 정상 done 종료·앱 중단이면 scene_intact=False, 조기 정지면 True.
    """
    step_dt = ge.step_dt
    max_steps = max(1, int(ge.max_episode_length) - 2) if stop_before_done else None
    steps_done = 0
    snap = None
    if record_traj is not None:
        # back2ini 역재생용: 에피소드의 전체 관절 자세를 컨트롤 스텝마다 기록.
        # 시작 자세(리셋 직후)도 포함해야 역재생의 종점이 초기 자세가 된다.
        record_traj.append(ge.robot.data.joint_pos[env_idx].detach().clone())
    while True:
        if not simulation_app.is_running():
            if snap is None:
                # 첫 스텝 전에 이미 앱이 닫힘 — ge 상태를 더 읽지 않고(torn-down 시뮬 접근 회피)
                # 합성 invalid 결과만 반환.
                return obs, EpisodeResult(
                    cell_idx=cell_idx, success=False, lifted=False,
                    grip_count=0.0, displacement=0.0, obj_idx=-1, invalid=True,
                    finger_contacts=(0.0,) * 5,
                ), False
            aborted = _snapshot_to_result(snap, env_idx, cell_idx=cell_idx)
            return obs, dataclasses.replace(aborted, invalid=True), False
        if max_steps is not None and steps_done >= max_steps:
            # 조기 정지: done(auto-reset) 전에 멈춰 장면 보존. 지표는 현재 상태 즉석 스냅샷.
            now = _snapshot_metrics(ge)
            return obs, _snapshot_to_result(now, env_idx, cell_idx=cell_idx), True
        start_time = time.time()
        obs, dones, snap = _eval_step(env, ge, agent, obs, provider)
        steps_done += 1
        if record_traj is not None and not bool(dones[env_idx]):
            # done 프레임은 제외 — env.step()이 내부 auto-reset을 이미 수행해 joint_pos가
            # "리셋 자세"로 바뀐 뒤라, 기록하면 역재생 첫 프레임이 순간이동이 된다.
            record_traj.append(ge.robot.data.joint_pos[env_idx].detach().clone())
        if real_time:
            sleep_time = step_dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        if bool(dones[env_idx]):
            return obs, _snapshot_to_result(snap, env_idx, cell_idx=cell_idx), False


def _replay_reverse(ge, traj: list, real_time: bool = True) -> None:
    """기록된 관절 궤적을 역순으로 kinematic 추종 재생 — 잡기→놓기 연속동작 관찰용.

    scripts/probes/probe_palm_orientation.py:146-151의 수동 스텝 idiom:
    set_joint_position_target → scene.write_data_to_sim → sim.step → scene.update.
    env.step()을 우회하므로 정책/리워드/done과 무관하게 순수 자세 추종이며, 물체는
    물리로 자연스럽게 놓인다(손이 열리면 낙하·안착). 컨트롤 스텝 1프레임당
    decimation번 물리 스텝(학습과 동일 120Hz)으로 부드럽게 추종한다.
    """
    decim = max(1, int(getattr(ge.cfg, "decimation", 2)))
    phys_dt = ge.sim.get_physics_dt()
    step_dt = ge.step_dt
    for q in reversed(traj):
        if not simulation_app.is_running():
            return
        start_time = time.time()
        ge.robot.set_joint_position_target(q.unsqueeze(0))
        for _ in range(decim):
            ge.scene.write_data_to_sim()
            ge.sim.step()
            ge.scene.update(phys_dt)
        if real_time:
            sleep_time = step_dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)


def _poll_stdin() -> str | None:
    """논블로킹 stdin 1줄 폴링. 입력 없으면 None. (GUI 렌더 루프를 막지 않기 위함)"""
    r, _, _ = select.select([sys.stdin], [], [], 0.0)
    if r:
        return sys.stdin.readline()
    return None


def _setup(task: str, num_envs: int, mode: str, cells: list, spec: GridSpec | None):
    """env cfg/checkpoint/agent/provider 공통 로더 — 배치(main)·인터랙티브 양쪽이 공유(DRY).

    main()에 인라인돼 있던 로더 블록을 순수 추출한 것 — 동작은 배치 모드 기준 완전히 동일하다.
    mode에 따라 최초 eval_fixed_spawn_local 만 달라진다: grid/single 은 기존과 동일하게
    각 모드의 스폰 좌표, interactive 는 STAGE_AWAY(작업공간 밖, 스펙 §4.6)로 파킹해
    세션 시작 시 물체가 로봇 앞에 놓이지 않도록 한다 — interactive_main() 이 spawn 명령을
    받을 때마다 이 값을 덮어쓰고 다시 reset() 한다.

    반환: (env, ge, agent, resume_path, provider, obs) — obs 는 최초 reset() 직후 관측값.
    """
    # ---- env cfg (play.py의 parse_env_cfg 경로, hydra 데코레이터 대신 probe_extract_success.py
    #      의 non-hydra 패턴 복제 — 이 스크립트는 hydra_task_config를 쓰지 않는다) ----
    env_cfg = parse_env_cfg(task, device=args_cli.device, num_envs=num_envs)

    # ---- checkpoint resolve + env.yaml/agent.yaml 복원 (play.py:275-307, :369-391 복제) ----
    resume_path = _resolve_checkpoint_path(args_cli.checkpoint)
    agent_cfg = load_cfg_from_registry(task, "rl_games_cfg_entry_point")
    workspace_root = os.path.abspath(os.path.join(_HDGP_ROOT, ".."))
    agent_cfg = _restore_run_cfg_if_available(
        env_cfg, agent_cfg, resume_path=resume_path, workspace_root=workspace_root,
    )
    # 복원된 logged env.yaml이 scene.num_envs(학습 시 배치 크기)를 되살릴 수 있으므로,
    # 그리드/단일 모드가 계산한 num_envs로 다시 확정한다 (play.py:_apply_playback_env_overrides
    # 가 args_cli.num_envs로 복원 이후 덮어쓰는 것과 동일한 이유).
    env_cfg.scene.num_envs = num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # ---- ADR 항상 off + demo/warm 스폰 분기 차단 (play.py:429-444 패턴 복제,
    #      이 하네스는 --disable_adr 플래그 없이 무조건 적용) ----
    _disabled_adr = []
    for adr_attr in (
        "enable_adr",
        "enable_noise_adr",
        "enable_bead_count_adr",
        "enable_success_adr",
        "enable_spill_adr",
    ):
        if hasattr(env_cfg, adr_attr):
            setattr(env_cfg, adr_attr, False)
            _disabled_adr.append(adr_attr)
    if _disabled_adr:
        print(f"[INFO] ADR disabled for eval: {', '.join(_disabled_adr)}")
    if hasattr(env_cfg, "enable_demo_grasp_reset"):
        env_cfg.enable_demo_grasp_reset = False
    if hasattr(env_cfg, "enable_warm_state_export"):
        env_cfg.enable_warm_state_export = False

    # 학습 스폰 분포 밖 경고 (분포외 측정이 목적이므로 차단하지 않고 경고만). interactive 는
    # cells=[] 라 항상 무경고.
    center_x = getattr(env_cfg, "object_spawn_x_center", 0.0)
    center_y = getattr(env_cfg, "object_spawn_y_center", 0.0)
    out_of_range = [
        (x, y) for (x, y) in cells
        if abs(x - center_x) > TRAIN_RANGE_WARN or abs(y - center_y) > TRAIN_RANGE_WARN
    ]
    if out_of_range:
        print(
            f"[WARN] {len(out_of_range)}개 좌표가 학습 스폰 범위"
            f"(center=({center_x:.3f},{center_y:.3f}) ± {TRAIN_RANGE_WARN:.3f}) 밖입니다 — "
            f"분포외 측정이 목적이면 무시해도 됩니다."
        )

    # ---- gym.make → RlGamesVecEnvWrapper → vecenv/env_configurations 등록 (play.py:616-644 복제,
    #      video/MARL 분기는 이 하네스 범위 밖이라 가져오지 않는다) ----
    env = gym.make(task, cfg=env_cfg, render_mode=None)

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    ge = env.unwrapped  # play.py:823 패턴 복제 (RlGamesVecEnvWrapper.unwrapped → 실제 grasp env)

    # eval_fixed_spawn_local: 훅이 env_ids로 인덱싱한 뒤 .to(self.device)를 호출하므로
    # (grasp_right_env.py:1541 부근 `_eval_spawn[env_ids].to(self.device)`), env_ids가
    # ge.device 텐서라면 인덱싱 이전에 이미 같은 device여야 한다 — 그래서 여기서 미리 .to(ge.device).
    if mode == "grid":
        initial_spawn = build_spawn_tensor(cells, spec.repeats, z=args_cli.object_z)
    elif mode == "single":
        initial_spawn = single_spawn_tensor(args_cli.object_x, args_cli.object_y, args_cli.object_z, 1)
    else:  # interactive — 세션 시작 시 STAGED, 물체는 작업공간 밖에 파킹
        initial_spawn = single_spawn_tensor(*STAGE_AWAY, 1)
    ge.eval_fixed_spawn_local = initial_spawn.to(ge.device)

    # ---- Runner/create_player/restore/reset (play.py:646-659 복제) ----
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = ge.num_envs
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    if args_cli.pose_source == "camera_frozen":
        provider = make_provider(
            "camera_frozen", poses_path=args_cli.poses, frames_meta_path=args_cli.frames_meta,
            robot=args_cli.robot,
        )
        # 생성 직후 pose 파일의 그리드(meta.json 기록)와 이번 실행의 그리드 인자를 대조한다 —
        # 다른 그리드로 캡처한 pose 파일을 잘못 재사용하면 셀별 좌표가 어긋난 채로 조용히
        # 집계될 수 있어 fail-fast로 막는다. spec이 없는 모드(단일/인터랙티브)는 대조할
        # 그리드가 없으므로 건너뛴다.
        if spec is not None:
            current_grid = {
                "x_min": spec.x_min, "x_max": spec.x_max, "nx": spec.nx,
                "y_min": spec.y_min, "y_max": spec.y_max, "ny": spec.ny,
                "repeats": spec.repeats,
            }
            if provider.expected_grid != current_grid:
                raise ValueError(
                    f"camera_frozen pose 파일의 그리드가 현재 실행 그리드 인자와 다릅니다: "
                    f"file={provider.expected_grid} current={current_grid}"
                )
    else:
        provider = make_provider(args_cli.pose_source)

    # reset + get_batch_size + RNN init (play.py:664-676 복제)
    # reset 전 override를 비워 stale 값(직전 세션/이전 에피소드)이 첫 관측에 새지 않도록 한다.
    ge.eval_cup_pos_override = None
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    all_ids = torch.arange(ge.num_envs, device=ge.device)
    provider.on_reset(ge, all_ids)  # 전 env 최초 캡처 (StateFrozenProvider의 NaN 검증 요건)

    return env, ge, agent, resume_path, provider, obs


def main():
    task = resolve_task(args_cli)
    spec, cells, num_envs = _build_grid_and_num_envs(args_cli, MODE)
    print(f"[INFO] mode={MODE} task={task} num_envs={num_envs}")

    env, ge, agent, resume_path, provider, obs = _setup(task, num_envs, MODE, cells, spec)

    # ---- 배치 루프 — 전 env가 episodes_per_env 채울 때까지 (play.py:682-710, :1053-1055 패턴) ----
    target = args_cli.episodes_per_env
    episode_counts = [0] * ge.num_envs
    results: list[EpisodeResult] = []
    step_dt = ge.step_dt

    # 지각(camera_frozen) 실패 env는 스텝 결과를 기록하지 않고 episodes_per_env개의 합성
    # perception_fail 에피소드로 미리 채운다. 주의: 이 env들도 씬에는 그대로 존재하고
    # 액션은 배치 전체에 대해 계산되므로 물리적으로는 계속 스텝을 밟는다(_sanitize_override_
    # for_failed_envs가 NaN pose를 안전한 더미로 치환해 배치 전체가 깨지지 않게 함) — 다만
    # 그 env가 만들어내는 실제 스텝 결과는 절대 기록하지 않고 버린다(discard). episode_counts를
    # 미리 target으로 채워두면 아래 배치 루프의 "이미 다 찬 env" 스킵 로직이 자연히 처리한다.
    for i in getattr(provider, "failed_envs", set()):
        cell_idx = env_to_cell(i, spec.repeats) if MODE == "grid" else 0
        for _ in range(target):
            results.append(EpisodeResult(
                cell_idx=cell_idx, success=False, lifted=False, grip_count=0.0,
                displacement=0.0, obj_idx=i % 8, invalid=False,
                finger_contacts=(0.0,) * 5, perception_fail=True,
            ))
        episode_counts[i] = target

    while any(c < target for c in episode_counts) and simulation_app.is_running():
        start_time = time.time()
        obs, dones, snap = _eval_step(env, ge, agent, obs, provider)
        if bool(dones.any()):
            for i in dones.nonzero(as_tuple=True)[0].tolist():
                # env당 episodes_per_env 초과분은 폐기(discard) — 그리드 셀은 repeats로 표본수가
                # 고정돼 있어, 일부 env가 먼저 끝나 더 많이 기록되면 셀별 표본이 불균등해진다.
                if episode_counts[i] >= target:
                    continue
                cell_idx = env_to_cell(i, spec.repeats) if MODE == "grid" else 0
                results.append(_snapshot_to_result(snap, i, cell_idx))
                episode_counts[i] += 1
        sleep_time = step_dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:  # play.py:1211-1213 패턴 복제
            time.sleep(sleep_time)

    # ---- 종료: 집계 → CSV/JSON(+git SHA) → (그리드면) 히트맵 → 콘솔 표 ----
    rows = aggregate(results, cells if MODE == "grid" else [(args_cli.object_x, args_cli.object_y)])

    out_dir = args_cli.out or os.path.join(_HDGP_ROOT, "log", "eval_s2r", f"{args_cli.robot}_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)

    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_HDGP_ROOT
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
        git_sha = None
        print(f"[WARN] git SHA 조회 실패: {e}")

    meta = {
        "args": vars(args_cli),
        "mode": MODE,
        "task": task,
        "checkpoint": resume_path,
        "git_sha": git_sha,
    }

    write_csv(rows, os.path.join(out_dir, "results.csv"))
    write_summary(rows, meta, os.path.join(out_dir, "summary.json"))
    if MODE == "grid":
        write_heatmap(rows, spec.nx, spec.ny, "success_rate", os.path.join(out_dir, "heatmap_success.png"))
        write_heatmap(rows, spec.nx, spec.ny, "lifted_rate", os.path.join(out_dir, "heatmap_lifted.png"))

    def _fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    print(f"\n[INFO] 결과 저장: {out_dir}")
    print(f"{'cell':>4} {'x':>8} {'y':>8} {'n':>4} {'invalid':>7} {'success':>8} {'lifted':>7} {'grip':>6} {'disp':>7}")
    for r in rows:
        print(
            f"{r['cell_idx']:>4} {r['x']:>8.3f} {r['y']:>8.3f} {r['n_episodes']:>4} {r['n_invalid']:>7} "
            f"{_fmt(r['success_rate']):>8} {_fmt(r['lifted_rate']):>7} "
            f"{_fmt(r['grip_finger_count']):>6} {_fmt(r['displacement_mean']):>7}"
        )

    env.close()


def interactive_main():
    """상주 GUI 세션: spawn → 리셋 → 1 에피소드 평가 → 결과 → STAGED 복귀 루프.

    STAGED(pending is None) 동안은 물리를 스텝하지 않고 simulation_app.update()만 돌리며
    stdin을 논블로킹 폴링한다 — GUI 응답성을 막지 않기 위함(스펙 §4.6). num_envs=1 고정.
    """
    task = resolve_task(args_cli)
    print(f"[INFO] mode={MODE} task={task} num_envs=1")
    env, ge, agent, _resume_path, provider, obs = _setup(task, 1, MODE, [], None)

    state = SessionState(last_spawn=None, obj_idx=None)
    session_results: list[EpisodeResult] = []
    print("[INTERACTIVE] 명령: spawn X Y [Z] | repeat | reset | back2ini | obj N | quit")
    print("> ", end="", flush=True)
    pending = None  # 실행 지시 dict (None 이면 STAGED)
    last_traj: list = []       # 직전 에피소드 관절 궤적 (back2ini 역재생용)
    scene_intact = False       # 조기 정지로 장면(잡은 상태)이 보존됐는가
    while simulation_app.is_running():
        if pending is None:
            # STAGED: 물리 스텝 없이 렌더만 돌리며 stdin 폴링 (GUI 응답성 유지)
            simulation_app.update()
            line = _poll_stdin()
            if line is None:
                continue
            if line == "":
                # stdin EOF(파이프 입력 소진) — readline()이 빈 문자열 반환. 사용자가 실제로
                # 빈 줄만 입력하면 "\n"이 오므로 ""와는 구분됨. 방치하면 select()가 EOF
                # 소켓을 계속 ready로 보고해 렌더 루프 속도로 [ERR] 빈 입력 스팸이 무한 반복된다.
                print("\n[INFO] stdin EOF — 세션 종료")
                break
            try:
                state, act = apply_command(state, parse_command(line))
            except ValueError as e:
                print(f"[ERR] {e}\n> ", end="", flush=True)
                continue
            if act["action"] == "quit":
                break
            if act["action"] == "noop":
                print(f"[OK] obj={state.obj_idx}\n> ", end="", flush=True)
                continue
            if act["action"] == "sweep":
                print("[ERR] sweep 은 인터랙티브 num_envs=1 세션에서 미지원 — "
                      "배치 모드로 별도 실행하세요\n> ", end="", flush=True)
                continue
            if act["action"] == "reset":
                # 환경 초기화 → 물체 대기 위치 파킹 → STAGED 유지
                ge.eval_cup_pos_override = None
                ge.eval_fixed_spawn_local = single_spawn_tensor(*STAGE_AWAY, 1).to(ge.device)
                obs = env.reset()
                if isinstance(obs, dict):
                    obs = obs["obs"]
                last_traj = []
                scene_intact = False
                print("[OK] 환경 초기화 — 대기 모드\n> ", end="", flush=True)
                continue
            if act["action"] == "back2ini":
                if not last_traj or not scene_intact:
                    print("[ERR] 역재생할 궤적이 없습니다 — spawn 으로 에피소드를 먼저 실행"
                          "하세요 (물체 낙하 등으로 장면이 리셋된 경우도 불가)\n> ",
                          end="", flush=True)
                    continue
                print(f"[BACK2INI] 역재생 시작 ({len(last_traj)} frames — 잡기→놓기)")
                _replay_reverse(ge, last_traj, real_time=args_cli.real_time)
                last_traj = []
                scene_intact = False
                print("[BACK2INI] 완료 — 초기 자세 복귀\n> ", end="", flush=True)
                continue
            pending = act
        else:
            # EVAL: 고정 스폰 설정 → reset → 1 에피소드 실행 → 결과 출력 → STAGED 복귀
            ge.eval_fixed_spawn_local = single_spawn_tensor(
                pending["x"], pending["y"], pending["z"], 1
            ).to(ge.device)
            # obj 선택: env0의 물체는 env_id%8=0 고정이라 obj N 은 경고만
            #   (MultiAsset 배정은 spawn 시점 고정 — 스펙 §7.4. obj 실선택은 num_envs=8
            #    기동 후 해당 env만 평가하는 후속 개선으로 미룸: YAGNI)
            if state.obj_idx not in (None, 0):
                print(f"[WARN] obj {state.obj_idx} 미지원(단일 env는 물체 0 고정) — 물체 0으로 진행")
            # reset 전 override를 비워 이전 에피소드의 고정 pose가 새 reset 관측에 새지 않도록 한다.
            ge.eval_cup_pos_override = None
            obs = env.reset()
            if isinstance(obs, dict):
                obs = obs["obs"]
            if agent.is_rnn:
                agent.init_rnn()
            provider.on_reset(ge, torch.arange(ge.num_envs, device=ge.device))
            last_traj = []
            obs, result, scene_intact = _run_one_episode(
                env, ge, agent, obs, provider, real_time=args_cli.real_time,
                record_traj=last_traj, stop_before_done=True,
            )
            session_results.append(result)
            if not simulation_app.is_running():
                # 에피소드 도중 GUI 창이 닫힘 — result 는 이미 invalid=True 로 기록됐다.
                # 더 이상 입력을 받을 앱이 없으므로 프롬프트를 다시 찍지 않고 세션을 정리한다.
                print("[WARN] 시뮬레이션 앱 종료 감지 — 에피소드 중단(invalid) 처리 후 세션 종료")
                break
            fingers = [int(v) for v in result.finger_contacts]  # T/I/M/R/P 순
            print(f"[RESULT] success={result.success} lifted={result.lifted} "
                  f"grip={result.grip_count:.1f} disp={result.displacement*100:.1f}cm "
                  f"obj={result.obj_idx} fingers={fingers}"
                  f"{' [INVALID]' if result.invalid else ''}")
            if scene_intact:
                print("       (장면 보존됨 — back2ini 로 잡기→놓기 역재생 가능)")
            print("> ", end="", flush=True)
            pending = None

    # 종료: 세션 이력 저장 (--out 지정 시)
    if args_cli.out and session_results:
        os.makedirs(args_cli.out, exist_ok=True)
        hist_path = os.path.join(args_cli.out, "interactive_history.json")
        with open(hist_path, "w") as f:
            json.dump([dataclasses.asdict(r) for r in session_results], f, indent=2)
        print(f"[INFO] 세션 이력 저장: {hist_path}")

    env.close()


if __name__ == "__main__":
    if MODE == "interactive":
        interactive_main()
    else:
        main()
    simulation_app.close()

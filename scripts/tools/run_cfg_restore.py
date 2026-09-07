"""저장된 run 설정(`params/env.yaml`)을 재생 cfg 에 되씌운다.

왜 여기 있나. 원래는 `rl_games/play.py` 안에만 있었다. 그런데 그림자 기록 프로브도
**같은 복원**을 해야 한다 — 기록과 재생이 다른 cfg 로 돌면 두 결과를 비교할 근거가 없다.
복사하면 그 순간 두 벌이 되고, 두 벌은 반드시 갈린다. 그래서 한 곳에 두고 둘 다 import 한다.

여기 담긴 세 가지는 전부 **사고에서 나온 것**이다:
  · `builtins.slice` 태그 — `SceneEntityCfg` 의 `joint_ids=slice(None)` 기본값이 이 태그로
    덤프되는데 `yaml.FullLoader` 가 거부한다. 이 태그 하나만 명시적으로 복원한다.
  · 절대경로 리베이스 — 다른 머신에서 학습한 run 의 cfg 는 그 머신의 절대경로를 담는다.
  · **설정 객체 리스트를 통째로 대입하지 않기** — `spawn.assets_cfg` 를 리스트째 씌우면
    configclass 가 dict 로 바뀌어 스폰 때 `dict has no attribute func` 로 깨진다.
"""

from __future__ import annotations

import os

import yaml as _yaml

try:                                        # Isaac 안에서 실행될 때
    from isaaclab.utils.io import load_yaml as _load_yaml
except ModuleNotFoundError:                 # 순수 파이썬 테스트
    def _load_yaml(path):
        with open(path) as handle:
            return _yaml.full_load(handle)


class RunCfgYamlLoader(_yaml.FullLoader):
    """logged env.yaml 전용 loader."""


RunCfgYamlLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:builtins.slice",
    lambda loader, node: slice(*loader.construct_sequence(node, deep=True)),
)


def load_run_yaml(path: str):
    try:
        return _load_yaml(path)
    except _yaml.constructor.ConstructorError:
        with open(path) as handle:
            return _yaml.load(handle, Loader=RunCfgYamlLoader)


def rebase_logged_paths(value, *, workspace_root: str):
    """다른 머신의 절대경로를 이 워크스페이스로 옮긴다.

    옮길 자리를 못 찾으면 **원본을 그대로 둔다** — 없는 경로를 지어내면 실패가 "파일 없음"
    이 아니라 "엉뚱한 자산으로 조용히 동작"이 된다.
    """
    if isinstance(value, dict):
        return {k: rebase_logged_paths(v, workspace_root=workspace_root) for k, v in value.items()}
    if isinstance(value, list):
        return [rebase_logged_paths(v, workspace_root=workspace_root) for v in value]
    if isinstance(value, tuple):
        return tuple(rebase_logged_paths(v, workspace_root=workspace_root) for v in value)
    if not isinstance(value, str) or os.path.exists(value):
        return value

    marker = "/rl_ws/"
    if marker not in value:
        return value
    rel = value.split(marker, 1)[1]
    candidate = os.path.join(workspace_root, rel)
    return candidate if os.path.exists(candidate) else value


def apply_logged_env_cfg(target, logged: dict) -> None:
    """기록된 cfg 값을 대상 설정 객체 위에 재귀적으로 얹는다."""
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
            apply_logged_env_cfg(current, value)
        elif isinstance(value, list) and isinstance(current, list) and any(
            isinstance(item, dict) for item in value
        ):
            # 설정 객체 리스트(spawn.assets_cfg 등): 통째로 교체하면 configclass 가
            # dict 로 바뀌어 asset_cfg.func 접근이 깨진다 → 같은 인덱스끼리 재귀 복원.
            for cur_item, val_item in zip(current, value):
                if isinstance(val_item, dict) and (
                    isinstance(cur_item, dict) or hasattr(cur_item, "__dict__")
                ):
                    apply_logged_env_cfg(cur_item, val_item)
        else:
            try:
                if isinstance(target, dict):
                    target[key] = value
                else:
                    setattr(target, key, value)
            except Exception:
                pass


def restore_run_cfg_if_available(env_cfg, agent_cfg: dict, *, resume_path: str,
                                 workspace_root: str) -> dict:
    """체크포인트 옆에 저장된 params 로 재생을 학습과 맞춘다."""
    run_dir = os.path.dirname(os.path.dirname(resume_path))
    params_dir = os.path.join(run_dir, "params")
    env_yaml = os.path.join(params_dir, "env.yaml")
    agent_yaml = os.path.join(params_dir, "agent.yaml")

    if os.path.exists(env_yaml):
        logged_env = rebase_logged_paths(load_run_yaml(env_yaml), workspace_root=workspace_root)
        apply_logged_env_cfg(env_cfg, logged_env)
        print(f"[INFO] Restored playback env cfg from: {env_yaml}")
    else:
        print(f"[WARN] Run env cfg not found; using current source cfg: {env_yaml}")

    if os.path.exists(agent_yaml):
        logged_agent = rebase_logged_paths(load_run_yaml(agent_yaml), workspace_root=workspace_root)
        if isinstance(logged_agent, dict) and "params" in logged_agent:
            print(f"[INFO] Restored playback agent cfg from: {agent_yaml}")
            return logged_agent
    return agent_cfg

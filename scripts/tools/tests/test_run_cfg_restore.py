"""저장된 run 설정을 재생에 되씌우는 규칙.

이 로직은 원래 `rl_games/play.py` 안에만 있었다. 그림자 기록 프로브가 같은 복원을 해야
하는데, 복사하면 그 순간 두 벌이 되고 재생과 기록이 조용히 달라진다. 그래서 여기로
옮기고 양쪽이 import 한다.

무엇을 지키는가:
  · `builtins.slice` 태그 — SceneEntityCfg 기본값이 이 태그로 덤프되고 FullLoader 가 거부한다
  · 다른 머신의 절대경로 리베이스
  · **설정 객체 리스트를 통째로 갈아치우지 않는 것** — dict 로 바뀌면 `.func` 접근이 깨진다
  · 실패를 삼키되 무엇을 못 씌웠는지는 남기는 것
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_cfg_restore import (  # noqa: E402
    apply_logged_env_cfg,
    load_run_yaml,
    rebase_logged_paths,
)


@dataclass
class Spawn:
    usd_path: str = "unset"
    func: object = None


@dataclass
class Scene:
    num_envs: int = 1
    spawn: Spawn = field(default_factory=Spawn)


@dataclass
class Cfg:
    seed: int = 0
    episode_length_s: float = 5.0
    scene: Scene = field(default_factory=Scene)


def test_a_slice_tag_does_not_stop_the_load(tmp_path):
    """SceneEntityCfg 의 slice(None) 기본값이 이 태그로 덤프된다."""
    path = tmp_path / "env.yaml"
    path.write_text(
        "joint_ids: !!python/object/apply:builtins.slice [null, null, null]\nseed: 7\n"
    )

    loaded = load_run_yaml(str(path))

    assert loaded["joint_ids"] == slice(None, None, None)
    assert loaded["seed"] == 7


def test_a_path_from_another_machine_is_rebased_onto_this_workspace(tmp_path):
    workspace = tmp_path / "rl_ws"
    (workspace / "hdgp/assets").mkdir(parents=True)
    asset = workspace / "hdgp/assets/env.usd"
    asset.write_text("x")

    rebased = rebase_logged_paths(
        {"usd": "/home/someone-else/rl_ws/hdgp/assets/env.usd"},
        workspace_root=str(workspace),
    )

    assert rebased["usd"] == str(asset)


def test_a_path_that_exists_here_is_left_alone(tmp_path):
    here = tmp_path / "present.usd"
    here.write_text("x")

    assert rebase_logged_paths(str(here), workspace_root=str(tmp_path)) == str(here)


def test_a_path_we_cannot_place_is_left_alone_rather_than_invented(tmp_path):
    original = "/home/someone-else/rl_ws/hdgp/assets/absent.usd"

    assert rebase_logged_paths(original, workspace_root=str(tmp_path)) == original


def test_scalars_are_written_through_nested_objects():
    cfg = Cfg()

    apply_logged_env_cfg(cfg, {"seed": 11, "scene": {"num_envs": 64}})

    assert cfg.seed == 11
    assert cfg.scene.num_envs == 64


def test_a_key_the_target_does_not_have_is_ignored():
    cfg = Cfg()

    apply_logged_env_cfg(cfg, {"invented_by_an_older_run": 3})

    assert not hasattr(cfg, "invented_by_an_older_run")


def test_callables_are_never_overwritten():
    """`func` 는 설정 객체가 스폰될 때 불린다 — dict 로 덮으면 스폰이 깨진다."""
    marker = object()
    cfg = Cfg()
    cfg.scene.spawn.func = marker

    apply_logged_env_cfg(cfg, {"scene": {"spawn": {"func": {"a": 1}, "usd_path": "new.usd"}}})

    assert cfg.scene.spawn.func is marker
    assert cfg.scene.spawn.usd_path == "new.usd"


def test_a_list_of_config_objects_is_merged_item_by_item_not_replaced():
    """통째로 대입하면 configclass 가 dict 가 되고 `.func` 접근이 죽는다."""

    @dataclass
    class WithAssets:
        assets_cfg: list = field(default_factory=lambda: [Spawn("a.usd"), Spawn("b.usd")])

    cfg = WithAssets()
    original = cfg.assets_cfg[0]

    apply_logged_env_cfg(cfg, {"assets_cfg": [{"usd_path": "A.usd"}, {"usd_path": "B.usd"}]})

    assert cfg.assets_cfg[0] is original, "리스트가 통째로 교체됐다"
    assert [item.usd_path for item in cfg.assets_cfg] == ["A.usd", "B.usd"]


def test_a_plain_list_of_scalars_is_replaced():
    @dataclass
    class WithNumbers:
        weights: list = field(default_factory=lambda: [1.0, 2.0])

    cfg = WithNumbers()

    apply_logged_env_cfg(cfg, {"weights": [9.0, 8.0]})

    assert cfg.weights == [9.0, 8.0]


def test_play_py_uses_this_module_rather_than_its_own_copy():
    """추출한 뒤에도 play.py 가 사본을 들고 있으면 두 벌이 다시 갈린다."""
    play = Path(__file__).resolve().parents[2] / "reinforcement_learning/rl_games/play.py"
    if not play.is_file():
        pytest.skip(f"play.py 없음: {play}")
    source = play.read_text()

    assert "run_cfg_restore" in source, "play.py 가 이 모듈을 쓰지 않는다"
    assert "def _apply_logged_env_cfg" not in source, "play.py 에 사본이 남아 있다"
    assert "def _rebase_logged_paths" not in source, "play.py 에 사본이 남아 있다"

"""visual_dr: 텍스처 뱅크 계약 (pxr/omni 없이 도는 부분만)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "visual_dr.py"


def _load():
    # visual_dr 은 pxr/omni 를 함수 안에서만 import 한다 → 모듈 로드에는 numpy 만 필요
    pkg = sys.modules.get("openarm.distillation")
    if pkg is None:
        openarm = types.ModuleType("openarm")
        openarm.__path__ = [str(_SRC.parents[1])]
        pkg = types.ModuleType("openarm.distillation")
        pkg.__path__ = [str(_SRC.parent)]
        sys.modules.setdefault("openarm", openarm)
        sys.modules.setdefault("openarm.distillation", pkg)

    spec = importlib.util.spec_from_file_location("_visual_dr_under_test", _SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


visual_dr = _load()


def test_texture_bank_raises_when_textures_missing(tmp_path):
    # 조용히 빈 리스트로 넘어가면 랜덤화가 no-op 이 되고, 학습은 멀쩡히 도는데
    # sim2real 에서만 무너진다 → 반드시 여기서 터져야 한다
    with pytest.raises(FileNotFoundError, match="텍스처가 없다"):
        visual_dr.TextureBank(str(tmp_path))


def test_texture_bank_error_names_every_missing_set(tmp_path):
    (tmp_path / "curated_table_textures").mkdir()
    (tmp_path / "curated_table_textures" / "wood.png").touch()

    with pytest.raises(FileNotFoundError) as excinfo:
        visual_dr.TextureBank(str(tmp_path))

    message = str(excinfo.value)
    assert "curated_table_textures" not in message   # 이건 채워졌다
    assert "dome_light_textures" in message
    assert "object_textures" in message


def test_texture_bank_collects_all_three_sets(tmp_path):
    for sub, name in (
        ("curated_table_textures", "wood.png"),
        ("dome_light_textures", "sky.exr"),
        ("object_textures", "metal.png"),
    ):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / name).touch()

    bank = visual_dr.TextureBank(str(tmp_path))

    assert len(bank.table) == 1
    assert len(bank.dome) == 1
    assert len(bank.object) == 1


def test_texture_bank_finds_object_textures_recursively(tmp_path):
    # object_textures 는 metropolis 에셋이라 하위 디렉토리로 깊게 중첩된다
    for sub in ("curated_table_textures", "dome_light_textures"):
        (tmp_path / sub).mkdir()
    (tmp_path / "curated_table_textures" / "wood.png").touch()
    (tmp_path / "dome_light_textures" / "sky.exr").touch()
    nested = tmp_path / "object_textures" / "wood" / "oak"
    nested.mkdir(parents=True)
    (nested / "oak_basecolor.png").touch()

    bank = visual_dr.TextureBank(str(tmp_path))

    assert len(bank.object) == 1


def test_dome_light_randomization_probability_matches_dextrah():
    assert visual_dr.DOME_LIGHT_RAND_PROB == 0.3

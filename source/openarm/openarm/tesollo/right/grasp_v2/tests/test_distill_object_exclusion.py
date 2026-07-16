"""distillation 실패물체 제외 매핑 검증 (kept_object_names_and_indices).

onehot 은 원본 물체군(153) 차원을 유지해야 teacher 체크포인트와 호환된다.
그래서 스폰·배정은 kept 로만 하되 object_idx 는 원본 슬롯 인덱스로 remap 한다.
이 매핑이 틀리면 (a) teacher onehot 슬롯이 어긋나 잘못된 물체로 조건화되거나
(b) 스폰 물체와 onehot 이 불일치한다. 좌우 헬퍼가 동일한지도 고정한다.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1]
LEFT_PKG = PKG.parents[1] / "left" / "grasp_v2"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


r_utils = _load("_excl_r_utils", PKG / "grasp_right_utils.py")
l_utils = _load("_excl_l_utils", LEFT_PKG / "grasp_left_utils.py")

NAMES = [f"obj_{i}" for i in range(10)]


def test_kept_excludes_and_preserves_order():
    excluded = ("obj_2", "obj_5")
    kept, orig = r_utils.kept_object_names_and_indices(NAMES, excluded)
    assert kept == ["obj_0", "obj_1", "obj_3", "obj_4", "obj_6", "obj_7", "obj_8", "obj_9"]
    # orig 는 kept 각 이름의 원본 인덱스 → onehot 슬롯. 제외된 2,5 는 절대 안 나온다.
    assert orig == [0, 1, 3, 4, 6, 7, 8, 9]
    assert 2 not in orig and 5 not in orig


def test_orig_maps_kept_to_original_slot():
    excluded = ("obj_0", "obj_9")
    kept, orig = r_utils.kept_object_names_and_indices(NAMES, excluded)
    # orig[i] 는 kept[i] 의 원본 인덱스여야 스폰 물체와 onehot 슬롯이 일치한다.
    for i, name in enumerate(kept):
        assert orig[i] == NAMES.index(name)


def test_empty_exclusion_is_identity():
    kept, orig = r_utils.kept_object_names_and_indices(NAMES, ())
    assert kept == NAMES
    assert orig == list(range(len(NAMES)))


def test_unknown_name_fails_loud():
    # 오타로 제외가 조용히 무력화되면 안 된다 → 즉시 실패.
    with pytest.raises(KeyError):
        r_utils.kept_object_names_and_indices(NAMES, ("obj_2", "typo_obj"))


def test_excluding_all_fails():
    with pytest.raises(ValueError):
        r_utils.kept_object_names_and_indices(["a", "b"], ("a", "b"))


def test_left_right_helpers_agree():
    excluded = ("obj_1", "obj_7")
    assert r_utils.kept_object_names_and_indices(
        NAMES, excluded
    ) == l_utils.kept_object_names_and_indices(NAMES, excluded)

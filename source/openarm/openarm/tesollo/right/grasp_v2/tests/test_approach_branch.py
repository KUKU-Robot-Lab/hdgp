"""접근 자세 분기(안 1): 회전 후 물체 높이로 top-down/side 를 가르는 규칙 검증.

이 규칙이 틀리면 접근 자세가 통째로 뒤집혀 기존 성공 물체까지 무너지므로,
경계 조건과 회전 반영을 명시적으로 고정한다.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

PKG = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[7]   # …/hdgp

MODULE_PATH = PKG / "grasp_right_utils.py"
SPEC = importlib.util.spec_from_file_location("grasp_right_utils_branch", MODULE_PATH)
grasp_right_utils = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = grasp_right_utils
SPEC.loader.exec_module(grasp_right_utils)

compute_flat_object_mask = grasp_right_utils.compute_flat_object_mask

THRESHOLD = 0.05  # grasp_right_env_cfg.flat_object_height_threshold 와 동일


def _identity(n: int) -> torch.Tensor:
    return torch.eye(3).unsqueeze(0).repeat(n, 1, 1)


def _rot_x_90(n: int) -> torch.Tensor:
    """x축 90° 회전 — 세워진 물체를 눕힌다(로컬 z 가 world y 로 간다)."""
    c, s = math.cos(math.pi / 2), math.sin(math.pi / 2)
    r = torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    return r.unsqueeze(0).repeat(n, 1, 1)


def test_flat_object_is_topdown_and_tall_object_is_side() -> None:
    # small_8_cyl(h 4cm, 실패) vs small_5_cyl(h 6cm, 성공) — 실측 bbox
    half = torch.tensor([
        [0.020, 0.020, 0.020],   # 4×4×4 cm  → 높이 4cm  < 5cm  → top-down
        [0.0125, 0.0125, 0.030],  # 2.5×2.5×6 cm → 높이 6cm > 5cm  → side
    ])

    mask = compute_flat_object_mask(half, _identity(2), THRESHOLD)

    assert bool(mask[0]) is True
    assert bool(mask[1]) is False


def test_upright_cylinder_becomes_flat_when_toppled() -> None:
    # small_5_cyl: 세워지면 높이 6cm(side), x축 90° 로 누우면 지름 2.5cm(top-down)
    half = torch.tensor([[0.0125, 0.0125, 0.030]])

    upright = compute_flat_object_mask(half, _identity(1), THRESHOLD)
    toppled = compute_flat_object_mask(half, _rot_x_90(1), THRESHOLD)

    assert bool(upright[0]) is False
    assert bool(toppled[0]) is True


def test_threshold_is_exclusive_at_boundary() -> None:
    # 높이가 정확히 임계와 같으면 "납작"이 아니다(< 비교) — 경계 물체의 자세가
    # 부동소수 오차로 흔들리지 않도록 규칙을 고정한다.
    half = torch.tensor([[0.02, 0.02, THRESHOLD / 2.0]])

    mask = compute_flat_object_mask(half, _identity(1), THRESHOLD)

    assert bool(mask[0]) is False


def test_bbox_table_covers_every_active_object() -> None:
    # bbox 누락은 env 생성 시 KeyError 로 죽는다 — 자산과 테이블의 동기화를 지킨다.
    bbox_path = REPO / "assets" / "object_bbox.json"
    usd_root = REPO / "assets" / "visdex_objects" / "USD"
    assert bbox_path.is_file(), "compute_object_bbox.py 로 bbox 를 먼저 생성하세요"

    table = json.loads(bbox_path.read_text(encoding="utf-8"))
    active = [
        d.name for d in usd_root.iterdir()
        if d.is_dir() and (d / f"{d.name}.usd").is_file()
    ]

    missing = [n for n in active if n not in table]
    assert not missing, f"bbox 누락: {missing}"
    assert all(len(table[n]) == 3 for n in active)


def test_known_failing_objects_are_all_routed_topdown() -> None:
    # 클린 평가에서 무너진 물체들(h ≤ 4cm)이 실제로 top-down 으로 분기되는지 —
    # 이 실험의 목적 자체를 고정하는 테스트.
    table = json.loads((REPO / "assets" / "object_bbox.json").read_text(encoding="utf-8"))
    failing = ["small_8_cyl", "small_8_cuboid", "small_12_cyl", "small_12_cuboid"]

    half = torch.tensor([[float(v) for v in table[n]] for n in failing])
    mask = compute_flat_object_mask(half, _identity(len(failing)), THRESHOLD)

    assert bool(mask.all()), f"top-down 분기 실패: {[failing[i] for i, m in enumerate(mask) if not m]}"


def test_successful_tall_objects_stay_side() -> None:
    # 잘 잡히던 물체들은 side 를 유지해야 한다(회귀 차단).
    table = json.loads((REPO / "assets" / "object_bbox.json").read_text(encoding="utf-8"))
    tall = ["large_5_cyl", "large_8_cyl", "small_5_cyl", "cup"]

    half = torch.tensor([[float(v) for v in table[n]] for n in tall])
    mask = compute_flat_object_mask(half, _identity(len(tall)), THRESHOLD)

    assert not bool(mask.any()), f"side 유지 실패: {[tall[i] for i, m in enumerate(mask) if m]}"

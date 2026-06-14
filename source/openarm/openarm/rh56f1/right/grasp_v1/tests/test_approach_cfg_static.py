from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]


def test_cfg_declares_rh56f1_adapted_pregrasp_and_approach_weights() -> None:
    cfg = (_ROOT / "grasp_right_env_cfg.py").read_text(encoding="utf-8")

    assert "실제 RH56F1/cup 기하 기준" in cfg
    assert "palm sensor는 palm_link 기준 (0.00, 0.03, 0.04)" in cfg
    assert "cup 반경은 약 0.035m" in cfg
    assert "thumb_1 루트가 palm sensor보다도 +x 방향으로 더 앞으로 나온다" in cfg
    assert "top3 fingertip shell error가 컵 반경보다 약 2cm 멀어서" in cfg
    assert "pregrasp_offset_x:     float = -0.045" in cfg
    assert "pregrasp_offset_y:     float = -0.055" in cfg
    assert "pregrasp_offset_z:     float = 0.015" in cfg
    assert "palm_delta_xyz:     float = 0.03" in cfg
    assert "ema_action_alpha: float = 0.7" in cfg
    assert "grasp_palm_delta_scale: float = 1.0" in cfg
    assert "grasp_palm_inward_offset: float = 0.025" in cfg
    assert "approach_weight: float = 2.0" in cfg
    assert "approach_sharpness: float = 8.0" in cfg
    assert "grasp_weight: float = 12.0" in cfg
    assert "stabilize_weight: float = 10.0" in cfg
    assert "success_bonus_weight: float = 20.0" in cfg
    assert "enclosure_weight:       float = 3.0" in cfg
    assert "enclosure_sharpness:    float = 15.0" in cfg

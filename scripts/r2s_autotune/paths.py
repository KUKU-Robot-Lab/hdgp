"""hdgp 루트 기준 경로 해석.

절대경로를 박아두면 server(다른 홈 디렉토리)에서 그대로 깨진다.
config의 경로는 hdgp 루트 기준 상대경로로 쓰고 여기서 해석한다.
"""

from __future__ import annotations

from pathlib import Path

# scripts/r2s_autotune/paths.py → hdgp/
HDGP_ROOT = Path(__file__).resolve().parents[2]


def resolve_hdgp_path(path: str | Path) -> Path:
    """상대경로는 hdgp 루트 기준으로, 절대경로는 그대로 돌려준다."""
    path = Path(path)
    return path if path.is_absolute() else (HDGP_ROOT / path)


def asset_dir(asset: str) -> Path:
    return HDGP_ROOT / "assets" / "robot" / asset


def asset_usd(asset: str) -> Path:
    return asset_dir(asset) / f"{asset}.usd"


def asset_manifest(asset: str) -> Path:
    """USD 옆의 manifest를 쓴다.

    urdf repo의 원본(`urdf/generated/rl/`)이 아니라 이 사본을 보는 이유는,
    관절 이름 계약이 '실제로 스폰되는 USD'와 맞아야 의미가 있기 때문이다.
    둘이 어긋나면 그건 asset 갱신 누락이지 autotune이 흡수할 문제가 아니다.
    """
    return asset_dir(asset) / f"{asset}_manifest.yaml"

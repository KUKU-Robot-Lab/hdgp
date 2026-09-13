"""Draw IKER keypoint markers and the axis legend (design spec §5-6)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

MARKER_RADIUS_PX = 11
LABEL_FONT_PX = 14
LEGEND_ARROW_PX = 38
LEGEND_HEAD_PX = 8
LEGEND_INSET_PX = 50
LABEL_COLOR = (255, 255, 255)
STATIC_COLOR = (70, 130, 255)
OCCLUDED_COLOR = (150, 150, 150)
LEGEND_COLOR = (255, 255, 0)
MOVABLE_COLORS = ((230, 60, 60), (60, 200, 90), (240, 170, 40), (190, 80, 220))


@dataclass(frozen=True)
class Marker:
    label: int
    uv: tuple[float, float]
    color: tuple[int, int, int]
    visible: bool


def movable_color(order_index: int) -> tuple[int, int, int]:
    return MOVABLE_COLORS[order_index % len(MOVABLE_COLORS)]


def rotate_image(rgb, theta_deg: float) -> Image.Image:
    """Rotate counter-clockwise about the centre, keeping the input size (black fill)."""
    pixels = np.asarray(rgb)
    if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
        raise ValueError("rgb must be an (H, W, 3) uint8 array")
    if not math.isfinite(theta_deg):
        raise ValueError("theta_deg must be finite")
    return Image.fromarray(pixels).rotate(
        theta_deg, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0)
    )


def draw_markers(image: Image.Image, markers: Sequence[Marker], axes: Mapping[str, tuple[float, float]]) -> Image.Image:
    """A copy of ``image`` with numbered markers and a bottom-right axis legend."""
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default(size=LABEL_FONT_PX)
    for marker in markers:
        u, v = marker.uv
        if not (math.isfinite(u) and math.isfinite(v)):
            raise ValueError(f"marker {marker.label} has non-finite coordinates")
        box = [u - MARKER_RADIUS_PX, v - MARKER_RADIUS_PX, u + MARKER_RADIUS_PX, v + MARKER_RADIUS_PX]
        if marker.visible:
            draw.ellipse(box, fill=marker.color, outline=LABEL_COLOR)
        else:
            draw.ellipse(box, outline=OCCLUDED_COLOR, width=2)
        draw.text((u, v), str(marker.label), fill=LABEL_COLOR, font=font, anchor="mm")
    _draw_legend(draw, out.size, axes, font)
    return out


def _draw_legend(draw: ImageDraw.ImageDraw, size: tuple[int, int], axes, font) -> None:
    width, height = size
    ox, oy = width - LEGEND_INSET_PX, height - LEGEND_INSET_PX
    for name, (du, dv) in axes.items():
        norm = math.hypot(du, dv)
        if not norm > 1e-9:
            raise ValueError(f"legend axis {name} has zero length")
        ux, uy = du / norm, dv / norm
        tip = (ox + LEGEND_ARROW_PX * ux, oy + LEGEND_ARROW_PX * uy)
        draw.line([(ox, oy), tip], fill=LEGEND_COLOR, width=3)
        back = (tip[0] - LEGEND_HEAD_PX * ux, tip[1] - LEGEND_HEAD_PX * uy)
        side = (-uy * LEGEND_HEAD_PX / 2.0, ux * LEGEND_HEAD_PX / 2.0)
        draw.polygon(
            [tip, (back[0] + side[0], back[1] + side[1]), (back[0] - side[0], back[1] - side[1])], fill=LEGEND_COLOR
        )
        label_at = (ox + (LEGEND_ARROW_PX + 12) * ux, oy + (LEGEND_ARROW_PX + 12) * uy)
        draw.text(label_at, name, fill=LEGEND_COLOR, font=font, anchor="mm")

"""Render English text over the original art.

No inpainting. Each bubble gets a *translucent* rounded box so the artwork
stays visible underneath, with fully-opaque text (white halo for legibility)
drawn on top. Compositing is done once per page via an RGBA overlay.
"""

from __future__ import annotations

import functools
import os
from typing import Iterable, List, Tuple

from PIL import Image, ImageDraw, ImageFont

Box = Tuple[int, int, int, int]

_FONT_CANDIDATES = [
    # Linux (origin machine)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    # macOS (modern paths under /System/Library/Fonts/Supplemental)
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


@functools.lru_cache(maxsize=256)
def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: float) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    cur = words[0]
    for word in words[1:]:
        trial = f"{cur} {word}"
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    return lines


def _block_size(draw, lines, font) -> Tuple[float, float]:
    widths = [draw.textlength(ln, font=font) for ln in lines] or [0]
    ascent, descent = font.getmetrics()
    return max(widths), (ascent + descent) * len(lines)


def _fit(draw, box: Box, text: str, min_font: int, max_font: int, pad: int):
    x1, y1, x2, y2 = box
    inner_w = max(1, (x2 - x1) - 2 * pad)
    inner_h = max(1, (y2 - y1) - 2 * pad)
    for size in range(max_font, min_font - 1, -1):
        font = _font(size)
        lines = _wrap(draw, text, font, inner_w)
        block_w, block_h = _block_size(draw, lines, font)
        if block_w <= inner_w and block_h <= inner_h:
            return font, lines
    return _font(min_font), _wrap(draw, text, _font(min_font), inner_w)


def render_page(
    image: Image.Image,
    items: Iterable[Tuple[Box, str]],
    *,
    bubble_alpha: int = 150,
    min_font: int = 9,
    max_font: int = 48,
    pad: int = 6,
) -> Image.Image:
    """Return a new RGB image with each (box, text) drawn as a translucent
    rounded box plus opaque, haloed text. `bubble_alpha` is 0 (invisible) to
    255 (fully opaque)."""
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    outline_alpha = min(255, bubble_alpha + 90)
    for box, text in items:
        if not text.strip():
            continue
        x1, y1, x2, y2 = box
        font, lines = _fit(draw, box, text, min_font, max_font, pad)

        # Optional translucent box. With bubble_alpha == 0 we draw no box at all
        # and rely on the white text halo for legibility over the art.
        if bubble_alpha > 0:
            radius = max(4, min((x2 - x1), (y2 - y1)) // 8)
            draw.rounded_rectangle(
                box,
                radius=radius,
                fill=(255, 255, 255, bubble_alpha),
                outline=(40, 40, 40, outline_alpha),
                width=2,
            )

        ascent, descent = font.getmetrics()
        line_h = ascent + descent
        _, block_h = _block_size(draw, lines, font)
        cy = y1 + ((y2 - y1) - block_h) / 2
        cx = (x1 + x2) / 2
        # Thicker halo when there's no box, since it carries legibility alone.
        stroke = max(2, font.size // 7) if bubble_alpha == 0 else max(1, font.size // 12)
        for line in lines:
            w = draw.textlength(line, font=font)
            draw.text(
                (cx - w / 2, cy),
                line,
                font=font,
                fill=(0, 0, 0, 255),           # text fully opaque...
                stroke_width=stroke,
                stroke_fill=(255, 255, 255, 255),  # ...with a white halo
            )
            cy += line_h

    return Image.alpha_composite(base, overlay).convert("RGB")

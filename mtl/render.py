"""Render English text into an opaque box laid over the original art.

No inpainting: we draw a filled rounded rectangle (white, thin dark outline)
over each bubble and typeset the wrapped English to fit. Surrounding Japanese
art is left untouched for atmosphere.
"""

from __future__ import annotations

import functools
import os
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont

Box = Tuple[int, int, int, int]

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]


@functools.lru_cache(maxsize=256)
def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    # last resort: PIL's bundled bitmap font (fixed size, ignores `size`)
    return ImageFont.load_default()


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: float) -> List[str]:
    """Greedy word wrap to a pixel width."""
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
    line_h = ascent + descent
    return max(widths), line_h * len(lines)


def draw_block(
    image: Image.Image,
    box: Box,
    text: str,
    *,
    min_font: int = 9,
    max_font: int = 48,
    pad: int = 6,
) -> None:
    """Fill `box` with an opaque rounded rectangle and centered, wrapped text,
    shrinking the font until the text fits."""
    if not text.strip():
        return

    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = box
    inner_w = max(1, (x2 - x1) - 2 * pad)
    inner_h = max(1, (y2 - y1) - 2 * pad)

    chosen_size = min_font
    chosen_lines: List[str] = [text]
    for size in range(max_font, min_font - 1, -1):
        font = _font(size)
        lines = _wrap(draw, text, font, inner_w)
        block_w, block_h = _block_size(draw, lines, font)
        if block_w <= inner_w and block_h <= inner_h:
            chosen_size, chosen_lines = size, lines
            break
    else:
        chosen_size = min_font
        chosen_lines = _wrap(draw, text, _font(min_font), inner_w)

    font = _font(chosen_size)
    radius = max(4, min((x2 - x1), (y2 - y1)) // 8)
    draw.rounded_rectangle(
        box, radius=radius, fill=(255, 255, 255), outline=(40, 40, 40), width=2
    )

    _, block_h = _block_size(draw, chosen_lines, font)
    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    cy = y1 + ((y2 - y1) - block_h) / 2
    cx = (x1 + x2) / 2
    for line in chosen_lines:
        w = draw.textlength(line, font=font)
        draw.text((cx - w / 2, cy), line, font=font, fill=(0, 0, 0))
        cy += line_h

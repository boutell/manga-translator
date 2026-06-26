"""End-to-end page translation: detect -> cluster -> OCR -> translate -> render."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from PIL import Image

from . import detect, cluster, ocr, render
from .translate import Translator

Box = Tuple[int, int, int, int]


@dataclass
class Region:
    box: Box
    japanese: str = ""
    english: str = ""


@dataclass
class PageResult:
    image: Image.Image
    regions: List[Region] = field(default_factory=list)


def _scaled_for_detection(image: Image.Image, max_side: int) -> Tuple[np.ndarray, float]:
    """Downscale a copy for detection (CRAFT is slow on huge phone photos).
    Returns the array plus the scale factor to map boxes back to full res."""
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return np.asarray(image), 1.0
    scale = max_side / longest
    small = image.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    return np.asarray(small), scale


def translate_page(
    image: Image.Image,
    translator: Translator,
    *,
    detect_max_side: int = 1600,
    min_box_area: int = 200,
    max_box_area_frac: float = 0.10,
    bubble_alpha: int = 0,
) -> PageResult:
    image = image.convert("RGB")
    page_area = image.width * image.height
    max_box_area = page_area * max_box_area_frac
    det_array, scale = _scaled_for_detection(image, detect_max_side)

    line_boxes = detect.detect_boxes(det_array)
    if scale != 1.0:
        inv = 1.0 / scale
        line_boxes = [
            (round(x1 * inv), round(y1 * inv), round(x2 * inv), round(y2 * inv))
            for (x1, y1, x2, y2) in line_boxes
        ]

    bubbles = cluster.cluster_boxes(line_boxes)

    # OCR every bubble first, then translate the whole page in one batch so an
    # LLM backend can use cross-bubble context (and we pay model overhead once).
    regions: List[Region] = []
    for box in bubbles:
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        # Skip too-small noise and implausibly large merges (usually facing-page
        # bleed-in on two-page photos clustered into one blob).
        if area < min_box_area or area > max_box_area:
            continue
        crop = image.crop(box)
        japanese = ocr.read(crop)
        if not japanese:
            continue
        regions.append(Region(box=box, japanese=japanese))

    englishes = translator.translate_batch([r.japanese for r in regions])
    for region, english in zip(regions, englishes):
        region.english = english

    out = render.render_page(
        image,
        [(r.box, r.english or r.japanese) for r in regions],
        bubble_alpha=bubble_alpha,
    )
    return PageResult(image=out, regions=regions)

"""Text-region detection via EasyOCR's CRAFT detector.

We use EasyOCR purely as a detector (its `.detect()` call); the actual reading
is done by manga-ocr, which is far stronger on stylized/vertical manga text.
"""

from __future__ import annotations

import functools
from typing import List, Tuple

import numpy as np

from .device import get_device

Box = Tuple[int, int, int, int]  # (x1, y1, x2, y2)


@functools.lru_cache(maxsize=1)
def _reader():
    import easyocr

    return easyocr.Reader(["ja"], gpu=(get_device() == "cuda"))


def detect_boxes(image: np.ndarray) -> List[Box]:
    """Return axis-aligned text-line boxes for an RGB image array."""
    reader = _reader()
    # EasyOCR returns (horizontal_list, free_list), one entry per image.
    horizontal_list, free_list = reader.detect(
        image,
        # tuned a little looser than defaults so we catch sparse manga lettering
        text_threshold=0.6,
        low_text=0.3,
        link_threshold=0.3,
    )
    boxes: List[Box] = []
    for x_min, x_max, y_min, y_max in horizontal_list[0]:
        boxes.append((int(x_min), int(y_min), int(x_max), int(y_max)))
    for poly in free_list[0]:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        boxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
    return boxes

"""Japanese OCR via manga-ocr (the gold-standard manga text reader).

manga-ocr automatically uses the GPU when torch sees one. It expects a single
text region (one bubble) per call and handles vertical, multi-line text.
"""

from __future__ import annotations

import functools

from PIL import Image

from .device import get_device


@functools.lru_cache(maxsize=1)
def _model():
    from manga_ocr import MangaOcr

    return MangaOcr(force_cpu=(get_device() != "cuda"))


def read(crop: Image.Image) -> str:
    text = _model()(crop).strip()
    return text

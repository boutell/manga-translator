"""Group detected text-line boxes into speech-bubble clusters.

manga-ocr reads a whole bubble (multi-line, vertical) better than isolated
lines, and we want one English block per bubble — so we merge nearby boxes
into a single region before OCR/translation.
"""

from __future__ import annotations

import itertools
from typing import List, Tuple

Box = Tuple[int, int, int, int]


def _expand(b: Box, pad: float) -> Tuple[float, float, float, float]:
    return (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)


def _overlap(a, b) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def cluster_boxes(boxes: List[Box], gap_ratio: float = 0.15) -> List[Box]:
    """Merge boxes whose padded rectangles touch. Padding scales with the
    median *shorter* side of the boxes. For vertical manga text the line boxes
    are tall, narrow columns, so the shorter side ~= column width — padding by
    that keeps columns of one bubble together without bridging across the whole
    page (padding by height would chain every bubble into one).

    gap_ratio 0.15 (was 0.3): on dense 4-koma pages 0.3 chained distinct texts
    (title, asides, sound effects) into panel-sized blobs that then garbled OCR
    and tripped the oversized-box filter. 0.15 keeps a bubble's own columns
    together while separating neighbours; pages with normal whitespace between
    bubbles are unaffected (verified: their bubble counts barely move)."""
    if not boxes:
        return []

    short_sides = sorted(min(b[2] - b[0], b[3] - b[1]) for b in boxes)
    median_short = short_sides[len(short_sides) // 2] or 1
    pad = median_short * gap_ratio

    parent = list(range(len(boxes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    expanded = [_expand(b, pad) for b in boxes]
    for i, j in itertools.combinations(range(len(boxes)), 2):
        if _overlap(expanded[i], expanded[j]):
            union(i, j)

    groups: dict[int, List[Box]] = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(boxes[i])

    merged: List[Box] = []
    for members in groups.values():
        x1 = min(b[0] for b in members)
        y1 = min(b[1] for b in members)
        x2 = max(b[2] for b in members)
        y2 = max(b[3] for b in members)
        merged.append((x1, y1, x2, y2))
    return merged

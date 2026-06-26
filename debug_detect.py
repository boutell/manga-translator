#!/usr/bin/env python3
"""Visualize raw detection vs clustering at several settings.

Draws raw line boxes (red) and clustered bubbles (green) so we can see what the
detector actually catches on these phone photos. Saves *_debug.jpg per config.
"""
import sys
from itertools import product

import numpy as np
from PIL import Image, ImageDraw

from mtl import cluster
from mtl.detect import _reader

img_path = sys.argv[1] if len(sys.argv) > 1 else "input/PXL_20260612_213545116.jpg"
image = Image.open(img_path).convert("RGB")
W, H = image.size
print(f"image: {W}x{H}")

reader = _reader()

for max_side, text_th, low_text in product([1600, 2400, 3200], [0.5, 0.3], [0.3, 0.2]):
    longest = max(W, H)
    scale = min(1.0, max_side / longest)
    work = image if scale == 1.0 else image.resize((round(W * scale), round(H * scale)), Image.LANCZOS)
    arr = np.asarray(work)
    h_list, f_list = reader.detect(arr, text_threshold=text_th, low_text=low_text, link_threshold=0.3)
    boxes = []
    for x1, x2, y1, y2 in h_list[0]:
        boxes.append((int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)))
    for poly in f_list[0]:
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        boxes.append((int(min(xs) / scale), int(min(ys) / scale), int(max(xs) / scale), int(max(ys) / scale)))
    bubbles = cluster.cluster_boxes(boxes)
    print(f"max_side={max_side} text_th={text_th} low_text={low_text}: {len(boxes)} lines -> {len(bubbles)} bubbles")

    vis = image.copy()
    d = ImageDraw.Draw(vis)
    for b in boxes:
        d.rectangle(b, outline=(255, 0, 0), width=2)
    for b in bubbles:
        d.rectangle(b, outline=(0, 200, 0), width=3)
    out = f"/tmp/mt_smoke_out/debug_ms{max_side}_t{text_th}_l{low_text}.jpg"
    vis.save(out)

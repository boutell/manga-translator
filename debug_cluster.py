#!/usr/bin/env python3
"""Detection runs ONCE; then sweep clustering tightness so we can see bubbles
separate. Saves *_cluster.jpg per gap_ratio (green = clustered bubbles)."""
import sys

import numpy as np
from PIL import Image, ImageDraw

from mtl import cluster
from mtl.detect import _reader

img_path = sys.argv[1] if len(sys.argv) > 1 else "input/PXL_20260612_213545116.jpg"
image = Image.open(img_path).convert("RGB")
W, H = image.size

max_side = 2400
scale = min(1.0, max_side / max(W, H))
work = image.resize((round(W * scale), round(H * scale)), Image.LANCZOS)
h_list, f_list = _reader().detect(np.asarray(work), text_threshold=0.5, low_text=0.3, link_threshold=0.3)
boxes = []
for x1, x2, y1, y2 in h_list[0]:
    boxes.append((int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)))
for poly in f_list[0]:
    xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
    boxes.append((int(min(xs) / scale), int(min(ys) / scale), int(max(xs) / scale), int(max(ys) / scale)))
print(f"{len(boxes)} raw line boxes")

for gap in [0.3, 0.5, 0.7, 1.0]:
    bubbles = cluster.cluster_boxes(boxes, gap_ratio=gap)
    print(f"gap_ratio={gap}: {len(bubbles)} bubbles")
    vis = image.copy()
    d = ImageDraw.Draw(vis)
    for b in boxes:
        d.rectangle(b, outline=(255, 0, 0), width=1)
    for b in bubbles:
        d.rectangle(b, outline=(0, 180, 0), width=4)
    vis.save(f"/tmp/mt_smoke_out/cluster_gap{gap}.jpg")

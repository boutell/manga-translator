#!/usr/bin/env python3
"""Hybrid step 1: local CRAFT detect+cluster -> numbered boxes per page.

Run from the project root (NO venv activation needed — use the venv python):
    .venv/bin/python hybrid_boxes.py [input_dir] [work_dir]

For each page it writes:
    <work>/boxed_<name>   : the page with numbered red boxes (for the translator
                            — Claude — to read and key translations to)
    <work>/boxes.json     : {page_name: {id: [x1,y1,x2,y2]}} in full-res pixels

Boxes are numbered in rough manga reading order (top->bottom bands, right->left).
This is the geometry half of the hybrid: the Mac finds tight boxes locally; a
strong translator supplies the idiom. See hybrid_render.py for the render half.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mtl import cluster, detect

MIN_AREA = 200
MAX_AREA_FRAC = 0.10  # drop facing-page bleed (matches pipeline.py)


def _font(size: int):
    for p in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _scaled_for_detection(image: Image.Image, max_side: int = 2000):
    """Downscale a copy for detection; return (array, scale_to_map_back)."""
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return np.asarray(image), 1.0
    scale = max_side / longest
    small = image.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    return np.asarray(small), scale


def page_boxes(im: Image.Image):
    max_area = im.width * im.height * MAX_AREA_FRAC
    arr, scale = _scaled_for_detection(im)
    lines = detect.detect_boxes(arr)
    if scale != 1.0:
        inv = 1.0 / scale
        lines = [(round(x1 * inv), round(y1 * inv), round(x2 * inv), round(y2 * inv))
                 for x1, y1, x2, y2 in lines]
    return [b for b in cluster.cluster_boxes(lines)
            if MIN_AREA <= (b[2] - b[0]) * (b[3] - b[1]) <= max_area]


def manga_sort(boxes, H):
    band = max(1, H // 14)
    return sorted(boxes, key=lambda b: (b[1] // band, -b[2]))


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "input"
    work_dir = sys.argv[2] if len(sys.argv) > 2 else "work"
    os.makedirs(work_dir, exist_ok=True)
    font = _font(64)
    all_boxes = {}
    for path in sorted(glob.glob(os.path.join(input_dir, "*.jpg"))):
        im = Image.open(path).convert("RGB")
        boxes = manga_sort(page_boxes(im), im.height)
        vis = im.copy()
        d = ImageDraw.Draw(vis)
        name = os.path.basename(path)
        page = {}
        for i, b in enumerate(boxes):
            d.rectangle(b, outline=(220, 0, 0), width=6)
            tb = d.textbbox((b[0] + 6, b[1] + 6), str(i), font=font)
            d.rectangle([tb[0] - 5, tb[1] - 3, tb[2] + 5, tb[3] + 3], fill=(220, 0, 0))
            d.text((b[0] + 6, b[1] + 6), str(i), fill=(255, 255, 255), font=font)
            page[i] = list(b)
        vis.save(os.path.join(work_dir, f"boxed_{name}"))
        all_boxes[name] = page
        print(f"{name}: {len(boxes)} boxes")
    json.dump(all_boxes, open(os.path.join(work_dir, "boxes.json"), "w"))
    print(f"-> {work_dir}/boxes.json and boxed_*.jpg")


if __name__ == "__main__":
    main()

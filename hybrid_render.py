#!/usr/bin/env python3
"""Hybrid step 3: render the translator's text into the local CRAFT boxes.

Run from the project root:
    .venv/bin/python hybrid_render.py [work_dir] [input_dir] [out_dir]

Reads:
    <work>/boxes.json         : {page: {id: [x1,y1,x2,y2]}}   (from hybrid_boxes.py)
    <work>/translations.json  : {page: {id: "english"}}       (written by the translator)
Writes <out>/<page> with the English composited over the art (white halo, no box,
Japanese showing through — the project's no-inpainting house style). Empty/missing
ids are skipped, so noise boxes can just be left blank in translations.json.
"""
from __future__ import annotations

import json
import os
import sys

from PIL import Image

from mtl.render import render_page


def main():
    work = sys.argv[1] if len(sys.argv) > 1 else "work"
    input_dir = sys.argv[2] if len(sys.argv) > 2 else "input"
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "output"
    os.makedirs(out_dir, exist_ok=True)

    boxes = json.load(open(os.path.join(work, "boxes.json")))
    trans = json.load(open(os.path.join(work, "translations.json")))

    for name, page_boxes in boxes.items():
        if name not in trans:
            print(f"{name}: no translations, skipped")
            continue
        im = Image.open(os.path.join(input_dir, name)).convert("RGB")
        page_tr = trans[name]
        items = []
        for bid, box in page_boxes.items():
            txt = page_tr.get(str(bid), "")
            if txt and txt.strip():
                items.append((tuple(box), txt))
        render_page(im, items, bubble_alpha=0).save(os.path.join(out_dir, name))
        print(f"{name}: rendered {len(items)} regions -> {out_dir}/{name}")


if __name__ == "__main__":
    main()

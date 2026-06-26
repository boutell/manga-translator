#!/usr/bin/env python3
"""Translate manga pages from Japanese to English, in place on the page.

Usage:
    python run.py                       # input/ -> output/, opus-mt translator
    python run.py --translator null     # skip translation (debug detection)
    python run.py -i pages/ -o out/     # custom folders

Fully self-hosted: models download once, then everything runs offline.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image

from mtl.device import describe_device
from mtl.pipeline import translate_page
from mtl.translate import TRANSLATORS, get_translator

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", default="input", type=Path, help="input folder (default: input)")
    p.add_argument("-o", "--output", default="output", type=Path, help="output folder (default: output)")
    p.add_argument(
        "-t", "--translator", default="opus", choices=sorted(TRANSLATORS),
        help="translation backend (default: opus)",
    )
    p.add_argument(
        "--detect-max-side", type=int, default=2000,
        help="downscale long edge to this many px for detection only (default: 1600)",
    )
    p.add_argument("--dump-text", action="store_true", help="also write a .txt of JA/EN per page")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.input.is_dir():
        print(f"error: input folder not found: {args.input}", file=sys.stderr)
        return 1
    args.output.mkdir(parents=True, exist_ok=True)

    pages = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not pages:
        print(f"error: no images in {args.input}", file=sys.stderr)
        return 1

    print(f"Device: {describe_device()}")
    print(f"Translator: {args.translator}")
    print(f"Loading models (first run downloads them once)...", flush=True)
    translator = get_translator(args.translator)

    print(f"Translating {len(pages)} page(s) -> {args.output}/\n")
    for idx, path in enumerate(pages, 1):
        t0 = time.time()
        image = Image.open(path)
        result = translate_page(image, translator, detect_max_side=args.detect_max_side)
        out_path = args.output / path.name
        result.image.save(out_path)
        dt = time.time() - t0
        print(f"[{idx}/{len(pages)}] {path.name}: {len(result.regions)} bubble(s), {dt:.1f}s -> {out_path.name}")

        if args.dump_text:
            txt = args.output / (path.stem + ".txt")
            with txt.open("w", encoding="utf-8") as fh:
                for r in result.regions:
                    fh.write(f"JA: {r.japanese}\nEN: {r.english}\n\n")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

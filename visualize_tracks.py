#!/usr/bin/env python3
"""Render tracking results (MOT format) over a sequence as an animated GIF.

One track file gives a single panel; two track files give a side-by-side
comparison of two trackers (e.g. a baseline vs. our tracker).

Example:
    python visualize_tracks.py --frames-dir dataset/test/aveo/img1 \\
        --tracks results/baseline/aveo.txt results/ours/aveo.txt \\
        --labels Baseline Ours --start 1 --end 300 --out assets/examples/aveo_compare.gif
"""
from __future__ import annotations

import argparse
import colorsys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames-dir", type=Path, required=True, help="Folder with frames named 000001.jpg, ...")
    p.add_argument("--tracks", type=Path, nargs="+", required=True, help="One or two MOT track files.")
    p.add_argument("--labels", nargs="+", default=None, help="Panel titles (one per track file).")
    p.add_argument("--start", type=int, default=1, help="First frame (1-indexed).")
    p.add_argument("--end", type=int, default=None, help="Last frame (inclusive). Default: last frame.")
    p.add_argument("--step", type=int, default=3, help="Keep one frame every N.")
    p.add_argument("--fps", type=float, default=10.0, help="GIF frame rate.")
    p.add_argument("--width", type=int, default=480, help="Width in pixels of each panel.")
    p.add_argument("--out", type=Path, required=True, help="Output .gif path.")
    p.add_argument("--colors", type=int, default=128, help="GIF palette size (smaller = lighter file).")
    return p.parse_args()


def load_tracks(path: Path) -> dict[int, list[tuple[int, float, float, float, float]]]:
    by_frame: dict[int, list] = defaultdict(list)
    for line in path.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        f, tid = int(float(parts[0])), int(float(parts[1]))
        x, y, w, h = (float(v) for v in parts[2:6])
        by_frame[f].append((tid, x, y, w, h))
    return by_frame


def id_color(tid: int) -> tuple[int, int, int]:
    """Stable, well-separated BGR colour per track id."""
    hue = (tid * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def draw_panel(img: np.ndarray, boxes: list, width: int, title: str | None) -> np.ndarray:
    scale = width / img.shape[1]
    img = cv2.resize(img, (width, int(round(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    for tid, x, y, w, h in boxes:
        x1, y1, x2, y2 = (int(round(v * scale)) for v in (x, y, x + w, y + h))
        color = id_color(tid)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"ID {tid}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(y1, th + 6)
        cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
        cv2.putText(img, text, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    if title:
        band = np.full((28, img.shape[1], 3), 30, np.uint8)
        cv2.putText(band, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        img = np.vstack([band, img])
    return img


def main() -> None:
    args = parse_args()
    if len(args.tracks) > 2:
        raise SystemExit("Pass one or two track files.")
    labels = args.labels or [None] * len(args.tracks)
    if len(labels) != len(args.tracks):
        raise SystemExit("--labels needs one title per track file.")

    tracks = [load_tracks(t) for t in args.tracks]
    frames = sorted(args.frames_dir.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"No .jpg frames in {args.frames_dir}")
    end = args.end or len(frames)

    gif_frames = []
    for idx in range(args.start, end + 1, args.step):
        img = cv2.imread(str(args.frames_dir / f"{idx:06d}.jpg"))
        if img is None:
            continue
        panels = [draw_panel(img.copy(), t.get(idx, []), args.width, lab) for t, lab in zip(tracks, labels)]
        canvas = np.hstack(panels) if len(panels) > 1 else panels[0]
        rgb = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        gif_frames.append(rgb.quantize(colors=args.colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))

    if not gif_frames:
        raise SystemExit("No frames rendered; check --start/--end and --frames-dir.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    gif_frames[0].save(
        args.out, save_all=True, append_images=gif_frames[1:],
        duration=int(round(1000 / args.fps)), loop=0, optimize=True,
    )
    print(f"{args.out}: {len(gif_frames)} frames, {args.out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()

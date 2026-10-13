#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from PIL.Image import Resampling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an animated GIF from sequential images in a folder."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("gif"),
        help="Folder containing input frames.",
    )
    parser.add_argument(
        "--pattern",
        default="*.jpg",
        help="Glob pattern used to match input images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gif/output.gif"),
        help="Output GIF path.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frames per second for the output GIF.",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="GIF loop count. Use 0 for infinite loop.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=540,
        help="Resize frames to this width while keeping aspect ratio. Use 0 to keep original size.",
    )
    parser.add_argument(
        "--colors",
        type=int,
        default=64,
        help="Maximum number of colors in the GIF palette.",
    )
    return parser.parse_args()


def prepare_frame(image: Image.Image, max_width: int, colors: int) -> Image.Image:
    frame = image.convert("RGB")
    if max_width > 0 and frame.width > max_width:
        height = int(round(frame.height * max_width / frame.width))
        frame = frame.resize((max_width, height), Resampling.LANCZOS)
    return frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=colors)


def main() -> None:
    args = parse_args()
    frame_paths = sorted(args.input_dir.glob(args.pattern))
    if not frame_paths:
        raise SystemExit(
            f"No frames found in '{args.input_dir}' matching pattern '{args.pattern}'."
        )

    duration_ms = max(1, int(round(1000 / args.fps)))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            frames.append(prepare_frame(image, args.max_width, args.colors))

    first_frame, *rest_frames = frames
    first_frame.save(
        args.output,
        save_all=True,
        append_images=rest_frames,
        duration=duration_ms,
        loop=args.loop,
        optimize=True,
        disposal=2,
    )

    print(f"Created GIF: {args.output}")
    print(f"Frames: {len(frame_paths)}")
    print(f"FPS: {args.fps}")
    print(f"Frame duration: {duration_ms} ms")
    print(f"Max width: {args.max_width if args.max_width > 0 else 'original'}")
    print(f"Colors: {args.colors}")


if __name__ == "__main__":
    main()

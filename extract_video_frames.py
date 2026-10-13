#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrae los fotogramas de un video y los guarda en una carpeta."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("val_viz/best_val_seq.mp4"),
        help="Ruta al video de entrada.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("val_viz/frames"),
        help="Carpeta donde se guardarán los fotogramas.",
    )
    parser.add_argument(
        "--prefix",
        default="frame",
        help="Prefijo de los nombres de archivo generados.",
    )
    parser.add_argument(
        "--ext",
        default="jpg",
        choices=("jpg", "png"),
        help="Formato de imagen de salida.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Guarda un fotograma cada N frames. Debe ser >= 1.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Máximo de fotogramas a guardar. Usa 0 para no limitar.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.step < 1:
        raise SystemExit("--step debe ser >= 1.")
    if args.max_frames < 0:
        raise SystemExit("--max-frames debe ser >= 0.")
    if not args.video.is_file():
        raise SystemExit(f"No se encontró el video: {args.video}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el video: {args.video}")

    saved_frames = 0
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            current_idx = frame_idx
            frame_idx += 1

            if current_idx % args.step == 0:
                out_name = f"{args.prefix}_{saved_frames:06d}.{args.ext}"
                out_path = args.output_dir / out_name
                if not cv2.imwrite(str(out_path), frame):
                    raise SystemExit(f"No se pudo guardar el fotograma: {out_path}")

                saved_frames += 1
                if args.max_frames and saved_frames >= args.max_frames:
                    break
    finally:
        cap.release()

    print(f"Video: {args.video}")
    print(f"Salida: {args.output_dir}")
    print(f"Fotogramas guardados: {saved_frames}")
    print(f"Frames leidos: {frame_idx}")


if __name__ == "__main__":
    main()

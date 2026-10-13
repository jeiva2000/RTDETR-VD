#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def even_size(value: int) -> int:
    value = max(2, int(value))
    return value if value % 2 == 0 else value - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera un video a partir de una secuencia de fotogramas."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("test_viz/aveo/frames"),
        help="Carpeta con los fotogramas de entrada.",
    )
    parser.add_argument(
        "--pattern",
        default="*.jpg",
        help="Patron glob para localizar los fotogramas.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_viz/aveo/video.mp4"),
        help="Ruta del video de salida.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames por segundo del video de salida.",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC a usar para el video de salida.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Usa un frame cada N para reducir el tamano del video.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=0,
        help="Redimensiona el video a este ancho manteniendo proporcion. Usa 0 para conservar.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Falla si encuentra frames ilegibles o con tamano inconsistente.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.fps <= 0:
        raise SystemExit("--fps debe ser > 0.")
    if args.step < 1:
        raise SystemExit("--step debe ser >= 1.")
    if args.max_width < 0:
        raise SystemExit("--max-width debe ser >= 0.")
    if len(args.codec) != 4:
        raise SystemExit("--codec debe tener exactamente 4 caracteres.")
    if not args.input_dir.is_dir():
        raise SystemExit(f"No se encontro la carpeta de frames: {args.input_dir}")

    frame_paths = sorted(
        p for p in args.input_dir.glob(args.pattern) if p.is_file() and not p.name.startswith(".")
    )
    if not frame_paths:
        raise SystemExit(
            f"No se encontraron frames en '{args.input_dir}' con el patron '{args.pattern}'."
        )

    first_frame = None
    for frame_path in frame_paths:
        first_frame = cv2.imread(str(frame_path))
        if first_frame is not None:
            break
        if args.strict:
            raise SystemExit(f"No se pudo leer el frame: {frame_path}")
        print(f"Aviso: se omite frame ilegible: {frame_path}")

    if first_frame is None:
        raise SystemExit("No se pudo leer ningun frame valido.")

    src_height, src_width = first_frame.shape[:2]
    width, height = src_width, src_height
    if args.max_width and src_width > args.max_width:
        width = even_size(args.max_width)
        height = even_size(round(src_height * width / src_width))

    args.output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*args.codec),
        args.fps,
        (width, height),
    )
    if not writer.isOpened():
        raise SystemExit(f"No se pudo crear el video: {args.output}")

    written_frames = 0
    skipped_frames = 0
    try:
        for frame_index, frame_path in enumerate(frame_paths):
            if frame_index % args.step != 0:
                skipped_frames += 1
                continue

            frame = cv2.imread(str(frame_path))
            if frame is None:
                if args.strict:
                    raise SystemExit(f"No se pudo leer el frame: {frame_path}")
                print(f"Aviso: se omite frame ilegible: {frame_path}")
                skipped_frames += 1
                continue

            frame_h, frame_w = frame.shape[:2]
            if frame_w != src_width or frame_h != src_height:
                if args.strict:
                    raise SystemExit(
                        f"El frame {frame_path} tiene tamano {frame_w}x{frame_h}, esperado {src_width}x{src_height}."
                    )
                print(
                    f"Aviso: se omite frame con tamano inconsistente: {frame_path} ({frame_w}x{frame_h})."
                )
                skipped_frames += 1
                continue

            if frame_w != width or frame_h != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            writer.write(frame)
            written_frames += 1
    finally:
        writer.release()

    print(f"Frames: {args.input_dir}")
    print(f"Patron: {args.pattern}")
    print(f"Video generado: {args.output}")
    print(f"Fotogramas escritos: {written_frames}")
    print(f"Fotogramas omitidos: {skipped_frames}")
    print(f"Resolucion: {width}x{height}")
    print(f"FPS: {args.fps}")
    print(f"Step: {args.step}")


if __name__ == "__main__":
    main()

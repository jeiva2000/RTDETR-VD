import os
import re
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import onnxruntime as ort
from cjm_byte_track.byte_tracker import BYTETracker  # pip install cjm_byte_track
from torchvision.transforms import ToTensor
import numpy as np
from pathlib import Path
from PIL import ImageFile
import argparse
from config import DATASET_ROOT, ONNX_MODEL, TRACKEVAL_PREDS_ROOT, VIEWS_ROOT
ImageFile.LOAD_TRUNCATED_IMAGES = True

def infer_rtdetr(sess, image_path, th=0.4):
    """
    Devuelve (labels, boxes, scores, queries) filtrados por umbral th.
    - labels: (N,)
    - boxes:  (N, 4) en coords absolutas del tamaño original (normalmente xyxy)
    - scores: (N,)
    - queries: (N, ...) (depende del modelo)
    """
    original_im = Image.open(image_path).convert("RGB")
    original_size = original_im.size  # (W, H)

    im = original_im.resize((640, 640))
    im_data = ToTensor()(im)[None]

    output = sess.run(
        output_names=None,
        input_feed={
            "images": im_data.data.numpy(),
            "orig_target_sizes": [list(original_size)],  # [[W, H]]
        },
    )

    labels, boxes, scores, queries = output

    labels = np.asarray(labels)
    boxes = np.asarray(boxes)
    scores = np.asarray(scores)

    keep = scores >= th
    return labels[keep], boxes[keep], scores[keep], np.asarray(queries)[keep]

def _natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def _load_gt_txt(txt_path: Path):
    if not txt_path.exists():
        return np.zeros((0, 4), dtype=np.float32)

    text = txt_path.read_text(encoding="utf-8").strip()
    if not text:
        return np.zeros((0, 4), dtype=np.float32)

    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",") if p.strip() != ""]
        if len(parts) < 4:
            continue
        try:
            x1, y1, x2, y2 = map(float, parts[:4])
        except ValueError:
            continue
        rows.append([x1, y1, x2, y2])
    if not rows:
        return np.zeros((0, 4), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)

def _box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: (N,4), b: (M,4) => iou (N,M)
    """
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0.0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0.0, a_max=None)
    inter = inter_w * inter_h

    area_a = np.clip(ax2 - ax1, a_min=0.0, a_max=None) * np.clip(ay2 - ay1, a_min=0.0, a_max=None)
    area_b = np.clip(bx2 - bx1, a_min=0.0, a_max=None) * np.clip(by2 - by1, a_min=0.0, a_max=None)
    union = area_a[:, None] + area_b[None, :] - inter

    iou = np.zeros_like(inter, dtype=np.float32)
    valid = union > 0
    iou[valid] = inter[valid] / union[valid]
    return iou

def _greedy_match_count(preds: np.ndarray, gts: np.ndarray, iou_th: float):
    if len(preds) == 0 or len(gts) == 0:
        return 0
    ious = _box_iou_xyxy(preds, gts)
    matched_gts = set()
    matched_preds = set()
    # greedy matching by highest IoU
    while True:
        max_idx = np.unravel_index(np.argmax(ious), ious.shape)
        max_iou = ious[max_idx]
        if max_iou < iou_th:
            break
        pi, gi = int(max_idx[0]), int(max_idx[1])
        if pi in matched_preds or gi in matched_gts:
            ious[pi, gi] = -1.0
            continue
        matched_preds.add(pi)
        matched_gts.add(gi)
        ious[pi, :] = -1.0
        ious[:, gi] = -1.0
    return len(matched_preds)


def _count_fp_for_folder_multi_iou(
    sess,
    folder: Path,
    det_th: float,
    iou_thresholds,
    exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp"),
):
    files = [p for p in folder.iterdir() if p.suffix.lower() in exts]
    files.sort(key=lambda p: _natural_key(p.name))

    stats_by_iou = {
        thr: {"preds": 0, "gts": 0, "fps": 0, "matches": 0}
        for thr in iou_thresholds
    }

    for img_path in files:
        labels, boxes_xyxy, scores, _ = infer_rtdetr(sess, str(img_path), th=det_th)
        preds = np.asarray(boxes_xyxy, dtype=np.float32)
        gt_path = img_path.with_suffix(".txt")
        gts = _load_gt_txt(gt_path)
        pred_n = len(preds)
        gt_n = len(gts)
        for thr in iou_thresholds:
            matches = _greedy_match_count(preds, gts, iou_th=thr)
            fps = max(pred_n - matches, 0)
            stats_by_iou[thr]["preds"] += pred_n
            stats_by_iou[thr]["gts"] += gt_n
            stats_by_iou[thr]["fps"] += fps
            stats_by_iou[thr]["matches"] += matches

    return stats_by_iou


def _visualize_preds_and_gts_for_views(
    sess,
    views_root: Path,
    out_root: Path,
    det_th: float,
    exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp"),
):
    out_root.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()

    total_images = 0
    for folder in sorted([p for p in views_root.iterdir() if p.is_dir()]):
        out_folder = out_root / folder.name
        out_folder.mkdir(parents=True, exist_ok=True)

        files = [p for p in folder.iterdir() if p.suffix.lower() in exts]
        files.sort(key=lambda p: _natural_key(p.name))

        for img_path in files:
            labels, boxes_xyxy, scores, _ = infer_rtdetr(sess, str(img_path), th=det_th)
            preds = np.asarray(boxes_xyxy, dtype=np.float32)
            gts = _load_gt_txt(img_path.with_suffix(".txt"))

            with Image.open(img_path).convert("RGB") as im:
                draw = ImageDraw.Draw(im)

                # GT en verde
                for i, gt in enumerate(gts):
                    x1, y1, x2, y2 = map(float, gt)
                    draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
                    draw.text((x1, max(0.0, y1 - 12.0)), f"GT {i}", fill="lime", font=font)

                # Pred en rojo
                for i, (pred, score) in enumerate(zip(preds, scores)):
                    x1, y1, x2, y2 = map(float, pred)
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    draw.text((x1, y1 + 2.0), f"Pred {i} {float(score):.2f}", fill="red", font=font)

                draw.text((8, 8), "GT=verde | Pred=rojo", fill="yellow", font=font)
                im.save(out_folder / img_path.name)
                total_images += 1

    print(f"[VIZ] Imágenes visualizadas: {total_images}")
    print(f"[VIZ] Carpeta de salida: {out_root}")


def _xyxy_to_tlwh_xyxy(box_xyxy):
    x1, y1, x2, y2 = box_xyxy
    x = float(x1)
    y = float(y1)
    w = float(x2 - x1)
    h = float(y2 - y1)
    return x, y, w, h


def _dets_for_bytetrack_xyxy(boxes_xyxy: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """
    ByteTrack (tipo YOLOX) suele esperar output_results como ndarray (N,5):
      [x1, y1, x2, y2, score]
    """
    if len(boxes_xyxy) == 0:
        return np.zeros((0, 5), dtype=np.float32)

    boxes_xyxy = boxes_xyxy.astype(np.float32)
    scores = scores.astype(np.float32).reshape(-1, 1)
    return np.concatenate([boxes_xyxy, scores], axis=1)


def run_bytetrack_from_rtdetr_folder(
    sess,
    images_dir: str,
    out_tracks_txt: str,
    det_th: float = 0.4,           # umbral para RT-DETR
    track_thresh: float = 0.5,    # ByteTrack: threshold para tracks
    track_buffer: int = 30,
    match_thresh: float = 0.7,
    frame_rate: int = 30,
    per_class: bool = False,       # True = 1 tracker por clase
    class_id_offset: int = 1_000_000,  # para evitar colisiones de IDs entre clases
    exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp"),
):
    """
    - Lee carpeta de frames, orden natural.
    - Hace inferencia con RT-DETR (tu infer_rtdetr) por frame.
    - Alimenta ByteTrack con detecciones (x1,y1,x2,y2,score).
    - Guarda tracks en un TXT estilo MOT.

    Salida por línea (10 columnas):
      frame, track_id, x, y, w, h, score, class, -1, -1

    Nota:
      - Si per_class=False, class se escribe como -1 (ByteTrack “single-class”).
      - Si per_class=True, se trackea por clase y se offsetean IDs por clase.
    """
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"No existe la carpeta: {images_dir}")

    files = [f for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in exts]
    files.sort(key=_natural_key)
    if not files:
        raise RuntimeError(f"No se encontraron imágenes en {images_dir}")

    os.makedirs(os.path.dirname(out_tracks_txt) or ".", exist_ok=True)

    # Trackers
    if per_class:
        trackers = {}  # class_id -> BYTETracker
    else:
        tracker = BYTETracker(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            frame_rate=frame_rate,
        )

    lines = []
    model_infer_total_s = 0.0
    model_infer_frames = 0

    for frame_idx, fname in enumerate(files, start=1):
        img_path = os.path.join(images_dir, fname)

        with Image.open(img_path) as im0:
            W, H = im0.size

        # --- RT-DETR detections (ya filtradas por det_th) ---
        infer_start = time.perf_counter()
        labels, boxes_xyxy, scores, _ = infer_rtdetr(sess, img_path, th=det_th)
        model_infer_total_s += time.perf_counter() - infer_start
        model_infer_frames += 1

        # ByteTracker.update pide (img_info: (H,W), img_size: (H,W)) según docs :contentReference[oaicite:2]{index=2}
        img_info = (H, W)
        img_size = (H, W)  # como ya pasas boxes a escala original, no queremos reescalar nada

        if not per_class:
            dets = _dets_for_bytetrack_xyxy(np.asarray(boxes_xyxy), np.asarray(scores))
            online_targets = tracker.update(dets, img_info, img_size)  # :contentReference[oaicite:3]{index=3}

            for t in online_targets:
                tlwh = t.tlwh  # :contentReference[oaicite:4]{index=4}
                tid = int(t.track_id)
                x, y, w, h = map(float, tlwh)
                score = float(getattr(t, "score", 1.0))

                lines.append(
                    f"{frame_idx},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.6f},-1,-1,-1\n"
                )

        else:
            # 1 tracker por clase (más estable para multi-clase)
            labels_np = np.asarray(labels).astype(int)
            boxes_np = np.asarray(boxes_xyxy)
            scores_np = np.asarray(scores)

            for cls in np.unique(labels_np):
                if cls not in trackers:
                    trackers[cls] = BYTETracker(
                        track_thresh=track_thresh,
                        track_buffer=track_buffer,
                        match_thresh=match_thresh,
                        frame_rate=frame_rate,
                    )

                mask = labels_np == cls
                dets_cls = _dets_for_bytetrack_xyxy(boxes_np[mask], scores_np[mask])
                online_targets = trackers[cls].update(dets_cls, img_info, img_size)

                for t in online_targets:
                    tlwh = t.tlwh  # :contentReference[oaicite:5]{index=5}
                    tid = int(t.track_id) + int(cls) * class_id_offset
                    x, y, w, h = map(float, tlwh)
                    score = float(getattr(t, "score", 1.0))

                    lines.append(
                        f"{frame_idx},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.6f},-1,-1,-1\n"
                    )

    with open(out_tracks_txt, "w", encoding="utf-8") as f:
        f.writelines(lines)

    infer_ms_per_frame = (model_infer_total_s / model_infer_frames * 1000.0) if model_infer_frames > 0 else 0.0
    infer_fps = (model_infer_frames / model_infer_total_s) if model_infer_total_s > 0 else 0.0
    return {
        "num_frames": len(files),
        "num_track_rows": len(lines),
        "images_dir": images_dir,
        "out_tracks_txt": out_tracks_txt,
        "det_threshold": det_th,
        "per_class": per_class,
        "model_infer_total_s": model_infer_total_s,
        "model_infer_frames": model_infer_frames,
        "model_infer_ms_per_frame": infer_ms_per_frame,
        "model_infer_fps": infer_fps,
    }


"""
sess = ort.InferenceSession(ONNX_MODEL, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
output_path = os.path.join(TRACKEVAL_PREDS_ROOT, 'rtdetr_onnx') + '/'
os.makedirs(output_path,exist_ok=True)
os.makedirs(output_path+'/data',exist_ok=True)
info = run_bytetrack_from_rtdetr_folder(
    sess,
    images_dir=os.path.join(DATASET_ROOT, "test", "aveo"),
    out_tracks_txt=os.path.join(TRACKEVAL_PREDS_ROOT, "rtdetr_onnx", "data", "aveo.txt"),
    det_th=0.4,
    per_class=True,   # si quieres IDs separados por clase
)
print(info)
"""

sess = ort.InferenceSession(ONNX_MODEL, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
test_root = Path(DATASET_ROOT) / "test"
views_root = Path(VIEWS_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument('--th_score', type=float)
parser.add_argument('--viz_views', action='store_true', help='Guardar visualizaciones de GT y predicciones en views')
parser.add_argument('--viz_out_dir', type=str, default=None, help='Carpeta de salida para visualizaciones')
args = parser.parse_args()
th_score = args.th_score

if args.viz_views and views_root.exists():
    viz_out_root = Path(args.viz_out_dir) if args.viz_out_dir else Path(f"views_viz_{th_score}")
    _visualize_preds_and_gts_for_views(
        sess=sess,
        views_root=views_root,
        out_root=viz_out_root,
        det_th=th_score,
    )

# --- Falsos positivos en /views antes de tracking ---
if views_root.exists():
    print(f"\n=== Evaluando falsos positivos en: {views_root} ===")
    iou_thresholds = [round(x / 10.0, 1) for x in range(1, 10)]
    total_stats_by_iou = {
        thr: {"preds": 0, "gts": 0, "fps": 0, "matches": 0}
        for thr in iou_thresholds
    }

    for folder in sorted([p for p in views_root.iterdir() if p.is_dir()]):
        stats_by_iou = _count_fp_for_folder_multi_iou(
            sess,
            folder,
            det_th=th_score,
            iou_thresholds=iou_thresholds,
        )
        for thr in iou_thresholds:
            s = stats_by_iou[thr]
            total_stats_by_iou[thr]["preds"] += s["preds"]
            total_stats_by_iou[thr]["gts"] += s["gts"]
            total_stats_by_iou[thr]["fps"] += s["fps"]
            total_stats_by_iou[thr]["matches"] += s["matches"]
            tp = s["matches"]
            fp = s["fps"]
            fn = max(s["gts"] - s["matches"], 0)
            recall_pct = (tp / s["gts"]) * 100.0 if s["gts"] > 0 else 0.0
            acc_den = tp + fp + fn
            accuracy_pct = (tp / acc_den) * 100.0 if acc_den > 0 else 0.0
            pred_den = tp + fp
            fpr_pct = (fp / pred_den) * 100.0 if pred_den > 0 else 0.0
            print(
                f"[views_fp][{folder.name}] preds={s['preds']} gts={s['gts']} matches={s['matches']} "
                f"fps={s['fps']} iou_th={thr:.2f} "
                f"recall={recall_pct:.2f}% accuracy={accuracy_pct:.2f}% "
                f"fpr={fpr_pct:.2f}%"
            )

    for thr in iou_thresholds:
        s = total_stats_by_iou[thr]
        tp = s["matches"]
        fp = s["fps"]
        fn = max(s["gts"] - s["matches"], 0)
        recall_pct = (tp / s["gts"]) * 100.0 if s["gts"] > 0 else 0.0
        acc_den = tp + fp + fn
        accuracy_pct = (tp / acc_den) * 100.0 if acc_den > 0 else 0.0
        pred_den = tp + fp
        fpr_pct = (fp / pred_den) * 100.0 if pred_den > 0 else 0.0
        print(
            f"[views_fp] preds={s['preds']} gts={s['gts']} matches={s['matches']} "
            f"fps={s['fps']} iou_th={thr:.2f} "
            f"recall={recall_pct:.2f}% accuracy={accuracy_pct:.2f}% "
            f"fpr={fpr_pct:.2f}%"
        )
else:
    print(f"\n[WARN] No existe la carpeta de views: {views_root}")


# Base de salida (TrackEval)
output_root = Path(TRACKEVAL_PREDS_ROOT) / ("rtdetr_onnx_" + str(th_score))
os.makedirs(output_root,exist_ok=True)
output_data = output_root / "data"
output_data.mkdir(parents=True, exist_ok=True)

# Recorre cada subcarpeta (cada "video")
total_model_infer_s = 0.0
total_model_infer_frames = 0
for video_dir in sorted([p for p in test_root.iterdir() if p.is_dir()]):
    video_name = video_dir.name

    out_txt = output_data / f"{video_name}.txt"

    print(f"\n=== Procesando: {video_name} ===")
    print(f"images_dir: {video_dir}")
    print(f"out_tracks_txt: {out_txt}")

    info = run_bytetrack_from_rtdetr_folder(
        sess,
        images_dir=str(video_dir),
        out_tracks_txt=str(out_txt),
        det_th=th_score,
        per_class=True,  # IDs separados por clase
    )

    print(f"OK -> {out_txt}")
    print(info)
    total_model_infer_s += float(info.get("model_infer_total_s", 0.0))
    total_model_infer_frames += int(info.get("model_infer_frames", 0))

if total_model_infer_frames > 0:
    total_ms_per_frame = (total_model_infer_s / total_model_infer_frames) * 1000.0
    total_fps = total_model_infer_frames / total_model_infer_s if total_model_infer_s > 0 else 0.0
    print(
        "[rtdetr_timing] "
        f"frames={total_model_infer_frames} "
        f"total_s={total_model_infer_s:.6f} "
        f"ms_per_frame={total_ms_per_frame:.3f} "
        f"fps={total_fps:.3f}"
    )

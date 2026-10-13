import argparse
import os
import random
import re
import subprocess
import shutil
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from config import CKPT_DIR, PREDS_DIR, TRACKEVAL_DIR, TRACKEVAL_PREDS_ROOT, VIEWS_ROOT


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


TRACKEVAL_METRICS: Tuple[str, ...] = (
    "HOTA",
    "DetA",
    "AssA",
    "DetRe",
    "DetPr",
    "AssRe",
    "AssPr",
    "MOTA",
    "IDF1",
)


def _parse_metric_value(raw: str) -> Optional[float]:
    s = raw.strip().replace("%", "")
    if "," in s and "." in s:
        s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_metrics_from_table_lines(lines: List[str]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    number_pattern = r"[-+]?[0-9]+(?:[.,][0-9]+)?(?:[eE][-+]?[0-9]+)?"
    for i in range(len(lines)):
        header_tokens = re.split(r"\s+", lines[i].strip())
        metric_indices = {tok: idx for idx, tok in enumerate(header_tokens) if tok in TRACKEVAL_METRICS}
        if len(metric_indices) < 2:
            continue

        combined_row: Optional[Dict[str, float]] = None
        fallback_row: Optional[Dict[str, float]] = None
        max_idx = max(metric_indices.values())
        for j in range(i + 1, min(i + 8, len(lines))):
            row_tokens = re.split(r"\s+", lines[j].strip())
            if not row_tokens:
                continue
            # Stop if we reached another header-like line.
            if len([tok for tok in row_tokens if tok in TRACKEVAL_METRICS]) >= 2:
                break

            offset = 1 if len(row_tokens) > max_idx + 1 and _parse_metric_value(row_tokens[0]) is None else 0
            parsed_row: Dict[str, float] = {}
            for metric_name, idx in metric_indices.items():
                val_idx = idx + offset
                if val_idx >= len(row_tokens):
                    continue
                raw = row_tokens[val_idx].rstrip("%")
                if not re.fullmatch(number_pattern, raw):
                    continue
                parsed = _parse_metric_value(raw)
                if parsed is not None:
                    parsed_row[metric_name] = parsed
            if not parsed_row:
                continue

            row_name = row_tokens[0].upper() if offset == 1 else ""
            if row_name == "COMBINED":
                combined_row = parsed_row
                break
            if fallback_row is None:
                fallback_row = parsed_row

        if combined_row is not None:
            metrics.update(combined_row)
        elif fallback_row is not None:
            metrics.update(fallback_row)
    return metrics


def _parse_metrics_from_combined_sections(output: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    section_columns: Dict[str, List[str]] = {
        "HOTA": [
            "HOTA",
            "DetA",
            "AssA",
            "DetRe",
            "DetPr",
            "AssRe",
            "AssPr",
            "LocA",
            "OWTA",
            "HOTA(0)",
            "LocA(0)",
            "HOTALocA(0)",
        ],
        "CLEAR": [
            "MOTA",
            "MOTP",
            "MODA",
            "CLR_Re",
            "CLR_Pr",
            "MTR",
            "PTR",
            "MLR",
            "sMOTA",
            "CLR_TP",
            "CLR_FN",
            "CLR_FP",
            "IDSW",
            "MT",
            "PT",
            "ML",
            "Frag",
        ],
        "Identity": [
            "IDF1",
            "IDR",
            "IDP",
            "IDTP",
            "IDFN",
            "IDFP",
        ],
    }
    number_re = r"[-+]?[0-9]+(?:[.,][0-9]+)?(?:[eE][-+]?[0-9]+)?"
    section_names = list(section_columns.keys())
    for idx, section in enumerate(section_names):
        next_sections = section_names[idx + 1 :]
        next_pat = "|".join(re.escape(ns) + ":" for ns in next_sections + ["Count", "Timing analysis"])
        if next_pat:
            pat = rf"{re.escape(section)}:\s*(.*?)(?=\n(?:{next_pat})|\Z)"
        else:
            pat = rf"{re.escape(section)}:\s*(.*?)(?=\Z)"
        blocks = list(re.finditer(pat, output, flags=re.DOTALL | re.IGNORECASE))
        if not blocks:
            continue
        block = blocks[-1].group(1)
        combined_match = re.search(r"(?mi)^\s*COMBINED\s+(.+?)\s*$", block)
        if not combined_match:
            continue
        values = re.findall(number_re, combined_match.group(1))
        if not values:
            continue
        cols = section_columns[section]
        for col_name, raw in zip(cols, values):
            if col_name in TRACKEVAL_METRICS:
                parsed = _parse_metric_value(raw)
                if parsed is not None:
                    metrics[col_name] = parsed
    return metrics


def _parse_metrics_from_text(output: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    number_pattern = r"[-+]?[0-9]+(?:[.,][0-9]+)?(?:[eE][-+]?[0-9]+)?"

    # 1) Parse explicit key/value occurrences (most precise).
    for metric_name in TRACKEVAL_METRICS:
        escaped = re.escape(metric_name)
        matches = re.findall(rf"(?<!\w){escaped}(?!\w)\s*[:=]\s*({number_pattern})\s*%?", output)
        if not matches:
            continue
        value = _parse_metric_value(matches[-1])
        if value is not None:
            metrics[metric_name] = value

    # 2) Parse TrackEval tables, prioritizing COMBINED rows.
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    metrics.update(_parse_metrics_from_table_lines(lines))
    # 3) Parse by section + COMBINED row, which is robust even when headers are malformed.
    metrics.update(_parse_metrics_from_combined_sections(output))

    return metrics


def _parse_metrics_from_summary(trackeval_dir: str, tracker_name: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    base_dir = Path(trackeval_dir) / "data" / "trackers" / "damages" / "test" / tracker_name
    for summary_path in base_dir.glob("**/summary.txt"):
        try:
            lines = [ln.strip() for ln in summary_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except OSError:
            continue
        metrics.update(_parse_metrics_from_table_lines(lines))
        for line in lines:
            for metric_name in TRACKEVAL_METRICS:
                m = re.search(
                    rf"(?<!\w){re.escape(metric_name)}(?!\w)\s*[:=]\s*([-+]?[0-9]+(?:[.,][0-9]+)?(?:[eE][-+]?[0-9]+)?)",
                    line,
                )
                if not m:
                    continue
                parsed = _parse_metric_value(m.group(1))
                if parsed is not None:
                    metrics[metric_name] = parsed
        if len(metrics) == len(TRACKEVAL_METRICS):
            return metrics
    return metrics


def run_trackeval_for_tracker(tracker_name: str, show_output: bool = False) -> Optional[Dict[str, float]]:
    trackeval_dir = TRACKEVAL_DIR
    cmd = [
        sys.executable,
        "scripts/run_mot_challenge.py",
        "--BENCHMARK",
        "damages",
        "--SPLIT_TO_EVAL",
        "test",
        "--TRACKERS_TO_EVAL",
        tracker_name,
        "--METRICS",
        "CLEAR",
        "Identity",
        "HOTA",
        "--PLOT_CURVES",
        "False",
    ]
    prev_cwd = os.getcwd()
    try:
        os.chdir(trackeval_dir)
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    finally:
        os.chdir(prev_cwd)
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    if show_output:
        if result.stdout:
            print("[trackeval][stdout]")
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print("[trackeval][stderr]")
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
    metrics = _parse_metrics_from_text(output)
    if not metrics:
        metrics = _parse_metrics_from_summary(trackeval_dir, tracker_name)
    return metrics if metrics else None


def format_trackeval_metrics(metrics: Dict[str, float]) -> str:
    ordered = [
        f"{k}={metrics[k]:.4f}" if k in metrics else f"{k}=NA"
        for k in TRACKEVAL_METRICS
    ]
    return " | ".join(ordered)


def average_trackeval_metrics(runs: List[Dict[str, float]]) -> Dict[str, float]:
    avg: Dict[str, float] = {}
    for metric_name in TRACKEVAL_METRICS:
        vals = [m[metric_name] for m in runs if metric_name in m]
        if vals:
            avg[metric_name] = float(sum(vals) / len(vals))
    return avg


def trackeval_metric_coverage(runs: List[Dict[str, float]]) -> Dict[str, int]:
    return {metric_name: sum(1 for m in runs if metric_name in m) for metric_name in TRACKEVAL_METRICS}


def build_track_tag(args: argparse.Namespace) -> str:
    train_random_skip_max = args.train_random_skip_max if args.train_random_skip_max > 1 else None
    sigmoid_tag = args.cross_attn_activation if args.use_cross_attn else "softmax"
    return (
        f"rtdetr_{sigmoid_tag}_{args.emb_dim}_seq{args.seq_len}_str{args.stride}"
        f"_skip{train_random_skip_max or 1}"
        f"_cross{int(args.use_cross_attn)}"
        f"_mask{int(args.use_cross_attn_mask)}"
        f"_coord{int(args.use_coord_time_embeds)}"
        f"_norm{int(args.use_normalized_dists)}"
    )


def build_child_argv_for_seed(seed: int) -> List[str]:
    argv = sys.argv[1:]
    child_argv: List[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--triple-seed-train":
            i += 1
            continue
        if token == "--triple-seeds":
            i += 4
            continue
        if token == "--seed":
            i += 2
            continue
        if token.startswith("--seed="):
            i += 1
            continue
        child_argv.append(token)
        i += 1
    child_argv.extend(["--seed", str(seed)])
    return child_argv


class MOTSeqDataset(Dataset):
    def __init__(
        self,
        npy_path: str,
        seq_len: int = 30,
        stride: int = 30,
        score_threshold: Optional[float] = None,
        random_skip_max: Optional[int] = None,
    ):
        self.data = np.load(npy_path, allow_pickle=True).item()
        self.seq_len = seq_len
        self.stride = stride
        self.score_threshold = score_threshold
        self.random_skip_max = random_skip_max

        videos: Dict[str, List[str]] = {}
        for path in self.data.keys():
            vid = path.split("/")[-2]
            videos.setdefault(vid, []).append(path)

        def frame_num(p: str) -> int:
            name = os.path.basename(p).split(".")[0]
            nums = re.findall(r"\d+", name)
            return int(nums[-1]) if nums else 0

        for vid in videos:
            videos[vid] = sorted(videos[vid], key=frame_num)

        self.videos = videos
        self.video_sequences: List[List[str]] = []
        for _, frames in videos.items():
            n = len(frames)
            if self.random_skip_max is not None and self.random_skip_max > 1:
                window = seq_len * self.random_skip_max
                if n >= window:
                    for i in range(0, n - window + 1, stride):
                        self.video_sequences.append(frames[i : i + window])
                else:
                    self.video_sequences.append(frames)
            else:
                if n >= seq_len:
                    for i in range(0, n - seq_len + 1, stride):
                        self.video_sequences.append(frames[i : i + seq_len])
                else:
                    self.video_sequences.append(frames)

    def __len__(self) -> int:
        return len(self.video_sequences)

    def _unwrap(self, entry: Any) -> Any:
        if isinstance(entry, list) and len(entry) > 0:
            return entry[0]
        return entry

    def _frame_from_path(self, path: str) -> Dict[str, torch.Tensor]:
        def _unwrap_results(res: Any) -> Any:
            while isinstance(res, list) and len(res) == 1:
                res = res[0]
            return res

        entry = self._unwrap(self.data[path])
        gt_boxes = None

        qd = entry.get("queries_det", None)
        if isinstance(qd, list):
            qd = qd[0] if len(qd) > 0 else None
        if qd is None:
            qd = torch.zeros((0, 256), dtype=torch.float32)
        elif not torch.is_tensor(qd):
            qd = torch.as_tensor(qd, dtype=torch.float32)

        ids = None
        if "targets" in entry and isinstance(entry["targets"], list) and len(entry["targets"]) > 0:
            t0 = entry["targets"][0]
            if isinstance(t0, dict) and "ids" in t0:
                ids = t0["ids"]
                gt_boxes = t0.get("boxes")
        if ids is None:
            ids = torch.zeros((0,), dtype=torch.long)
        elif not torch.is_tensor(ids):
            ids = torch.as_tensor(ids, dtype=torch.long)
        ids = ids.view(-1)
        if gt_boxes is None:
            gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
        elif not torch.is_tensor(gt_boxes):
            gt_boxes = torch.as_tensor(gt_boxes, dtype=torch.float32)
        if gt_boxes.numel() == 0:
            gt_boxes = gt_boxes.view(0, 4)
        else:
            gt_boxes = gt_boxes.view(-1, gt_boxes.shape[-1])[:, :4]

        scores = entry.get("scores_det", None)
        if scores is None:
            scores = entry.get("scores", None)
        if isinstance(scores, list):
            scores = scores[0] if len(scores) > 0 else None
        if scores is not None and not torch.is_tensor(scores):
            scores = torch.as_tensor(scores, dtype=torch.float32)
        if torch.is_tensor(scores):
            scores = scores.view(-1)

        boxes = entry.get("boxes_det_norm", None)
        if boxes is None:
            boxes = entry.get("boxes_det", None)
        if isinstance(boxes, list):
            boxes = boxes[0] if len(boxes) > 0 else None
        if boxes is None:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        elif not torch.is_tensor(boxes):
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if boxes.numel() == 0:
            boxes = boxes.view(0, 4)
        else:
            if boxes.dim() == 1:
                cols = 5 if scores is None and boxes.numel() % 5 == 0 else 4
                boxes = boxes.view(-1, cols)
            else:
                boxes = boxes.view(-1, boxes.shape[-1])
            if boxes.shape[1] >= 5:
                if scores is None:
                    scores = boxes[:, 4].clone()
                boxes = boxes[:, :4]
            else:
                boxes = boxes[:, :4]

        if self.score_threshold is not None and torch.is_tensor(scores):
            scores = scores.view(-1)
            n = min(scores.shape[0], qd.shape[0], ids.shape[0], boxes.shape[0])
            if n == 0:
                qd = qd[:0]
                ids = ids[:0]
                boxes = boxes[:0]
                scores = scores[:0]
            else:
                scores = scores[:n]
                mask = scores >= self.score_threshold
                qd = qd[:n][mask]
                ids = ids[:n][mask]
                boxes = boxes[:n][mask]
                scores = scores[mask]

        results = _unwrap_results(entry.get("results", None))
        return {
            "queries": qd,
            "ids": ids,
            "boxes": boxes,
            "scores": scores,
            "gt_boxes": gt_boxes,
            "path": path,
            "results": results,
            "width": entry.get("width"),
            "height": entry.get("height"),
        }

    def __getitem__(self, idx: int) -> List[Dict[str, torch.Tensor]]:
        seq_paths = self.video_sequences[idx]
        if (
            self.random_skip_max is not None
            and self.random_skip_max > 1
            and len(seq_paths) >= self.seq_len * self.random_skip_max
        ):
            max_step = self.random_skip_max
            start = random.randint(0, max_step - 1)
            pos = start
            picked = []
            for _ in range(self.seq_len):
                picked.append(seq_paths[pos])
                pos += random.randint(1, max_step)
            seq_paths = picked
        frames: List[Dict[str, torch.Tensor]] = []
        for path in seq_paths:
            frames.append(self._frame_from_path(path))
        return frames


class QueryProjector(nn.Module):
    def __init__(
        self,
        d_in: int = 256,
        d_hidden: int = 512,
        d_out: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


class SigmoidMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        attn_activation: str = "sigmoid",
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if attn_activation not in {"sigmoid", "softmax"}:
            raise ValueError("attn_activation must be 'sigmoid' or 'softmax'")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.attn_activation = attn_activation

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, tgt_len, _ = query.shape
        _, src_len, _ = key.shape

        q = self.q_proj(query).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            scores = scores + attn_mask
        if key_padding_mask is not None:
            mask = key_padding_mask[:, None, None, :].to(dtype=torch.bool)
            scores = scores.masked_fill(mask, -1e9)

        if self.attn_activation == "softmax":
            attn = torch.softmax(scores, dim=-1)
        else:
            attn = torch.sigmoid(scores)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        out = self.out_proj(out)
        attn_weights = attn.mean(dim=1)
        return out, attn_weights


class SelfCrossTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        cross_attn_activation: str = "sigmoid",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = SigmoidMultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            attn_activation=cross_attn_activation,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sa_out, _ = self.self_attn(x, x, x, key_padding_mask=self_key_padding_mask, need_weights=False)
        x = self.norm1(x + self.dropout(sa_out))

        ca_out, _ = self.cross_attn(
            x,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            attn_mask=cross_attn_mask,
        )
        x = self.norm2(x + self.dropout(ca_out))

        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm3(x + self.dropout(ff))
        return x


class QueryProjectorWithCross(nn.Module):
    def __init__(
        self,
        d_in: int = 256,
        d_hidden: int = 512,
        d_out: int = 256,
        dropout: float = 0.2,
        nhead: int = 8,
        dim_feedforward: int = 512,
        attn_dropout: float = 0.1,
        use_coord_time: bool = True,
        cross_attn_activation: str = "sigmoid",
    ):
        super().__init__()
        self.projector = QueryProjector(d_in=d_in, d_hidden=d_hidden, d_out=d_out, dropout=dropout)
        self.use_coord_time = use_coord_time
        self.coord_proj = nn.Sequential(
            nn.Linear(4, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(1, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )
        self.fuse_proj = nn.Linear(d_out * 3, d_out)
        self.proj_boxes = nn.Sequential(
            nn.Linear(d_out, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 4),
        )
        self.cross_block = SelfCrossTransformer(
            d_model=d_out,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=attn_dropout,
            cross_attn_activation=cross_attn_activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
        frame_idx: Optional[torch.Tensor] = None,
        hist_z: Optional[torch.Tensor] = None,
        use_cross: bool = False,
        use_cross_attn_mask: bool = False,
        use_coord_time: bool = True,
    ) -> torch.Tensor:
        z = self.projector(x)
        use_coord_time = use_coord_time and self.use_coord_time
        if use_coord_time and boxes is not None and frame_idx is not None and z.numel() > 0:
            boxes_t = boxes if torch.is_tensor(boxes) else torch.as_tensor(boxes, dtype=torch.float32)
            boxes_t = boxes_t.to(z.device, dtype=torch.float32)
            if boxes_t.dim() == 1:
                boxes_t = boxes_t.view(-1, 4)
            n = min(z.shape[0], boxes_t.shape[0])
            if n > 0:
                z = z[:n]
                boxes_t = boxes_t[:n]
                if not torch.is_tensor(frame_idx):
                    frame_idx_t = torch.as_tensor(frame_idx, dtype=torch.float32, device=z.device)
                else:
                    frame_idx_t = frame_idx.to(z.device, dtype=torch.float32)
                if frame_idx_t.dim() == 0:
                    frame_idx_t = frame_idx_t.view(1, 1).repeat(n, 1)
                elif frame_idx_t.dim() == 1:
                    frame_idx_t = frame_idx_t.view(-1, 1)
                    if frame_idx_t.shape[0] == 1:
                        frame_idx_t = frame_idx_t.repeat(n, 1)
                elif frame_idx_t.dim() == 2 and frame_idx_t.shape[0] == 1:
                    frame_idx_t = frame_idx_t.repeat(n, 1)
                if frame_idx_t.shape[0] != n:
                    frame_idx_t = frame_idx_t[:n]
                z_coord = self.coord_proj(boxes_t)
                z_time = self.time_proj(frame_idx_t)
                z = self.fuse_proj(torch.cat([z, z_coord, z_time], dim=-1))
                z = F.normalize(z, dim=-1)
        if not use_cross or hist_z is None or hist_z.numel() == 0 or z.numel() == 0:
            return z

        q = z.unsqueeze(0)
        kv = hist_z.unsqueeze(0)
        cross_attn_mask = None
        if use_cross_attn_mask:
            dist = torch.cdist(z, hist_z)
            cross_attn_mask = (-dist).unsqueeze(0).unsqueeze(0)
        z = self.cross_block(q, kv, cross_attn_mask=cross_attn_mask).squeeze(0)
        return F.normalize(z, dim=-1)


def contrastive_loss_with_history(
    z_cur: torch.Tensor,
    ids_cur: torch.Tensor,
    z_hist: Optional[torch.Tensor],
    ids_hist: Optional[torch.Tensor],
    margin: float = 1.0,
) -> torch.Tensor:
    if z_cur.numel() == 0 or ids_cur.numel() == 0:
        return z_cur.sum() * 0.0
    if z_hist is None or ids_hist is None or z_hist.numel() == 0 or ids_hist.numel() == 0:
        return z_cur.sum() * 0.0

    n = min(z_cur.shape[0], ids_cur.shape[0])
    m = min(z_hist.shape[0], ids_hist.shape[0])
    if n == 0 or m == 0:
        return z_cur.sum() * 0.0

    z_cur = z_cur[:n]
    ids_cur = ids_cur[:n]
    z_hist = z_hist[:m]
    ids_hist = ids_hist[:m]

    dists = torch.cdist(z_cur, z_hist)
    pos_mask = ids_cur[:, None] == ids_hist[None, :]
    neg_mask = ~pos_mask

    loss = z_cur.sum() * 0.0
    if pos_mask.any():
        pos_d = dists[pos_mask]
        loss = loss + (pos_d**2).mean()
    if neg_mask.any():
        neg_d = dists[neg_mask]
        loss = loss + F.relu(margin - neg_d).pow(2).mean()
    return loss


def id_switch_loss(
    z_cur: torch.Tensor,
    ids_cur: torch.Tensor,
    z_prev: Optional[torch.Tensor],
    ids_prev: Optional[torch.Tensor],
    margin: float = 0.2,
) -> torch.Tensor:
    if z_cur.numel() == 0 or ids_cur.numel() == 0:
        return z_cur.sum() * 0.0
    if z_prev is None or ids_prev is None or z_prev.numel() == 0 or ids_prev.numel() == 0:
        return z_cur.sum() * 0.0

    n = min(z_cur.shape[0], ids_cur.shape[0])
    m = min(z_prev.shape[0], ids_prev.shape[0])
    if n == 0 or m == 0:
        return z_cur.sum() * 0.0

    z_cur = z_cur[:n]
    ids_cur = ids_cur[:n]
    z_prev = z_prev[:m]
    ids_prev = ids_prev[:m]

    dists = torch.cdist(z_cur, z_prev)
    losses = []
    for i in range(n):
        pos_mask = ids_prev == ids_cur[i]
        neg_mask = ~pos_mask
        if not pos_mask.any() or not neg_mask.any():
            continue
        pos_d = dists[i][pos_mask].min()
        neg_d = dists[i][neg_mask].min()
        losses.append(F.relu(pos_d - neg_d + margin))

    if len(losses) == 0:
        return z_cur.sum() * 0.0
    return torch.stack(losses).mean()


def new_id_repel_loss(
    z_cur: torch.Tensor,
    ids_cur: torch.Tensor,
    z_hist: Optional[torch.Tensor],
    ids_hist: Optional[torch.Tensor],
    margin: float = 1.0,
) -> torch.Tensor:
    if z_cur.numel() == 0 or ids_cur.numel() == 0:
        return z_cur.sum() * 0.0
    if z_hist is None or ids_hist is None or z_hist.numel() == 0 or ids_hist.numel() == 0:
        return z_cur.sum() * 0.0

    n = min(z_cur.shape[0], ids_cur.shape[0])
    m = min(z_hist.shape[0], ids_hist.shape[0])
    if n == 0 or m == 0:
        return z_cur.sum() * 0.0

    z_cur = z_cur[:n]
    ids_cur = ids_cur[:n]
    z_hist = z_hist[:m]
    ids_hist = ids_hist[:m]

    hist_ids_set = set(ids_hist.detach().cpu().tolist())
    new_mask = torch.tensor(
        [int(i.item()) not in hist_ids_set for i in ids_cur],
        device=ids_cur.device,
        dtype=torch.bool,
    )
    if not new_mask.any():
        return z_cur.sum() * 0.0

    z_new = z_cur[new_mask]
    dists = torch.cdist(z_new, z_hist)
    return F.relu(margin - dists).pow(2).mean()


def apply_score_threshold(
    queries: torch.Tensor,
    ids: Optional[torch.Tensor] = None,
    boxes: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    threshold: Optional[float] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if threshold is None or scores is None:
        return queries, ids, boxes, scores
    scores_t = scores if torch.is_tensor(scores) else torch.as_tensor(scores, dtype=torch.float32)
    scores_t = scores_t.view(-1)
    if torch.is_tensor(ids) and ids.numel() == 0:
        ids = None
    if torch.is_tensor(boxes) and boxes.numel() == 0:
        boxes = None
    n = scores_t.shape[0]
    if torch.is_tensor(queries):
        n = min(n, queries.shape[0])
    if torch.is_tensor(ids):
        n = min(n, ids.shape[0])
    if torch.is_tensor(boxes):
        n = min(n, boxes.shape[0])
    scores_t = scores_t[:n]
    mask = scores_t >= threshold
    queries = queries[:n][mask]
    if torch.is_tensor(ids):
        ids = ids[:n]
        ids = ids[mask.to(ids.device)]
    if torch.is_tensor(boxes):
        boxes = boxes[:n]
        boxes = boxes[mask.to(boxes.device)]
    scores_t = scores_t[mask]
    return queries, ids, boxes, scores_t


def postprocess_track_lines(
    lines: List[str],
    min_track_len: Optional[int] = None,
) -> List[str]:
    if min_track_len is None or min_track_len <= 1:
        return lines
    def parse_tid(line: str) -> Optional[int]:
        parts = line.split(",")
        if len(parts) < 2:
            return None
        try:
            return int(parts[1])
        except ValueError:
            return None

    counts: Dict[int, int] = {}
    for line in lines:
        tid = parse_tid(line)
        if tid is None:
            continue
        counts[tid] = counts.get(tid, 0) + 1
    filtered: List[str] = []
    for line in lines:
        tid = parse_tid(line)
        if tid is None:
            continue
        if counts.get(tid, 0) >= min_track_len:
            filtered.append(line)
    return filtered


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    margin: float = 0.2,
    max_hist_items: int = 5000,
    use_history: bool = True,
    use_cross_attn: bool = False,
    use_cross_attn_mask: bool = False,
    use_coord_time_embeds: bool = False,
    id_switch_weight: float = 1.0,
    id_switch_margin: float = 0.2,
    new_id_repel_weight: float = 0.0,
    new_id_repel_margin: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    total_seqs = 0

    for seq in tqdm(dataloader, desc="train", leave=False):
        unique_ids = set()
        for frame in seq:
            ids = frame["ids"]
            if torch.is_tensor(ids) and ids.numel() > 0:
                unique_ids.update(ids.view(-1).tolist())
        if len(unique_ids) < 2:
            continue

        hist_z = None
        hist_ids = None
        seq_loss = None
        seq_loss_frames = 0

        last_z = None
        last_ids = None
        seq_len = len(seq)
        for t, frame in enumerate(seq):
            queries = frame["queries"].to(device)
            ids = frame["ids"].to(device)
            boxes = frame.get("boxes")
            boxes_t = None
            if boxes is not None:
                boxes_t = boxes if torch.is_tensor(boxes) else torch.as_tensor(boxes, dtype=torch.float32)
                boxes_t = boxes_t.to(device, dtype=torch.float32)

            if queries.numel() == 0 or ids.numel() == 0:
                continue

            queries, ids, boxes_t, _ = apply_score_threshold(
                queries,
                ids=ids,
                boxes=boxes_t,
                scores=None,
                threshold=None,
            )
            if queries.numel() == 0 or ids is None or ids.numel() == 0:
                continue

            n = min(queries.shape[0], ids.shape[0])
            if boxes_t is not None:
                n = min(n, boxes_t.shape[0])
            if n == 0:
                continue

            queries = queries[:n]
            ids = ids[:n]

            frame_pos = 0.0 if seq_len <= 1 else float(t) / float(seq_len - 1)
            frame_pos_t = torch.full((n, 1), frame_pos, device=device)
            z = model(
                queries,
                boxes=boxes_t[:n] if boxes_t is not None else None,
                frame_idx=frame_pos_t,
                hist_z=hist_z if use_history else None,
                use_cross=use_cross_attn,
                use_cross_attn_mask=use_cross_attn_mask,
                use_coord_time=use_coord_time_embeds,
            )
            loss = contrastive_loss_with_history(
                z,
                ids,
                hist_z if use_history else None,
                hist_ids if use_history else None,
                margin=margin,
            )
            if last_z is not None and last_ids is not None and id_switch_weight != 0.0:
                switch_loss = id_switch_loss(z, ids, last_z, last_ids, margin=id_switch_margin)
                loss = loss + id_switch_weight * switch_loss
            if new_id_repel_weight != 0.0 and use_history:
                repel_loss = new_id_repel_loss(z, ids, hist_z, hist_ids, margin=new_id_repel_margin)
                loss = loss + new_id_repel_weight * repel_loss
            seq_loss = loss if seq_loss is None else (seq_loss + loss)
            seq_loss_frames += 1
            last_z = z
            last_ids = ids

            if use_history:
                z_detach = z.detach()
                ids_detach = ids.detach()
                if hist_z is None:
                    hist_z = z_detach
                    hist_ids = ids_detach
                else:
                    hist_z = torch.cat([hist_z, z_detach], dim=0)
                    hist_ids = torch.cat([hist_ids, ids_detach], dim=0)

                if hist_z.shape[0] > max_hist_items:
                    hist_z = hist_z[-max_hist_items:]
                    hist_ids = hist_ids[-max_hist_items:]

        if seq_loss is None:
            continue
        seq_loss = seq_loss / max(seq_loss_frames, 1)

        optimizer.zero_grad(set_to_none=True)
        seq_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += seq_loss.item()
        total_seqs += 1

    avg_loss = total_loss / max(total_seqs, 1)
    return avg_loss


def eval_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    margin: float = 0.2,
    max_hist_items: int = 5000,
    use_history: bool = True,
    use_cross_attn: bool = False,
    use_cross_attn_mask: bool = False,
    use_coord_time_embeds: bool = False,
    id_switch_weight: float = 1.0,
    id_switch_margin: float = 0.2,
    new_id_repel_weight: float = 0.0,
    new_id_repel_margin: float = 1.0,
    mode: str = "val",
    match_threshold: float = 1.0,
    score_threshold: Optional[float] = None,
    generate_video: bool = False,
    video_out_path: Optional[str] = None,
    video_seq_idx: int = 0,
    video_fps: int = 10,
    frames_out_dir: Optional[str] = None,
    fp_calc_frames_dir: Optional[str] = None,
    use_ema_updates: bool = True,
    ema_momentum: float = 0.95,
    use_normalized_dists: bool = False,
    test_max_frames: Optional[int] = None,
    test_track_max_age: Optional[int] = None,
    video_post_min_track_len: Optional[int] = None,
    calc_false_positives: bool = False,
) -> Optional[float]:
    model.eval()
    if mode not in {"val", "test"}:
        raise ValueError("mode must be 'val' or 'test'")
    total_loss = 0.0
    total_seqs = 0
    writer = None
    writer_opened = False
    blank = None
    use_postprocess = (
        mode == "test"
        and video_post_min_track_len is not None
        and video_post_min_track_len > 1
    )
    use_video_post = use_postprocess and generate_video
    fp_calc_enabled = mode == "test" and use_postprocess and calc_false_positives
    if fp_calc_enabled and fp_calc_frames_dir is None:
        if frames_out_dir:
            parent = os.path.dirname(frames_out_dir) or "."
            fp_calc_frames_dir = os.path.join(parent, "views_preds_gts")
        elif video_out_path:
            video_parent = os.path.dirname(video_out_path) or "."
            video_stem = os.path.splitext(os.path.basename(video_out_path))[0]
            fp_calc_frames_dir = os.path.join(video_parent, f"{video_stem}_views_preds_gts")
    if fp_calc_enabled and fp_calc_frames_dir is not None:
        os.makedirs(fp_calc_frames_dir, exist_ok=True)
        if mode == "test" and use_postprocess:
            print(f"[views_fp] views_preds_gts_dir={fp_calc_frames_dir}")
    views_root = VIEWS_ROOT
    views_index: Dict[str, set[str]] = {}
    if fp_calc_enabled and os.path.isdir(views_root):
        for vid in os.listdir(views_root):
            vid_dir = os.path.join(views_root, vid)
            if not os.path.isdir(vid_dir):
                continue
            names = set()
            for name in os.listdir(vid_dir):
                if name.lower().endswith((".jpg", ".jpeg", ".png")):
                    names.add(name)
            if names:
                views_index[vid] = names

    views_active_tracks: Dict[str, List[int]] = {}
    views_iou_thresholds = [round(x / 10.0, 1) for x in range(1, 10)]
    views_vis_iou_threshold = 0.5
    views_stats_by_iou: Dict[float, Dict[str, int]] = {
        thr: {"frames": 0, "preds": 0, "gts": 0, "fps": 0, "matches": 0}
        for thr in views_iou_thresholds
    }
    views_stats_by_video_by_iou: Dict[str, Dict[float, Dict[str, int]]] = {}
    views_saved_frames_by_video: Dict[str, int] = {}

    render_video = generate_video

    def _open_writer(frame_path: str, w: int, h: int) -> None:
        nonlocal writer, writer_opened
        if writer_opened or not render_video or video_out_path is None:
            return
        out_dir = os.path.dirname(video_out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        try:
            import cv2
        except Exception:
            return
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_out_path, fourcc, video_fps, (w, h))
        if writer.isOpened():
            writer_opened = True
        else:
            writer = None

    def _scale_boxes_to_image(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes
        if boxes.max().item() <= 1.5:
            scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
            return boxes * scale
        return boxes

    def _format_dist_matrix(
        mat: np.ndarray,
        row_labels: Optional[List[int]] = None,
        col_labels: Optional[List[int]] = None,
        max_rows: int = 6,
        max_cols: int = 6,
    ) -> List[str]:
        rows = min(mat.shape[0], max_rows)
        cols = min(mat.shape[1], max_cols)
        lines = [f"dists {mat.shape[0]}x{mat.shape[1]} (show {rows}x{cols})"]
        if col_labels:
            col_head = "cols " + " ".join(f"{col_labels[c]:>6}" for c in range(cols))
            lines.append(col_head)
        for r in range(rows):
            prefix = ""
            if row_labels:
                prefix = f"{row_labels[r]:>4} | "
            row_vals = " ".join(f"{mat[r, c]:.2f}" for c in range(cols))
            lines.append(prefix + row_vals)
        return lines

    def _ensure_frames_dir() -> Optional[str]:
        if not render_video or frames_out_dir is None:
            return None
        os.makedirs(frames_out_dir, exist_ok=True)
        return frames_out_dir

    def _is_view_frame(frame: Dict[str, Any]) -> bool:
        frame_path = frame.get("path", "")
        if not frame_path:
            return False
        if not views_index:
            return False
        parts = frame_path.replace("\\", "/").split("/")
        vid = parts[-2] if len(parts) >= 2 else ""
        if vid not in views_index:
            return False
        fname = os.path.basename(frame_path)
        return fname in views_index[vid]

    def _view_video_name(frame: Dict[str, Any]) -> str:
        frame_path = frame.get("path", "")
        if not frame_path:
            return "unknown"
        parts = frame_path.replace("\\", "/").split("/")
        if len(parts) >= 2:
            return parts[-2]
        return "unknown"

    def _xyxy_iou_matrix(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
            return torch.zeros((pred_boxes.shape[0], gt_boxes.shape[0]), dtype=torch.float32)
        p = pred_boxes.to(dtype=torch.float32).view(-1, 4)
        g = gt_boxes.to(dtype=torch.float32).view(-1, 4)
        p = torch.stack(
            [
                torch.minimum(p[:, 0], p[:, 2]),
                torch.minimum(p[:, 1], p[:, 3]),
                torch.maximum(p[:, 0], p[:, 2]),
                torch.maximum(p[:, 1], p[:, 3]),
            ],
            dim=1,
        )
        g = torch.stack(
            [
                torch.minimum(g[:, 0], g[:, 2]),
                torch.minimum(g[:, 1], g[:, 3]),
                torch.maximum(g[:, 0], g[:, 2]),
                torch.maximum(g[:, 1], g[:, 3]),
            ],
            dim=1,
        )

        x1 = torch.maximum(p[:, None, 0], g[None, :, 0])
        y1 = torch.maximum(p[:, None, 1], g[None, :, 1])
        x2 = torch.minimum(p[:, None, 2], g[None, :, 2])
        y2 = torch.minimum(p[:, None, 3], g[None, :, 3])
        inter = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)

        area_p = (p[:, 2] - p[:, 0]).clamp(min=0.0) * (p[:, 3] - p[:, 1]).clamp(min=0.0)
        area_g = (g[:, 2] - g[:, 0]).clamp(min=0.0) * (g[:, 3] - g[:, 1]).clamp(min=0.0)
        union = area_p[:, None] + area_g[None, :] - inter
        iou = torch.zeros_like(inter)
        valid = union > 0
        iou[valid] = inter[valid] / union[valid]
        return iou

    def _load_view_gt_boxes(frame: Dict[str, Any]) -> torch.Tensor:
        frame_path = frame.get("path", "")
        if not frame_path:
            return torch.zeros((0, 4), dtype=torch.float32)
        parts = frame_path.replace("\\", "/").split("/")
        vid = parts[-2] if len(parts) >= 2 else ""
        frame_name = os.path.basename(frame_path)
        frame_stem, _ = os.path.splitext(frame_name)
        txt_path = os.path.join(views_root, vid, f"{frame_stem}.txt")
        if not os.path.exists(txt_path):
            return torch.zeros((0, 4), dtype=torch.float32)
        rows: List[List[float]] = []
        try:
            with open(txt_path, "r") as f:
                for line in f:
                    cols = [p.strip() for p in line.split(",") if p.strip() != ""]
                    if len(cols) < 4:
                        continue
                    try:
                        x1, y1, x2, y2 = map(float, cols[:4])
                    except ValueError:
                        continue
                    rows.append([x1, y1, x2, y2])
        except OSError:
            return torch.zeros((0, 4), dtype=torch.float32)
        if not rows:
            return torch.zeros((0, 4), dtype=torch.float32)
        return torch.as_tensor(rows, dtype=torch.float32)

    def _match_view_preds_to_gt(
        pred_boxes_t: torch.Tensor,
        gt_boxes: torch.Tensor,
        iou_threshold: float,
    ) -> Tuple[int, Set[int]]:
        pred_count = int(pred_boxes_t.shape[0])
        gt_count = int(gt_boxes.shape[0])
        if pred_count == 0 or gt_count == 0:
            return 0, set()
        ious = _xyxy_iou_matrix(pred_boxes_t, gt_boxes).cpu().numpy()
        pairs: List[Tuple[float, int, int]] = []
        for pi in range(pred_count):
            for gi in range(gt_count):
                iou = float(ious[pi, gi])
                if iou >= iou_threshold:
                    pairs.append((iou, pi, gi))
        pairs.sort(key=lambda x: x[0], reverse=True)
        used_preds: Set[int] = set()
        used_gts: Set[int] = set()
        for _, pi, gi in pairs:
            if pi in used_preds or gi in used_gts:
                continue
            used_preds.add(pi)
            used_gts.add(gi)
        return len(used_preds), used_preds

    def _accumulate_view_stats(
        frame: Dict[str, Any], pred_boxes: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        gt_boxes = _load_view_gt_boxes(frame)
        if pred_boxes is None:
            pred_boxes_t = torch.zeros((0, 4), dtype=torch.float32)
        elif not torch.is_tensor(pred_boxes):
            pred_boxes_t = torch.as_tensor(pred_boxes, dtype=torch.float32)
        else:
            pred_boxes_t = pred_boxes.detach().to(dtype=torch.float32)
        if pred_boxes_t.numel() == 0:
            pred_boxes_t = pred_boxes_t.view(0, 4)
        else:
            pred_boxes_t = pred_boxes_t.view(-1, pred_boxes_t.shape[-1])[:, :4]

        w = int(frame.get("width") or 0)
        h = int(frame.get("height") or 0)
        if w > 0 and h > 0:
            pred_boxes_t = _scale_boxes_to_image(pred_boxes_t, w, h)
            gt_boxes = _scale_boxes_to_image(gt_boxes, w, h)

        pred_count = int(pred_boxes_t.shape[0])
        gt_count = int(gt_boxes.shape[0])
        vid = _view_video_name(frame)
        frame_matches_vis = 0
        frame_fp_vis = pred_count
        matched_pred_idx_vis: Set[int] = set()
        for thr in views_iou_thresholds:
            frame_matches, matched_pred_idx = _match_view_preds_to_gt(
                pred_boxes_t, gt_boxes, iou_threshold=thr
            )
            frame_fp = max(pred_count - frame_matches, 0)
            total_stats = views_stats_by_iou[thr]
            total_stats["frames"] += 1
            total_stats["preds"] += pred_count
            total_stats["gts"] += gt_count
            total_stats["fps"] += frame_fp
            total_stats["matches"] += frame_matches
            vid_stats = views_stats_by_video_by_iou.setdefault(vid, {}).setdefault(
                thr,
                {"frames": 0, "preds": 0, "gts": 0, "fps": 0, "matches": 0},
            )
            vid_stats["frames"] += 1
            vid_stats["preds"] += pred_count
            vid_stats["gts"] += gt_count
            vid_stats["fps"] += frame_fp
            vid_stats["matches"] += frame_matches
            if abs(thr - views_vis_iou_threshold) < 1e-9:
                frame_matches_vis = frame_matches
                frame_fp_vis = frame_fp
                matched_pred_idx_vis = matched_pred_idx
        return {
            "pred_boxes_t": pred_boxes_t,
            "gt_boxes_t": gt_boxes,
            "matched_pred_idx": matched_pred_idx_vis,
            "pred_count": pred_count,
            "gt_count": gt_count,
            "frame_matches": frame_matches_vis,
            "frame_fp": frame_fp_vis,
        }

    def _write_fp_calc_frame(
        frame: Dict[str, Any],
        t: int,
        assigned_ids: List[int],
        pred_boxes_t: Optional[torch.Tensor],
        gt_boxes_t: Optional[torch.Tensor],
        matched_pred_idx: Optional[Set[int]],
    ) -> None:
        if fp_calc_frames_dir is None:
            return
        try:
            import cv2
        except Exception:
            return
        img_path = frame.get("path", "")
        img = None
        if img_path and os.path.exists(img_path):
            img = cv2.imread(img_path)
        if img is None:
            h = int(frame.get("height") or 720)
            w = int(frame.get("width") or 1280)
            img = np.zeros((h, w, 3), dtype=np.uint8)
        h, w = img.shape[:2]

        gt_n = 0
        if gt_boxes_t is not None and torch.is_tensor(gt_boxes_t) and gt_boxes_t.numel() > 0:
            gt_vis = gt_boxes_t.detach().to(dtype=torch.float32)
            gt_vis = _scale_boxes_to_image(gt_vis, w, h).cpu().numpy()
            gt_n = gt_vis.shape[0]
            for i in range(gt_n):
                x1, y1, x2, y2 = gt_vis[i].tolist()
                cv2.rectangle(
                    img,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    img,
                    f"GT {i}",
                    (int(x1), max(int(y1) - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        pred_n = 0
        if pred_boxes_t is not None and torch.is_tensor(pred_boxes_t) and pred_boxes_t.numel() > 0:
            pred_vis = pred_boxes_t.detach().to(dtype=torch.float32)
            pred_vis = _scale_boxes_to_image(pred_vis, w, h).cpu().numpy()
            pred_n = pred_vis.shape[0]
            n = min(pred_n, len(assigned_ids))
            for i in range(n):
                x1, y1, x2, y2 = pred_vis[i].tolist()
                tid = int(assigned_ids[i])
                is_tp = matched_pred_idx is not None and i in matched_pred_idx
                color = (0, 255, 0) if is_tp else (0, 0, 255)
                cv2.rectangle(
                    img,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    2,
                )
                tag = "TP" if is_tp else "FP"
                cv2.putText(
                    img,
                    f"pred id {tid} {tag}",
                    (int(x1), min(int(y2) + 14, h - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        cv2.putText(
            img,
            f"GT={gt_n} Pred={pred_n}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        vid = _view_video_name(frame)
        out_dir = os.path.join(fp_calc_frames_dir, vid)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, _frame_name(frame, t))
        cv2.imwrite(out_path, img)
        views_saved_frames_by_video[vid] = views_saved_frames_by_video.get(vid, 0) + 1

    def _frame_name(frame: Dict[str, Any], t: int) -> str:
        frame_path = frame.get("path", "")
        base = os.path.splitext(os.path.basename(frame_path))[0] if frame_path else ""
        if base:
            return f"{base}.jpg"
        return f"frame_{t:06d}.jpg"

    def _write_frame_only(frame: Dict[str, Any], t: int) -> None:
        nonlocal blank
        if not render_video or video_out_path is None:
            return
        try:
            import cv2
        except Exception:
            return
        frames_dir = _ensure_frames_dir()
        img_path = frame.get("path", "")
        img = None
        if img_path and os.path.exists(img_path):
            img = cv2.imread(img_path)
        if img is None:
            if blank is None:
                h = int(frame.get("height") or 720)
                w = int(frame.get("width") or 1280)
                blank = np.zeros((h, w, 3), dtype=np.uint8)
            img = blank.copy()
        if mode == "test":
            target_w, target_h = 640, 480
            if img.shape[1] != target_w or img.shape[0] != target_h:
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        _open_writer(img_path, w, h)
        if writer_opened and writer is not None:
            writer.write(img)
        if frames_dir is not None:
            out_path = os.path.join(frames_dir, _frame_name(frame, t))
            cv2.imwrite(out_path, img)

    def _draw_frame(
        frame: Dict[str, Any],
        assigned_ids: List[int],
        boxes_t: Optional[torch.Tensor],
        dist_lines: Optional[List[str]],
        t: int,
        matched_pred_idx: Optional[Set[int]] = None,
    ) -> None:
        nonlocal blank
        if not render_video or video_out_path is None:
            return
        try:
            import cv2
        except Exception:
            return
        frames_dir = _ensure_frames_dir()
        img_path = frame.get("path", "")
        img = None
        if img_path and os.path.exists(img_path):
            img = cv2.imread(img_path)
        if img is None:
            if blank is None:
                h = int(frame.get("height") or 720)
                w = int(frame.get("width") or 1280)
                blank = np.zeros((h, w, 3), dtype=np.uint8)
            img = blank.copy()
        h, w = img.shape[:2]
        if boxes_t is None:
            return
        if dist_lines:
            x0, y0 = 10, 30
            for i, line in enumerate(dist_lines):
                y = y0 + i * 14
                if y > h - 10:
                    break
                cv2.putText(
                    img,
                    line,
                    (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
        n = min(len(assigned_ids), boxes_t.shape[0])
        if n == 0:
            if mode == "test":
                target_w, target_h = 640, 480
                if img.shape[1] != target_w or img.shape[0] != target_h:
                    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
            _open_writer(img_path, w, h)
            if writer_opened and writer is not None:
                writer.write(img)
            if frames_dir is not None:
                out_path = os.path.join(frames_dir, _frame_name(frame, t))
                cv2.imwrite(out_path, img)
            return
        boxes_vis = boxes_t[:n].detach()
        boxes_vis = _scale_boxes_to_image(boxes_vis, w, h).cpu().numpy()
        for i in range(n):
            x1, y1, x2, y2 = boxes_vis[i].tolist()
            tid = int(assigned_ids[i])
            has_match_info = matched_pred_idx is not None
            is_tp = has_match_info and i in matched_pred_idx
            color = (0, 255, 0) if is_tp else (0, 0, 255)
            cv2.rectangle(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                2,
            )
            if has_match_info:
                box_label = f"id {tid} {'TP' if is_tp else 'FP'}"
            else:
                box_label = f"id {tid}"
            cv2.putText(
                img,
                box_label,
                (int(x1), max(int(y1) - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        if mode == "test":
            target_w, target_h = 640, 480
            if img.shape[1] != target_w or img.shape[0] != target_h:
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        _open_writer(img_path, w, h)
        if writer_opened and writer is not None:
            writer.write(img)
        if frames_dir is not None:
            out_path = os.path.join(frames_dir, _frame_name(frame, t))
            cv2.imwrite(out_path, img)

    with torch.no_grad():
        for seq_idx, seq in enumerate(tqdm(dataloader, desc=mode, leave=False)):
            if mode == "test" and generate_video:
                render_video = video_seq_idx == seq_idx
            else:
                render_video = generate_video
            if mode == "test" and test_max_frames is not None:
                seq = seq[:test_max_frames]
            unique_ids = set()
            for frame in seq:
                ids = frame["ids"]
                if torch.is_tensor(ids) and ids.numel() > 0:
                    unique_ids.update(ids.view(-1).tolist())
            if mode == "val" and len(unique_ids) < 2:
                continue

            hist_z = None
            hist_ids = None
            seq_loss = None
            seq_loss_frames = 0

            last_z = None
            last_ids = None
            seq_len = len(seq)
            track_embs = None
            track_ids: List[int] = []
            track_last_seen: List[int] = []
            next_track_id = 0
            video_buffer: List[Dict[str, Any]] = []
            track_counts: Dict[int, int] = {}
            for t, frame in enumerate(seq):
                dist_lines = None
                queries = frame["queries"].to(device)
                ids = frame["ids"].to(device)
                boxes = frame.get("boxes")
                scores = frame.get("scores")
                boxes_t = None
                if boxes is not None:
                    boxes_t = boxes if torch.is_tensor(boxes) else torch.as_tensor(boxes, dtype=torch.float32)
                    boxes_t = boxes_t.to(device, dtype=torch.float32)

                if queries.numel() == 0 or (mode == "val" and ids.numel() == 0):
                    if mode == "test":
                        if use_postprocess:
                            video_buffer.append({"frame": frame, "write_only": True, "t": t})
                        elif render_video:
                            _write_frame_only(frame, t)
                    continue

                queries, ids, boxes_t, _ = apply_score_threshold(
                    queries,
                    ids=ids,
                    boxes=boxes_t,
                    scores=scores if mode == "test" else None,
                    threshold=score_threshold if mode == "test" else None,
                )
                if queries.numel() == 0 or (mode == "val" and (ids is None or ids.numel() == 0)):
                    if mode == "test":
                        if use_postprocess:
                            video_buffer.append({"frame": frame, "write_only": True, "t": t})
                        elif render_video:
                            _write_frame_only(frame, t)
                    continue

                n = queries.shape[0]
                if mode == "val":
                    n = min(n, ids.shape[0])
                if boxes_t is not None:
                    n = min(n, boxes_t.shape[0])
                if n == 0:
                    if mode == "test":
                        if use_postprocess:
                            video_buffer.append({"frame": frame, "write_only": True, "t": t})
                        elif render_video:
                            _write_frame_only(frame, t)
                    continue

                queries = queries[:n]
                if mode == "val" and ids is not None:
                    ids = ids[:n]

                frame_pos = 0.0 if seq_len <= 1 else float(t) / float(seq_len - 1)
                frame_pos_t = torch.full((n, 1), frame_pos, device=device)
                z = model(
                    queries,
                    boxes=boxes_t[:n] if boxes_t is not None else None,
                    frame_idx=frame_pos_t,
                    hist_z=(hist_z if mode == "val" else track_embs) if use_history else None,
                    use_cross=use_cross_attn,
                    use_cross_attn_mask=use_cross_attn_mask,
                    use_coord_time=use_coord_time_embeds,
                )
                if mode == "val":
                    loss = contrastive_loss_with_history(
                        z,
                        ids,
                        hist_z if use_history else None,
                        hist_ids if use_history else None,
                        margin=margin,
                    )
                    if last_z is not None and last_ids is not None and id_switch_weight != 0.0:
                        switch_loss = id_switch_loss(z, ids, last_z, last_ids, margin=id_switch_margin)
                        loss = loss + id_switch_weight * switch_loss
                    if new_id_repel_weight != 0.0 and use_history:
                        repel_loss = new_id_repel_loss(z, ids, hist_z, hist_ids, margin=new_id_repel_margin)
                        loss = loss + new_id_repel_weight * repel_loss
                    seq_loss = loss if seq_loss is None else (seq_loss + loss)
                    seq_loss_frames += 1
                    last_z = z
                    last_ids = ids

                    if use_history:
                        z_detach = z.detach()
                        ids_detach = ids.detach()
                        if hist_z is None:
                            hist_z = z_detach
                            hist_ids = ids_detach
                        else:
                            hist_z = torch.cat([hist_z, z_detach], dim=0)
                            hist_ids = torch.cat([hist_ids, ids_detach], dim=0)

                        if hist_z.shape[0] > max_hist_items:
                            hist_z = hist_z[-max_hist_items:]
                            hist_ids = hist_ids[-max_hist_items:]
                else:
                    if not use_history:
                        track_embs = None
                        track_ids = []
                        track_last_seen = []
                    if (
                        test_track_max_age is not None
                        and track_embs is not None
                        and track_embs.numel() > 0
                    ):
                        keep = [
                            idx
                            for idx, last_seen in enumerate(track_last_seen)
                            if (t - last_seen) <= test_track_max_age
                        ]
                        if not keep:
                            track_embs = None
                            track_ids = []
                            track_last_seen = []
                        else:
                            track_embs = track_embs[keep]
                            track_ids = [track_ids[idx] for idx in keep]
                            track_last_seen = [track_last_seen[idx] for idx in keep]
                    assigned_ids = [-1] * n
                    if track_embs is None or track_embs.numel() == 0:
                        assigned_ids = list(range(next_track_id, next_track_id + n))
                        next_track_id += n
                        track_ids.extend(assigned_ids)
                        track_last_seen.extend([t] * n)
                        track_embs = z.detach().clone()
                    else:
                        dists = torch.cdist(z, track_embs)
                        if use_normalized_dists:
                            dists = torch.cdist(F.normalize(z, dim=-1), F.normalize(track_embs, dim=-1))
                        row_labels = None
                        if ids is not None and ids.numel() > 0:
                            row_labels = [int(v) for v in ids[:n].detach().cpu().tolist()]
                        else:
                            row_labels = list(range(n))
                        col_labels = [int(tid) for tid in track_ids]
                        dist_lines = _format_dist_matrix(
                            dists.detach().cpu().numpy(),
                            row_labels=row_labels,
                            col_labels=col_labels,
                        )
                        pairs = []
                        has_candidate = [False] * n
                        for i in range(n):
                            for j in range(track_embs.shape[0]):
                                dist = float(dists[i, j].item())
                                if dist <= match_threshold:
                                    pairs.append((dist, i, j))
                                    has_candidate[i] = True
                        pairs.sort(key=lambda x: x[0])
                        used_tracks = set()
                        used_dets = set()
                        for _, i, j in pairs:
                            if i in used_dets or j in used_tracks:
                                continue
                            assigned_ids[i] = track_ids[j]
                            used_dets.add(i)
                            used_tracks.add(j)
                            track_last_seen[j] = t
                            if use_ema_updates:
                                update = ema_momentum * track_embs[j] + (1.0 - ema_momentum) * z[i].detach()
                            else:
                                update = z[i].detach()
                            track_embs[j] = F.normalize(update, dim=-1) if use_normalized_dists else update
                        for i in range(n):
                            if assigned_ids[i] == -1 and not has_candidate[i]:
                                assigned_ids[i] = next_track_id
                                next_track_id += 1
                                track_ids.append(assigned_ids[i])
                                track_last_seen.append(t)
                                track_embs = torch.cat([track_embs, z[i].detach().unsqueeze(0)], dim=0)

                    if use_postprocess:
                        for tid in assigned_ids:
                            track_counts[tid] = track_counts.get(tid, 0) + 1
                        video_buffer.append(
                            {
                                "frame": frame,
                                "assigned_ids": assigned_ids,
                                "boxes_t": boxes_t[:n].detach().cpu() if boxes_t is not None else None,
                                "dist_lines": dist_lines,
                                "t": t,
                            }
                        )
                    elif render_video and video_out_path is not None:
                        _draw_frame(frame, assigned_ids, boxes_t, dist_lines, t)

            if mode == "val":
                if seq_loss is None:
                    continue
                seq_loss = seq_loss / max(seq_loss_frames, 1)
                total_loss += seq_loss.item()
                total_seqs += 1
            elif use_postprocess:
                keep_tids = {
                    tid for tid, count in track_counts.items() if count >= int(video_post_min_track_len)
                }
                for entry in video_buffer:
                    frame = entry["frame"]
                    is_view = _is_view_frame(frame)
                    if entry.get("write_only"):
                        if is_view:
                            views_active_tracks[frame.get("path", "")] = []
                            match_info = _accumulate_view_stats(frame, pred_boxes=None)
                            _write_fp_calc_frame(
                                frame=frame,
                                t=entry.get("t", 0),
                                assigned_ids=[],
                                pred_boxes_t=match_info.get("pred_boxes_t"),
                                gt_boxes_t=match_info.get("gt_boxes_t"),
                                matched_pred_idx=match_info.get("matched_pred_idx"),
                            )
                        if render_video and video_out_path is not None:
                            _write_frame_only(entry["frame"], entry.get("t", 0))
                        continue
                    assigned_ids = entry["assigned_ids"]
                    if not keep_tids:
                        if is_view:
                            views_active_tracks[frame.get("path", "")] = []
                            match_info = _accumulate_view_stats(frame, pred_boxes=None)
                            _write_fp_calc_frame(
                                frame=frame,
                                t=entry.get("t", 0),
                                assigned_ids=[],
                                pred_boxes_t=match_info.get("pred_boxes_t"),
                                gt_boxes_t=match_info.get("gt_boxes_t"),
                                matched_pred_idx=match_info.get("matched_pred_idx"),
                            )
                        if render_video and video_out_path is not None:
                            _write_frame_only(entry["frame"], entry.get("t", 0))
                        continue
                    keep_idx = [i for i, tid in enumerate(assigned_ids) if tid in keep_tids]
                    if not keep_idx:
                        if is_view:
                            views_active_tracks[frame.get("path", "")] = []
                            match_info = _accumulate_view_stats(frame, pred_boxes=None)
                            _write_fp_calc_frame(
                                frame=frame,
                                t=entry.get("t", 0),
                                assigned_ids=[],
                                pred_boxes_t=match_info.get("pred_boxes_t"),
                                gt_boxes_t=match_info.get("gt_boxes_t"),
                                matched_pred_idx=match_info.get("matched_pred_idx"),
                            )
                        if render_video and video_out_path is not None:
                            _write_frame_only(entry["frame"], entry.get("t", 0))
                        continue
                    boxes_t = entry["boxes_t"]
                    if boxes_t is not None:
                        boxes_t = boxes_t[keep_idx]
                    filtered_ids = [assigned_ids[i] for i in keep_idx]
                    match_info = None
                    if is_view:
                        views_active_tracks[frame.get("path", "")] = filtered_ids
                        match_info = _accumulate_view_stats(frame, pred_boxes=boxes_t)
                        _write_fp_calc_frame(
                            frame=frame,
                            t=entry.get("t", 0),
                            assigned_ids=filtered_ids,
                            pred_boxes_t=match_info.get("pred_boxes_t") if match_info is not None else None,
                            gt_boxes_t=match_info.get("gt_boxes_t") if match_info is not None else None,
                            matched_pred_idx=match_info.get("matched_pred_idx") if match_info is not None else None,
                        )
                    if render_video and video_out_path is not None:
                        matched_pred_idx = None
                        if match_info is not None:
                            matched_pred_idx = match_info.get("matched_pred_idx")
                        _draw_frame(
                            entry["frame"],
                            filtered_ids,
                            boxes_t,
                            None,
                            entry.get("t", 0),
                            matched_pred_idx=matched_pred_idx,
                        )

    if writer_opened and writer is not None:
        writer.release()
    if fp_calc_enabled and views_stats_by_iou[views_iou_thresholds[0]]["frames"] > 0:
        for thr in views_iou_thresholds:
            stats = views_stats_by_iou[thr]
            total_tp = stats["matches"]
            total_fp = stats["fps"]
            total_fn = max(stats["gts"] - stats["matches"], 0)
            total_acc_den = total_tp + total_fp + total_fn
            total_pred_den = total_tp + total_fp
            fpr = (total_fp / total_pred_den) * 100.0 if total_pred_den > 0 else 0.0
            accuracy = (total_tp / total_acc_den) * 100.0 if total_acc_den > 0 else 0.0
            recall = (total_tp / stats["gts"]) * 100.0 if stats["gts"] > 0 else 0.0
            print(
                "[views_fp] "
                f"frames={stats['frames']} preds={stats['preds']} gts={stats['gts']} "
                f"matches={stats['matches']} fps={stats['fps']} iou_th={thr:.2f} "
                f"recall={recall:.2f}% accuracy={accuracy:.2f}% "
                f"fpr={fpr:.2f}%"
            )
        for vid in sorted(views_stats_by_video_by_iou.keys()):
            for thr in views_iou_thresholds:
                stats = views_stats_by_video_by_iou.get(vid, {}).get(
                    thr, {"frames": 0, "preds": 0, "gts": 0, "fps": 0, "matches": 0}
                )
                vid_tp = stats["matches"]
                vid_fp = stats["fps"]
                vid_fn = max(stats["gts"] - stats["matches"], 0)
                vid_acc_den = vid_tp + vid_fp + vid_fn
                vid_pred_den = vid_tp + vid_fp
                vid_fpr = (vid_fp / vid_pred_den) * 100.0 if vid_pred_den > 0 else 0.0
                vid_accuracy = (vid_tp / vid_acc_den) * 100.0 if vid_acc_den > 0 else 0.0
                vid_recall = (vid_tp / stats["gts"]) * 100.0 if stats["gts"] > 0 else 0.0
                print(
                    f"[views_fp][{vid}] "
                    f"frames={stats['frames']} preds={stats['preds']} gts={stats['gts']} "
                    f"matches={stats['matches']} fps={stats['fps']} iou_th={thr:.2f} "
                    f"recall={vid_recall:.2f}% accuracy={vid_accuracy:.2f}% "
                    f"fpr={vid_fpr:.2f}%"
                )
        if fp_calc_frames_dir is not None:
            total_saved = sum(views_saved_frames_by_video.values())
            print(
                f"[views_fp] saved_views_frames={total_saved} "
                f"videos={len(views_saved_frames_by_video)} dir={fp_calc_frames_dir}"
            )
            for vid in sorted(views_saved_frames_by_video.keys()):
                print(f"[views_fp][saved][{vid}] frames={views_saved_frames_by_video[vid]}")
    if mode == "val":
        avg_loss = total_loss / max(total_seqs, 1)
        return avg_loss
    return None


def generate_direct_npy_videos(
    dataset: MOTSeqDataset,
    out_dir: str,
    fps: int = 10,
    score_threshold: Optional[float] = None,
    frames_out_dir: Optional[str] = None,
    test_max_frames: Optional[int] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if frames_out_dir is not None:
        os.makedirs(frames_out_dir, exist_ok=True)
    try:
        import cv2
    except Exception:
        return

    def _scale_boxes_to_image(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes
        if boxes.max().item() <= 1.5:
            scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
            return boxes * scale
        return boxes

    def frame_num(p: str) -> Optional[int]:
        name = os.path.basename(p).split(".")[0]
        nums = re.findall(r"\d+", name)
        return int(nums[-1]) if nums else None

    videos = getattr(dataset, "videos", None)
    if not isinstance(videos, dict) or not videos:
        videos = {}
        for path in dataset.data.keys():
            vid = path.split("/")[-2]
            videos.setdefault(vid, []).append(path)
        for vid in videos:
            videos[vid] = sorted(videos[vid], key=frame_num)

    for vid, frame_paths in videos.items():
        if test_max_frames is not None:
            frame_paths = frame_paths[:test_max_frames]
        if len(frame_paths) == 0:
            continue
        out_path = os.path.join(out_dir, f"{vid}.mp4")
        vid_frames_dir = None
        if frames_out_dir is not None:
            vid_frames_dir = os.path.join(frames_out_dir, vid)
            os.makedirs(vid_frames_dir, exist_ok=True)
        writer = None
        writer_opened = False
        blank = None
        for t, path in enumerate(frame_paths):
            frame = dataset._frame_from_path(path)
            img_path = frame.get("path", "")
            img = None
            if img_path and os.path.exists(img_path):
                img = cv2.imread(img_path)
            if img is None:
                if blank is None:
                    h = int(frame.get("height") or 720)
                    w = int(frame.get("width") or 1280)
                    blank = np.zeros((h, w, 3), dtype=np.uint8)
                img = blank.copy()
            h, w = img.shape[:2]
            if not writer_opened:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                writer_opened = bool(writer is not None and writer.isOpened())
            gt_boxes = frame.get("gt_boxes")
            boxes = frame.get("boxes")
            scores = frame.get("scores")
            ids = frame.get("ids")
            boxes_t = None
            if gt_boxes is not None and torch.is_tensor(gt_boxes) and gt_boxes.numel() > 0:
                boxes_t = gt_boxes
                scores = None
            elif boxes is not None:
                boxes_t = boxes if torch.is_tensor(boxes) else torch.as_tensor(boxes, dtype=torch.float32)
            if boxes_t is None:
                boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            dummy_queries = torch.zeros((boxes_t.shape[0], 1), dtype=torch.float32)
            _, ids, boxes_t, _ = apply_score_threshold(
                dummy_queries,
                ids=ids,
                boxes=boxes_t,
                scores=scores,
                threshold=score_threshold if scores is not None else None,
            )
            if boxes_t is not None and boxes_t.numel() > 0:
                boxes_vis = _scale_boxes_to_image(boxes_t, w, h).cpu().numpy()
                if torch.is_tensor(ids) and ids.numel() > 0:
                    assigned_ids = [int(v) for v in ids[: boxes_vis.shape[0]].detach().cpu().tolist()]
                else:
                    assigned_ids = list(range(boxes_vis.shape[0]))
                for i, (x1, y1, x2, y2) in enumerate(boxes_vis.tolist()):
                    tid = assigned_ids[i] if i < len(assigned_ids) else i
                    cv2.rectangle(
                        img,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        img,
                        f"id {tid}",
                        (int(x1), max(int(y1) - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
            if writer_opened and writer is not None:
                writer.write(img)
            if vid_frames_dir is not None:
                base = os.path.splitext(os.path.basename(img_path))[0] if img_path else ""
                name = f"{base}.jpg" if base else f"frame_{t:06d}.jpg"
                cv2.imwrite(os.path.join(vid_frames_dir, name), img)
        if writer_opened and writer is not None:
            writer.release()


def export_test_predictions_mot(
    model: nn.Module,
    dataset: MOTSeqDataset,
    out_dir: str,
    match_threshold: float = 1.0,
    score_threshold: Optional[float] = None,
    use_cross_attn: bool = False,
    use_cross_attn_mask: bool = False,
    use_coord_time_embeds: bool = False,
    use_ema_updates: bool = True,
    ema_momentum: float = 0.95,
    use_normalized_dists: bool = False,
    test_max_frames: Optional[int] = None,
    test_track_max_age: Optional[int] = None,
    test_min_track_len: Optional[int] = None,
    use_history: bool = True,
    apply_postprocess: bool = True,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    is_cuda_device = device.type == "cuda"
    model_infer_time_total_s = 0.0
    model_infer_frames = 0

    def frame_num(p: str) -> Optional[int]:
        name = os.path.basename(p).split(".")[0]
        nums = re.findall(r"\d+", name)
        return int(nums[-1]) if nums else None

    videos = getattr(dataset, "videos", None)
    if not isinstance(videos, dict) or not videos:
        videos = {}
        for path in dataset.data.keys():
            vid = path.split("/")[-2]
            videos.setdefault(vid, []).append(path)
        for vid in videos:
            videos[vid] = sorted(videos[vid], key=frame_num)

    with torch.no_grad():
        for vid, frame_paths in videos.items():
            if test_max_frames is not None:
                frame_paths = frame_paths[:test_max_frames]
            if len(frame_paths) == 0:
                continue
            frames = [dataset._frame_from_path(p) for p in frame_paths]
            out_path = os.path.join(out_dir, f"{vid}.txt")

            track_embs = None
            track_ids: List[int] = []
            track_last_seen: List[int] = []
            next_track_id = 0
            lines: List[str] = []
            last_frame_id = 0
            total_frames = len(frames)

            for t, frame in enumerate(frames):
                boxes = frame.get("boxes")
                queries = frame.get("queries")
                scores = frame.get("scores")
                frame_w = frame.get("width")
                frame_h = frame.get("height")
                if boxes is None or queries is None:
                    continue

                q_t = queries if torch.is_tensor(queries) else torch.as_tensor(queries)
                b_t = boxes if torch.is_tensor(boxes) else torch.as_tensor(boxes)
                q_t, _, b_t, scores_t = apply_score_threshold(
                    q_t,
                    ids=None,
                    boxes=b_t,
                    scores=scores,
                    threshold=score_threshold,
                )
                if q_t.numel() == 0 or b_t.numel() == 0:
                    continue
                if torch.is_tensor(scores_t):
                    scores_t = scores_t.view(-1)
                else:
                    scores_t = None

                b_out_t = b_t
                if (
                    torch.is_tensor(b_out_t)
                    and b_out_t.numel() > 0
                    and frame_w is not None
                    and frame_h is not None
                    and b_out_t.max().item() <= 1.5
                ):
                    scale = torch.tensor(
                        [frame_w, frame_h, frame_w, frame_h],
                        dtype=b_out_t.dtype,
                        device=b_out_t.device,
                    )
                    b_out_t = b_out_t * scale

                boxes_np = b_out_t.detach().cpu().numpy()
                n = min(len(boxes_np), q_t.shape[0])
                if n == 0:
                    continue
                if torch.is_tensor(scores_t):
                    n = min(n, scores_t.shape[0])
                    scores_t = scores_t[:n]

                queries_t = q_t[:n].to(device)
                frame_pos = 0.0 if total_frames <= 1 else float(t) / float(total_frames - 1)
                frame_pos_t = torch.full((n, 1), frame_pos, device=device)
                if is_cuda_device:
                    torch.cuda.synchronize(device)
                infer_start = time.perf_counter()
                z = model(
                    queries_t,
                    boxes=b_t[:n].to(device),
                    frame_idx=frame_pos_t,
                    hist_z=track_embs if use_history else None,
                    use_cross=use_cross_attn,
                    use_cross_attn_mask=use_cross_attn_mask,
                    use_coord_time=use_coord_time_embeds,
                )
                if is_cuda_device:
                    torch.cuda.synchronize(device)
                model_infer_time_total_s += time.perf_counter() - infer_start
                model_infer_frames += 1

                if not use_history:
                    track_embs = None
                    track_ids = []
                    track_last_seen = []

                if test_track_max_age is not None and track_embs is not None and track_embs.numel() > 0:
                    keep = [
                        idx
                        for idx, last_seen in enumerate(track_last_seen)
                        if (t - last_seen) <= test_track_max_age
                    ]
                    if not keep:
                        track_embs = None
                        track_ids = []
                        track_last_seen = []
                    else:
                        track_embs = track_embs[keep]
                        track_ids = [track_ids[idx] for idx in keep]
                        track_last_seen = [track_last_seen[idx] for idx in keep]

                assigned_ids = [-1] * n
                if track_embs is None or track_embs.numel() == 0:
                    assigned_ids = list(range(next_track_id, next_track_id + n))
                    track_embs = z.detach().clone()
                    track_ids.extend(assigned_ids)
                    track_last_seen.extend([t] * n)
                    next_track_id += n
                else:
                    if use_normalized_dists:
                        dists = torch.cdist(F.normalize(z, dim=-1), F.normalize(track_embs, dim=-1))
                    else:
                        dists = torch.cdist(z, track_embs)
                    pairs = []
                    has_candidate = [False] * n
                    for i in range(n):
                        for j in range(track_embs.shape[0]):
                            dist = float(dists[i, j].item())
                            if dist <= match_threshold:
                                pairs.append((dist, i, j))
                                has_candidate[i] = True
                    pairs.sort(key=lambda x: x[0])
                    used_tracks = set()
                    used_dets = set()
                    for _, i, j in pairs:
                        if i in used_dets or j in used_tracks:
                            continue
                        assigned_ids[i] = track_ids[j]
                        used_dets.add(i)
                        used_tracks.add(j)
                        track_last_seen[j] = t
                        if use_ema_updates:
                            update = ema_momentum * track_embs[j] + (1.0 - ema_momentum) * z[i].detach()
                        else:
                            update = z[i].detach()
                        track_embs[j] = F.normalize(update, dim=-1) if use_normalized_dists else update
                    for i in range(n):
                        if assigned_ids[i] == -1 and not has_candidate[i]:
                            assigned_ids[i] = next_track_id
                            next_track_id += 1
                            track_ids.append(assigned_ids[i])
                            track_last_seen.append(t)
                            track_embs = torch.cat([track_embs, z[i].detach().unsqueeze(0)], dim=0)

                frame_id = frame_num(frame.get("path", "")) or (t + 1)
                if frame_id <= last_frame_id:
                    frame_id = last_frame_id + 1
                last_frame_id = frame_id
                for i in range(n):
                    tid = int(assigned_ids[i])
                    if tid < 0:
                        continue
                    x1, y1, x2, y2 = boxes_np[i].tolist()
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    if torch.is_tensor(scores_t):
                        conf = float(scores_t[i].item())
                    else:
                        conf = 1.0
                    lines.append(
                        f"{frame_id},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1"
                    )

            if apply_postprocess:
                lines = postprocess_track_lines(lines, min_track_len=test_min_track_len)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"[test_preds] Wrote {out_path}")
    if model_infer_frames > 0:
        avg_infer_ms = (model_infer_time_total_s / model_infer_frames) * 1000.0
        print(
            "[test_timing] "
            f"model_infer_total_s={model_infer_time_total_s:.6f} "
            f"frames={model_infer_frames} "
            f"model_infer_ms_per_frame={avg_infer_ms:.3f} "
            f"apply_postprocess={apply_postprocess}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/eval MOT embedding model.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--triple-seed-train",
        action="store_true",
        help="Run training 3 times with fixed seeds and report averaged TrackEval metrics.",
    )
    parser.add_argument(
        "--triple-seeds",
        type=int,
        nargs=3,
        default=[11, 22, 33],
        help="Exactly 3 seeds used when --triple-seed-train is enabled.",
    )
    parser.add_argument("--use-cross-attn", action="store_true", help="Enable cross-attention.")
    parser.add_argument("--use-cross-attn-mask", action="store_true", help="Enable cross-attention mask.")
    parser.add_argument(
        "--cross-attn-activation",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "softmax"],
        help="Activation for cross-attention weights.",
    )
    parser.add_argument(
        "--use-coord-time-embeds",
        action="store_true",
        help="Fuse box coords + time into embeddings.",
    )
    parser.add_argument(
        "--use-normalized-dists",
        action="store_true",
        help="Use L2-normalized embeddings for matching distances.",
    )
    parser.add_argument("--no-ema-updates", action="store_false", dest="use_ema_updates")
    parser.add_argument("--no-history", action="store_false", dest="use_history")
    parser.add_argument("--no-test", action="store_false", dest="run_test")
    parser.add_argument("--no-trackeval", action="store_false", dest="run_trackeval")
    parser.add_argument("--seq-len", type=int, default=240)
    parser.add_argument("--stride", type=int, default=240)
    parser.add_argument("--test-seq-len", type=int, default=100000)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--test-max-frames", type=int, default=None)
    parser.add_argument("--test-track-max-age", type=int, default=None)
    parser.add_argument("--test-min-track-len", type=int, default=3)
    parser.add_argument("--train-random-skip-max", type=int, default=1)
    parser.add_argument("--new-id-repel-weight", type=float, default=0.0)
    parser.add_argument("--new-id-repel-margin", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=1.5)
    parser.add_argument("--trackeval-each-epoch", action="store_true")
    parser.add_argument(
        "--emit-videos",
        action="store_true",
        help="Emit test visualization videos/frames.",
    )
    parser.add_argument(
        "--calc-false-positives",
        action="store_true",
        help="Enable false positive calculation during test postprocess.",
    )
    args = parser.parse_args()
    if args.triple_seed_train:
        run_metrics: List[Dict[str, float]] = []
        tracker_name = build_track_tag(args)
        for run_idx, seed in enumerate(args.triple_seeds, start=1):
            child_argv = build_child_argv_for_seed(seed)
            cmd = [sys.executable, os.path.abspath(__file__), *child_argv]
            print(
                f"[triple-seed] Run {run_idx}/3 starting with seed={seed}. "
                f"cmd={' '.join(cmd)}"
            )
            subprocess.run(cmd, check=True)
            if args.run_test and args.run_trackeval:
                metrics = run_trackeval_for_tracker(tracker_name)
                if metrics is not None:
                    run_metrics.append(metrics)
                    print(
                        f"[triple-seed] Run {run_idx}/3 metrics: "
                        f"{format_trackeval_metrics(metrics)}"
                    )
                else:
                    print(f"[triple-seed] Run {run_idx}/3 without parsed TrackEval metrics.")
        if args.run_test and args.run_trackeval:
            avg_metrics = average_trackeval_metrics(run_metrics)
            if avg_metrics:
                expected_runs = len(args.triple_seeds)
                coverage = trackeval_metric_coverage(run_metrics)
                coverage_str = " | ".join(
                    f"{metric_name}:{coverage[metric_name]}/{expected_runs}"
                    for metric_name in TRACKEVAL_METRICS
                )
                print(f"[triple-seed] Metric coverage: {coverage_str}")
                if len(run_metrics) != expected_runs or any(c != expected_runs for c in coverage.values()):
                    print(
                        f"[triple-seed][warning] Average uses parsed metrics only "
                        f"({len(run_metrics)}/{expected_runs} runs with parsed output)."
                    )
                print(f"[triple-seed] Average over {len(run_metrics)} parsed runs: {format_trackeval_metrics(avg_metrics)}")
            else:
                print("[triple-seed] Could not compute average TrackEval metrics.")
        return
    # Seed setting disabled by default; enable by passing --seed.
    if args.seed is not None:
        set_seed(args.seed)

    train_npy_path = os.path.join(PREDS_DIR, "train_damages.npy")
    val_npy_path = os.path.join(PREDS_DIR, "val_damages.npy")
    test_npy_path = os.path.join(PREDS_DIR, "test_damages.npy")
    seq_len = args.seq_len
    stride = args.stride
    test_seq_len = args.test_seq_len

    train_random_skip_max = args.train_random_skip_max if args.train_random_skip_max > 1 else None
    train_ds = MOTSeqDataset(
        train_npy_path,
        seq_len=seq_len,
        stride=stride,
        random_skip_max=train_random_skip_max,
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: x[0],
    )
    val_ds = MOTSeqDataset(val_npy_path, seq_len=seq_len, stride=stride)
    val_dl = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x[0],
    )

    use_cross_attn = args.use_cross_attn
    use_cross_attn_mask = args.use_cross_attn_mask
    cross_attn_activation = args.cross_attn_activation
    use_coord_time_embeds = args.use_coord_time_embeds
    emb_dim = args.emb_dim
    margin = args.margin
    max_hist_items = 6000
    epochs = 100
    run_test = args.run_test if hasattr(args, "run_test") else True
    test_match_threshold = 1.0
    test_score_threshold = 0.6
    test_viz_out_path = "test_viz/test_seq.mp4"
    test_viz_frames_dir = "test_viz/test_seq_frames"
    direct_npy_test_viz_out_dir = "test_viz/epoch01_direct_npy"
    direct_npy_test_viz_frames_dir = "test_viz/epoch01_direct_npy_frames"
    best_test_viz_out_path = "test_viz/best_seq_test.mp4"
    best_post_test_viz_out_path = "test_viz/best_seq_test_post.mp4"
    test_viz_seq_idx = 0
    test_viz_fps = 10
    test_use_ema_updates = args.use_ema_updates
    test_ema_momentum = 0.95
    use_history = args.use_history
    test_use_normalized_dists = args.use_normalized_dists
    test_max_frames = args.test_max_frames
    test_track_max_age = args.test_track_max_age
    test_min_track_len = args.test_min_track_len
    test_emit_videos = args.emit_videos
    test_calc_false_positives = args.calc_false_positives
    test_preds_root = TRACKEVAL_PREDS_ROOT
    track_tag = build_track_tag(args)
    test_preds_dir = os.path.join(test_preds_root, track_tag, "data")
    track_tag_no_post = f"{track_tag}_no_post"
    test_preds_no_post_dir = os.path.join(test_preds_root, track_tag_no_post, "data")
    ckpt_root = CKPT_DIR
    ckpt_dir = os.path.join(ckpt_root, track_tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, "mlp_queries_triplet_best.pt")
    run_trackeval = args.run_trackeval if hasattr(args, "run_trackeval") else True
    trackeval_each_epoch = args.trackeval_each_epoch
    model = QueryProjectorWithCross(
        d_in=256,
        d_hidden=128,
        d_out=emb_dim,
        dropout=0.5,
        cross_attn_activation=cross_attn_activation,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    id_switch_weight = 2.5 #1.0
    id_switch_margin = 0.5 #0.2
    new_id_repel_weight = args.new_id_repel_weight
    new_id_repel_margin = args.new_id_repel_margin
    es_patience = 10
    es_min_delta = 0.0
    no_improve = 0

    def _make_improve_viz_paths() -> dict[str, str]:
        base_dir = os.path.join("test_viz", "val_improve_latest")
        raw_dir = os.path.join(base_dir, "best_seq_test")
        post_dir = os.path.join(base_dir, "best_seq_test_post")
        return {
            "raw_video": os.path.join(raw_dir, "video.mp4"),
            "raw_frames": os.path.join(raw_dir, "frames"),
            "post_video": os.path.join(post_dir, "video.mp4"),
            "post_frames": os.path.join(post_dir, "frames"),
        }

    def _reset_improve_viz_dirs(base_dir: str) -> None:
        if os.path.isdir(base_dir):
            shutil.rmtree(base_dir, ignore_errors=True)

    def _iter_test_video_seq_indices(dataset: MOTSeqDataset) -> List[tuple[int, str]]:
        seqs = getattr(dataset, "video_sequences", None)
        if not isinstance(seqs, list) or not seqs:
            return []
        pairs: List[tuple[int, str]] = []
        for idx, seq in enumerate(seqs):
            if not seq:
                continue
            first_path = seq[0]
            vid = os.path.basename(os.path.dirname(first_path)) or f"seq_{idx:03d}"
            pairs.append((idx, vid))
        return pairs

    best_val_loss = float("inf")
    best_hota = float("-inf")
    print(
        "[config] use_cross_attn="
        f"{use_cross_attn} use_cross_attn_mask={use_cross_attn_mask} "
        f"cross_attn_activation={cross_attn_activation} "
        f"use_coord_time_embeds={use_coord_time_embeds} emb_dim={emb_dim} "
        f"margin={margin} max_hist_items={max_hist_items} epochs={epochs} "
        f"id_switch_weight={id_switch_weight} id_switch_margin={id_switch_margin} "
        f"new_id_repel_weight={new_id_repel_weight} new_id_repel_margin={new_id_repel_margin} "
        f"seq_len={seq_len} stride={stride} test_seq_len={test_seq_len} "
        f"train_random_skip_max={train_random_skip_max} "
        f"train_npy_path={train_npy_path} val_npy_path={val_npy_path} "
        f"run_test={run_test} test_npy_path={test_npy_path} test_match_threshold={test_match_threshold} "
        f"test_score_threshold={test_score_threshold} test_viz_out_path={test_viz_out_path} "
        f"test_viz_frames_dir={test_viz_frames_dir} "
        f"direct_npy_test_viz_out_dir={direct_npy_test_viz_out_dir} "
        f"direct_npy_test_viz_frames_dir={direct_npy_test_viz_frames_dir} "
        f"best_test_viz_out_path={best_test_viz_out_path} "
        f"best_post_test_viz_out_path={best_post_test_viz_out_path} "
        f"best_test_viz_frames_dir=None "
        f"best_post_test_viz_frames_dir=None "
        f"test_viz_seq_idx={test_viz_seq_idx} test_viz_fps={test_viz_fps} "
        f"test_use_ema_updates={test_use_ema_updates} test_ema_momentum={test_ema_momentum} "
        f"use_history={use_history} "
        f"test_use_normalized_dists={test_use_normalized_dists} test_max_frames={test_max_frames} "
        f"test_track_max_age={test_track_max_age} test_min_track_len={test_min_track_len} "
        f"test_emit_videos={test_emit_videos} "
        f"test_calc_false_positives={test_calc_false_positives} "
        f"test_preds_dir={test_preds_dir} test_preds_no_post_dir={test_preds_no_post_dir} "
        f"run_trackeval={run_trackeval} "
        f"trackeval_each_epoch={trackeval_each_epoch}"
    )
    test_dl = None
    test_ds = None
    if run_test:
        test_ds = MOTSeqDataset(test_npy_path, seq_len=test_seq_len, stride=stride)
        test_dl = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda x: x[0],
        )
    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(
            model,
            train_dl,
            opt,
            margin=margin,
            max_hist_items=max_hist_items,
            use_history=use_history,
            use_cross_attn=use_cross_attn,
            use_cross_attn_mask=use_cross_attn_mask,
            use_coord_time_embeds=use_coord_time_embeds,
            id_switch_weight=id_switch_weight,
            id_switch_margin=id_switch_margin,
            new_id_repel_weight=new_id_repel_weight,
            new_id_repel_margin=new_id_repel_margin,
        )
        val_loss = eval_one_epoch(
            model,
            val_dl,
            margin=margin,
            max_hist_items=max_hist_items,
            use_history=use_history,
            use_cross_attn=use_cross_attn,
            use_cross_attn_mask=use_cross_attn_mask,
            use_coord_time_embeds=use_coord_time_embeds,
            id_switch_weight=id_switch_weight,
            id_switch_margin=id_switch_margin,
            new_id_repel_weight=new_id_repel_weight,
            new_id_repel_margin=new_id_repel_margin,
        )
        print(
            f"[Epoch {epoch:02d}] Train Contrastive Loss={tr_loss:.4f} | "
            f"Val Contrastive Loss={val_loss:.4f}"
        )
        scheduler.step(val_loss)
        if run_test and run_trackeval and trackeval_each_epoch:
            if test_ds is not None:
                export_test_predictions_mot(
                    model,
                    test_ds,
                    test_preds_dir,
                    match_threshold=test_match_threshold,
                    score_threshold=test_score_threshold,
                    use_cross_attn=use_cross_attn,
                    use_cross_attn_mask=use_cross_attn_mask,
                    use_coord_time_embeds=use_coord_time_embeds,
                    use_ema_updates=test_use_ema_updates,
                    ema_momentum=test_ema_momentum,
                    use_normalized_dists=test_use_normalized_dists,
                    test_max_frames=test_max_frames,
                    test_track_max_age=test_track_max_age,
                    test_min_track_len=test_min_track_len,
                    use_history=use_history,
                )
                export_test_predictions_mot(
                    model,
                    test_ds,
                    test_preds_no_post_dir,
                    match_threshold=test_match_threshold,
                    score_threshold=test_score_threshold,
                    use_cross_attn=use_cross_attn,
                    use_cross_attn_mask=use_cross_attn_mask,
                    use_coord_time_embeds=use_coord_time_embeds,
                    use_ema_updates=test_use_ema_updates,
                    ema_momentum=test_ema_momentum,
                    use_normalized_dists=test_use_normalized_dists,
                    test_max_frames=test_max_frames,
                    test_track_max_age=test_track_max_age,
                    test_min_track_len=test_min_track_len,
                    use_history=use_history,
                    apply_postprocess=False,
                )
            trackeval_metrics = run_trackeval_for_tracker(track_tag, show_output=True)
            if trackeval_metrics is not None:
                print(f"[trackeval][epoch={epoch:02d}] {format_trackeval_metrics(trackeval_metrics)}")
            hota = trackeval_metrics.get("HOTA") if trackeval_metrics is not None else None
            if hota is not None and hota > best_hota:
                best_hota = hota
                # Checkpoint saving disabled for now.
                if run_test and test_dl is not None and test_emit_videos:
                    eval_one_epoch(
                        model,
                        test_dl,
                        max_hist_items=max_hist_items,
                        use_history=use_history,
                        use_cross_attn=use_cross_attn,
                        use_cross_attn_mask=use_cross_attn_mask,
                        use_coord_time_embeds=use_coord_time_embeds,
                        mode="test",
                        match_threshold=test_match_threshold,
                        score_threshold=test_score_threshold,
                        generate_video=test_emit_videos,
                        video_out_path=best_test_viz_out_path,
                        video_seq_idx=test_viz_seq_idx,
                        video_fps=test_viz_fps,
                        use_ema_updates=test_use_ema_updates,
                        ema_momentum=test_ema_momentum,
                        use_normalized_dists=test_use_normalized_dists,
                        test_max_frames=test_max_frames,
                        test_track_max_age=test_track_max_age,
                        calc_false_positives=test_calc_false_positives,
                    )
                    eval_one_epoch(
                        model,
                        test_dl,
                        max_hist_items=max_hist_items,
                        use_history=use_history,
                        use_cross_attn=use_cross_attn,
                        use_cross_attn_mask=use_cross_attn_mask,
                        use_coord_time_embeds=use_coord_time_embeds,
                        mode="test",
                        match_threshold=test_match_threshold,
                        score_threshold=test_score_threshold,
                        generate_video=test_emit_videos,
                        video_out_path=best_post_test_viz_out_path,
                        video_seq_idx=test_viz_seq_idx,
                        video_fps=test_viz_fps,
                        use_ema_updates=test_use_ema_updates,
                        ema_momentum=test_ema_momentum,
                        use_normalized_dists=test_use_normalized_dists,
                        test_max_frames=test_max_frames,
                        test_track_max_age=test_track_max_age,
                        video_post_min_track_len=test_min_track_len,
                        calc_false_positives=test_calc_false_positives,
                    )
        if val_loss < best_val_loss - es_min_delta:
            best_val_loss = val_loss
            no_improve = 0
            if run_test and test_dl is not None and test_emit_videos:
                improve_paths = _make_improve_viz_paths()
                base_dir = os.path.dirname(os.path.dirname(improve_paths["raw_video"]))
                _reset_improve_viz_dirs(base_dir)
                if test_ds is not None:
                    for seq_idx, vid in _iter_test_video_seq_indices(test_ds):
                        raw_video = os.path.join(os.path.dirname(improve_paths["raw_video"]), vid, "video.mp4")
                        raw_frames = os.path.join(os.path.dirname(improve_paths["raw_frames"]), vid, "frames")
                        eval_one_epoch(
                            model,
                            test_dl,
                            max_hist_items=max_hist_items,
                            use_history=use_history,
                            use_cross_attn=use_cross_attn,
                            use_cross_attn_mask=use_cross_attn_mask,
                            use_coord_time_embeds=use_coord_time_embeds,
                            mode="test",
                            match_threshold=test_match_threshold,
                            score_threshold=test_score_threshold,
                            generate_video=test_emit_videos,
                            video_out_path=raw_video,
                            frames_out_dir=raw_frames,
                            video_seq_idx=seq_idx,
                            video_fps=test_viz_fps,
                            use_ema_updates=test_use_ema_updates,
                            ema_momentum=test_ema_momentum,
                            use_normalized_dists=test_use_normalized_dists,
                            test_max_frames=test_max_frames,
                            test_track_max_age=test_track_max_age,
                            calc_false_positives=test_calc_false_positives,
                        )
                        post_video = os.path.join(os.path.dirname(improve_paths["post_video"]), vid, "video.mp4")
                        post_frames = os.path.join(os.path.dirname(improve_paths["post_frames"]), vid, "frames")
                        eval_one_epoch(
                            model,
                            test_dl,
                            max_hist_items=max_hist_items,
                            use_history=use_history,
                            use_cross_attn=use_cross_attn,
                            use_cross_attn_mask=use_cross_attn_mask,
                            use_coord_time_embeds=use_coord_time_embeds,
                            mode="test",
                            match_threshold=test_match_threshold,
                            score_threshold=test_score_threshold,
                            generate_video=test_emit_videos,
                            video_out_path=post_video,
                            frames_out_dir=post_frames,
                            video_seq_idx=seq_idx,
                            video_fps=test_viz_fps,
                            use_ema_updates=test_use_ema_updates,
                            ema_momentum=test_ema_momentum,
                            use_normalized_dists=test_use_normalized_dists,
                            test_max_frames=test_max_frames,
                            test_track_max_age=test_track_max_age,
                            video_post_min_track_len=test_min_track_len,
                            calc_false_positives=test_calc_false_positives,
                        )
            if not trackeval_each_epoch:
                # Checkpoint saving disabled for now.
                if run_test and test_dl is not None:
                    export_test_predictions_mot(
                        model,
                        test_ds,
                        test_preds_dir,
                        match_threshold=test_match_threshold,
                        score_threshold=test_score_threshold,
                        use_cross_attn=use_cross_attn,
                        use_cross_attn_mask=use_cross_attn_mask,
                        use_coord_time_embeds=use_coord_time_embeds,
                        use_ema_updates=test_use_ema_updates,
                        ema_momentum=test_ema_momentum,
                        use_normalized_dists=test_use_normalized_dists,
                        test_max_frames=test_max_frames,
                        test_track_max_age=test_track_max_age,
                        test_min_track_len=test_min_track_len,
                        use_history=use_history,
                    )
                    export_test_predictions_mot(
                        model,
                        test_ds,
                        test_preds_no_post_dir,
                        match_threshold=test_match_threshold,
                        score_threshold=test_score_threshold,
                        use_cross_attn=use_cross_attn,
                        use_cross_attn_mask=use_cross_attn_mask,
                        use_coord_time_embeds=use_coord_time_embeds,
                        use_ema_updates=test_use_ema_updates,
                        ema_momentum=test_ema_momentum,
                        use_normalized_dists=test_use_normalized_dists,
                        test_max_frames=test_max_frames,
                        test_track_max_age=test_track_max_age,
                        test_min_track_len=test_min_track_len,
                        use_history=use_history,
                        apply_postprocess=False,
                    )
                    if test_emit_videos:
                        eval_one_epoch(
                            model,
                            test_dl,
                            max_hist_items=max_hist_items,
                            use_history=use_history,
                            use_cross_attn=use_cross_attn,
                            use_cross_attn_mask=use_cross_attn_mask,
                            use_coord_time_embeds=use_coord_time_embeds,
                            mode="test",
                            match_threshold=test_match_threshold,
                            score_threshold=test_score_threshold,
                            generate_video=test_emit_videos,
                            video_out_path=best_test_viz_out_path,
                            video_seq_idx=test_viz_seq_idx,
                            video_fps=test_viz_fps,
                            use_ema_updates=test_use_ema_updates,
                            ema_momentum=test_ema_momentum,
                            use_normalized_dists=test_use_normalized_dists,
                            test_max_frames=test_max_frames,
                            test_track_max_age=test_track_max_age,
                            calc_false_positives=test_calc_false_positives,
                        )
                        eval_one_epoch(
                            model,
                            test_dl,
                            max_hist_items=max_hist_items,
                            use_history=use_history,
                            use_cross_attn=use_cross_attn,
                            use_cross_attn_mask=use_cross_attn_mask,
                            use_coord_time_embeds=use_coord_time_embeds,
                            mode="test",
                            match_threshold=test_match_threshold,
                            score_threshold=test_score_threshold,
                            generate_video=test_emit_videos,
                            video_out_path=best_post_test_viz_out_path,
                            video_seq_idx=test_viz_seq_idx,
                            video_fps=test_viz_fps,
                            use_ema_updates=test_use_ema_updates,
                            ema_momentum=test_ema_momentum,
                            use_normalized_dists=test_use_normalized_dists,
                            test_max_frames=test_max_frames,
                            test_track_max_age=test_track_max_age,
                            video_post_min_track_len=test_min_track_len,
                            calc_false_positives=test_calc_false_positives,
                        )
        else:
            no_improve += 1
            if no_improve >= es_patience:
                print(f"[EarlyStop] No improvement in {es_patience} epochs. Best={best_val_loss:.4f}")
                break

    # no save of last checkpoint to avoid disk usage
    if run_test and test_dl is not None and test_emit_videos:
        eval_one_epoch(
            model,
            test_dl,
            max_hist_items=max_hist_items,
            use_history=use_history,
            use_cross_attn=use_cross_attn,
            use_cross_attn_mask=use_cross_attn_mask,
            use_coord_time_embeds=use_coord_time_embeds,
            mode="test",
            match_threshold=test_match_threshold,
            score_threshold=test_score_threshold,
            generate_video=test_emit_videos,
            video_out_path=test_viz_out_path,
            frames_out_dir=test_viz_frames_dir,
            video_seq_idx=test_viz_seq_idx,
            video_fps=test_viz_fps,
            use_ema_updates=test_use_ema_updates,
            ema_momentum=test_ema_momentum,
            use_normalized_dists=test_use_normalized_dists,
            test_max_frames=test_max_frames,
            test_track_max_age=test_track_max_age,
            video_post_min_track_len=test_min_track_len,
            calc_false_positives=test_calc_false_positives,
        )
    if run_test and run_trackeval and not trackeval_each_epoch:
        final_metrics = run_trackeval_for_tracker(track_tag)
        if final_metrics is not None:
            print(f"[trackeval][final] {format_trackeval_metrics(final_metrics)}")
        else:
            print("[trackeval][final] No se pudieron parsear metricas.")


if __name__ == "__main__":
    main()

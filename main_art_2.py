import os
import random
import re
import subprocess
import sys
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MOTSeqDataset(Dataset):
    def __init__(
        self,
        npy_path: str,
        seq_len: int = 30,
        stride: int = 30,
        score_threshold: Optional[float] = None,
    ):
        self.data = np.load(npy_path, allow_pickle=True).item()
        self.seq_len = seq_len
        self.stride = stride
        self.score_threshold = score_threshold

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
        for vid, frames in videos.items():
            n = len(frames)
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
            if scores.numel() == 0:
                print(f"[loader] empty scores for {path}")

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
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

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
        # Expected shapes: (batch, seq, embed_dim)
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
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = SigmoidMultiheadAttention(d_model, nhead, dropout=dropout)
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
    ) -> torch.Tensor:
        sa_out, _ = self.self_attn(x, x, x, key_padding_mask=self_key_padding_mask, need_weights=False)
        x = self.norm1(x + self.dropout(sa_out))

        ca_out, _ = self.cross_attn(
            x,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
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
        )

    def forward(
        self,
        x: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
        frame_idx: Optional[torch.Tensor] = None,
        hist_z: Optional[torch.Tensor] = None,
        use_cross: bool = False,
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
        z = self.cross_block(q, kv).squeeze(0)
        return F.normalize(z, dim=-1)


def triplet_loss_with_history(
    z_cur: torch.Tensor,
    ids_cur: torch.Tensor,
    z_hist: Optional[torch.Tensor],
    ids_hist: Optional[torch.Tensor],
    margin: float = 0.2,
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

    losses = []
    for i in range(n):
        pos_mask = ids_hist == ids_cur[i]
        neg_mask = ids_hist != ids_cur[i]
        if not pos_mask.any() or not neg_mask.any():
            continue
        dists = torch.norm(z_hist - z_cur[i], dim=-1)
        pos_dist = dists[pos_mask].max()
        neg_dist = dists[neg_mask].min()
        losses.append(F.relu(pos_dist - neg_dist + margin))

    if len(losses) == 0:
        return z_cur.sum() * 0.0
    return torch.stack(losses).mean()


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
        loss = loss + (pos_d ** 2).mean()
    if neg_mask.any():
        neg_d = dists[neg_mask]
        loss = loss + F.relu(margin - neg_d).pow(2).mean()
    return loss


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    w = (boxes[:, 2] - boxes[:, 0]).clamp(min=0.0)
    h = (boxes[:, 3] - boxes[:, 1]).clamp(min=0.0)
    return w * h


def giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x1 = torch.max(pred[:, 0], target[:, 0])
    y1 = torch.max(pred[:, 1], target[:, 1])
    x2 = torch.min(pred[:, 2], target[:, 2])
    y2 = torch.min(pred[:, 3], target[:, 3])

    inter = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
    area_p = _box_area(pred)
    area_t = _box_area(target)
    union = area_p + area_t - inter
    iou = inter / (union + 1e-7)

    cx1 = torch.min(pred[:, 0], target[:, 0])
    cy1 = torch.min(pred[:, 1], target[:, 1])
    cx2 = torch.max(pred[:, 2], target[:, 2])
    cy2 = torch.max(pred[:, 3], target[:, 3])
    c_area = (cx2 - cx1).clamp(min=0.0) * (cy2 - cy1).clamp(min=0.0)
    giou = iou - (c_area - union) / (c_area + 1e-7)
    return (1.0 - giou).mean()


def iou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x1 = torch.max(pred[:, 0], target[:, 0])
    y1 = torch.max(pred[:, 1], target[:, 1])
    x2 = torch.min(pred[:, 2], target[:, 2])
    y2 = torch.min(pred[:, 3], target[:, 3])

    inter = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
    area_p = _box_area(pred)
    area_t = _box_area(target)
    union = area_p + area_t - inter
    iou = inter / (union + 1e-7)
    return (1.0 - iou).mean()


def _pairwise_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device, dtype=boxes1.dtype)
    b1 = boxes1.view(-1, 4)
    b2 = boxes2.view(-1, 4)
    x1 = torch.max(b1[:, None, 0], b2[None, :, 0])
    y1 = torch.max(b1[:, None, 1], b2[None, :, 1])
    x2 = torch.min(b1[:, None, 2], b2[None, :, 2])
    y2 = torch.min(b1[:, None, 3], b2[None, :, 3])
    inter = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
    area1 = _box_area(b1)
    area2 = _box_area(b2)
    union = area1[:, None] + area2[None, :] - inter
    return torch.where(union > 0.0, inter / (union + 1e-7), torch.zeros_like(union))


def _pairwise_dist(x: torch.Tensor, y: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    if normalize:
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
    return torch.cdist(x, y)


def _standardize_pred_boxes(
    pred: torch.Tensor,
    target: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if pred.numel() == 0:
        return pred
    pred_t = pred.view(-1, 4)
    if target is not None and target.numel() > 0:
        try:
            t_max = float(target.max().item())
        except RuntimeError:
            t_max = float("inf")
        if t_max <= 1.5:
            pred_t = pred_t.sigmoid()
    x1 = torch.minimum(pred_t[:, 0], pred_t[:, 2])
    y1 = torch.minimum(pred_t[:, 1], pred_t[:, 3])
    x2 = torch.maximum(pred_t[:, 0], pred_t[:, 2])
    y2 = torch.maximum(pred_t[:, 1], pred_t[:, 3])
    return torch.stack([x1, y1, x2, y2], dim=-1)


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


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    margin: float = 0.2,
    max_hist_items: int = 5000,
    log_every: int = 500,
    log_frames_prob: float = 0.02,
    log_once_per_epoch: bool = True,
    log_distances: bool = False,
    use_cross_attn: bool = False,
    use_coord_time_embeds: bool = False,
    id_switch_weight: float = 1.0,
    id_switch_margin: float = 0.2,
    use_box_pred_loss: bool = True,
    box_loss_weight: float = 1.0,
    box_l1_weight: float = 1.0,
    box_iou_weight: float = 1.0,
    use_giou_loss: bool = True,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_box_loss = 0.0
    total_box_items = 0
    total_seqs = 0
    logged_dist = False

    for batch_idx, seq in enumerate(tqdm(dataloader, desc="train", leave=False), start=1):
        """
        if log_frames_prob > 0 and random.random() < log_frames_prob:
            frame_names = [os.path.basename(f.get("path", "")) for f in seq]
            print(f"[Batch {batch_idx}] frames={frame_names}")
        """

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
                hist_z=hist_z,
                use_cross=use_cross_attn,
                use_coord_time=use_coord_time_embeds,
            )
            loss = contrastive_loss_with_history(z, ids, hist_z, hist_ids, margin=margin)
            if use_box_pred_loss and boxes_t is not None and boxes_t.numel() > 0:
                pred_boxes = model.proj_boxes(z)
                target_boxes = boxes_t[: pred_boxes.shape[0]]
                pred_boxes = _standardize_pred_boxes(pred_boxes, target_boxes)
                if use_giou_loss:
                    box_loss = giou_loss(pred_boxes, target_boxes)
                else:
                    box_loss = iou_loss(pred_boxes, target_boxes)
                l1_loss = F.l1_loss(pred_boxes, target_boxes)
                box_term = box_loss_weight * (box_iou_weight * box_loss + box_l1_weight * l1_loss)
                loss = loss + box_term
                total_box_loss += float(box_term.item())
                total_box_items += 1
            if last_z is not None and last_ids is not None and id_switch_weight != 0.0:
                switch_loss = id_switch_loss(z, ids, last_z, last_ids, margin=id_switch_margin)
                loss = loss + id_switch_weight * switch_loss
            seq_loss = loss if seq_loss is None else (seq_loss + loss)
            seq_loss_frames += 1
            last_z = z
            last_ids = ids

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

        if (
            log_distances
            and log_every > 0
            and (batch_idx % log_every == 0 or (log_once_per_epoch and not logged_dist))
            and hist_z is not None
            and hist_ids is not None
            and hist_z.numel() > 0
            and hist_ids.numel() > 0
            and last_z is not None
            and last_ids is not None
        ):
            with torch.no_grad():
                dists = torch.cdist(last_z, hist_z)
                same_inst = (last_ids[:, None] == hist_ids[None, :]) & (dists < 1e-6)
                if same_inst.any():
                    dists = dists.masked_fill(same_inst, float("inf"))
                ids_cpu = last_ids.detach().cpu().tolist()
                hist_ids_cpu = hist_ids.detach().cpu().tolist()
                below = (dists < 1.0).detach().cpu()
                for i, row_id in enumerate(ids_cpu):
                    col_idxs = torch.nonzero(below[i], as_tuple=False).flatten().tolist()
                    col_ids = [hist_ids_cpu[j] for j in col_idxs]
                    print(f"[Batch {batch_idx}] row_id={row_id} cols_lt1={col_ids}")
                logged_dist = True

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
    avg_box_loss = total_box_loss / max(total_box_items, 1)
    return avg_loss, avg_box_loss


def evaluate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    margin: float = 0.2,
    max_hist_items: int = 5000,
    acc_threshold: Optional[float] = None,
    use_cross_attn: bool = False,
    use_coord_time_embeds: bool = False,
    use_box_pred_loss: bool = True,
    box_loss_weight: float = 1.0,
    box_l1_weight: float = 1.0,
    box_iou_weight: float = 1.0,
    use_giou_loss: bool = True,
    score_threshold: Optional[float] = None,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_box_loss = 0.0
    total_box_items = 0
    total_seqs = 0
    acc_correct = 0
    acc_total = 0
    thresh = margin if acc_threshold is None else acc_threshold
    printed_dist = False
    printed_pred_boxes = False
    printed_box_loss = False

    with torch.no_grad():
        for seq in dataloader:
            unique_ids = set()
            for frame in seq:
                ids = frame["ids"]
                if torch.is_tensor(ids) and ids.numel() > 0:
                    unique_ids.update(ids.view(-1).tolist())
            if len(unique_ids) < 2:
                continue

            hist_z = None
            hist_ids = None
            seq_loss = 0.0
            has_loss = False
            seq_loss_frames = 0
            prev_z = None

            seq_len = len(seq)
            for t, frame in enumerate(seq):
                queries = frame["queries"].to(device)
                ids = frame["ids"].to(device)
                scores = frame.get("scores")
                boxes = frame.get("boxes")
                boxes_t = None
                if boxes is not None:
                    boxes_t = boxes if torch.is_tensor(boxes) else torch.as_tensor(boxes, dtype=torch.float32)
                    boxes_t = boxes_t.to(device, dtype=torch.float32)

                if queries.numel() == 0 or ids.numel() == 0:
                    continue

                queries, ids, boxes_t, scores = apply_score_threshold(
                    queries,
                    ids=ids,
                    boxes=boxes_t,
                    scores=scores,
                    threshold=score_threshold,
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
                if boxes_t is not None:
                    boxes_t = boxes_t[:n]

                frame_pos = 0.0 if seq_len <= 1 else float(t) / float(seq_len - 1)
                frame_pos_t = torch.full((n, 1), frame_pos, device=device)
                z = model(
                    queries,
                    boxes=boxes_t,
                    frame_idx=frame_pos_t,
                    hist_z=hist_z,
                    use_cross=use_cross_attn,
                    use_coord_time=use_coord_time_embeds,
                )
                if use_box_pred_loss and boxes_t is not None and boxes_t.numel() > 0:
                    pred_boxes = model.proj_boxes(z)
                    target_boxes = boxes_t[: pred_boxes.shape[0]]
                    pred_boxes = _standardize_pred_boxes(pred_boxes, target_boxes)
                    if use_giou_loss:
                        box_loss = giou_loss(pred_boxes, target_boxes)
                    else:
                        box_loss = iou_loss(pred_boxes, target_boxes)
                    l1_loss = F.l1_loss(pred_boxes, target_boxes)
                    if not printed_box_loss:
                        print(
                            f"[eval] box_loss={box_loss.item():.6f} "
                            f"l1_loss={l1_loss.item():.6f} "
                            f"use_giou={use_giou_loss}"
                        )
                        printed_box_loss = True
                    box_term = box_loss_weight * (box_iou_weight * box_loss + box_l1_weight * l1_loss)
                    seq_loss += float(box_term.item())
                    total_box_loss += float(box_term.item())
                    total_box_items += 1
                if (
                    prev_z is not None
                    and not printed_dist
                    and z.shape[0] > 1
                    and prev_z.shape[0] > 1
                ):
                    dists = torch.cdist(z, prev_z)
                    dists_np = dists.detach().cpu().numpy()
                    print(
                        f"[eval] track-det distance matrix (tracks={prev_z.shape[0]} x dets={z.shape[0]}):"
                    )
                    print(dists_np)
                    printed_dist = True
                loss = contrastive_loss_with_history(z, ids, hist_z, hist_ids, margin=margin)
                seq_loss += float(loss.item())
                has_loss = True
                seq_loss_frames += 1

                if hist_z is not None and hist_ids is not None and hist_z.numel() > 0:
                    dists = torch.cdist(z, hist_z)
                    same_mask = ids[:, None] == hist_ids[None, :]
                    pred_same = dists < thresh
                    acc_correct += (pred_same == same_mask).sum().item()
                    acc_total += dists.numel()

                if hist_z is None:
                    hist_z = z
                    hist_ids = ids
                else:
                    hist_z = torch.cat([hist_z, z], dim=0)
                    hist_ids = torch.cat([hist_ids, ids], dim=0)
                prev_z = z

                if hist_z.shape[0] > max_hist_items:
                    hist_z = hist_z[-max_hist_items:]
                    hist_ids = hist_ids[-max_hist_items:]

            if not has_loss:
                continue

            seq_loss = seq_loss / max(seq_loss_frames, 1)
            total_loss += seq_loss
            total_seqs += 1

    avg_loss = total_loss / max(total_seqs, 1)
    acc = acc_correct / max(acc_total, 1)
    avg_box_loss = total_box_loss / max(total_box_items, 1)
    return avg_loss, acc, avg_box_loss


def visualize_val_sequence(
    model: nn.Module,
    dataset: MOTSeqDataset,
    out_path: str,
    seq_idx: int = 0,
    fps: int = 10,
    max_frames: Optional[int] = None,
    match_threshold: float = 1.0,
    ema_momentum: float = 0.95,
    use_ema_updates: bool = True,
    use_track_emb_boxes: bool = False,
    show_gt: bool = False,
    score_threshold: Optional[float] = None,
    show_scores: bool = False,
    use_coord_time_embeds: bool = False,
    spatial_weight: float = 0.0,
    min_iou: Optional[float] = None,
    use_normalized_dists: bool = False,
) -> None:
    try:
        import cv2
    except Exception:
        return

    if model is None:
        return

    if seq_idx < 0 or seq_idx >= len(dataset):
        return

    frames = dataset[seq_idx]
    if max_frames is not None:
        frames = frames[:max_frames]

    first_img = None
    for frame in frames:
        img_path = frame.get("path", "")
        if os.path.exists(img_path):
            first_img = cv2.imread(img_path)
            if first_img is not None:
                break

    if first_img is None:
        return

    h, w = first_img.shape[:2]
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    writer = None
    for fourcc_code in ("mp4v", "avc1", "H264", "XVID"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if writer.isOpened():
            break
        writer.release()
        writer = None
    if writer is None:
        return

    model.eval()
    device = next(model.parameters()).device
    track_embs = None
    track_ids: List[int] = []
    next_track_id = 0
    track_pred_boxes: Dict[int, torch.Tensor] = {}
    blank = np.zeros((h, w, 3), dtype=np.uint8)

    def _scale_xyxy_to_image(box: torch.Tensor, w: int, h: int) -> torch.Tensor:
        b = box.clone()
        if b.numel() != 4:
            return b
        if b.max().item() <= 1.5:
            scale = torch.tensor([w, h, w, h], dtype=b.dtype, device=b.device)
            b = b * scale
        return b

    def _scale_xyxy_batch(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes
        if boxes.max().item() <= 1.5:
            scale = torch.tensor([w, h, w, h], dtype=boxes.dtype, device=boxes.device)
            return boxes * scale
        return boxes

    with torch.no_grad():
        total_frames = len(frames)
        for t, frame in enumerate(frames):
            img_path = frame.get("path", "")
            img = None
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
            if img is None:
                img = blank.copy()
                cv2.putText(
                    img,
                    "missing frame",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            boxes = frame.get("boxes")
            queries = frame.get("queries")
            scores = frame.get("scores")
            gt_ids = frame.get("ids")
            #print('scores:',scores)
            if boxes is not None and queries is not None:
                q_t = queries if torch.is_tensor(queries) else torch.as_tensor(queries)
                b_t = boxes if torch.is_tensor(boxes) else torch.as_tensor(boxes)
                ids_t = (
                    gt_ids
                    if torch.is_tensor(gt_ids)
                    else (torch.as_tensor(gt_ids) if gt_ids is not None else None)
                )
                #print('llegaa')
                #print('score_th:',score_threshold)
                q_t, ids_t, b_t, scores_t = apply_score_threshold(
                    q_t,
                    ids=ids_t,
                    boxes=b_t,
                    scores=scores,
                    threshold=score_threshold,
                )
                #print('q_t:',q_t)
                if q_t.numel() > 0 and b_t.numel() > 0:
                    boxes_np = b_t.detach().cpu().numpy()
                    n = min(len(boxes_np), q_t.shape[0])
                    scores_np = None
                    if scores_t is not None and not torch.is_tensor(scores_t):
                        scores_t = torch.as_tensor(scores_t, dtype=torch.float32)
                    if torch.is_tensor(scores_t):
                        scores_np = scores_t.detach().cpu().numpy()
                    if n > 0:
                        queries_t = q_t[:n].to(device)
                        frame_pos = 0.0 if total_frames <= 1 else float(t) / float(total_frames - 1)
                        frame_pos_t = torch.full((n, 1), frame_pos, device=device)
                        z = model(
                            queries_t,
                            boxes=b_t[:n].to(device),
                            frame_idx=frame_pos_t,
                            use_coord_time=use_coord_time_embeds,
                        )

                        assigned_ids = [-1] * n
                        if track_embs is None or track_embs.numel() == 0:
                            assigned_ids = list(range(next_track_id, next_track_id + n))
                            track_embs = z.detach().clone()
                            track_ids.extend(assigned_ids)
                            next_track_id += n
                            for i in range(n):
                                if use_track_emb_boxes:
                                    pred_box = model.proj_boxes(z[i : i + 1].detach()).squeeze(0)
                                    ref_box = b_t[i : i + 1].to(pred_box.device) if b_t is not None else None
                                    pred_box = _standardize_pred_boxes(pred_box, ref_box).squeeze(0)
                                    pred_box = _scale_xyxy_to_image(pred_box, w, h)
                                    track_pred_boxes[assigned_ids[i]] = pred_box.detach()
                                else:
                                    track_pred_boxes[assigned_ids[i]] = b_t[i].detach().to(device)
                        else:
                            dists = _pairwise_dist(z, track_embs, normalize=use_normalized_dists)
                            cost = dists
                            valid_mask = dists < match_threshold
                            if spatial_weight > 0.0 and track_pred_boxes:
                                det_boxes_cost = b_t[:n].to(device)
                                if use_track_emb_boxes:
                                    det_boxes_cost = _scale_xyxy_batch(det_boxes_cost, w, h)
                                track_boxes_list = [track_pred_boxes[tid] for tid in track_ids]
                                if track_boxes_list:
                                    track_boxes_t = torch.stack(track_boxes_list, dim=0).to(device)
                                    ious = _pairwise_iou_xyxy(det_boxes_cost, track_boxes_t)
                                    cost = dists + spatial_weight * (1.0 - ious)
                                    if min_iou is not None:
                                        valid_mask = valid_mask & (ious >= min_iou)
                            pairs = []
                            for i in range(n):
                                for j in range(track_embs.shape[0]):
                                    if not valid_mask[i, j]:
                                        continue
                                    d = float(cost[i, j].item())
                                    pairs.append((d, i, j))
                            pairs.sort(key=lambda x: x[0])
                            used_tracks = set()
                            used_dets = set()
                            for _, i, j in pairs:
                                if i in used_dets or j in used_tracks:
                                    continue
                                assigned_ids[i] = track_ids[j]
                                used_dets.add(i)
                                used_tracks.add(j)
                                if use_ema_updates:
                                    update = ema_momentum * track_embs[j] + (1.0 - ema_momentum) * z[i].detach()
                                else:
                                    update = z[i].detach()
                                track_embs[j] = (
                                    F.normalize(update, dim=-1) if use_normalized_dists else update
                                )
                                if use_track_emb_boxes:
                                    pred_box = model.proj_boxes(z[i : i + 1].detach()).squeeze(0)
                                    ref_box = b_t[i : i + 1].to(pred_box.device) if b_t is not None else None
                                    pred_box = _standardize_pred_boxes(pred_box, ref_box).squeeze(0)
                                    pred_box = _scale_xyxy_to_image(pred_box, w, h)
                                    track_pred_boxes[track_ids[j]] = pred_box.detach()
                            for i in range(n):
                                if assigned_ids[i] == -1:
                                    assigned_ids[i] = next_track_id
                                    next_track_id += 1
                                    track_ids.append(assigned_ids[i])
                                    track_embs = torch.cat(
                                        [track_embs, z[i].detach().unsqueeze(0)], dim=0
                                    )
                                    if use_track_emb_boxes:
                                        pred_box = model.proj_boxes(z[i : i + 1].detach()).squeeze(0)
                                        ref_box = b_t[i : i + 1].to(pred_box.device) if b_t is not None else None
                                        pred_box = _standardize_pred_boxes(pred_box, ref_box).squeeze(0)
                                        pred_box = _scale_xyxy_to_image(pred_box, w, h)
                                        track_pred_boxes[assigned_ids[i]] = pred_box.detach()
                                    else:
                                        track_pred_boxes[assigned_ids[i]] = b_t[i].detach().to(device)

                        for i in range(n):
                            x1, y1, x2, y2 = boxes_np[i].tolist()
                            tid = int(assigned_ids[i])
                            if use_track_emb_boxes and tid in track_pred_boxes:
                                x1, y1, x2, y2 = track_pred_boxes[tid].detach().cpu().tolist()
                            cv2.rectangle(
                                img,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                (0, 255, 255),
                                2,
                            )
                            cv2.putText(
                                img,
                                f"pred {tid}",
                                (int(x1), max(int(y1) - 5, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )
                            if show_scores and scores_np is not None and i < len(scores_np):
                                cv2.putText(
                                    img,
                                    f"s {scores_np[i]:.2f}",
                                    (int(x1), min(int(y2) + 30, h - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 0),
                                    1,
                                    cv2.LINE_AA,
                                )

                            if show_gt and ids_t is not None and i < len(ids_t):
                                gt_id = int(ids_t[i]) if torch.is_tensor(ids_t) else int(ids_t[i])
                                cv2.putText(
                                    img,
                                    f"gt {gt_id}",
                                    (int(x1), min(int(y2) + 15, h - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    1,
                                    cv2.LINE_AA,
                                )

            cv2.putText(
                img,
                f"frame {t}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(img)

    writer.release()


def visualize_loader_predictions(
    model: Optional[nn.Module],
    dataset: MOTSeqDataset,
    out_path: str,
    seq_idx: int = 0,
    fps: int = 10,
    max_frames: Optional[int] = None,
    show_scores: bool = True,
    min_track_seconds: float = 5.0,
    forget_after_frames: Optional[int] = None,
    use_ema_updates: bool = True,
    use_track_emb_boxes: bool = False,
    spatial_weight: float = 0.0,
    min_iou: Optional[float] = None,
    use_normalized_dists: bool = False,
    match_threshold: float = 1.0,
    score_threshold: Optional[float] = None,
    use_cross_attn: bool = False,
    use_coord_time_embeds: bool = False,
    allow_new_tracks: bool = True,
) -> None:
    try:
        import cv2
    except Exception:
        return

    if seq_idx < 0 or seq_idx >= len(dataset):
        return

    frames = dataset[seq_idx]
    if max_frames is not None:
        frames = frames[:max_frames]

    first_img = None
    for frame in frames:
        img_path = frame.get("path", "")
        if os.path.exists(img_path):
            first_img = cv2.imread(img_path)
            if first_img is not None:
                break

    if first_img is None:
        return

    h, w = first_img.shape[:2]
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        return

    def _to_xyxy(box: torch.Tensor, w: int, h: int) -> torch.Tensor:
        b = box.clone()
        if b.numel() != 4:
            return b
        if b.max().item() <= 1.5:
            b = b * torch.tensor([w, h, w, h], dtype=b.dtype, device=b.device)
        xyxy = b
        xywh = torch.stack([b[0], b[1], b[0] + b[2], b[1] + b[3]])
        cxcywh = torch.stack(
            [b[0] - b[2] / 2.0, b[1] - b[3] / 2.0, b[0] + b[2] / 2.0, b[1] + b[3] / 2.0]
        )

        def _oob_score(cand: torch.Tensor) -> torch.Tensor:
            invalid = (cand[2] < cand[0]) | (cand[3] < cand[1])
            oob = (
                (-cand[0]).clamp(min=0)
                + (-cand[1]).clamp(min=0)
                + (cand[2] - w).clamp(min=0)
                + (cand[3] - h).clamp(min=0)
            )
            return oob + invalid.float() * (w + h)

        scores = torch.stack([_oob_score(xyxy), _oob_score(xywh), _oob_score(cxcywh)])
        best = int(torch.argmin(scores).item())
        b = [xyxy, xywh, cxcywh][best]
        b[0] = b[0].clamp(0, w)
        b[2] = b[2].clamp(0, w)
        b[1] = b[1].clamp(0, h)
        b[3] = b[3].clamp(0, h)
        return b

    def _iou(a: List[float], b: List[float]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter <= 0.0:
            return 0.0
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0.0 else 0.0
    def _count_false_positives(
        pred_boxes: List[List[float]],
        gt_boxes: Optional[List[List[float]]],
        iou_threshold: float = 0.5,
    ) -> int:
        if not pred_boxes:
            return 0
        if not gt_boxes:
            return len(pred_boxes)
        used_gt = set()
        fp = 0
        for p in pred_boxes:
            best_iou = 0.0
            best_j = -1
            for j, g in enumerate(gt_boxes):
                if j in used_gt:
                    continue
                iou = _iou(p, g)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_threshold and best_j >= 0:
                used_gt.add(best_j)
            else:
                fp += 1
        return fp

    def _format_dist_matrix(mat: np.ndarray, max_rows: int = 6, max_cols: int = 6) -> List[str]:
        rows = min(mat.shape[0], max_rows)
        cols = min(mat.shape[1], max_cols)
        clipped = mat[:rows, :cols]
        lines = [f"dists {mat.shape[0]}x{mat.shape[1]} (show {rows}x{cols})"]
        txt = np.array2string(clipped, precision=2, separator=", ")
        lines.extend(txt.splitlines())
        return lines
    min_track_len = max(1, int(round(min_track_seconds * fps)))
    ema_momentum = 0.95
    tracks: Dict[int, Dict[str, Any]] = {}
    next_track_id = 0
    frame_tracks: List[Dict[int, Dict[str, Any]]] = []
    frame_dists: List[Optional[np.ndarray]] = []
    track_ids: List[int] = []
    track_embs: Optional[torch.Tensor] = None
    track_pred_boxes: Dict[int, torch.Tensor] = {}
    track_last_seen: List[int] = []
    for t, frame in enumerate(frames):
        dists_np = None
        boxes = frame.get("boxes")
        queries = frame.get("queries")
        scores = frame.get("scores")
        frame_w = frame.get("width")
        frame_h = frame.get("height")
        if boxes is None or queries is None:
            frame_tracks.append({})
            frame_dists.append(None)
            continue

        if forget_after_frames is not None and track_last_seen:
            keep = [i for i, last in enumerate(track_last_seen) if (t - last) <= forget_after_frames]
            if len(keep) != len(track_last_seen):
                track_ids = [track_ids[i] for i in keep]
                track_last_seen = [track_last_seen[i] for i in keep]
                if track_embs is not None and track_embs.numel() > 0:
                    track_embs = track_embs[keep] if keep else None
        q_t = queries if torch.is_tensor(queries) else torch.as_tensor(queries)
        b_t = boxes if torch.is_tensor(boxes) else torch.as_tensor(boxes)
        q_t, _, b_t, _ = apply_score_threshold(
            q_t,
            ids=None,
            boxes=b_t,
            scores=scores,
            threshold=score_threshold,
        )
        if q_t.numel() == 0 or b_t.numel() == 0:
            frame_tracks.append({})
            frame_dists.append(None)
            continue
        scores_t = scores if torch.is_tensor(scores) else (
            torch.as_tensor(scores, dtype=torch.float32) if scores is not None else None
        )
        if torch.is_tensor(scores_t):
            scores_t = scores_t.view(-1)

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
            frame_tracks.append({})
            frame_dists.append(None)
            continue
        if torch.is_tensor(scores_t):
            n = min(n, scores_t.shape[0])
            scores_t = scores_t[:n]

        queries_t = q_t[:n].to(device)
        frame_pos = 0.0 if len(frames) <= 1 else float(t) / float(len(frames) - 1)
        frame_pos_t = torch.full((n, 1), frame_pos, device=device)
        z = model(
            queries_t,
            boxes=b_t[:n].to(device),
            frame_idx=frame_pos_t,
            hist_z=track_embs,
            use_cross=use_cross_attn,
            use_coord_time=use_coord_time_embeds,
        )

        if track_embs is None or track_embs.numel() == 0:
            for di in range(n):
                tid = next_track_id
                next_track_id += 1
                tracks[tid] = {"last_box": b_out_t[di].detach().cpu().tolist(), "frames": {}}
                track_ids.append(tid)
                track_embs = z[di].detach().unsqueeze(0) if track_embs is None else torch.cat(
                    [track_embs, z[di].detach().unsqueeze(0)], dim=0
                )
                track_last_seen.append(t)
                if use_track_emb_boxes and model is not None:
                    pred_box = model.proj_boxes(z[di : di + 1].detach()).squeeze(0)
                    ref_box = b_t[di : di + 1].to(pred_box.device) if b_t is not None else None
                    pred_box = _standardize_pred_boxes(pred_box, ref_box).squeeze(0)
                    pred_box = _to_xyxy(pred_box, w, h)
                    track_pred_boxes[tid] = pred_box.detach()
                    box_out = pred_box.detach().cpu().tolist()
                else:
                    track_pred_boxes[tid] = b_out_t[di].detach().to(device)
                    box_out = b_out_t[di].detach().cpu().tolist()
                tracks[tid]["last_box"] = box_out
                score_val = None
                if torch.is_tensor(scores_t):
                    score_val = float(scores_t[di].item())
                tracks[tid]["frames"][t] = {"box": box_out, "score": score_val}
        else:
            assigned_track = {}
            used_tracks = set()
            used_dets = set()
            dists = _pairwise_dist(z, track_embs, normalize=use_normalized_dists)
            dists_np = dists.detach().cpu().numpy()
            pairs = []
            for di in range(dists.shape[0]):
                for tj in range(dists.shape[1]):
                    dist = float(dists[di, tj].item())
                    if dist <= match_threshold:
                        if spatial_weight > 0.0 and track_pred_boxes:
                            track_box = track_pred_boxes.get(track_ids[tj])
                            if track_box is not None:
                                iou = _iou(
                                    b_out_t[di].detach().cpu().tolist(),
                                    track_box.detach().cpu().tolist(),
                                )
                                if min_iou is not None and iou < min_iou:
                                    continue
                                dist = dist + spatial_weight * (1.0 - iou)
                        pairs.append((dist, di, tj))
            pairs.sort(key=lambda x: x[0])
            for _, di, tj in pairs:
                if di in used_dets or tj in used_tracks:
                    continue
                assigned_track[di] = track_ids[tj]
                used_dets.add(di)
                used_tracks.add(tj)

            for di in range(n):
                tid = assigned_track.get(di)
                if tid is None:
                    if not allow_new_tracks:
                        continue
                    tid = next_track_id
                    next_track_id += 1
                    tracks[tid] = {"last_box": b_out_t[di].detach().cpu().tolist(), "frames": {}}
                    track_ids.append(tid)
                    track_embs = torch.cat([track_embs, z[di].detach().unsqueeze(0)], dim=0)
                    track_last_seen.append(t)
                    if use_track_emb_boxes and model is not None:
                        pred_box = model.proj_boxes(z[di : di + 1].detach()).squeeze(0)
                        ref_box = b_t[di : di + 1].to(pred_box.device) if b_t is not None else None
                        pred_box = _standardize_pred_boxes(pred_box, ref_box).squeeze(0)
                        pred_box = _to_xyxy(pred_box, w, h)
                        track_pred_boxes[tid] = pred_box.detach()
                        box_out = pred_box.detach().cpu().tolist()
                    else:
                        track_pred_boxes[tid] = b_out_t[di].detach().to(device)
                        box_out = b_out_t[di].detach().cpu().tolist()
                else:
                    tj = track_ids.index(tid)
                    if use_ema_updates:
                        update = ema_momentum * track_embs[tj] + (1.0 - ema_momentum) * z[di].detach()
                    else:
                        update = z[di].detach()
                    track_embs[tj] = F.normalize(update, dim=-1) if use_normalized_dists else update
                    track_last_seen[tj] = t
                    if use_track_emb_boxes and model is not None:
                        pred_box = model.proj_boxes(z[di : di + 1].detach()).squeeze(0)
                        ref_box = b_t[di : di + 1].to(pred_box.device) if b_t is not None else None
                        pred_box = _standardize_pred_boxes(pred_box, ref_box).squeeze(0)
                        pred_box = _to_xyxy(pred_box, w, h)
                        track_pred_boxes[tid] = pred_box.detach()
                        box_out = pred_box.detach().cpu().tolist()
                    else:
                        box_out = b_out_t[di].detach().cpu().tolist()
                tracks[tid]["last_box"] = box_out
                score_val = None
                if torch.is_tensor(scores_t):
                    score_val = float(scores_t[di].item())
                tracks[tid]["frames"][t] = {"box": box_out, "score": score_val}

        frame_tracks.append({tid: tracks[tid]["frames"][t] for tid in tracks if t in tracks[tid]["frames"]})
        frame_dists.append(dists_np)

    keep_ids = {tid for tid, trk in tracks.items() if len(trk["frames"]) >= min_track_len}

    blank = np.zeros((h, w, 3), dtype=np.uint8)
    total_fp = 0
    total_frames = 0
    for t, frame in enumerate(frames):
        img_path = frame.get("path", "")
        img = None
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
        if img is None:
            img = blank.copy()
            cv2.putText(
                img,
                "missing frame",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        dists_np = frame_dists[t] if t < len(frame_dists) else None
        if dists_np is not None and dists_np.size > 0:
            lines = _format_dist_matrix(dists_np)
            x0, y0 = 10, 60
            for i, line in enumerate(lines):
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

        pred_boxes = [det["box"] for tid, det in frame_tracks[t].items() if tid in keep_ids]
        gt_boxes = frame.get("gt_boxes")
        gt_list = None
        if torch.is_tensor(gt_boxes):
            if gt_boxes.numel() > 0:
                gt_list = gt_boxes.detach().cpu().numpy().reshape(-1, 4).tolist()
            else:
                gt_list = []
        elif gt_boxes is not None:
            gt_list = gt_boxes
        fp_count = _count_false_positives(pred_boxes, gt_list, iou_threshold=0.5)
        total_fp += fp_count
        total_frames += 1

        for tid, det in frame_tracks[t].items():
            if tid not in keep_ids:
                continue
            x1, y1, x2, y2 = det["box"]
            cv2.rectangle(
                img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 0, 255),
                2,
            )
            label = f"id {tid}"
            if show_scores and det.get("score") is not None:
                label = f"id {tid} s {det['score']:.2f}"
            cv2.putText(
                img,
                label,
                (int(x1), max(int(y1) - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            img,
            f"frame {t} fp {fp_count}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        writer.write(img)

    writer.release()


def export_val_predictions_mot(
    model: nn.Module,
    dataset: MOTSeqDataset,
    out_dir: str,
    match_threshold: float = 1.0,
    ema_momentum: float = 0.95,
    use_cross_attn: bool = False,
    use_coord_time_embeds: bool = False,
    use_ema_updates: bool = True,
    use_track_emb_boxes: bool = False,
    allow_new_tracks: bool = True,
    max_frames: Optional[int] = None,
    score_threshold: Optional[float] = None,
    forget_after_frames: Optional[int] = None,
    spatial_weight: float = 0.0,
    min_iou: Optional[float] = None,
    use_normalized_dists: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device

    def frame_num(p: str) -> Optional[int]:
        name = os.path.basename(p).split(".")[0]
        nums = re.findall(r"\d+", name)
        return int(nums[-1]) if nums else None
    def collect_videos() -> Dict[str, List[str]]:
        videos = getattr(dataset, "videos", None)
        if isinstance(videos, dict) and videos:
            return videos
        videos = {}
        for path in dataset.data.keys():
            vid = path.split("/")[-2]
            videos.setdefault(vid, []).append(path)
        for vid in videos:
            videos[vid] = sorted(videos[vid], key=frame_num)
        return videos

    with torch.no_grad():
        videos = collect_videos()
        for vid, frame_paths in videos.items():
            if max_frames is not None:
                frame_paths = frame_paths[:max_frames]
            if len(frame_paths) == 0:
                continue
            frames = [dataset._frame_from_path(p) for p in frame_paths]
            out_path = os.path.join(out_dir, f"{vid}.txt")

            track_embs = None
            track_ids: List[int] = []
            next_track_id = 0
            track_last_seen: List[int] = []
            track_pred_boxes: Dict[int, torch.Tensor] = {}
            lines: List[str] = []
            last_frame_id = 0

            total_frames = len(frames)
            for t, frame in enumerate(frames):
                frame_used_ids = set()
                boxes = frame.get("boxes")
                queries = frame.get("queries")
                scores = frame.get("scores")
                frame_w = frame.get("width")
                frame_h = frame.get("height")
                if boxes is None or queries is None:
                    continue

                if forget_after_frames is not None and track_last_seen:
                    keep = [i for i, last in enumerate(track_last_seen) if (t - last) <= forget_after_frames]
                    if len(keep) != len(track_last_seen):
                        track_ids = [track_ids[i] for i in keep]
                        track_last_seen = [track_last_seen[i] for i in keep]
                        if track_embs is not None and track_embs.numel() > 0:
                            track_embs = track_embs[keep] if keep else None

                q_t = queries if torch.is_tensor(queries) else torch.as_tensor(queries)
                b_t = boxes if torch.is_tensor(boxes) else torch.as_tensor(boxes)
                q_t, _, b_t, _ = apply_score_threshold(
                    q_t,
                    ids=None,
                    boxes=b_t,
                    scores=scores,
                    threshold=score_threshold,
                )
                if q_t.numel() == 0 or b_t.numel() == 0:
                    continue
                scores_t = scores if torch.is_tensor(scores) else (
                    torch.as_tensor(scores, dtype=torch.float32) if scores is not None else None
                )
                if torch.is_tensor(scores_t):
                    scores_t = scores_t.view(-1)

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
                z = model(
                    queries_t,
                    boxes=b_t[:n].to(device),
                    frame_idx=frame_pos_t,
                    hist_z=track_embs,
                    use_cross=use_cross_attn,
                    use_coord_time=use_coord_time_embeds,
                )

                assigned_ids = [-1] * n
                if track_embs is None or track_embs.numel() == 0:
                    assigned_ids = list(range(next_track_id, next_track_id + n))
                    track_embs = z.detach().clone()
                    track_ids.extend(assigned_ids)
                    track_last_seen.extend([t] * n)
                    next_track_id += n
                    for i in range(n):
                        if use_track_emb_boxes:
                            pred_box = model.proj_boxes(z[i : i + 1].detach()).squeeze(0)
                            ref_box = b_t[i : i + 1].to(pred_box.device) if b_t is not None else None
                            pred_box = _standardize_pred_boxes(pred_box, ref_box).squeeze(0)
                            if (
                                frame_w is not None
                                and frame_h is not None
                                and pred_box.numel() == 4
                                and pred_box.max().item() <= 1.5
                            ):
                                pred_box = pred_box * torch.tensor(
                                    [frame_w, frame_h, frame_w, frame_h],
                                    dtype=pred_box.dtype,
                                    device=pred_box.device,
                                )
                            track_pred_boxes[assigned_ids[i]] = pred_box.detach()
                        else:
                            track_pred_boxes[assigned_ids[i]] = b_out_t[i].detach().to(device)
                else:
                    dists = _pairwise_dist(z, track_embs, normalize=use_normalized_dists)
                    cost = dists
                    valid_mask = dists < match_threshold
                    if spatial_weight > 0.0 and track_pred_boxes:
                        det_boxes_cost = b_out_t[:n].to(device)
                        track_boxes_list = [track_pred_boxes[tid] for tid in track_ids]
                        if track_boxes_list:
                            track_boxes_t = torch.stack(track_boxes_list, dim=0).to(device)
                            ious = _pairwise_iou_xyxy(det_boxes_cost, track_boxes_t)
                            cost = dists + spatial_weight * (1.0 - ious)
                            if min_iou is not None:
                                valid_mask = valid_mask & (ious >= min_iou)
                    pairs = []
                    for i in range(n):
                        for j in range(track_embs.shape[0]):
                            if not valid_mask[i, j]:
                                continue
                            d = float(cost[i, j].item())
                            pairs.append((d, i, j))
                    pairs.sort(key=lambda x: x[0])
                    used_tracks = set()
                    used_dets = set()
                    for _, i, j in pairs:
                        if i in used_dets or j in used_tracks:
                            continue
                        assigned_ids[i] = track_ids[j]
                        used_dets.add(i)
                        used_tracks.add(j)
                        if use_ema_updates:
                            update = ema_momentum * track_embs[j] + (1.0 - ema_momentum) * z[i].detach()
                        else:
                            update = z[i].detach()
                        track_embs[j] = (
                            F.normalize(update, dim=-1) if use_normalized_dists else update
                        )
                        track_last_seen[j] = t
                        if use_track_emb_boxes:
                            pred_box = model.proj_boxes(z[i : i + 1].detach()).squeeze(0)
                            ref_box = b_t[i : i + 1].to(pred_box.device) if b_t is not None else None
                            pred_box = _standardize_pred_boxes(pred_box, ref_box).squeeze(0)
                            if (
                                frame_w is not None
                                and frame_h is not None
                                and pred_box.numel() == 4
                                and pred_box.max().item() <= 1.5
                            ):
                                pred_box = pred_box * torch.tensor(
                                    [frame_w, frame_h, frame_w, frame_h],
                                    dtype=pred_box.dtype,
                                    device=pred_box.device,
                                )
                            track_pred_boxes[track_ids[j]] = pred_box.detach()
                    if allow_new_tracks:
                        for i in range(n):
                            if assigned_ids[i] == -1:
                                assigned_ids[i] = next_track_id
                                next_track_id += 1
                                track_ids.append(assigned_ids[i])
                                track_embs = torch.cat([track_embs, z[i].detach().unsqueeze(0)], dim=0)
                                track_last_seen.append(t)
                                if use_track_emb_boxes:
                                    pred_box = model.proj_boxes(z[i : i + 1].detach()).squeeze(0)
                                    ref_box = b_t[i : i + 1].to(pred_box.device) if b_t is not None else None
                                    pred_box = _standardize_pred_boxes(pred_box, ref_box).squeeze(0)
                                    if (
                                        frame_w is not None
                                        and frame_h is not None
                                        and pred_box.numel() == 4
                                        and pred_box.max().item() <= 1.5
                                    ):
                                        pred_box = pred_box * torch.tensor(
                                            [frame_w, frame_h, frame_w, frame_h],
                                            dtype=pred_box.dtype,
                                            device=pred_box.device,
                                        )
                                    track_pred_boxes[assigned_ids[i]] = pred_box.detach()
                                else:
                                    track_pred_boxes[assigned_ids[i]] = b_out_t[i].detach().to(device)

                frame_id = frame_num(frame.get("path", "")) or (t + 1)
                if frame_id <= last_frame_id:
                    frame_id = last_frame_id + 1
                last_frame_id = frame_id
                for i in range(n):
                    if assigned_ids[i] == -1:
                        continue
                    x1, y1, x2, y2 = boxes_np[i].tolist()
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    tid = int(assigned_ids[i])
                    if use_track_emb_boxes and tid in track_pred_boxes:
                        x1, y1, x2, y2 = track_pred_boxes[tid].detach().cpu().tolist()
                        w = max(0.0, x2 - x1)
                        h = max(0.0, y2 - y1)
                    if tid in frame_used_ids:
                        continue
                    frame_used_ids.add(tid)
                    if torch.is_tensor(scores_t):
                        conf = float(scores_t[i].item())
                    else:
                        conf = 1.0
                    lines.append(
                        f"{frame_id},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1"
                    )

            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            print(f"[val_preds] Wrote {out_path}")


def main() -> None:
    set_seed(0)

    train_npy_path = "/data/backup/serperzar/models/custom/preds/train_damages.npy"
    val_npy_path = "/data/backup/serperzar/models/custom/preds/val_damages.npy"
    test_npy_path = "/data/backup/serperzar/models/custom/preds/test_damages.npy"
    seq_len = 240
    stride = 240

    train_ds = MOTSeqDataset(train_npy_path, seq_len=seq_len, stride=stride)
    train_dl = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: x[0],
    )
    val_ds = MOTSeqDataset(val_npy_path, seq_len=1000, stride=stride)
    val_dl = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x[0],
    )
    test_score_threshold = 0.5
    test_ds = MOTSeqDataset(
        test_npy_path,
        seq_len=1000,
        stride=stride,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x[0],
    )

    use_cross_attn = False
    use_coord_time_embeds = False
    use_ema_updates = True
    use_track_emb_boxes = False
    use_box_pred_loss = False
    box_loss_weight = 1.0
    box_l1_weight = 1.0
    box_iou_weight = 1.0
    use_giou_loss = True
    allow_new_tracks = True
    emb_dim = 64
    sigmoid_tag = "sigmoid" if use_cross_attn else "softmax"
    model = QueryProjectorWithCross(d_in=256, d_hidden=128, d_out=emb_dim, dropout=0.5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    epochs = 100
    es_patience = 10
    es_min_delta = 0.0
    no_improve = 0
    best_val = float("inf")
    best_epoch = 0
    id_switch_weight = 1.0
    id_switch_margin = 0.2
    viz_seq_idx = 0
    viz_out_path = "val_viz/best_val_seq.mp4"
    viz_fps = 10
    export_val_preds = False
    val_preds_dir = "val_preds"
    export_test_preds = True
    test_preds_root = "/home/serperzar/custom_mot_2/TrackEval/data/trackers/mot_challenge/damages-test"
    test_preds_dir = os.path.join(test_preds_root, f"rtdetr_{sigmoid_tag}_{emb_dim}", "data")
    export_test_viz = True
    test_viz_seq_idx = 0
    test_viz_out_path = "test_viz/best_test_seq.mp4"
    test_min_track_seconds = 0.0
    match_spatial_weight = 0.4
    match_min_iou = 0.05
    use_normalized_dists = False
    for epoch in range(1, epochs + 1):
        tr_loss, tr_box_loss = train_one_epoch(
            model,
            train_dl,
            opt,
            margin=1.5,
            max_hist_items=6000,
            use_cross_attn=use_cross_attn,
            use_coord_time_embeds=use_coord_time_embeds,
            id_switch_weight=id_switch_weight,
            id_switch_margin=id_switch_margin,
            use_box_pred_loss=use_box_pred_loss,
            box_loss_weight=box_loss_weight,
            box_l1_weight=box_l1_weight,
            box_iou_weight=box_iou_weight,
            use_giou_loss=use_giou_loss,
        )
        val_loss, val_acc, val_box_loss = evaluate_one_epoch(
            model,
            val_dl,
            margin=1.5,
            max_hist_items=6000,
            use_cross_attn=use_cross_attn,
            use_coord_time_embeds=use_coord_time_embeds,
            use_box_pred_loss=use_box_pred_loss,
            box_loss_weight=box_loss_weight,
            box_l1_weight=box_l1_weight,
            box_iou_weight=box_iou_weight,
            use_giou_loss=use_giou_loss,
        )
        test_loss, test_acc, test_box_loss = evaluate_one_epoch(
            model,
            test_dl,
            margin=1.5,
            max_hist_items=6000,
            use_cross_attn=use_cross_attn,
            use_coord_time_embeds=use_coord_time_embeds,
            use_box_pred_loss=use_box_pred_loss,
            box_loss_weight=box_loss_weight,
            box_l1_weight=box_l1_weight,
            box_iou_weight=box_iou_weight,
            use_giou_loss=use_giou_loss,
            score_threshold=test_score_threshold,
        )
        if export_val_preds:
            export_val_predictions_mot(
                model,
                val_ds,
                val_preds_dir,
                match_threshold=1.0,
                ema_momentum=0.95,
                use_cross_attn=use_cross_attn,
                use_coord_time_embeds=use_coord_time_embeds,
                use_ema_updates=use_ema_updates,
                use_track_emb_boxes=use_track_emb_boxes,
                spatial_weight=match_spatial_weight,
                min_iou=match_min_iou,
                use_normalized_dists=use_normalized_dists,
            )
        scheduler.step(val_loss)
        lr = opt.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch:02d}] Train Contrastive Loss={tr_loss:.4f} | "
            f"Train Box Loss={tr_box_loss:.4f} | "
            f"Val Contrastive Loss={val_loss:.4f} | Val Acc={val_acc:.4f} | "
            f"Val Box Loss={val_box_loss:.4f} | "
            f"Test Contrastive Loss={test_loss:.4f} | Test Acc={test_acc:.4f} | "
            f"Test Box Loss={test_box_loss:.4f} | LR={lr:.6g}"
        )
        if val_loss < best_val - es_min_delta:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), "mlp_queries_triplet_best.pt")
            if export_test_preds:
                export_val_predictions_mot(
                    model,
                    test_ds,
                    test_preds_dir,
                    match_threshold=1.0,
                    ema_momentum=0.95,
                    use_cross_attn=use_cross_attn,
                    use_coord_time_embeds=use_coord_time_embeds,
                    use_ema_updates=use_ema_updates,
                    use_track_emb_boxes=use_track_emb_boxes,
                    allow_new_tracks=allow_new_tracks,
                    forget_after_frames=180,
                    spatial_weight=match_spatial_weight,
                    min_iou=match_min_iou,
                    use_normalized_dists=use_normalized_dists,
                )
            if export_test_viz:
                visualize_loader_predictions(
                    model,
                    test_ds,
                    test_viz_out_path,
                    seq_idx=test_viz_seq_idx,
                    fps=viz_fps,
                    show_scores=True,
                    min_track_seconds=test_min_track_seconds,
                    forget_after_frames=180,
                    use_ema_updates=use_ema_updates,
                    use_track_emb_boxes=use_track_emb_boxes,
                    spatial_weight=match_spatial_weight,
                    min_iou=match_min_iou,
                    use_normalized_dists=use_normalized_dists,
                    match_threshold=1.0,
                    use_cross_attn=use_cross_attn,
                    use_coord_time_embeds=use_coord_time_embeds,
                    allow_new_tracks=allow_new_tracks,
                )
                print(f"[test_viz] saved {test_viz_out_path}")
            """
            visualize_val_sequence(
                model,
                val_ds,
                viz_out_path,
                seq_idx=viz_seq_idx,
                fps=viz_fps,
            )
            """
        else:
            no_improve += 1
            if no_improve >= es_patience:
                print(f"[EarlyStop] No improvement in {es_patience} epochs. Best epoch={best_epoch}")
                break

    torch.save(model.state_dict(), "mlp_queries_triplet.pt")
    tracker_name = f"rtdetr_{sigmoid_tag}_{emb_dim}"
    trackeval_dir = "TrackEval"
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
    ]
    prev_cwd = os.getcwd()
    try:
        os.chdir(trackeval_dir)
        subprocess.run(cmd, check=True)
    finally:
        os.chdir(prev_cwd)


if __name__ == "__main__":
    main()

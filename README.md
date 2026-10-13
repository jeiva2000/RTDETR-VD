# RTDETR-VD

Official code and dataset for **"Vehicle damage detection with transformer-based multi-object tracking"** (*Engineering Applications of Artificial Intelligence*, 2026).

[![Paper](https://img.shields.io/badge/Paper-EAAI%202026-blue)](https://doi.org/10.1016/j.engappai.2026.114725)
[![Code license: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)
[![Dataset license: CC BY-NC 4.0](https://img.shields.io/badge/dataset-CC%20BY--NC%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)

https://github.com/user-attachments/assets/1ab382b7-c5df-4350-aa68-8c1f79ccd33a

![Demo](animacion_sub25mb.gif)

## Overview

We detect vehicle damage with RT-DETR and track each damage instance across a video, so that the same damage is counted once and followed through the walk-around. This repository contains the tracker training code, the inference scripts and a video-based vehicle damage dataset in MOT Challenge format.

## Dataset

17 video sequences of vehicles with annotated damage, in MOT Challenge format, with a single class (`damage`). License plates are anonymized.

| Split | Sequences | Frames | Boxes | Tracks |
|---|---|---|---|---|
| train | 10 | 16,908 | 21,506 | 109 |
| val | 1 | 1,838 | 2,715 | 8 |
| test | 6 | 5,843 | 2,804 | 42 |
| **Total** | **17** | **24,589** | **27,025** | **159** |

Layout:

```
<split>/<sequence>/
├── img1/000001.jpg ...     # frames (1-indexed)
├── gt/gt.txt               # frame,id,x,y,w,h,1,-1,1
├── seqinfo.ini             # frame rate, resolution, length
└── <sequence>.mp4          # source video
```

Train and val frames are 480x848 (portrait), test frames are 848x480 (landscape), at 30 fps.

### Request access

To request access to the dataset, please fill out the following form: [Dataset request form](https://forms.gle/oFPwwhZofzhr8S27A)

## Installation

```bash
git clone https://github.com/jeiva2000/RTDETR-VD
cd RTDETR-VD
pip install -r requirements.txt
git clone https://github.com/JonathonLuiten/TrackEval   # evaluation
```

Install a `torch`/`torchvision` build that matches your CUDA version first (see the comments in `requirements.txt`).

## Usage

| File | Purpose |
|---|---|
| `generate_preds.py` | Runs the RT-DETR ONNX detector and stores per-frame predictions and decoder queries (`.npy`). |
| `train_tracker.py` | Trains and evaluates the query-based tracker (cross-attention variants, multiple seeds, TrackEval). |
| `run_seed_experiments.sh` | The experiment grid used for the paper (embedding size, cross-attention, masks). |
| `infer_rtdetr_bytetrack.py` | ByteTrack baseline on RT-DETR detections. |
| `losses.py`, `infer_onnx_rtdetrv2.py` | Loss functions and ONNX inference helpers. |
| `extract_video_frames.py`, `generate_video_from_frames.py`, `create_gif.py` | Video and visualization utilities. |

Example:

```bash
python train_tracker.py --use-coord-time-embeds --use-cross-attn --emb-dim 128 \
    --margin 3 --test-track-max-age 5 --test-min-track-len 10 \
    --triple-seed-train --triple-seeds 11 22 33
```

### Paths

All paths are defined in [`config.py`](config.py) and can be overridden with environment variables. Defaults are relative to the repository root.

| Variable | Default | Content |
|---|---|---|
| `DATASET_ROOT` | `./dataset` | Dataset (`<split>/<sequence>/`) |
| `VIEWS_ROOT` | `./views` | Frames for the false-positive analysis |
| `PREDS_DIR` | `./preds` | Detector predictions and queries (`.npy`) |
| `CKPT_DIR` | `./checkpoints` | Tracker checkpoints |
| `TRACKEVAL_DIR` | `./TrackEval` | TrackEval clone and tracker outputs |
| `ONNX_MODEL` | `./rtdetr_damages.onnx` | Exported RT-DETR detector |

```bash
DATASET_ROOT=/path/to/dataset PREDS_DIR=/path/to/preds python train_tracker.py ...
```

## License

- **Code:** [MIT](LICENSE).
- **Dataset:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). It can be used for academic research free of charge, citing the paper below. If you seek to use the data for commercial purposes please contact us.

## Citation

If you think this project is helpful, please feel free to leave a :star: and cite our paper:

```tex
@article{PEREZZARATE2026114725,
title = {Vehicle damage detection with transformer-based multi-object tracking},
journal = {Engineering Applications of Artificial Intelligence},
volume = {176},
pages = {114725},
year = {2026},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2026.114725},
url = {https://www.sciencedirect.com/science/article/pii/S0952197626010079},
author = {Sergio Andrés Pérez-Zarate and Jose Luis Pro-Martín and Juan Antonio Álvarez-García and Miguel Ángel Martínez-del-Amor and Daniel Corzo-García and David Fernández-Cabrera},
keywords = {Multi-object tracking, Vehicle damage detection, Real-time detection transformer, Computer vision in insurance, Attention mechanism, Video-based car damage dataset},
}
```



"""Central location for the paths used by the scripts.

Every path can be overridden with an environment variable; defaults are
relative to the repository root, so nothing depends on the current directory.

    DATASET_ROOT   dataset with <split>/<sequence>/ folders          (default: ./dataset)
    VIEWS_ROOT     frames used for the false-positive analysis      (default: ./views)
    PREDS_DIR      detector predictions/queries (.npy)              (default: ./preds)
    CKPT_DIR       tracker checkpoints                              (default: ./checkpoints)
    TRACKEVAL_DIR  TrackEval clone, also used for tracker outputs   (default: ./TrackEval)
    ONNX_MODEL     exported RT-DETR detector                        (default: ./rtdetr_damages.onnx)
"""
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _path(env: str, default: str) -> str:
    return os.path.abspath(os.environ.get(env, os.path.join(REPO_ROOT, default)))


DATASET_ROOT = _path("DATASET_ROOT", "dataset")
VIEWS_ROOT = _path("VIEWS_ROOT", "views")
PREDS_DIR = _path("PREDS_DIR", "preds")
CKPT_DIR = _path("CKPT_DIR", "checkpoints")
TRACKEVAL_DIR = _path("TRACKEVAL_DIR", "TrackEval")
ONNX_MODEL = _path("ONNX_MODEL", "rtdetr_damages.onnx")

# TrackEval folder where the tracker outputs of the "damages-test" benchmark go.
TRACKEVAL_PREDS_ROOT = os.path.join(TRACKEVAL_DIR, "data", "trackers", "mot_challenge", "damages-test")

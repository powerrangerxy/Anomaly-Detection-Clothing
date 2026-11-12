"""Stage 3: train PaDiM / Reverse Distillation on one fold and predict its test patches.

Predictions are cached per model and fold, so the evaluation can be repeated without retraining.
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

# anomalib checkpoints contain more than plain weights, which torch >= 2.6 refuses to load by default
_torch_load = torch.load
torch.load = lambda *args, **kwargs: _torch_load(*args, **{**kwargs, "weights_only": False})

from anomalib.data import MVTecAD  # noqa: E402
from anomalib.engine import Engine  # noqa: E402
from anomalib.models import Padim, ReverseDistillation  # noqa: E402
from lightning.pytorch.loggers import CSVLogger  # noqa: E402

from settings import BACKBONE, MAX_EPOCHS, Paths  # noqa: E402

MODELS = {
    "padim": {
        "fn": lambda: Padim(backbone=BACKBONE, layers=["layer2", "layer3"], visualizer=False),
        "train_batch": 32, "eval_batch": 32,
    },
    "reverse_distillation": {
        "fn": lambda: ReverseDistillation(backbone=BACKBONE, layers=["layer1", "layer2", "layer3"],
                                          visualizer=False),
        "train_batch": 16, "eval_batch": 4,
    },
}

PRED_KEYS = ["classes", "scores", "raw_scores", "maps", "gt", "fold"]


def make_datamodule(fold: dict, cfg: dict, pool: Path):
    """Build the MVTec folder layout that anomalib expects, using symlinks into the patch pool."""
    def link(dst: Path, src: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(src.resolve())

    tmp = Path(tempfile.mkdtemp())
    ds = tmp / "dataset"
    for r in fold["train"]:
        link(ds / "train" / "good" / f"{r['name']}.png", pool / r["cls"] / f"{r['name']}.png")
    for cls, rows in fold["test"].items():
        for r in rows:
            link(ds / "test" / cls / f"{r['name']}.png", pool / cls / f"{r['name']}.png")
            if cls != "good":
                link(ds / "ground_truth" / cls / f"{r['name']}_mask.png",
                     pool / "masks" / cls / f"{r['name']}_mask.png")
    val_dir = tmp / "val_good"
    val_dir.mkdir()
    for r in fold["val"]:
        link(val_dir / f"{r['name']}.png", pool / r["cls"] / f"{r['name']}.png")
    dm = MVTecAD(root=str(tmp), category="dataset",
                 train_batch_size=cfg["train_batch"], eval_batch_size=cfg["eval_batch"])
    return dm, tmp, val_dir


def anomalib_metrics(results: list) -> dict:
    """anomalib's own test metrics, kept only for comparison with the earlier experiments."""
    out = {"anomalib_image_AUROC": None, "anomalib_pixel_AUROC": None,
           "anomalib_image_F1": None, "anomalib_pixel_F1": None}
    flat = results[0] if results and isinstance(results[0], dict) else {}
    for k, v in flat.items():
        kl = k.lower()
        if "image_auroc" in kl:
            out["anomalib_image_AUROC"] = round(float(v), 4)
        elif "pixel_auroc" in kl:
            out["anomalib_pixel_AUROC"] = round(float(v), 4)
        elif "pixel" in kl and "f1" in kl:
            out["anomalib_pixel_F1"] = round(float(v), 4)
        elif "f1" in kl:
            out["anomalib_image_F1"] = round(float(v), 4)
    return out


def collect_predictions(batches, with_gt: bool = True) -> dict:
    """Turn anomalib's prediction batches into arrays (maps as float16, masks as bool)."""
    paths, scores, maps, gts = [], [], [], []
    for b in batches:
        image_paths = [b.image_path] if isinstance(b.image_path, str) else b.image_path
        paths.extend(str(p) for p in image_paths)
        scores.append(b.pred_score.detach().cpu().float().numpy().reshape(-1))
        am = b.anomaly_map.detach().cpu().float().numpy()
        if am.ndim == 4:
            am = am[:, 0]
        maps.append(am.astype(np.float16))
        gm = getattr(b, "gt_mask", None) if with_gt else None
        if gm is not None:
            gm = gm.detach().cpu().numpy()
            if gm.ndim == 4:
                gm = gm[:, 0]
            gts.append(gm > 0)
        else:
            gts.append(np.zeros(am.shape, dtype=bool))
    return {"names": [Path(p).stem for p in paths],
            "classes": np.array([Path(p).parent.name for p in paths]),
            "scores": np.concatenate(scores), "maps": np.concatenate(maps), "gt": np.concatenate(gts)}


def robust_scale(x: np.ndarray):
    med = float(np.median(x))
    q1, q3 = np.percentile(x, [25, 75])
    iqr = float(q3 - q1) or float(np.std(x)) or 1.0
    return med, iqr


def train_and_predict(model_name: str, cfg: dict, fold: dict, paths: Paths, keep_ckpt: bool):
    dm, tmp, val_dir = make_datamodule(fold, cfg, paths.pool)
    ckpt_dir = paths.split_dir / "checkpoints" / f"fold{fold['fold']}" / model_name
    try:
        model = cfg["fn"]()
        logger = CSVLogger(save_dir=str(paths.split_dir / "logs"), name=f"fold{fold['fold']}_{model_name}")
        engine = Engine(max_epochs=MAX_EPOCHS, logger=logger, default_root_dir=str(ckpt_dir))
        engine.fit(model=model, datamodule=dm)
        reported = anomalib_metrics(engine.test(model=model, datamodule=dm))

        # anomalib normalises scores to [0, 1] and clips them; we want the raw scores
        model.post_processor.enable_normalization = False
        model.post_processor.enable_thresholding = False
        pred = collect_predictions(engine.predict(model=model, datamodule=dm, return_predictions=True))
        val = collect_predictions(engine.predict(model=model, data_path=str(val_dir),
                                                 return_predictions=True), with_gt=False)

        # Standardise with good validation patches of the training garments (no test labels used),
        # so that the scores of all folds can be pooled with one threshold.
        s_med, s_iqr = robust_scale(val["scores"])
        p_med, p_iqr = robust_scale(val["maps"].astype(np.float32).ravel())
        pred["raw_scores"] = pred["scores"].copy()
        pred["scores"] = (pred["scores"] - s_med) / s_iqr
        pred["maps"] = ((pred["maps"].astype(np.float32) - p_med) / p_iqr).astype(np.float16)
        pred["fold"] = np.full(len(pred["scores"]), fold["fold"])
        pred["norm"] = {"score_median": s_med, "score_iqr": s_iqr, "pixel_median": p_med, "pixel_iqr": p_iqr}
        return reported, pred
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if not keep_ckpt:
            shutil.rmtree(ckpt_dir, ignore_errors=True)


def cache_path(paths: Paths, model_name: str, fold: int) -> Path:
    return paths.split_dir / "cache" / f"{model_name}_fold{fold}.npz"


def save_cache(paths: Paths, model_name: str, fold: int, reported: dict, pred: dict):
    p = cache_path(paths, model_name, fold)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, names=np.array(pred["names"]), **{k: pred[k] for k in PRED_KEYS},
             meta=json.dumps({"reported": reported, "norm": pred["norm"]}))


def load_cache(paths: Paths, model_name: str, fold: int):
    p = cache_path(paths, model_name, fold)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    pred = {k: z[k] for k in PRED_KEYS}
    pred["names"] = z["names"].tolist()
    pred["norm"] = meta["norm"]
    return meta["reported"], pred


def concat_preds(preds: list[dict]) -> dict:
    out = {k: np.concatenate([p[k] for p in preds]) for k in PRED_KEYS}
    out["names"] = [n for p in preds for n in p["names"]]
    return out

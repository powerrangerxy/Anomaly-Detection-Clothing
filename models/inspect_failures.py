"""Failure analysis of the phase 1 benchmark models (loads the checkpoints, no training).

The threshold is set so that 5% of the good test patches are flagged. Per defect class we
compute recall and the mean IoU of the thresholded heat map, and save image grids of
missed defects and false alarms. Run run_focused_benchmark.py first.
"""

import csv
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import torch

# anomalib checkpoints contain their own classes, torch >= 2.6 only loads them with weights_only=False
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore, ReverseDistillation, Supersimplenet

# config
DATASET_ROOT   = Path("datasets/focused_jeans")
RESULTS_ROOT   = Path("results")
INSPECT_DIR    = Path("results/inspection")
EROSION_LABELS = ["erosion_00px", "erosion_20px", "erosion_45px"]

TEST_CLASSES = [
    "abrasion", "hole_tear", "missing_functional_elements",
    "other_defects", "seam_defect", "stain_dirt",
]

TARGET_FPR  = 0.05   # calibrate threshold so ~5% of good test patches are flagged
PATCH_SIZE  = 336
GRID_COLS   = 5
MAX_GRID    = 20     # max patches per failure grid

MODELS = {
    "padim": (
        lambda: Padim(backbone="resnet18", layers=["layer2", "layer3"],
                      visualizer=False),
        "Padim",
    ),
    "patchcore": (
        lambda: Patchcore(backbone="resnet18", layers=["layer2", "layer3"],
                          coreset_sampling_ratio=0.15, visualizer=False),
        "Patchcore",
    ),
    "reverse_distillation": (
        lambda: ReverseDistillation(backbone="resnet18",
                                    layers=["layer1", "layer2", "layer3"],
                                    visualizer=False),
        "ReverseDistillation",
    ),
    "supersimplenet": (
        lambda: Supersimplenet(backbone="resnet18", layers=["layer2", "layer3"],
                               visualizer=False),
        "Supersimplenet",
    ),
}


def find_checkpoint(model_dir_name: str, erosion_label: str) -> Path | None:
    base  = RESULTS_ROOT / model_dir_name / "MVTecAD" / erosion_label
    ckpts = sorted(base.glob("v*/weights/lightning/model.ckpt")) if base.exists() else []
    return ckpts[-1] if ckpts else None


def build_test_symlinks(src: Path) -> Path:
    """
    Build a minimal MVTecAD-compatible temp directory with only test data.
    We add a single train/good symlink because anomalib requires the folder to exist.
    """
    tmp = Path(tempfile.mkdtemp()) / src.name
    tmp.mkdir(parents=True)

    # train/good, at least one image required for datamodule setup
    (tmp / "train" / "good").mkdir(parents=True)
    train_imgs = sorted((src / "train" / "good").glob("*.png"))
    if train_imgs:
        (tmp / "train" / "good" / train_imgs[0].name).symlink_to(train_imgs[0].resolve())

    # val/good, can be empty but directory must exist
    (tmp / "val" / "good").mkdir(parents=True)

    # test/good
    (tmp / "test" / "good").mkdir(parents=True)
    for p in sorted((src / "test" / "good").glob("*.png")):
        (tmp / "test" / "good" / p.name).symlink_to(p.resolve())

    # test/{defect_class} + ground_truth/{defect_class}
    for cls in TEST_CLASSES:
        cls_test = src / "test" / cls
        cls_gt   = src / "ground_truth" / cls
        if not cls_test.exists():
            continue
        (tmp / "test" / cls).mkdir(parents=True)
        (tmp / "ground_truth" / cls).mkdir(parents=True)
        for p in sorted(cls_test.glob("*.png")):
            (tmp / "test" / cls / p.name).symlink_to(p.resolve())
            mask = cls_gt / p.name
            if mask.exists():
                (tmp / "ground_truth" / cls / p.name).symlink_to(mask.resolve())

    return tmp


def collect_predictions(model_fn, ckpt_path: Path, tmp_dataset: Path,
                         erosion_label: str) -> list[dict]:
    """
    Run predict with a saved checkpoint and return one record per test patch:
        image_name, defect_class, true_label (0=good / 1=defect),
        score (float), anomaly_map (np.ndarray H×W float32)
    """
    datamodule = MVTecAD(
        root=str(tmp_dataset.parent),
        category=tmp_dataset.name,
        train_batch_size=1,
        eval_batch_size=1,
    )
    model  = model_fn()
    engine = Engine()

    batches = engine.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=str(ckpt_path),
    )
    if not batches:
        return []

    records = []
    for batch in batches:
        # atleast_1d handles the batch_size=1 case where tensors are squeezed to 0-dim scalars
        scores    = np.atleast_1d(batch.pred_score.cpu().numpy())
        gt_labels = np.atleast_1d(batch.gt_label.cpu().numpy())
        anom_maps = np.atleast_1d(batch.anomaly_map.cpu().numpy())
        img_paths = batch.image_path
        if isinstance(img_paths, str):
            img_paths = [img_paths]

        if anom_maps.ndim == 4:
            anom_maps = anom_maps[:, 0]                # drop channel dim -> (B, H, W)
        elif anom_maps.ndim == 2:
            anom_maps = anom_maps[np.newaxis]          # (H, W) -> (1, H, W)

        for i in range(len(scores)):
            fname        = Path(img_paths[i]).name
            defect_class = Path(img_paths[i]).parent.name   # "good" | "abrasion" | ...
            records.append({
                "image_name":   fname,
                "defect_class": defect_class,
                "true_label":   int(gt_labels[i]),
                "score":        float(scores[i]),
                "anomaly_map":  anom_maps[i].astype(np.float32),
                # store the canonical (non-temp) path for GT mask lookup
                "gt_mask_path": (
                    DATASET_ROOT / erosion_label / "ground_truth" / defect_class / fname
                    if defect_class != "good" else None
                ),
                "patch_path": (
                    DATASET_ROOT / erosion_label / "test" / defect_class / fname
                ),
            })

    return records


def threshold_at_fpr(records: list[dict], target_fpr: float) -> float:
    """
    Find the score threshold so that at most target_fpr of good patches are flagged.
    Calibrated on the test good set (conservative choice, no leakage from defect classes).
    """
    good_scores = sorted(r["score"] for r in records if r["true_label"] == 0)
    if not good_scores:
        return 0.5
    # we want the (1 - target_fpr) quantile of good scores as the threshold
    idx = int(np.ceil((1.0 - target_fpr) * len(good_scores))) - 1
    idx = max(0, min(idx, len(good_scores) - 1))
    return good_scores[idx]


def mask_iou(pred_map: np.ndarray, gt_mask_path: Path) -> float | None:
    """
    Binarize the anomaly map at its median value and compute IoU vs GT mask.
    Returns None if GT mask has no positive pixels.
    """
    if not gt_mask_path.exists():
        return None
    gt  = np.array(Image.open(gt_mask_path).convert("L"))
    # resize gt to match heatmap if needed
    if gt.shape != pred_map.shape:
        gt = np.array(
            Image.fromarray(gt).resize(
                (pred_map.shape[1], pred_map.shape[0]), Image.NEAREST
            )
        )
    gt_bin = gt > 127
    if not gt_bin.any():
        return None

    # threshold the heatmap at the midpoint of its range
    lo, hi   = pred_map.min(), pred_map.max()
    pred_bin = pred_map > (lo + (hi - lo) * 0.5)

    inter = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    return float(inter / union) if union > 0 else 0.0


def compute_metrics(records: list[dict], threshold: float) -> dict:
    good    = [r for r in records if r["true_label"] == 0]
    defects = [r for r in records if r["true_label"] == 1]

    fp = sum(1 for r in good if r["score"] > threshold)
    tn = len(good) - fp
    fpr = fp / len(good) if good else 0.0

    per_class: dict[str, dict] = defaultdict(lambda: {
        "TP": 0, "FN": 0, "ious": []
    })

    for r in defects:
        cls = r["defect_class"]
        detected = r["score"] > threshold
        if detected:
            per_class[cls]["TP"] += 1
            iou = mask_iou(r["anomaly_map"], r["gt_mask_path"])
            if iou is not None:
                per_class[cls]["ious"].append(iou)
        else:
            per_class[cls]["FN"] += 1

    return {
        "good_total": len(good),
        "fp": fp, "tn": tn, "fpr": fpr,
        "per_class": dict(per_class),
    }


def _load_font(size: int = 24):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def save_failure_grid(records: list[dict], threshold: float,
                      kind: str, defect_class: str, out_path: Path):
    """
    kind="fn" -> patches the model missed (score <= threshold, true label = defect)
    kind="fp" -> good patches wrongly flagged (score > threshold, true label = good)

    Each cell shows:   original | heatmap overlay | [gt mask if fn]
    """
    if kind == "fn":
        pool = [r for r in records
                if r["true_label"] == 1
                and r["defect_class"] == defect_class
                and r["score"] <= threshold]
        # worst first = lowest score among misses
        pool = sorted(pool, key=lambda r: r["score"])[:MAX_GRID]
        panels = 3
    else:
        pool = [r for r in records
                if r["true_label"] == 0 and r["score"] > threshold]
        # worst first = highest score among false alarms
        pool = sorted(pool, key=lambda r: r["score"], reverse=True)[:MAX_GRID]
        panels = 2

    if not pool:
        return

    P    = PATCH_SIZE
    cols = min(GRID_COLS, len(pool))
    rows = (len(pool) + cols - 1) // cols
    cw   = P * panels
    grid = Image.new("RGB", (cw * cols, P * rows), (20, 20, 20))
    draw = ImageDraw.Draw(grid)
    font = _load_font(20)

    for idx, r in enumerate(pool):
        col = idx % cols
        row = idx // cols
        ox  = col * cw
        oy  = row * P

        # panel 1: original patch
        orig = Image.open(r["patch_path"]).convert("RGB").resize((P, P))
        grid.paste(orig, (ox, oy))

        # panel 2: heatmap blended onto original
        hmap      = r["anomaly_map"]
        hmap_norm = ((hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8) * 255).astype(np.uint8)
        heat_rgb  = np.stack([hmap_norm,
                               np.zeros_like(hmap_norm),
                               np.zeros_like(hmap_norm)], axis=-1)
        heat_img  = Image.fromarray(heat_rgb).resize((P, P))
        blend     = Image.blend(orig, heat_img, alpha=0.55)
        grid.paste(blend, (ox + P, oy))

        # panel 3 (fn only): ground-truth mask
        if kind == "fn":
            gtp = r["gt_mask_path"]
            if gtp and gtp.exists():
                gt_img = Image.open(gtp).convert("RGB").resize((P, P))
            else:
                gt_img = Image.new("RGB", (P, P), (40, 40, 40))
            grid.paste(gt_img, (ox + P * 2, oy))

        # score label
        label = f"{r['score']:.3f}"
        draw.text((ox + 4, oy + 4), label, font=font, fill=(0, 0, 0))
        draw.text((ox + 3, oy + 3), label, font=font, fill=(255, 255, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(f"      grid saved: {out_path.name}  ({len(pool)} patches)")


def run_one(model_name: str, model_fn, model_dir: str, erosion_label: str) -> list[dict]:
    ckpt = find_checkpoint(model_dir, erosion_label)
    if ckpt is None:
        print(f"  [SKIP] checkpoint not found for {model_dir}/MVTecAD/{erosion_label}")
        return []

    print(f"  checkpoint : {ckpt}")
    src     = DATASET_ROOT / erosion_label
    tmp     = build_test_symlinks(src)
    out_dir = INSPECT_DIR / f"{model_name}_{erosion_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        records = collect_predictions(model_fn, ckpt, tmp, erosion_label)
        if not records:
            print("  [WARN] predict returned no results")
            return []

        threshold = threshold_at_fpr(records, TARGET_FPR)
        print(f"  threshold @ FPR<={TARGET_FPR:.0%} : {threshold:.4f}")

        # per-sample scores CSV
        with open(out_dir / "scores.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "defect_class", "true_label", "score", "predicted"])
            for r in records:
                w.writerow([
                    r["image_name"],
                    r["defect_class"],
                    r["true_label"],
                    f"{r['score']:.6f}",
                    int(r["score"] > threshold),
                ])

        # metrics
        m   = compute_metrics(records, threshold)
        fpr = m["fpr"]
        print(f"  good: {m['good_total']}  FP={m['fp']}  actual FPR={fpr:.3f}")

        summary = []
        for cls, c in sorted(m["per_class"].items()):
            tp, fn_ = c["TP"], c["FN"]
            total   = tp + fn_
            recall  = tp / total if total > 0 else 0.0
            mean_iou = float(np.mean(c["ious"])) if c["ious"] else float("nan")

            print(f"  {cls:<35} n={total:>3}  recall={recall:.2f}  "
                  f"IoU={mean_iou:.3f}  (TP={tp} FN={fn_})")

            # failure grids
            save_failure_grid(records, threshold, "fn", cls,
                              out_dir / f"fn_{cls}.png")

            summary.append({
                "model":     model_name,
                "erosion":   erosion_label,
                "class":     cls,
                "total":     total,
                "TP":        tp,
                "FN":        fn_,
                "recall":    round(recall, 4),
                "mean_IoU":  round(mean_iou, 4) if not np.isnan(mean_iou) else "",
            })

        # FP grid for good patches
        save_failure_grid(records, threshold, "fp", "good",
                          out_dir / "fp_good.png")

        return summary

    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


def main():
    INSPECT_DIR.mkdir(parents=True, exist_ok=True)
    all_summary = []

    for erosion_label in EROSION_LABELS:
        for model_name, (model_fn, model_dir) in MODELS.items():
            print(f"\n{'='*60}")
            print(f"  {model_name}  |  {erosion_label}")
            print(f"{'='*60}")
            rows = run_one(model_name, model_fn, model_dir, erosion_label)
            all_summary.extend(rows)

    if all_summary:
        csv_path = INSPECT_DIR / "summary.csv"
        fields   = ["model", "erosion", "class", "total", "TP", "FN", "recall", "mean_IoU"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_summary)
        print(f"\nsummary CSV  ->  {csv_path.resolve()}")

    print("\ndone.")


if __name__ == "__main__":
    main()

"""Failure analysis of the phase 2 PaDiM and RD models (20 px erosion).

Per defect class: recall, missed defects and mean IoU, plus image grids of
missed defects (patch | heat map | ground truth) and false alarms (patch | heat map).
Set SUBCATEGORY to jeans or t_shirt.
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
from anomalib.models import Padim, ReverseDistillation

# config
SUBCATEGORY = "jeans"       # "jeans" or "t_shirt"
EROSION_PX  = 20
TARGET_FPR  = 0.05          # calibrate threshold so ~5% of good patches are flagged
PATCH_SIZE  = 336
GRID_COLS   = 5
MAX_GRID    = 20

TEST_CLASSES = [
    "abrasion", "hole_tear", "missing_functional_elements",
    "other_defects", "seam_defect", "stain_dirt",
]

# checkpoint mapping
# Identified by file date:
#   jeans   -> Padim v1 (Jul 15), RD v2 (Jul 15)
#   t_shirt -> Padim v2 (Jul 18), RD v3 (Jul 18)
CHECKPOINTS = {
    "jeans": {
        "padim":               "results/Padim/MVTecAD/erosion_20px/v1/weights/lightning/model.ckpt",
        "reverse_distillation":"results/ReverseDistillation/MVTecAD/erosion_20px/v2/weights/lightning/model.ckpt",
    },
    "t_shirt": {
        "padim":               "results/Padim/MVTecAD/erosion_20px/v2/weights/lightning/model.ckpt",
        "reverse_distillation":"results/ReverseDistillation/MVTecAD/erosion_20px/v3/weights/lightning/model.ckpt",
    },
}

MODELS = {
    "padim": lambda: Padim(backbone="resnet18", layers=["layer2", "layer3"],
                           visualizer=False),
    "reverse_distillation": lambda: ReverseDistillation(backbone="resnet18",
                                                        layers=["layer1", "layer2", "layer3"],
                                                        visualizer=False),
}

# paths
DATASET_ROOT = Path("second_phase_rd_padim") / SUBCATEGORY / f"erosion_{EROSION_PX:02d}px"
INSPECT_DIR  = Path("second_phase_rd_padim") / SUBCATEGORY / "results" / "inspection"


def build_test_symlinks() -> Path:
    tmp = Path(tempfile.mkdtemp()) / DATASET_ROOT.name
    tmp.mkdir(parents=True)

    (tmp / "train" / "good").mkdir(parents=True)
    train_imgs = sorted((DATASET_ROOT / "train" / "good").glob("*.png"))
    if train_imgs:
        (tmp / "train" / "good" / train_imgs[0].name).symlink_to(train_imgs[0].resolve())

    (tmp / "val" / "good").mkdir(parents=True)

    (tmp / "test" / "good").mkdir(parents=True)
    for p in sorted((DATASET_ROOT / "test" / "good").glob("*.png")):
        (tmp / "test" / "good" / p.name).symlink_to(p.resolve())

    for cls in TEST_CLASSES:
        cls_test = DATASET_ROOT / "test" / cls
        cls_gt   = DATASET_ROOT / "ground_truth" / cls
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


def collect_predictions(model_fn, ckpt_path: Path, tmp_dataset: Path) -> list[dict]:
    datamodule = MVTecAD(
        root=str(tmp_dataset.parent),
        category=tmp_dataset.name,
        train_batch_size=1,
        eval_batch_size=1,
    )
    model   = model_fn()
    engine  = Engine()
    batches = engine.predict(model=model, datamodule=datamodule,
                             ckpt_path=str(ckpt_path))
    if not batches:
        return []

    records = []
    for batch in batches:
        scores    = np.atleast_1d(batch.pred_score.cpu().numpy())
        gt_labels = np.atleast_1d(batch.gt_label.cpu().numpy())
        anom_maps = np.atleast_1d(batch.anomaly_map.cpu().numpy())
        img_paths = batch.image_path
        if isinstance(img_paths, str):
            img_paths = [img_paths]
        if anom_maps.ndim == 4:
            anom_maps = anom_maps[:, 0]
        elif anom_maps.ndim == 2:
            anom_maps = anom_maps[np.newaxis]

        for i in range(len(scores)):
            fname        = Path(img_paths[i]).name
            defect_class = Path(img_paths[i]).parent.name
            records.append({
                "image_name":   fname,
                "defect_class": defect_class,
                "true_label":   int(gt_labels[i]),
                "score":        float(scores[i]),
                "anomaly_map":  anom_maps[i].astype(np.float32),
                "gt_mask_path": (
                    DATASET_ROOT / "ground_truth" / defect_class / fname
                    if defect_class != "good" else None
                ),
                "patch_path": DATASET_ROOT / "test" / defect_class / fname,
            })
    return records


def threshold_at_fpr(records: list[dict], target_fpr: float) -> float:
    good_scores = sorted(r["score"] for r in records if r["true_label"] == 0)
    if not good_scores:
        return 0.5
    idx = int(np.ceil((1.0 - target_fpr) * len(good_scores))) - 1
    idx = max(0, min(idx, len(good_scores) - 1))
    return good_scores[idx]


def mask_iou(pred_map: np.ndarray, gt_mask_path: Path) -> float | None:
    if not gt_mask_path or not gt_mask_path.exists():
        return None
    gt = np.array(Image.open(gt_mask_path).convert("L"))
    if gt.shape != pred_map.shape:
        gt = np.array(
            Image.fromarray(gt).resize(
                (pred_map.shape[1], pred_map.shape[0]), Image.NEAREST))
    gt_bin = gt > 127
    if not gt_bin.any():
        return None
    lo, hi   = pred_map.min(), pred_map.max()
    pred_bin = pred_map > (lo + (hi - lo) * 0.5)
    inter    = (pred_bin & gt_bin).sum()
    union    = (pred_bin | gt_bin).sum()
    return float(inter / union) if union > 0 else 0.0


def _load_font(size: int = 20):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def save_failure_grid(records: list[dict], threshold: float,
                      kind: str, defect_class: str, out_path: Path):
    """
    kind="fn" -> patches the model missed  (3 panels: original | heatmap | GT mask)
    kind="fp" -> good patches wrongly flagged (2 panels: original | heatmap)
    """
    if kind == "fn":
        pool   = [r for r in records
                  if r["true_label"] == 1
                  and r["defect_class"] == defect_class
                  and r["score"] <= threshold]
        pool   = sorted(pool, key=lambda r: r["score"])[:MAX_GRID]
        panels = 3
    else:
        pool   = [r for r in records if r["true_label"] == 0 and r["score"] > threshold]
        pool   = sorted(pool, key=lambda r: r["score"], reverse=True)[:MAX_GRID]
        panels = 2

    if not pool:
        return

    P    = PATCH_SIZE
    cols = min(GRID_COLS, len(pool))
    rows = (len(pool) + cols - 1) // cols
    grid = Image.new("RGB", (P * panels * cols, P * rows), (20, 20, 20))
    draw = ImageDraw.Draw(grid)
    font = _load_font()

    for idx, r in enumerate(pool):
        col = idx % cols
        row = idx // cols
        ox  = col * P * panels
        oy  = row * P

        orig = Image.open(r["patch_path"]).convert("RGB").resize((P, P))
        grid.paste(orig, (ox, oy))

        hmap      = r["anomaly_map"]
        hmap_norm = ((hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8) * 255).astype(np.uint8)
        heat_rgb  = np.stack([hmap_norm, np.zeros_like(hmap_norm),
                               np.zeros_like(hmap_norm)], axis=-1)
        blend     = Image.blend(orig, Image.fromarray(heat_rgb).resize((P, P)), alpha=0.55)
        grid.paste(blend, (ox + P, oy))

        if kind == "fn":
            gtp = r["gt_mask_path"]
            gt_img = (Image.open(gtp).convert("RGB").resize((P, P))
                      if gtp and gtp.exists()
                      else Image.new("RGB", (P, P), (40, 40, 40)))
            grid.paste(gt_img, (ox + P * 2, oy))

        label = f"{r['score']:.3f}"
        draw.text((ox + 4, oy + 4), label, font=font, fill=(0, 0, 0))
        draw.text((ox + 3, oy + 3), label, font=font, fill=(255, 255, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(f"      saved: {out_path.name}  ({len(pool)} patches)")


def run_one(model_name: str, model_fn, ckpt_path: Path) -> list[dict]:
    out_dir = INSPECT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp = build_test_symlinks()
    try:
        records = collect_predictions(model_fn, ckpt_path, tmp)
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)

    if not records:
        print("  [WARN] no predictions returned")
        return []

    threshold = threshold_at_fpr(records, TARGET_FPR)
    print(f"  threshold @ FPR<={TARGET_FPR:.0%} : {threshold:.4f}")

    # per-sample scores CSV
    with open(out_dir / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "defect_class", "true_label", "score", "predicted"])
        for r in records:
            w.writerow([r["image_name"], r["defect_class"], r["true_label"],
                        f"{r['score']:.6f}", int(r["score"] > threshold)])

    # metrics
    good    = [r for r in records if r["true_label"] == 0]
    defects = [r for r in records if r["true_label"] == 1]
    fp      = sum(1 for r in good if r["score"] > threshold)
    fpr     = fp / len(good) if good else 0.0
    print(f"  good: {len(good)}  FP={fp}  actual FPR={fpr:.3f}")

    per_class: dict = defaultdict(lambda: {"TP": 0, "FN": 0, "ious": []})
    for r in defects:
        cls      = r["defect_class"]
        detected = r["score"] > threshold
        if detected:
            per_class[cls]["TP"] += 1
            iou = mask_iou(r["anomaly_map"], r["gt_mask_path"])
            if iou is not None:
                per_class[cls]["ious"].append(iou)
        else:
            per_class[cls]["FN"] += 1

    summary = []
    for cls in TEST_CLASSES:
        if cls not in per_class:
            continue
        c        = per_class[cls]
        tp, fn_  = c["TP"], c["FN"]
        total    = tp + fn_
        recall   = tp / total if total > 0 else 0.0
        mean_iou = float(np.mean(c["ious"])) if c["ious"] else float("nan")

        print(f"  {cls:<35} n={total:>3}  recall={recall:.2f}  "
              f"IoU={mean_iou:.3f}  (TP={tp} FN={fn_})")

        save_failure_grid(records, threshold, "fn", cls,
                          out_dir / f"fn_{cls}.png")

        summary.append({
            "subcategory": SUBCATEGORY,
            "model":       model_name,
            "class":       cls,
            "total":       total,
            "TP":          tp,
            "FN":          fn_,
            "recall":      round(recall, 4),
            "mean_IoU":    round(mean_iou, 4) if not np.isnan(mean_iou) else "",
        })

    save_failure_grid(records, threshold, "fp", "good", out_dir / "fp_good.png")
    return summary


def main():
    if not DATASET_ROOT.exists():
        print(f"[ERROR] dataset not found: {DATASET_ROOT}")
        print("  Run the appropriate build script first.")
        return

    ckpts = CHECKPOINTS.get(SUBCATEGORY)
    if not ckpts:
        print(f"[ERROR] no checkpoints configured for subcategory '{SUBCATEGORY}'")
        return

    INSPECT_DIR.mkdir(parents=True, exist_ok=True)
    all_summary = []

    for model_name, model_fn in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  {model_name}  |  {SUBCATEGORY}  erosion_{EROSION_PX:02d}px")
        print(f"{'='*60}")

        ckpt = Path(ckpts[model_name])
        if not ckpt.exists():
            print(f"  [SKIP] checkpoint not found: {ckpt}")
            continue
        print(f"  checkpoint: {ckpt}")

        rows = run_one(model_name, model_fn, ckpt)
        all_summary.extend(rows)

    if all_summary:
        fields   = ["subcategory", "model", "class", "total", "TP", "FN", "recall", "mean_IoU"]
        csv_path = INSPECT_DIR / "summary.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(all_summary)
        print(f"\nsummary -> {csv_path.resolve()}")

    print("\ndone.")


if __name__ == "__main__":
    main()

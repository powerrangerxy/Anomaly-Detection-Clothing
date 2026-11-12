"""
Build erosion_20px dataset for t-shirts, then train PaDiM and ReverseDistillation from scratch.
Assumes dataset already built by run_tshirt_zero_shot.py (skips build if folder exists).
Run from project root: python models/run_rd_padim_tshirt_20px.py

Outputs (under second_phase_rd_padim/t_shirt/):
  erosion_20px/                          <- dataset (built if missing)
  results/
    results_t_shirt.csv
    masks/padim/...                        <- binary pred masks
    masks/reverse_distillation/...
    visualizations_trained/padim/...       <- 4-panel PNGs
    visualizations_trained/reverse_distillation/...
    training_logs/reverse_distillation/  <- CSVLogger output for loss curves
    plots/training_curves.png
    plots/metric_comparison.png          <- trained vs zero-shot (reads zero_shot CSV if present)
"""

import csv
import json
import random
import shutil
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

# PyTorch 2.6 compat
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim, ReverseDistillation
from lightning.pytorch.loggers import CSVLogger

# settings
FILTER_FIELD       = "garment_sub_category"
FILTER_VALUE       = "t_shirt"
EROSION_PX         = 20
PATCH_SIZE         = 336
GARMENT_COVERAGE   = 0.30
DEFECT_THRESHOLD   = 0.02
GOOD_STRIDE        = 336
EXCLUDE_CATEGORIES = {"brand_marker"}
TRAIN_SPLIT        = 0.80
RANDOM_SEED        = 42
TEST_CAP           = 300
VAL_SAMPLE         = 200
BACKBONE           = "resnet18"
MAX_EPOCHS         = 50

# paths (auto-derived from FILTER_VALUE / EROSION_PX)
DATA_ROOT   = Path("202603_ReUseTex_Setup1/01_data")
SEG_ROOT    = Path("202603_ReUseTex_Setup1/02_segmentation")
CACHE_ROOT  = Path("datasets/custom_ivc_dataset/by_category")
OUTPUT_ROOT = Path("second_phase_rd_padim") / FILTER_VALUE / f"erosion_{EROSION_PX:02d}px"
RESULTS_DIR = Path("second_phase_rd_padim") / FILTER_VALUE / "results"

TEST_CLASSES = [
    "abrasion", "hole_tear", "missing_functional_elements",
    "other_defects", "seam_defect", "stain_dirt",
]


# dataset building

def load_garment_mask(png_path: Path):
    rel     = png_path.relative_to(DATA_ROOT)
    npy     = CACHE_ROOT / FILTER_VALUE / rel.parent / png_path.stem / "garment_mask.npy"
    if npy.exists():
        return np.load(str(npy))
    seg_png = SEG_ROOT / rel.parent / (png_path.stem + "_mask.png")
    if seg_png.exists():
        return np.array(Image.open(seg_png).convert("L")) > 0
    return None


def load_coco(coco_path: Path):
    with open(coco_path) as f:
        data = json.load(f)
    return {c["id"]: c["name"] for c in data["categories"]}, data.get("annotations", [])


def get_mask_bbox(mask: np.ndarray):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return int(y_min), int(y_max), int(x_min), int(x_max)


def create_defect_mask(shape_hw, annotations, categories, x_off, y_off):
    h, w   = shape_hw
    canvas = Image.new("L", (w, h), 0)
    draw   = ImageDraw.Draw(canvas)
    for ann in annotations:
        cat = categories.get(ann["category_id"], "")
        if cat in EXCLUDE_CATEGORIES:
            continue
        for seg in ann.get("segmentation", []):
            pts = [(seg[i] - x_off, seg[i + 1] - y_off) for i in range(0, len(seg), 2)]
            draw.polygon(pts, fill=255)
    return np.array(canvas) > 0


def erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    pil = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.array(pil.filter(ImageFilter.MinFilter(px * 2 + 1))) > 0


def annotation_centroids(annotations, categories, x_off, y_off, img_h, img_w):
    centroids = []
    for ann in annotations:
        cat = categories.get(ann["category_id"], "")
        if cat in EXCLUDE_CATEGORIES:
            continue
        segs = ann.get("segmentation", [])
        if segs:
            seg = segs[0]
            xs  = [seg[i]     - x_off for i in range(0, len(seg), 2)]
            ys  = [seg[i]     - y_off for i in range(1, len(seg), 2)]
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
        else:
            x, y, w, h = ann["bbox"]
            cx = int(x + w / 2) - x_off
            cy = int(y + h / 2) - y_off
        cx = max(PATCH_SIZE // 2, min(cx, img_w - PATCH_SIZE // 2))
        cy = max(PATCH_SIZE // 2, min(cy, img_h - PATCH_SIZE // 2))
        centroids.append((cx, cy, cat))
    return centroids


def extract_defective_patches(image_np, garment_mask, eroded_mask, raw_mask,
                               annotations, categories, x_off, y_off):
    h, w      = image_np.shape[:2]
    centroids = annotation_centroids(annotations, categories, x_off, y_off, h, w)
    patches, seen = [], set()
    for cx, cy, cat in centroids:
        r0, c0 = cy - PATCH_SIZE // 2, cx - PATCH_SIZE // 2
        r1, c1 = r0 + PATCH_SIZE, c0 + PATCH_SIZE
        if (r0, c0) in seen:
            continue
        seen.add((r0, c0))
        if np.mean(garment_mask[r0:r1, c0:c1]) < GARMENT_COVERAGE:
            continue
        if np.mean(eroded_mask[r0:r1, c0:c1]) < DEFECT_THRESHOLD:
            continue
        patches.append((image_np[r0:r1, c0:c1].copy(),
                        (raw_mask[r0:r1, c0:c1] * 255).astype(np.uint8),
                        cat, r0, c0))
    return patches


def extract_good_patches(image_np, garment_mask, raw_mask):
    h, w    = image_np.shape[:2]
    patches = []
    for r0 in range(0, h - PATCH_SIZE + 1, GOOD_STRIDE):
        for c0 in range(0, w - PATCH_SIZE + 1, GOOD_STRIDE):
            r1, c1 = r0 + PATCH_SIZE, c0 + PATCH_SIZE
            if np.mean(garment_mask[r0:r1, c0:c1]) < GARMENT_COVERAGE:
                continue
            if raw_mask[r0:r1, c0:c1].any():
                continue
            patches.append(image_np[r0:r1, c0:c1].copy())
    return patches


def find_pairs():
    pairs = []
    for png in sorted(DATA_ROOT.rglob("*.png")):
        coco = png.with_name(png.stem + "_coco.json")
        if not coco.exists():
            continue
        try:
            data = json.load(open(coco))
            if data.get("extension", {}).get(FILTER_FIELD, "").strip().lower() == FILTER_VALUE:
                pairs.append((png, coco))
        except Exception:
            continue
    return pairs


def save_patch(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def build_dataset():
    random.seed(RANDOM_SEED)
    pairs = find_pairs()
    print(f"\n[BUILD] {len(pairs)} {FILTER_VALUE} images found")

    all_good, all_defective = [], []
    ok = failed = skipped = 0

    for i, (png, coco) in enumerate(pairs, 1):
        print(f"  [{i}/{len(pairs)}] {png.name}", end="  ", flush=True)
        try:
            image_np         = np.array(Image.open(png).convert("RGB"))
            categories, anns = load_coco(coco)
            garment_mask     = load_garment_mask(png)
            if garment_mask is None:
                print("no mask - skipped"); skipped += 1; continue

            y_min, y_max, x_min, x_max = get_mask_bbox(garment_mask)
            cropped   = image_np[y_min:y_max, x_min:x_max].copy()
            crop_mask = garment_mask[y_min:y_max, x_min:x_max].copy()

            ch, cw = cropped.shape[:2]
            pad_h  = (PATCH_SIZE - ch % PATCH_SIZE) % PATCH_SIZE
            pad_w  = (PATCH_SIZE - cw % PATCH_SIZE) % PATCH_SIZE
            if pad_h or pad_w:
                cropped   = np.pad(cropped,   ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)
                crop_mask = np.pad(crop_mask, ((0, pad_h), (0, pad_w)),          constant_values=False)

            raw_defect = create_defect_mask(cropped.shape[:2], anns, categories, x_min, y_min)
            eroded     = erode_mask(raw_defect, EROSION_PX)
            good       = extract_good_patches(cropped, crop_mask, raw_defect)
            defective  = extract_defective_patches(
                cropped, crop_mask, eroded, raw_defect, anns, categories, x_min, y_min)

            all_good.extend(good); all_defective.extend(defective)
            print(f"good={len(good)}  def={len(defective)}")
            ok += 1
        except Exception as e:
            print(f"ERROR: {e}"); traceback.print_exc(); failed += 1

    print(f"\n  {ok} ok  {failed} failed  {skipped} skipped (no mask)")
    print(f"  good patches: {len(all_good)}  defective: {len(all_defective)}")

    random.shuffle(all_good)
    split      = int(len(all_good) * TRAIN_SPLIT)
    train_good = all_good[:split]
    test_good  = all_good[split:]

    print(f"\n  writing -> {OUTPUT_ROOT}")
    for idx, patch in enumerate(train_good):
        save_patch(patch, OUTPUT_ROOT / "train" / "good" / f"{idx:05d}.png")
    for idx, patch in enumerate(test_good[:TEST_CAP]):
        save_patch(patch, OUTPUT_ROOT / "test" / "good" / f"{idx:05d}.png")

    type_counters: dict = defaultdict(int)
    for patch, gt_mask, cat, _, _ in all_defective:
        if not cat or cat in EXCLUDE_CATEGORIES: continue
        idx = type_counters[cat]
        if idx >= TEST_CAP: continue
        name = f"{idx:05d}.png"
        save_patch(patch,   OUTPUT_ROOT / "test"         / cat / name)
        save_patch(gt_mask, OUTPUT_ROOT / "ground_truth" / cat / name)
        type_counters[cat] += 1

    print(f"  train/good: {len(train_good)}  test/good: {min(len(test_good), TEST_CAP)}")
    for cat, cnt in sorted(type_counters.items()):
        print(f"  test/{cat}: {cnt}")
    print("[BUILD] done.\n")


# model runner

def build_temp_dataset(src: Path, train_n: int) -> Path:
    tmp = Path(tempfile.mkdtemp()) / src.name
    tmp.mkdir(parents=True)
    random.seed(RANDOM_SEED)

    good_all = sorted((src / "train" / "good").glob("*.png"))
    sampled  = random.sample(good_all, min(train_n, len(good_all)))

    (tmp / "train" / "good").mkdir(parents=True)
    for p in sampled:
        (tmp / "train" / "good" / p.name).symlink_to(p.resolve())

    remaining = [p for p in good_all if p not in set(sampled)]
    (tmp / "val" / "good").mkdir(parents=True)
    for p in random.sample(remaining, min(VAL_SAMPLE, len(remaining))):
        (tmp / "val" / "good" / p.name).symlink_to(p.resolve())

    (tmp / "test" / "good").mkdir(parents=True)
    for p in sorted((src / "test" / "good").glob("*.png"))[:TEST_CAP]:
        (tmp / "test" / "good" / p.name).symlink_to(p.resolve())

    for cls in TEST_CLASSES:
        cls_test = src / "test" / cls
        cls_gt   = src / "ground_truth" / cls
        if not cls_test.exists(): continue
        (tmp / "test" / cls).mkdir(parents=True)
        (tmp / "ground_truth" / cls).mkdir(parents=True)
        for p in sorted(cls_test.glob("*.png"))[:TEST_CAP]:
            (tmp / "test" / cls / p.name).symlink_to(p.resolve())
            mask = cls_gt / p.name
            if mask.exists():
                (tmp / "ground_truth" / cls / p.name).symlink_to(mask.resolve())
    return tmp


def extract_metrics(results: list) -> dict:
    out = {"image_AUROC": None, "pixel_AUROC": None, "image_F1": None, "pixel_F1": None}
    if not results: return out
    flat = results[0] if isinstance(results[0], dict) else {}
    for k, v in flat.items():
        kl = k.lower()
        if   "image_auroc" in kl: out["image_AUROC"] = round(float(v), 4)
        elif "pixel_auroc" in kl: out["pixel_AUROC"] = round(float(v), 4)
        elif "pixel" in kl and "f1" in kl: out["pixel_F1"] = round(float(v), 4)
        elif "f1" in kl and "pixel" not in kl and out["image_F1"] is None:
            out["image_F1"] = round(float(v), 4)
    return out


# visualization

def apply_jet(arr: np.ndarray) -> np.ndarray:
    import matplotlib.cm as cm
    lo, hi = float(arr.min()), float(arr.max())
    norm   = (arr - lo) / (hi - lo + 1e-8)
    return (cm.jet(norm)[:, :, :3] * 255).astype(np.uint8)


def make_four_panel(orig_path: Path, gt_mask_path: Path | None,
                    anomaly_map: np.ndarray, pred_mask: np.ndarray) -> Image.Image:
    orig     = Image.open(orig_path).convert("RGB")
    W, H     = orig.size
    orig_arr = np.array(orig)

    if gt_mask_path and gt_mask_path.exists():
        gt = np.array(Image.open(gt_mask_path).convert("L").resize((W, H), Image.NEAREST))
        p2 = Image.fromarray(np.stack([gt, gt, gt], axis=-1))
    else:
        p2 = Image.new("RGB", (W, H), (0, 0, 0))

    if anomaly_map.ndim == 3: anomaly_map = anomaly_map[0]
    amap = np.array(Image.fromarray(anomaly_map).resize((W, H), Image.BILINEAR))
    p3   = Image.blend(orig, Image.fromarray(apply_jet(amap)), alpha=0.5)

    if pred_mask.ndim == 3: pred_mask = pred_mask[0]
    pmask   = np.array(
        Image.fromarray((pred_mask > 0.5).astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
    ) > 127
    overlay = orig_arr.copy()
    overlay[pmask, 0] = 255
    overlay[pmask, 1] = (orig_arr[pmask, 1] * 0.3).astype(np.uint8)
    overlay[pmask, 2] = (orig_arr[pmask, 2] * 0.3).astype(np.uint8)
    p4 = Image.fromarray(overlay)

    strip = Image.new("RGB", (W * 4, H))
    for j, panel in enumerate([orig, p2, p3, p4]):
        strip.paste(panel, (W * j, 0))
    return strip


def save_visualizations(batches, out_base: Path):
    out_base.mkdir(parents=True, exist_ok=True)
    saved = 0
    for batch in batches:
        img_paths  = batch.image_path
        if isinstance(img_paths, str): img_paths = [img_paths]
        anom_maps  = batch.anomaly_map.cpu().numpy()
        pred_masks = batch.pred_mask.cpu().numpy()
        if anom_maps.ndim  == 4: anom_maps  = anom_maps[:, 0]
        if pred_masks.ndim == 4: pred_masks = pred_masks[:, 0]
        for i, img_path in enumerate(img_paths):
            img_path = Path(img_path)
            category = img_path.parent.name
            gt_path  = (OUTPUT_ROOT / "ground_truth" / category / img_path.name
                        if category != "good" else None)
            try:
                vis = make_four_panel(img_path, gt_path, anom_maps[i], pred_masks[i])
            except Exception as e:
                print(f"    [WARN] {category}/{img_path.stem}: {e}"); continue
            out = out_base / category / f"{img_path.stem}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            vis.save(out)
            saved += 1
    print(f"  saved {saved} visualizations -> {out_base}")


# plots

def plot_training_curves():
    """Plot train_loss per epoch from CSVLogger (RD only - PaDiM has no gradient training)."""
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False

    for model_name, label in [("reverse_distillation", "RD"), ("padim", "PaDiM")]:
        log_dir = RESULTS_DIR / "training_logs" / model_name
        csvs    = sorted(log_dir.glob("version_*/metrics.csv")) if log_dir.exists() else []
        if not csvs:
            print(f"  [INFO] no training log for {model_name} - skipping curve")
            continue
        rows = []
        with open(csvs[-1]) as f:
            for row in csv.DictReader(f):
                # CSVLogger writes the per-epoch loss as train_loss_epoch
                loss = (row.get("train_loss_epoch") or row.get("train_loss") or "").strip()
                if loss:
                    try:
                        rows.append((int(float(row["epoch"])), float(loss)))
                    except ValueError:
                        pass
        if not rows:
            print(f"  [INFO] no train_loss rows for {model_name}")
            continue
        epochs = [r[0] for r in rows]
        losses = [r[1] for r in rows]
        ax.plot(epochs, losses, marker="o", markersize=3, label=f"{label} train loss")
        plotted = True

    if plotted:
        ax.set_xlabel("Epoch"); ax.set_ylabel("Train Loss")
        ax.set_title(f"Training loss - {FILTER_VALUE}  erosion_{EROSION_PX:02d}px")
        ax.legend(); ax.grid(True, alpha=0.3)
        out = plots_dir / "training_curves.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  training curves -> {out}")
    else:
        print("  [WARN] no training loss data - skipping training_curves.png")
    plt.close(fig)


def plot_metric_comparison(trained_rows: list[dict]):
    """Bar chart: zero-shot (jeans) vs trained (t-shirt). Reads zero-shot CSV if present."""
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    zero_shot_csv  = RESULTS_DIR / f"results_{FILTER_VALUE}_zero_shot.csv"
    zero_shot_rows = []
    if zero_shot_csv.exists():
        with open(zero_shot_csv) as f:
            zero_shot_rows = list(csv.DictReader(f))
    else:
        print("  [INFO] zero-shot CSV not found - comparison shows trained results only")

    model_names  = ["padim", "reverse_distillation"]
    model_labels = ["PaDiM", "RD"]
    metrics      = ["image_AUROC", "pixel_AUROC"]
    phases       = [("zero-shot (jeans)", zero_shot_rows, "#4C72B0"),
                    (f"trained ({FILTER_VALUE})", trained_rows, "#DD8452")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(model_names)); width = 0.35

    for ax, metric in zip(axes, metrics):
        for i, (label, rows, color) in enumerate(phases):
            vals = []
            for m in model_names:
                match = next((r for r in rows if r["model"] == m), None)
                v = float(match[metric]) if (match and match.get(metric) not in (None, "", "None")) else 0.0
                vals.append(v)
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)
            for bar, val in zip(bars, vals):
                if val:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(model_labels)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(metric)
        ax.legend(); ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Zero-shot (jeans model) vs Trained ({FILTER_VALUE})  |  erosion_{EROSION_PX:02d}px",
        fontsize=12)
    fig.tight_layout()
    out = plots_dir / "metric_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  metric comparison -> {out}")
    plt.close(fig)


# run models

def run_models():
    models = {
        "padim": {
            "fn":          lambda: Padim(backbone=BACKBONE, layers=["layer2", "layer3"],
                                         visualizer=False),
            "train_n":     5000,
            "train_batch": 32,
            "eval_batch":  32,
        },
        "reverse_distillation": {
            "fn":          lambda: ReverseDistillation(backbone=BACKBONE,
                                                       layers=["layer1", "layer2", "layer3"],
                                                       visualizer=False),
            "train_n":     5000,
            "train_batch": 16,
            "eval_batch":  4,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    src  = OUTPUT_ROOT

    for model_name, cfg in models.items():
        print(f"\n{'='*60}")
        print(f"  {model_name}  |  {FILTER_VALUE}  erosion_{EROSION_PX:02d}px")
        print(f"{'='*60}")

        tmp = build_temp_dataset(src, cfg["train_n"])
        try:
            datamodule = MVTecAD(
                root=str(tmp.parent),
                category=src.name,
                train_batch_size=cfg["train_batch"],
                eval_batch_size=cfg["eval_batch"],
            )
            model  = cfg["fn"]()
            logger = CSVLogger(
                save_dir=str(RESULTS_DIR / "training_logs"),
                name=model_name,
            )
            engine = Engine(max_epochs=MAX_EPOCHS, logger=logger)
            engine.fit(model=model, datamodule=datamodule)
            results = engine.test(model=model, datamodule=datamodule)
            print(f"  raw: {results}")
            metrics = extract_metrics(results)

            predictions = engine.predict(model=model, datamodule=datamodule,
                                         return_predictions=True)
            if predictions:
                save_visualizations(predictions,
                    RESULTS_DIR / "visualizations_trained" / model_name)

        except Exception as e:
            print(f"  [ERROR] {e}"); traceback.print_exc()
            metrics = {"image_AUROC": None, "pixel_AUROC": None,
                       "image_F1": None, "pixel_F1": None}
        finally:
            shutil.rmtree(tmp.parent, ignore_errors=True)

        rows.append({
            "erosion":     f"erosion_{EROSION_PX:02d}px",
            "model":       model_name,
            "image_AUROC": metrics["image_AUROC"],
            "pixel_AUROC": metrics["pixel_AUROC"],
            "image_F1":    metrics["image_F1"],
            "pixel_F1":    metrics["pixel_F1"],
        })

        csv_path = RESULTS_DIR / f"results_{FILTER_VALUE}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["erosion", "model", "image_AUROC", "pixel_AUROC", "image_F1", "pixel_F1"])
            writer.writeheader(); writer.writerows(rows)

    print(f"\n\n{'model':<25} {'img_AUROC':>10} {'pix_AUROC':>10} {'img_F1':>8} {'pix_F1':>8}")
    print("-" * 64)
    for r in rows:
        print(f"{r['model']:<25} {str(r['image_AUROC']):>10} {str(r['pixel_AUROC']):>10} "
              f"{str(r['image_F1']):>8} {str(r['pixel_F1']):>8}")
    print(f"\nresults -> {(RESULTS_DIR / f'results_{FILTER_VALUE}.csv').resolve()}")

    # plots
    print("\n[PLOTS]")
    plot_training_curves()
    plot_metric_comparison(rows)


# entry point

if __name__ == "__main__":
    if not OUTPUT_ROOT.exists() or not any(OUTPUT_ROOT.iterdir()):
        build_dataset()
    else:
        print(f"[BUILD] dataset already exists at {OUTPUT_ROOT}, skipping.")
        print("  Delete the folder to rebuild.\n")

    run_models()

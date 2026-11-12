"""
Build erosion_20px dataset from 202603_ReUseTex_Setup1, then run PaDiM and ReverseDistillation.
Garment masks: old dates use cached .npy, new dates fall back to _mask.png from 02_segmentation.
Run from project root: python models/run_rd_padim_20px.py
"""

import csv
import json
import random
import shutil
import tempfile
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# settings
FILTER_FIELD       = "garment_sub_category"
FILTER_VALUE       = "jeans"          # change this to run a different subcategory
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


def load_garment_mask(png_path: Path):
    """Load garment mask: try cached .npy first, then _mask.png from SegMasks."""
    rel = png_path.relative_to(DATA_ROOT)

    npy = CACHE_ROOT / FILTER_VALUE / rel.parent / png_path.stem / "garment_mask.npy"
    if npy.exists():
        return np.load(str(npy))

    seg_png = SEG_ROOT / rel.parent / (png_path.stem + "_mask.png")
    if seg_png.exists():
        return np.array(Image.open(seg_png).convert("L")) > 0

    return None


def load_coco(coco_path: Path):
    with open(coco_path) as f:
        data = json.load(f)
    categories  = {c["id"]: c["name"] for c in data["categories"]}
    return categories, data.get("annotations", [])


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
    patches   = []
    seen      = set()
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
        patch   = image_np[r0:r1, c0:c1].copy()
        gt_mask = (raw_mask[r0:r1, c0:c1] * 255).astype(np.uint8)
        patches.append((patch, gt_mask, cat, r0, c0))
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


# step 1: build dataset

def build_dataset():
    random.seed(RANDOM_SEED)
    pairs = find_pairs()
    print(f"\n[BUILD] {len(pairs)} {FILTER_VALUE} images found")

    all_good      = []
    all_defective = []
    ok = failed = skipped = 0

    for i, (png, coco) in enumerate(pairs, 1):
        print(f"  [{i}/{len(pairs)}] {png.name}", end="  ", flush=True)
        try:
            image_np         = np.array(Image.open(png).convert("RGB"))
            categories, anns = load_coco(coco)
            garment_mask     = load_garment_mask(png)

            if garment_mask is None:
                print("no mask - skipped")
                skipped += 1
                continue

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

            good      = extract_good_patches(cropped, crop_mask, raw_defect)
            defective = extract_defective_patches(
                cropped, crop_mask, eroded, raw_defect, anns, categories, x_min, y_min
            )

            all_good.extend(good)
            all_defective.extend(defective)
            print(f"good={len(good)}  def={len(defective)}")
            ok += 1

        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n  {ok} ok  {failed} failed  {skipped} skipped (no mask)")
    print(f"  good patches: {len(all_good)}  defective: {len(all_defective)}")

    random.shuffle(all_good)
    split      = int(len(all_good) * TRAIN_SPLIT)
    train_good = all_good[:split]
    test_good  = all_good[split:]

    out = OUTPUT_ROOT
    print(f"\n  writing -> {out}")

    for idx, patch in enumerate(train_good):
        save_patch(patch, out / "train" / "good" / f"{idx:05d}.png")
    for idx, patch in enumerate(test_good[:TEST_CAP]):
        save_patch(patch, out / "test" / "good" / f"{idx:05d}.png")

    type_counters: dict = defaultdict(int)
    for patch, gt_mask, cat, _, _ in all_defective:
        if not cat or cat in EXCLUDE_CATEGORIES:
            continue
        idx = type_counters[cat]
        if idx >= TEST_CAP:
            continue
        name = f"{idx:05d}.png"
        save_patch(patch,   out / "test"         / cat / name)
        save_patch(gt_mask, out / "ground_truth" / cat / name)
        type_counters[cat] += 1

    print(f"  train/good: {len(train_good)}  test/good: {min(len(test_good), TEST_CAP)}")
    for cat, cnt in sorted(type_counters.items()):
        print(f"  test/{cat}: {cnt}")
    print("[BUILD] done.\n")


# step 2: run models

def build_temp_dataset(src: Path, train_n: int) -> Path:
    tmp = Path(tempfile.mkdtemp()) / src.name
    tmp.mkdir(parents=True)
    random.seed(RANDOM_SEED)

    good_all = sorted((src / "train" / "good").glob("*.png"))
    sampled  = random.sample(good_all, min(train_n, len(good_all)))

    train_dir = tmp / "train" / "good"
    train_dir.mkdir(parents=True)
    for p in sampled:
        (train_dir / p.name).symlink_to(p.resolve())

    remaining = [p for p in good_all if p not in set(sampled)]
    val_dir = tmp / "val" / "good"
    val_dir.mkdir(parents=True)
    for p in random.sample(remaining, min(VAL_SAMPLE, len(remaining))):
        (val_dir / p.name).symlink_to(p.resolve())

    (tmp / "test" / "good").mkdir(parents=True)
    for p in sorted((src / "test" / "good").glob("*.png"))[:TEST_CAP]:
        (tmp / "test" / "good" / p.name).symlink_to(p.resolve())

    for cls in TEST_CLASSES:
        cls_test = src / "test" / cls
        cls_gt   = src / "ground_truth" / cls
        if not cls_test.exists():
            continue
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
    if not results:
        return out
    flat = results[0] if isinstance(results[0], dict) else {}
    for k, v in flat.items():
        kl = k.lower()
        if "image_auroc" in kl:
            out["image_AUROC"] = round(float(v), 4)
        elif "pixel_auroc" in kl:
            out["pixel_AUROC"] = round(float(v), 4)
        elif "pixel" in kl and "f1" in kl:
            out["pixel_F1"] = round(float(v), 4)
        elif "f1" in kl and "pixel" not in kl and out["image_F1"] is None:
            out["image_F1"] = round(float(v), 4)
    return out


def save_pred_masks(predictions, masks_dir: Path):
    """Save predicted binary masks from engine.predict() output."""
    import torch
    masks_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for batch in predictions:
        paths = getattr(batch, "image_path", None)
        masks = getattr(batch, "pred_mask", None)
        if paths is None or masks is None:
            continue
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        for i, img_path in enumerate(paths):
            stem     = Path(img_path).stem
            mask_np  = (masks[i] * 255).astype(np.uint8)
            Image.fromarray(mask_np).save(masks_dir / f"{stem}_pred_mask.png")
            saved += 1
    print(f"  saved {saved} prediction masks -> {masks_dir}")


def run_models():
    from anomalib.data import MVTecAD
    from anomalib.engine import Engine
    from anomalib.models import Padim, ReverseDistillation

    models = {
        "padim": {
            "fn":          lambda: Padim(backbone=BACKBONE, layers=["layer2", "layer3"]),
            "train_n":     5000,
            "train_batch": 32,
            "eval_batch":  32,
        },
        "reverse_distillation": {
            "fn":          lambda: ReverseDistillation(backbone=BACKBONE,
                                                       layers=["layer1", "layer2", "layer3"]),
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
        print(f"  {model_name}  |  erosion_{EROSION_PX:02d}px")
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
            engine = Engine(max_epochs=MAX_EPOCHS)
            engine.fit(model=model, datamodule=datamodule)
            results = engine.test(model=model, datamodule=datamodule)
            print(f"  raw: {results}")
            metrics = extract_metrics(results)

            # save predicted masks for all test images
            predictions = engine.predict(model=model, datamodule=datamodule, return_predictions=True)
            if predictions:
                save_pred_masks(predictions, RESULTS_DIR / "masks" / model_name)

        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            metrics = {"image_AUROC": None, "pixel_AUROC": None, "image_F1": None, "pixel_F1": None}
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
                f, fieldnames=["erosion", "model", "image_AUROC", "pixel_AUROC", "image_F1", "pixel_F1"]
            )
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n\n{'model':<25} {'img_AUROC':>10} {'pix_AUROC':>10} {'img_F1':>8} {'pix_F1':>8}")
    print("-" * 64)
    for r in rows:
        print(f"{r['model']:<25} {str(r['image_AUROC']):>10} {str(r['pixel_AUROC']):>10} "
              f"{str(r['image_F1']):>8} {str(r['pixel_F1']):>8}")
    print(f"\nresults -> {(RESULTS_DIR / f'results_{FILTER_VALUE}.csv').resolve()}")


# entry point

if __name__ == "__main__":
    if not OUTPUT_ROOT.exists() or not any(OUTPUT_ROOT.iterdir()):
        build_dataset()
    else:
        print(f"[BUILD] dataset already exists at {OUTPUT_ROOT}, skipping.")
        print("  Delete the folder to rebuild.\n")

    run_models()

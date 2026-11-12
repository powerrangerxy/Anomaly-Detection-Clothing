"""
Zero-shot evaluation: run the jeans-trained PaDiM + RD checkpoints on t-shirt patches.
No t-shirt training - purely tests how well the jeans model generalises.

Run from project root:  python models/run_tshirt_zero_shot.py

Outputs (under second_phase_rd_padim/t_shirt/):
  erosion_20px/                           <- t-shirt patch dataset (built if missing)
  results/
    results_t_shirt_zero_shot.csv
    visualizations_zero_shot/padim/...
    visualizations_zero_shot/reverse_distillation/...
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

# settings
FILTER_FIELD       = "garment_sub_category"
FILTER_VALUE       = "t_shirt"
JEANS_VALUE        = "jeans"
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

# paths
DATA_ROOT   = Path("202603_ReUseTex_Setup1/01_data")
SEG_ROOT    = Path("202603_ReUseTex_Setup1/02_segmentation")
CACHE_ROOT  = Path("datasets/custom_ivc_dataset/by_category")
OUTPUT_ROOT = Path("second_phase_rd_padim") / FILTER_VALUE / f"erosion_{EROSION_PX:02d}px"
RESULTS_DIR = Path("second_phase_rd_padim") / FILTER_VALUE / "results"
CKPT_BASE   = Path("results")

TEST_CLASSES = [
    "abrasion", "hole_tear", "missing_functional_elements",
    "other_defects", "seam_defect", "stain_dirt",
]

MODELS = {
    "padim": {
        "fn":          lambda: Padim(backbone=BACKBONE, layers=["layer2", "layer3"],
                                     visualizer=False),
        "ckpt_dir":    "Padim",
        "train_batch": 32,
        "eval_batch":  32,
    },
    "reverse_distillation": {
        "fn":          lambda: ReverseDistillation(backbone=BACKBONE,
                                                   layers=["layer1", "layer2", "layer3"],
                                                   visualizer=False),
        "ckpt_dir":    "ReverseDistillation",
        "train_batch": 16,
        "eval_batch":  4,
    },
}


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


def load_coco(path: Path):
    with open(path) as f:
        d = json.load(f)
    return {c["id"]: c["name"] for c in d["categories"]}, d.get("annotations", [])


def get_mask_bbox(mask):
    rows = np.any(mask, axis=1); cols = np.any(mask, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return int(y0), int(y1), int(x0), int(x1)


def create_defect_mask(shape_hw, anns, cats, x_off, y_off):
    h, w   = shape_hw
    canvas = Image.new("L", (w, h), 0)
    draw   = ImageDraw.Draw(canvas)
    for ann in anns:
        if cats.get(ann["category_id"], "") in EXCLUDE_CATEGORIES:
            continue
        for seg in ann.get("segmentation", []):
            pts = [(seg[i] - x_off, seg[i+1] - y_off) for i in range(0, len(seg), 2)]
            draw.polygon(pts, fill=255)
    return np.array(canvas) > 0


def erode_mask(mask, px):
    pil = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.array(pil.filter(ImageFilter.MinFilter(px * 2 + 1))) > 0


def annotation_centroids(anns, cats, x_off, y_off, img_h, img_w):
    centroids = []
    for ann in anns:
        if cats.get(ann["category_id"], "") in EXCLUDE_CATEGORIES:
            continue
        segs = ann.get("segmentation", [])
        if segs:
            seg = segs[0]
            xs  = [seg[i]   - x_off for i in range(0, len(seg), 2)]
            ys  = [seg[i]   - y_off for i in range(1, len(seg), 2)]
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
        else:
            x, y, w, h = ann["bbox"]
            cx = int(x + w/2) - x_off
            cy = int(y + h/2) - y_off
        cx = max(PATCH_SIZE//2, min(cx, img_w - PATCH_SIZE//2))
        cy = max(PATCH_SIZE//2, min(cy, img_h - PATCH_SIZE//2))
        centroids.append((cx, cy, cats.get(ann["category_id"], "")))
    return centroids


def extract_defective_patches(image_np, garment_mask, eroded, raw_mask,
                               anns, cats, x_off, y_off):
    h, w      = image_np.shape[:2]
    centroids = annotation_centroids(anns, cats, x_off, y_off, h, w)
    patches, seen = [], set()
    for cx, cy, cat in centroids:
        r0, c0 = cy - PATCH_SIZE//2, cx - PATCH_SIZE//2
        r1, c1 = r0 + PATCH_SIZE, c0 + PATCH_SIZE
        if (r0, c0) in seen: continue
        seen.add((r0, c0))
        if np.mean(garment_mask[r0:r1, c0:c1]) < GARMENT_COVERAGE: continue
        if np.mean(eroded[r0:r1, c0:c1]) < DEFECT_THRESHOLD: continue
        patches.append((image_np[r0:r1, c0:c1].copy(),
                        (raw_mask[r0:r1, c0:c1] * 255).astype(np.uint8),
                        cat))
    return patches


def extract_good_patches(image_np, garment_mask, raw_mask):
    h, w    = image_np.shape[:2]
    patches = []
    for r0 in range(0, h - PATCH_SIZE + 1, GOOD_STRIDE):
        for c0 in range(0, w - PATCH_SIZE + 1, GOOD_STRIDE):
            r1, c1 = r0 + PATCH_SIZE, c0 + PATCH_SIZE
            if np.mean(garment_mask[r0:r1, c0:c1]) < GARMENT_COVERAGE: continue
            if raw_mask[r0:r1, c0:c1].any(): continue
            patches.append(image_np[r0:r1, c0:c1].copy())
    return patches


def find_pairs():
    pairs = []
    for png in sorted(DATA_ROOT.rglob("*.png")):
        coco = png.with_name(png.stem + "_coco.json")
        if not coco.exists(): continue
        try:
            d = json.load(open(coco))
            if d.get("extension", {}).get(FILTER_FIELD, "").strip().lower() == FILTER_VALUE:
                pairs.append((png, coco))
        except Exception:
            continue
    return pairs


def save_patch(arr, path):
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
            image_np     = np.array(Image.open(png).convert("RGB"))
            cats, anns   = load_coco(coco)
            garment_mask = load_garment_mask(png)
            if garment_mask is None:
                print("no mask - skipped"); skipped += 1; continue

            y0, y1, x0, x1 = get_mask_bbox(garment_mask)
            cropped   = image_np[y0:y1, x0:x1].copy()
            crop_mask = garment_mask[y0:y1, x0:x1].copy()

            ch, cw = cropped.shape[:2]
            pad_h  = (PATCH_SIZE - ch % PATCH_SIZE) % PATCH_SIZE
            pad_w  = (PATCH_SIZE - cw % PATCH_SIZE) % PATCH_SIZE
            if pad_h or pad_w:
                cropped   = np.pad(cropped,   ((0,pad_h),(0,pad_w),(0,0)), constant_values=0)
                crop_mask = np.pad(crop_mask, ((0,pad_h),(0,pad_w)),       constant_values=False)

            raw_defect = create_defect_mask(cropped.shape[:2], anns, cats, x0, y0)
            eroded     = erode_mask(raw_defect, EROSION_PX)
            good       = extract_good_patches(cropped, crop_mask, raw_defect)
            defective  = extract_defective_patches(
                cropped, crop_mask, eroded, raw_defect, anns, cats, x0, y0)

            all_good.extend(good); all_defective.extend(defective)
            print(f"good={len(good)}  def={len(defective)}")
            ok += 1
        except Exception as e:
            print(f"ERROR: {e}"); traceback.print_exc(); failed += 1

    print(f"\n  {ok} ok  {failed} failed  {skipped} skipped")
    print(f"  good: {len(all_good)}  defective: {len(all_defective)}")

    random.shuffle(all_good)
    split = int(len(all_good) * TRAIN_SPLIT)
    for idx, p in enumerate(all_good[:split]):
        save_patch(p, OUTPUT_ROOT / "train" / "good" / f"{idx:05d}.png")
    for idx, p in enumerate(all_good[split:][:TEST_CAP]):
        save_patch(p, OUTPUT_ROOT / "test" / "good" / f"{idx:05d}.png")

    counters: dict = defaultdict(int)
    for patch, gt_mask, cat in all_defective:
        if not cat or cat in EXCLUDE_CATEGORIES: continue
        idx = counters[cat]
        if idx >= TEST_CAP: continue
        name = f"{idx:05d}.png"
        save_patch(patch,   OUTPUT_ROOT / "test"         / cat / name)
        save_patch(gt_mask, OUTPUT_ROOT / "ground_truth" / cat / name)
        counters[cat] += 1

    print(f"  train/good: {split}  test/good: {min(len(all_good)-split, TEST_CAP)}")
    for cat, cnt in sorted(counters.items()):
        print(f"  test/{cat}: {cnt}")
    print("[BUILD] done.\n")


# helpers

def build_temp_dataset() -> Path:
    """Minimal temp dataset for zero-shot (test images + 1 train image for datamodule)."""
    tmp = Path(tempfile.mkdtemp()) / OUTPUT_ROOT.name
    tmp.mkdir(parents=True)

    (tmp / "train" / "good").mkdir(parents=True)
    train_imgs = sorted((OUTPUT_ROOT / "train" / "good").glob("*.png"))
    if train_imgs:
        (tmp / "train" / "good" / train_imgs[0].name).symlink_to(train_imgs[0].resolve())

    (tmp / "val" / "good").mkdir(parents=True)

    (tmp / "test" / "good").mkdir(parents=True)
    for p in sorted((OUTPUT_ROOT / "test" / "good").glob("*.png"))[:TEST_CAP]:
        (tmp / "test" / "good" / p.name).symlink_to(p.resolve())

    for cls in TEST_CLASSES:
        cls_test = OUTPUT_ROOT / "test" / cls
        cls_gt   = OUTPUT_ROOT / "ground_truth" / cls
        if not cls_test.exists(): continue
        (tmp / "test" / cls).mkdir(parents=True)
        (tmp / "ground_truth" / cls).mkdir(parents=True)
        for p in sorted(cls_test.glob("*.png"))[:TEST_CAP]:
            (tmp / "test" / cls / p.name).symlink_to(p.resolve())
            mask = cls_gt / p.name
            if mask.exists():
                (tmp / "ground_truth" / cls / p.name).symlink_to(mask.resolve())
    return tmp


def find_jeans_checkpoint(ckpt_dir: str) -> Path | None:
    base  = CKPT_BASE / ckpt_dir / "MVTecAD" / f"erosion_{EROSION_PX:02d}px"
    ckpts = sorted(base.glob("v*/weights/lightning/model.ckpt")) if base.exists() else []
    return ckpts[-1] if ckpts else None


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
        gt  = np.array(Image.open(gt_mask_path).convert("L").resize((W, H), Image.NEAREST))
        p2  = Image.fromarray(np.stack([gt, gt, gt], axis=-1))
    else:
        p2  = Image.new("RGB", (W, H), (0, 0, 0))

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


# main

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not OUTPUT_ROOT.exists() or not any(OUTPUT_ROOT.iterdir()):
        build_dataset()
    else:
        print(f"[BUILD] dataset already exists at {OUTPUT_ROOT}, skipping.\n")

    rows = []
    for model_name, cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  {model_name}  |  zero-shot from {JEANS_VALUE} checkpoint")
        print(f"{'='*60}")

        ckpt = find_jeans_checkpoint(cfg["ckpt_dir"])
        if ckpt is None:
            print(f"  [SKIP] no jeans checkpoint found"); continue
        print(f"  jeans ckpt: {ckpt}")

        tmp = build_temp_dataset()
        try:
            datamodule = MVTecAD(
                root=str(tmp.parent),
                category=OUTPUT_ROOT.name,
                train_batch_size=cfg["train_batch"],
                eval_batch_size=cfg["eval_batch"],
            )
            model  = cfg["fn"]()
            engine = Engine()

            results = engine.test(model=model, datamodule=datamodule, ckpt_path=str(ckpt))
            print(f"  raw: {results}")
            metrics = extract_metrics(results)

            batches = engine.predict(model=model, datamodule=datamodule,
                                     ckpt_path=str(ckpt), return_predictions=True)
            if batches:
                save_visualizations(batches,
                    RESULTS_DIR / "visualizations_zero_shot" / model_name)

        except Exception as e:
            print(f"  [ERROR] {e}"); traceback.print_exc()
            metrics = {"image_AUROC": None, "pixel_AUROC": None,
                       "image_F1": None, "pixel_F1": None}
        finally:
            shutil.rmtree(tmp.parent, ignore_errors=True)

        rows.append({"model": model_name, **metrics})

    csv_path = RESULTS_DIR / f"results_{FILTER_VALUE}_zero_shot.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","image_AUROC","pixel_AUROC","image_F1","pixel_F1"])
        w.writeheader(); w.writerows(rows)

    print(f"\n{'model':<25} {'img_AUROC':>10} {'pix_AUROC':>10} {'img_F1':>8} {'pix_F1':>8}")
    print("-" * 64)
    for r in rows:
        print(f"{r['model']:<25} {str(r['image_AUROC']):>10} {str(r['pixel_AUROC']):>10}"
              f" {str(r['image_F1']):>8} {str(r['pixel_F1']):>8}")
    print(f"\nresults -> {csv_path.resolve()}")


if __name__ == "__main__":
    main()

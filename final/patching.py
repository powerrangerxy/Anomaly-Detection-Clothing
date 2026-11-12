"""Stage 1: cut every image of one garment type into patches (the patch pool).

Defect patches are centred on the annotated defects. Good patches come from a regular grid
and must not touch any annotated defect. Each patch is tagged with its garment id, so the
split in stage 2 can keep all patches of a garment on the same side.
"""

import csv
import json
import shutil
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from settings import (DATA_ROOT, DEFECT_THRESHOLD, EROSION_PX, EXCLUDE_CATEGORIES, GARMENT_COVERAGE,
                      GOOD_STRIDE, MASK_CACHE_ROOT, PATCH_SIZE, SEG_ROOT, Paths)

INDEX_FIELDS = ["name", "kind", "cls", "iid", "image", "area"]


def find_pairs(field: str, value: str) -> list[tuple[Path, Path]]:
    """All (image, COCO annotation) pairs whose metadata field matches the value."""
    pairs = []
    for png in sorted(DATA_ROOT.rglob("*.png")):
        coco = png.with_name(png.stem + "_coco.json")
        if not coco.exists():
            continue
        try:
            data = json.load(open(coco))
            if data.get("extension", {}).get(field, "").strip().lower() == value:
                pairs.append((png, coco))
        except Exception:
            continue
    return pairs


def garment_id(png: Path) -> int:
    """The garment id is the 2nd field of the file name, e.g. 000950_00638_... -> 638."""
    return int(png.stem.split("_")[1])


def load_garment_mask(png: Path, value: str):
    # For jeans, an older SAM3 mask is used if it exists. Otherwise the dataset's own segmentation.
    rel = png.relative_to(DATA_ROOT)
    npy = MASK_CACHE_ROOT / value / rel.parent / png.stem / "garment_mask.npy"
    if npy.exists():
        return np.load(str(npy))
    seg_png = SEG_ROOT / rel.parent / (png.stem + "_mask.png")
    if seg_png.exists():
        return np.array(Image.open(seg_png).convert("L")) > 0
    return None


def load_coco(coco_path: Path):
    with open(coco_path) as f:
        data = json.load(f)
    return {c["id"]: c["name"] for c in data["categories"]}, data.get("annotations", [])


def mask_bbox(mask: np.ndarray):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return int(y_min), int(y_max), int(x_min), int(x_max)


def defect_mask(shape_hw, annotations, categories, x_off, y_off) -> np.ndarray:
    """Draw all annotated defect polygons into one binary mask (shifted to the garment crop)."""
    h, w = shape_hw
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    for ann in annotations:
        if categories.get(ann["category_id"], "") in EXCLUDE_CATEGORIES:
            continue
        for seg in ann.get("segmentation", []):
            pts = [(seg[i] - x_off, seg[i + 1] - y_off) for i in range(0, len(seg), 2)]
            if len(pts) >= 2:
                draw.polygon(pts, fill=255)
    return np.array(canvas) > 0


def erode(mask: np.ndarray, px: int) -> np.ndarray:
    # same result as PIL's MinFilter(2*px+1), but much faster
    return ndimage.minimum_filter(mask, size=px * 2 + 1, mode="nearest")


def defect_centres(annotations, categories, x_off, y_off, img_h, img_w):
    """Centre of each defect, moved inwards so that a full patch fits into the image."""
    centres = []
    for ann in annotations:
        cat = categories.get(ann["category_id"], "")
        if cat in EXCLUDE_CATEGORIES:
            continue
        segs = ann.get("segmentation", [])
        if segs:
            seg = segs[0]
            cx = int(np.mean([seg[i] - x_off for i in range(0, len(seg), 2)]))
            cy = int(np.mean([seg[i] - y_off for i in range(1, len(seg), 2)]))
        else:
            x, y, w, h = ann["bbox"]
            cx = int(x + w / 2) - x_off
            cy = int(y + h / 2) - y_off
        half = PATCH_SIZE // 2
        cx = max(half, min(cx, img_w - half))
        cy = max(half, min(cy, img_h - half))
        centres.append((cx, cy, cat))
    return centres


def defect_patches(image, garment, eroded, raw, annotations, categories, x_off, y_off):
    h, w = image.shape[:2]
    patches, seen = [], set()
    for cx, cy, cat in defect_centres(annotations, categories, x_off, y_off, h, w):
        r0, c0 = cy - PATCH_SIZE // 2, cx - PATCH_SIZE // 2
        r1, c1 = r0 + PATCH_SIZE, c0 + PATCH_SIZE
        if (r0, c0) in seen:
            continue
        seen.add((r0, c0))
        if np.mean(garment[r0:r1, c0:c1]) < GARMENT_COVERAGE:
            continue
        if np.mean(eroded[r0:r1, c0:c1]) < DEFECT_THRESHOLD:
            continue
        patches.append((image[r0:r1, c0:c1].copy(), (raw[r0:r1, c0:c1] * 255).astype(np.uint8), cat))
    return patches


def good_patches(image, garment, raw):
    h, w = image.shape[:2]
    patches = []
    for r0 in range(0, h - PATCH_SIZE + 1, GOOD_STRIDE):
        for c0 in range(0, w - PATCH_SIZE + 1, GOOD_STRIDE):
            r1, c1 = r0 + PATCH_SIZE, c0 + PATCH_SIZE
            if np.mean(garment[r0:r1, c0:c1]) < GARMENT_COVERAGE:
                continue
            if raw[r0:r1, c0:c1].any():
                continue
            patches.append(image[r0:r1, c0:c1].copy())
    return patches


def patches_of_image(png: Path, coco: Path, value: str):
    """Return (good patches, defect patches) of one image, or None if it has no garment mask."""
    image = np.array(Image.open(png).convert("RGB"))
    categories, anns = load_coco(coco)
    garment = load_garment_mask(png, value)
    if garment is None or not garment.any():
        return None

    # crop to the garment and pad to a multiple of the patch size
    y_min, y_max, x_min, x_max = mask_bbox(garment)
    image = image[y_min:y_max, x_min:x_max].copy()
    garment = garment[y_min:y_max, x_min:x_max].copy()
    h, w = image.shape[:2]
    pad_h = (PATCH_SIZE - h % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - w % PATCH_SIZE) % PATCH_SIZE
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)
        garment = np.pad(garment, ((0, pad_h), (0, pad_w)), constant_values=False)

    raw = defect_mask(image.shape[:2], anns, categories, x_min, y_min)
    eroded = erode(raw, EROSION_PX)
    good = good_patches(image, garment, raw)
    defective = defect_patches(image, garment, eroded, raw, anns, categories, x_min, y_min)
    return good, defective


def save_png(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def build_pool(paths: Paths) -> dict:
    pairs = find_pairs(paths.field, paths.value)
    if not pairs:
        raise SystemExit(f"no images with {paths.field} = '{paths.value}' in {DATA_ROOT}. "
                         f"Check the spelling, or use --field garment_main_category for knit_wool and shirt_blouse.")
    shutil.rmtree(paths.pool, ignore_errors=True)  # remove patches of an earlier run
    print(f"\n[POOL] {len(pairs)} {paths.value} images found")
    index = []
    ok = failed = skipped = 0
    garments = set()
    for i, (png, coco) in enumerate(pairs, 1):
        print(f"  [{i}/{len(pairs)}] {png.name}", end="  ", flush=True)
        try:
            result = patches_of_image(png, coco, paths.value)
            if result is None:
                print("no mask - skipped")
                skipped += 1
                continue
            good, defective = result
            iid = garment_id(png)
            garments.add(iid)
            for k, patch in enumerate(good):
                name = f"{png.stem}_g{k:03d}"
                save_png(patch, paths.pool / "good" / f"{name}.png")
                index.append({"name": name, "kind": "good", "cls": "good", "iid": iid,
                              "image": png.stem, "area": 0.0})
            for k, (patch, gt, cat) in enumerate(defective):
                if not cat or cat in EXCLUDE_CATEGORIES:
                    continue
                name = f"{png.stem}_d{k:03d}"
                save_png(patch, paths.pool / cat / f"{name}.png")
                save_png(gt, paths.pool / "masks" / cat / f"{name}_mask.png")
                index.append({"name": name, "kind": "defect", "cls": cat, "iid": iid,
                              "image": png.stem, "area": round(float((gt > 0).mean()), 5)})
            print(f"good={len(good)}  def={len(defective)}")
            ok += 1
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            failed += 1

    with open(paths.pool / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(index)
    counts = defaultdict(int)
    for r in index:
        counts[r["cls"]] += 1
    stats = {"images_found": len(pairs), "images_ok": ok, "images_failed": failed,
             "images_no_mask": skipped, "garments": len(garments),
             "patches": dict(sorted(counts.items()))}
    with open(paths.pool / "pool_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print("[POOL] done.\n")
    return stats


def load_pool(paths: Paths) -> list[dict]:
    with open(paths.pool / "index.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["iid"] = int(r["iid"])
        r["area"] = float(r["area"])
    return rows

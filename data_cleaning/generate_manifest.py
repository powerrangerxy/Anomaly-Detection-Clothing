"""Write a manifest.csv for every erosion level of the focused jeans dataset.

build_focused_dataset.py only saves the patches with numbers as names, so we could not
tell which image a patch came from. This script repeats the same patch selection
(same settings and seed, but without saving images) and writes for every patch:
file name, split, class, source image, date, garment crop offset and patch position.
The settings must stay the same as in build_focused_dataset.py.
"""

import csv
import json
import random
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

FILTER_FIELD = "garment_sub_category"
FILTER_VALUE = "jeans"

DATA_ROOT = Path("202603_ReUseTex_Setup1")
CACHE_ROOT = Path("datasets/custom_ivc_dataset/by_category")
OUTPUT_ROOT = Path("datasets/focused_jeans")

EROSION_LEVELS_PX = [0, 20, 45]
PATCH_SIZE = 336
GARMENT_COVERAGE = 0.30
DEFECT_THRESHOLD = 0.02
GOOD_STRIDE = 336
EXCLUDE_CATEGORIES = {"brand_marker"}

TRAIN_SPLIT = 0.80
RANDOM_SEED = 42
TEST_CAP = 300

FIELDS = ["filename", "split", "defect_class", "source_image", "date",
          "garment_x_off", "garment_y_off", "patch_col", "patch_row"]


def load_coco(coco_path: Path):
    with open(coco_path) as f:
        data = json.load(f)
    categories = {c["id"]: c["name"] for c in data["categories"]}
    return categories, data.get("annotations", [])


def get_mask_bbox(mask: np.ndarray):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return int(y_min), int(y_max), int(x_min), int(x_max)


def create_defect_mask(shape_hw, annotations, categories, x_off, y_off):
    h, w = shape_hw
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    for ann in annotations:
        if categories.get(ann["category_id"], "") in EXCLUDE_CATEGORIES:
            continue
        for seg in ann.get("segmentation", []):
            pts = [(seg[i] - x_off, seg[i + 1] - y_off) for i in range(0, len(seg), 2)]
            draw.polygon(pts, fill=255)
    return np.array(canvas) > 0


def erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px == 0:
        return mask
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
            cx = int(np.mean([seg[i] - x_off for i in range(0, len(seg), 2)]))
            cy = int(np.mean([seg[i] - y_off for i in range(1, len(seg), 2)]))
        else:
            x, y, w, h = ann["bbox"]
            cx = int(x + w / 2) - x_off
            cy = int(y + h / 2) - y_off
        cx = max(PATCH_SIZE // 2, min(cx, img_w - PATCH_SIZE // 2))
        cy = max(PATCH_SIZE // 2, min(cy, img_h - PATCH_SIZE // 2))
        centroids.append((cx, cy, cat))
    return centroids


def extract_defective_patches(garment_mask, eroded_mask, raw_mask,
                              annotations, categories, x_off, y_off, h, w):
    """Same selection as in build_focused_dataset.py, but only returns (class, row, col)."""
    patches, seen = [], set()
    for cx, cy, cat in annotation_centroids(annotations, categories, x_off, y_off, h, w):
        r0, c0 = cy - PATCH_SIZE // 2, cx - PATCH_SIZE // 2
        r1, c1 = r0 + PATCH_SIZE, c0 + PATCH_SIZE
        if (r0, c0) in seen:
            continue
        seen.add((r0, c0))
        if np.mean(garment_mask[r0:r1, c0:c1]) < GARMENT_COVERAGE:
            continue
        if np.mean(eroded_mask[r0:r1, c0:c1]) < DEFECT_THRESHOLD:
            continue
        patches.append((cat, r0, c0))
    return patches


def extract_good_patches(garment_mask, raw_mask, h, w):
    patches = []
    for r0 in range(0, h - PATCH_SIZE + 1, GOOD_STRIDE):
        for c0 in range(0, w - PATCH_SIZE + 1, GOOD_STRIDE):
            r1, c1 = r0 + PATCH_SIZE, c0 + PATCH_SIZE
            if np.mean(garment_mask[r0:r1, c0:c1]) < GARMENT_COVERAGE:
                continue
            if raw_mask[r0:r1, c0:c1].any():
                continue
            patches.append((r0, c0))
    return patches


def load_cached_mask(png_path: Path):
    rel = png_path.relative_to(DATA_ROOT)
    path = CACHE_ROOT / FILTER_VALUE / rel.parent / png_path.stem / "garment_mask.npy"
    return np.load(str(path)) if path.exists() else None


def parse_date(stem: str) -> str:
    # 000319_00164_20260305_154150 -> 2026-03-05
    for p in stem.split("_"):
        if len(p) == 8 and p.isdigit():
            return f"{p[:4]}-{p[4:6]}-{p[6:]}"
    return ""


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


def row(filename, split, cls, stem, date, gx, gy, r0, c0):
    return {"filename": filename, "split": split, "defect_class": cls, "source_image": stem,
            "date": date, "garment_x_off": gx, "garment_y_off": gy, "patch_col": c0, "patch_row": r0}


def main():
    random.seed(RANDOM_SEED)
    pairs = find_pairs()
    print(f"found {len(pairs)} {FILTER_VALUE} images")

    all_good = []                                    # (r0, c0, stem, date, x_off, y_off)
    all_defective = [[] for _ in EROSION_LEVELS_PX]  # (cat, r0, c0, stem, date, x_off, y_off)
    ok = failed = skipped = 0

    for i, (png, coco) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {png.name}", end="  ", flush=True)
        try:
            garment_mask = load_cached_mask(png)
            if garment_mask is None:
                print("no cached mask - skipped")
                skipped += 1
                continue

            categories, annotations = load_coco(coco)
            y_min, y_max, x_min, x_max = get_mask_bbox(garment_mask)

            # size of the cropped and padded garment, the image itself is not needed
            crop_h, crop_w = y_max - y_min, x_max - x_min
            pad_h = (PATCH_SIZE - crop_h % PATCH_SIZE) % PATCH_SIZE
            pad_w = (PATCH_SIZE - crop_w % PATCH_SIZE) % PATCH_SIZE
            h, w = crop_h + pad_h, crop_w + pad_w

            crop_mask = garment_mask[y_min:y_max, x_min:x_max].copy()
            if pad_h or pad_w:
                crop_mask = np.pad(crop_mask, ((0, pad_h), (0, pad_w)), constant_values=False)
            raw_defect = create_defect_mask((h, w), annotations, categories, x_min, y_min)

            stem, date = png.stem, parse_date(png.stem)
            good = extract_good_patches(crop_mask, raw_defect, h, w)
            all_good.extend((r0, c0, stem, date, x_min, y_min) for r0, c0 in good)

            n_def = []
            for lvl, erosion_px in enumerate(EROSION_LEVELS_PX):
                defective = extract_defective_patches(crop_mask, erode_mask(raw_defect, erosion_px), raw_defect,
                                                      annotations, categories, x_min, y_min, h, w)
                all_defective[lvl].extend((cat, r0, c0, stem, date, x_min, y_min) for cat, r0, c0 in defective)
                n_def.append(len(defective))

            print(f"good={len(good)}  def={n_def}")
            ok += 1
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\nprocessed: {ok} ok  {failed} failed  {skipped} skipped")

    # same shuffle as build_focused_dataset.py, so the file names match
    random.shuffle(all_good)
    split = int(len(all_good) * TRAIN_SPLIT)
    train_good = all_good[:split]
    test_good = all_good[split:]

    for lvl, erosion_px in enumerate(EROSION_LEVELS_PX):
        out_csv = OUTPUT_ROOT / f"erosion_{erosion_px:02d}px" / "manifest.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for idx, (r0, c0, stem, date, gx, gy) in enumerate(train_good):
            rows.append(row(f"{idx:05d}.png", "train", "good", stem, date, gx, gy, r0, c0))
        for idx, (r0, c0, stem, date, gx, gy) in enumerate(test_good[:TEST_CAP]):
            rows.append(row(f"{idx:05d}.png", "test", "good", stem, date, gx, gy, r0, c0))

        type_counters = defaultdict(int)
        for cat, r0, c0, stem, date, gx, gy in all_defective[lvl]:
            if not cat or cat in EXCLUDE_CATEGORIES:
                continue
            idx = type_counters[cat]
            if idx >= TEST_CAP:
                continue
            rows.append(row(f"{idx:05d}.png", "test", cat, stem, date, gx, gy, r0, c0))
            type_counters[cat] += 1

        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {out_csv}")

    print("\ndone.")


if __name__ == "__main__":
    main()

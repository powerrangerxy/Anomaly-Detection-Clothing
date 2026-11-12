"""Build the first jeans dataset (224 px grid patches) in the MVTec AD folder layout.

    train/good/, test/good/, test/<defect>/, ground_truth/<defect>/

Uses the SAM3 masks saved by filter_and_run_pipeline.py. 80% of the good patches go to
training, the rest to test. A patch with several defect types is put into the folder of
the type with the most pixels in it.
"""

import json
import random
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import patch_pipeline as pp  # noqa: E402

DATA_ROOT = "../202603_ReUseTex_Setup1"
CACHE_ROOT = "../results/by_category"
OUTPUT_ROOT = "../results/by_category"
FILTER_FIELD = "garment_sub_category"
FILTER_VALUE = "jeans"
TRAIN_SPLIT = 0.80
RANDOM_SEED = 42


def matches_filter(json_path: Path, field: str, value: str) -> bool:
    try:
        with open(json_path) as f:
            data = json.load(f)
        return data.get("extension", {}).get(field, "").strip().lower() == value.strip().lower()
    except Exception:
        return False


def find_filtered_pairs(root: Path, field: str, value: str):
    pairs = []
    for png in sorted(root.rglob("*.png")):
        coco = png.with_name(png.stem + "_coco.json")
        if coco.exists() and matches_filter(coco, field, value):
            pairs.append((png, coco))
    return pairs


def load_cached_mask(png_path: Path, data_root: Path, cache_root: Path, filter_value: str):
    rel = png_path.relative_to(data_root)
    cache_path = cache_root / filter_value.lower() / rel.parent / png_path.stem / "garment_mask.npy"
    return np.load(str(cache_path)) if cache_path.exists() else None


def create_per_category_masks(shape_hw, annotations, categories, x_offset, y_offset):
    """One bool mask per defect type in the image, as {name: mask}."""
    h, w = shape_hw
    cat_masks = {}
    for ann in annotations:
        cat_name = categories[ann["category_id"]]
        if cat_name not in cat_masks:
            cat_masks[cat_name] = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(cat_masks[cat_name])
        for seg in ann["segmentation"]:
            remapped = pp.remap_polygon(seg, x_offset, y_offset)
            poly_tuples = [(remapped[i], remapped[i + 1]) for i in range(0, len(remapped), 2)]
            draw.polygon(poly_tuples, fill=255)
    return {name: np.array(img) > 0 for name, img in cat_masks.items()}


def dominant_defect_type(cat_masks_cropped: dict, row_start, col_start):
    """The defect type with the most pixels in this patch (None if there are none)."""
    row_end = row_start + pp.PATCH_SIZE
    col_end = col_start + pp.PATCH_SIZE
    best_name, best_count = None, 0
    for name, mask in cat_masks_cropped.items():
        count = int(np.sum(mask[row_start:row_end, col_start:col_end]))
        if count > best_count:
            best_count = count
            best_name = name
    return best_name


def process_one(png_path, json_path, data_root, cache_root, filter_value):
    """Returns the good patches and a list of (patch, gt mask, defect type)."""
    image = Image.open(png_path).convert("RGB")
    image_np = np.array(image)
    _, categories, annotations = pp.load_coco_annotations(json_path)

    garment_mask = load_cached_mask(png_path, data_root, cache_root, filter_value)
    if garment_mask is not None:
        print("  mask : from cache")
    else:
        print("  mask : running SAM3")
        candidates = pp.detect_garment_prompts(json_path)
        garment_mask, prompt = pp.segment_garment_with_fallback(image, candidates)
        print(f"  prompt : '{prompt}'")

    y_min, y_max, x_min, x_max = pp.get_mask_bbox(garment_mask)
    cropped_image = image_np[y_min:y_max, x_min:x_max].copy()
    cropped_garment_mask = garment_mask[y_min:y_max, x_min:x_max].copy()
    cropped_image[~cropped_garment_mask] = 0

    ch, cw = cropped_image.shape[:2]
    pad_h = (pp.PATCH_SIZE - ch % pp.PATCH_SIZE) % pp.PATCH_SIZE
    pad_w = (pp.PATCH_SIZE - cw % pp.PATCH_SIZE) % pp.PATCH_SIZE
    if pad_h or pad_w:
        cropped_image = np.pad(cropped_image, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)
        cropped_garment_mask = np.pad(cropped_garment_mask, ((0, pad_h), (0, pad_w)), constant_values=False)

    # the combined mask is the ground truth, the per-type masks decide the folder
    combined_defect_mask = pp.create_defect_mask(cropped_image.shape[:2], annotations, categories, x_min, y_min)
    per_cat_masks = create_per_category_masks(cropped_image.shape[:2], annotations, categories, x_min, y_min)

    h, w = cropped_image.shape[:2]
    good_patches, defective_patches = [], []
    for row_start in range(0, h - pp.PATCH_SIZE + 1, pp.STRIDE):
        for col_start in range(0, w - pp.PATCH_SIZE + 1, pp.STRIDE):
            row_end = row_start + pp.PATCH_SIZE
            col_end = col_start + pp.PATCH_SIZE

            if np.mean(cropped_garment_mask[row_start:row_end, col_start:col_end]) < pp.GARMENT_COVERAGE_THRESHOLD:
                continue

            patch = cropped_image[row_start:row_end, col_start:col_end].copy()
            defect_patch = combined_defect_mask[row_start:row_end, col_start:col_end]
            if np.mean(defect_patch) >= pp.DEFECT_THRESHOLD:
                gt_mask = (defect_patch * 255).astype(np.uint8)
                defect_type = dominant_defect_type(per_cat_masks, row_start, col_start)
                defective_patches.append((patch, gt_mask, defect_type or "other_defects"))
            else:
                good_patches.append(patch)

    print(f"  patches : {len(good_patches) + len(defective_patches)} total  "
          f"({len(good_patches)} good, {len(defective_patches)} defective)")
    return good_patches, defective_patches


def save_patch(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def main():
    random.seed(RANDOM_SEED)
    data_root = Path(DATA_ROOT)
    cache_root = Path(CACHE_ROOT)
    output_dir = Path(OUTPUT_ROOT) / (FILTER_VALUE.lower().replace(" ", "_") + "_mvtec")

    print(f"data root  : {data_root.resolve()}")
    print(f"cache root : {cache_root.resolve()}")
    print(f"output     : {output_dir.resolve()}")
    print(f"filter     : extension['{FILTER_FIELD}'] == '{FILTER_VALUE}'")
    print(f"train split: {int(TRAIN_SPLIT * 100)}% of good patches")
    print()

    pairs = find_filtered_pairs(data_root, FILTER_FIELD, FILTER_VALUE)
    print(f"found {len(pairs)} matching pairs\n")

    all_good_patches, all_defective_patches = [], []
    ok, failed = 0, 0
    for i, (png, coco) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {png.relative_to(data_root)}")
        try:
            good, defective = process_one(png, coco, data_root, cache_root, FILTER_VALUE)
            all_good_patches.extend(good)
            all_defective_patches.extend(defective)
            ok += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            failed += 1
        print()

    type_counts = Counter(dt for _, _, dt in all_defective_patches)
    print("=" * 60)
    print(f"processed  : {ok} ok,  {failed} failed")
    print(f"good total : {len(all_good_patches)}")
    print(f"defective  : {len(all_defective_patches)}")
    print("defect breakdown:")
    for name, count in sorted(type_counts.items()):
        print(f"  {name:<35} {count}")
    print()

    # split the good patches into train and test
    random.shuffle(all_good_patches)
    split_idx = int(len(all_good_patches) * TRAIN_SPLIT)
    train_good = all_good_patches[:split_idx]
    test_good = all_good_patches[split_idx:]

    print(f"saving {len(train_good)} train/good ...")
    for idx, patch in enumerate(train_good):
        save_patch(patch, output_dir / "train" / "good" / f"patch_{idx:05d}.png")

    print(f"saving {len(test_good)} test/good ...")
    for idx, patch in enumerate(test_good):
        save_patch(patch, output_dir / "test" / "good" / f"patch_{idx:05d}.png")

    # file names are numbered per defect type
    type_counters = {}
    print(f"saving {len(all_defective_patches)} defective patches + GT masks ...")
    for patch, gt_mask, defect_type in all_defective_patches:
        idx = type_counters.get(defect_type, 0)
        type_counters[defect_type] = idx + 1
        name = f"patch_{idx:05d}.png"
        save_patch(patch, output_dir / "test" / defect_type / name)
        save_patch(gt_mask, output_dir / "ground_truth" / defect_type / name)

    print()
    print("done.")
    print(f"  train/good     : {len(train_good)}")
    print(f"  test/good      : {len(test_good)}")
    for name, count in sorted(type_counters.items()):
        print(f"  test/{name:<30} {count}  (+ GT masks)")
    print(f"\noutput: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

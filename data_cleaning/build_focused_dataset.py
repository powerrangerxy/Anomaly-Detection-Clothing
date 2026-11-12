"""Build the defect-centred jeans dataset with 0, 20 and 45 px erosion.

Same patching as patch_pipeline_focused.py, but for all jeans images and without SAM3:
the garment masks saved by filter_and_run_pipeline.py are loaded instead.
brand_marker is not used as a defect, and the background around the garment is kept.

Output: datasets/focused_jeans/erosion_<px>/{train, test, ground_truth}
and two check images per image and erosion level in datasets/focused_jeans/vis/.
"""

import json
import random
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

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
TEST_CAP = 300  # max. test patches per class


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
        # move inwards so the patch fits into the image
        cx = max(PATCH_SIZE // 2, min(cx, img_w - PATCH_SIZE // 2))
        cy = max(PATCH_SIZE // 2, min(cy, img_h - PATCH_SIZE // 2))
        centroids.append((cx, cy, cat))
    return centroids


def extract_defective_patches(image_np, garment_mask, eroded_mask, raw_mask,
                              annotations, categories, x_off, y_off):
    h, w = image_np.shape[:2]
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
        patch = image_np[r0:r1, c0:c1].copy()
        gt_mask = (raw_mask[r0:r1, c0:c1] * 255).astype(np.uint8)
        patches.append((patch, gt_mask, cat, r0, c0))
    return patches


def extract_good_patches(image_np, garment_mask, raw_mask):
    h, w = image_np.shape[:2]
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


def _remap_polygon(seg, x_off, y_off):
    return [(seg[i] - x_off, seg[i + 1] - y_off) for i in range(0, len(seg), 2)]


def _load_font():
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", max(24, PATCH_SIZE // 6))
    except OSError:
        return ImageFont.load_default()


def save_debug_visualization(cropped_np, annotations, categories, x_off, y_off,
                             erosion_px, eroded_mask, out_path: Path):
    """Annotations in red and the part left after erosion in yellow."""
    base = Image.fromarray(cropped_np).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for ann in annotations:
        if categories.get(ann["category_id"], "") in EXCLUDE_CATEGORIES:
            continue
        for seg in ann.get("segmentation", []):
            pts = _remap_polygon(seg, x_off, y_off)
            if len(pts) >= 3:
                draw.polygon(pts, fill=(255, 0, 0, 80), outline=(255, 0, 0, 220))

    if erosion_px > 0:
        ero_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        ero_layer.paste((255, 215, 0, 130), mask=Image.fromarray(eroded_mask.astype(np.uint8) * 255))
        overlay = Image.alpha_composite(overlay, ero_layer)

    composite = Image.alpha_composite(base, overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(out_path)


def save_grid_visualization(cropped_np, garment_mask, eroded_mask, raw_mask,
                            annotations, categories, x_off, y_off,
                            erosion_px, defective_patches, out_path: Path):
    """All used patches on the image: defect patches in red, good patches in green, with their number."""
    h, w = cropped_np.shape[:2]
    base = Image.fromarray(cropped_np).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font()

    for ann in annotations:
        for seg in ann.get("segmentation", []):
            pts = _remap_polygon(seg, x_off, y_off)
            if len(pts) >= 3:
                draw.polygon(pts, fill=(255, 0, 0, 70), outline=(255, 0, 0, 200))

    for x in range(0, w, PATCH_SIZE):
        draw.line([(x, 0), (x, h)], fill=(255, 255, 255, 140), width=2)
    for y in range(0, h, PATCH_SIZE):
        draw.line([(0, y), (w, y)], fill=(255, 255, 255, 140), width=2)

    # (row, col, is_defect, number, defect type)
    cells = [(r0, c0, True, idx, cat) for idx, (_, _, cat, r0, c0) in enumerate(defective_patches)]
    def_positions = {(r0, c0) for _, _, _, r0, c0 in defective_patches}
    good_idx = 0
    for r0 in range(0, h - PATCH_SIZE + 1, GOOD_STRIDE):
        for c0 in range(0, w - PATCH_SIZE + 1, GOOD_STRIDE):
            if (r0, c0) in def_positions:
                continue
            if np.mean(garment_mask[r0:r0 + PATCH_SIZE, c0:c0 + PATCH_SIZE]) < GARMENT_COVERAGE:
                continue
            if raw_mask[r0:r0 + PATCH_SIZE, c0:c0 + PATCH_SIZE].any():
                continue
            cells.append((r0, c0, False, good_idx, ""))
            good_idx += 1

    for r0, c0, is_def, idx, cat in cells:
        r1, c1 = r0 + PATCH_SIZE, c0 + PATCH_SIZE
        color = (255, 60, 60, 220) if is_def else (60, 220, 60, 220)
        draw.rectangle([c0 + 1, r0 + 1, c1 - 1, r1 - 1], outline=color, width=4)

        bbox = draw.textbbox((0, 0), str(idx), font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = c0 + (PATCH_SIZE - tw) // 2
        ty = r0 + (PATCH_SIZE - th) // 2
        draw.text((tx + 2, ty + 2), str(idx), font=font, fill=(0, 0, 0, 200))
        draw.text((tx, ty), str(idx), font=font, fill=(255, 255, 255, 255))
        if is_def and cat:
            draw.text((c0 + 4, r0 + 4), cat[:10], font=font, fill=(255, 220, 0, 255))

    composite = Image.alpha_composite(base, overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composite.save(out_path)
    print(f"  grid saved: {out_path.name}  ({len(cells)} cells)")


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


def load_cached_mask(png_path: Path):
    rel = png_path.relative_to(DATA_ROOT)
    path = CACHE_ROOT / FILTER_VALUE / rel.parent / png_path.stem / "garment_mask.npy"
    return np.load(str(path)) if path.exists() else None


def process_image(png_path: Path, coco_path: Path):
    image_np = np.array(Image.open(png_path).convert("RGB"))
    categories, annotations = load_coco(coco_path)

    garment_mask = load_cached_mask(png_path)
    if garment_mask is None:
        return None

    # crop to the garment (background kept) and pad to a multiple of the patch size
    y_min, y_max, x_min, x_max = get_mask_bbox(garment_mask)
    cropped = image_np[y_min:y_max, x_min:x_max].copy()
    crop_mask = garment_mask[y_min:y_max, x_min:x_max].copy()
    ch, cw = cropped.shape[:2]
    pad_h = (PATCH_SIZE - ch % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - cw % PATCH_SIZE) % PATCH_SIZE
    if pad_h or pad_w:
        cropped = np.pad(cropped, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)
        crop_mask = np.pad(crop_mask, ((0, pad_h), (0, pad_w)), constant_values=False)

    raw_defect = create_defect_mask(cropped.shape[:2], annotations, categories, x_min, y_min)
    good = extract_good_patches(cropped, crop_mask, raw_defect)

    # the good patches are the same for all erosion levels, only the defect patches change
    defective_per_level = []
    for erosion_px in EROSION_LEVELS_PX:
        eroded = erode_mask(raw_defect, erosion_px)
        defective_per_level.append(extract_defective_patches(
            cropped, crop_mask, eroded, raw_defect, annotations, categories, x_min, y_min))

    return good, defective_per_level, cropped, crop_mask, raw_defect, annotations, categories, x_min, y_min


def save_patch(arr: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def main():
    random.seed(RANDOM_SEED)
    pairs = find_pairs()
    print(f"found {len(pairs)} {FILTER_VALUE} images")

    all_good = []
    all_defective = [[] for _ in EROSION_LEVELS_PX]
    ok = failed = skipped = 0

    for i, (png, coco) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {png.name}", end="  ", flush=True)
        try:
            result = process_image(png, coco)
            if result is None:
                print("no cached mask - skipped")
                skipped += 1
                continue
            good, defective_per_level, cropped, crop_mask, raw_defect, anns, cats, x_min, y_min = result
            all_good.extend(good)
            for lvl, defs in enumerate(defective_per_level):
                all_defective[lvl].extend(defs)

            for erosion_px, defective in zip(EROSION_LEVELS_PX, defective_per_level):
                eroded = erode_mask(raw_defect, erosion_px)
                vis_dir = OUTPUT_ROOT / "vis" / png.stem
                suffix = f"erosion_{erosion_px:02d}px"
                save_debug_visualization(cropped, anns, cats, x_min, y_min, erosion_px, eroded,
                                         vis_dir / f"garment_debug_{suffix}.png")
                save_grid_visualization(cropped, crop_mask, eroded, raw_defect, anns, cats, x_min, y_min,
                                        erosion_px, defective, vis_dir / f"garment_grid_{suffix}.png")

            print(f"good={len(good)}  def={[len(d) for d in defective_per_level]}")
            ok += 1
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\nprocessed: {ok} ok  {failed} failed  {skipped} skipped")
    print(f"total good patches : {len(all_good)}")
    for lvl, px in enumerate(EROSION_LEVELS_PX):
        print(f"  erosion {px:>2}px defective : {len(all_defective[lvl])}")

    random.shuffle(all_good)
    split = int(len(all_good) * TRAIN_SPLIT)
    train_good = all_good[:split]
    test_good = all_good[split:]

    for lvl, erosion_px in enumerate(EROSION_LEVELS_PX):
        out_dir = OUTPUT_ROOT / f"erosion_{erosion_px:02d}px"
        print(f"\n-- writing {out_dir}")

        for idx, patch in enumerate(train_good):
            save_patch(patch, out_dir / "train" / "good" / f"{idx:05d}.png")
        for idx, patch in enumerate(test_good[:TEST_CAP]):
            save_patch(patch, out_dir / "test" / "good" / f"{idx:05d}.png")

        type_counters = defaultdict(int)
        for patch, gt_mask, cat, _, _ in all_defective[lvl]:
            if not cat or cat in EXCLUDE_CATEGORIES:
                continue
            idx = type_counters[cat]
            if idx >= TEST_CAP:
                continue
            name = f"{idx:05d}.png"
            save_patch(patch, out_dir / "test" / cat / name)
            save_patch(gt_mask, out_dir / "ground_truth" / cat / name)
            type_counters[cat] += 1

        print(f"  train/good : {len(train_good)}")
        print(f"  test/good  : {min(len(test_good), TEST_CAP)}")
        for cat, cnt in sorted(type_counters.items()):
            print(f"  test/{cat} : {cnt}")

    print("\ndone.")


if __name__ == "__main__":
    main()

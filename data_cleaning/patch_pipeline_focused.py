"""Defect-centred patching for one image, to compare erosion levels.

With the 224 px grid, one defect at a patch border was split over up to four patches.
Here every defect patch (336 px) is centred on an annotated defect instead. The annotation
is shrunk (eroded) first, so a patch only counts if it contains the core of the defect.
Good patches come from a grid and must not touch any annotated pixel.

The image is processed with 0, 20 and 45 px erosion, and one picture per level shows
the original annotation (red), the eroded one (yellow) and the defect patches (green).
"""

import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).parent.parent))
import data_cleaning.patch_pipeline as pp  # noqa: E402

PATCH_SIZE = 336
GARMENT_COVERAGE_THRESHOLD = 0.30
DEFECT_THRESHOLD = 0.02   # min. share of (eroded) defect pixels in a defect patch
GOOD_STRIDE = 336
EROSION_LEVELS_PX = [0, 20, 45]

BASE_DIR = Path(__file__).parent
IMAGE_PATH = BASE_DIR / "third.png"
JSON_PATH = BASE_DIR / "third.json"
OUTPUT_DIR = BASE_DIR / "patches_focused"

# garment mask saved by an earlier SAM3 run, so SAM3 does not have to run again (None = run SAM3)
CACHED_MASK_PATH = BASE_DIR / "../datasets/custom_ivc_dataset/by_category/jeans/2026-03-09/2026-03-09_10-30-28/000404_00222_20260309_103028/garment_mask.npy"


def erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    if px == 0:
        return mask
    pil = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.array(pil.filter(ImageFilter.MinFilter(size=px * 2 + 1))) > 0


def annotation_centroids(annotations, categories, x_offset, y_offset, img_h, img_w):
    """Centre of every annotation (polygon mean, or bbox centre if there is no polygon)."""
    centroids = []
    for ann in annotations:
        cat = categories.get(ann["category_id"], "")
        if not ann.get("segmentation"):
            x, y, w, h = ann["bbox"]
            cx = int(x + w / 2) - x_offset
            cy = int(y + h / 2) - y_offset
        else:
            seg = ann["segmentation"][0]
            cx = int(np.mean([seg[i] - x_offset for i in range(0, len(seg), 2)]))
            cy = int(np.mean([seg[i] - y_offset for i in range(1, len(seg), 2)]))
        # move inwards so the patch fits into the image
        cx = max(PATCH_SIZE // 2, min(cx, img_w - PATCH_SIZE // 2))
        cy = max(PATCH_SIZE // 2, min(cy, img_h - PATCH_SIZE // 2))
        centroids.append((cx, cy, cat))
    return centroids


def extract_defective_patches(image_np, garment_mask, eroded_defect_mask, raw_defect_mask,
                              annotations, categories, x_offset, y_offset):
    h, w = image_np.shape[:2]
    patches = []
    seen = set()  # two annotations close together can give the same patch
    for cx, cy, cat in annotation_centroids(annotations, categories, x_offset, y_offset, h, w):
        r0 = cy - PATCH_SIZE // 2
        c0 = cx - PATCH_SIZE // 2
        r1, c1 = r0 + PATCH_SIZE, c0 + PATCH_SIZE
        if (r0, c0) in seen:
            continue
        seen.add((r0, c0))

        if np.mean(garment_mask[r0:r1, c0:c1]) < GARMENT_COVERAGE_THRESHOLD:
            continue
        if np.mean(eroded_defect_mask[r0:r1, c0:c1]) < DEFECT_THRESHOLD:
            continue  # too little defect left after erosion

        patch = image_np[r0:r1, c0:c1].copy()
        gt_mask = (raw_defect_mask[r0:r1, c0:c1] * 255).astype(np.uint8)
        patches.append((patch, gt_mask, cat, r0, c0))
    return patches


def extract_good_patches(image_np, garment_mask, raw_defect_mask):
    # good patches are checked against the original (not eroded) mask,
    # so not a single annotated pixel ends up in a good patch
    h, w = image_np.shape[:2]
    patches = []
    for r0 in range(0, h - PATCH_SIZE + 1, GOOD_STRIDE):
        for c0 in range(0, w - PATCH_SIZE + 1, GOOD_STRIDE):
            r1, c1 = r0 + PATCH_SIZE, c0 + PATCH_SIZE
            if np.mean(garment_mask[r0:r1, c0:c1]) < GARMENT_COVERAGE_THRESHOLD:
                continue
            if raw_defect_mask[r0:r1, c0:c1].any():
                continue
            patches.append(image_np[r0:r1, c0:c1].copy())
    return patches


def save_erosion_comparison(cropped_np, raw_defect, erosion_levels, defective_per_level, out_dir: Path):
    scale = 0.15  # the full image is too big to look at
    h, w = cropped_np.shape[:2]
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    base_pil = Image.fromarray(cropped_np).resize((sw, sh), Image.LANCZOS)

    frames = []
    for erosion_px, defective in zip(erosion_levels, defective_per_level):
        eroded = erode_mask(raw_defect, erosion_px)
        canvas = base_pil.convert("RGBA")

        # original annotation in red, eroded one in yellow
        for mask, color in [(raw_defect, (220, 50, 50, 90)), (eroded, (255, 220, 0, 140))]:
            scaled = Image.fromarray(mask.astype(np.uint8) * 255).resize((sw, sh), Image.NEAREST)
            layer = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
            layer.paste(color, mask=scaled)
            canvas = Image.alpha_composite(canvas, layer)

        # defect patches in green
        draw = ImageDraw.Draw(canvas)
        for _, _, _, r0, c0 in defective:
            draw.rectangle([int(c0 * scale), int(r0 * scale),
                            int((c0 + PATCH_SIZE) * scale), int((r0 + PATCH_SIZE) * scale)],
                           outline=(60, 220, 60, 230), width=3)

        draw.rectangle([0, 0, sw, 22], fill=(0, 0, 0, 160))
        draw.text((6, 4), f"erosion: {erosion_px}px  |  defective patches: {len(defective)}",
                  fill=(255, 255, 255, 255))

        rgb = canvas.convert("RGB")
        frames.append(rgb)
        rgb.save(out_dir / f"vis_erosion_{erosion_px:02d}px.png")

    # all levels next to each other
    strip = Image.new("RGB", (sw * len(frames) + 4 * (len(frames) - 1), sh), (30, 30, 30))
    for i, f in enumerate(frames):
        strip.paste(f, (i * (sw + 4), 0))
    strip.save(out_dir / "vis_comparison_strip.png")
    print(f"  visualisation saved to: {out_dir}")


def save_patches(defective, good, out_dir: Path):
    def_dir, good_dir, gt_dir = out_dir / "defective", out_dir / "good", out_dir / "ground_truth"
    for d in (def_dir, good_dir, gt_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    for idx, (patch, gt, cat, _, _) in enumerate(defective):
        Image.fromarray(patch).save(def_dir / f"{idx:04d}_{cat}.png")
        Image.fromarray(gt).save(gt_dir / f"{idx:04d}_{cat}.png")
    for idx, patch in enumerate(good):
        Image.fromarray(patch).save(good_dir / f"{idx:04d}.png")

    print(f"  defective : {len(defective)}")
    print(f"  good      : {len(good)}")


def main():
    print(f"patch: {PATCH_SIZE}px  |  defect threshold: {DEFECT_THRESHOLD:.1%}")
    print(f"erosion levels: {EROSION_LEVELS_PX} px\n")

    image = Image.open(IMAGE_PATH).convert("RGB")
    image_np = np.array(image)
    _, categories, annotations = pp.load_coco_annotations(JSON_PATH)

    if CACHED_MASK_PATH and CACHED_MASK_PATH.exists():
        print(f"loading cached mask: {CACHED_MASK_PATH}")
        garment_mask = np.load(CACHED_MASK_PATH)
    else:
        print("running SAM3...")
        garment_mask, _ = pp.segment_garment_with_fallback(image, pp.detect_garment_prompts(JSON_PATH))
    y_min, y_max, x_min, x_max = pp.get_mask_bbox(garment_mask)

    # unlike the grid version we keep the background, it gives the models
    # the shadows and lighting around the garment
    cropped = image_np[y_min:y_max, x_min:x_max].copy()
    crop_mask = garment_mask[y_min:y_max, x_min:x_max].copy()
    ch, cw = cropped.shape[:2]
    pad_h = (PATCH_SIZE - ch % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - cw % PATCH_SIZE) % PATCH_SIZE
    if pad_h or pad_w:
        cropped = np.pad(cropped, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)
        crop_mask = np.pad(crop_mask, ((0, pad_h), (0, pad_w)), constant_values=False)

    raw_defect = pp.create_defect_mask(cropped.shape[:2], annotations, categories, x_min, y_min)

    defective_per_level = []
    for erosion_px in EROSION_LEVELS_PX:
        label = f"erosion_{erosion_px:02d}px" if erosion_px > 0 else "erosion_none"
        print(f"\n-- {label}")
        eroded_defect = erode_mask(raw_defect, erosion_px)
        defective = extract_defective_patches(cropped, crop_mask, eroded_defect, raw_defect,
                                              annotations, categories, x_min, y_min)
        good = extract_good_patches(cropped, crop_mask, raw_defect)
        defective_per_level.append(defective)
        save_patches(defective, good, OUTPUT_DIR / label)
        print(f"  saved to: {OUTPUT_DIR / label}")

    print("\ngenerating visualisation...")
    save_erosion_comparison(cropped, raw_defect, EROSION_LEVELS_PX, defective_per_level, OUTPUT_DIR)
    print("\ndone.")


if __name__ == "__main__":
    main()

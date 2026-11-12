"""First patching pipeline, for one image.

SAM3 segments the garment, the image is cropped to the garment and cut into a
224x224 grid. A patch counts as defective if at least 1% of it is annotated defect.
Also saves two images to check the result (defect polygons and the numbered grid).
"""

import json
import os
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


PATCH_SIZE = 224
STRIDE = 224                       # no overlap
DEFECT_THRESHOLD = 0.01            # min. share of defect pixels for a defective patch
GARMENT_COVERAGE_THRESHOLD = 0.30  # min. share of garment pixels, otherwise the patch is skipped
EXCLUDE_CATEGORIES = []            # here brand_marker still counts as a defect

GARMENT_PROMPT = None              # set a text prompt by hand, otherwise it is taken from the json
BASE_DIR = Path(__file__).parent
IMAGE_PATH = BASE_DIR / "third.png"
JSON_PATH = BASE_DIR / "third.json"
OUTPUT_DIR = BASE_DIR / "patches_third"
GOOD_DIR = OUTPUT_DIR / "good"
DEFECTIVE_DIR = OUTPUT_DIR / "defective"
GARMET_VIS = OUTPUT_DIR / "garment_vis.png"
GRID_VIS = OUTPUT_DIR / "garment_grid.png"


def detect_garment_prompts(json_path):
    """Text prompts for SAM3: sub and main category from the json, then some general words."""
    def clean(s):
        # SAM3 needs plain english, without the "group_" prefix
        return s.replace("group_", "").replace("_", " ").strip()

    with open(json_path) as f:
        data = json.load(f)

    ext = data.get("extension", {})
    sub = ext.get("garment_sub_category", "")
    main = ext.get("garment_main_category", "")

    candidates = []
    if sub:
        candidates.append(clean(sub))
    if main:
        candidates.append(clean(main))
    for fb in ["trousers", "pants", "shirt", "jacket", "clothing", "garment"]:
        if fb not in candidates:
            candidates.append(fb)

    print(f"  prompt candidates : {candidates}")
    return candidates


def load_coco_annotations(json_path):
    with open(json_path) as f:
        data = json.load(f)
    categories = {cat["id"]: cat["name"] for cat in data["categories"]}
    return data["images"][0], categories, data["annotations"]


@lru_cache(maxsize=1)
def load_sam3():
    model = build_sam3_image_model().to("cuda").float()
    model.eval()
    return Sam3Processor(model)


def segment_garment_sam3(image_pil, prompt):
    """Run SAM3 with a text prompt. Returns one bool mask (H, W), True = garment."""
    processor = load_sam3()
    state = processor.set_image(image_pil)
    output = processor.set_text_prompt(state=state, prompt=prompt)
    masks = output["masks"]

    if isinstance(masks, torch.Tensor):
        masks_np = masks.cpu().numpy().astype(bool)
    else:
        masks_np = np.stack([np.array(m, dtype=bool) for m in masks])

    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np.squeeze(1)

    if masks_np.shape[0] == 0:
        raise RuntimeError(f"SAM3 returned no masks for prompt '{prompt}'.")

    # SAM3 can return several masks (e.g. left and right leg), so we merge them
    return masks_np.any(axis=0)


def segment_garment_with_fallback(image_pil, candidates):
    """Try the prompts one after the other until SAM3 finds something."""
    for prompt in candidates:
        try:
            print(f"  trying prompt: '{prompt}' ...", end=" ", flush=True)
            mask = segment_garment_sam3(image_pil, prompt)
            print("OK")
            return mask, prompt
        except RuntimeError:
            print("no mask")
    raise RuntimeError(f"SAM3 returned no masks for any candidate prompt: {candidates}")


def get_mask_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return int(y_min), int(y_max), int(x_min), int(x_max)


def remap_polygon(polygon, x_offset, y_offset):
    """Move a COCO polygon [x0, y0, x1, y1, ...] into the coordinates of the cropped image."""
    remapped = []
    for i in range(0, len(polygon), 2):
        remapped.append(polygon[i] - x_offset)
        remapped.append(polygon[i + 1] - y_offset)
    return remapped


def create_defect_mask(shape_hw, annotations, categories, x_offset, y_offset):
    h, w = shape_hw
    defect_mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(defect_mask)

    for ann in annotations:
        cat_name = categories[ann["category_id"]]
        if cat_name in EXCLUDE_CATEGORIES:
            continue
        for seg in ann["segmentation"]:
            remapped = remap_polygon(seg, x_offset, y_offset)
            poly_tuples = [(remapped[i], remapped[i + 1]) for i in range(0, len(remapped), 2)]
            draw.polygon(poly_tuples, fill=255)

    return np.array(defect_mask) > 0


def save_debug_visualization(cropped_image_np, annotations, categories,
                             x_offset, y_offset, out_path):
    """Cropped garment with the defect polygons in red, to check that the remapping is right."""
    base = Image.fromarray(cropped_image_np).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for ann in annotations:
        cat_name = categories[ann["category_id"]]
        if cat_name in EXCLUDE_CATEGORIES:
            continue
        for seg in ann["segmentation"]:
            remapped = remap_polygon(seg, x_offset, y_offset)
            poly_tuples = [(remapped[i], remapped[i + 1]) for i in range(0, len(remapped), 2)]
            draw.polygon(poly_tuples, fill=(255, 0, 0, 80), outline=(255, 0, 0, 220))

    composite = Image.alpha_composite(base, overlay).convert("RGB")
    os.makedirs(out_path.parent, exist_ok=True)
    composite.save(out_path)
    print(f"  debug visualisation saved to: {out_path}")


def save_grid_visualization(cropped_image_np, cropped_garment_mask, defect_mask,
                            annotations, categories, x_offset, y_offset, out_path):
    """Cropped garment with the patch grid on top.

    Every used patch gets a green (good) or red (defective) border and its number,
    which is the same number as in the saved file name.
    """
    h, w = cropped_image_np.shape[:2]
    base = Image.fromarray(cropped_image_np).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # all polygons, also the excluded ones, so we see everything that is annotated
    for ann in annotations:
        for seg in ann["segmentation"]:
            remapped = remap_polygon(seg, x_offset, y_offset)
            poly_tuples = [(remapped[i], remapped[i + 1]) for i in range(0, len(remapped), 2)]
            draw.polygon(poly_tuples, fill=(255, 0, 0, 70), outline=(255, 0, 0, 200))

    for x in range(0, w, PATCH_SIZE):
        draw.line([(x, 0), (x, h)], fill=(255, 255, 255, 140), width=2)
    for y in range(0, h, PATCH_SIZE):
        draw.line([(0, y), (w, y)], fill=(255, 255, 255, 140), width=2)

    try:
        font_size = max(24, PATCH_SIZE // 6)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    patch_idx = 0
    for row_start in range(0, h - PATCH_SIZE + 1, STRIDE):
        for col_start in range(0, w - PATCH_SIZE + 1, STRIDE):
            row_end = row_start + PATCH_SIZE
            col_end = col_start + PATCH_SIZE

            garment_patch = cropped_garment_mask[row_start:row_end, col_start:col_end]
            if np.mean(garment_patch) < GARMENT_COVERAGE_THRESHOLD:
                continue

            defect_patch = defect_mask[row_start:row_end, col_start:col_end]
            is_defective = np.mean(defect_patch) >= DEFECT_THRESHOLD

            border_color = (255, 60, 60, 220) if is_defective else (60, 220, 60, 220)
            draw.rectangle([col_start + 1, row_start + 1, col_end - 1, row_end - 1],
                           outline=border_color, width=4)

            # number in the middle of the cell, with a dark shadow so it is readable
            label = str(patch_idx)
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx = col_start + (PATCH_SIZE - tw) // 2
            ty = row_start + (PATCH_SIZE - th) // 2
            draw.text((tx + 2, ty + 2), label, font=font, fill=(0, 0, 0, 200))
            draw.text((tx, ty), label, font=font, fill=(255, 255, 255, 255))

            patch_idx += 1

    composite = Image.alpha_composite(base, overlay).convert("RGB")
    os.makedirs(out_path.parent, exist_ok=True)
    composite.save(out_path)
    print(f"  grid visualisation saved to: {out_path}  ({patch_idx} cells numbered)")


def extract_and_classify_patches(image_np, garment_mask, defect_mask):
    """Slide over the image and return (patch, is_defective, row, col, defect_ratio)."""
    h, w = image_np.shape[:2]
    patches = []

    for row_start in range(0, h - PATCH_SIZE + 1, STRIDE):
        for col_start in range(0, w - PATCH_SIZE + 1, STRIDE):
            row_end = row_start + PATCH_SIZE
            col_end = col_start + PATCH_SIZE

            garment_patch = garment_mask[row_start:row_end, col_start:col_end]
            if np.mean(garment_patch) < GARMENT_COVERAGE_THRESHOLD:
                continue  # mostly background

            patch = image_np[row_start:row_end, col_start:col_end].copy()
            defect_ratio = np.mean(defect_mask[row_start:row_end, col_start:col_end])
            is_defective = defect_ratio >= DEFECT_THRESHOLD

            patches.append((patch, is_defective, row_start, col_start, defect_ratio))

    return patches


def main():
    print(f"threshold: {DEFECT_THRESHOLD * 100:.1f}%  patch: {PATCH_SIZE}x{PATCH_SIZE}  stride: {STRIDE}")
    print()

    image = Image.open(IMAGE_PATH).convert("RGB")
    image_np = np.array(image)
    img_info, categories, annotations = load_coco_annotations(JSON_PATH)

    print(f"  image: {image_np.shape[1]}x{image_np.shape[0]}")
    print(f"  categories: {list(categories.values())}")
    print(f"  annotations: {len(annotations)}")
    print()

    candidates = [GARMENT_PROMPT] if GARMENT_PROMPT else detect_garment_prompts(JSON_PATH)

    print("running SAM3...")
    garment_mask, prompt = segment_garment_with_fallback(image, candidates)
    y_min, y_max, x_min, x_max = get_mask_bbox(garment_mask)
    print(f"  bbox: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
    print()

    # crop to the garment and set the background to black
    cropped_image = image_np[y_min:y_max, x_min:x_max].copy()
    cropped_garment_mask = garment_mask[y_min:y_max, x_min:x_max].copy()
    cropped_image[~cropped_garment_mask] = 0

    # pad so that the grid fits exactly
    ch, cw = cropped_image.shape[:2]
    pad_h = (PATCH_SIZE - ch % PATCH_SIZE) % PATCH_SIZE
    pad_w = (PATCH_SIZE - cw % PATCH_SIZE) % PATCH_SIZE
    if pad_h or pad_w:
        cropped_image = np.pad(cropped_image, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)
        cropped_garment_mask = np.pad(cropped_garment_mask, ((0, pad_h), (0, pad_w)), constant_values=False)
    print(f"  padded to: {cropped_image.shape[1]}x{cropped_image.shape[0]} "
          f"({cropped_image.shape[1] // PATCH_SIZE}x{cropped_image.shape[0] // PATCH_SIZE} grid)")

    defect_mask = create_defect_mask(cropped_image.shape[:2], annotations, categories,
                                     x_offset=x_min, y_offset=y_min)
    print(f"  defect annotations: {len([a for a in annotations if categories[a['category_id']] != 'brand_marker'])}")
    print()

    print("saving vis...")
    save_debug_visualization(cropped_image, annotations, categories, x_min, y_min, GARMET_VIS)
    save_grid_visualization(cropped_image, cropped_garment_mask, defect_mask, annotations, categories,
                            x_min, y_min, GRID_VIS)
    print()

    print("extracting patches...")
    patches = extract_and_classify_patches(cropped_image, cropped_garment_mask, defect_mask)
    good_count = sum(1 for p in patches if not p[1])
    defective_count = sum(1 for p in patches if p[1])
    print(f"  total: {len(patches)}  good: {good_count}  defective: {defective_count}")
    print()

    # remove the patches of an earlier run
    for d in (GOOD_DIR, DEFECTIVE_DIR):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    for idx, (patch, is_defective, row, col, defect_ratio) in enumerate(patches):
        save_dir = DEFECTIVE_DIR if is_defective else GOOD_DIR
        label = "defective" if is_defective else "good"
        Image.fromarray(patch).save(save_dir / f"patch_{idx:04d}_{label}.png")

    print(f"done, saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

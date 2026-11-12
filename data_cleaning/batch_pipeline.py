"""Run patch_pipeline.py on every image in a folder.

    python batch_pipeline.py ../202603_ReUseTex_Setup1/2026-03-04 ../results/patches_20260304

The SAM3 garment mask of every image is saved as garment_mask.npy next to its patches.
If the file exists, it is loaded instead of running SAM3 again.
"""

import argparse
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import patch_pipeline as pp  # noqa: E402


def find_pairs(root: Path):
    """All <name>.png files under root that have a <name>_coco.json next to them."""
    pairs = []
    for png in sorted(root.rglob("*.png")):
        coco = png.with_name(png.stem + "_coco.json")
        if coco.exists():
            pairs.append((png, coco))
        else:
            print(f"  [skip] no matching coco json for {png.name}")
    return pairs


def run_one(image_path: Path, json_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    good_dir = output_dir / "good"
    defective_dir = output_dir / "defective"

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    img_info, categories, annotations = pp.load_coco_annotations(json_path)
    print(f"  image size  : {image_np.shape[1]}x{image_np.shape[0]}")
    print(f"  annotations : {len(annotations)}")

    candidates = pp.detect_garment_prompts(json_path)
    mask_cache = output_dir / "garment_mask.npy"
    if mask_cache.exists():
        garment_mask = np.load(str(mask_cache))
        print("  mask        : loaded from cache")
    else:
        garment_mask, prompt = pp.segment_garment_with_fallback(image, candidates)
        print(f"  prompt used : '{prompt}'")
        np.save(str(mask_cache), garment_mask)

    y_min, y_max, x_min, x_max = pp.get_mask_bbox(garment_mask)
    print(f"  bbox        : x=[{x_min},{x_max}], y=[{y_min},{y_max}]")

    # crop to the garment, black background, pad to a multiple of the patch size
    cropped_image = image_np[y_min:y_max, x_min:x_max].copy()
    cropped_garment_mask = garment_mask[y_min:y_max, x_min:x_max].copy()
    cropped_image[~cropped_garment_mask] = 0

    ch, cw = cropped_image.shape[:2]
    pad_h = (pp.PATCH_SIZE - ch % pp.PATCH_SIZE) % pp.PATCH_SIZE
    pad_w = (pp.PATCH_SIZE - cw % pp.PATCH_SIZE) % pp.PATCH_SIZE
    if pad_h or pad_w:
        cropped_image = np.pad(cropped_image, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)
        cropped_garment_mask = np.pad(cropped_garment_mask, ((0, pad_h), (0, pad_w)), constant_values=False)

    defect_mask = pp.create_defect_mask(cropped_image.shape[:2], annotations, categories, x_min, y_min)
    active_defects = sum(1 for a in annotations if categories[a["category_id"]] not in pp.EXCLUDE_CATEGORIES)
    print(f"  defect anns : {active_defects}")

    pp.save_debug_visualization(cropped_image, annotations, categories, x_min, y_min,
                                output_dir / "garment_debug.png")
    pp.save_grid_visualization(cropped_image, cropped_garment_mask, defect_mask,
                               annotations, categories, x_min, y_min, output_dir / "garment_grid.png")

    patches = pp.extract_and_classify_patches(cropped_image, cropped_garment_mask, defect_mask)
    for d in (good_dir, defective_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    for idx, (patch, is_defective, row, col, defect_ratio) in enumerate(patches):
        save_dir = defective_dir if is_defective else good_dir
        label = "defective" if is_defective else "good"
        Image.fromarray(patch).save(save_dir / f"patch_{idx:04d}_{label}.png")

    good_count = sum(1 for p in patches if not p[1])
    defective_count = sum(1 for p in patches if p[1])
    print(f"  patches     : {len(patches)} total  ({good_count} good, {defective_count} defective)")
    return {"total": len(patches), "good": good_count, "defective": defective_count}


def main():
    parser = argparse.ArgumentParser(description="Batch garment patch pipeline")
    parser.add_argument("input_folder", type=Path, help="folder to search for PNG + JSON pairs")
    parser.add_argument("output_folder", type=Path, help="folder for the patches")
    args = parser.parse_args()

    pairs = find_pairs(args.input_folder)
    print(f"found {len(pairs)} PNG/JSON pairs under {args.input_folder}\n")

    ok, failed = 0, 0
    failed_list = []
    for i, (png, coco) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {png.name}")
        try:
            run_one(png, coco, args.output_folder / png.stem)
            ok += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            failed += 1
            failed_list.append(png.name)
        print()

    print("=" * 60)
    print(f"done.  {ok} succeeded,  {failed} failed")
    if failed_list:
        print("failed files:")
        for f in failed_list:
            print(f"  {f}")


if __name__ == "__main__":
    main()

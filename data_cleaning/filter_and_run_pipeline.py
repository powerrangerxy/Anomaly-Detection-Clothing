"""Run batch_pipeline.py on all images of one garment type.

The garment type is read from the "extension" block of the COCO json.
The output keeps the folder structure of the dataset, so images of different
sessions do not overwrite each other. We used it with FILTER_VALUE = "jeans".
"""

import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import batch_pipeline as bp  # noqa: E402

DATA_ROOT = "../202603_ReUseTex_Setup1"
OUTPUT_ROOT = "../results/by_category"
FILTER_FIELD = "garment_sub_category"
FILTER_VALUE = "jeans"


def matches_filter(json_path: Path, field: str, value: str) -> bool:
    try:
        with open(json_path) as f:
            data = json.load(f)
        actual = data.get("extension", {}).get(field, "")
        return actual.strip().lower() == value.strip().lower()
    except Exception:
        return False


def find_filtered_pairs(root: Path, field: str, value: str):
    pairs = []
    for png in sorted(root.rglob("*.png")):
        coco = png.with_name(png.stem + "_coco.json")
        if not coco.exists():
            continue
        if matches_filter(coco, field, value):
            pairs.append((png, coco))
    return pairs


def main():
    data_root = Path(DATA_ROOT)
    output_dir = Path(OUTPUT_ROOT) / FILTER_VALUE.lower().replace(" ", "_")

    print(f"data root    : {data_root.resolve()}")
    print(f"filter       : extension['{FILTER_FIELD}'] == '{FILTER_VALUE}'")
    print(f"output dir   : {output_dir.resolve()}")
    print()

    pairs = find_filtered_pairs(data_root, FILTER_FIELD, FILTER_VALUE)
    print(f"found {len(pairs)} matching PNG/JSON pairs\n")
    if not pairs:
        print("nothing to process, check the filter settings.")
        return

    ok, failed = 0, 0
    failed_list = []
    for i, (png, coco) in enumerate(pairs, 1):
        rel = png.relative_to(data_root)
        print(f"[{i}/{len(pairs)}] {rel}")
        try:
            bp.run_one(png, coco, output_dir / rel.parent / png.stem)
            ok += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            failed += 1
            failed_list.append(str(rel))
        print()

    print("=" * 60)
    print(f"done.  {ok} succeeded,  {failed} failed")
    if failed_list:
        print("failed files:")
        for f in failed_list:
            print(f"  {f}")


if __name__ == "__main__":
    main()

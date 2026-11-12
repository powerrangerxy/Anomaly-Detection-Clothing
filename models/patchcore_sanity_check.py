"""Quick check of the jeans_mvtec dataset before the long PatchCore run.

Checks the folders, image counts, that every test image has a mask, that images open,
and runs PatchCore on 5 images. Takes about 30 seconds.
"""
import random
import shutil
import tempfile
import sys
from pathlib import Path

from PIL import Image

DATASET_ROOT = "./datasets/custom_ivc_dataset/by_category"
CATEGORY     = "jeans_mvtec"
RANDOM_SEED  = 42
random.seed(RANDOM_SEED)

src = Path(DATASET_ROOT) / CATEGORY
errors   = []
warnings = []

print("=" * 60)
print("SANITY CHECK, PatchCore / jeans_mvtec")
print("=" * 60)

# required folders exist
print("\n[1] Checking required folders...")
required = ["train/good", "test/good", "ground_truth"]
for rel in required:
    p = src / rel
    if not p.exists():
        errors.append(f"Missing required folder: {p}")
    else:
        print(f"    OK  {rel}")

# image counts
print("\n[2] Counting images...")
train_good = sorted((src / "train" / "good").glob("*.png"))
print(f"    train/good        : {len(train_good):>6} images")
if len(train_good) < 10:
    errors.append(f"train/good has only {len(train_good)} images, too few to train")

test_good = sorted((src / "test" / "good").glob("*.png"))
print(f"    test/good         : {len(test_good):>6} images")

defect_dirs = [d for d in (src / "test").iterdir() if d.is_dir() and d.name != "good"]
total_anomalies = 0
for d in sorted(defect_dirs):
    count = len(sorted(d.glob("*.png")))
    total_anomalies += count
    print(f"    test/{d.name:<22}: {count:>6} images")
if total_anomalies == 0:
    errors.append("No anomaly images found in test/, cannot evaluate model")

# ground truth mask alignment
print("\n[3] Checking ground_truth mask alignment...")
gt_root = src / "ground_truth"
for d in sorted(defect_dirs):
    gt_dir = gt_root / d.name
    test_imgs = {p.stem for p in d.glob("*.png")}
    if not gt_dir.exists():
        warnings.append(f"No ground_truth folder for defect class '{d.name}'")
        print(f"    WARN  ground_truth/{d.name}, folder missing")
        continue
    # masks use same filename as test images (no _mask suffix)
    gt_masks = {p.stem for p in gt_dir.glob("*.png")}
    missing = test_imgs - gt_masks
    extra   = gt_masks - test_imgs
    if missing:
        warnings.append(f"{d.name}: {len(missing)} test images have no mask")
        print(f"    WARN  {d.name}: {len(missing)} images missing masks")
    elif extra:
        warnings.append(f"{d.name}: {len(extra)} masks have no matching test image")
        print(f"    WARN  {d.name}: {len(extra)} orphan masks")
    else:
        print(f"    OK    ground_truth/{d.name} ({len(gt_masks)} masks aligned)")

# spot-check images are readable and not corrupt
print("\n[4] Spot-checking image readability (5 random images per split)...")
def check_images(paths, label):
    sample = random.sample(paths, min(5, len(paths)))
    for p in sample:
        try:
            img = Image.open(p)
            img.verify()
        except Exception as e:
            errors.append(f"Corrupt image in {label}: {p.name}, {e}")
            print(f"    ERROR {label}: {p.name} is corrupt")
            return
    print(f"    OK  {label}, {len(sample)} images readable")

all_test_imgs = [p for d in defect_dirs for p in d.glob("*.png")]
check_images(train_good,    "train/good")
check_images(list(test_good), "test/good")
if all_test_imgs:
    check_images(all_test_imgs, "test/anomalies")

# quick pipeline dry run (5 images, CPU, no GPU needed)
print("\n[5] Dry-run pipeline with 5 images...")
try:
    import torch
    from anomalib.data import MVTecAD
    from anomalib.models import Patchcore
    from anomalib.engine import Engine

    tmp = Path(tempfile.mkdtemp()) / CATEGORY
    tmp.mkdir(parents=True)

    # train: 5 good images
    dr_train = tmp / "train" / "good"
    dr_train.mkdir(parents=True)
    for p in train_good[:5]:
        (dr_train / p.name).symlink_to(p.resolve())

    # val: good-only (MVTecAD does not support anomalous val images, crashes on mask_path=None)
    dr_val_good = tmp / "val" / "good"
    dr_val_good.mkdir(parents=True)
    for p in test_good[:5]:
        (dr_val_good / p.name).symlink_to(p.resolve())

    # test: symlink full test + ground_truth
    (tmp / "test").symlink_to((src / "test").resolve())
    (tmp / "ground_truth").symlink_to((src / "ground_truth").resolve())

    datamodule = MVTecAD(
        root=str(tmp.parent),
        category=CATEGORY,
        train_batch_size=5,
        eval_batch_size=5,
    )
    model = Patchcore(backbone="resnet18", layers=["layer2", "layer3"], coreset_sampling_ratio=0.1)
    engine = Engine()
    engine.fit(model=model, datamodule=datamodule)

    results = engine.test(datamodule=datamodule, model=model)
    print(f"\n    Dry-run results: {results}")

    # Check for all-zero metrics
    if results:
        r = results[0]
        auroc = r.get("image_AUROC", r.get("AUROC", None))
        if auroc is not None and auroc == 0.0:
            warnings.append("AUROC is 0.0 in dry run, check mask alignment or label assignment")

    shutil.rmtree(tmp.parent, ignore_errors=True)
    print("\n    Pipeline dry-run PASSED")

except Exception as e:
    errors.append(f"Pipeline dry-run FAILED: {e}")
    print(f"\n    ERROR: {e}")

# summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
if warnings:
    print(f"\nWarnings ({len(warnings)}):")
    for w in warnings:
        print(f"  WARN  {w}")
if errors:
    print(f"\nErrors ({len(errors)}):")
    for e in errors:
        print(f"  ERROR {e}")
    print("\nFix the errors above before running full training.")
    sys.exit(1)
else:
    print("\nAll checks passed. Safe to run full training.")

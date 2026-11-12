
# run_patchcore_focused.py
#
# Runs PatchCore on all 3 focused erosion datasets.
# Kept separate so GPU memory is clean (no leftover from other models).
# Appends results to results/focused_benchmark/results.csv

import csv
import random
import shutil
import tempfile
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

# config
DATASET_ROOT   = Path("datasets/focused_jeans")
RESULTS_CSV    = Path("results/focused_benchmark/results.csv")
EROSION_LABELS = ["erosion_00px", "erosion_20px", "erosion_45px"]

TEST_CLASSES = [
    "abrasion", "hole_tear", "missing_functional_elements",
    "other_defects", "seam_defect", "stain_dirt",
]

BACKBONE    = "resnet18"
TRAIN_N     = 1000   # reduced from 2000 - coreset OOM'd with leftover GPU memory
VAL_SAMPLE  = 200
TEST_CAP    = 300
RANDOM_SEED = 42


def build_temp(src: Path) -> Path:
    tmp = Path(tempfile.mkdtemp()) / src.name
    tmp.mkdir(parents=True)

    random.seed(RANDOM_SEED)
    good_all = sorted((src / "train" / "good").glob("*.png"))
    sampled  = random.sample(good_all, min(TRAIN_N, len(good_all)))

    (tmp / "train" / "good").mkdir(parents=True)
    for p in sampled:
        (tmp / "train" / "good" / p.name).symlink_to(p.resolve())

    remaining = [p for p in good_all if p not in set(sampled)]
    (tmp / "val" / "good").mkdir(parents=True)
    for p in random.sample(remaining, min(VAL_SAMPLE, len(remaining))):
        (tmp / "val" / "good" / p.name).symlink_to(p.resolve())

    (tmp / "test" / "good").mkdir(parents=True)
    for p in sorted((src / "test" / "good").glob("*.png"))[:TEST_CAP]:
        (tmp / "test" / "good" / p.name).symlink_to(p.resolve())

    for cls in TEST_CLASSES:
        cls_src = src / "test" / cls
        cls_gt  = src / "ground_truth" / cls
        if not cls_src.exists():
            continue
        (tmp / "test" / cls).mkdir(parents=True)
        (tmp / "ground_truth" / cls).mkdir(parents=True)
        for p in sorted(cls_src.glob("*.png"))[:TEST_CAP]:
            (tmp / "test" / cls / p.name).symlink_to(p.resolve())
            mask = cls_gt / p.name
            if mask.exists():
                (tmp / "ground_truth" / cls / p.name).symlink_to(mask.resolve())

    return tmp


def extract_metrics(results):
    out = {"image_AUROC": None, "pixel_AUROC": None, "F1": None}
    if not results:
        return out
    flat = results[0] if isinstance(results[0], dict) else {}
    for k, v in flat.items():
        kl = k.lower()
        if "image_auroc" in kl:
            out["image_AUROC"] = round(float(v), 4)
        elif "pixel_auroc" in kl:
            out["pixel_AUROC"] = round(float(v), 4)
        elif "f1" in kl and out["F1"] is None:
            out["F1"] = round(float(v), 4)
    return out


def load_existing():
    rows = []
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV) as f:
            rows = list(csv.DictReader(f))
    return rows


def save_csv(rows):
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["erosion", "model", "image_AUROC", "pixel_AUROC", "F1"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = load_existing()
    # remove old empty patchcore rows
    rows = [r for r in rows if not (r["model"] == "patchcore" and not r["image_AUROC"])]

    for erosion_label in EROSION_LABELS:
        src = DATASET_ROOT / erosion_label
        print(f"\n{'='*55}")
        print(f"  patchcore  |  {erosion_label}  |  train_n={TRAIN_N}")
        print(f"{'='*55}")

        tmp = build_temp(src)
        try:
            datamodule = MVTecAD(
                root=str(tmp.parent),
                category=src.name,
                train_batch_size=32,
                eval_batch_size=8,
            )
            model  = Patchcore(backbone=BACKBONE, layers=["layer2", "layer3"],
                               coreset_sampling_ratio=0.15)
            engine = Engine()
            engine.fit(model=model, datamodule=datamodule)
            results = engine.test(model=model, datamodule=datamodule)
            print(f"  raw: {results}")
            metrics = extract_metrics(results)
        except Exception as e:
            print(f"  [ERROR] {e}")
            metrics = {"image_AUROC": None, "pixel_AUROC": None, "F1": None}
        finally:
            shutil.rmtree(tmp.parent, ignore_errors=True)

        rows.append({"erosion": erosion_label, "model": "patchcore", **metrics})
        save_csv(rows)
        print(f"  saved -> {RESULTS_CSV}")

    # final print
    patchcore_rows = [r for r in rows if r["model"] == "patchcore"]
    print("\n-- PatchCore Results ------------------------------")
    print(f"{'erosion':<20} {'img_AUROC':>10} {'pix_AUROC':>10} {'F1':>8}")
    for r in patchcore_rows:
        print(f"{r['erosion']:<20} {str(r['image_AUROC']):>10} {str(r['pixel_AUROC']):>10} {str(r['F1']):>8}")


if __name__ == "__main__":
    main()

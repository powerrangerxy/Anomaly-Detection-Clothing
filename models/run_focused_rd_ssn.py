
# run_focused_rd_ssn.py
#
# Runs ReverseDistillation and SuperSimpleNet on the full focused dataset
# (no test-image cap - safe since neither model uses an embedding bank).
# Saves image_AUROC, pixel_AUROC, F1 to results/focused_rd_ssn/results.csv
#
# Run from the project root:  python models/run_focused_rd_ssn.py

import csv
import random
import shutil
import tempfile
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import ReverseDistillation, Supersimplenet

# config
DATASET_ROOT  = Path("datasets/focused_jeans")
RESULTS_DIR   = Path("results/focused_rd_ssn")
EROSION_LABELS = ["erosion_00px", "erosion_20px", "erosion_45px"]

TEST_CLASSES = [
    "abrasion", "hole_tear", "missing_functional_elements",
    "other_defects", "seam_defect", "stain_dirt",
]

BACKBONE    = "resnet18"
MAX_EPOCHS  = 50
RANDOM_SEED = 42
VAL_SAMPLE  = 200

MODELS = {
    "reverse_distillation": {
        "fn":          lambda: ReverseDistillation(backbone=BACKBONE,
                                                   layers=["layer1", "layer2", "layer3"]),
        "train_n":     5000,
        "train_batch": 16,
        "eval_batch":  4,
    },
    "supersimplenet": {
        "fn":          lambda: Supersimplenet(backbone=BACKBONE, layers=["layer2", "layer3"]),
        "train_n":     5000,
        "train_batch": 16,
        "eval_batch":  4,
    },
}


def build_temp_dataset(src: Path, train_n: int, seed: int) -> Path:
    tmp = Path(tempfile.mkdtemp()) / src.name
    tmp.mkdir(parents=True)

    random.seed(seed)
    good_all = sorted((src / "train" / "good").glob("*.png"))
    sampled  = random.sample(good_all, min(train_n, len(good_all)))

    train_dir = tmp / "train" / "good"
    train_dir.mkdir(parents=True)
    for p in sampled:
        (train_dir / p.name).symlink_to(p.resolve())

    remaining = [p for p in good_all if p not in set(sampled)]
    val_dir = tmp / "val" / "good"
    val_dir.mkdir(parents=True)
    for p in random.sample(remaining, min(VAL_SAMPLE, len(remaining))):
        (val_dir / p.name).symlink_to(p.resolve())

    # full test set - no cap
    (tmp / "test" / "good").mkdir(parents=True)
    for p in sorted((src / "test" / "good").glob("*.png")):
        (tmp / "test" / "good" / p.name).symlink_to(p.resolve())

    for cls in TEST_CLASSES:
        cls_test = src / "test" / cls
        cls_gt   = src / "ground_truth" / cls
        if not cls_test.exists():
            continue
        (tmp / "test" / cls).mkdir(parents=True)
        (tmp / "ground_truth" / cls).mkdir(parents=True)
        for p in sorted(cls_test.glob("*.png")):
            (tmp / "test" / cls / p.name).symlink_to(p.resolve())
            mask = cls_gt / p.name
            if mask.exists():
                (tmp / "ground_truth" / cls / p.name).symlink_to(mask.resolve())

    return tmp


def extract_metrics(results: list) -> dict:
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


def run_one(model_name: str, cfg: dict, src: Path) -> dict:
    tmp = build_temp_dataset(src, cfg["train_n"], RANDOM_SEED)
    try:
        datamodule = MVTecAD(
            root=str(tmp.parent),
            category=src.name,
            train_batch_size=cfg["train_batch"],
            eval_batch_size=cfg["eval_batch"],
        )
        model  = cfg["fn"]()
        engine = Engine(max_epochs=MAX_EPOCHS)
        engine.fit(model=model, datamodule=datamodule)
        results = engine.test(model=model, datamodule=datamodule)
        print(f"  raw results: {results}")
        return extract_metrics(results)
    except Exception as e:
        print(f"  [ERROR] {e}")
        return {"image_AUROC": None, "pixel_AUROC": None, "F1": None}
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for erosion_label in EROSION_LABELS:
        src = DATASET_ROOT / erosion_label
        if not src.exists():
            print(f"[SKIP] {src} not found - run build_focused_dataset.py first")
            continue

        for model_name, cfg in MODELS.items():
            print(f"\n{'='*60}")
            print(f"  {model_name}  |  {erosion_label}")
            print(f"{'='*60}")

            metrics = run_one(model_name, cfg, src)
            rows.append({
                "erosion":     erosion_label,
                "model":       model_name,
                "image_AUROC": metrics["image_AUROC"],
                "pixel_AUROC": metrics["pixel_AUROC"],
                "F1":          metrics["F1"],
            })

            csv_path = RESULTS_DIR / "results.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["erosion", "model", "image_AUROC", "pixel_AUROC", "F1"])
                writer.writeheader()
                writer.writerows(rows)

    print(f"\n\n{'erosion':<20} {'model':<25} {'img_AUROC':>10} {'pix_AUROC':>10} {'F1':>8}")
    print("-" * 76)
    for r in rows:
        print(f"{r['erosion']:<20} {r['model']:<25} "
              f"{str(r['image_AUROC']):>10} {str(r['pixel_AUROC']):>10} {str(r['F1']):>8}")
    print(f"\nresults -> {(RESULTS_DIR / 'results.csv').resolve()}")


if __name__ == "__main__":
    main()

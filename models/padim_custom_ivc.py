import random
import shutil
import tempfile
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim

# Config
DATASET_ROOT = "./datasets/custom_ivc_dataset/by_category"
CATEGORY     = "jeans_mvtec"
TRAIN_SAMPLE = 13000  # PaDiM fits a Gaussian -> lighter than PatchCore memory bank
RANDOM_SEED  = 42

# Defect classes to include in test - excludes "defective" which is a duplicate
# superset of all other classes and breaks mask assignment in Anomalib
TEST_CLASSES = ["abrasion", "brand_marker", "hole_tear", "missing_functional_elements",
                "other_defects", "seam_defect", "stain_dirt"]

src = Path(DATASET_ROOT) / CATEGORY
tmp = Path(tempfile.mkdtemp()) / CATEGORY
tmp.mkdir(parents=True)

random.seed(RANDOM_SEED)
good_patches = sorted((src / "train" / "good").glob("*.png"))
sampled = random.sample(good_patches, min(TRAIN_SAMPLE, len(good_patches)))

# train
train_dir = tmp / "train" / "good"
train_dir.mkdir(parents=True)
for p in sampled:
    (train_dir / p.name).symlink_to(p.resolve())

# val: good-only (MVTecAD does not support anomalous val images)
remaining = [p for p in good_patches if p not in set(sampled)]
val_dir = tmp / "val" / "good"
val_dir.mkdir(parents=True)
for p in random.sample(remaining, min(200, len(remaining))):
    (val_dir / p.name).symlink_to(p.resolve())

# test: good + specific defect classes only (no catch-all "defective" folder)
# Cap per-class images to limit GPU memory used by pixel-level metric accumulation.
# Anomalib stacks ALL pixel maps (N x 224 x 224) for argsort -> OOM with 12K images.
TEST_CAP = 300  # images per class; 300 x 8 classes x 224^2 x 4B ~ 1.2 GB -> fits

(tmp / "test" / "good").mkdir(parents=True)
for p in sorted((src / "test" / "good").glob("*.png"))[:TEST_CAP]:
    (tmp / "test" / "good" / p.name).symlink_to(p.resolve())

for cls in TEST_CLASSES:
    (tmp / "test" / cls).mkdir(parents=True)
    (tmp / "ground_truth" / cls).mkdir(parents=True)
    for p in sorted((src / "test" / cls).glob("*.png"))[:TEST_CAP]:
        (tmp / "test" / cls / p.name).symlink_to(p.resolve())
        mask = src / "ground_truth" / cls / p.name
        if mask.exists():
            (tmp / "ground_truth" / cls / p.name).symlink_to(mask.resolve())

print(f"Training on {len(sampled)} / {len(good_patches)} good images")
for cls in TEST_CLASSES:
    n = len(list((tmp / "test" / cls).glob("*.png")))
    print(f"  test/{cls}: {n}")

# Datamodule
datamodule = MVTecAD(
    root=str(tmp.parent),
    category=CATEGORY,
    train_batch_size=32,
    eval_batch_size=32,
)

# Model
model = Padim(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    pre_trained=True
)

# Train + Evaluate
engine = Engine()
engine.fit(model=model, datamodule=datamodule)

test_results = engine.test(model=model, datamodule=datamodule)
print(test_results)

shutil.rmtree(tmp.parent, ignore_errors=True)

import random
import shutil
import tempfile
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.models import ReverseDistillation
from anomalib.engine import Engine

# Config
DATASET_ROOT = "./datasets/custom_ivc_dataset/by_category"
CATEGORY     = "jeans_mvtec"
RANDOM_SEED  = 42
TRAIN_SAMPLE = 5000

TEST_CLASSES = [
    "abrasion", "brand_marker", "hole_tear",
    "missing_functional_elements", "other_defects",
    "seam_defect", "stain_dirt",
]
TEST_CAP = 300

# Sanity-check settings (swap for proper training)
MAX_EPOCHS = 50          # bump to ~100 for real training
BACKBONE   = "resnet18" # use "wide_resnet50_2" for full run

# Build temp dataset with symlinks
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

# val: good-only
remaining = [p for p in good_patches if p not in set(sampled)]
val_dir = tmp / "val" / "good"
val_dir.mkdir(parents=True)
for p in random.sample(remaining, min(200, len(remaining))):
    (val_dir / p.name).symlink_to(p.resolve())

# test: capped good + specific defect classes
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
    train_batch_size=16,
    eval_batch_size=4,
)

# Model
# ReverseDistillation trains an encoder-decoder pair: encoder mirrors a pretrained
# backbone, decoder tries to reconstruct only normal features -> anomalies cause
# high reconstruction error. No embedding bank -> memory scales with batch size.
model = ReverseDistillation(
    backbone=BACKBONE,
    layers=["layer1", "layer2", "layer3"],
    pre_trained=True,
)

# Train + Evaluate
engine = Engine(max_epochs=MAX_EPOCHS)
engine.fit(model=model, datamodule=datamodule)

test_results = engine.test(model=model, datamodule=datamodule)
print(test_results)

shutil.rmtree(tmp.parent, ignore_errors=True)

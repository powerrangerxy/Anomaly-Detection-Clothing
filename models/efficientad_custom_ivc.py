import random
import shutil
import tempfile
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.models import EfficientAd
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
MAX_EPOCHS = 5    # bump to ~200 for real training (EfficientAD needs more epochs)

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
    train_batch_size=1,   # EfficientAD internally uses a fixed batch of 1 teacher + student
    eval_batch_size=4,
)

# Model
# EfficientAD trains a student-teacher pair + a separate autoencoder.
# Teacher is pretrained on ImageNette (auto-downloaded on first run to ./datasets/imagenette).
# No embedding bank -> memory scales with batch size only.
model = EfficientAd(
    imagenet_dir="./datasets/imagenette",
    model_size="small",   # "small" or "medium" - small is faster, medium is more accurate
)

# Train + Evaluate
engine = Engine(max_epochs=MAX_EPOCHS)
engine.fit(model=model, datamodule=datamodule)

test_results = engine.test(model=model, datamodule=datamodule)
print(test_results)

shutil.rmtree(tmp.parent, ignore_errors=True)

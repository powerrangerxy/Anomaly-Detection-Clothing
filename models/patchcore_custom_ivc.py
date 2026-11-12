import random
import shutil
import tempfile
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

# Subsample train/good to a manageable number
# PatchCore collects all training embeddings before coreset selection.
# wide_resnet50_2 produces 1536-dim embeddings -> 2000 patches x 28x28 = 9GB GPU.
# resnet18 produces 384-dim embeddings -> 2000 patches x 28x28 = 2.3GB GPU -> fits fine.

DATASET_ROOT = "./datasets/custom_ivc_dataset/by_category"
CATEGORY     = "jeans_mvtec"
TRAIN_SAMPLE = 2000  # resnet18 raw bank ~4.7GB -> safe with ~8GB used by model during vstack
RANDOM_SEED  = 42

src = Path(DATASET_ROOT) / CATEGORY
tmp = Path(tempfile.mkdtemp()) / CATEGORY
tmp.mkdir(parents=True)

(tmp / "test").symlink_to((src / "test").resolve())
(tmp / "ground_truth").symlink_to((src / "ground_truth").resolve())

good_patches = sorted((src / "train" / "good").glob("*.png"))
random.seed(RANDOM_SEED)
sampled = random.sample(good_patches, min(TRAIN_SAMPLE, len(good_patches)))

train_good = tmp / "train" / "good"
train_good.mkdir(parents=True)
for p in sampled:
    (train_good / p.name).symlink_to(p.resolve())

print(f"Training on {len(sampled)} / {len(good_patches)} good patches")

# Validation split
# Note: MVTecAD datamodule does NOT support anomalous val images (it only assigns
# masks to test-split samples). Val is good-only; the adaptive threshold warning
# is a known Anomalib limitation when no anomalies are in val.
VAL_GOOD_SAMPLE = 200

remaining_good = [p for p in good_patches if p not in set(sampled)]
val_good_imgs = random.sample(remaining_good, min(VAL_GOOD_SAMPLE, len(remaining_good)))
val_good_dir = tmp / "val" / "good"
val_good_dir.mkdir(parents=True)
for p in val_good_imgs:
    (val_good_dir / p.name).symlink_to(p.resolve())

print(f"Validation: {len(val_good_imgs)} good images (anomalous val not supported by MVTecAD datamodule)")

# Datamodule
datamodule = MVTecAD(
    root=str(tmp.parent),
    category=CATEGORY,
    train_batch_size=32,
    eval_batch_size=8,
)

# Model
model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.15,  # slightly richer bank than 0.1
)

# Train + Evaluate
engine = Engine()
engine.fit(model=model, datamodule=datamodule)

test_results = engine.test(datamodule=datamodule, model=model)
print(test_results)

shutil.rmtree(tmp.parent, ignore_errors=True)
 
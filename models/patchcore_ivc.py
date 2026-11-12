import random
import shutil
import tempfile
from pathlib import Path

from anomalib.data import Folder
from anomalib.data.utils import ValSplitMode
from anomalib.engine import Engine
from anomalib.models import Patchcore

# dataset path
SRC = Path("./datasets/custom_ivc_dataset/by_category/jeans_mvtec")

# patchcore stores ALL patch embeddings before coreset selection
# resnet18 produces 384-dim embeddings, 28x28 patches per image
# 3000 images -> ~3.6 GB raw bank, ~7 GB peak during vstack -> fits in 16 GB
# dont go above 4000 or it will OOM
TRAIN_SAMPLE = 2000

# same issue as padim - 12k test images causes OOM during pixel metric argsort
TEST_CAP = 300

SEED = 42
random.seed(SEED)

defect_classes = ["abrasion", "brand_marker", "hole_tear",
                  "missing_functional_elements", "other_defects", "seam_defect", "stain_dirt"]

# build temp folder with sampled images
tmp = Path(tempfile.mkdtemp())

train_dir = tmp / "normal";      train_dir.mkdir()
test_good = tmp / "normal_test"; test_good.mkdir()
abnormal  = tmp / "abnormal";    abnormal.mkdir()
masks_dir = tmp / "masks";       masks_dir.mkdir()

good_images = sorted((SRC / "train" / "good").glob("*.png"))
for p in random.sample(good_images, min(TRAIN_SAMPLE, len(good_images))):
    (train_dir / p.name).symlink_to(p.resolve())

for p in sorted((SRC / "test" / "good").glob("*.png"))[:TEST_CAP]:
    (test_good / p.name).symlink_to(p.resolve())

for cls in defect_classes:
    for p in sorted((SRC / "test" / cls).glob("*.png"))[:TEST_CAP]:
        (abnormal / f"{cls}__{p.name}").symlink_to(p.resolve())
        mask = SRC / "ground_truth" / cls / p.name
        if mask.exists():
            (masks_dir / f"{cls}__{p.name}").symlink_to(mask.resolve())

print(f"training on {len(list(train_dir.glob('*.png')))} good images")
print(f"testing on {len(list(abnormal.glob('*.png')))} defective + {len(list(test_good.glob('*.png')))} good")

datamodule = Folder(
    name="jeans",
    root=tmp,
    normal_dir="normal",
    abnormal_dir="abnormal",
    normal_test_dir="normal_test",
    mask_dir="masks",
    val_split_mode=ValSplitMode.FROM_TRAIN,
    val_split_ratio=0.1,
    train_batch_size=32,
    eval_batch_size=4,
)

model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1,  # keeps 10% of patches in final memory bank
)

engine = Engine()
engine.fit(model=model, datamodule=datamodule)

results = engine.test(model=model, datamodule=datamodule)
print(results)

shutil.rmtree(tmp)

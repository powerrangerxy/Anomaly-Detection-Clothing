import random
import shutil
import tempfile
from pathlib import Path

from anomalib.data import Folder
from anomalib.data.utils import ValSplitMode
from anomalib.engine import Engine
from anomalib.models import Padim

# dataset path
SRC = Path("./datasets/custom_ivc_dataset/by_category/jeans_mvtec")

# how many training images to use - using all 21k causes GPU OOM during fitting
TRAIN_SAMPLE = 3000

# max images per defect class in test - using all 12k causes OOM during metric computation
TEST_CAP = 300

SEED = 42
random.seed(SEED)

# the "defective" folder was removed because it was a duplicate of all classes below
defect_classes = ["abrasion", "brand_marker", "hole_tear",
                  "missing_functional_elements", "other_defects", "seam_defect", "stain_dirt"]

# create a temp folder to put the sampled images in (symlinks, not copies)
tmp = Path(tempfile.mkdtemp())

train_dir = tmp / "normal";      train_dir.mkdir()
test_good = tmp / "normal_test"; test_good.mkdir()
abnormal  = tmp / "abnormal";    abnormal.mkdir()
masks_dir = tmp / "masks";       masks_dir.mkdir()

# sample from the 21k good training images
good_images = sorted((SRC / "train" / "good").glob("*.png"))
for p in random.sample(good_images, min(TRAIN_SAMPLE, len(good_images))):
    (train_dir / p.name).symlink_to(p.resolve())

# take a subset of test good images
for p in sorted((SRC / "test" / "good").glob("*.png"))[:TEST_CAP]:
    (test_good / p.name).symlink_to(p.resolve())

# take a subset of each defect class + corresponding masks
# prefix filename with class name to avoid collisions (all classes use patch_00000.png etc.)
for cls in defect_classes:
    for p in sorted((SRC / "test" / cls).glob("*.png"))[:TEST_CAP]:
        (abnormal / f"{cls}__{p.name}").symlink_to(p.resolve())
        mask = SRC / "ground_truth" / cls / p.name
        if mask.exists():
            (masks_dir / f"{cls}__{p.name}").symlink_to(mask.resolve())

print(f"training on {len(list(train_dir.glob('*.png')))} good images")
print(f"testing on {len(list(abnormal.glob('*.png')))} defective + {len(list(test_good.glob('*.png')))} good")

# load dataset
datamodule = Folder(
    name="jeans",
    root=tmp,
    normal_dir="normal",
    abnormal_dir="abnormal",
    normal_test_dir="normal_test",
    mask_dir="masks",
    val_split_mode=ValSplitMode.FROM_TRAIN,  # take 10% of train as validation
    val_split_ratio=0.1,
    train_batch_size=32,
    eval_batch_size=32,
)

model = Padim(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    pre_trained=True
)

engine = Engine()
engine.fit(model=model, datamodule=datamodule)

results = engine.test(model=model, datamodule=datamodule)
print(results)

# cleanup temp folder
shutil.rmtree(tmp)

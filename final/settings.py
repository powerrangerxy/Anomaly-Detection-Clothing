"""Fixed settings of the final experiments.

The values are written to split_<x>/config.json for every run.
"""

from dataclasses import dataclass
from pathlib import Path

# patching
EROSION_PX = 20             # defect masks are shrunk by this many pixels before checking defect coverage
PATCH_SIZE = 336
GARMENT_COVERAGE = 0.30     # a patch must lie at least 30% on the garment
DEFECT_THRESHOLD = 0.02     # a defect patch must contain at least 2% (eroded) defect pixels
GOOD_STRIDE = 336
EXCLUDE_CATEGORIES = {"brand_marker"}

# splits
HOLDOUT_TEST_FRAC = 0.20
N_FOLDS = 5
RANDOM_SEED = 42
TEST_CAP = 300              # per class per test fold (good and each defect class)
TRAIN_N = 5000              # good patches used for training per fold
VAL_SAMPLE = 200            # held-out good patches of the train garments, used to standardise scores

# models
BACKBONE = "resnet18"
MAX_EPOCHS = 50

# evaluation
TARGET_TPR = 0.95
TARGET_FPR = 0.10           # second operating point: 10% of good patches flagged
N_BOOT = 1000
PIX_BINS = 4096
# defect area as a fraction of the patch; a patch needs >= 2% eroded defect, so raw areas start at about 6%
SIZE_BINS = [(0.00, 0.20, "small (<20%)"),
             (0.20, 0.40, "medium (20-40%)"),
             (0.40, 1.01, "large (>=40%)")]
N_EXAMPLES = 6              # lowest / highest scored example strips per class

DATA_ROOT = Path("202603_ReUseTex_Setup1/01_data")
SEG_ROOT = Path("202603_ReUseTex_Setup1/02_segmentation")
MASK_CACHE_ROOT = Path("datasets/custom_ivc_dataset/by_category")  # older SAM3 garment masks (jeans only)

TEST_CLASSES = ["abrasion", "hole_tear", "missing_functional_elements",
                "other_defects", "seam_defect", "stain_dirt"]
MODEL_LABELS = {"padim": "PaDiM", "reverse_distillation": "RD"}


@dataclass(frozen=True)
class Paths:
    """Input filter and output folders of one run, e.g. jeans with the 5-fold split."""
    field: str
    value: str
    split: str

    @property
    def pool(self) -> Path:
        return Path("final") / self.value / "pool"

    @property
    def split_dir(self) -> Path:
        return Path("final") / self.value / f"split_{self.split}"

    @property
    def results(self) -> Path:
        return self.split_dir / "results"

    @property
    def plots(self) -> Path:
        return self.split_dir / "plots"


def to_dict(paths: Paths, pred_keys: list[str]) -> dict:
    """Settings in the same form as the existing config.json files."""
    return {
        "EROSION_PX": EROSION_PX, "PATCH_SIZE": PATCH_SIZE, "GARMENT_COVERAGE": GARMENT_COVERAGE,
        "DEFECT_THRESHOLD": DEFECT_THRESHOLD, "GOOD_STRIDE": GOOD_STRIDE,
        "EXCLUDE_CATEGORIES": sorted(EXCLUDE_CATEGORIES), "HOLDOUT_TEST_FRAC": HOLDOUT_TEST_FRAC,
        "N_FOLDS": N_FOLDS, "RANDOM_SEED": RANDOM_SEED, "TEST_CAP": TEST_CAP, "TRAIN_N": TRAIN_N,
        "VAL_SAMPLE": VAL_SAMPLE, "BACKBONE": BACKBONE, "MAX_EPOCHS": MAX_EPOCHS,
        "TARGET_TPR": TARGET_TPR, "TARGET_FPR": TARGET_FPR, "N_BOOT": N_BOOT, "PIX_BINS": PIX_BINS,
        "SIZE_BINS": [list(b) for b in SIZE_BINS], "N_EXAMPLES": N_EXAMPLES,
        "TEST_CLASSES": TEST_CLASSES, "MODEL_LABELS": MODEL_LABELS,
        "FILTER_FIELD": paths.field, "FILTER_VALUE": paths.value, "PRED_KEYS": pred_keys,
    }

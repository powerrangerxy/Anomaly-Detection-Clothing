"""Stage 2: garment-level train/test splits.

Garments, not patches, are assigned to the folds. So a garment is never in training and test at once.
80_20 gives one split with 20% test garments, 5fold makes every garment a test garment exactly once.
"""

import random

import numpy as np

from settings import HOLDOUT_TEST_FRAC, N_FOLDS, RANDOM_SEED, TEST_CAP, TEST_CLASSES, TRAIN_N, VAL_SAMPLE


def make_folds(index: list[dict], split: str) -> list[dict]:
    rng = random.Random(RANDOM_SEED)
    garments = sorted({r["iid"] for r in index})
    rng.shuffle(garments)
    if split == "80_20":
        n_test = int(round(len(garments) * HOLDOUT_TEST_FRAC))
        groups = [garments[:n_test]]
    else:
        groups = [list(g) for g in np.array_split(garments, N_FOLDS)]

    folds = []
    for f, test_garments in enumerate(groups):
        test_garments = set(int(g) for g in test_garments)
        rng_f = random.Random(RANDOM_SEED + f)

        train_good = [r for r in index if r["kind"] == "good" and r["iid"] not in test_garments]
        rng_f.shuffle(train_good)
        train = train_good[:TRAIN_N]
        val = train_good[TRAIN_N:TRAIN_N + VAL_SAMPLE]
        if len(val) < VAL_SAMPLE:  # small garment type: keep the validation set, use fewer training patches
            val, train = train_good[:VAL_SAMPLE], train_good[VAL_SAMPLE:VAL_SAMPLE + TRAIN_N]

        test = {}
        for cls in ["good"] + TEST_CLASSES:
            candidates = [r for r in index if r["cls"] == cls and r["iid"] in test_garments]
            rng_f.shuffle(candidates)
            if candidates:
                test[cls] = sorted(candidates[:TEST_CAP], key=lambda r: r["name"])

        folds.append({"fold": f, "test_garments": sorted(test_garments),
                      "train": train, "val": val, "test": test})
    return folds


def fold_summary(folds: list[dict]) -> list[dict]:
    """Patch counts and test garment ids per fold (written to folds.json)."""
    return [{"fold": fd["fold"], "n_test_garments": len(fd["test_garments"]),
             "train_good": len(fd["train"]), "val_good": len(fd["val"]),
             "test": {c: len(v) for c, v in fd["test"].items()},
             "test_garments": fd["test_garments"]} for fd in folds]

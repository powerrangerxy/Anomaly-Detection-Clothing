"""Zero-shot: apply the jeans models to another garment type without retraining.

We use the jeans models of the 80/20 split and test them on the same test patches as the
5-fold results of the target type, so the numbers can be compared directly.

    python final/run_zero_shot.py --sub t_shirt
    python final/run_zero_shot.py --sub knit_wool --field garment_main_category

Results go to final/<sub>/zero_shot_jeans/.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

import patching
import splits
import training
from metrics import evaluate, write_csv
from settings import RANDOM_SEED, Paths
from anomalib.engine import Engine
from training import Padim, ReverseDistillation

CKPT_DIR = Path("final/jeans/split_80_20/checkpoints/fold0")
MODEL_CLASSES = {"padim": Padim, "reverse_distillation": ReverseDistillation}


def jeans_checkpoint(model_name):
    ckpts = sorted((CKPT_DIR / model_name).rglob("model.ckpt"))
    if len(ckpts) != 1:
        raise FileNotFoundError(f"expected one {model_name} checkpoint in {CKPT_DIR}, found {len(ckpts)}")
    return ckpts[0]


def predict(model_name, test_set, pool):
    # anomalib needs a train folder, but nothing is trained here
    data = {"train": test_set["good"][:1], "val": [], "test": test_set}
    dm, tmp, _ = training.make_datamodule(data, training.MODELS[model_name], pool)
    try:
        model = MODEL_CLASSES[model_name].load_from_checkpoint(str(jeans_checkpoint(model_name)))
        # raw scores, without anomalib's normalisation
        model.post_processor.enable_normalization = False
        model.post_processor.enable_thresholding = False
        engine = Engine(logger=False)
        return training.collect_predictions(engine.predict(model=model, datamodule=dm, return_predictions=True))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True, help="target garment type, e.g. t_shirt")
    ap.add_argument("--field", default="garment_sub_category")
    args = ap.parse_args()

    paths = Paths(field=args.field, value=args.sub, split="5fold")
    out_dir = Path("final") / args.sub / "zero_shot_jeans"
    index = patching.load_pool(paths)
    garment_of = {r["name"]: r["iid"] for r in index}
    folds = splits.make_folds(index, "5fold")

    # Every garment is in exactly one test fold, so all test sets together cover the garment type once.
    # We remember the fold of each patch to also report per-fold numbers.
    test_set, fold_of = {}, {}
    for fd in folds:
        for cls, rows in fd["test"].items():
            test_set.setdefault(cls, []).extend(rows)
            fold_of.update({r["name"]: fd["fold"] for r in rows})

    rng = np.random.default_rng(RANDOM_SEED)
    overall, per_defect = [], []
    for model_name in MODEL_CLASSES:
        print(f"\n{model_name}: jeans -> {args.sub}")
        pred = predict(model_name, test_set, paths.pool)
        # One model scores all patches, so the scores need no per-fold standardisation.
        pred["raw_scores"] = pred["scores"]
        pred["fold"] = np.array([fold_of[n] for n in pred["names"]])
        reported = [{} for _ in folds]  # anomalib's own metrics are not used here

        o, pdf, _, rows, _, _, _ = evaluate(model_name, pred, reported, rng)
        for r in pdf:
            r["n_garments"] = len({garment_of[n] for n, c in zip(pred["names"], pred["classes"])
                                   if c == r["defect"]})
        overall.extend(o)
        per_defect.extend(pdf)
        write_csv(out_dir / f"predictions_{model_name}.csv", rows)
        torch.cuda.empty_cache()

    write_csv(out_dir / "metrics_overall.csv", overall)
    write_csv(out_dir / "per_defect.csv", per_defect)

    print()
    for o in overall:
        if o["fold"] in ("pooled", "mean", "std"):
            print(f"{o['model']:<22} {o['fold']:<7} image AUROC {o['image_AUROC']}  pixel AUROC {o['pixel_AUROC']}")


if __name__ == "__main__":
    main()

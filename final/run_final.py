"""PaDiM and Reverse Distillation on one garment type with a garment-level split.

Run from the project root:
    python final/run_final.py --sub t_shirt --split 5fold
    python final/run_final.py --sub t_shirt --split 80_20
    python final/run_final.py --sub knit_wool --field garment_main_category --split 5fold

Stages:
  1. patching.py  cut all images into patches once (shared by both splits)
  2. splits.py    assign garments to folds
  3. training.py  train and predict per fold (cached in split_<x>/cache/)
  4. metrics.py   evaluate the pooled predictions, plots.py draws the figures

All outputs go to final/<sub>/, see final/README.md.
"""

import argparse
import json
import shutil
import traceback

import numpy as np
import torch

import patching
import plots
import splits
import training
from metrics import evaluate, write_csv
from settings import RANDOM_SEED, Paths, to_dict


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True, help="garment type to filter on, e.g. t_shirt")
    ap.add_argument("--field", default="garment_sub_category")
    ap.add_argument("--split", choices=["80_20", "5fold"], required=True)
    ap.add_argument("--rebuild", action="store_true", help="delete and rebuild the patch pool")
    return ap.parse_args()


def prepare_pool(paths: Paths, rebuild: bool) -> dict:
    # pool_stats.json is in the repository, but the patches and index.csv are not
    if rebuild or not (paths.pool / "index.csv").exists():
        shutil.rmtree(paths.pool, ignore_errors=True)
        return patching.build_pool(paths)
    print(f"[POOL] exists at {paths.pool}, skipping (use --rebuild).")
    return json.load(open(paths.pool / "pool_stats.json"))


def write_split_info(paths: Paths, folds: list[dict], pool_stats: dict):
    paths.split_dir.mkdir(parents=True, exist_ok=True)
    summary = splits.fold_summary(folds)
    with open(paths.split_dir / "folds.json", "w") as f:
        json.dump(summary, f, indent=2)
    for s in summary:
        print(f"  fold {s['fold']}: {s['n_test_garments']} test garments  train={s['train_good']}  "
              f"val={s['val_good']}  test={s['test']}")

    config = to_dict(paths, training.PRED_KEYS)
    config.update({"split": paths.split, "n_folds": len(folds), "pool_stats": pool_stats})
    with open(paths.split_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def predict_all_folds(model_name: str, cfg: dict, folds: list[dict], paths: Paths):
    """Load cached predictions or train the model, for every fold. Returns None if a fold failed."""
    preds, reported = [], []
    for fd in folds:
        print(f"\n{'=' * 60}\n  {model_name}  |  {paths.value}  split {paths.split}  fold {fd['fold']}\n{'=' * 60}")
        cached = training.load_cache(paths, model_name, fd["fold"])
        if cached:
            rep, pred = cached
            print(f"  loaded cached predictions {training.cache_path(paths, model_name, fd['fold'])}")
        else:
            torch.manual_seed(RANDOM_SEED + fd["fold"])
            try:
                rep, pred = training.train_and_predict(model_name, cfg, fd, paths,
                                                       keep_ckpt=(paths.split == "80_20"))
            except Exception as e:
                print(f"  [ERROR] {e}")
                traceback.print_exc()
                continue
            training.save_cache(paths, model_name, fd["fold"], rep, pred)
        print(f"  fold {fd['fold']} anomalib: {rep}  norm: {pred['norm']}")
        preds.append(pred)
        reported.append(rep)
        torch.cuda.empty_cache()

    if len(preds) != len(folds):
        print(f"  [ERROR] {model_name}: only {len(preds)}/{len(folds)} folds finished - skipping evaluation")
        return None
    return training.concat_preds(preds), reported


def print_summary(overall: list[dict], per_defect: list[dict]):
    print("\n" + "=" * 90)
    for o in overall:
        print(f"{o['model']:<22} {o['fold']:<7} imgAUROC {o.get('image_AUROC')}  FPR@95 {o.get('image_FPR@95TPR')}  "
              f"pixAUROC {o.get('pixel_AUROC')}  pixFPR@95 {o.get('pixel_FPR@95TPR')}")
    for r in per_defect:
        print(f"  {r['model']:<22} {r['defect']:<28} n={r['n']:<4} ({r['n_garments']} garments) "
              f"AUROC {r['image_AUROC']:.3f} [{r['image_AUROC_lo']:.3f},{r['image_AUROC_hi']:.3f}]  "
              f"recall {r['recall@95TPR']:.3f}  pixAUROC {r['pixel_AUROC']:.3f}  area {r['median_area']:.3f}")


def main():
    args = parse_args()
    paths = Paths(field=args.field, value=args.sub, split=args.split)

    pool_stats = prepare_pool(paths, args.rebuild)
    index = patching.load_pool(paths)
    folds = splits.make_folds(index, args.split)
    write_split_info(paths, folds, pool_stats)

    garment_of = {r["name"]: r["iid"] for r in index}
    rng = np.random.default_rng(RANDOM_SEED)
    preds, overall, per_defect, per_size = {}, [], [], []
    for model_name, cfg in training.MODELS.items():
        result = predict_all_folds(model_name, cfg, folds, paths)
        if result is None:
            continue
        pred, reported = result

        o, pdf, ps, rows, pix_thr, area, thr95 = evaluate(model_name, pred, reported, rng)
        for r in pdf:
            r["n_garments"] = len({garment_of[n] for n, c in zip(pred["names"], pred["classes"])
                                   if c == r["defect"]})
        overall.extend(o)
        per_defect.extend(pdf)
        per_size.extend(ps)

        write_csv(paths.results / f"predictions_{model_name}.csv", rows)
        plots.save_examples(model_name, pred, area, pix_thr, thr95, paths)
        pred.pop("maps")  # free memory, the plots only need scores and classes
        preds[model_name] = pred
        write_csv(paths.results / "metrics_overall.csv", overall)
        write_csv(paths.results / "per_defect.csv", per_defect)
        write_csv(paths.results / "per_defect_size.csv", per_size)

    if preds:
        plots.plot_all(preds, overall, per_defect, per_size, f"{paths.value} ({args.split})", paths)
    print_summary(overall, per_defect)


if __name__ == "__main__":
    main()

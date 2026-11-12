# final

PaDiM and Reverse Distillation on every garment type, with 336 px patches and 20 px erosion.
All results in the thesis come from this folder.

```bash
python final/run_final.py --sub t_shirt --split 5fold
python final/run_zero_shot.py --sub t_shirt
```

For `knit_wool` and `shirt_blouse` add `--field garment_main_category`.
Running `run_final.py` a second time loads the saved predictions and only repeats the evaluation.
Delete `split_<x>/cache/` to train again, and use `--rebuild` to cut the patches again.

## How it works

1. `patching.py` cuts all images of one garment type into patches (defect patches centred on the defects, good patches from a grid).
2. `splits.py` splits by garment, not by patch, so a garment is never in training and test at the same time.
   With `5fold` every garment is a test garment once. `80_20` is a single split.
3. `training.py` trains the models per fold. The scores of each fold are standardised with 200 good patches
   of the training garments, so the folds can be pooled.
4. `metrics.py` and `plots.py` compute the metrics and draw the plots. All settings are in `settings.py`.

`run_zero_shot.py` takes the jeans models of the 80/20 split and tests them on the 5-fold test patches of another type.

We use the raw anomaly scores and not anomalib's normalised ones, because anomalib uses the test set
as validation set by default. FPR@95%TPR is the share of good patches flagged when 95% of the defects are found,
TPR@10%FPR the share of defects found when 10% of the good patches are flagged.

## Output per garment type

In the repository: `split_<x>/config.json` (settings), `folds.json` (garments per fold),
`results/*.csv` (metrics overall, per defect, per defect size, and one row per test patch) and `plots/`.
The patches, cached predictions, example images and checkpoints are created when the script runs.

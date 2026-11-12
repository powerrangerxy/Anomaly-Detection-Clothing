"""Image- and pixel-level metrics, per fold, pooled over folds, per defect type and per defect size."""

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from settings import N_BOOT, PIX_BINS, SIZE_BINS, TARGET_FPR, TARGET_TPR, TEST_CLASSES

METRIC_KEYS = ["image_AUROC", "image_AUPR", "image_F1max", "image_FPR@95TPR", "image_TPR@10FPR",
               "pixel_AUROC", "pixel_F1max", "pixel_FPR@95TPR",
               "anomalib_image_AUROC", "anomalib_image_F1", "anomalib_pixel_AUROC", "anomalib_pixel_F1"]


def fpr_at_tpr(y, s, target=TARGET_TPR):
    """False positive rate (and threshold) at the first point where the TPR reaches the target."""
    fpr, tpr, thr = roc_curve(y, s)
    i = int(np.argmax(tpr >= target))
    return float(fpr[i]), float(thr[i])


def f1_max(y, s):
    p, r, _ = precision_recall_curve(y, s)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(f1.max())


def bootstrap_auroc_ci(pos, neg, rng):
    """95% confidence interval of the AUROC by resampling defect and good scores."""
    vals = []
    for _ in range(N_BOOT):
        bp = rng.choice(pos, len(pos))
        bn = rng.choice(neg, len(neg))
        vals.append(roc_auc_score(np.r_[np.ones(len(bp)), np.zeros(len(bn))], np.r_[bp, bn]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def wilson_ci(k, n, z=1.96):
    """95% Wilson interval for a recall of k out of n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0, c - h)), float(min(1, c + h))


class PixelHist:
    """Histograms of defect and good pixel scores per patch, on one shared bin grid.

    Summing the histograms of any subset of patches gives its pixel ROC curve,
    without keeping all pixel scores in memory.
    """

    def __init__(self, maps, gt):
        lo, hi = float(np.min(maps)), float(np.max(maps))
        self.edges = np.linspace(lo, hi + 1e-6, PIX_BINS + 1)
        n = len(maps)
        self.pos = np.zeros((n, PIX_BINS), np.int64)
        self.neg = np.zeros((n, PIX_BINS), np.int64)
        for i in range(n):
            m = maps[i].astype(np.float32)
            g = gt[i]
            self.pos[i] = np.histogram(m[g], self.edges)[0]
            self.neg[i] = np.histogram(m[~g], self.edges)[0]

    def metrics(self, idx):
        # cumulative counts from the highest bin down = TP / FP for each threshold
        pos = self.pos[idx].sum(0)[::-1].cumsum()
        neg = self.neg[idx].sum(0)[::-1].cumsum()
        P, N = pos[-1], neg[-1]
        if P == 0 or N == 0:
            return {"AUROC": np.nan, "F1max": np.nan, "FPR@95TPR": np.nan, "thr_f1": np.nan}
        tpr = np.r_[0, pos / P]
        fpr = np.r_[0, neg / N]
        auroc = float(np.trapz(tpr, fpr))
        prec = pos / np.clip(pos + neg, 1, None)
        rec = pos / P
        f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
        k = int(np.argmax(f1))
        thr_f1 = float(self.edges[::-1][1:][k])
        fpr95 = float(fpr[int(np.argmax(tpr >= TARGET_TPR))])
        return {"AUROC": auroc, "F1max": float(f1[k]), "FPR@95TPR": fpr95, "thr_f1": thr_f1}


def overall_metrics(classes, scores, ph: PixelHist, idx) -> dict:
    y = (classes[idx] != "good").astype(int)
    s = scores[idx]
    fpr95, thr95 = fpr_at_tpr(y, s)
    thr10 = float(np.quantile(s[y == 0], 1 - TARGET_FPR))
    pix = ph.metrics(idx)
    return {"n_good": int((y == 0).sum()), "n_defect": int(y.sum()),
            "image_AUROC": round(roc_auc_score(y, s), 4),
            "image_AUPR": round(average_precision_score(y, s), 4),
            "image_F1max": round(f1_max(y, s), 4),
            "image_FPR@95TPR": round(fpr95, 4),
            "image_TPR@10FPR": round(float((s[y == 1] > thr10).mean()), 4),
            "pixel_AUROC": round(pix["AUROC"], 4),
            "pixel_F1max": round(pix["F1max"], 4),
            "pixel_FPR@95TPR": round(pix["FPR@95TPR"], 4),
            "thr95_image": round(thr95, 5), "thr10fpr_image": round(thr10, 5)}


def recall_by_size(model_name, defect, idx, area, scores, thr95, thr10) -> list[dict]:
    rows = []
    for lo, hi, label in SIZE_BINS:
        bi = idx[(area[idx] >= lo) & (area[idx] < hi)]
        k95, k10 = int((scores[bi] >= thr95).sum()), int((scores[bi] > thr10).sum())
        rows.append({"model": model_name, "defect": defect, "size_bin": label, "n": len(bi),
                     "recall@95TPR": round(k95 / len(bi), 4) if len(bi) else None,
                     "recall@10FPR": round(k10 / len(bi), 4) if len(bi) else None})
    return rows


def evaluate(model_name: str, pred: dict, reported: list[dict], rng):
    classes, scores, folds = pred["classes"], pred["scores"], pred["fold"]
    y = (classes != "good").astype(int)
    area = pred["gt"].reshape(len(scores), -1).mean(1)
    good_idx = np.where(y == 0)[0]
    ph = PixelHist(pred["maps"], pred["gt"])

    # overall: per fold, mean and std over folds, and all folds pooled
    overall = []
    fold_ids = sorted(set(folds.tolist()))
    for f in fold_ids:
        overall.append({"model": model_name, "fold": str(f),
                        **overall_metrics(classes, scores, ph, np.where(folds == f)[0]), **reported[f]})
    pooled = {"model": model_name, "fold": "pooled",
              **overall_metrics(classes, scores, ph, np.arange(len(scores)))}
    if len(fold_ids) > 1:
        for stat, fn in [("mean", np.mean), ("std", np.std)]:
            row = {"model": model_name, "fold": stat}
            for k in METRIC_KEYS:
                vals = [o[k] for o in overall if o.get(k) is not None]
                row[k] = round(float(fn(vals)), 4) if vals else None
            overall.append(row)
    else:
        pooled.update(reported[0])
    overall.append(pooled)
    thr95 = pooled["thr95_image"]
    thr10 = pooled["thr10fpr_image"]

    # per defect type: each defect class against all good patches, thresholds from the pooled set
    per_defect, per_size = [], []
    for cls in TEST_CLASSES:
        ci = np.where(classes == cls)[0]
        if len(ci) == 0:
            continue
        s_pos, s_neg = scores[ci], scores[good_idx]
        auroc = roc_auc_score(np.r_[np.ones(len(ci)), np.zeros(len(good_idx))], np.r_[s_pos, s_neg])
        lo, hi = bootstrap_auroc_ci(s_pos, s_neg, rng) if len(ci) >= 2 else (np.nan, np.nan)
        k95 = int((s_pos >= thr95).sum())
        r95_lo, r95_hi = wilson_ci(k95, len(ci))
        k10 = int((s_pos > thr10).sum())
        r10_lo, r10_hi = wilson_ci(k10, len(ci))
        pix = ph.metrics(np.r_[ci, good_idx])
        per_defect.append({
            "model": model_name, "defect": cls, "n": len(ci),
            "n_garments": None,  # filled in by run_final.py
            "image_AUROC": round(auroc, 4), "image_AUROC_lo": round(lo, 4), "image_AUROC_hi": round(hi, 4),
            "recall@95TPR": round(k95 / len(ci), 4), "recall_lo": round(r95_lo, 4), "recall_hi": round(r95_hi, 4),
            "recall@10FPR": round(k10 / len(ci), 4), "recall10_lo": round(r10_lo, 4), "recall10_hi": round(r10_hi, 4),
            "pixel_AUROC": round(pix["AUROC"], 4),
            "median_area": round(float(np.median(area[ci])), 4),
        })
        per_size += recall_by_size(model_name, cls, ci, area, scores, thr95, thr10)
    per_size += recall_by_size(model_name, "all", np.where(y == 1)[0], area, scores, thr95, thr10)

    pred_rows = [{"fold": int(f), "name": n, "class": c, "raw_score": round(float(r), 6),
                  "score": round(float(s), 5), "defect_area": round(float(a), 5),
                  "detected@95TPR": int(s >= thr95), "detected@10FPR": int(s > thr10)}
                 for f, n, c, r, s, a in zip(folds, pred["names"], classes, pred["raw_scores"], scores, area)]
    pix_thr = ph.metrics(np.arange(len(scores)))["thr_f1"]
    return overall, per_defect, per_size, pred_rows, pix_thr, area, thr95


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

"""Result plots and example strips (image | ground truth | anomaly map | prediction)."""

import csv
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from settings import MODEL_LABELS, N_EXAMPLES, SIZE_BINS, TARGET_FPR, TARGET_TPR, TEST_CLASSES, Paths  # noqa: E402

COLORS = {"padim": "#4C72B0", "reverse_distillation": "#DD8452"}
BAR_WIDTH = 0.38


def save_fig(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def example_strip(orig, gt, amap, vmin, vmax, pix_thr):
    W, H = orig.size
    gt = Image.fromarray((gt * 255).astype(np.uint8)).resize((W, H), Image.NEAREST).convert("RGB")
    norm = np.clip((amap - vmin) / (vmax - vmin + 1e-8), 0, 1)
    heat = Image.fromarray((cm.jet(norm)[:, :, :3] * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    heat = Image.blend(orig, heat, 0.5)
    pred_mask = np.array(Image.fromarray((amap > pix_thr).astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
    overlay = np.array(orig).copy()
    overlay[pred_mask] = (0.4 * overlay[pred_mask] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)
    strip = Image.new("RGB", (W * 4, H))
    for j, panel in enumerate([orig, gt, heat, Image.fromarray(overlay)]):
        strip.paste(panel, (W * j, 0))
    return strip


def save_examples(model_name: str, pred: dict, area, pix_thr: float, img_thr: float, paths: Paths):
    """Strips for the lowest and highest scored patches of each class.

    The heat map uses one colour scale for the whole test set, so patches can be compared.
    """
    out_base = paths.split_dir / "examples" / model_name
    shutil.rmtree(out_base, ignore_errors=True)
    maps, scores, classes = pred["maps"], pred["scores"], pred["classes"]
    sample = maps[:: max(1, len(maps) // 300)].astype(np.float32)
    vmin, vmax = float(np.percentile(sample, 1)), float(np.percentile(sample, 99.9))
    for cls in ["good"] + TEST_CLASSES:
        ci = np.where(classes == cls)[0]
        if len(ci) == 0:
            continue
        order = ci[np.argsort(scores[ci])]
        picks = [("lowest", i) for i in order[:N_EXAMPLES]] + [("highest", i) for i in order[::-1][:N_EXAMPLES]]
        for tag, i in picks:
            orig = Image.open(paths.pool / cls / f"{pred['names'][i]}.png").convert("RGB")
            strip = example_strip(orig, pred["gt"][i], maps[i].astype(np.float32), vmin, vmax, pix_thr)
            det = "det" if scores[i] >= img_thr else "miss"
            name = f"{tag}_{pred['names'][i]}_s{scores[i]:.3f}_a{area[i]*100:.1f}_{det}.png"
            dst = out_base / cls / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            strip.save(dst)
    print(f"  examples -> {out_base}")


def plot_roc(preds, pooled, title, out):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for m, pred in preds.items():
        y = (pred["classes"] != "good").astype(int)
        fpr, tpr, _ = roc_curve(y, pred["scores"])
        o = pooled[m]
        ax.plot(fpr, tpr, color=COLORS[m], label=f"{MODEL_LABELS[m]} (AUROC {o['image_AUROC']:.3f})")
        ax.scatter([o["image_FPR@95TPR"]], [TARGET_TPR], color=COLORS[m], zorder=3)
    ax.axhline(TARGET_TPR, ls="--", c="grey", lw=0.8)
    ax.plot([0, 1], [0, 1], c="lightgrey", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Image-level ROC - {title}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    save_fig(fig, out)


def plot_per_defect(preds, per_defect, defects, key, lo_key, hi_key, ylabel, title, out):
    x = np.arange(len(defects))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for j, m in enumerate(preds):
        rows = [next(r for r in per_defect if r["model"] == m and r["defect"] == d) for d in defects]
        values = np.array([r[key] for r in rows], float)
        err = None
        if lo_key:
            err = np.nan_to_num(np.array([[r[key] - r[lo_key] for r in rows],
                                          [r[hi_key] - r[key] for r in rows]], float))
        ax.bar(x + (j - 0.5) * BAR_WIDTH, values, BAR_WIDTH, yerr=err, capsize=3,
               color=COLORS[m], label=MODEL_LABELS[m])
    ns = [next(r for r in per_defect if r["defect"] == d)["n"] for d in defects]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n(n={n})" for d, n in zip(defects, ns)], fontsize=8)
    if key == "recall@95TPR":
        ax.axhline(TARGET_TPR, ls="--", c="grey", lw=0.8)
    elif key == "recall@10FPR":
        ax.axhline(TARGET_FPR, ls=":", c="grey", lw=0.8)  # what a random detector would reach
    else:
        ax.axhline(0.5, ls=":", c="grey", lw=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"{ylabel} per defect - {title}")
    save_fig(fig, out)


def plot_score_distributions(preds, pooled, defects, out):
    fig, axes = plt.subplots(1, len(preds), figsize=(6 * len(preds), 4.5))
    cls_list = ["good"] + defects
    for ax, (m, pred) in zip(np.atleast_1d(axes), preds.items()):
        ax.boxplot([pred["scores"][pred["classes"] == c] for c in cls_list], showfliers=False)
        ax.set_xticks(range(1, len(cls_list) + 1))
        ax.set_xticklabels(cls_list, rotation=35, ha="right", fontsize=8)
        ax.axhline(pooled[m]["thr95_image"], ls="--", c="red", lw=0.8, label="95% TPR threshold")
        ax.axhline(pooled[m]["thr10fpr_image"], ls="--", c="green", lw=0.8, label="10% FPR threshold")
        ax.set_ylabel("standardised score (val median / IQR)")
        ax.set_title(f"{MODEL_LABELS[m]} - image score")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out)


def plot_recall_by_size(preds, per_size, title, out):
    labels = [b[2] for b in SIZE_BINS]
    xb = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, ylabel in [(axes[0], "recall@95TPR", "Recall @ 95% TPR threshold"),
                            (axes[1], "recall@10FPR", f"Recall @ {int(TARGET_FPR*100)}% FPR threshold")]:
        for j, m in enumerate(preds):
            rows = [next(r for r in per_size if r["model"] == m and r["defect"] == "all" and r["size_bin"] == lb)
                    for lb in labels]
            bars = ax.bar(xb + (j - 0.5) * BAR_WIDTH, [r[key] or 0 for r in rows], BAR_WIDTH,
                          color=COLORS[m], label=MODEL_LABELS[m])
            for bar, r in zip(bars, rows):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"n={r['n']}",
                        ha="center", fontsize=7)
        ax.set_xticks(xb)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Defect area (fraction of patch)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Recall by defect size - {title}")
    fig.tight_layout()
    save_fig(fig, out)


def plot_rd_loss(log_dir, title, out):
    """Training loss of Reverse Distillation, one line per fold (read from the CSV logs)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for d in sorted(log_dir.glob("fold*_reverse_distillation")):
        csvs = sorted(d.glob("version_*/metrics.csv"))
        if not csvs:
            continue
        epochs, loss = [], []
        with open(csvs[-1]) as f:
            for row in csv.DictReader(f):
                v = (row.get("train_loss_epoch") or "").strip()
                if v:
                    epochs.append(int(float(row["epoch"])))
                    loss.append(float(v))
        if epochs:
            ax.plot(epochs, loss, marker="o", ms=2, label=d.name.split("_")[0])
            plotted = True
    if plotted:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train loss")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_title(f"RD training loss - {title}")
        fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_all(preds: dict, overall: list, per_defect: list, per_size: list, title: str, paths: Paths):
    out = paths.plots
    out.mkdir(parents=True, exist_ok=True)
    pooled = {o["model"]: o for o in overall if o["fold"] == "pooled"}
    defects = [d for d in TEST_CLASSES if any(r["defect"] == d for r in per_defect)]

    plot_roc(preds, pooled, title, out / "roc_image.png")
    for key, lo_key, hi_key, fname, ylabel in [
        ("image_AUROC", "image_AUROC_lo", "image_AUROC_hi", "per_defect_image_auroc.png", "Image AUROC (vs good)"),
        ("recall@95TPR", "recall_lo", "recall_hi", "per_defect_recall.png",
         f"Recall @ global {int(TARGET_TPR*100)}% TPR threshold"),
        ("recall@10FPR", "recall10_lo", "recall10_hi", "per_defect_recall_10fpr.png",
         f"Recall @ {int(TARGET_FPR*100)}% FPR threshold"),
        ("pixel_AUROC", None, None, "per_defect_pixel_auroc.png", "Pixel AUROC"),
    ]:
        plot_per_defect(preds, per_defect, defects, key, lo_key, hi_key, ylabel, title, out / fname)
    plot_score_distributions(preds, pooled, defects, out / "score_distributions.png")
    plot_recall_by_size(preds, per_size, title, out / "recall_by_size.png")
    plot_rd_loss(paths.split_dir / "logs", title, out / "training_loss_rd.png")
    print(f"  plots -> {out}")

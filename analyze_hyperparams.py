import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# config
FILTER_FIELD = "garment_sub_category"
FILTER_VALUE = "jeans"
DATA_ROOT    = "202603_ReUseTex_Setup1"
PATCH_SIZES  = [112, 224, 336, 448]
THRESHOLDS   = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]


def load_annotations():
    anns = []
    for p in sorted(Path(DATA_ROOT).rglob("*_coco.json")):
        try:
            data = json.load(open(p))
        except Exception:
            continue
        ext = data.get("extension", {})
        if ext.get(FILTER_FIELD, "").strip().lower() != FILTER_VALUE:
            continue
        cats = {c["id"]: c["name"] for c in data["categories"]}
        for a in data.get("annotations", []):
            x, y, w, h = a["bbox"]
            anns.append({
                "cat":  cats.get(a["category_id"], "unknown"),
                "bw":   w,
                "bh":   h,
                "area": a.get("area", w * h),
            })
    print(f"loaded {len(anns)} annotations ({FILTER_VALUE})")
    return anns


def plot_sizes(anns):
    side = np.sqrt([a["area"] for a in anns])

    by_cat = defaultdict(list)
    for a in anns:
        by_cat[a["cat"]].append(np.sqrt(a["area"]))

    line_colors = ["#e63946", "#f4a261", "#2a9d8f", "#6d6875"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Defect Size  -  {FILTER_VALUE} (subcategory)", fontsize=13)

    out = Path("plots"); out.mkdir(exist_ok=True)

    # all defects together
    ax = axes[0]
    ax.hist(side, bins=50, color="#457b9d", edgecolor="white")
    ymax = ax.get_ylim()[1]
    for ps, col in zip(PATCH_SIZES, line_colors):
        ax.axvline(ps, linewidth=2, color=col)
        ax.text(ps + 4, ymax * 0.95, f"{ps}px", color=col, fontsize=9, fontweight="bold", va="top")
    ax.set_xlabel("Defect side  (sqrt(area), px)")
    ax.set_ylabel("Count")
    ax.set_title("All categories")

    # per category box plot
    ax = axes[1]
    cats   = sorted(by_cat, key=lambda c: -len(by_cat[c]))
    data   = [by_cat[c] for c in cats]
    labels = [c.replace("_", "\n") for c in cats]
    bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "black"})
    box_colors = ["#e63946","#457b9d","#2a9d8f","#e9c46a","#f4a261","#264653","#a8dadc"]
    for patch, col in zip(bp["boxes"], box_colors):
        patch.set_facecolor(col)
    xmax = len(cats) + 0.5
    for ps, col in zip(PATCH_SIZES, line_colors):
        ax.axhline(ps, linestyle="--", linewidth=1.5, color=col)
        ax.text(xmax, ps, f" {ps}px", color=col, fontsize=9, fontweight="bold", va="center")
    ax.set_xticks(range(1, len(cats) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Defect side  (px)")
    ax.set_title("By category")

    plt.tight_layout()
    fname = out / f"{FILTER_VALUE}_patch_size_analysis.png"
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved: {fname}")


def plot_thresholds(anns):
    recalls = []
    for dt in THRESHOLDS:
        # a defect is 'captured' if its area fills >= dt fraction of any candidate patch
        captured = 0
        for a in anns:
            for ps in PATCH_SIZES:
                best_overlap = min(a["bw"], ps) * min(a["bh"], ps)
                fill = min(a["area"] / max(a["bw"] * a["bh"], 1), 1.0)
                if best_overlap * fill / (ps * ps) >= dt:
                    captured += 1
                    break   # at least one patch size captures it - enough
        recalls.append(captured / len(anns))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([t * 100 for t in THRESHOLDS], [r * 100 for r in recalls],
            marker="o", color="#e63946", linewidth=2)
    ax.axvline(1.0,  linestyle="--", color="gray",    label="current (1%)")
    ax.axhline(90,   linestyle="--", color="#2a9d8f", label="90% recall target")
    for dt, r in zip(THRESHOLDS, recalls):
        ax.annotate(f"{r:.0%}", (dt * 100, r * 100),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Defect threshold  (% of patch area must be defect)")
    ax.set_ylabel("Defect recall  (%)")
    ax.set_title(f"How threshold affects defect capture  -  {FILTER_VALUE}")
    ax.legend()
    plt.tight_layout()
    out = Path("plots"); out.mkdir(exist_ok=True)
    fname = out / f"{FILTER_VALUE}_defect_threshold_recall.png"
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved: {fname}")


if __name__ == "__main__":
    anns = load_annotations()
    plot_sizes(anns)
    plot_thresholds(anns)

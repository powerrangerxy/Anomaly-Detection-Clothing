"""Example images for the phase 2 PaDiM and RD models (loads the checkpoints, no training).

Every image shows: patch | ground truth | anomaly map | predicted mask.
Set SUBCATEGORY to the garment type that was trained.
"""

import shutil
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# anomalib checkpoints contain their own classes, torch >= 2.6 only loads them with weights_only=False
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim, ReverseDistillation

# paths
SUBCATEGORY  = "jeans"          # change this to match the subcategory you ran
EROSION_PX   = 20

DATA_ROOT    = Path("second_phase_rd_padim") / SUBCATEGORY / f"erosion_{EROSION_PX:02d}px"
RESULTS_ROOT = Path("results")
VIS_DIR      = Path("second_phase_rd_padim") / SUBCATEGORY / "results" / "visualizations"

TEST_CLASSES = [
    "abrasion", "hole_tear", "missing_functional_elements",
    "other_defects", "seam_defect", "stain_dirt",
]

MODELS = {
    "padim": {
        "fn":  lambda: Padim(backbone="resnet18", layers=["layer2", "layer3"],
                             visualizer=False),
        "dir": "Padim",
    },
    "reverse_distillation": {
        "fn":  lambda: ReverseDistillation(backbone="resnet18",
                                           layers=["layer1", "layer2", "layer3"],
                                           visualizer=False),
        "dir": "ReverseDistillation",
    },
}


def find_checkpoint(model_dir: str) -> Path | None:
    """Return the latest checkpoint (highest version) for this model."""
    base  = RESULTS_ROOT / model_dir / "MVTecAD" / DATA_ROOT.name
    ckpts = sorted(base.glob("v*/weights/lightning/model.ckpt")) if base.exists() else []
    return ckpts[-1] if ckpts else None


def build_temp_dataset() -> Path:
    """Symlink the second-phase dataset into a temp MVTecAD-compatible directory."""
    tmp = Path(tempfile.mkdtemp()) / DATA_ROOT.name
    tmp.mkdir(parents=True)
    src = DATA_ROOT

    # train/good, anomalib requires at least one image here
    (tmp / "train" / "good").mkdir(parents=True)
    train_imgs = sorted((src / "train" / "good").glob("*.png"))
    if train_imgs:
        (tmp / "train" / "good" / train_imgs[0].name).symlink_to(train_imgs[0].resolve())

    # val/good, directory must exist
    (tmp / "val" / "good").mkdir(parents=True)

    # test/good
    (tmp / "test" / "good").mkdir(parents=True)
    for p in sorted((src / "test" / "good").glob("*.png")):
        (tmp / "test" / "good" / p.name).symlink_to(p.resolve())

    # test/{defect} + ground_truth/{defect}
    for cls in TEST_CLASSES:
        cls_test = src / "test" / cls
        cls_gt   = src / "ground_truth" / cls
        if not cls_test.exists():
            continue
        (tmp / "test" / cls).mkdir(parents=True)
        (tmp / "ground_truth" / cls).mkdir(parents=True)
        for p in sorted(cls_test.glob("*.png")):
            (tmp / "test" / cls / p.name).symlink_to(p.resolve())
            mask = cls_gt / p.name
            if mask.exists():
                (tmp / "ground_truth" / cls / p.name).symlink_to(mask.resolve())

    return tmp


def apply_jet(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize arr and apply matplotlib jet colormap -> RGB uint8 H×W×3."""
    import matplotlib.cm as cm
    lo, hi = float(arr.min()), float(arr.max())
    norm   = (arr - lo) / (hi - lo + 1e-8)
    rgba   = cm.jet(norm)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def make_four_panel(orig_path: Path,
                    gt_mask_path: Path | None,
                    anomaly_map: np.ndarray,
                    pred_mask: np.ndarray) -> Image.Image:
    """
    Build a horizontal 4-panel strip at the original image resolution:
      [original] [GT mask] [original + anomaly map] [original + pred mask]
    """
    orig     = Image.open(orig_path).convert("RGB")
    W, H     = orig.size
    orig_arr = np.array(orig)

    # panel 1: original
    p1 = orig

    # panel 2: GT mask
    if gt_mask_path and gt_mask_path.exists():
        gt_arr = np.array(Image.open(gt_mask_path).convert("L").resize((W, H), Image.NEAREST))
        p2 = Image.fromarray(np.stack([gt_arr, gt_arr, gt_arr], axis=-1))
    else:
        p2 = Image.new("RGB", (W, H), (0, 0, 0))

    # panel 3: original + anomaly map (jet heatmap blend)
    if anomaly_map.ndim == 3:
        anomaly_map = anomaly_map[0]          # (1, H, W) -> (H, W)
    amap_resized = np.array(
        Image.fromarray(anomaly_map).resize((W, H), Image.BILINEAR)
    )
    heat_rgb = apply_jet(amap_resized)
    p3 = Image.blend(orig, Image.fromarray(heat_rgb), alpha=0.5)

    # panel 4: original + pred mask (red overlay)
    if pred_mask.ndim == 3:
        pred_mask = pred_mask[0]              # (1, H, W) -> (H, W)
    pmask = np.array(
        Image.fromarray((pred_mask > 0.5).astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
    ) > 127
    overlay = orig_arr.copy()
    overlay[pmask, 0] = 255
    overlay[pmask, 1] = (orig_arr[pmask, 1] * 0.3).astype(np.uint8)
    overlay[pmask, 2] = (orig_arr[pmask, 2] * 0.3).astype(np.uint8)
    p4 = Image.fromarray(overlay)

    # stitch
    strip = Image.new("RGB", (W * 4, H))
    strip.paste(p1, (0,     0))
    strip.paste(p2, (W,     0))
    strip.paste(p3, (W * 2, 0))
    strip.paste(p4, (W * 3, 0))
    return strip


def run_visualization(model_name: str, cfg: dict):
    ckpt = find_checkpoint(cfg["dir"])
    if ckpt is None:
        print(f"  [SKIP] no checkpoint found for {cfg['dir']}/MVTecAD/{DATA_ROOT.name}")
        return

    print(f"  checkpoint : {ckpt}")
    tmp = build_temp_dataset()
    try:
        datamodule = MVTecAD(
            root=str(tmp.parent),
            category=DATA_ROOT.name,
            train_batch_size=1,
            eval_batch_size=1,
        )
        model   = cfg["fn"]()
        engine  = Engine()

        batches = engine.predict(
            model=model,
            datamodule=datamodule,
            ckpt_path=str(ckpt),
        )
        if not batches:
            print("  [WARN] predict returned no results")
            return

        out_base = VIS_DIR / model_name
        saved    = 0

        for batch in batches:
            img_paths = batch.image_path
            if isinstance(img_paths, str):
                img_paths = [img_paths]

            anom_maps  = batch.anomaly_map.cpu().numpy()   # (B, 1, H, W) or (B, H, W)
            pred_masks = batch.pred_mask.cpu().numpy()     # (B, H, W) or (B, 1, H, W)

            if anom_maps.ndim == 4:
                anom_maps = anom_maps[:, 0]                # -> (B, H, W)
            if pred_masks.ndim == 4:
                pred_masks = pred_masks[:, 0]              # -> (B, H, W)

            for i, img_path in enumerate(img_paths):
                img_path = Path(img_path)
                category = img_path.parent.name            # "abrasion", "good", ...
                stem     = img_path.stem

                gt_mask_path = (
                    DATA_ROOT / "ground_truth" / category / img_path.name
                    if category != "good" else None
                )

                try:
                    vis = make_four_panel(
                        img_path, gt_mask_path,
                        anom_maps[i], pred_masks[i],
                    )
                except Exception as e:
                    print(f"    [WARN] {category}/{stem}: {e}")
                    continue

                out_path = out_base / category / f"{stem}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                vis.save(out_path)
                saved += 1

        print(f"  saved {saved} visualizations -> {out_base}")

    except Exception as e:
        print(f"  [ERROR] {e}")
        traceback.print_exc()
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)


def main():
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    for model_name, cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"  {model_name}")
        print(f"{'='*60}")
        run_visualization(model_name, cfg)
    print("\ndone.")


if __name__ == "__main__":
    main()

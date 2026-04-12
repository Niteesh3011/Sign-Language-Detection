"""
Train YOLO11l (Large) on American Sign Language Dataset
=======================================================
Model: yolo11l.pt (Large variant - ~49M parameters)
Dataset: American-sign-language-2 (Roboflow)
  - 106 classes (ASL letters + food/restaurant words)
  - Train: 17,906 images
  - Valid: 1,546 images
  - Test: 822 images
  - Total: 20,274 images

This script trains the larger YOLO11l model for better accuracy
compared to the nano/small variants previously used.
"""

import gc
import os
import sys
import traceback
from pathlib import Path

import torch
from ultralytics import YOLO, settings


def main():
    # ─── Configuration ─────────────────────────────────────────────
    MODEL_WEIGHTS = "yolo11l.pt"  # YOLO11 Large pretrained weights
    DATASET_DIR = Path(r"American-sign-language-2")
    DATA_YAML = DATASET_DIR / "data_fixed.yaml"
    PROJECT_DIR = "runs/detect"
    RUN_NAME = "asl_yolo11l"

    # Training hyperparameters (tuned for yolo11l - larger model)
    EPOCHS = 100              # More epochs for larger model to converge
    IMGSZ = 640               # Standard YOLO input size
    BATCH_SIZE = 4            # Reduced batch for large model GPU memory
    WORKERS = 2               # Dataloader workers
    LR0 = 0.01                # Initial learning rate
    LRF = 0.01                # Final learning rate fraction
    PATIENCE = 15             # Early stopping patience
    SAVE_PERIOD = 10          # Save checkpoint every N epochs

    # ─── Sanity Checks ─────────────────────────────────────────────
    print("=" * 60)
    print("  YOLO11l Training - American Sign Language Detection")
    print("=" * 60)

    # Check model weights exist
    model_path = Path(MODEL_WEIGHTS)
    if not model_path.exists():
        print(f"ERROR: Model weights not found: {model_path.resolve()}")
        print("Please ensure yolo11l.pt is in the project directory.")
        sys.exit(1)
    print(f"✓ Model weights: {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")

    # Check dataset
    if not (DATASET_DIR / "images" / "train").exists():
        print(f"ERROR: Dataset not found at {DATASET_DIR.resolve()}")
        sys.exit(1)
    print(f"✓ Dataset root: {DATASET_DIR.resolve()}")

    # Check data.yaml
    if not DATA_YAML.exists():
        # Fallback to data.yaml
        DATA_YAML_ALT = DATASET_DIR / "data.yaml"
        if DATA_YAML_ALT.exists():
            DATA_YAML = DATA_YAML_ALT
        else:
            print(f"ERROR: data.yaml not found in {DATASET_DIR}")
            sys.exit(1)
    print(f"✓ Data config: {DATA_YAML}")

    # Check GPU
    device = 0 if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("⚠ No GPU detected - training on CPU (will be very slow)")

    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    print()

    # ─── Disable telemetry ─────────────────────────────────────────
    settings.update({"mlflow": False})
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS"] = "false"

    # ─── Reproducibility ──────────────────────────────────────────
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # ─── Training with fallback profiles ───────────────────────────
    # yolo11l is memory-hungry; try progressively lighter settings
    profiles = [
        {"imgsz": IMGSZ, "batch": BATCH_SIZE, "workers": WORKERS},
        {"imgsz": 640,    "bat  ch": 2,          "workers": 2},
        {"imgsz": 512,    "batch": 2,          "workers": 1},
        {"imgsz": 416,    "batch": 1,          "workers": 0},
    ]

    train_result = None
    attempt_log = []

    for i, profile in enumerate(profiles):
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"{'─' * 60}")
            print(f"  Attempt {i+1}/{len(profiles)}")
            print(f"  imgsz={profile['imgsz']}  batch={profile['batch']}  workers={profile['workers']}")
            print(f"{'─' * 60}")

            model = YOLO(str(model_path))
            train_result = model.train(
                data=str(DATA_YAML),
                epochs=EPOCHS,
                imgsz=profile["imgsz"],
                batch=profile["batch"],
                device=device,
                amp=torch.cuda.is_available(),       # Mixed precision on GPU
                workers=profile["workers"],
                lr0=LR0,
                lrf=LRF,
                patience=PATIENCE,
                project=PROJECT_DIR,
                name=RUN_NAME,
                exist_ok=True,
                save=True,
                save_period=SAVE_PERIOD,
                verbose=True,
                # Augmentation settings for better generalization
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=10.0,
                translate=0.1,
                scale=0.5,
                shear=2.0,
                flipud=0.0,       # No vertical flip (signs have orientation)
                fliplr=0.5,       # Horizontal flip
                mosaic=1.0,
                mixup=0.1,
            )

            print("\n✓ Training completed successfully!")
            break

        except torch.cuda.OutOfMemoryError:
            attempt_log.append((profile, "CUDA OOM"))
            print(f"✗ CUDA OOM with batch={profile['batch']}, trying lighter profile...")
            gc.collect()
            torch.cuda.empty_cache()
            continue

        except Exception as e:
            short_err = f"{type(e).__name__}: {e}"
            attempt_log.append((profile, short_err))
            print(f"✗ Failed: {short_err}")
            traceback.print_exc()
            continue

    if train_result is None:
        print("\n" + "=" * 60)
        print("  ALL TRAINING ATTEMPTS FAILED")
        print("=" * 60)
        for p, err in attempt_log:
            print(f"  - imgsz={p['imgsz']}, batch={p['batch']} → {err}")
        sys.exit(1)

    # ─── Post-training Summary ────────────────────────────────────
    best_pt = Path(train_result.save_dir) / "weights" / "best.pt"
    last_pt = Path(train_result.save_dir) / "weights" / "last.pt"

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best model:  {best_pt}")
    print(f"  Last model:  {last_pt}")
    print(f"  Results dir: {train_result.save_dir}")

    if best_pt.exists():
        print(f"  Best model size: {best_pt.stat().st_size / 1e6:.1f} MB")

    # ─── Validation on test set ───────────────────────────────────
    print("\n" + "─" * 60)
    print("  Running validation on test set...")
    print("─" * 60)

    best_model = YOLO(str(best_pt))
    val_results = best_model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=640,
        batch=4,
        device=device,
    )

    print(f"\n  Test mAP50:    {val_results.box.map50:.4f}")
    print(f"  Test mAP50-95: {val_results.box.map:.4f}")
    print(f"  Precision:     {val_results.box.mp:.4f}")
    print(f"  Recall:        {val_results.box.mr:.4f}")

    # ─── Copy best model to project root ──────────────────────────
    import shutil
    root_best = Path("best.pt")
    if best_pt.exists():
        shutil.copy2(best_pt, root_best)
        print(f"\n  ✓ Copied best.pt to project root: {root_best.resolve()}")

    print("\n" + "=" * 60)
    print("  Done! Use 'best.pt' for inference in app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

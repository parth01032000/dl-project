"""
train_part1.py
==============
Standalone training script for Part 1 - CNN Glyph Classifier.

Run from the project root:
    python3 scripts/train_part1.py

Outputs:
    checkpoints/part1_glyph_model.pth   -- best model checkpoint
    notebooks/figs/part1_loss_curve.png -- training curves
    notebooks/figs/part1_confusion.png  -- confusion matrix (test set)
"""

import sys
import os

# Always run from project root regardless of where script is called from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import torch
import matplotlib
matplotlib.use('Agg')  # non-interactive backend - works in terminal with no display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from scripts.utils import seed_all, get_dataloader
from scripts.part1_preprocessing import (
    part1_build_preprocess_args,
    part1_preprocess_x,
    part1_preprocess_y,
    part1_pad_collate,
)
from models.part1_glyph_model import (
    part1_build_vocab,
    part1_build_model_args,
    part1_glyph_classification_model,
    part1_train_model,
    part1_test_model,
)

# =============================================================
# CONFIGURATION  -- edit these to experiment with hyperparams
# =============================================================
DEBUG       = False          # True = use 1k dataset for quick smoke-test
BATCH_SIZE  = 128
NUM_EPOCHS  = 20
LR          = 1e-3
SEED        = 2026

H5_FULL     = Path("datasets/glyph_80k.h5")
H5_SMALL    = Path("datasets/glyph_1k.h5")
CHECKPOINT  = Path("checkpoints/part1_glyph_model.pth")
FIGS_DIR    = Path("notebooks/figs")
# =============================================================

def main():
    seed_all(SEED)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Select dataset ----
    H5 = H5_SMALL if DEBUG else H5_FULL
    if not H5.exists():
        print(f"ERROR: Dataset not found at {H5}")
        print("  For the 1k debug set use: DEBUG=True")
        print("  For the full set, download glyph_80k.h5 into datasets/")
        sys.exit(1)

    # ---- Device ----
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("=" * 60)
    print("PART 1 – CNN Glyph Classifier")
    print("=" * 60)
    print(f"  Dataset : {H5}")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {NUM_EPOCHS}")
    print(f"  LR      : {LR}")
    print(f"  Batch   : {BATCH_SIZE}")
    print(f"  Seed    : {SEED}")
    print("=" * 60)

    # ---- Vocabulary ----
    vocab_obj       = part1_build_vocab()
    preprocess_args = part1_build_preprocess_args()
    print(f"\nVocabulary ({len(vocab_obj)} classes): {vocab_obj.itos}\n")

    # ---- Data loaders ----
    loader_setup = (
        vocab_obj,
        preprocess_args,
        part1_preprocess_x,
        part1_preprocess_y,
        part1_pad_collate,
    )

    print("Loading data loaders...")
    train_loader = get_dataloader(
        H5, BATCH_SIZE, split="train",
        loader_setup=loader_setup, shuffle=True, use_cache=True,
    )
    valid_loader = get_dataloader(
        H5, BATCH_SIZE, split="valid",
        loader_setup=loader_setup, shuffle=False, use_cache=True,
    )
    test_loader = get_dataloader(
        H5, BATCH_SIZE, split="test",
        loader_setup=loader_setup, shuffle=False, use_cache=True,
    )
    print(f"  train batches : {len(train_loader)}")
    print(f"  valid batches : {len(valid_loader)}")
    print(f"  test  batches : {len(test_loader)}\n")

    # ---- Model ----
    model_args = part1_build_model_args(vocab_obj)
    model      = part1_glyph_classification_model(**model_args)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: GlyphCNN")
    print(f"  Architecture : {model}")
    print(f"  Trainable params : {total_params:,}")
    assert total_params < 50_000, (
        f"Model has {total_params:,} params -- exceeds 50k limit!"
    )
    print(f"  [OK] Under 50k parameter budget\n")

    # ---- Train ----
    seed_all(SEED)  # re-seed just before training for reproducibility
    print("Starting training...\n")
    history = part1_train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        device=device,
        save_path=str(CHECKPOINT),
        resume=False,
    )

    best_val = max(history["val_acc"])
    print(f"\nTraining complete. Best val accuracy: {best_val*100:.2f}%")

    # ---- Plot training curves ----
    print("\nSaving training curves...")
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], "o-", label="Train", color="teal",   lw=2, ms=4)
    axes[0].plot(epochs, history["val_loss"],   "s-", label="Val",   color="orange", lw=2, ms=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Part 1 – Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [a*100 for a in history["train_acc"]], "o-", label="Train", color="teal",   lw=2, ms=4)
    axes[1].plot(epochs, [a*100 for a in history["val_acc"]],   "s-", label="Val",   color="orange", lw=2, ms=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Part 1 – Accuracy")
    axes[1].set_ylim([0, 100])
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    curve_path = FIGS_DIR / "part1_loss_curve.png"
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {curve_path}")

    # ---- Test evaluation ----
    print("\nEvaluating on test set...")
    test_acc = part1_test_model(
        model=model,
        test_loader=test_loader,
        checkpoint_path=CHECKPOINT,
        device=device,
    )
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")

    # ---- Confusion matrix ----
    print("\nGenerating confusion matrix...")
    all_preds  = []
    all_labels = []

    model.eval()
    model.to(device)
    with torch.no_grad():
        for X, Y in test_loader:
            X = X.to(device)
            Y = Y.to(device).view(-1)
            preds = model(X).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(Y.cpu().numpy())

    class_names = vocab_obj.itos
    cm   = confusion_matrix(all_labels, all_preds)
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax2, colorbar=True, cmap="Blues")
    ax2.set_title("Part 1 – Confusion Matrix (Test Set)")
    plt.tight_layout()
    conf_path = FIGS_DIR / "part1_confusion.png"
    plt.savefig(conf_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {conf_path}")

    # ---- Final summary ----
    print()
    print("=" * 60)
    print("PART 1 FINAL SUMMARY")
    print("=" * 60)
    print(f"  Model             : GlyphCNN (1D CNN)")
    print(f"  Trainable params  : {total_params:,}")
    print(f"  Epochs trained    : {len(history['train_loss'])}")
    print(f"  Best val accuracy : {best_val*100:.2f}%")
    print(f"  Test accuracy     : {test_acc*100:.2f}%")
    print(f"  Checkpoint        : {CHECKPOINT}")
    print(f"  Loss curve        : {curve_path}")
    print(f"  Confusion matrix  : {conf_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

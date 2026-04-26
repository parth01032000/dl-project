"""
train_part2.py
==============
Standalone training script for Part 2 - RNN Infix Expression Recognition.

Run from the project root:
    python3 scripts/train_part2.py

Outputs:
    checkpoints/part2_infix_model.pth      -- best model checkpoint
    notebooks/figs/part2_loss_curve.png    -- training curves
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from scripts.utils import seed_all, get_dataloader
from scripts.part2_preprocessing import (
    part2_build_preprocess_args,
    part2_preprocess_x,
    part2_preprocess_y,
    part2_pad_collate,
)
from models.part2_infix_model import (
    part2_build_vocab,
    part2_build_model_args,
    part2_infix_recognition_model,
    part2_train_model,
    part2_test_model,
)

# =============================================================
# CONFIGURATION
# =============================================================
DEBUG      = False        # True = use 1k dataset for quick smoke-test
BATCH_SIZE = 128
NUM_EPOCHS = 30
LR         = 1e-3
SEED       = 2026

H5_FULL    = Path("datasets/infix_88k.h5")
H5_SMALL   = Path("datasets/infix_1k.h5")
CHECKPOINT = Path("checkpoints/part2_infix_model.pth")
FIGS_DIR   = Path("notebooks/figs")
# =============================================================


def main():
    seed_all(SEED)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Select dataset ----
    H5 = H5_SMALL if DEBUG else H5_FULL
    if not H5.exists():
        print(f"ERROR: Dataset not found at {H5}")
        print("  For the 1k debug set use: DEBUG=True")
        print("  For the full set, download infix_88k.h5 into datasets/")
        sys.exit(1)

    # ---- Device ----
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("=" * 60)
    print("PART 2 - RNN Infix Expression Recognition")
    print("=" * 60)
    print(f"  Dataset : {H5}")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {NUM_EPOCHS}")
    print(f"  LR      : {LR}")
    print(f"  Batch   : {BATCH_SIZE}")
    print(f"  Seed    : {SEED}")
    print("=" * 60)

    # ---- Vocabulary ----
    vocab_obj       = part2_build_vocab()
    preprocess_args = part2_build_preprocess_args(vocab_obj)
    print(f"\nVocabulary ({len(vocab_obj)} tokens): {vocab_obj.itos}")
    print(f"  pad_id={preprocess_args['pad_token_value']}")
    print()

    # ---- Data loaders ----
    loader_setup = (
        vocab_obj,
        preprocess_args,
        part2_preprocess_x,
        part2_preprocess_y,
        part2_pad_collate,
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

    # ---- Inspect one batch shape ----
    sample = next(iter(train_loader))
    X_s, X_lens_s, Y_s = sample
    print(f"Sample batch shapes:")
    print(f"  X (strokes)  : {X_s.shape}   (B, N_strokes, stroke_len)")
    print(f"  X_lens       : {X_lens_s.shape}")
    print(f"  Y (tokens)   : {Y_s.shape}    (B, T)\n")

    # ---- Model ----
    model_args = part2_build_model_args(vocab_obj)
    model      = part2_infix_recognition_model(**model_args)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: InfixRNN (manual GRU encoder-decoder)")
    print(f"  Trainable params : {total_params:,}")
    assert total_params < 200_000, (
        f"Model has {total_params:,} params -- exceeds 200k limit!"
    )
    print(f"  [OK] Under 200k parameter budget\n")

    # ---- Train ----
    seed_all(SEED)
    print("Starting training...\n")
    history = part2_train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        device=device,
        save_path=str(CHECKPOINT),
        resume=False,
    )

    best_val_la = max(history["val_acc"])
    print(f"\nTraining complete.")
    print(f"  Best val LA : {best_val_la*100:.2f}%")

    # ---- Plot training curves ----
    print("\nSaving training curves...")
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], "o-",
                 label="Train Loss", color="teal",   lw=2, ms=4)
    axes[0].plot(epochs, history["val_loss"],   "s-",
                 label="Val Loss",   color="orange", lw=2, ms=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Part 2 - Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [a*100 for a in history["val_acc"]], "s-",
                 label="Val LA (%)", color="orange", lw=2, ms=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Levenshtein Accuracy (%)")
    axes[1].set_title("Part 2 - Validation LA")
    axes[1].set_ylim([0, 100])
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    curve_path = FIGS_DIR / "part2_loss_curve.png"
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {curve_path}")

    # ---- Test evaluation ----
    print("\nEvaluating on test set...")
    avg_la, avg_cer = part2_test_model(
        model=model,
        test_loader=test_loader,
        checkpoint_path=CHECKPOINT,
        device=device,
    )

    # ---- Final summary ----
    print()
    print("=" * 60)
    print("PART 2 FINAL SUMMARY")
    print("=" * 60)
    print(f"  Model              : InfixRNN (manual GRU)")
    print(f"  Trainable params   : {total_params:,}")
    print(f"  Epochs trained     : {len(history['train_loss'])}")
    print(f"  Best val LA        : {best_val_la*100:.2f}%")
    print(f"  Test LA            : {avg_la*100:.2f}%")
    print(f"  Test forced CER    : {avg_cer*100:.2f}%")
    print(f"  Checkpoint         : {CHECKPOINT}")
    print(f"  Loss curve         : {curve_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

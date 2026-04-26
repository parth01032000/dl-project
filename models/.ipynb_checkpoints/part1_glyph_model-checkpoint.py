# ------------------------------------------------------------
# Copyright (c) 2026 UCD COMP47650
# Version: 1.0.3
#
# Private coursework for University College Dublin.
# Do NOT share publicly or upload to repositories.
# Do NOT submit this code to AI tools or external services.
# ------------------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scripts.utils import Vocab
from pathlib import Path
from collections import Counter, OrderedDict

# ----------------------
# DO NOT MODIFY
# Build the glyph vocabulary
def part1_build_vocab() -> Vocab:
    """
    Create and return the scripts.utils.Vocab for the glyph classification task.
    """
    token_counter = Counter({
        '0': 1, '1': 1, '2': 1, '3': 1, '4': 1,
        '5': 1, '6': 1, '7': 1, '8': 1, '9': 1,
        '+': 1, '-': 1, '*': 1, '/': 1,
        '.': 1, '(': 1, ')': 1, '=': 1,
    })
    ordered_dict = OrderedDict(
        sorted(token_counter.items(), key=lambda x: x[1], reverse=True)
    )
    return Vocab(ordered_dict)


# ----------------------
# Build model argument dictionary based on the vocabulary
def part1_build_model_args(vocab: Vocab) -> dict:
    """
    Build model argument dictionary for the 1D CNN glyph classifier.
    """
    model_args = {
        'num_classes': len(vocab),   # 18 output classes
        'in_channels': 1,            # 1D conv: treat the feature vector as 1 channel
        'vec_length': 3 * 128 + 2,   # Length of flattened input vector (386)
        # CNN architecture hyperparameters (tuned for <50k params)
        'channels': [32, 64],        # Number of filters per conv block
        'kernel_size': 5,            # Convolution kernel width
        'dropout': 0.3,              # Dropout rate for regularisation
        'hidden_dim': 128,           # MLP hidden dimension after global pooling
    }
    return model_args


# ----------------------
# Build the CNN model from kwargs
def part1_glyph_classification_model(**kwargs) -> nn.Module:
    """
    Build and return the GlyphCNN model.

    Args (via kwargs):
        num_classes (int): Number of output classes (18).
        in_channels (int): Input channels for Conv1d (1).
        vec_length (int): Length of flattened stroke input (386).
        channels (list): Output channels for each conv block.
        kernel_size (int): Kernel size for conv layers.
        dropout (float): Dropout probability.
        hidden_dim (int): Hidden size of the MLP head.

    Returns:
        nn.Module: GlyphCNN model.
    """
    return GlyphCNN(
        num_classes=kwargs.get('num_classes', 18),
        in_channels=kwargs.get('in_channels', 1),
        channels=kwargs.get('channels', [32, 64]),
        kernel_size=kwargs.get('kernel_size', 5),
        dropout=kwargs.get('dropout', 0.3),
        hidden_dim=kwargs.get('hidden_dim', 128),
    )


# ----------------------
# The real CNN model
class GlyphCNN(nn.Module):
    """
    Compact 1D CNN for glyph classification from flattened stroke sequences.

    Architecture:
        Input (B, L) → unsqueeze → (B, 1, L)
        Conv Block 1: Conv1d(1, 32, k=5) → BN → ReLU → MaxPool(2)
        Conv Block 2: Conv1d(32, 64, k=5) → BN → ReLU → MaxPool(2)
        Global Average Pool → (B, 64)
        MLP: Linear(64, 128) → ReLU → Dropout → Linear(128, 18)

    Total parameters: ~15-20k (well under 50k limit).
    """

    def __init__(
        self,
        num_classes: int = 18,
        in_channels: int = 1,
        channels: list = None,
        kernel_size: int = 5,
        dropout: float = 0.3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64]

        # Build convolutional blocks dynamically from channels list
        conv_blocks = []
        prev_ch = in_channels
        for out_ch in channels:
            conv_blocks.append(nn.Conv1d(
                in_channels=prev_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                padding=kernel_size // 2,   # same-padding to preserve length
            ))
            conv_blocks.append(nn.BatchNorm1d(out_ch))
            conv_blocks.append(nn.ReLU())
            conv_blocks.append(nn.MaxPool1d(kernel_size=2))
            prev_ch = out_ch

        self.conv_net = nn.Sequential(*conv_blocks)

        # Global average pooling reduces (B, C, L') → (B, C)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # MLP classification head
        self.classifier = nn.Sequential(
            nn.Linear(prev_ch, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Flattened stroke input of shape (B, L)

        Returns:
            Tensor: Logits of shape (B, num_classes)
        """
        # Add channel dimension: (B, L) → (B, 1, L)
        x = x.unsqueeze(1)

        # Convolutional feature extraction: (B, 1, L) → (B, C, L')
        x = self.conv_net(x)

        # Global average pooling: (B, C, L') → (B, C, 1) → (B, C)
        x = self.global_avg_pool(x).squeeze(-1)

        # Classification: (B, C) → (B, num_classes)
        logits = self.classifier(x)
        return logits


# ----------------------
# Real training function
def part1_train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    num_epochs: int,
    lr: float = 1e-3,
    device: str = "cpu",
    save_path: str | None = None,
    resume: bool = False
) -> dict:
    """
    Train the glyph CNN classifier.

    Implements:
        - Adam optimiser with given learning rate
        - CrossEntropyLoss
        - Per-epoch train/validation accuracy and loss tracking
        - Best-model checkpoint saving (by validation accuracy)
        - Optional resume from checkpoint

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training dataloader.
        valid_loader (DataLoader): Validation dataloader.
        num_epochs (int): Number of epochs to train.
        lr (float): Learning rate for Adam.
        device (str): Device ('cpu', 'cuda', 'mps').
        save_path (str | Path): Path to save the best checkpoint.
        resume (bool): If True, resume from an existing checkpoint.

    Returns:
        dict: History with keys 'train_loss', 'train_acc', 'val_loss', 'val_acc'.
    """
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "train_acc":  [],
        "val_loss":   [],
        "val_acc":    [],
    }

    checkpoint_path = Path(save_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    start_epoch = 1

    # --- Optional: resume from existing checkpoint ---
    if resume and checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_acc = ckpt.get("val_acc", 0.0)
        start_epoch = ckpt.get("epoch", 0) + 1
        history = ckpt.get("history", history)
        print(f"  Resumed at epoch {start_epoch - 1}, best_val_acc={best_val_acc:.4f}")
    elif resume:
        print(f"Warning: resume=True but no checkpoint found at {checkpoint_path}. Starting fresh.")

    # --- Training loop ---
    for epoch in range(start_epoch, start_epoch + num_epochs):

        # ---- Train phase ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{start_epoch + num_epochs - 1} [Train]", leave=True)
        for inputs, targets in pbar:
            inputs  = inputs.to(device)
            targets = targets.to(device).view(-1)   # (B,)

            optimizer.zero_grad()
            logits = model(inputs)                  # (B, num_classes)
            loss   = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            running_loss += loss.item() * targets.size(0)
            preds   = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total   += targets.size(0)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc":  f"{correct / total:.4f}"
            })

        train_loss = running_loss / total
        train_acc  = correct / total

        # ---- Validation phase ----
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            pbar_val = tqdm(valid_loader, desc=f"Epoch {epoch}/{start_epoch + num_epochs - 1} [Valid]", leave=True)
            for inputs, targets in pbar_val:
                inputs  = inputs.to(device)
                targets = targets.to(device).view(-1)

                logits = model(inputs)
                loss   = criterion(logits, targets)

                val_running_loss += loss.item() * targets.size(0)
                preds       = logits.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total   += targets.size(0)

                pbar_val.set_postfix({
                    "val_loss": f"{loss.item():.4f}",
                    "val_acc":  f"{val_correct / val_total:.4f}"
                })

        val_loss = val_running_loss / val_total
        val_acc  = val_correct / val_total

        # ---- Log history ----
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        # ---- Save best checkpoint ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # DO NOT MODIFY THESE DICTIONARY KEYS
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc":          best_val_acc,
                    "history":          history,
                },
                checkpoint_path,
            )
            print(f"  ✓ Saved best checkpoint (val_acc={best_val_acc:.4f}) → {checkpoint_path}")

    return history


# ----------------------
# DO NOT MODIFY
# Model testing function for the evaluation notebook
def part1_test_model(
    model: nn.Module,
    test_loader: DataLoader,
    checkpoint_path,
    device,
):
    """
    Evaluate a trained model on the test dataset.

    Args:
        model (nn.Module): Model to evaluate.
        test_loader (DataLoader): DataLoader containing test samples.
        checkpoint_path (Path | str): Path to a saved model checkpoint.
        device (str): Device for evaluation ('cpu', 'cuda', 'mps').

    Returns:
        float: Test accuracy.
    """
    print(f"Using device: {device}")
    epoch = -1

    # Load weights from checkpoint
    assert checkpoint_path.exists(), f"Checkpoint not found at {checkpoint_path}"
    if checkpoint_path and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        val_acc = checkpoint["val_acc"]
        epoch = checkpoint["epoch"]
        print(
            f"Model from checkpoint at Epoch {epoch}, "
            f"(Valid acc={val_acc:.4f}): "
            f"{checkpoint_path.parent.name}/{checkpoint_path.name}"
        )

    model.to(device)
    model.eval()

    correct_preds  = 0
    total_samples  = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Epoch {epoch} [Test]", leave=True)

        for inputs, targets in pbar:
            inputs  = inputs.to(device)
            targets = targets.to(device).view(-1)

            logits = model(inputs)
            preds  = torch.argmax(logits, dim=1)

            correct_preds += (preds == targets).sum().item()
            total_samples += targets.size(0)

            running_acc = correct_preds / total_samples

            pbar.set_postfix({
                "Batch Class Acc": f"{running_acc:.4f}"
            })

    test_accuracy = correct_preds / total_samples
    return test_accuracy
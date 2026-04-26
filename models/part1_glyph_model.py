# ------------------------------------------------------------
# COMP47650 Deep Learning – Part 1: CNN Glyph Classifier
# ------------------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from scripts.utils import Vocab
from pathlib import Path
from collections import Counter, OrderedDict


# -------------------------------------------------------
# DO NOT MODIFY – Build the glyph vocabulary
# -------------------------------------------------------
def part1_build_vocab() -> Vocab:
    """Build and return the 18-class glyph vocabulary."""
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


# -------------------------------------------------------
# Build model argument dictionary
# -------------------------------------------------------
def part1_build_model_args(vocab: Vocab) -> dict:
    """
    Return hyperparameters for GlyphCNN.

    Architecture stays well under the 50k parameter budget:
        channels=[32,64], kernel=5, hidden=128  =>  ~19k params
    """
    return {
        'num_classes': len(vocab),   # 18 output classes
        'channels':    [32, 64],     # filters per Conv1d block
        'kernel_size': 5,            # convolution kernel width
        'dropout':     0.3,          # dropout rate in MLP head
        'hidden_dim':  128,          # MLP hidden layer width
    }


# -------------------------------------------------------
# Model factory function
# -------------------------------------------------------
def part1_glyph_classification_model(**kwargs) -> nn.Module:
    """
    Instantiate and return the GlyphCNN model.

    All parameters are passed via kwargs so the evaluation
    notebook can call this function without modification.
    """
    return GlyphCNN(
        num_classes=kwargs.get('num_classes', 18),
        channels=kwargs.get('channels',    [32, 64]),
        kernel_size=kwargs.get('kernel_size', 5),
        dropout=kwargs.get('dropout',      0.3),
        hidden_dim=kwargs.get('hidden_dim', 128),
    )


# -------------------------------------------------------
# GlyphCNN – compact 1D convolutional classifier
# -------------------------------------------------------
class GlyphCNN(nn.Module):
    """
    1D CNN for glyph classification from flattened stroke vectors.

    Architecture
    ------------
    Input  : (B, L)  where L = 3*128+2 = 386 (flattened strokes)
    Unsqueeze channel dim -> (B, 1, L)

    Block 1: Conv1d(1,  32, k=5, same-pad) -> BN -> ReLU -> MaxPool(2)
    Block 2: Conv1d(32, 64, k=5, same-pad) -> BN -> ReLU -> MaxPool(2)

    Global Average Pool -> (B, 64)

    MLP head: Linear(64, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 18)

    Total trainable parameters: ~19 000  (well under 50k limit)
    """

    def __init__(
        self,
        num_classes: int = 18,
        channels:    list = None,
        kernel_size: int  = 5,
        dropout:     float = 0.3,
        hidden_dim:  int  = 128,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64]

        # ----- Convolutional feature extractor -----
        conv_blocks = []
        in_ch = 1  # single input channel (flattened vector treated as 1D signal)
        for out_ch in channels:
            conv_blocks += [
                nn.Conv1d(in_ch, out_ch, kernel_size,
                          padding=kernel_size // 2),  # same-length padding
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
            ]
            in_ch = out_ch

        self.conv_net = nn.Sequential(*conv_blocks)

        # Global average pool collapses spatial dimension: (B,C,L') -> (B,C,1)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # ----- MLP classification head -----
        self.head = nn.Sequential(
            nn.Linear(in_ch, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L) flattened stroke input
        Returns:
            logits: (B, num_classes)
        """
        x = x.unsqueeze(1)          # (B, L) -> (B, 1, L)
        x = self.conv_net(x)         # (B, 1, L) -> (B, C, L')
        x = self.gap(x).squeeze(-1)  # (B, C, L') -> (B, C)
        return self.head(x)          # (B, C) -> (B, num_classes)


# -------------------------------------------------------
# Training function
# -------------------------------------------------------
def part1_train_model(
    model:        nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    num_epochs:   int,
    lr:           float = 1e-3,
    device:       str   = "cpu",
    save_path:    str   = None,
    resume:       bool  = False,
) -> dict:
    """
    Train GlyphCNN with Adam + CrossEntropyLoss.

    Saves the best checkpoint (by validation accuracy) to save_path.
    Checkpoint keys (DO NOT CHANGE): epoch, model_state_dict, val_acc, history.

    Returns
    -------
    history : dict with keys train_loss, train_acc, val_loss, val_acc
    """
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }

    checkpoint_path = Path(save_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    start_epoch  = 1

    # ------ Optional resume ------
    if resume and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_acc = ckpt.get("val_acc",   0.0)
        start_epoch  = ckpt.get("epoch",     0) + 1
        history      = ckpt.get("history",   history)
        print(f"Resumed from epoch {start_epoch-1}, best_val_acc={best_val_acc:.4f}")
    elif resume:
        print(f"Warning: resume=True but no checkpoint at {checkpoint_path}. Starting fresh.")

    # ------ Epoch loop ------
    for epoch in range(start_epoch, start_epoch + num_epochs):

        # === TRAIN ===
        model.train()
        run_loss = correct = total = 0

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch}/{start_epoch+num_epochs-1} [Train]",
                    leave=True)
        for X, Y in pbar:
            X = X.to(device)
            Y = Y.to(device).view(-1)

            optimizer.zero_grad()
            logits = model(X)
            loss   = criterion(logits, Y)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * Y.size(0)
            correct  += (logits.argmax(1) == Y).sum().item()
            total    += Y.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{correct/total:.4f}")

        train_loss = run_loss / total
        train_acc  = correct  / total

        # === VALIDATE ===
        model.eval()
        v_loss = v_correct = v_total = 0

        with torch.no_grad():
            vbar = tqdm(valid_loader,
                        desc=f"Epoch {epoch}/{start_epoch+num_epochs-1} [Valid]",
                        leave=True)
            for X, Y in vbar:
                X = X.to(device)
                Y = Y.to(device).view(-1)
                logits = model(X)
                loss   = criterion(logits, Y)

                v_loss    += loss.item() * Y.size(0)
                v_correct += (logits.argmax(1) == Y).sum().item()
                v_total   += Y.size(0)

                vbar.set_postfix(val_loss=f"{loss.item():.4f}",
                                 val_acc=f"{v_correct/v_total:.4f}")

        val_loss = v_loss    / v_total
        val_acc  = v_correct / v_total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
              f" | val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        # === Save best checkpoint ===
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc":          best_val_acc,
                    "history":          history,
                },
                checkpoint_path,
            )
            print(f"  -> Saved best checkpoint  val_acc={best_val_acc:.4f}  [{checkpoint_path}]")

    return history


# -------------------------------------------------------
# DO NOT MODIFY – Test function used by evaluation notebook
# -------------------------------------------------------
def part1_test_model(
    model:           nn.Module,
    test_loader:     DataLoader,
    checkpoint_path,
    device,
):
    """
    Load best checkpoint and evaluate on test set.
    Returns test accuracy (float).
    """
    print(f"Using device: {device}")

    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint  epoch={ckpt['epoch']}  val_acc={ckpt['val_acc']:.4f}")

    model.to(device)
    model.eval()

    correct = total = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"[Test]", leave=True)
        for X, Y in pbar:
            X = X.to(device)
            Y = Y.to(device).view(-1)
            preds    = model(X).argmax(1)
            correct += (preds == Y).sum().item()
            total   += Y.size(0)
            pbar.set_postfix(acc=f"{correct/total:.4f}")

    return correct / total

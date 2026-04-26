"""
part2_infix_model.py
====================
RNN encoder-decoder for infix mathematical expression recognition.

Architecture
------------
1. Stroke pre-encoder : two Conv1d layers compress each stroke vector
                        from length 128 down to a fixed embedding dim.
2. Encoder            : single-layer bidirectional GRU (manual cell).
                        Hidden states from both directions are concatenated
                        then projected to decoder hidden size.
3. Decoder            : single-layer unidirectional GRU (manual cell)
                        with teacher forcing during training and greedy
                        autoregressive decoding at inference.

All RNN cells are implemented manually using nn.Linear + torch ops.
nn.RNN / nn.LSTM / nn.GRU / nn.MultiheadAttention are NOT used.

Parameter budget: well under 200k.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from scripts.utils import Vocab, batch_LA
from collections import Counter, OrderedDict


# -------------------------------------------------------
# Vocabulary
# -------------------------------------------------------
def part2_build_vocab() -> Vocab:
    """22-token vocabulary for infix recognition."""
    token_counter = Counter({
        '0': 1, '1': 1, '2': 1, '3': 1, '4': 1,
        '5': 1, '6': 1, '7': 1, '8': 1, '9': 1,
        '+': 1, '-': 1, '*': 1, '/': 1,
        '.': 1, '(': 1, ')': 1, '=': 1,
    })
    ordered_dict = OrderedDict(
        sorted(token_counter.items(), key=lambda x: x[1], reverse=True)
    )
    vocab_obj = Vocab(ordered_dict, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    vocab_obj.set_default_index(vocab_obj['<unk>'])
    assert vocab_obj['<pad>'] == 1, "Expected <pad>=1"
    assert vocab_obj['<bos>'] == 2, "Expected <bos>=2"
    assert vocab_obj['<eos>'] == 3, "Expected <eos>=3"
    return vocab_obj


# -------------------------------------------------------
# Model arguments
# -------------------------------------------------------
def part2_build_model_args(vocab: Vocab) -> dict:
    """Hyperparameters for the RNN encoder-decoder."""
    return {
        "vocab_size":   len(vocab),  # 22
        "max_len":      64,          # max output sequence length
        "pad_id":       1,
        "bos_id":       2,
        "eos_id":       3,
        # Architecture (tuned to stay under 200k params)
        "stroke_dim":   128,         # raw stroke vector length
        "conv_hidden":  16,          # conv pre-encoder output channels
        "enc_hidden":   48,          # GRU encoder hidden size (each direction)
        "dec_hidden":   96,          # GRU decoder hidden size
        "embed_dim":    32,          # token embedding dimension
        "dropout":      0.3,
    }


# -------------------------------------------------------
# Model factory
# -------------------------------------------------------
def part2_infix_recognition_model(**kwargs) -> nn.Module:
    """Build and return the InfixRNN model."""
    return InfixRNN(
        vocab_size  = kwargs["vocab_size"],
        max_len     = kwargs["max_len"],
        pad_id      = kwargs["pad_id"],
        bos_id      = kwargs["bos_id"],
        eos_id      = kwargs["eos_id"],
        stroke_dim  = kwargs.get("stroke_dim",   128),
        conv_hidden = kwargs.get("conv_hidden",   32),
        enc_hidden  = kwargs.get("enc_hidden",   128),
        dec_hidden  = kwargs.get("dec_hidden",   256),
        embed_dim   = kwargs.get("embed_dim",     64),
        dropout     = kwargs.get("dropout",      0.3),
    )


# -------------------------------------------------------
# Manual GRU Cell
# -------------------------------------------------------
class GRUCell(nn.Module):
    """
    Single-step GRU cell implemented with nn.Linear only.
    No nn.GRU / nn.RNN / nn.LSTM used.

    h_t = GRU(x_t, h_{t-1})
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        # Reset, update, new gates (input + hidden -> hidden)
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_n = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, input_size)
            h : (B, hidden_size)
        Returns:
            h_new : (B, hidden_size)
        """
        xh = torch.cat([x, h], dim=-1)
        r  = torch.sigmoid(self.W_r(xh))           # reset gate
        z  = torch.sigmoid(self.W_z(xh))           # update gate
        xh_r = torch.cat([x, r * h], dim=-1)
        n  = torch.tanh(self.W_n(xh_r))            # new gate
        return (1 - z) * n + z * h                 # h_new


# -------------------------------------------------------
# Full InfixRNN model
# -------------------------------------------------------
class InfixRNN(nn.Module):
    """
    Stroke-to-infix encoder-decoder with manual GRU cells.

    Encoder:
        - Conv1d pre-encoder compresses each stroke (length 128) -> conv_hidden
        - Bidirectional GRU over stroke sequence
        - Final hidden projected to decoder initial hidden

    Decoder:
        - Embedding layer for output tokens
        - Unidirectional GRU cell
        - Linear projection -> vocab logits
        - Teacher forcing during training
        - Greedy decoding at inference
    """

    def __init__(
        self,
        vocab_size:  int,
        max_len:     int,
        pad_id:      int,
        bos_id:      int,
        eos_id:      int,
        stroke_dim:  int   = 128,
        conv_hidden: int   = 32,
        enc_hidden:  int   = 128,
        dec_hidden:  int   = 256,
        embed_dim:   int   = 64,
        dropout:     float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len    = max_len
        self.pad_id     = pad_id
        self.bos_id     = bos_id
        self.eos_id     = eos_id
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden

        # ------ Stroke pre-encoder (Conv1d) ------
        # Input per stroke: (B, N, stroke_dim) treated as (B*N, 1, stroke_dim)
        self.stroke_encoder = nn.Sequential(
            nn.Conv1d(1, conv_hidden, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),   # (B*N, conv_hidden, 1)
        )
        # After squeeze: each stroke -> conv_hidden-dim vector

        # ------ Bidirectional GRU encoder ------
        # Forward and backward GRU cells over stroke sequence
        self.enc_fwd = GRUCell(conv_hidden, enc_hidden)
        self.enc_bwd = GRUCell(conv_hidden, enc_hidden)

        # Project concatenated bi-directional hidden to decoder hidden size
        self.enc2dec = nn.Linear(enc_hidden * 2, dec_hidden)

        # ------ Decoder ------
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dec_gru = GRUCell(embed_dim, dec_hidden)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dec_hidden, vocab_size)

    # --------------------------------------------------
    def _encode(
        self,
        strokes: torch.Tensor,          # (B, N, stroke_dim)
        strokes_lengths: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """
        Encode stroke sequences into a single context vector.

        Returns
        -------
        h_dec : (B, dec_hidden)  -- initial decoder hidden state
        enc_out : (B, N, enc_hidden*2) -- all encoder hidden states
        """
        B, N, D = strokes.shape

        # -- Conv pre-encoding: compress each stroke independently --
        x = strokes.view(B * N, 1, D)           # (B*N, 1, 128)
        x = self.stroke_encoder(x)               # (B*N, conv_hidden, 1)
        x = x.squeeze(-1)                        # (B*N, conv_hidden)
        x = x.view(B, N, -1)                    # (B, N, conv_hidden)

        # -- Forward GRU pass --
        h_fwd = torch.zeros(B, self.enc_hidden, device=strokes.device)
        fwd_states = []
        for t in range(N):
            h_fwd = self.enc_fwd(x[:, t, :], h_fwd)
            fwd_states.append(h_fwd)             # each: (B, enc_hidden)

        # -- Backward GRU pass --
        h_bwd = torch.zeros(B, self.enc_hidden, device=strokes.device)
        bwd_states = []
        for t in reversed(range(N)):
            h_bwd = self.enc_bwd(x[:, t, :], h_bwd)
            bwd_states.insert(0, h_bwd)

        # Stack all hidden states: (B, N, enc_hidden*2)
        fwd_out = torch.stack(fwd_states, dim=1)  # (B, N, enc_hidden)
        bwd_out = torch.stack(bwd_states, dim=1)  # (B, N, enc_hidden)
        enc_out = torch.cat([fwd_out, bwd_out], dim=-1)  # (B, N, enc_hidden*2)

        # Use final valid hidden states (respecting actual lengths)
        # For simplicity use last fwd + first bwd as context
        # Index the actual last stroke for each sequence in batch
        idx = (strokes_lengths - 1).clamp(min=0).long()  # (B,)
        h_fwd_last = fwd_states[-1]              # fallback: last timestep
        # Gather per-sequence last valid hidden
        idx_exp = idx.view(B, 1, 1).expand(B, 1, self.enc_hidden)
        h_fwd_last = fwd_out.gather(1, idx_exp).squeeze(1)  # (B, enc_hidden)
        h_bwd_first = bwd_out[:, 0, :]           # (B, enc_hidden)

        context = torch.cat([h_fwd_last, h_bwd_first], dim=-1)  # (B, enc_hidden*2)
        h_dec   = torch.tanh(self.enc2dec(context))              # (B, dec_hidden)

        return h_dec, enc_out

    # --------------------------------------------------
    def forward(
        self,
        strokes:               torch.Tensor,        # (B, N, stroke_dim)
        strokes_lengths:       torch.Tensor,        # (B,)
        target_tokens:         torch.Tensor = None, # (B, T)
        teacher_forcing_ratio: float        = 0.5,
    ) -> torch.Tensor:
        """
        Forward pass with optional teacher forcing.

        Returns
        -------
        logits : (B, T, vocab_size)
        """
        B = strokes.shape[0]
        device = strokes.device

        h, _ = self._encode(strokes, strokes_lengths)  # (B, dec_hidden)

        T = target_tokens.shape[1] if target_tokens is not None else self.max_len

        # Start token for every sequence in batch
        token = torch.full((B,), self.bos_id, dtype=torch.long, device=device)

        logits_list = []

        for t in range(T):
            emb    = self.embed(token)                  # (B, embed_dim)
            emb    = self.dropout(emb)
            h      = self.dec_gru(emb, h)              # (B, dec_hidden)
            logit  = self.out_proj(h)                  # (B, vocab_size)
            logits_list.append(logit)

            # Teacher forcing: feed ground-truth token with probability
            if (target_tokens is not None
                    and t + 1 < T
                    and torch.rand(1).item() < teacher_forcing_ratio):
                token = target_tokens[:, t]
            else:
                token = logit.argmax(dim=-1)           # greedy

        return torch.stack(logits_list, dim=1)         # (B, T, vocab_size)

    # --------------------------------------------------
    @torch.no_grad()
    def greedy_decode(
        self,
        strokes:         torch.Tensor,  # (B, N, stroke_dim)
        strokes_lengths: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """
        Greedy autoregressive decoding at inference time.

        Returns
        -------
        tokens : (B, max_len)
        """
        B      = strokes.shape[0]
        device = strokes.device

        h, _  = self._encode(strokes, strokes_lengths)

        token  = torch.full((B,), self.bos_id, dtype=torch.long, device=device)
        tokens = []

        for _ in range(self.max_len):
            emb   = self.embed(token)
            h     = self.dec_gru(emb, h)
            logit = self.out_proj(h)
            token = logit.argmax(dim=-1)
            tokens.append(token)

        return torch.stack(tokens, dim=1)  # (B, max_len)

    # --------------------------------------------------
    @torch.no_grad()
    def teacher_forced_cer(
        self,
        strokes:         torch.Tensor,  # (B, N, stroke_dim)
        strokes_lengths: torch.Tensor,  # (B,)
        target_tokens:   torch.Tensor,  # (B, T)
    ) -> torch.Tensor:
        """
        Teacher-forced CER: at each step feed ground-truth token,
        count prediction errors, normalise by sequence length.

        Returns
        -------
        cer : (B,)  per-sequence character error rate
        """
        B, T   = target_tokens.shape
        device = strokes.device

        h, _  = self._encode(strokes, strokes_lengths)
        token  = torch.full((B,), self.bos_id, dtype=torch.long, device=device)

        errors = torch.zeros(B, device=device)
        lengths = torch.zeros(B, device=device)

        for t in range(T):
            emb   = self.embed(token)
            h     = self.dec_gru(emb, h)
            logit = self.out_proj(h)
            pred  = logit.argmax(dim=-1)        # (B,)

            gt = target_tokens[:, t]            # ground-truth at step t

            # Only count positions that are not padding
            not_pad = (gt != self.pad_id)
            errors  += ((pred != gt) & not_pad).float()
            lengths += not_pad.float()

            # Feed ground-truth (teacher forcing)
            token = gt

        # CER per sequence
        cer = errors / lengths.clamp(min=1)
        return cer


# -------------------------------------------------------
# Training function
# -------------------------------------------------------
def part2_train_model(
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
    Train the InfixRNN encoder-decoder.

    Uses:
        - Adam optimiser
        - CrossEntropyLoss (padding tokens ignored)
        - Gradient clipping (max_norm=1.0)
        - Teacher forcing (ratio annealed from 0.8 -> 0.3)
        - Best checkpoint saved by validation LA
    """
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=model.pad_id)
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
        best_val_acc = ckpt.get("val_acc",  0.0)
        start_epoch  = ckpt.get("epoch",    0) + 1
        history      = ckpt.get("history",  history)
        print(f"Resumed from epoch {start_epoch-1}, best_val_acc={best_val_acc:.4f}")
    elif resume:
        print(f"Warning: no checkpoint at {checkpoint_path}. Starting fresh.")

    for epoch in range(start_epoch, start_epoch + num_epochs):

        # Anneal teacher forcing ratio linearly from 0.8 to 0.3
        tf_ratio = max(0.3, 0.8 - (epoch - 1) * 0.5 / max(num_epochs - 1, 1))

        # ===== TRAIN =====
        model.train()
        run_loss = 0.0
        run_la   = 0.0
        n_batches = 0

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch}/{start_epoch+num_epochs-1} [Train]",
                    leave=True)
        for batch in pbar:
            X, X_lens, Y = [b.to(device) for b in batch]

            optimizer.zero_grad()
            # Y input to decoder: all tokens except last  (B, T-1)
            # Y target          : all tokens except first (B, T-1)
            logits = model(X, X_lens, Y[:, :-1],
                           teacher_forcing_ratio=tf_ratio)  # (B, T-1, V)

            # Flatten for loss
            B, T, V = logits.shape
            loss = criterion(
                logits.reshape(B * T, V),
                Y[:, 1:].reshape(B * T),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            run_loss  += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", tf=f"{tf_ratio:.2f}")

        train_loss = run_loss / max(n_batches, 1)

        # ===== VALIDATE =====
        model.eval()
        val_loss  = 0.0
        val_la    = 0.0
        n_val     = 0

        with torch.no_grad():
            vbar = tqdm(valid_loader,
                        desc=f"Epoch {epoch}/{start_epoch+num_epochs-1} [Valid]",
                        leave=True)
            for batch in vbar:
                X, X_lens, Y = [b.to(device) for b in batch]

                logits = model(X, X_lens, Y[:, :-1],
                               teacher_forcing_ratio=0.0)  # (B, T-1, V)
                B, T, V = logits.shape
                loss = criterion(
                    logits.reshape(B * T, V),
                    Y[:, 1:].reshape(B * T),
                )

                # Greedy decode for LA
                Y_hat = model.greedy_decode(X, X_lens)
                la    = batch_LA(Y, Y_hat, model.pad_id,
                                 model.bos_id, model.eos_id)

                val_loss += loss.item()
                val_la   += la
                n_val    += 1
                vbar.set_postfix(val_loss=f"{loss.item():.4f}",
                                 val_LA=f"{la:.4f}")

        val_loss = val_loss / max(n_val, 1)
        val_la   = val_la   / max(n_val, 1)

        # Use LA as the "accuracy" metric for checkpoint selection
        history["train_loss"].append(train_loss)
        history["train_acc"].append(0.0)       # placeholder (LA not computed on train)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_la)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f}  val_LA={val_la:.4f}")

        if val_la > best_val_acc:
            best_val_acc = val_la
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc":          best_val_acc,
                    "history":          history,
                },
                checkpoint_path,
            )
            print(f"  -> Saved best checkpoint  val_LA={best_val_acc:.4f}  [{checkpoint_path}]")

    return history


# -------------------------------------------------------
# DO NOT MODIFY – Test function for evaluation notebook
# -------------------------------------------------------
def part2_test_model(
    model:           nn.Module,
    test_loader:     DataLoader,
    checkpoint_path,
    device,
):
    """Load checkpoint and evaluate LA + forced CER on test set."""
    print(f"Using device: {device}")

    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint  epoch={ckpt['epoch']}  val_acc={ckpt['val_acc']:.4f}")

    model.to(device)
    model.eval()

    total_la  = 0.0
    total_cer = 0.0
    n_batches = 0

    pbar = tqdm(test_loader, desc="[Test]", leave=True)
    for batch in pbar:
        X, X_lens, Y = [b.to(device) for b in batch]

        Y_hat     = model.greedy_decode(X, X_lens)
        batch_la  = batch_LA(Y, Y_hat, model.pad_id, model.bos_id, model.eos_id)
        batch_cer = model.teacher_forced_cer(X, X_lens, Y).mean().item()

        total_la  += batch_la
        total_cer += batch_cer
        n_batches += 1

        pbar.set_postfix(LA=f"{batch_la:.4f}", CER=f"{batch_cer:.4f}")

    avg_la  = total_la  / max(n_batches, 1)
    avg_cer = total_cer / max(n_batches, 1)
    return avg_la, avg_cer

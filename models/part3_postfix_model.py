"""
part3_postfix_model.py — Encoder-Decoder Transformer for postfix RPN.
Manual implementation: no nn.Transformer / nn.MultiheadAttention used.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from collections import Counter, OrderedDict
from scripts.utils import Vocab, batch_LA


# ── Vocabulary ────────────────────────────────────────────────────────────────
def part3_build_vocab() -> Vocab:
    token_counter = Counter({
        '0':1,'1':1,'2':1,'3':1,'4':1,'5':1,'6':1,'7':1,'8':1,'9':1,
        '+':1,'-':1,'*':1,'/':1,'.':1,'(':1,')':1,'=':1,',':1,
    })
    od = OrderedDict(sorted(token_counter.items(),
                            key=lambda x: x[1], reverse=True))
    v  = Vocab(od, specials=['<unk>','<pad>','<bos>','<eos>'])
    v.set_default_index(v['<unk>'])
    return v


# ── Model args (<50k params) ──────────────────────────────────────────────────
def part3_build_model_args(vocab) -> dict:
    return dict(
        vocab_size   = len(vocab),
        max_len      = 64,
        pad_id       = 1,
        bos_id       = 2,
        eos_id       = 3,
        stroke_dim   = 128,
        conv_hidden  = 8,
        d_model      = 32,
        n_heads      = 2,
        n_enc_layers = 2,
        n_dec_layers = 2,
        d_ff         = 64,
        dropout      = 0.1,
    )


# ── Factory ───────────────────────────────────────────────────────────────────
def part3_postfix_recognition_model(**kw) -> nn.Module:
    return PostfixTransformer(**kw)


# ── Positional Encoding ───────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


# ── Multi-Head Attention ──────────────────────────────────────────────────────
class MHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h  = n_heads
        self.dk = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        def split(W, x):
            return W(x).view(B, -1, self.h, self.dk).transpose(1, 2)
        Q, K, V = split(self.Wq, q), split(self.Wk, k), split(self.Wv, v)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dk)
        if mask is not None:
            scores = scores + mask
        out = torch.matmul(F.softmax(scores, dim=-1), V)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.h * self.dk)
        return self.Wo(out)


# ── Encoder Layer ─────────────────────────────────────────────────────────────
class EncLayer(nn.Module):
    def __init__(self, d, h, ff, drop):
        super().__init__()
        self.attn = MHA(d, h)
        self.ff   = nn.Sequential(
            nn.Linear(d, ff), nn.ReLU(inplace=True), nn.Linear(ff, d))
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.dp = nn.Dropout(drop)

    def forward(self, x):
        x = self.n1(x + self.dp(self.attn(x, x, x)))
        x = self.n2(x + self.dp(self.ff(x)))
        return x


# ── Decoder Layer ─────────────────────────────────────────────────────────────
class DecLayer(nn.Module):
    def __init__(self, d, h, ff, drop):
        super().__init__()
        self.sa = MHA(d, h)
        self.ca = MHA(d, h)
        self.ff = nn.Sequential(
            nn.Linear(d, ff), nn.ReLU(inplace=True), nn.Linear(ff, d))
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.n3 = nn.LayerNorm(d)
        self.dp = nn.Dropout(drop)

    def forward(self, x, enc, causal_mask):
        x = self.n1(x + self.dp(self.sa(x, x, x, causal_mask)))
        x = self.n2(x + self.dp(self.ca(x, enc, enc)))
        x = self.n3(x + self.dp(self.ff(x)))
        return x


# ── Full Model ────────────────────────────────────────────────────────────────
class PostfixTransformer(nn.Module):

    def __init__(self, vocab_size, max_len, pad_id, bos_id, eos_id,
                 stroke_dim=128, conv_hidden=8, d_model=32, n_heads=2,
                 n_enc_layers=2, n_dec_layers=2, d_ff=64, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len    = max_len
        self.pad_id     = pad_id
        self.bos_id     = bos_id
        self.eos_id     = eos_id
        self.d_model    = d_model

        self.conv = nn.Sequential(
            nn.Conv1d(1, conv_hidden, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(conv_hidden, d_model)

        self.enc_pe = PositionalEncoding(d_model, dropout)
        self.enc    = nn.ModuleList([
            EncLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_enc_layers)])

        self.emb    = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.dec_pe = PositionalEncoding(d_model, dropout)
        self.dec    = nn.ModuleList([
            DecLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_dec_layers)])
        self.out = nn.Linear(d_model, vocab_size)
        self.out.weight = self.emb.weight

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _causal_mask(self, T, device):
        m = torch.triu(torch.ones(T, T, device=device), diagonal=1)
        return m.masked_fill(m.bool(), float('-inf')).unsqueeze(0).unsqueeze(0)

    def _encode(self, strokes):
        B, N, D = strokes.shape
        x = self.conv(strokes.view(B * N, 1, D)).squeeze(-1)
        x = self.proj(x.view(B, N, -1)) * math.sqrt(self.d_model)
        x = self.enc_pe(x)
        for layer in self.enc:
            x = layer(x)
        return x

    def forward(self, strokes, lengths, tgt):
        """
        strokes : (B, N, 128)
        lengths : (B,)
        tgt     : (B, T)  Long tensor
        returns   (B, T-1, V)
        """
        enc_out = self._encode(strokes)

        # CRITICAL: cast to long to handle any bool/int8 tensors from dataloader
        dec_in = tgt[:, :-1].long()
        T      = dec_in.size(1)
        x      = self.emb(dec_in) * math.sqrt(self.d_model)
        x      = self.dec_pe(x)
        caus   = self._causal_mask(T, strokes.device)

        for layer in self.dec:
            x = layer(x, enc_out, caus)

        return self.out(x)

    @torch.no_grad()
    def greedy_decode(self, strokes, lengths):
        B      = strokes.size(0)
        device = strokes.device
        enc    = self._encode(strokes)
        tokens = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)

        for _ in range(self.max_len):
            T    = tokens.size(1)
            x    = self.emb(tokens.long()) * math.sqrt(self.d_model)
            x    = self.dec_pe(x)
            caus = self._causal_mask(T, device)
            for layer in self.dec:
                x = layer(x, enc, caus)
            nxt    = self.out(x[:, -1, :]).argmax(-1, keepdim=True)
            tokens = torch.cat([tokens, nxt], dim=1)
            if (nxt.squeeze(-1) == self.eos_id).all():
                break

        return tokens[:, 1:]

    @torch.no_grad()
    def teacher_forced_cer(self, strokes, lengths, tgt):
        preds   = self.forward(strokes, lengths, tgt).argmax(-1)
        gt      = tgt[:, 1:].long()
        not_pad = gt != self.pad_id
        errors  = ((preds != gt) & not_pad).float().sum(1)
        lens    = not_pad.float().sum(1).clamp(min=1)
        return errors / lens


# ── Training ──────────────────────────────────────────────────────────────────
def part3_train_model(model, train_loader, valid_loader,
                      num_epochs, lr=5e-4, device="cpu",
                      save_path=None, resume=False, warmup_steps=200):

    model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    crit  = nn.CrossEntropyLoss(ignore_index=model.pad_id)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(max(s, 1) / max(warmup_steps, 1), 1.0))

    ckpt_path = Path(save_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    history  = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    best_val = 0.0
    start_ep = 1

    if resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        best_val = ck.get("val_acc", 0.0)
        start_ep = ck.get("epoch",   0) + 1
        history  = ck.get("history", history)
        print(f"Resumed from epoch {start_ep-1}, best={best_val:.4f}")

    for epoch in range(start_ep, start_ep + num_epochs):

        # Train
        model.train()
        tl, nb = 0.0, 0
        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch}/{start_ep+num_epochs-1} [Train]",
                    leave=True)
        for batch in pbar:
            X, Y_in, X_lens, Y = [b.to(device) for b in batch]
            # Cast to long - fixes CPUBoolType error
            X     = X.float()
            Y     = Y.long()
            X_lens = X_lens.long()

            opt.zero_grad()
            lg = model(X, X_lens, Y)
            B, T, V = lg.shape
            loss = crit(lg.reshape(B*T, V), Y[:, 1:].reshape(B*T))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += loss.item()
            nb += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validate
        model.eval()
        vl, vla, nv = 0.0, 0.0, 0
        with torch.no_grad():
            vbar = tqdm(valid_loader,
                        desc=f"Epoch {epoch}/{start_ep+num_epochs-1} [Valid]",
                        leave=True)
            for batch in vbar:
                X, Y_in, X_lens, Y = [b.to(device) for b in batch]
                X      = X.float()
                Y      = Y.long()
                X_lens = X_lens.long()

                lg = model(X, X_lens, Y)
                B, T, V = lg.shape
                loss = crit(lg.reshape(B*T, V), Y[:, 1:].reshape(B*T))

                Yh = model.greedy_decode(X, X_lens)
                Ty = Y.size(1); Th = Yh.size(1)
                if Th < Ty:
                    Yh = F.pad(Yh, (0, Ty-Th), value=model.pad_id)
                else:
                    Yh = Yh[:, :Ty]

                la = batch_LA(Y, Yh, model.pad_id, model.bos_id, model.eos_id)
                vl  += loss.item()
                vla += la
                nv  += 1
                vbar.set_postfix(val_loss=f"{loss.item():.4f}", val_LA=f"{la:.4f}")

        tl  /= max(nb, 1)
        vl  /= max(nv, 1)
        vla /= max(nv, 1)

        history["train_loss"].append(tl)
        history["train_acc"].append(0.0)
        history["val_loss"].append(vl)
        history["val_acc"].append(vla)

        print(f"Epoch {epoch}: train_loss={tl:.4f} | val_loss={vl:.4f}  val_LA={vla:.4f}")

        if vla >= best_val:
            best_val = vla
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_acc":          best_val,
                "history":          history,
            }, ckpt_path)
            print(f"  -> Saved best  val_LA={best_val:.4f}")

    return history


# ── Evaluation ────────────────────────────────────────────────────────────────
def part3_test_model(model, test_loader, checkpoint_path, device):
    print(f"Using device: {device}")
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
    ck = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    print(f"Loaded checkpoint  epoch={ck['epoch']}  val_acc={ck['val_acc']:.4f}")
    model.to(device)
    model.eval()

    total_la, total_cer, nb = 0.0, 0.0, 0
    for batch in tqdm(test_loader, desc="[Test]", leave=True):
        X, Y_in, X_lens, Y = [b.to(device) for b in batch]
        X      = X.float()
        Y      = Y.long()
        X_lens = X_lens.long()

        Yh = model.greedy_decode(X, X_lens)
        Ty = Y.size(1); Th = Yh.size(1)
        if Th < Ty:
            Yh = F.pad(Yh, (0, Ty-Th), value=model.pad_id)
        else:
            Yh = Yh[:, :Ty]

        total_la  += batch_LA(Y, Yh, model.pad_id, model.bos_id, model.eos_id)
        total_cer += model.teacher_forced_cer(X, X_lens, Y).mean().item()
        nb        += 1

    return total_la / max(nb, 1), total_cer / max(nb, 1)

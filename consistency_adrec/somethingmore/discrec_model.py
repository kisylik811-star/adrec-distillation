"""
DiscRec: Discrete Diffusion Model for Sequential Recommendation.

Model components:
  1. Item embedding layer (shared between input and output).
  2. History encoder (causal Transformer, SASRec-style).
  3. Time embedding (sinusoidal + MLP).
  4. Score network (Transformer that predicts categorical distribution over items).

Forward process: absorbing diffusion - target token replaced by [MASK] with
probability (1 - alpha_t).
Reverse process: one-step argmax over predicted distribution at t = T.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Time embedding
# ----------------------------------------------------------------------
def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal time embedding (as in DDPM). t: (B,) long tensor."""
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device).float() / half
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb  # (B, dim)


# ----------------------------------------------------------------------
# Transformer block (standard pre-LN)
# ----------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Pre-LN
        h = self.ln1(x)
        attn_out, _ = self.attn(
            h, h, h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)
        x = x + self.ff(self.ln2(x))
        return x


# ----------------------------------------------------------------------
# History encoder (SASRec-style causal Transformer)
# ----------------------------------------------------------------------
class HistoryEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_layers: int,
        max_len: int,
        dropout: float,
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.max_len = max_len

    def forward(self, history_emb, padding_mask):
        """
        history_emb: (B, L, d)
        padding_mask: (B, L) bool - True where PADDED (ignored)
        Returns: (B, d) - last non-pad position representation
        """
        B, L, d = history_emb.shape
        positions = torch.arange(L, device=history_emb.device).unsqueeze(0).expand(B, L)
        x = history_emb + self.pos_emb(positions)

        causal = torch.triu(
            torch.full((L, L), float("-inf"), device=x.device), diagonal=1
        )

        for block in self.blocks:
            x = block(x, attn_mask=causal, key_padding_mask=padding_mask)

        x = self.ln_final(x)

        # Take last non-pad position per sequence
        valid = (~padding_mask).long()
        last_idx = valid.sum(dim=1) - 1
        last_idx = last_idx.clamp(min=0)
        out = x[torch.arange(B, device=x.device), last_idx]
        return out


# ----------------------------------------------------------------------
# Score network
# ----------------------------------------------------------------------
class ScoreNetwork(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, num_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(3 * d_model, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, h, x_t_emb, t_emb):
        """
        h:       (B, d)
        x_t_emb: (B, d)
        t_emb:   (B, d)
        Returns: (B, d)
        """
        z = torch.cat([h, x_t_emb, t_emb], dim=-1)
        z = self.input_proj(z).unsqueeze(1)
        for block in self.blocks:
            z = block(z)
        z = self.ln_final(z).squeeze(1)
        z = self.out_proj(z)
        return z


# ----------------------------------------------------------------------
# Full DiscRec model
# ----------------------------------------------------------------------
class DiscRec(nn.Module):
    """
    Token IDs:
      0       : PAD
      1..N    : real items (N = num_items)
      N+1     : [MASK]
    """

    def __init__(
        self,
        num_items: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        num_layers_history: int = 2,
        num_layers_score: int = 2,
        dropout: float = 0.1,
        max_len: int = 50,
        T: int = 50,
    ):
        super().__init__()
        self.num_items = num_items
        self.pad_id = 0
        self.mask_id = num_items + 1
        self.vocab_size = num_items + 2
        self.T = T
        self.d_model = d_model
        self.max_len = max_len

        self.item_emb = nn.Embedding(self.vocab_size, d_model, padding_idx=0)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.history_encoder = HistoryEncoder(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            num_layers=num_layers_history, max_len=max_len, dropout=dropout,
        )

        self.score_network = ScoreNetwork(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            num_layers=num_layers_score, dropout=dropout,
        )

        # Linear absorbing schedule: alpha_t = 1 - t/T
        alphas = 1.0 - torch.arange(1, T + 1).float() / T
        alphas = alphas.clamp(min=1e-4, max=1.0 - 1e-4)
        self.register_buffer("alphas", alphas)

    def encode_history(self, history_ids):
        padding_mask = (history_ids == self.pad_id)
        h_emb = self.item_emb(history_ids)
        return self.history_encoder(h_emb, padding_mask)

    def time_features(self, t):
        sin_emb = sinusoidal_time_embedding(t, self.d_model)
        return self.time_mlp(sin_emb)

    def forward_logits(self, history_ids, x_t, t):
        h = self.encode_history(history_ids)
        x_t_emb = self.item_emb(x_t)
        t_emb = self.time_features(t)
        z = self.score_network(h, x_t_emb, t_emb)
        # Dot product with item embeddings for items 1..N
        item_table = self.item_emb.weight[1:self.num_items + 1]
        logits = z @ item_table.T
        return logits

    def sample_x_t(self, x_0, t):
        alpha_t = self.alphas[t - 1]
        rand = torch.rand_like(alpha_t)
        masked = rand > alpha_t
        x_t = torch.where(
            masked,
            torch.full_like(x_0, self.mask_id),
            x_0,
        )
        return x_t, masked

    def compute_loss(self, history_ids, x_0):
        B = x_0.size(0)
        device = x_0.device

        t = torch.randint(1, self.T + 1, (B,), device=device)
        x_t, mask = self.sample_x_t(x_0, t)

        logits = self.forward_logits(history_ids, x_t, t)
        target = x_0 - 1
        ce = F.cross_entropy(logits, target, reduction="none")

        alpha_t = self.alphas[t - 1]
        w = 1.0 / (1.0 - alpha_t + 1e-8)

        mask_f = mask.float()
        denom = mask_f.sum().clamp(min=1.0)
        loss = (mask_f * w * ce).sum() / denom
        return loss

    @torch.no_grad()
    def predict_one_step(self, history_ids):
        B = history_ids.size(0)
        device = history_ids.device
        x_T = torch.full((B,), self.mask_id, dtype=torch.long, device=device)
        t = torch.full((B,), self.T, dtype=torch.long, device=device)
        logits = self.forward_logits(history_ids, x_T, t)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_multi_step(self, history_ids, num_steps: int):
        B = history_ids.size(0)
        device = history_ids.device
        x_t = torch.full((B,), self.mask_id, dtype=torch.long, device=device)
        num_steps = max(num_steps, 1)
        steps = torch.linspace(self.T, 1, num_steps + 1).long().to(device)

        for i in range(num_steps):
            t_curr = steps[i].expand(B)
            logits = self.forward_logits(history_ids, x_t, t_curr)
            x_0_pred = logits.argmax(dim=-1) + 1
            if i < num_steps - 1:
                t_next = steps[i + 1].expand(B)
                x_t, _ = self.sample_x_t(x_0_pred, t_next)

        return F.softmax(logits, dim=-1)
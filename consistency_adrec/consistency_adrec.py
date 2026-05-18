"""
Consistency distillation for ADRec — token-level CD with optional RCCD and
adaptive (min-SNR) weighting.

ADRec is a token-level diffusion sequential recommender: each position in the
sequence has its own independent timestep t_i, all positions denoise in
parallel through a shared `net`. This is preserved end-to-end in the student.

Key components:
  ConsistencyADRec
    Student diffusion module mirroring the teacher's AdRec. Reuses the
    teacher's noise schedule (frozen). `ag_encoder` and `net` are initialised
    from the teacher and finetuned via consistency distillation.

  ConsistencyADRecStudent
    Full student wrapper mirroring Att_Diffuse_model: item_embedding +
    hist_norm + embed_dropout + ConsistencyADRec.

Inference protocol (matches teacher.denoise_sample):
  - ag_encoder runs once on history (the only encoder pass; not duplicated
    across NFE steps).
  - At every step, positions 0..L-2 of x_t hold clean target embeddings
    (autoregressive teacher forcing on past targets, as in ADRec eval).
  - Only the last position is iteratively denoised. NFE=1 means one
    `net` forward pass; NFE>1 alternates denoise/re-noise on the last
    position only (Song et al. 2023 multi-step CM inference).

Loss components (all per-token, padding masked out):
  L_cons (consistency MSE, weighted by min-SNR γ=5 if adaptive_weighting=True)
    Online student f_theta(x_t, t)  vs  EMA student f_theta_ema(x_{t-1}, t-1).
    x_{t-1} is one deterministic DDIM step of the *teacher* from x_t.
  L_ce (next-token cross-entropy through item table)
    Standard ADRec ranking loss. Kept verbatim from the teacher.
  L_contrast (RCCD: ranking-aligned per-token InfoNCE)
    Anchor: student's predicted x_0 at position i.
    Positive: embedding of the true next item at position i.
    Negatives: K items sampled uniformly from the catalog, shared across the
    batch (cheap, ~sampled softmax). Optional (β=0 disables it).

References:
  Song et al., "Consistency Models", ICML 2023.
  Hang et al., "Efficient Diffusion Training via Min-SNR Weighting Strategy",
    ICCV 2023.
"""
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Imports from the adrec_original source tree. The caller must arrange
# PYTHONPATH so these are visible (see README).
from adrec import AdRec, DenoisedModel
from common import LayerNorm, TransformerEncoder
from utils import _extract_into_tensor


# ===================================================================
# ConsistencyADRec: student diffusion module (mirrors AdRec)
# ===================================================================

class ConsistencyADRec(nn.Module):
    """
    Student diffusion module trained via consistency distillation.

    Mirrors the teacher's AdRec: an `ag_encoder` for history conditioning and
    a `net` (DenoisedModel) for token-level denoising. Both are initialised
    from the teacher and finetuned. The noise schedule (betas, alphas_cumprod,
    ...) is reused verbatim — it is a fixed property of the teacher's
    diffusion process.
    """

    def __init__(self, teacher_adrec, args):
        super().__init__()
        # ----- Frozen schedule (just numpy arrays, not Parameters) -----
        self.betas = teacher_adrec.betas
        self.alphas_cumprod = teacher_adrec.alphas_cumprod
        self.alphas_cumprod_prev = teacher_adrec.alphas_cumprod_prev
        self.sqrt_alphas_cumprod = teacher_adrec.sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = teacher_adrec.sqrt_one_minus_alphas_cumprod
        self.posterior_mean_coef1 = teacher_adrec.posterior_mean_coef1
        self.posterior_mean_coef2 = teacher_adrec.posterior_mean_coef2
        self.posterior_variance = teacher_adrec.posterior_variance
        self.num_timesteps = teacher_adrec.num_timesteps
        self.rescale_timesteps = teacher_adrec.rescale_timesteps
        self.independent_diffusion = teacher_adrec.independent_diffusion
        # We deliberately do NOT support CFG/geodesic in the student (per
        # user spec: assume teacher trained with defaults cfg_scale=1,
        # geodesic=False).
        self.cfg_scale = 1.0
        self.geodesic = False

        # ----- Trainable copies of teacher's encoder and denoiser -----
        # We use a fresh args object and load state_dict from teacher to
        # guarantee architectural identity.
        self.ag_encoder = TransformerEncoder(args, num_blocks=2, norm_first=False)
        self.ag_encoder.load_state_dict(teacher_adrec.ag_encoder.state_dict())

        self.net = DenoisedModel(args)
        self.net.load_state_dict(teacher_adrec.net.state_dict())

    # ----- Forward-noise helpers (verbatim from AdRec, no geodesic) -----

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_sample(self, x_start, t, noise, mask=None):
        """x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*noise.

        Works with t of shape (B, L) and x_start of shape (B, L, H).
        """
        x_t = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )
        if mask is None:
            return x_t
        mask_b = mask.unsqueeze(-1).expand_as(x_start)
        return torch.where(mask_b == 0, x_start, x_t)

    # ----- Teacher one-step DDIM (used to build consistency targets) -----

    @torch.no_grad()
    def teacher_ddim_step(self, teacher_adrec, item_rep, x_t_high,
                          t_high, t_low, mask_seq, mask_tag):
        """
        Deterministic DDIM step from t_high down to t_low using the frozen
        teacher. Operates per-token: t_high, t_low are (B, L) int tensors.

        x_0_pred = teacher.net(history, x_t_high, t_high)
        eps_pred = (x_t_high - sqrt(alpha_t)*x_0) / sqrt(1-alpha_t)
        x_t_low  = sqrt(alpha_{t-1})*x_0 + sqrt(1-alpha_{t-1})*eps_pred
        """
        x_0_t = teacher_adrec.net(
            item_rep, x_t_high, teacher_adrec._scale_timesteps(t_high),
            mask_seq, mask_tag,
        )

        sa_h = _extract_into_tensor(teacher_adrec.sqrt_alphas_cumprod,
                                    t_high, x_t_high.shape)
        som_h = _extract_into_tensor(teacher_adrec.sqrt_one_minus_alphas_cumprod,
                                     t_high, x_t_high.shape)
        sa_l = _extract_into_tensor(teacher_adrec.sqrt_alphas_cumprod,
                                    t_low, x_t_high.shape)
        som_l = _extract_into_tensor(teacher_adrec.sqrt_one_minus_alphas_cumprod,
                                     t_low, x_t_high.shape)

        # Numerical safety: som_h could be 0 at t=0 only. We sample
        # t_high >= 1 for valid positions; padding positions get t_high=0
        # but those are masked out of the loss. Clamp to be safe.
        eps_pred = (x_t_high - sa_h * x_0_t) / som_h.clamp(min=1e-8)
        x_t_low = sa_l * x_0_t + som_l * eps_pred
        return x_t_low

    # ----- Single-pass prediction (the "consistency function") -----

    def predict_x0(self, item_rep, x_t, t, mask_seq, mask_tag):
        """f_theta(x_t, t, history) -> predicted x_0. Shape (B, L, H)."""
        return self.net(item_rep, x_t, self._scale_timesteps(t),
                        mask_seq, mask_tag)

    # ----- Inference: ADRec-style denoising of the last position only -----

    @torch.no_grad()
    def sample(self, item_emb_history, tgt_emb_known, mask_seq, mask_tag,
               num_steps=1):
        """
        Generate x_0 for the next-item slot at position L-1.

        Arguments
        ---------
        item_emb_history : (B, L, H)
            Output of item_embedding + dropout + LayerNorm on the history
            sequence. ag_encoder is applied INSIDE this function so callers
            don't need to know about it.
        tgt_emb_known : (B, L, H)
            Embeddings of the autoregressive context. Positions 0..L-2 hold
            CLEAN ground-truth-shifted-by-one item embeddings (these are the
            past observed targets in eval mode). Position L-1 is ignored —
            it is replaced by pure noise.
        mask_seq, mask_tag : (B, L)
            Padding masks.
        num_steps : int
            NFE. 1 = canonical CM inference, single forward pass.
            >1 = multi-step alternating denoise/re-noise on the last position.
        """
        device = item_emb_history.device
        B, L, H = item_emb_history.shape
        T = self.num_timesteps

        # Encode history ONCE. This cost is shared with the teacher, not
        # scaled by NFE — speedup comes from `net`, not `ag_encoder`.
        item_rep = self.ag_encoder(item_emb_history, mask_seq)

        # Build x_t: clean context at 0..L-2, pure Gaussian at L-1.
        noise_last = torch.randn(B, 1, H, device=device)
        x_t = torch.cat([tgt_emb_known[:, :-1], noise_last], dim=1)

        if num_steps == 1:
            t = torch.zeros(B, L, dtype=torch.long, device=device)
            t[:, -1] = T - 1
            x_0 = self.predict_x0(item_rep, x_t, t, mask_seq, mask_tag)
            return x_0

        # Multi-step CM inference: alternating denoise on the last position
        # only, re-noising the predicted x_0 to the next timestep.
        ts = np.linspace(T - 1, 1, num_steps).round().astype(int)
        x_0 = None
        for i, t_val in enumerate(ts):
            t = torch.zeros(B, L, dtype=torch.long, device=device)
            t[:, -1] = int(t_val)
            x_0 = self.predict_x0(item_rep, x_t, t, mask_seq, mask_tag)
            if i < len(ts) - 1:
                t_next_scalar = int(ts[i + 1])
                t_next = torch.full((B, 1), t_next_scalar, dtype=torch.long,
                                    device=device)
                # Re-noise only the last position via q_sample.
                noise = torch.randn_like(x_0[:, -1:])
                sa = _extract_into_tensor(self.sqrt_alphas_cumprod,
                                          t_next, x_0[:, -1:].shape)
                som = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                           t_next, x_0[:, -1:].shape)
                x_t_last = sa * x_0[:, -1:] + som * noise
                x_t = torch.cat([tgt_emb_known[:, :-1], x_t_last], dim=1)
        return x_0


# ===================================================================
# ConsistencyADRecStudent: full student wrapper
# ===================================================================

class ConsistencyADRecStudent(nn.Module):
    """
    Full student model: item_embedding + LayerNorm + ConsistencyADRec.

    Mirrors Att_Diffuse_model with the teacher's `diffu = AdRec` replaced
    by `diffu_student = ConsistencyADRec`. Item embeddings and the LayerNorm
    are initialised from the teacher and trained jointly.

    Loss components are returned separately by `consistency_loss` so the
    training loop can log and combine them (with optional ablation
    switches: --use_rccd / --use_adaptive_weighting).
    """

    def __init__(self, teacher_model, args, ema_decay=0.999):
        super().__init__()
        self.args = args
        self.ema_decay = ema_decay
        self.emb_dim = args.hidden_size
        self.item_num = args.item_num  # ADRec convention: item_num WITHOUT +1
        self.K_neg = getattr(args, 'rccd_num_neg', 128)

        # --- Item embedding (copied from teacher, then trained) ---
        # Teacher used padding_idx=0; preserve it.
        self.item_embedding = nn.Embedding(self.item_num + 1, self.emb_dim,
                                           padding_idx=0)
        self.item_embedding.load_state_dict(
            teacher_model.item_embedding.state_dict()
        )

        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.hist_norm = LayerNorm(args.hidden_size, eps=1e-12)
        self.hist_norm.load_state_dict(teacher_model.hist_norm.state_dict())

        # --- Online + EMA student diffusion modules ---
        self.diffu_student = ConsistencyADRec(teacher_model.diffu, args)
        self.diffu_student_ema = copy.deepcopy(self.diffu_student)
        for p in self.diffu_student_ema.parameters():
            p.requires_grad = False

        self.loss_ce = nn.CrossEntropyLoss(ignore_index=0)

    # ----- EMA update -----

    @torch.no_grad()
    def update_ema(self):
        for p_t, p_o in zip(self.diffu_student_ema.parameters(),
                            self.diffu_student.parameters()):
            p_t.data.mul_(self.ema_decay).add_(p_o.data,
                                               alpha=1.0 - self.ema_decay)

    # ----- History encoding (mirror Att_Diffuse_model preprocessing) -----

    def encode_history(self, sequence):
        """Embed -> dropout -> LayerNorm. Returns (item_emb, mask_seq)."""
        item_emb = self.item_embedding(sequence)
        item_emb = self.embed_dropout(item_emb)
        item_emb = self.hist_norm(item_emb)
        mask_seq = (sequence > 0).float()
        return item_emb, mask_seq

    # ----- Min-SNR per-token weight (Hang et al. 2023) -----

    def _min_snr_weight(self, t, gamma=5.0):
        """
        Per-token min-SNR weight: w(t) = min(SNR(t), gamma) / SNR(t)
        where SNR(t) = alpha_bar_t / (1 - alpha_bar_t).

        Equivalent form (numerically stabler at low SNR):
            w(t) = min(1, gamma / SNR(t))
                 = min(1, gamma * (1 - alpha_bar) / alpha_bar)

        t : LongTensor (B, L). Returns FloatTensor (B, L).
        """
        alpha_bar = (
            torch.from_numpy(self.diffu_student.alphas_cumprod)
            .to(t.device).float()
        )[t]  # (B, L)
        # Avoid division by zero at alpha_bar = 1 (i.e. t=0). We sample
        # t >= 1 for valid positions, but be defensive for padding.
        snr = alpha_bar / (1.0 - alpha_bar).clamp(min=1e-8)
        return torch.clamp(snr, max=gamma) / snr.clamp(min=1e-8)

    # ----- Main training loss -----

    def consistency_loss(self, sequence, target, teacher_diffu,
                         use_rccd=True, use_adaptive_weighting=True,
                         contrast_temperature=0.1):
        """
        Token-level Consistency Distillation for ADRec.

        Returns (cons_loss, ce_loss, contrast_loss). The training loop
        weights and sums them: loss = cons + ce + beta * contrast. The
        adaptive (min-SNR) weight is applied to L_cons only — it would not
        make sense for L_ce (no per-token timestep on the CE target) and
        would distort the contrastive temperature.

        Mathematical sketch
        -------------------
        For each user, for each valid position i in the sequence:
          1. Sample t_i ~ Uniform[1, T-1] independently (per-token).
          2. Forward-noise: x_t^i = q_sample(target_i, t_i, noise_i).
          3. Teacher one-step DDIM: x_{t-1}^i = ddim_step(teacher, x_t^i).
          4. Online student: y_high^i = f_theta(x_t^i, t_i, history).
          5. EMA student:    y_low^i  = f_theta_ema(x_{t-1}^i, t_i-1, history).
          6. L_cons^i = w_minsnr(t_i) * ||y_high^i - y_low^i||^2_2.
        Aggregate over valid positions (mask_tag==1) and average.

        L_ce  : standard per-token CE on y_high vs target IDs.
        L_rccd: per-token InfoNCE on normalised y_high against
                (positive: target_i embedding, negatives: K shared catalog
                 items uniformly sampled per batch).
        """
        device = sequence.device

        # ----- 1. Encode history (item_embedding -> dropout -> LN) -----
        item_emb, mask_seq = self.encode_history(sequence)
        item_rep = self.diffu_student.ag_encoder(item_emb, mask_seq)
        # For the teacher pass we MUST use the SAME ag_encoder output
        # because the consistency target should depend on the same context
        # the online student sees. Using teacher.ag_encoder here would
        # introduce a second moving target (the student's ag_encoder is
        # also learning). We therefore reuse the online student's
        # `item_rep` for both the online forward pass and the teacher's
        # DDIM target step. This is the standard CD recipe.

        # ----- 2. Target embeddings (no LayerNorm — matches teacher) -----
        # Teacher's Att_Diffuse_model uses raw item_embedding(tag) (no
        # embed_dropout / hist_norm) for the diffusion target. Preserve it.
        tgt_emb = self.item_embedding(target)  # (B, L, H)
        mask_tag = (target > 0).float()  # (B, L)

        B, L, H = tgt_emb.shape
        T = self.diffu_student.num_timesteps

        # ----- 3. Sample per-token timesteps in [1, T-1] -----
        # We avoid t=0 (trivial: x_t = x_0, consistency target is identical).
        # For padding positions, mask_tag=0 zeroes the loss anyway, so the
        # specific t there does not matter; we just pass the same range.
        t_high = torch.randint(1, T, (B, L), device=device)
        t_low = (t_high - 1).clamp(min=0)

        # ----- 4. Forward noise (mask-aware to match teacher training) -----
        noise = torch.randn_like(tgt_emb)
        x_t_high = self.diffu_student.q_sample(tgt_emb, t_high, noise,
                                               mask=mask_tag)

        # ----- 5. Teacher one-step DDIM (consistency target builder) -----
        # teacher.net is frozen and in eval mode (enforced in distill_trainer).
        with torch.no_grad():
            x_t_low = self.diffu_student.teacher_ddim_step(
                teacher_diffu, item_rep, x_t_high, t_high, t_low,
                mask_seq, mask_tag,
            )

        # ----- 6. Online student forward (with gradient) -----
        pred_high = self.diffu_student.predict_x0(item_rep, x_t_high, t_high,
                                                  mask_seq, mask_tag)

        # ----- 7. EMA target forward (no gradient) -----
        # The EMA target sees the SAME conditioning (item_rep) so that
        # consistency is enforced only on the (x_t, t) -> x_0 mapping, not
        # on the encoder. This matches Song et al. 2023.
        with torch.no_grad():
            pred_low = self.diffu_student_ema.predict_x0(
                item_rep, x_t_low, t_low, mask_seq, mask_tag,
            )

        # =================== L_cons (consistency MSE) ===================
        diff_sq = (pred_high - pred_low).pow(2).sum(-1)  # (B, L)
        if use_adaptive_weighting:
            w = self._min_snr_weight(t_high, gamma=5.0)  # (B, L)
        else:
            w = torch.ones_like(diff_sq)
        cons_per_token = diff_sq * w * mask_tag
        denom = mask_tag.sum().clamp(min=1.0)
        cons_loss = cons_per_token.sum() / denom

        # =================== L_ce (full-item CE) ========================
        # Same as teacher's calculate_loss but on student's pred_high.
        valid = target > 0
        if valid.any():
            scores_flat = torch.matmul(
                pred_high[valid], self.item_embedding.weight.t(),
            )  # (N_valid, V)
            ce_loss = self.loss_ce(scores_flat, target[valid])
        else:
            ce_loss = torch.zeros((), device=device)

        # =================== L_contrast (RCCD per-token) ================
        if use_rccd:
            contrast_loss = self._rccd_loss(
                pred_high, tgt_emb, mask_tag,
                temperature=contrast_temperature,
            )
        else:
            contrast_loss = torch.zeros((), device=device)

        return cons_loss, ce_loss, contrast_loss

    def _rccd_loss(self, pred_high, tgt_emb, mask_tag, temperature=0.1):
        """
        Per-token InfoNCE with K shared catalog negatives per batch.

        Anchor   : pred_high[b, l]        — student's predicted x_0.
        Positive : tgt_emb[b, l]          — embedding of the true next item.
        Negatives: K items sampled uniformly from item IDs in [1, item_num]
                   (excluding padding id 0). Negatives are SHARED across the
                   batch (sampled once per call) — standard sampled-softmax
                   practice, cheap, decouples negatives from sequence
                   correlations.

        All vectors L2-normalised → cosine similarity in [-1, 1] →
        temperature τ is scale-invariant.
        """
        device = pred_high.device
        B, L, H = pred_high.shape

        anchor = F.normalize(pred_high, dim=-1)             # (B, L, H)
        pos = F.normalize(tgt_emb, dim=-1)                  # (B, L, H)

        # Sample K negatives from [1, item_num] (exclude padding id 0).
        neg_ids = torch.randint(1, self.item_num + 1, (self.K_neg,),
                                device=device)
        neg_emb = F.normalize(self.item_embedding(neg_ids), dim=-1)  # (K, H)

        # Positive logits: cos(anchor[b,l], pos[b,l])
        pos_logits = (anchor * pos).sum(-1, keepdim=True)   # (B, L, 1)

        # Negative logits: anchor @ neg^T
        neg_logits = torch.matmul(anchor, neg_emb.t())      # (B, L, K)

        # InfoNCE: -log(exp(pos)/(exp(pos)+sum_k exp(neg_k))) with positive
        # at index 0 in the concatenated logits matrix.
        logits = torch.cat([pos_logits, neg_logits], dim=-1) / temperature
        labels = torch.zeros(B, L, dtype=torch.long, device=device)

        ce = F.cross_entropy(
            logits.reshape(-1, self.K_neg + 1),
            labels.reshape(-1),
            reduction='none',
        ).reshape(B, L)

        return (ce * mask_tag).sum() / mask_tag.sum().clamp(min=1.0)

    # ----- Inference: scores over all items for the LAST position -----

    @torch.no_grad()
    def predict_scores(self, sequence, target, num_steps=1):
        """
        Inference at the last sequence position.

        Mirrors Att_Diffuse_model.forward(train_flag=False):
          - sequence : (B, L) history tokens (hist_pad)
          - target   : (B, L) answer_pad. Positions 0..L-2 are the shifted
                       history (CLEAN, used as autoregressive context).
                       Position L-1 is the held-out answer (used only for
                       scoring after this function returns).
          - num_steps: NFE for the student. 1 = single net forward pass.

        Returns scores : (B, V+1) for the last position only.
        """
        # Embed history (with hist_norm).
        item_emb, mask_seq = self.encode_history(sequence)
        # Embed target context (NO hist_norm, matches teacher).
        tgt_emb = self.item_embedding(target)
        mask_tag = (target > 0).float()

        x_0 = self.diffu_student.sample(item_emb, tgt_emb, mask_seq, mask_tag,
                                        num_steps=num_steps)
        last_item = x_0[:, -1, :]  # (B, H)
        scores = torch.matmul(last_item, self.item_embedding.weight.t())
        return scores
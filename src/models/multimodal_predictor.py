import math
import torch
import torch.nn as nn
from attention import SpatialFeatureProjector, MultiScaleTemporalTransformer, CrossAttentionFusion, AttentionPooling

# =========================================================
# Internal Latent Dynamics Model
# =========================================================

class LatentDynamicsModel(nn.Module):
    """
    Refines the predicted next latent by combining:
      - z_t   : the last temporal hidden state  [B, hidden_dim]
      - context: attention-pooled sequence mean  [B, hidden_dim]
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, z_t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, context], dim=-1))


# =========================================================
# Sequence Predictor
# =========================================================

class SequencePredictor(nn.Module):
    """
    Predicts the next image AND next text given a sequence of
    (image, text) pairs (default window = 4 steps).

    Args:
        visual_autoencoder : VisualAutoencoder  – provides .encoder / .decoder
        text_autoencoder   : Seq2Seq            – provides .encoder / .decoder
        latent_dim         : visual VAE latent size (default 256)
        hidden_dim         : shared working dimension across all temporal /
                             fusion modules (default 512)
        temporal_layers    : depth of MultiScaleTemporalTransformer (default 4)
        temporal_heads     : attention heads in temporal transformer (default 8)
        cross_attention_heads : heads in CrossAttentionFusion (default 8)
        dropout            : dropout rate throughout (default 0.1)

    Forward inputs:
        image_seq               [B, T, C, H, W]
        input_ids_text_encoder  [B, T, L]   – token ids for each timestep
        attention_mask_text_enc [B, T, L]   – padding mask for encoder
        target_seq_text_decoder [B, S]      – decoder input ids (teacher-forced)

    Forward outputs (dict):
        pred_image       [B, 3, H, W]   – reconstructed next frame
        pred_text_logits [B, S, vocab]  – next-text logits (teacher-forced)
        z_next           [B, hidden_dim]
        spatial_z        [B, spatial_dim, 14, 14]
        mu               [B, latent_dim]   – for KL loss
        logvar           [B, latent_dim]   – for KL loss
    """

    def __init__(
        self,
        visual_autoencoder,
        text_autoencoder,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        temporal_layers: int = 4,
        temporal_heads: int = 8,
        cross_attention_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ── frozen / partially-frozen backbones ──────────────────────────
        # (do NOT modify these sub-modules)
        self.image_encoder = visual_autoencoder.encoder   # CLIPEncoderWrapper
        self.image_decoder = visual_autoencoder.decoder   # VisualDecoder

        self.text_encoder  = text_autoencoder.encoder    # RobertaEncoder
        self.text_decoder  = text_autoencoder.decoder    # TransformerDecoder

        # infer text encoder output dim (RoBERTa-base → 768)
        text_enc_dim: int = self.text_encoder.roberta.config.hidden_size  # 768

        # ── projections into shared hidden_dim ───────────────────────────
        # visual latent z (latent_dim) → hidden_dim
        self.visual_projector = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # RoBERTa token sequence (text_enc_dim) → hidden_dim
        # Applied per-token before temporal modeling
        self.text_projector = nn.Sequential(
            nn.Linear(text_enc_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # ── spatial feature projector ─────────────────────────────────────
        # SpatialFeatureProjector: [B, 256, 14, 14] → [B, hidden_dim]
        # (used to build the per-step spatial token fed into temporal model)
        self.spatial_projector = SpatialFeatureProjector(
            in_channels=256,
            hidden_channels=256,
            output_dim=hidden_dim,
        )

        # ── temporal transformers ─────────────────────────────────────────
        # Visual: operates on sequence of projected z vectors [B, T, hidden_dim]
        self.visual_temporal = MultiScaleTemporalTransformer(
            dim=hidden_dim,
            num_heads=temporal_heads,
            num_layers=temporal_layers,
            dropout=dropout,
        )

        # Text: operates on sequence of mean-pooled text tokens [B, T, hidden_dim]
        self.text_temporal = MultiScaleTemporalTransformer(
            dim=hidden_dim,
            num_heads=temporal_heads,
            num_layers=temporal_layers,
            dropout=dropout,
        )

        # ── attention pooling ─────────────────────────────────────────────
        # Collapses [B, T, hidden_dim] → [B, hidden_dim] with learned weights
        self.visual_attn_pool = AttentionPooling(dim=hidden_dim)
        self.text_attn_pool   = AttentionPooling(dim=hidden_dim)

        # ── cross-modal fusion ────────────────────────────────────────────
        # Queries: pooled visual features  [B, 1, hidden_dim]
        # Keys/Values: pooled text features [B, 1, hidden_dim]
        # Output: fused visual representation [B, 1, hidden_dim]
        self.cross_attention_fusion = CrossAttentionFusion(
            vision_dim=hidden_dim,
            text_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=cross_attention_heads,
            dropout=dropout,
        )

        # ── latent dynamics model ─────────────────────────────────────────
        # Maps (fused_z, context) → z_next in hidden_dim space
        self.dynamics_model = LatentDynamicsModel(hidden_dim=hidden_dim)

        # ── projection back to latent_dim for the image decoder ──────────
        # VisualDecoder expects z of size latent_dim (256)
        self.z_to_latent = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # ── projection to text decoder memory dim (768) ──────────────────
        # TransformerDecoder.memory must be [B, *, 768] (d_model of RoBERTa)
        self.z_to_text_memory = nn.Sequential(
            nn.Linear(hidden_dim, text_enc_dim),
            nn.GELU(),
            nn.LayerNorm(text_enc_dim),
        )

    # ─────────────────────────────────────────────────────────────────────
    def forward(
        self,
        image_seq: torch.Tensor,               # [B, T, C, H, W]
        input_ids_text_encoder: torch.Tensor,  # [B, T, L]
        attention_mask_text_encoder: torch.Tensor,  # [B, T, L]
        target_seq_text_decoder: torch.Tensor, # [B, S]  teacher-forced target
    ) -> dict:

        B, T, C, H, W = image_seq.shape
        device = image_seq.device

        # ── 1. Encode every frame ─────────────────────────────────────────
        z_list, mu_list, logvar_list, spatial_list = [], [], [], []

        for t in range(T):
            z, mu, logvar, spatial_z = self.image_encoder(image_seq[:, t])
            z_list.append(z)           # each [B, latent_dim]
            mu_list.append(mu)
            logvar_list.append(logvar)
            spatial_list.append(spatial_z)  # each [B, 256, 14, 14]

        # Stack along time axis
        z_seq      = torch.stack(z_list, dim=1)       # [B, T, latent_dim]
        spatial_seq = torch.stack(spatial_list, dim=1) # [B, T, 256, 14, 14]

        # Keep mu/logvar of LAST frame for KL loss during sequence training
        mu_last     = mu_list[-1]
        logvar_last = logvar_list[-1]

        # ── 2. Project visual latents into hidden_dim ─────────────────────
        z_seq_proj = self.visual_projector(z_seq)     # [B, T, hidden_dim]

        # ── 3. Encode each text step, project, and mean-pool tokens ───────
        text_seq_list = []
        for t in range(T):
            # input_ids / attention_mask: [B, L]
            mem = self.text_encoder(
                input_ids=input_ids_text_encoder[:, t],
                attention_mask=attention_mask_text_encoder[:, t],
            )                                          # [B, L, 768]
            mem_proj = self.text_projector(mem)        # [B, L, hidden_dim]
            # Mean-pool over token dim to get a single vector per step
            mask = attention_mask_text_encoder[:, t].unsqueeze(-1).float()
            pooled = (mem_proj * mask).sum(1) / mask.sum(1).clamp(min=1)
            text_seq_list.append(pooled)               # [B, hidden_dim]

        text_seq_proj = torch.stack(text_seq_list, dim=1)  # [B, T, hidden_dim]

        # ── 4. Temporal modeling ──────────────────────────────────────────
        visual_temp = self.visual_temporal(z_seq_proj)     # [B, T, hidden_dim]
        text_temp   = self.text_temporal(text_seq_proj)    # [B, T, hidden_dim]

        # ── 5. Attention pooling over the T-step sequence ─────────────────
        # attn_weights shape: [B, T, 1]  — how much each timestep contributed
        visual_pooled, visual_attn_w = self.visual_attn_pool(visual_temp)  # [B, hidden_dim], [B, T, 1]
        text_pooled,   text_attn_w   = self.text_attn_pool(text_temp)      # [B, hidden_dim], [B, T, 1]

        # ── 6. Cross-modal fusion (vision queries text) ───────────────────
        # CrossAttentionFusion expects [B, seq, dim]; unsqueeze to add seq=1
        fused, _ = self.cross_attention_fusion(
            vision_seq=visual_pooled.unsqueeze(1),   # [B, 1, hidden_dim]
            text_seq=text_pooled.unsqueeze(1),        # [B, 1, hidden_dim]
        )                                             # [B, 1, hidden_dim]
        fused = fused.squeeze(1)                      # [B, hidden_dim]

        # ── 7. Latent dynamics: predict z_next ────────────────────────────
        # z_t    = last temporal visual hidden state
        # context = attention-pooled visual summary
        z_t     = visual_temp[:, -1]                  # [B, hidden_dim]
        context = visual_pooled                        # [B, hidden_dim]
        z_next  = self.dynamics_model(fused, context) # [B, hidden_dim]

        # ── 8. Spatial feature for image decoder ─────────────────────────
        # Project each frame's spatial map, then take the last one
        # (could be extended to a temporal fusion over spatial features)
        spatial_proj_list = [
            self.spatial_projector(spatial_seq[:, t])  # [B, hidden_dim]
            for t in range(T)
        ]
        # We still pass raw spatial of last frame to VisualDecoder
        # (decoder expects [B, spatial_dim=256, 14, 14])
        spatial_t = spatial_seq[:, -1]                 # [B, 256, 14, 14]

        # ── 9. Decode image ────────────────────────────────────────────────
        z_for_decoder = self.z_to_latent(z_next)       # [B, latent_dim]
        pred_image = self.image_decoder(z_for_decoder, spatial_t)
        # pred_image: [B, 3, H', W']

        # ── 10. Decode text ───────────────────────────────────────────────
        # Build memory for the TransformerDecoder from z_next
        # Memory shape must be [B, *, 768]; we use a single summary token
        text_memory = self.z_to_text_memory(z_next).unsqueeze(1)  # [B, 1, 768]

        # Causal mask for teacher-forced decoding
        S = target_seq_text_decoder.size(1)
        tgt_mask = _generate_causal_mask(S, device=device)

        pred_text_logits = self.text_decoder(
            tgt=target_seq_text_decoder,
            memory=text_memory,
            tgt_mask=tgt_mask,
        )                                              # [B, S, vocab_size]

        return {
            "pred_image":          pred_image,            # [B, 3, H', W']
            "pred_text_logits":    pred_text_logits,      # [B, S, vocab_size]
            "z_next":              z_next,                # [B, hidden_dim]
            "z_for_decoder":       z_for_decoder,         # [B, latent_dim]
            "spatial_z":           spatial_t,             # [B, 256, 14, 14]
            "mu":                  mu_last,               # [B, latent_dim]  for KL
            "logvar":              logvar_last,            # [B, latent_dim]  for KL
            # Learnable pooler weights – how much each of the T timesteps
            # contributed to the prediction. Shape [B, T], sums to ~1 over T.
            "visual_attn_weights": visual_attn_w.squeeze(-1),  # [B, T]
            "text_attn_weights":   text_attn_w.squeeze(-1),    # [B, T]
        }


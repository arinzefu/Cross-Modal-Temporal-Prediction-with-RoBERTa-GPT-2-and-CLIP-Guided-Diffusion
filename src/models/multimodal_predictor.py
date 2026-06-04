import torch
import torch.nn as nn

from attention import MultiScaleTemporalTransformer, CrossAttentionFusion, AttentionPooling


# =========================================================
# Internal Latent Dynamics Model
# =========================================================
class LatentDynamicsModel(nn.Module):
    """Refine the predicted next latent from (fused_z, context), both [B, hidden_dim]."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, z_t, context):
        return self.net(torch.cat([z_t, context], dim=-1))


# =========================================================
# Sequence Predictor
# ---------------------------------------------------------
# Predicts the NEXT (image, text) from the previous T (image, text) pairs.
#
# Backbones (used read-only; NOT modified here):
#   visual_autoencoder : GaussianDiffusion  -> .model.clip.global_latent(x01) gives a
#                        frozen 512-d CLIP embedding per frame (the visual latent).
#   text_autoencoder   : Seq2Seq (RoBERTa2GPT2)
#                        .encoder -> HF RobertaModel (returns .last_hidden_state [B,L,768])
#                        .decoder -> GPT-2 decoder (takes encoder_hidden_states, causal mask
#                                    handled internally)
#
# Outputs:
#   pred_image_latent  [B, 512]      predicted next-frame CLIP latent
#   pred_text_logits   [B, S, vocab] next-text logits (teacher-forced, GPT-2 vocab)
#
# NOTE: the diffusion visual model is self-conditioned (it reads CLIP features from the
# image it denoises), so it cannot render `pred_image_latent` to pixels without a small
# hook to accept an external condition. Train against the true next frame's CLIP latent;
# add that hook later if you want actual generated frames.
# =========================================================
class SequencePredictor(nn.Module):
    def __init__(
        self,
        visual_autoencoder,                 # GaussianDiffusion
        text_autoencoder,                   # Seq2Seq (RoBERTa2GPT2)
        hidden_dim: int = 512,
        temporal_layers: int = 4,
        temporal_heads: int = 8,
        cross_attention_heads: int = 8,
        dropout: float = 0.1,
        text_memory_tokens: int = 1,
    ):
        super().__init__()

        # ── backbones (read-only feature extractors / generator) ──────────
        self.visual_clip = visual_autoencoder.model.clip   # CLIPMultiScale (frozen)
        self.text_encoder = text_autoencoder.encoder       # HF RobertaModel
        self.text_decoder = text_autoencoder.decoder       # GPT-2 decoder (cross-attn)

        # infer dims from the actual backbones
        visual_dim = getattr(self.visual_clip.clip.config, "projection_dim", 512)  # 512 (ViT-B)
        text_enc_dim = self.text_encoder.config.hidden_size                        # 768

        # ── projections into the shared hidden_dim ───────────────────────
        self.visual_projector = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.text_projector = nn.Sequential(
            nn.Linear(text_enc_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )

        # ── temporal transformers ─────────────────────────────────────────
        self.visual_temporal = MultiScaleTemporalTransformer(
            dim=hidden_dim, num_heads=temporal_heads, num_layers=temporal_layers, dropout=dropout
        )
        self.text_temporal = MultiScaleTemporalTransformer(
            dim=hidden_dim, num_heads=temporal_heads, num_layers=temporal_layers, dropout=dropout
        )

        # ── pooling + cross-modal fusion ──────────────────────────────────
        self.visual_attn_pool = AttentionPooling(dim=hidden_dim)
        self.text_attn_pool = AttentionPooling(dim=hidden_dim)
        self.cross_attention_fusion = CrossAttentionFusion(
            vision_dim=hidden_dim, text_dim=hidden_dim, hidden_dim=hidden_dim,
            num_heads=cross_attention_heads, dropout=dropout,
        )
        self.dynamics_model = LatentDynamicsModel(hidden_dim=hidden_dim)

        # ── prediction heads ──────────────────────────────────────────────
        # visual: hidden_dim -> next-frame CLIP latent
        self.z_to_visual_latent = nn.Sequential(
            nn.Linear(hidden_dim, visual_dim), nn.LayerNorm(visual_dim)
        )
        # text: hidden_dim -> [M, 768] memory for the GPT-2 decoder cross-attention
        self.text_memory_tokens = text_memory_tokens
        self.z_to_text_memory = nn.Sequential(
            nn.Linear(hidden_dim, text_enc_dim * text_memory_tokens), nn.GELU()
        )
        self.text_memory_norm = nn.LayerNorm(text_enc_dim)

    @torch.no_grad()
    def _encode_frame(self, frames01):
        """frames01: [B,C,H,W] in [0,1] -> [B,512] frozen CLIP latent."""
        return self.visual_clip.global_latent(frames01)

    def forward(
        self,
        image_seq,                       # [B, T, C, H, W]  in [0,1]
        input_ids_text_encoder,          # [B, T, L]  RoBERTa token ids
        attention_mask_text_encoder,     # [B, T, L]
        target_seq_text_decoder,         # [B, S]  GPT-2 token ids (teacher-forced)
    ) -> dict:
        B, T = image_seq.shape[:2]

        # ── 1. per-frame visual latents (frozen CLIP) ─────────────────────
        v_seq = torch.stack([self._encode_frame(image_seq[:, t]) for t in range(T)], dim=1)  # [B,T,512]
        v_proj = self.visual_projector(v_seq)                                                # [B,T,hidden]

        # ── 2. per-step text features (RoBERTa -> project -> mean-pool) ────
        txt_steps = []
        for t in range(T):
            out = self.text_encoder(
                input_ids=input_ids_text_encoder[:, t],
                attention_mask=attention_mask_text_encoder[:, t],
            )
            mem = out.last_hidden_state                          # [B, L, 768]
            mp = self.text_projector(mem)                        # [B, L, hidden]
            mask = attention_mask_text_encoder[:, t].unsqueeze(-1).float()
            pooled = (mp * mask).sum(1) / mask.sum(1).clamp(min=1)   # [B, hidden]
            txt_steps.append(pooled)
        t_seq = torch.stack(txt_steps, dim=1)                    # [B, T, hidden]

        # ── 3. temporal modeling ──────────────────────────────────────────
        v_temp = self.visual_temporal(v_proj)                    # [B, T, hidden]
        t_temp = self.text_temporal(t_seq)                       # [B, T, hidden]

        # ── 4. attention pooling ──────────────────────────────────────────
        v_pool, v_w = self.visual_attn_pool(v_temp)              # [B,hidden], [B,T,1]
        t_pool, t_w = self.text_attn_pool(t_temp)

        # ── 5. cross-modal fusion (vision queries text) ───────────────────
        fused, _ = self.cross_attention_fusion(
            v_pool.unsqueeze(1), t_pool.unsqueeze(1)
        )
        fused = fused.squeeze(1)                                 # [B, hidden]

        # ── 6. latent dynamics -> z_next ──────────────────────────────────
        z_next = self.dynamics_model(fused, v_pool)              # [B, hidden]

        # ── 7. visual prediction: next-frame CLIP latent ──────────────────
        pred_image_latent = self.z_to_visual_latent(z_next)      # [B, 512]

        # ── 8. text prediction: build memory, run GPT-2 decoder ───────────
        mem = self.z_to_text_memory(z_next).view(B, self.text_memory_tokens, -1)  # [B, M, 768]
        mem = self.text_memory_norm(mem)
        dec_out = self.text_decoder(
            input_ids=target_seq_text_decoder,
            encoder_hidden_states=mem,
        )
        pred_text_logits = dec_out.logits                        # [B, S, vocab]

        return {
            "pred_image_latent": pred_image_latent,   # [B, 512]  -> train vs true next-frame CLIP latent
            "pred_text_logits": pred_text_logits,     # [B, S, vocab]
            "z_next": z_next,                         # [B, hidden]
            "visual_attn_weights": v_w.squeeze(-1),   # [B, T]
            "text_attn_weights": t_w.squeeze(-1),     # [B, T]
        }

    @torch.no_grad()
    def target_image_latent(self, next_frame01):
        """Helper: the true next frame's CLIP latent, to use as the visual target."""
        return self.visual_clip.global_latent(next_frame01)      # [B, 512]

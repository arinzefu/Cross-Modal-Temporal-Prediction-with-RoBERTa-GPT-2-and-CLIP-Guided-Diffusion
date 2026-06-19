
# =========================================================
# multimodal_predictor.py
# Predicts the 5th image and story from the previous 4
# image/story pairs using pretrained visual/text autoencoders.
# =========================================================

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Helpers
# =========================================================

def _set_requires_grad(module: nn.Module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad


def _group_count(channels: int, preferred: int = 8) -> int:
    for g in (preferred, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ConvResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        groups = _group_count(channels)

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


# =========================================================
# Attention Pooling
# =========================================================

class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        x:
            [B, T, D]

        mask:
            optional [B, T], True for valid positions

        returns:
            pooled  [B, D]
            weights [B, T, 1]
        """
        scores = self.score(x).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)

        return pooled, weights.unsqueeze(-1)


# =========================================================
# Temporal Transformer
# =========================================================

class TemporalTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_context_frames: int = 8,
    ):
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.zeros(1, max_context_frames, dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask: Optional[torch.Tensor] = None):
        """
        x:
            [B, T, D]

        key_padding_mask:
            optional [B, T], True for padded positions
        """
        t = x.size(1)

        if t > self.pos_embedding.size(1):
            raise ValueError(
                f"Sequence length {t} exceeds max_context_frames="
                f"{self.pos_embedding.size(1)}."
            )

        x = x + self.pos_embedding[:, :t]
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        return self.norm(x)


# =========================================================
# Cross-Modal Attention Fusion
# =========================================================

class CrossModalFusion(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vision_to_text = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.text_to_vision = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)

        self.vision_norm = nn.LayerNorm(dim)
        self.text_norm = nn.LayerNorm(dim)

        self.vision_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        self.text_ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

        self.vision_ffn_norm = nn.LayerNorm(dim)
        self.text_ffn_norm = nn.LayerNorm(dim)

        self.fuse = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

    def forward(
        self,
        vision_seq,
        text_seq,
        vision_padding_mask: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        vision_seq:
            [B, T, D]

        text_seq:
            [B, T, D]
        """
        vision_attended, vision_text_attn = self.vision_to_text(
            query=vision_seq,
            key=text_seq,
            value=text_seq,
            key_padding_mask=text_padding_mask,
        )

        text_attended, text_vision_attn = self.text_to_vision(
            query=text_seq,
            key=vision_seq,
            value=vision_seq,
            key_padding_mask=vision_padding_mask,
        )

        vision_seq = self.vision_norm(
            vision_seq + self.dropout(vision_attended)
        )

        text_seq = self.text_norm(
            text_seq + self.dropout(text_attended)
        )

        vision_seq = self.vision_ffn_norm(
            vision_seq + self.vision_ffn(vision_seq)
        )

        text_seq = self.text_ffn_norm(
            text_seq + self.text_ffn(text_seq)
        )

        if vision_seq.size(1) != text_seq.size(1):
            raise ValueError(
                "Expected aligned visual/text sequences with the same T. "
                f"Got vision T={vision_seq.size(1)}, text T={text_seq.size(1)}."
            )

        fused = torch.cat(
            [
                vision_seq,
                text_seq,
                vision_seq * text_seq,
                torch.abs(vision_seq - text_seq),
            ],
            dim=-1,
        )

        fused = self.fuse(fused)

        return fused, {
            "vision_to_text": vision_text_attn,
            "text_to_vision": text_vision_attn,
        }


# =========================================================
# Spatial Latent Projection
# =========================================================

class SpatialFeatureProjector(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 256,
        output_dim: int = 512,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.GELU(),
            ConvResidualBlock(hidden_channels),
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.final_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * 4 * 4, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        """
        x:
            [B, C, H, W] or [B, T, C, H, W]
        """
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            x = x.reshape(b * t, c, h, w)
            x = self.conv(x)
            x = self.pool(x)
            x = self.final_proj(x)
            return x.view(b, t, -1)

        x = self.conv(x)
        x = self.pool(x)

        return self.final_proj(x)


class SpatialLatentHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        spatial_dim: int = 256,
        spatial_size: int = 14,
        seed_channels: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.spatial_dim = spatial_dim
        self.spatial_size = spatial_size
        self.seed_channels = seed_channels

        self.seed = nn.Sequential(
            nn.Linear(hidden_dim, seed_channels * spatial_size * spatial_size),
            nn.GELU(),
        )

        self.net = nn.Sequential(
            ConvResidualBlock(seed_channels, dropout=dropout),
            nn.Conv2d(seed_channels, spatial_dim, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(spatial_dim), spatial_dim),
            nn.GELU(),
            ConvResidualBlock(spatial_dim, dropout=dropout),
            nn.Conv2d(spatial_dim, spatial_dim, kernel_size=3, padding=1),
        )

    def forward(self, x):
        b = x.size(0)

        x = self.seed(x)
        x = x.view(
            b,
            self.seed_channels,
            self.spatial_size,
            self.spatial_size,
        )

        return self.net(x)


# =========================================================
# Latent Dynamics
# =========================================================

class LatentDynamicsModel(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, last_state, context):
        update = self.net(torch.cat([last_state, context], dim=-1))
        return self.norm(last_state + update)


# =========================================================
# Sequence Predictor
# =========================================================

class SequencePredictor(nn.Module):
    """
    Predicts the next image/story from the previous image/story sequence.

    Intended use:
        input images/stories:
            frames 1-4 and stories 1-4

        target:
            frame 5 and story 5

    Visual path:
        VisualAutoencoder.encode(image) -> z, spatial
        predictor predicts next z and next spatial
        VisualAutoencoder.decode(pred_z, pred_spatial) -> predicted image

    Text path:
        RoBERTa encodes the previous stories.
        The predictor builds GPT-2 cross-attention memory.
        GPT-2 predicts the next story with teacher forcing.
    """

    def __init__(
        self,
        visual_autoencoder,
        text_autoencoder,
        hidden_dim: int = 512,
        temporal_layers: int = 4,
        temporal_heads: int = 8,
        cross_attention_heads: int = 8,
        dropout: float = 0.1,
        text_memory_tokens: int = 24,
        expected_context_frames: Optional[int] = 4,
        max_context_frames: int = 8,
        freeze_visual_encoder: bool = True,
        freeze_visual_decoder: bool = True,
        freeze_text_encoder: bool = True,
        train_text_decoder: bool = False,
        predict_residual: bool = True,
        residual_delta: float = 0.15,
        ground_text_on_inputs: bool = True,
        frame_blend: bool = True,
    ):
        super().__init__()

        self.visual_autoencoder = visual_autoencoder
        self.text_autoencoder = text_autoencoder

        self.text_encoder = text_autoencoder.encoder
        self.text_decoder = text_autoencoder.decoder

        self.expected_context_frames = expected_context_frames
        self.freeze_visual_encoder = freeze_visual_encoder
        self.freeze_visual_decoder = freeze_visual_decoder
        self.freeze_text_encoder = freeze_text_encoder
        self.train_text_decoder = train_text_decoder

        self.visual_latent_dim = getattr(
            visual_autoencoder.decoder,
            "latent_dim",
            128,
        )

        self.spatial_dim = getattr(
            visual_autoencoder.decoder,
            "spatial_dim",
            256,
        )

        self.spatial_size = 14
        self.text_enc_dim = self.text_encoder.config.hidden_size

        if freeze_visual_encoder:
            _set_requires_grad(self.visual_autoencoder.encoder, False)

        if freeze_visual_decoder:
            _set_requires_grad(self.visual_autoencoder.decoder, False)

        if freeze_text_encoder:
            _set_requires_grad(self.text_encoder, False)

        if not train_text_decoder:
            _set_requires_grad(self.text_decoder, False)

        self.visual_z_projector = nn.Sequential(
            nn.Linear(self.visual_latent_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.visual_spatial_projector = SpatialFeatureProjector(
            in_channels=self.spatial_dim,
            hidden_channels=min(hidden_dim, 256),
            output_dim=hidden_dim,
        )

        self.visual_token_norm = nn.LayerNorm(hidden_dim)

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_enc_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.visual_temporal = TemporalTransformer(
            dim=hidden_dim,
            num_heads=temporal_heads,
            num_layers=temporal_layers,
            dropout=dropout,
            max_context_frames=max_context_frames,
        )

        self.text_temporal = TemporalTransformer(
            dim=hidden_dim,
            num_heads=temporal_heads,
            num_layers=temporal_layers,
            dropout=dropout,
            max_context_frames=max_context_frames,
        )

        self.cross_modal_fusion = CrossModalFusion(
            dim=hidden_dim,
            num_heads=cross_attention_heads,
            dropout=dropout,
        )

        self.fusion_temporal = TemporalTransformer(
            dim=hidden_dim,
            num_heads=temporal_heads,
            num_layers=max(1, temporal_layers // 2),
            dropout=dropout,
            max_context_frames=max_context_frames,
        )

        self.visual_attn_pool = AttentionPooling(hidden_dim)
        self.text_attn_pool = AttentionPooling(hidden_dim)
        self.fusion_attn_pool = AttentionPooling(hidden_dim)

        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.dynamics_model = LatentDynamicsModel(
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # No trailing LayerNorm: in residual mode this head outputs a *delta*
        # added to the anchor latent, so it must be free to have a small scale.
        self.z_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.visual_latent_dim),
            nn.LayerNorm(self.visual_latent_dim),
        )

        self.spatial_head = SpatialLatentHead(
            hidden_dim=hidden_dim,
            spatial_dim=self.spatial_dim,
            spatial_size=self.spatial_size,
            seed_channels=min(128, hidden_dim),
            dropout=dropout,
        )

        # Residual prediction with a HARD bound. The previous learnable gate let
        # the head output explode (z std blew up to ~4.5x target), pushing the
        # latent off the decoder manifold -> grey. A fixed tanh bound keeps the
        # prediction within +/- residual_delta of the (sharp, on-manifold) anchor.
        self.predict_residual = predict_residual
        self.residual_delta = residual_delta

        # Frame-blend anchor: instead of anchoring the prediction on frame 4 alone
        # (which makes it copy frame 4), predict softmax weights over ALL context
        # frames and anchor on their weighted blend. A blend of on-manifold frame
        # latents stays near the manifold (so it decodes sharp), and the output is
        # a function of all 4 frames. Zero-init -> starts as a uniform average.
        self.frame_blend = frame_blend
        self.frame_score = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.frame_score.weight)
        nn.init.zeros_(self.frame_score.bias)

        self.text_memory_tokens = text_memory_tokens

        self.text_memory_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.text_enc_dim * text_memory_tokens),
        )

        self.text_memory_pos = nn.Parameter(
            torch.zeros(1, text_memory_tokens, self.text_enc_dim)
        )

        nn.init.trunc_normal_(self.text_memory_pos, std=0.02)

        self.text_memory_norm = nn.LayerNorm(self.text_enc_dim)

        # Ground the decoder on the actual input-story token features so it can
        # copy entities (names, objects) instead of hallucinating from the GPT-2
        # prior. These raw RoBERTa features match what the decoder cross-attended
        # to during text-autoencoder pretraining.
        self.ground_text_on_inputs = ground_text_on_inputs
        self.text_ground_norm = nn.LayerNorm(self.text_enc_dim)

        self.train(True)

    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze_visual_encoder:
            self.visual_autoencoder.encoder.eval()

        if self.freeze_visual_decoder:
            self.visual_autoencoder.decoder.eval()

        if self.freeze_text_encoder:
            self.text_encoder.eval()

        if not self.train_text_decoder:
            self.text_decoder.eval()

        return self

    def _encode_visual_sequence(self, image_seq):
        """
        image_seq:
            [B, T, 3, H, W] in [0, 1]

        returns:
            z_seq       [B, T, latent_dim]
            spatial_seq [B, T, spatial_dim, 14, 14]
        """
        b, t, c, h, w = image_seq.shape
        flat = image_seq.reshape(b * t, c, h, w)

        if self.freeze_visual_encoder:
            with torch.no_grad():
                z, spatial = self.visual_autoencoder.encode(flat)
        else:
            z, spatial = self.visual_autoencoder.encode(flat)

        z = z.view(b, t, -1)
        spatial = spatial.view(
            b,
            t,
            spatial.size(1),
            spatial.size(2),
            spatial.size(3),
        )

        return z, spatial

    def _encode_text_sequence(self, input_ids, attention_mask):
        """
        input_ids:
            [B, T, L]

        attention_mask:
            [B, T, L]

        returns:
            text_seq [B, T, hidden_dim]
        """
        b, t, l = input_ids.shape

        flat_ids = input_ids.reshape(b * t, l)
        flat_mask = attention_mask.reshape(b * t, l)

        if self.freeze_text_encoder:
            with torch.no_grad():
                enc = self.text_encoder(
                    input_ids=flat_ids,
                    attention_mask=flat_mask,
                    return_dict=True,
                )
        else:
            enc = self.text_encoder(
                input_ids=flat_ids,
                attention_mask=flat_mask,
                return_dict=True,
            )

        raw_feats = enc.last_hidden_state                 # [B*T, L, text_enc_dim]
        token_features = self.text_projector(raw_feats)   # [B*T, L, hidden_dim]

        mask = flat_mask.unsqueeze(-1).float()
        pooled = (token_features * mask).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp_min(1.0)
        pooled = pooled.view(b, t, -1)

        # Flatten the per-frame token features into one grounding memory bank.
        ground_feats = raw_feats.reshape(b, t * l, -1)    # [B, T*L, text_enc_dim]
        ground_mask = flat_mask.reshape(b, t * l)         # [B, T*L]

        return pooled, ground_feats, ground_mask

    def _build_next_state(
        self,
        image_seq,
        input_ids_text_encoder,
        attention_mask_text_encoder,
    ):
        b, t = image_seq.shape[:2]

        if self.expected_context_frames is not None:
            if t != self.expected_context_frames:
                raise ValueError(
                    f"Expected {self.expected_context_frames} context frames "
                    f"for 5th-frame prediction, got T={t}."
                )

        z_seq, spatial_seq = self._encode_visual_sequence(image_seq)

        visual_tokens = self.visual_z_projector(z_seq)
        spatial_tokens = self.visual_spatial_projector(spatial_seq)

        visual_tokens = self.visual_token_norm(
            visual_tokens + spatial_tokens
        )

        text_tokens, text_ground_feats, text_ground_mask = self._encode_text_sequence(
            input_ids_text_encoder,
            attention_mask_text_encoder,
        )

        visual_temp = self.visual_temporal(visual_tokens)
        text_temp = self.text_temporal(text_tokens)

        fused_tokens, cross_attn = self.cross_modal_fusion(
            visual_temp,
            text_temp,
        )

        fused_temp = self.fusion_temporal(fused_tokens)

        visual_pool, visual_weights = self.visual_attn_pool(visual_temp)
        text_pool, text_weights = self.text_attn_pool(text_temp)
        fusion_pool, fusion_weights = self.fusion_attn_pool(fused_temp)

        last_state = fused_temp[:, -1]

        context = self.context_fusion(
            torch.cat(
                [
                    last_state,
                    fusion_pool,
                    visual_pool,
                    text_pool,
                ],
                dim=-1,
            )
        )

        next_state = self.dynamics_model(last_state, context)

        # Per-frame blend weights over the context frames (softmax over T).
        frame_scores = self.frame_score(fused_temp).squeeze(-1)   # [B, T]
        frame_weights = torch.softmax(frame_scores, dim=1)        # [B, T]

        return {
            "next_state": next_state,
            "visual_pool": visual_pool,
            "text_pool": text_pool,
            "fusion_pool": fusion_pool,
            "visual_attn_weights": visual_weights.squeeze(-1),
            "text_attn_weights": text_weights.squeeze(-1),
            "fusion_attn_weights": fusion_weights.squeeze(-1),
            "cross_attn": cross_attn,
            "input_visual_z": z_seq,
            "input_visual_spatial": spatial_seq,
            "text_ground_feats": text_ground_feats,
            "text_ground_mask": text_ground_mask,
            "frame_weights": frame_weights,
        }

    def _make_text_memory(self, next_state):
        b = next_state.size(0)

        mem = self.text_memory_head(next_state)
        mem = mem.view(b, self.text_memory_tokens, self.text_enc_dim)

        mem = mem + self.text_memory_pos[:, : self.text_memory_tokens]
        mem = self.text_memory_norm(mem)

        return mem

    def _decoder_memory(self, next_state, ground_feats=None, ground_mask=None):
        """
        Full cross-attention memory the decoder sees:
        the predicted "next-state" tokens, followed (optionally) by the grounding
        bank of input-story token features. Returns (memory, attention_mask).
        """
        pred_mem = self._make_text_memory(next_state)
        b = pred_mem.size(0)
        device = pred_mem.device

        mask = torch.ones(b, pred_mem.size(1), dtype=torch.long, device=device)

        if self.ground_text_on_inputs and ground_feats is not None:
            ground = self.text_ground_norm(ground_feats)
            memory = torch.cat([pred_mem, ground], dim=1)

            if ground_mask is None:
                ground_mask = torch.ones(
                    b, ground.size(1), dtype=torch.long, device=device
                )
            mask = torch.cat([mask, ground_mask.long()], dim=1)
            return memory, mask

        return pred_mem, mask

    def forward(
        self,
        image_seq,
        input_ids_text_encoder,
        attention_mask_text_encoder,
        target_seq_text_decoder=None,
        target_attention_mask_text_decoder=None,
        image_target=None,
        decode_image: bool = True,
    ):
        """
        image_seq:
            [B, 4, 3, H, W]

        input_ids_text_encoder:
            [B, 4, L]

        attention_mask_text_encoder:
            [B, 4, L]

        target_seq_text_decoder:
            [B, S], GPT-2 teacher-forced ids for story 5

        image_target:
            optional [B, 3, H, W], true frame 5
        """
        b = image_seq.size(0)

        state = self._build_next_state(
            image_seq=image_seq,
            input_ids_text_encoder=input_ids_text_encoder,
            attention_mask_text_encoder=attention_mask_text_encoder,
        )

        next_state = state["next_state"]

        z_delta = self.z_head(next_state)
        spatial_delta = self.spatial_head(next_state)

        if self.predict_residual:
            # Anchor on a weighted blend of ALL context frames (not just frame 4),
            # then add a *bounded* change. tanh keeps the change within
            # +/- residual_delta, so the latent stays near the (on-manifold) blend.
            if self.frame_blend:
                fw = state["frame_weights"]                       # [B, T]
                anchor_z = torch.einsum(
                    "bt,btd->bd", fw, state["input_visual_z"]
                )
                anchor_spatial = torch.einsum(
                    "bt,btchw->bchw", fw, state["input_visual_spatial"]
                )
            else:
                anchor_z = state["input_visual_z"][:, -1]
                anchor_spatial = state["input_visual_spatial"][:, -1]

            pred_image_z = anchor_z + self.residual_delta * torch.tanh(z_delta)
            pred_image_spatial = (
                anchor_spatial + self.residual_delta * torch.tanh(spatial_delta)
            )
        else:
            pred_image_z = z_delta
            pred_image_spatial = spatial_delta

        pred_image = None

        if decode_image:
            pred_image = self.visual_autoencoder.decode(
                pred_image_z,
                pred_image_spatial,
                add_noise=False,
            )

        text_memory, encoder_attention_mask = self._decoder_memory(
            next_state,
            state.get("text_ground_feats"),
            state.get("text_ground_mask"),
        )

        pred_text_logits = None

        if target_seq_text_decoder is not None:
            dec_out = self.text_decoder(
                input_ids=target_seq_text_decoder,
                attention_mask=target_attention_mask_text_decoder,
                encoder_hidden_states=text_memory,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=True,
            )

            pred_text_logits = dec_out.logits

        out = {
            "pred_image": pred_image,
            "pred_image_z": pred_image_z,
            "pred_image_spatial": pred_image_spatial,
            "pred_text_logits": pred_text_logits,
            "text_memory": self._make_text_memory(next_state),
            "z_next": next_state,
            "v_pool": state["visual_pool"],
            "t_pool": state["text_pool"],
            "fusion_pool": state["fusion_pool"],
            "visual_attn_weights": state["visual_attn_weights"],
            "text_attn_weights": state["text_attn_weights"],
            "fusion_attn_weights": state["fusion_attn_weights"],
            "cross_attn": state["cross_attn"],
            "input_visual_z": state["input_visual_z"],
            "input_visual_spatial": state["input_visual_spatial"],
            "frame_weights": state["frame_weights"],

            # Compatibility aliases for older training code.
            "pred_image_latent": pred_image_z,
            "pred_image_cond": pred_image_spatial,
        }

        if image_target is not None:
            target_z, target_spatial = self.target_image_latents(image_target)
            out["target_image_z"] = target_z
            out["target_image_spatial"] = target_spatial
            out["target_image_latent"] = target_z
            out["target_image_cond"] = target_spatial

        return out

    @torch.no_grad()
    def target_image_latents(self, next_frame01):
        """
        Encodes the true 5th image into the pretrained visual AE latent space.
        """
        was_training = self.visual_autoencoder.training
        self.visual_autoencoder.eval()

        z, spatial = self.visual_autoencoder.encode(next_frame01)

        if was_training:
            self.visual_autoencoder.train()

        return z, spatial

    @torch.no_grad()
    def target_image_latent(self, next_frame01):
        """
        Compatibility helper. Returns only the global visual latent z.
        """
        z, _ = self.target_image_latents(next_frame01)
        return z

    @torch.no_grad()
    def target_image_spatial(self, next_frame01):
        """
        Returns only the spatial visual latent.
        """
        _, spatial = self.target_image_latents(next_frame01)
        return spatial

    @torch.no_grad()
    def target_image_cond(self, next_frame01):
        """
        Compatibility helper for older code.

        In this autoencoder version, the old diffusion condition is replaced
        by the spatial latent used by VisualAutoencoder.decode.
        """
        return self.target_image_spatial(next_frame01)

    def decode_image_latents(self, z, spatial):
        """
        Decodes predicted or target visual latents into image space.
        Gradients flow to z/spatial if they require grad.
        """
        return self.visual_autoencoder.decode(
            z,
            spatial,
            add_noise=False,
        )

    def _decoder_start_token_id(self):
        candidates = []

        if hasattr(self.text_autoencoder, "model"):
            candidates.extend(
                [
                    getattr(
                        self.text_autoencoder.model.config,
                        "decoder_start_token_id",
                        None,
                    ),
                    getattr(
                        self.text_autoencoder.model.config,
                        "bos_token_id",
                        None,
                    ),
                    getattr(
                        self.text_autoencoder.model.config,
                        "eos_token_id",
                        None,
                    ),
                    getattr(
                        self.text_autoencoder.model.config,
                        "pad_token_id",
                        None,
                    ),
                ]
            )

        if getattr(self.text_autoencoder, "dec_tokenizer", None) is not None:
            tok = self.text_autoencoder.dec_tokenizer
            candidates.extend(
                [
                    tok.bos_token_id,
                    tok.eos_token_id,
                    tok.pad_token_id,
                ]
            )

        for value in candidates:
            if value is not None:
                return int(value)

        raise ValueError(
            "Could not resolve decoder start token id. "
            "Pass target_seq_text_decoder during training or attach dec_tokenizer."
        )

    def _decoder_eos_token_id(self):
        if getattr(self.text_autoencoder, "dec_tokenizer", None) is not None:
            return self.text_autoencoder.dec_tokenizer.eos_token_id

        if hasattr(self.text_autoencoder, "model"):
            return getattr(
                self.text_autoencoder.model.config,
                "eos_token_id",
                None,
            )

        return None

    @staticmethod
    def _apply_repetition_penalty(logits, prev_ids, penalty):
        """HF-style repetition penalty on already-generated tokens."""
        if penalty is None or penalty == 1.0:
            return logits
        for b in range(logits.size(0)):
            seen = torch.unique(prev_ids[b])
            vals = logits[b, seen]
            vals = torch.where(vals > 0, vals / penalty, vals * penalty)
            logits[b, seen] = vals
        return logits

    @staticmethod
    def _block_repeat_ngrams(logits, prev_ids, ngram_size):
        """Forbid tokens that would complete an n-gram already seen in the output."""
        if not ngram_size or ngram_size <= 0:
            return logits
        seq_len = prev_ids.size(1)
        if seq_len < ngram_size:
            return logits
        for b in range(prev_ids.size(0)):
            tokens = prev_ids[b].tolist()
            prefix = tuple(tokens[-(ngram_size - 1):]) if ngram_size > 1 else tuple()
            banned = set()
            for i in range(len(tokens) - ngram_size + 1):
                if tuple(tokens[i:i + ngram_size - 1]) == prefix:
                    banned.add(tokens[i + ngram_size - 1])
            if banned:
                logits[b, list(banned)] = torch.finfo(logits.dtype).min
        return logits

    @torch.no_grad()
    def generate_text_ids(
        self,
        image_seq,
        input_ids_text_encoder,
        attention_mask_text_encoder,
        max_new_tokens: int = 80,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 3,
    ):
        """
        Greedy generation for story 5.

        Returns:
            token ids without the initial decoder-start token.
        """
        self.eval()

        state = self._build_next_state(
            image_seq=image_seq,
            input_ids_text_encoder=input_ids_text_encoder,
            attention_mask_text_encoder=attention_mask_text_encoder,
        )

        text_memory, encoder_attention_mask = self._decoder_memory(
            state["next_state"],
            state.get("text_ground_feats"),
            state.get("text_ground_mask"),
        )

        b = image_seq.size(0)
        device = image_seq.device

        start_id = self._decoder_start_token_id()
        eos_id = self._decoder_eos_token_id()

        ids = torch.full(
            (b, 1),
            start_id,
            dtype=torch.long,
            device=device,
        )

        finished = torch.zeros(b, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            attention_mask = torch.ones_like(ids)

            dec_out = self.text_decoder(
                input_ids=ids,
                attention_mask=attention_mask,
                encoder_hidden_states=text_memory,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=True,
            )

            logits = dec_out.logits[:, -1, :].clone()
            logits = self._apply_repetition_penalty(logits, ids, repetition_penalty)
            logits = self._block_repeat_ngrams(logits, ids, no_repeat_ngram_size)

            next_id = logits.argmax(dim=-1, keepdim=True)

            if eos_id is not None:
                # Once a sequence has emitted eos, keep padding with eos so the
                # decoded string is not corrupted by post-eos tokens.
                next_id = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_id, eos_id),
                    next_id,
                )

            ids = torch.cat([ids, next_id], dim=1)

            if eos_id is not None:
                finished = finished | (next_id.squeeze(1) == eos_id)

                if bool(finished.all()):
                    break

        return ids[:, 1:]

    @torch.no_grad()
    def predict_next(
        self,
        image_seq,
        input_ids_text_encoder,
        attention_mask_text_encoder,
        max_new_tokens: int = 80,
    ):
        """
        Convenience inference helper.

        Returns:
            dict with predicted image, visual latents, and generated text ids.
        """
        out = self.forward(
            image_seq=image_seq,
            input_ids_text_encoder=input_ids_text_encoder,
            attention_mask_text_encoder=attention_mask_text_encoder,
            target_seq_text_decoder=None,
            decode_image=True,
        )

        out["generated_text_ids"] = self.generate_text_ids(
            image_seq=image_seq,
            input_ids_text_encoder=input_ids_text_encoder,
            attention_mask_text_encoder=attention_mask_text_encoder,
            max_new_tokens=max_new_tokens,
        )

        return out

# =========================================================
# visual_autoencoder.py
# =========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel
from torchvision import models
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:          # guard — your version assumed bias exists
            nn.init.zeros_(m.bias)

# =========================================================
# Residual Block
# =========================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
        )

        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


# =========================================================
# CLIP Encoder
# =========================================================

class CLIPEncoderWrapper(nn.Module):
    """
    CLIP encoder that returns:

    z:
        compact global latent [B, latent_dim]

    spatial:
        compressed spatial latent [B, spatial_dim, 14, 14]

    low/mid/high:
        optional raw CLIP feature maps for analysis or auxiliary losses

    NOTE on the spatial path: low/mid/high come from CLIP hidden_states
    [3], [6], [9], which are ALWAYS frozen (only the last `unfreeze_layers`
    vision layers are trainable). So the spatial latent's geometry is fixed
    CLIP semantics; only the 1x1 projections adapt. That is fine for
    reconstruction but it is the part of the latent that is hardest to
    predict, so keep an eye on it when training the SequencePredictor.
    """

    def __init__(
        self,
        latent_dim=128,
        spatial_dim=256,
        unfreeze_layers=2,
        clip_model_name="openai/clip-vit-base-patch16",
    ):
        super().__init__()

        self.clip = CLIPModel.from_pretrained(clip_model_name)

        self.clip.vision_model.config.output_hidden_states = True
        self.clip.config.output_hidden_states = True

        # Freeze CLIP first
        for p in self.clip.parameters():
            p.requires_grad = False

        # Optionally unfreeze last N CLIP vision layers
        if unfreeze_layers > 0:
            for layer in self.clip.vision_model.encoder.layers[-unfreeze_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True

        hidden_dim = self.clip.config.vision_config.hidden_size  # 768 for ViT-B/16

        # Global latent projection
        self.global_projection = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # Compress low/mid/high CLIP features into one spatial latent
        self.low_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, spatial_dim // 4, kernel_size=1),
            nn.GroupNorm(8, spatial_dim // 4),
            nn.GELU(),
        )

        self.mid_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, spatial_dim // 4, kernel_size=1),
            nn.GroupNorm(8, spatial_dim // 4),
            nn.GELU(),
        )

        self.high_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, spatial_dim // 2, kernel_size=1),
            nn.GroupNorm(8, spatial_dim // 2),
            nn.GELU(),
        )

        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(spatial_dim, spatial_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, spatial_dim),
            nn.GELU(),
            ResidualBlock(spatial_dim),
        )

        self.register_buffer(
            "clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
        )

        self.register_buffer(
            "clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
        )

    def _preprocess(self, x):
        """
        x expected in [0, 1].
        """
        x = F.interpolate(
            x,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        return (x - self.clip_mean) / self.clip_std

    def _hidden_to_map(self, h):
        """
        h: [B, 197, 768]
        returns: [B, 768, 14, 14]
        """
        h = h[:, 1:, :]  # remove CLS token
        b, n, c = h.shape
        side = int(n ** 0.5)
        return h.transpose(1, 2).contiguous().view(b, c, side, side)

    def forward(self, x, return_raw=False):
        """
        x: image in [0, 1]
        """
        x_clip = self._preprocess(x)

        vision_outputs = self.clip.vision_model(
            pixel_values=x_clip,
            output_hidden_states=True,
            return_dict=True,
        )

        # Global latent
        z = self.global_projection(vision_outputs.pooler_output)

        # Multi-scale CLIP hidden states
        low_feat = self._hidden_to_map(vision_outputs.hidden_states[3])
        mid_feat = self._hidden_to_map(vision_outputs.hidden_states[6])
        high_feat = self._hidden_to_map(vision_outputs.hidden_states[9])

        low = self.low_proj(low_feat)
        mid = self.mid_proj(mid_feat)
        high = self.high_proj(high_feat)

        spatial = torch.cat([low, mid, high], dim=1)
        spatial = self.spatial_fusion(spatial)

        if return_raw:
            return z, spatial, low_feat, mid_feat, high_feat

        return z, spatial


# =========================================================
# Decoder
# =========================================================

class VisualDecoder(nn.Module):
    """
    Decoder for sequence prediction.

    It reconstructs from:

        z       [B, latent_dim]
        spatial [B, spatial_dim, 14, 14]

    This means the sequence predictor only needs to predict z and spatial.
    It does not need raw CLIP low/mid/high features.

    NOTE: the 4-stage transposed-conv stack hard-codes 14 -> 224. `output_size`
    is retained for API compatibility but only 224 is supported by this stack;
    it is asserted rather than silently ignored.
    """

    def __init__(
        self,
        latent_dim=128,
        spatial_dim=256,
        base_dim=256,
        output_size=224,
    ):
        super().__init__()

        assert output_size == 224, (
            "VisualDecoder upsampling stack only reaches 224x224; "
            f"got output_size={output_size}."
        )

        self.latent_dim = latent_dim
        self.spatial_dim = spatial_dim
        self.base_dim = base_dim
        self.output_size = output_size

        # Global latent -> 14x14 feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, base_dim * 14 * 14),
            nn.GELU(),
        )

        # Fuse global map and spatial latent
        self.initial_fusion = nn.Sequential(
            nn.Conv2d(base_dim + spatial_dim, base_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_dim),
            nn.GELU(),
            ResidualBlock(base_dim),
        )

        # 14 -> 28
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_dim, 128, 4, 2, 1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            ResidualBlock(128),
        )

        # 28 -> 56
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            ResidualBlock(64),
        )

        # 56 -> 112
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            ResidualBlock(32),
        )

        # 112 -> 224
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.GroupNorm(8, 16),
            nn.GELU(),
            ResidualBlock(16),
        )

        self.final = nn.Sequential(
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z, spatial):
        b = z.size(0)

        global_map = self.fc(z).view(b, self.base_dim, 14, 14)

        x = torch.cat([global_map, spatial], dim=1)
        x = self.initial_fusion(x)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        x = self.final(x)

        return x


# =========================================================
# Visual Autoencoder
# =========================================================

class VisualAutoencoder(nn.Module):
    """
    Sequence-ready visual autoencoder.

    Main methods:

    forward(x):
        image reconstruction (injects latent noise while training)

    encode(x):
        returns z, spatial

    decode(z, spatial, add_noise=None):
        reconstructs image from (possibly predicted) latents

    Latent-noise robustness
    ------------------------
    `z_noise_std` and `spatial_noise_std` add Gaussian noise to the latents
    *before decoding*, scaled per-sample by the latent's own std. This is
    active ONLY in training mode, so:
      - your already-converged eval-time decode is unchanged;
      - a short fine-tune with noise on hardens the decoder against the
        approximate latents the SequencePredictor produces, which is the
        single biggest robustness win for the 5th-frame prediction task.
    Set both to 0.0 to recover the original behaviour exactly.
    """

    def __init__(
        self,
        latent_dim=128,
        spatial_dim=256,
        unfreeze_layers=2,
        z_noise_std=0.05,
        spatial_noise_std=0.10,
    ):
        super().__init__()

        self.encoder = CLIPEncoderWrapper(
            latent_dim=latent_dim,
            spatial_dim=spatial_dim,
            unfreeze_layers=unfreeze_layers,
        )

        self.decoder = VisualDecoder(
            latent_dim=latent_dim,
            spatial_dim=spatial_dim,
        )

        # Robustness knobs (non-parametric; not in state_dict)
        self.z_noise_std = z_noise_std
        self.spatial_noise_std = spatial_noise_std

    # ----- latent noise -----
    def _maybe_noise(self, x, std):
        """
        Add std-relative Gaussian noise (training only). Scaling by the
        per-sample std keeps the perturbation sensible for both the 128-dim
        z and the ~50k-dim spatial tensor without separate tuning.
        """
        if std <= 0.0 or not self.training:
            return x
        dims = tuple(range(1, x.dim()))
        scale = x.detach().std(dim=dims, keepdim=True).clamp_min(1e-6)
        return x + torch.randn_like(x) * std * scale

    # ----- core API -----
    def encode(self, x):
        """
        x: image in [0, 1]

        returns:
            z       [B, latent_dim]
            spatial [B, spatial_dim, 14, 14]
        """
        return self.encoder(x)

    def decode(self, z, spatial, add_noise=None):
        """
        Decode from latents.

        add_noise:
            None  -> follow self.training (noise while training, clean in eval)
            True  -> force noise
            False -> force clean (use this when rendering predicted frames)
        """
        if add_noise is None:
            add_noise = self.training
        if add_noise:
            z = self._maybe_noise(z, self.z_noise_std)
            spatial = self._maybe_noise(spatial, self.spatial_noise_std)
        return self.decoder(z, spatial)

    def forward(self, x):
        z, spatial = self.encode(x)
        x_hat = self.decode(z, spatial)  # noise applied iff training
        return x_hat

    # ----- predictor-facing helpers -----
    @torch.no_grad()
    def global_latent(self, x):
        """
        Convenience method for sequence predictor (no grad — logging/inference).
        """
        z, _ = self.encode(x)
        return z

    def encode_global(self, x):
        """
        Same as global_latent but keeps gradients, for end-to-end
        predictor training paths.
        """
        z, _ = self.encode(x)
        return z

    def freeze_decoder(self):
        """
        Freeze the decoder so the SequencePredictor can be trained by decoding
        its predicted latent and applying a pixel + perceptual loss in image
        space (the recommended way to make latent errors tolerable).
        """
        for p in self.decoder.parameters():
            p.requires_grad = False
        self.decoder.eval()
        return self

    @torch.no_grad()
    def latent_usage(self, x):
        """
        Diagnostic for posterior-collapse on z. Decodes with z zeroed and with
        spatial zeroed; if drop_z_l1 is tiny, the decoder barely uses z and the
        predictor's effort on z is largely wasted.
        """
        was_training = self.training
        self.eval()
        z, spatial = self.encode(x)
        full = self.decode(z, spatial, add_noise=False)
        no_z = self.decode(torch.zeros_like(z), spatial, add_noise=False)
        no_sp = self.decode(z, torch.zeros_like(spatial), add_noise=False)
        if was_training:
            self.train()
        return {
            "drop_z_l1": F.l1_loss(no_z, full).item(),
            "drop_spatial_l1": F.l1_loss(no_sp, full).item(),
        }


# =========================================================
# Perceptual Loss (multi-layer)
# =========================================================

class PerceptualLoss(nn.Module):
    """
    Multi-layer VGG16 perceptual loss (relu1_2, relu2_2, relu3_3).
    Frozen and forced to stay in eval() even under .train() so it never
    drifts. No trainable params.
    """

    # vgg16.features indices for relu1_2 / relu2_2 / relu3_3
    DEFAULT_LAYER_IDS = (3, 8, 15)

    def __init__(self, layer_ids=DEFAULT_LAYER_IDS, layer_weights=None):
        super().__init__()

        vgg = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_FEATURES
        ).features

        self.blocks = nn.ModuleList()
        prev = 0
        for idx in layer_ids:
            self.blocks.append(nn.Sequential(*[vgg[i] for i in range(prev, idx + 1)]))
            prev = idx + 1

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

        if layer_weights is None:
            layer_weights = [1.0] * len(self.blocks)
        self.layer_weights = layer_weights

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )

        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def train(self, mode=True):
        # Keep VGG frozen/eval regardless of parent .train() calls.
        return super().train(False)

    def _norm(self, x):
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        p = self._norm(pred)
        t = self._norm(target)

        loss = 0.0
        for blk, w in zip(self.blocks, self.layer_weights):
            p = blk(p)
            t = blk(t)
            loss = loss + w * F.l1_loss(p, t)

        return loss


# =========================================================
# Combined Reconstruction Loss
# =========================================================

class ReconstructionLoss(nn.Module):
    """
    Image-space loss (pixel + perceptual) plus an optional latent-matching
    term used when training the SequencePredictor against target latents.

    Latent term changes vs original:
      - z and spatial now have separate weights (they live on very different
        scales / importance);
      - z uses cosine distance by default (global CLIP-style codes behave
        better under cosine than raw MSE). Set z_distance="mse" to revert.
    """

    def __init__(
        self,
        pixel_weight=0.6,
        perceptual_weight=0.3,
        latent_weight=0.1,
        z_weight=1.0,
        spatial_weight=1.0,
        z_distance="cosine",  # "cosine" | "mse"
    ):
        super().__init__()

        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.latent_weight = latent_weight
        self.z_weight = z_weight
        self.spatial_weight = spatial_weight
        assert z_distance in ("cosine", "mse")
        self.z_distance = z_distance

        self.perceptual = PerceptualLoss()

    def _z_loss(self, z_pred, z_target):
        if self.z_distance == "cosine":
            return 1.0 - F.cosine_similarity(z_pred, z_target, dim=-1).mean()
        return F.mse_loss(z_pred, z_target)

    def forward(
        self,
        pred,
        target,
        z_pred=None,
        z_target=None,
        spatial_pred=None,
        spatial_target=None,
    ):
        pixel_loss = (
            0.5 * F.mse_loss(pred, target)
            + 0.5 * F.l1_loss(pred, target)
        )

        perceptual_loss = self.perceptual(pred, target)

        total = (
            self.pixel_weight * pixel_loss
            + self.perceptual_weight * perceptual_loss
        )

        latent_loss = torch.tensor(0.0, device=pred.device)

        if z_pred is not None and z_target is not None:
            latent_loss = latent_loss + self.z_weight * self._z_loss(z_pred, z_target)

        if spatial_pred is not None and spatial_target is not None:
            latent_loss = latent_loss + self.spatial_weight * F.mse_loss(
                spatial_pred, spatial_target
            )

        total = total + self.latent_weight * latent_loss

        return total
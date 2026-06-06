# @title Importing the necessary libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from transformers import CLIPProcessor, CLIPModel, RobertaModel, RobertaTokenizer
import evaluate
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import UNet2DConditionModel, DDPMScheduler
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.transforms.functional as FT
import math


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def zero_init_output(model):
    nn.init.zeros_(model.out.weight)
    nn.init.zeros_(model.out.bias)# =========================================================
# Helpers
# =========================================================

def extract(a, t, x_shape):
    """Gather schedule values a[t] and reshape to broadcast over x."""
    b = t.shape[0]
    out = a.gather(0, t)
    return out.reshape(b, *([1] * (len(x_shape) - 1)))


def gn(ch, groups=8):
    # all channel counts used here are multiples of 8
    return nn.GroupNorm(groups, ch)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=device) * -freqs)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


# =========================================================
# Time-conditioned Residual Block (FiLM-style time injection)
# =========================================================

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time = nn.Linear(time_dim, out_ch)
        self.norm2 = gn(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time(t_emb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm = gn(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)
        out, _ = self.attn(h, h, h)
        return x + out.transpose(1, 2).view(B, C, H, W)


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return self.op(x)


# =========================================================
# CLIP multi-scale conditioning encoder (low / mid / high)
# =========================================================

class CLIPMultiScale(nn.Module):
    """
    Extracts hidden states from 3 depths of CLIP ViT-B/16:
        low  -> texture / edges
        mid  -> parts / local structure
        high -> semantics
    Each is [B, 196, 768] -> reshaped to [B, 768, 14, 14] -> projected.
    Returned concatenated as conditioning at the U-Net bottleneck.
    """
    def __init__(self, layers=(4, 8, 12), proj_ch=128, unfreeze=4, backbone=None):
        super().__init__()
        if backbone is not None:
            self.clip = backbone
        else:
            from transformers import CLIPModel  # lazy import
            self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

        for p in self.clip.parameters():
            p.requires_grad = False
        for layer in self.clip.vision_model.encoder.layers[-unfreeze:]:
            for p in layer.parameters():
                p.requires_grad = True

        self.layers = layers
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(768, proj_ch, 1), gn(proj_ch), nn.SiLU())
            for _ in layers
        ])
        self.out_ch = proj_ch * len(layers)

        # --- robust feature capture via forward hooks ---
        # hidden_states[idx] == output of encoder layer (idx-1), so we hook
        # layer (idx-1). This does not depend on return_dict behavior, which
        # varies across transformers versions.
        self._feats = {}
        enc = self.clip.vision_model.encoder.layers
        for idx in layers:
            enc[idx - 1].register_forward_hook(self._make_hook(idx))

        self.register_buffer("mean", torch.tensor([0.481, 0.457, 0.408]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.268, 0.261, 0.275]).view(1, 3, 1, 1))

    def _make_hook(self, idx):
        def hook(module, inp, out):
            self._feats[idx] = out[0] if isinstance(out, (tuple, list)) else out
        return hook

    def _prep(self, x01):
        x = F.interpolate(x01, size=(224, 224), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def forward(self, x01):
        x = self._prep(x01)
        self._feats.clear()
        self.clip.vision_model(pixel_values=x)            # triggers hooks
        feats = []
        for idx, proj in zip(self.layers, self.projs):
            h = self._feats[idx][:, 1:, :]                # drop CLS -> [B,196,768]
            h = h.transpose(1, 2).contiguous().view(h.shape[0], 768, 14, 14)
            feats.append(proj(h))
        return torch.cat(feats, dim=1)                    # [B, proj_ch*3, 14, 14]

    @torch.no_grad()
    def global_latent(self, x01):
        """512-d normalized CLIP image embedding — handy for the sequence predictor."""
        x = self._prep(x01)

        feat = self.clip.get_image_features(pixel_values=x)

        # Robustly handle different HuggingFace CLIP return types
        if not torch.is_tensor(feat):
            if hasattr(feat, "image_embeds") and feat.image_embeds is not None:
                feat = feat.image_embeds
            elif hasattr(feat, "pooler_output") and feat.pooler_output is not None:
                feat = feat.pooler_output
            elif hasattr(feat, "last_hidden_state") and feat.last_hidden_state is not None:
                feat = feat.last_hidden_state[:, 0]
            else:
                raise TypeError(f"Unsupported CLIP output type: {type(feat)}")

        return F.normalize(feat, dim=-1)


# =========================================================
# The Denoiser U-Net (predicts eps)
# =========================================================

class CLIPDiffusionUNet(nn.Module):
    def __init__(self, base=64, time_dim=256, clip_layers=(4, 8, 12),
                 clip_proj=128, use_clip=True, clip_module=None):
        super().__init__()
        self.use_clip = use_clip

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base),
            nn.Linear(base, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        clip_ch = 0
        if use_clip:
            self.clip = clip_module if clip_module is not None else \
                CLIPMultiScale(clip_layers, clip_proj)
            clip_ch = self.clip.out_ch

        b = base
        # ---- encoder (conv path produces real spatial skips) ----
        self.init = nn.Conv2d(3, b, 3, padding=1)        # 224, b
        self.rb1  = ResBlock(b, b, time_dim)             # 224, b      -> s0
        self.down1 = Down(b)                             # -> 112
        self.rb2  = ResBlock(b, b * 2, time_dim)         # 112, 2b     -> s1
        self.down2 = Down(b * 2)                         # -> 56
        self.rb3  = ResBlock(b * 2, b * 4, time_dim)     # 56, 4b      -> s2
        self.down3 = Down(b * 4)                         # -> 28
        self.rb4  = ResBlock(b * 4, b * 4, time_dim)     # 28, 4b      -> s3
        self.down4 = Down(b * 4)                         # -> 14
        self.rb5  = ResBlock(b * 4, b * 8, time_dim)     # 14, 8b      -> s4

        # ---- bottleneck: fuse CLIP semantic features ----
        self.fuse = nn.Conv2d(b * 8 + clip_ch, b * 8, 1)
        self.mid1 = ResBlock(b * 8, b * 8, time_dim)
        self.attn = AttentionBlock(b * 8)
        self.mid2 = ResBlock(b * 8, b * 8, time_dim)

        # ---- decoder (upsample + skip concat) ----
        self.up4  = Up(b * 8)                            # 14 -> 28
        self.urb4 = ResBlock(b * 8 + b * 4, b * 4, time_dim)
        self.up3  = Up(b * 4)                            # 28 -> 56
        self.urb3 = ResBlock(b * 4 + b * 4, b * 4, time_dim)
        self.up2  = Up(b * 4)                            # 56 -> 112
        self.urb2 = ResBlock(b * 4 + b * 2, b * 2, time_dim)
        self.up1  = Up(b * 2)                            # 112 -> 224
        self.urb1 = ResBlock(b * 2 + b, b, time_dim)

        self.out_norm = gn(b)
        self.out = nn.Conv2d(b, 3, 3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x, t, cond_feat=None):
        """x: noisy image in [-1, 1], t: long [B]. Returns predicted noise.

        cond_feat: optional external CLIP condition [B, clip_ch, 14, 14]. If given
        (or set via self._external_cond), it REPLACES the features normally computed
        from x. This is what lets the predictor drive generation; weights unchanged.
        """
        temb = self.time_mlp(t)

        cfeat = None
        if self.use_clip:
            if cond_feat is None:
                cond_feat = getattr(self, "_external_cond", None)
            if cond_feat is not None:
                cfeat = cond_feat                                   # <-- use the external condition
            else:
                x01 = (x.clamp(-1, 1) + 1) * 0.5
                cfeat = self.clip(x01)

        h0 = self.init(x)
        s0 = self.rb1(h0, temb)
        s1 = self.rb2(self.down1(s0), temb)
        s2 = self.rb3(self.down2(s1), temb)
        s3 = self.rb4(self.down3(s2), temb)
        s4 = self.rb5(self.down4(s3), temb)              # 14x14, 8b

        if self.use_clip:
            s4 = self.fuse(torch.cat([s4, cfeat], dim=1))
        else:
            s4 = self.fuse(s4)

        h = self.mid1(s4, temb)
        h = self.attn(h)
        h = self.mid2(h, temb)

        h = self.urb4(torch.cat([self.up4(h), s3], 1), temb)
        h = self.urb3(torch.cat([self.up3(h), s2], 1), temb)
        h = self.urb2(torch.cat([self.up2(h), s1], 1), temb)
        h = self.urb1(torch.cat([self.up1(h), s0], 1), temb)

        return self.out(self.act(self.out_norm(h)))
# =========================================================
# Gaussian Diffusion (forward noising + reverse sampling)
# =========================================================

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    ac = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return torch.clip(betas, 1e-4, 0.999)


class GaussianDiffusion:
    def __init__(self, model, timesteps=1000, device="cuda"):
        self.model = model
        self.T = timesteps
        self.device = device

        betas = cosine_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, dim=0)
        ac_prev = F.pad(ac[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas_cumprod = ac
        self.sqrt_ac = torch.sqrt(ac)
        self.sqrt_one_minus_ac = torch.sqrt(1 - ac)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_var = betas * (1 - ac_prev) / (1 - ac)

    # ---- forward process q(x_t | x_0) ----
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        return (extract(self.sqrt_ac, t, x0.shape) * x0 +
                extract(self.sqrt_one_minus_ac, t, x0.shape) * noise)

    # ---- training objective (predict eps) ----
    def p_losses(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred = self.model(x_t, t)
        loss = F.mse_loss(pred, noise)
        # recover predicted x0 for logging / viz
        with torch.no_grad():
            x0_hat = (x_t - extract(self.sqrt_one_minus_ac, t, x0.shape) * pred) \
                     / extract(self.sqrt_ac, t, x0.shape)
        return loss, x0_hat

    # ---- single reverse step ----
    @torch.no_grad()
    def p_sample(self, x_t, t):
        eps = self.model(x_t, t)
        mean = extract(self.sqrt_recip_alphas, t, x_t.shape) * (
            x_t - extract(self.betas, t, x_t.shape) /
            extract(self.sqrt_one_minus_ac, t, x_t.shape) * eps
        )
        if (t == 0).all():
            return mean
        var = extract(self.posterior_var, t, x_t.shape)
        return mean + torch.sqrt(var) * torch.randn_like(x_t)

    # ---- full reverse loop, optionally starting partway ----
    @torch.no_grad()
    def sample(self, shape=None, x_start=None, t_start=None):
        if x_start is None:
            x = torch.randn(shape, device=self.device)
            t_start = self.T
        else:
            x = x_start
            t_start = t_start if t_start is not None else self.T
        for i in reversed(range(t_start)):
            t = torch.full((x.shape[0],), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t)
        return x


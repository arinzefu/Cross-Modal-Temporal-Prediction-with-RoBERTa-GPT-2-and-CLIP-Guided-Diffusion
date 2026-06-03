# @title Importing the necessary libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt
import numpy as np
import os
from nltk.translate.bleu_score import sentence_bleu
import json
import pandas as pd
from torchinfo import summary
from transformers import CLIPProcessor, CLIPModel, RobertaModel, RobertaTokenizer
import evaluate
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import tqdm
from datasets.fingerprint import random
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.transforms.functional as FT
import math
from transformers import BertTokenizer
import gc
import random

from typing import Dict, Any, List, Optional, Tuple
import textwrap
from tqdm import tqdm

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # Use a slightly higher gain for leaky_relu to boost signal
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01) # Small positive bias to keep neurons alive
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
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
            nn.GroupNorm(8, channels)
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


# =========================================================
# Attention Block
# =========================================================

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)
        attn_out, _ = self.attn(h, h, h)
        return x + attn_out.transpose(1, 2).view(B, C, H, W)


# =========================================================
# CLIP Encoder (Predictive + Stochastic)
# =========================================================

class CLIPEncoderWrapper(nn.Module):
    def __init__(self, latent_dim=256, spatial_dim=256, unfreeze_layers=4):
        super().__init__()

        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch16"
        )

        self.clip.vision_model.config.output_hidden_states = True

        for p in self.clip.parameters():
            p.requires_grad = False

        for layer in self.clip.vision_model.encoder.layers[-unfreeze_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

        hidden = self.clip.config.vision_config.hidden_size

        # -------- VAE heads --------
        self.mu = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.GELU(),
            nn.Linear(512, latent_dim)
        )

        self.logvar = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.GELU(),
            nn.Linear(512, latent_dim)
        )

        # -------- spatial latent --------
        self.spatial = nn.Sequential(
            nn.Conv2d(768, 512, 1),
            nn.GroupNorm(8, 512),
            nn.GELU(),
            nn.Conv2d(512, spatial_dim, 1)
        )

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):

        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        mean = torch.tensor([0.481, 0.457, 0.408], device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.268, 0.261, 0.275], device=x.device).view(1,3,1,1)
        x = (x - mean) / std

        out = self.clip.vision_model(pixel_values=x, return_dict=True, output_hidden_states=True)

        pooled = out.pooler_output

        mu = self.mu(pooled)
        logvar = self.logvar(pooled)
        z = self.reparam(mu, logvar)

        # Ensure hidden_states exist (needed for torchinfo)
        h_states = out.hidden_states if out.hidden_states is not None else [out.last_hidden_state]
        patches = h_states[-1][:, 1:, :].transpose(1, 2)
        B = patches.shape[0]
        patches = patches.reshape(B, 768, 14, 14)

        spatial_z = self.spatial(patches)

        return z, mu, logvar, spatial_z


# =========================================================
# Decoder
# =========================================================

class VisualDecoder(nn.Module):
    def __init__(self, latent_dim=256, spatial_dim=256):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 256 * 14 * 14)

        self.fuse = nn.Conv2d(256 + spatial_dim, 256, 3, padding=1)

        self.up = nn.Sequential(
            ResidualBlock(256),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.GELU(),
            ResidualBlock(128),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.GELU(),
            ResidualBlock(64),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.GELU(),
            ResidualBlock(32),

            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.GELU(),
            ResidualBlock(16),

            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z, spatial_z):

        B = z.size(0)

        x = self.fc(z).view(B, 256, 14, 14)

        x = torch.cat([x, spatial_z], dim=1)

        x = self.fuse(x)

        return self.up(x)


# =========================================================
# FULL AUTOENCODER (NOW PREDICTIVE)
# =========================================================

class VisualAutoencoder(nn.Module):
    def __init__(self, latent_dim=256, spatial_dim=256):
        super().__init__()

        self.encoder = CLIPEncoderWrapper(latent_dim, spatial_dim)
        self.decoder = VisualDecoder(latent_dim, spatial_dim)

        # -------- CRITICAL ADDITION --------
        # latent predictor (temporal modeling)
        self.latent_predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x):

        z, mu, logvar, spatial = self.encoder(x)

        x_hat = self.decoder(z, spatial)

        return x_hat, z, mu, logvar, spatial

    # predict next latent (KEY FOR SEQUENCE MODEL)
    def predict_next_latent(self, z_t):
        return self.latent_predictor(z_t)


# =========================================================
# LOSSES
# =========================================================

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES)
        self.features = nn.Sequential(*list(vgg.features[:16])).eval()
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        return F.l1_loss(self.features(pred), self.features(target))


class KLLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.perc = PerceptualLoss()
        self.kl = KLLoss()

    def forward(self, pred, target, mu, logvar):

        pixel = 0.5 * F.mse_loss(pred, target) + 0.5 * F.l1_loss(pred, target)
        perceptual = self.perc(pred, target)
        kl = self.kl(mu, logvar)

        return {
            "total": pixel + 0.3 * perceptual + 0.01 * kl,
            "pixel": pixel,
            "perceptual": perceptual,
            "kl": kl
        }
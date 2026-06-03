# =========================================================
# Learnable Attention Pooling
# =========================================================

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )

    def forward(self, x):
        """
        x: [B, T, D]
        """

        scores = self.score(x)                    # [B, T, 1]
        weights = torch.softmax(scores, dim=1)   # [B, T, 1]

        pooled = (x * weights).sum(dim=1)        # [B, D]

        return pooled, weights


# =========================================================
# Cross Attention Fusion
# =========================================================

class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        vision_dim,
        text_dim,
        hidden_dim,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, vision_seq, text_seq):
        """
        vision_seq: [B, T, vision_dim]
        text_seq:   [B, L, text_dim]
        """

        q = self.vision_proj(vision_seq)
        k = self.text_proj(text_seq)
        v = self.text_proj(text_seq)

        attn_out, attn_weights = self.cross_attn(
            query=q,
            key=k,
            value=v
        )

        fused = self.norm(q + attn_out)

        return fused, attn_weights


# =========================================================
# Multi Scale Temporal Transformer
# =========================================================

class MultiScaleTemporalTransformer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        num_layers=4,
        dropout=0.1
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.layers = nn.ModuleList([
            nn.TransformerEncoder(encoder_layer, num_layers=1)
            for _ in range(num_layers)
        ])

        self.scale_fusion = nn.Sequential(
            nn.Linear(dim * num_layers, dim),
            nn.GELU(),
            nn.LayerNorm(dim)
        )

    def forward(self, x):
        """
        x: [B, T, D]
        """

        multi_scale_features = []

        for layer in self.layers:

            x = layer(x)

            multi_scale_features.append(x)

        fused = torch.cat(multi_scale_features, dim=-1)

        fused = self.scale_fusion(fused)

        return fused


# =========================================================
# Spatial Preserving Visual Projection
# =========================================================

class SpatialFeatureProjector(nn.Module):
    def __init__(
        self,
        in_channels=256,
        hidden_channels=256,
        output_dim=512
    ):
        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1
            ),

            nn.GroupNorm(8, hidden_channels),
            nn.GELU(),

            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1
            ),

            nn.GroupNorm(8, hidden_channels),
            nn.GELU()
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.final_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * 4 * 4, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):

        x = self.conv(x)

        x = self.pool(x)

        x = self.final_proj(x)

        return x


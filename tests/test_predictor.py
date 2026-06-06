import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.training.train_predictor import _unpack, generate_frames, generate_text


# =========================================================
# 1. Temporal attention over the T input frames (visual + text)
#    Replaces plot_attention_heatmaps / plot_attention_heatmap_text.
# =========================================================
@torch.no_grad()
def plot_temporal_attention(predictor, dataloader, dec_tokenizer, device, max_samples=8):
    predictor.eval()
    batch = next(iter(dataloader))
    frames, _, text_dict = _unpack(batch)
    frames = frames.to(device)
    enc_ids = text_dict["input_ids"].to(device)
    enc_mask = text_dict["attention_mask"].to(device)
    start = dec_tokenizer.bos_token_id or dec_tokenizer.eos_token_id
    dummy = torch.full((frames.size(0), 1), start, dtype=torch.long, device=device)

    out = predictor(frames, enc_ids, enc_mask, dummy)
    v = out["visual_attn_weights"].cpu().numpy()   # [B,T]
    t = out["text_attn_weights"].cpu().numpy()      # [B,T]
    n, T = min(max_samples, v.shape[0]), v.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 0.5 * n + 2))
    for ax, mat, title in [(axes[0], v[:n], "Visual"), (axes[1], t[:n], "Text")]:
        im = ax.imshow(mat, cmap="viridis", aspect="auto")
        ax.set_title(f"{title} attention over input frames")
        ax.set_xlabel("input frame index"); ax.set_ylabel("sample")
        ax.set_xticks(range(T))
        plt.colorbar(im, ax=ax, fraction=0.046)
        for i in range(mat.shape[0]):
            for j in range(T):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white", fontsize=7)
    plt.tight_layout(); plt.show(); plt.close(fig)

    # mean importance bar
    plt.figure(figsize=(7, 3))
    width = 0.4
    x = np.arange(T)
    plt.bar(x - width / 2, v.mean(0), width, label="visual")
    plt.bar(x + width / 2, t.mean(0), width, label="text")
    plt.xticks(x, [f"frame {i}" for i in range(T)])
    plt.ylabel("mean attention"); plt.title("Average temporal attention")
    plt.legend(); plt.tight_layout(); plt.show(); plt.close()


# =========================================================
# 2. CLIP patch-attention overlay on an input frame
#    Replaces plot_feature_maps. Reads the CLIP ViT inside the frozen encoder.
# =========================================================
@torch.no_grad()
def plot_clip_attention(predictor, dataloader, device, layer_id=-1, head=None, alpha=0.6):
    predictor.eval()
    clip = predictor.visual_clip.clip            # the underlying CLIP model
    try:
        clip.vision_model.config._attn_implementation = "eager"   # needed for output_attentions
    except Exception:
        pass

    batch = next(iter(dataloader))
    frames, *_ = _unpack(batch)
    img01 = frames[0, 0].to(device)              # first frame of first sample, [3,H,W] in [0,1]

    x = F.interpolate(img01.unsqueeze(0), (224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    x_norm = (x - mean) / std

    vis = clip.vision_model(pixel_values=x_norm, output_attentions=True, return_dict=True)
    attn = vis.attentions[layer_id][0]           # [heads, tokens, tokens]
    attn = attn.mean(0) if head is None else attn[head]
    cls_to_patches = attn[0, 1:]                 # CLS -> patches
    g = int(np.sqrt(cls_to_patches.shape[0]))
    heat = cls_to_patches.reshape(g, g)
    heat = F.interpolate(heat[None, None], (224, 224), mode="bilinear", align_corners=False)[0, 0]
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat = heat.cpu().numpy()

    base = x[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(base); ax[0].set_title("input frame"); ax[0].axis("off")
    ax[1].imshow(heat, cmap="jet"); ax[1].axis("off")
    ax[1].set_title(f"CLIP attention (layer {layer_id}, "
                    f"{'avg heads' if head is None else f'head {head}'})")
    ax[2].imshow(base); ax[2].imshow(heat, cmap="jet", alpha=alpha)
    ax[2].set_title("overlay"); ax[2].axis("off")
    plt.tight_layout(); plt.show(); plt.close(fig)


# =========================================================
# 3. Cross-modal alignment (batch image vs text pooled features)
#    Replaces plot_cross_modal_alignment, using v_pool / t_pool.
# =========================================================
@torch.no_grad()
def plot_cross_modal_alignment(predictor, dataloader, dec_tokenizer, device):
    predictor.eval()
    batch = next(iter(dataloader))
    frames, _, text_dict = _unpack(batch)
    frames = frames.to(device)
    enc_ids = text_dict["input_ids"].to(device)
    enc_mask = text_dict["attention_mask"].to(device)
    start = dec_tokenizer.bos_token_id or dec_tokenizer.eos_token_id
    dummy = torch.full((frames.size(0), 1), start, dtype=torch.long, device=device)

    out = predictor(frames, enc_ids, enc_mask, dummy)
    zi = F.normalize(out["v_pool"], dim=-1)
    zt = F.normalize(out["t_pool"], dim=-1)
    sim = (zi @ zt.t()).cpu().numpy()

    plt.figure(figsize=(7, 6))
    im = plt.imshow(sim, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, label="cosine similarity")
    plt.title("Cross-modal alignment (image rows vs text cols)")
    plt.xlabel("text sample"); plt.ylabel("image sample")
    plt.tight_layout(); plt.show(); plt.close()


# =========================================================
# 4. Predicted-frame figures + error map + text
#    Replaces plot_reconstruction_error_maps / get_prediction_examples / show_full_example.
# =========================================================
@torch.no_grad()
def show_predictions(predictor, diffusion, dataloader, dec_tokenizer, device, n=3):
    predictor.eval()
    batch = next(iter(dataloader))
    frames, image_target, text_dict = _unpack(batch)
    frames = frames.to(device)
    image_target = image_target.to(device)
    enc_ids = text_dict["input_ids"].to(device)
    enc_mask = text_dict["attention_mask"].to(device)
    tgt_ids = text_dict["target_ids"].to(device)

    n = min(n, frames.size(0))
    out = predictor(frames[:n], enc_ids[:n], enc_mask[:n],
                    tgt_ids[:n, :1] if tgt_ids.dim() == 2 else tgt_ids[:n])
    gen = generate_frames(diffusion, out["pred_image_cond"], 224)        # [n,3,224,224] in [0,1]
    texts = generate_text(predictor, frames[:n], enc_ids[:n], enc_mask[:n], dec_tokenizer, device)

    T = frames.size(1)
    for i in range(n):
        gt = F.interpolate(image_target[i:i+1], (224, 224), mode="bilinear",
                           align_corners=False)[0].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        pr = gen[i].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        err = np.abs(gt - pr).mean(axis=2)

        fig, axes = plt.subplots(1, T + 3, figsize=(3 * (T + 3), 3))
        for f in range(T):
            axes[f].imshow(frames[i, f].cpu().permute(1, 2, 0).numpy().clip(0, 1))
            axes[f].set_title(f"input {f}"); axes[f].axis("off")
        axes[T].imshow(gt); axes[T].set_title("true next"); axes[T].axis("off")
        axes[T + 1].imshow(pr); axes[T + 1].set_title("predicted"); axes[T + 1].axis("off")
        im = axes[T + 2].imshow(err, cmap="hot"); axes[T + 2].set_title("error"); axes[T + 2].axis("off")
        plt.colorbar(im, ax=axes[T + 2], fraction=0.046)
        plt.tight_layout(); plt.show(); plt.close(fig)

        gt_text = dec_tokenizer.decode(tgt_ids[i], skip_special_tokens=True)
        print(f"--- sample {i} ---\n  GT text   : {gt_text}\n  pred text : {texts[i]}\n")

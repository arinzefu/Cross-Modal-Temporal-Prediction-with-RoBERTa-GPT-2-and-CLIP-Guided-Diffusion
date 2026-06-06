import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


GEN_SIZE = 224


# =========================================================
# Batch unpacking
# =========================================================

def _unpack(batch):
    """
    Your current dataloader returns a list/tuple of items.
    The text dictionary is one of those items.

    Expected text_dict keys:
        enc_input_ids
        enc_attention_mask
        target_ids
        target_attention_mask
    """
    frames = batch[0]
    image_target = batch[1]
    text_dict = next(x for x in batch if isinstance(x, dict))

    return frames, image_target, text_dict


def _get_text_tensors(text_dict, device):
    """
    Handles your current tokenizer dictionary format.
    """
    enc_ids = text_dict["enc_input_ids"].to(device)
    enc_mask = text_dict["enc_attention_mask"].to(device)
    tgt_ids = text_dict["target_ids"].to(device)
    tgt_mask = text_dict["target_attention_mask"].to(device)

    return enc_ids, enc_mask, tgt_ids, tgt_mask


def _make_dummy_decoder_input(batch_size, dec_tokenizer, device):
    """
    Creates a one-token decoder input so the predictor can run
    when we only need condition/attention outputs.
    """
    start_id = dec_tokenizer.bos_token_id

    if start_id is None:
        start_id = dec_tokenizer.eos_token_id

    if start_id is None:
        start_id = dec_tokenizer.pad_token_id

    if start_id is None:
        raise ValueError("Decoder tokenizer has no bos/eos/pad token id.")

    dummy_ids = torch.full(
        (batch_size, 1),
        start_id,
        dtype=torch.long,
        device=device,
    )

    dummy_mask = torch.ones_like(dummy_ids)

    return dummy_ids, dummy_mask


# =========================================================
# DDIM pure-noise generation
# =========================================================

@torch.no_grad()
def generate_frames_ddim_cond(
    diffusion,
    pred_cond,
    image_size=GEN_SIZE,
    steps=100,
    eta=0.0,
):
    """
    Pure-noise DDIM generation using external condition.

    This is the correct path after external-condition fine-tuning:

        random noise + predicted condition -> generated next frame

    It does NOT use frame 4.
    """
    diffusion.model.eval()

    device = pred_cond.device
    B = pred_cond.size(0)

    x = torch.randn(
        B,
        3,
        image_size,
        image_size,
        device=device,
    )

    times = torch.linspace(
        diffusion.T - 1,
        0,
        steps,
        device=device,
    ).long()

    for i in range(len(times)):
        t_int = int(times[i].item())

        if i == len(times) - 1:
            prev_t_int = -1
        else:
            prev_t_int = int(times[i + 1].item())

        t = torch.full(
            (B,),
            t_int,
            device=device,
            dtype=torch.long,
        )

        eps = diffusion.model(
            x,
            t,
            cond_feat=pred_cond,
        )

        alpha_t = diffusion.alphas_cumprod[t_int].to(device)

        if prev_t_int >= 0:
            alpha_prev = diffusion.alphas_cumprod[prev_t_int].to(device)
        else:
            alpha_prev = torch.tensor(1.0, device=device)

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

        x0_pred = (
            x - sqrt_one_minus_alpha_t * eps
        ) / sqrt_alpha_t.clamp(min=1e-8)

        x0_pred = x0_pred.clamp(-1, 1)

        if prev_t_int < 0:
            x = x0_pred
        else:
            if eta == 0.0:
                sigma = 0.0
            else:
                sigma = eta * torch.sqrt(
                    (1 - alpha_prev) / (1 - alpha_t)
                    * (1 - alpha_t / alpha_prev)
                )

            direction = torch.sqrt(
                torch.clamp(1 - alpha_prev - sigma ** 2, min=0.0)
            ) * eps

            noise = torch.randn_like(x) if eta > 0 else 0.0

            x = torch.sqrt(alpha_prev) * x0_pred + direction + sigma * noise

    return (x.clamp(-1, 1) + 1) / 2


# =========================================================
# Text generation
# =========================================================

@torch.no_grad()
def generate_text(
    predictor,
    image_seq,
    enc_ids,
    enc_mask,
    dec_tokenizer,
    device,
    max_len=80,
):
    """
    Greedy GPT-2 decode conditioned on predictor's predicted text memory.
    Compatible with predictor.forward(..., target_ids, target_mask).
    """
    predictor.eval()

    B = image_seq.size(0)

    start_id = dec_tokenizer.bos_token_id

    if start_id is None:
        start_id = dec_tokenizer.eos_token_id

    if start_id is None:
        start_id = dec_tokenizer.pad_token_id

    if start_id is None:
        raise ValueError("Decoder tokenizer has no bos/eos/pad token id.")

    ids = torch.full(
        (B, 1),
        start_id,
        dtype=torch.long,
        device=device,
    )

    dummy_tgt_mask = torch.ones_like(ids)

    out = predictor(
        image_seq,
        enc_ids,
        enc_mask,
        ids,
        dummy_tgt_mask,
    )

    mem = out["text_memory"]

    eos_id = dec_tokenizer.eos_token_id
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len):
        attn = torch.ones_like(ids)

        dec_out = predictor.text_decoder(
            input_ids=ids,
            attention_mask=attn,
            encoder_hidden_states=mem,
        )

        next_id = dec_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        ids = torch.cat([ids, next_id], dim=1)

        if eos_id is not None:
            finished = finished | (next_id.squeeze(1) == eos_id)

            if bool(finished.all()):
                break

    return [
        dec_tokenizer.decode(ids[b, 1:], skip_special_tokens=True)
        for b in range(B)
    ]


# =========================================================
# 1. Temporal attention over the T input frames
# =========================================================

@torch.no_grad()
def plot_temporal_attention(
    predictor,
    dataloader,
    dec_tokenizer,
    device,
    max_samples=8,
):
    """
    Plots visual/text temporal attention over the input frames.
    """
    predictor.eval()

    batch = next(iter(dataloader))
    frames, _, text_dict = _unpack(batch)

    frames = frames.to(device)
    enc_ids, enc_mask, _, _ = _get_text_tensors(text_dict, device)

    dummy_ids, dummy_mask = _make_dummy_decoder_input(
        batch_size=frames.size(0),
        dec_tokenizer=dec_tokenizer,
        device=device,
    )

    out = predictor(
        frames,
        enc_ids,
        enc_mask,
        dummy_ids,
        dummy_mask,
    )

    v = out["visual_attn_weights"].detach().cpu().numpy()
    t = out["text_attn_weights"].detach().cpu().numpy()

    n = min(max_samples, v.shape[0])
    T = v.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 0.5 * n + 2))

    for ax, mat, title in [
        (axes[0], v[:n], "Visual"),
        (axes[1], t[:n], "Text"),
    ]:
        im = ax.imshow(mat, cmap="viridis", aspect="auto")
        ax.set_title(f"{title} attention over input frames")
        ax.set_xlabel("input frame index")
        ax.set_ylabel("sample")
        ax.set_xticks(range(T))

        plt.colorbar(im, ax=ax, fraction=0.046)

        for i in range(mat.shape[0]):
            for j in range(T):
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=7,
                )

    plt.tight_layout()
    plt.show()
    plt.close(fig)

    plt.figure(figsize=(7, 3))

    width = 0.4
    x = np.arange(T)

    plt.bar(x - width / 2, v.mean(0), width, label="visual")
    plt.bar(x + width / 2, t.mean(0), width, label="text")

    plt.xticks(x, [f"frame {i + 1}" for i in range(T)])
    plt.ylabel("mean attention")
    plt.title("Average temporal attention")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


# =========================================================
# 2. CLIP patch-attention overlay on an input frame
# =========================================================

@torch.no_grad()
def plot_clip_attention(
    predictor,
    dataloader,
    device,
    layer_id=-1,
    head=None,
    alpha=0.6,
    frame_index=0,
):
    """
    Plots CLIP ViT patch attention overlay for one input frame.

    Uses predictor.visual_clip.clip.vision_model attentions.
    """
    predictor.eval()

    visual_clip = predictor.visual_clip
    clip_model = visual_clip.clip

    try:
        clip_model.vision_model.config._attn_implementation = "eager"
    except Exception:
        pass

    batch = next(iter(dataloader))
    frames, _, _ = _unpack(batch)

    img01 = frames[0, frame_index].to(device)

    x = F.interpolate(
        img01.unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )

    # Prefer your visual encoder's own preprocessing if available.
    if hasattr(visual_clip, "_prep"):
        x_norm = visual_clip._prep(x)
    else:
        mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073],
            device=device,
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711],
            device=device,
        ).view(1, 3, 1, 1)

        x_norm = (x - mean) / std

    vis = clip_model.vision_model(
        pixel_values=x_norm,
        output_attentions=True,
        return_dict=True,
    )

    if vis.attentions is None:
        raise RuntimeError(
            "CLIP did not return attentions. Make sure output_attentions=True works "
            "for your installed transformers version."
        )

    attn = vis.attentions[layer_id][0]  # [heads, tokens, tokens]

    if head is None:
        attn = attn.mean(0)
        head_label = "avg heads"
    else:
        attn = attn[head]
        head_label = f"head {head}"

    cls_to_patches = attn[0, 1:]

    grid_size = int(np.sqrt(cls_to_patches.shape[0]))

    if grid_size * grid_size != cls_to_patches.shape[0]:
        raise ValueError(
            f"Patch count {cls_to_patches.shape[0]} is not a square number."
        )

    heat = cls_to_patches.reshape(grid_size, grid_size)

    heat = F.interpolate(
        heat[None, None],
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat = heat.detach().cpu().numpy()

    base = x[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(base)
    ax[0].set_title(f"input frame {frame_index + 1}")
    ax[0].axis("off")

    ax[1].imshow(heat, cmap="jet")
    ax[1].set_title(f"CLIP attention\nlayer {layer_id}, {head_label}")
    ax[1].axis("off")

    ax[2].imshow(base)
    ax[2].imshow(heat, cmap="jet", alpha=alpha)
    ax[2].set_title("overlay")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


# =========================================================
# 3. Cross-modal alignment
# =========================================================

@torch.no_grad()
def plot_cross_modal_alignment(
    predictor,
    dataloader,
    dec_tokenizer,
    device,
):
    """
    Plots cosine similarity between pooled visual and text features.
    """
    predictor.eval()

    batch = next(iter(dataloader))
    frames, _, text_dict = _unpack(batch)

    frames = frames.to(device)
    enc_ids, enc_mask, _, _ = _get_text_tensors(text_dict, device)

    dummy_ids, dummy_mask = _make_dummy_decoder_input(
        batch_size=frames.size(0),
        dec_tokenizer=dec_tokenizer,
        device=device,
    )

    out = predictor(
        frames,
        enc_ids,
        enc_mask,
        dummy_ids,
        dummy_mask,
    )

    zi = F.normalize(out["v_pool"], dim=-1)
    zt = F.normalize(out["t_pool"], dim=-1)

    sim = (zi @ zt.t()).detach().cpu().numpy()

    plt.figure(figsize=(7, 6))

    im = plt.imshow(sim, cmap="coolwarm", vmin=-1, vmax=1)

    plt.colorbar(im, label="cosine similarity")
    plt.title("Cross-modal alignment: visual rows vs text columns")
    plt.xlabel("text sample")
    plt.ylabel("visual sample")

    plt.tight_layout()
    plt.show()
    plt.close()


# =========================================================
# 4. Predicted-frame figures + error map + text
# =========================================================

@torch.no_grad()
def show_predictions(
    predictor,
    diffusion,
    dataloader,
    dec_tokenizer,
    device,
    n=3,
    ddim_steps=100,
    max_text_len=80,
    show_true_cond=True,
):
    """
    Shows test predictions.

    For each sample:
        input frames
        true next frame
        DDIM pure noise + PRED condition
        optional DDIM pure noise + TRUE condition
        error map
        GT text and predicted text

    No frame-4 generation is used.
    """
    predictor.eval()
    diffusion.model.eval()

    batch = next(iter(dataloader))

    frames, image_target, text_dict = _unpack(batch)

    frames = frames.to(device)
    image_target = image_target.to(device)

    enc_ids, enc_mask, tgt_ids, tgt_mask = _get_text_tensors(text_dict, device)

    n = min(n, frames.size(0))

    frames_n = frames[:n]
    image_target_n = image_target[:n]
    enc_ids_n = enc_ids[:n]
    enc_mask_n = enc_mask[:n]
    tgt_ids_n = tgt_ids[:n]
    tgt_mask_n = tgt_mask[:n]

    out = predictor(
        frames_n,
        enc_ids_n,
        enc_mask_n,
        tgt_ids_n,
        tgt_mask_n,
    )

    pred_cond = out["pred_image_cond"]

    pred_frames = generate_frames_ddim_cond(
        diffusion=diffusion,
        pred_cond=pred_cond,
        image_size=GEN_SIZE,
        steps=ddim_steps,
        eta=0.0,
    )

    true_frames = None

    if show_true_cond:
        true_cond = predictor.target_image_cond(image_target_n)

        true_frames = generate_frames_ddim_cond(
            diffusion=diffusion,
            pred_cond=true_cond,
            image_size=GEN_SIZE,
            steps=ddim_steps,
            eta=0.0,
        )

    pred_texts = generate_text(
        predictor=predictor,
        image_seq=frames_n,
        enc_ids=enc_ids_n,
        enc_mask=enc_mask_n,
        dec_tokenizer=dec_tokenizer,
        device=device,
        max_len=max_text_len,
    )

    T = frames.size(1)

    for i in range(n):
        gt_img = F.interpolate(
            image_target_n[i:i + 1],
            size=(GEN_SIZE, GEN_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0]

        pred_img = pred_frames[i]

        gt_np = gt_img.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        pred_np = pred_img.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)

        err = np.abs(gt_np - pred_np).mean(axis=2)

        extra_cols = 4 if show_true_cond else 3

        fig, axes = plt.subplots(
            1,
            T + extra_cols,
            figsize=(3 * (T + extra_cols), 3),
        )

        for f in range(T):
            img = frames_n[i, f].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)

            axes[f].imshow(img)
            axes[f].set_title(f"input {f + 1}")
            axes[f].axis("off")

        target_col = T
        pred_col = T + 1

        axes[target_col].imshow(gt_np)
        axes[target_col].set_title("true next")
        axes[target_col].axis("off")

        axes[pred_col].imshow(pred_np)
        axes[pred_col].set_title(f"DDIM pred\n{ddim_steps} steps")
        axes[pred_col].axis("off")

        if show_true_cond:
            true_col = T + 2
            error_col = T + 3

            true_np = true_frames[i].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)

            axes[true_col].imshow(true_np)
            axes[true_col].set_title("DDIM true cond\nupper bound")
            axes[true_col].axis("off")
        else:
            error_col = T + 2

        im = axes[error_col].imshow(err, cmap="hot")
        axes[error_col].set_title("error map")
        axes[error_col].axis("off")

        plt.colorbar(im, ax=axes[error_col], fraction=0.046)

        plt.tight_layout()
        plt.show()
        plt.close(fig)

        gt_text = dec_tokenizer.decode(
            tgt_ids_n[i],
            skip_special_tokens=True,
        )

        print(f"--- sample {i} ---")
        print("GT text:")
        print(gt_text)
        print("\nPredicted text:")
        print(pred_texts[i])
        print()
import os
import math
import textwrap
import contextlib

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm.auto import tqdm


# =========================================================
# Config
# =========================================================

DRIVE_CHECKPOINT_FOLDER = "/content/drive/MyDrive/DL_Checkpoints"
GEN_SIZE = 224


# =========================================================
# Early stopping
# =========================================================

class EarlyStopping:
    """
    Early stops the training if validation loss does not improve after patience.
    """
    def __init__(self, patience=5, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = None
        self.num_bad_epochs = 0
        self.stop = False

    def step(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss

        elif current_loss < self.best_loss - self.min_delta:
            if self.verbose:
                print(
                    f"Loss improved from {self.best_loss:.4f} "
                    f"to {current_loss:.4f}. Resetting patience."
                )
            self.best_loss = current_loss
            self.num_bad_epochs = 0

        else:
            self.num_bad_epochs += 1

            if self.verbose:
                print(
                    f"Loss did not improve. "
                    f"Patience: {self.num_bad_epochs}/{self.patience}"
                )

            if self.num_bad_epochs >= self.patience:
                self.stop = True

        return self.stop


# =========================================================
# History
# =========================================================

def make_predictor_history():
    keys = [
        "epoch",
        "train_loss",
        "text_loss",
        "image_loss",
        "bleu",
        "meteor",
        "rougeL",
        "mse",
        "ssim",
        "lpips",
        "psnr",
    ]
    return {k: [] for k in keys}


# =========================================================
# Checkpointing
# =========================================================

def save_predictor_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    history,
    filename="sequence_predictor.pth",
):
    os.makedirs(DRIVE_CHECKPOINT_FOLDER, exist_ok=True)
    path = os.path.join(DRIVE_CHECKPOINT_FOLDER, filename)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "history": history,
        },
        path,
    )

    print(f"✓ Predictor checkpoint saved: {path} (epoch {epoch})")


def load_predictor_checkpoint(
    model,
    optimizer=None,
    filename="sequence_predictor.pth",
):
    path = os.path.join(DRIVE_CHECKPOINT_FOLDER, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # weights_only=False is needed for older checkpoints containing numpy/python objects.
    # Only use this for your own trusted checkpoints.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    epoch = ckpt.get("epoch", 0)
    loss = ckpt.get("loss", float("inf"))
    history = ckpt.get("history", make_predictor_history())

    # Make history resume-safe.
    n = len(history.get("epoch", []))

    for k in make_predictor_history():
        history.setdefault(k, [])

        if k != "epoch":
            history[k] = (history[k] + [None] * n)[:n]

    print(f"✓ Predictor checkpoint loaded: {path} (epoch {epoch})")

    return model, optimizer, epoch, loss, history


# =========================================================
# Log export
# =========================================================

def export_predictor_log(history, filename="training_predictor.txt"):
    os.makedirs(DRIVE_CHECKPOINT_FOLDER, exist_ok=True)
    path = os.path.join(DRIVE_CHECKPOINT_FOLDER, filename)

    def f(v):
        return "  -   " if v is None else f"{v:.4f}"

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("=== SEQUENCE PREDICTOR TRAINING LOG ===\n\n")

        for i in range(len(history["epoch"])):
            fh.write(f"Epoch {history['epoch'][i]}\n")
            fh.write(
                f"  train_loss : {f(history['train_loss'][i])}"
                f"   text_loss : {f(history['text_loss'][i])}"
                f"   image_loss : {f(history['image_loss'][i])}\n"
            )
            fh.write(
                f"  TEXT   | BLEU {f(history['bleu'][i])}"
                f"  METEOR {f(history['meteor'][i])}"
                f"  ROUGE-L {f(history['rougeL'][i])}\n"
            )
            fh.write(
                f"  VISUAL | MSE {f(history['mse'][i])}"
                f"  SSIM {f(history['ssim'][i])}"
                f"  LPIPS {f(history['lpips'][i])}"
                f"  PSNR {f(history['psnr'][i])}\n\n"
            )

        if history["epoch"]:
            fh.write("=== SUMMARY ===\n")
            fh.write(f"Epochs run : {len(history['epoch'])}\n")

            tl = [x for x in history["train_loss"] if x is not None]

            if tl:
                best = min(tl)
                best_epoch = history["epoch"][history["train_loss"].index(best)]
                fh.write(f"Best train_loss : {best:.4f} (epoch {best_epoch})\n")

    print(f"✓ Predictor log saved: {path}")


# =========================================================
# Plot helpers
# =========================================================

def _xy(history, key):
    xs, ys = [], []

    for e, v in zip(history["epoch"], history.get(key, [])):
        if v is not None:
            xs.append(e)
            ys.append(v)

    return xs, ys


def plot_loss(history):
    if not history["epoch"]:
        return

    plt.figure(figsize=(9, 4))

    for key, label in [
        ("train_loss", "total"),
        ("text_loss", "text"),
        ("image_loss", "image"),
    ]:
        xs, ys = _xy(history, key)

        if xs:
            plt.plot(xs, ys, marker="o", markersize=3, label=label)

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Sequence predictor loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_text_metrics(history):
    if not history["epoch"]:
        return

    plt.figure(figsize=(9, 4))

    for key, label in [
        ("bleu", "BLEU"),
        ("meteor", "METEOR"),
        ("rougeL", "ROUGE-L"),
    ]:
        xs, ys = _xy(history, key)

        if xs:
            plt.plot(xs, ys, marker="o", markersize=3, label=label)

    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.title("Text metrics vs epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_visual_metrics(history):
    if not history["epoch"]:
        return

    fig, ax = plt.subplots(figsize=(9, 4))

    for key, label in [
        ("ssim", "SSIM"),
        ("lpips", "LPIPS"),
    ]:
        xs, ys = _xy(history, key)

        if xs:
            ax.plot(xs, ys, marker="o", markersize=3, label=label)

    ax.set_xlabel("epoch")
    ax.set_ylabel("SSIM / LPIPS")

    ax2 = ax.twinx()

    xs, ys = _xy(history, "mse")

    if xs:
        ax2.plot(xs, ys, marker="s", markersize=3, label="MSE")

    ax2.set_ylabel("MSE")

    lines = ax.get_lines() + ax2.get_lines()

    if lines:
        ax.legend(lines, [l.get_label() for l in lines], loc="best", fontsize=8)

    ax.set_title("Visual metrics vs epoch")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_all_metrics_bar(metrics):
    unit = [
        ("BLEU", metrics.get("bleu")),
        ("ROUGE-L", metrics.get("rougeL")),
        ("METEOR", metrics.get("meteor")),
        ("SSIM", metrics.get("ssim")),
        ("LPIPS", metrics.get("lpips")),
        ("MSE", metrics.get("mse")),
    ]

    unit = [(k, float(v)) for k, v in unit if v is not None]

    fig, ax = plt.subplots(figsize=(11, 6))

    labels = [k for k, _ in unit]
    vals = [v for _, v in unit]
    x = np.arange(len(labels))

    bars = ax.bar(x, vals)

    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_ylabel("score")

    tick_pos = list(x)
    tick_lab = list(labels)

    psnr = metrics.get("psnr")

    if psnr is not None:
        ax2 = ax.twinx()
        pos = len(labels)

        ax2.bar([pos], [float(psnr)], width=0.6)
        ax2.text(
            pos,
            float(psnr),
            f"{float(psnr):.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax2.set_ylabel("PSNR (dB)")

        tick_pos.append(pos)
        tick_lab.append("PSNR")

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    ax.set_title("Final evaluation metrics")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


# =========================================================
# Optional metric libraries
# =========================================================

try:
    from skimage.metrics import structural_similarity as _ssim
    from skimage.metrics import peak_signal_noise_ratio as _psnr
except Exception:
    _ssim = None
    _psnr = None

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _SMOOTH = SmoothingFunction().method1
except Exception:
    sentence_bleu = None
    _SMOOTH = None


# =========================================================
# Batch unpacking
# =========================================================

def _unpack(batch):
    """
    Your current dataloader returns a list of 9 items.
    The text dictionary is one of those items.
    """
    frames = batch[0]
    image_target = batch[1]
    text_dict = next(x for x in batch if isinstance(x, dict))

    return frames, image_target, text_dict


# =========================================================
# External condition helper
# =========================================================

@contextlib.contextmanager
def _external_condition(unet_model, cond_feat):
    """
    Temporarily forces CLIPDiffusionUNet.forward() to use an external condition.
    This does not change model weights.
    """
    prev = getattr(unet_model, "_external_cond", None)

    unet_model._external_cond = cond_feat

    try:
        yield

    finally:
        if prev is None:
            if hasattr(unet_model, "_external_cond"):
                delattr(unet_model, "_external_cond")
        else:
            unet_model._external_cond = prev


# =========================================================
# Visual generation
# =========================================================

@torch.no_grad()
def generate_frames(
    diffusion,
    pred_cond,
    image_size=GEN_SIZE,
):
    """
    Pure-noise generation.
    Keep this for debugging/comparison only.
    Your current diffusion model is weak at this path.
    """
    device = pred_cond.device

    x = torch.randn(
        pred_cond.size(0),
        3,
        image_size,
        image_size,
        device=device,
    )

    with _external_condition(diffusion.model, pred_cond):
        out = diffusion.sample(
            x_start=x,
            t_start=diffusion.T - 1,
        )

    return (out.clamp(-1, 1) + 1) / 2


@torch.no_grad()
def generate_frames_from_context(
    diffusion,
    pred_cond,
    context_frame,
    t_start=250,
    image_size=GEN_SIZE,
):
    """
    Better generation for your current pretrained visual model.

    Uses the 4th input frame as the starting visual canvas:
        frame_4 -> add noise -> denoise using predicted next-frame condition.

    This still predicts frame 5 from frames 1-4.
    It does not use the target frame as input.
    """
    device = pred_cond.device

    context_frame = F.interpolate(
        context_frame,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )

    x0 = context_frame * 2 - 1

    t_start = min(int(t_start), diffusion.T - 1)

    t = torch.full(
        (x0.size(0),),
        t_start,
        device=device,
        dtype=torch.long,
    )

    noise = torch.randn_like(x0)
    x_t = diffusion.q_sample(x0, t, noise=noise)

    with _external_condition(diffusion.model, pred_cond):
        out = diffusion.sample(
            x_start=x_t,
            t_start=t_start,
        )

    return (out.clamp(-1, 1) + 1) / 2


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
    max_len=40,
):
    """
    Greedy GPT-2 decode conditioned on predictor's predicted text memory.

    Fixed version:
    predictor.forward now expects target_ids and target_attention_mask.
    """
    predictor.eval()

    B = image_seq.size(0)

    start = dec_tokenizer.bos_token_id

    if start is None:
        start = dec_tokenizer.eos_token_id

    if start is None:
        start = dec_tokenizer.pad_token_id

    ids = torch.full(
        (B, 1),
        start,
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

    eos = dec_tokenizer.eos_token_id
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len):
        attn = torch.ones_like(ids)

        dec = predictor.text_decoder(
            input_ids=ids,
            attention_mask=attn,
            encoder_hidden_states=mem,
        )

        nxt = dec.logits[:, -1, :].argmax(-1, keepdim=True)

        ids = torch.cat([ids, nxt], dim=1)

        if eos is not None:
            finished = finished | (nxt.squeeze(1) == eos)

            if bool(finished.all()):
                break

    return [
        dec_tokenizer.decode(ids[b, 1:], skip_special_tokens=True)
        for b in range(B)
    ]


# =========================================================
# Evaluation
# =========================================================

@torch.no_grad()
def evaluate_predictor(
    predictor,
    diffusion,
    dataloader,
    dec_tokenizer,
    device,
    n_visual=1,
    lpips_fn=None,
    max_text_len=40,
    rouge_metric=None,
    meteor_metric=None,
    visual_t_start=250,
    use_context_generation=True,
):
    predictor.eval()

    pad_id = dec_tokenizer.pad_token_id

    if pad_id is None:
        pad_id = dec_tokenizer.eos_token_id

    preds_txt, gts_txt, bleu = [], [], []
    ssim_v, psnr_v, mse_v, lpips_v = [], [], [], []

    text_loss_sum = 0.0
    nb = 0
    vis_done = 0

    for batch in dataloader:
        frames, image_target, text_dict = _unpack(batch)

        frames = frames.to(device)
        image_target = image_target.to(device)

        enc_ids = text_dict["enc_input_ids"].to(device)
        enc_mask = text_dict["enc_attention_mask"].to(device)
        tgt_ids = text_dict["target_ids"].to(device)
        tgt_mask = text_dict["target_attention_mask"].to(device)

        out = predictor(
            frames,
            enc_ids,
            enc_mask,
            tgt_ids,
            tgt_mask,
        )

        logits = out["pred_text_logits"]
        V = logits.size(-1)

        text_loss_sum += F.cross_entropy(
            logits[:, :-1].reshape(-1, V),
            tgt_ids[:, 1:].reshape(-1),
            ignore_index=pad_id,
        ).item()

        nb += 1

        gen = generate_text(
            predictor,
            frames,
            enc_ids,
            enc_mask,
            dec_tokenizer,
            device,
            max_len=max_text_len,
        )

        for b in range(frames.size(0)):
            gt = dec_tokenizer.decode(
                tgt_ids[b],
                skip_special_tokens=True,
            )

            preds_txt.append(gen[b])
            gts_txt.append(gt)

            if sentence_bleu is not None:
                ref = gt.split()

                bleu.append(
                    sentence_bleu(
                        [ref],
                        gen[b].split(),
                        smoothing_function=_SMOOTH,
                    )
                    if ref else 0.0
                )

        if n_visual > 0 and vis_done < n_visual:
            take = min(frames.size(0), n_visual - vis_done)

            if use_context_generation:
                gen_frames = generate_frames_from_context(
                    diffusion=diffusion,
                    pred_cond=out["pred_image_cond"][:take],
                    context_frame=frames[:take, -1],
                    t_start=visual_t_start,
                    image_size=GEN_SIZE,
                )
            else:
                gen_frames = generate_frames(
                    diffusion=diffusion,
                    pred_cond=out["pred_image_cond"][:take],
                    image_size=GEN_SIZE,
                )

            tgt = F.interpolate(
                image_target[:take],
                size=(GEN_SIZE, GEN_SIZE),
                mode="bilinear",
                align_corners=False,
            )

            for b in range(take):
                g = gen_frames[b].detach().cpu().permute(1, 2, 0).numpy()
                t = tgt[b].detach().cpu().permute(1, 2, 0).numpy()

                mse_v.append(float(np.mean((g - t) ** 2)))

                if _ssim is not None:
                    try:
                        ssim_v.append(
                            _ssim(
                                t,
                                g,
                                channel_axis=2,
                                data_range=1.0,
                            )
                        )
                    except TypeError:
                        ssim_v.append(
                            _ssim(
                                t,
                                g,
                                multichannel=True,
                                data_range=1.0,
                            )
                        )

                if _psnr is not None:
                    psnr_v.append(
                        _psnr(
                            t,
                            g,
                            data_range=1.0,
                        )
                    )

            if lpips_fn is not None:
                lpips_v.append(
                    lpips_fn(
                        gen_frames * 2 - 1,
                        tgt * 2 - 1,
                    ).mean().item()
                )

            vis_done += take

    metrics = {
        "text_loss": text_loss_sum / max(1, nb),
        "bleu": float(np.mean(bleu)) if bleu else None,
        "meteor": None,
        "rougeL": None,
        "mse": float(np.mean(mse_v)) if mse_v else None,
        "ssim": float(np.mean(ssim_v)) if ssim_v else None,
        "psnr": float(np.mean(psnr_v)) if psnr_v else None,
        "lpips": float(np.mean(lpips_v)) if lpips_v else None,
    }

    if rouge_metric is not None and preds_txt:
        try:
            metrics["rougeL"] = rouge_metric.compute(
                predictions=preds_txt,
                references=gts_txt,
            )["rougeL"]
        except Exception:
            pass

    if meteor_metric is not None and preds_txt:
        try:
            metrics["meteor"] = meteor_metric.compute(
                predictions=preds_txt,
                references=gts_txt,
            )["meteor"]
        except Exception:
            pass

    return metrics


# =========================================================
# Validation display
# =========================================================

@torch.no_grad()
def show_validation_prediction_debug(
    predictor,
    diffusion,
    dataloader,
    dec_tokenizer,
    device,
    max_text_len=80,
    sample_idx=0,
    enc_tokenizer=None,
    visual_t_start=250,
    show_pure_noise_debug=False,
):
    """
    Shows:
    - Input 1-4 frames and texts
    - Target frame/text
    - Context + TRUE condition generation
    - Context + PRED condition generation

    This is the most useful validation view for your current model.
    """
    predictor.eval()
    diffusion.model.eval()

    batch = next(iter(dataloader))

    frames, image_target, text_dict = _unpack(batch)

    frames = frames.to(device)
    image_target = image_target.to(device)

    enc_ids = text_dict["enc_input_ids"].to(device)
    enc_mask = text_dict["enc_attention_mask"].to(device)
    tgt_ids = text_dict["target_ids"].to(device)
    tgt_mask = text_dict["target_attention_mask"].to(device)

    sample_idx = min(sample_idx, frames.size(0) - 1)

    out = predictor(
        frames,
        enc_ids,
        enc_mask,
        tgt_ids,
        tgt_mask,
    )

    pred_cond = out["pred_image_cond"]
    true_cond = predictor.target_image_cond(image_target)

    pred_latent = out["pred_image_latent"]
    true_latent = predictor.target_image_latent(image_target)

    def tensor_stats(name, x):
        x = x.detach().float()
        finite = torch.isfinite(x)

        if finite.any():
            xf = x[finite]

            print(
                f"{name}: "
                f"shape={tuple(x.shape)} | "
                f"min={xf.min().item():.4f} | "
                f"max={xf.max().item():.4f} | "
                f"mean={xf.mean().item():.4f} | "
                f"std={xf.std().item():.4f} | "
                f"nan={(~finite).sum().item()}"
            )
        else:
            print(f"{name}: all values are NaN/Inf")

    cond_mse = F.mse_loss(pred_cond, true_cond).item()
    cond_mae = F.l1_loss(pred_cond, true_cond).item()

    cond_cos = F.cosine_similarity(
        pred_cond.flatten(1),
        true_cond.flatten(1),
        dim=-1,
    ).mean().item()

    latent_cos = F.cosine_similarity(
        F.normalize(pred_latent, dim=-1),
        F.normalize(true_latent, dim=-1),
        dim=-1,
    ).mean().item()

    print("\n================ VALIDATION DEBUG ================")
    tensor_stats("pred_cond", pred_cond)
    tensor_stats("true_cond", true_cond)
    tensor_stats("pred_latent", pred_latent)
    tensor_stats("true_latent", true_latent)

    print(f"cond_mse     : {cond_mse:.6f}")
    print(f"cond_mae     : {cond_mae:.6f}")
    print(f"cond_cosine  : {cond_cos:.6f}")
    print(f"latent_cosine: {latent_cos:.6f}")

    last_context_frame = frames[sample_idx:sample_idx + 1, -1]

    pred_frame_context = generate_frames_from_context(
        diffusion=diffusion,
        pred_cond=pred_cond[sample_idx:sample_idx + 1],
        context_frame=last_context_frame,
        t_start=visual_t_start,
        image_size=GEN_SIZE,
    )[0]

    true_frame_context = generate_frames_from_context(
        diffusion=diffusion,
        pred_cond=true_cond[sample_idx:sample_idx + 1],
        context_frame=last_context_frame,
        t_start=visual_t_start,
        image_size=GEN_SIZE,
    )[0]

    tensor_stats("generated_context_pred_cond", pred_frame_context)
    tensor_stats("generated_context_true_cond", true_frame_context)
    tensor_stats("target_image", image_target[sample_idx])

    pure_true_frame = None
    pure_pred_frame = None

    if show_pure_noise_debug:
        pure_true_frame = generate_frames(
            diffusion=diffusion,
            pred_cond=true_cond[sample_idx:sample_idx + 1],
            image_size=GEN_SIZE,
        )[0]

        pure_pred_frame = generate_frames(
            diffusion=diffusion,
            pred_cond=pred_cond[sample_idx:sample_idx + 1],
            image_size=GEN_SIZE,
        )[0]

        tensor_stats("pure_noise_true_cond", pure_true_frame)
        tensor_stats("pure_noise_pred_cond", pure_pred_frame)

    # Text generation from already-computed memory
    mem = out["text_memory"][sample_idx:sample_idx + 1]

    start_id = dec_tokenizer.bos_token_id

    if start_id is None:
        start_id = dec_tokenizer.eos_token_id

    if start_id is None:
        start_id = dec_tokenizer.pad_token_id

    ids = torch.full(
        (1, 1),
        start_id,
        dtype=torch.long,
        device=device,
    )

    eos_id = dec_tokenizer.eos_token_id

    for _ in range(max_text_len):
        attn = torch.ones_like(ids)

        dec_out = predictor.text_decoder(
            input_ids=ids,
            attention_mask=attn,
            encoder_hidden_states=mem,
        )

        next_id = dec_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        ids = torch.cat([ids, next_id], dim=1)

        if eos_id is not None and next_id.item() == eos_id:
            break

    pred_text = dec_tokenizer.decode(
        ids[0, 1:],
        skip_special_tokens=True,
    )

    target_text = dec_tokenizer.decode(
        tgt_ids[sample_idx],
        skip_special_tokens=True,
    )

    input_tokenizer = enc_tokenizer if enc_tokenizer is not None else dec_tokenizer

    print("\nTarget text:")
    print(target_text)

    print("\nPredicted text:")
    print(pred_text)

    n_input = frames.shape[1]
    extra_cols = 3

    if show_pure_noise_debug:
        extra_cols = 5

    n_cols = n_input + extra_cols

    fig, ax = plt.subplots(
        2,
        n_cols,
        figsize=(4 * n_cols, 6),
        gridspec_kw={"height_ratios": [2, 1.4]},
    )

    def show_img(axis, img, title):
        img = img.detach().cpu()

        if img.dim() == 3:
            img = img.permute(1, 2, 0)

        img = img.clamp(0, 1)

        axis.imshow(img)
        axis.set_title(title)
        axis.axis("off")

    for t in range(n_input):
        show_img(
            ax[0, t],
            frames[sample_idx, t],
            f"Input {t + 1}",
        )

        input_text = input_tokenizer.decode(
            enc_ids[sample_idx, t],
            skip_special_tokens=True,
        )

        ax[1, t].text(
            0.5,
            0.95,
            textwrap.fill(input_text, width=35),
            ha="center",
            va="top",
            fontsize=9,
            wrap=True,
        )
        ax[1, t].axis("off")

    target_col = n_input
    true_col = n_input + 1
    pred_col = n_input + 2

    show_img(
        ax[0, target_col],
        image_target[sample_idx],
        "Target",
    )

    ax[1, target_col].text(
        0.5,
        0.95,
        textwrap.fill(target_text, width=35),
        ha="center",
        va="top",
        fontsize=9,
        wrap=True,
    )
    ax[1, target_col].axis("off")

    show_img(
        ax[0, true_col],
        true_frame_context,
        f"Frame 4 + TRUE cond\nt={visual_t_start}",
    )

    ax[1, true_col].text(
        0.5,
        0.95,
        "Uses frame 4 as starting canvas and the real target condition.",
        ha="center",
        va="top",
        fontsize=9,
        wrap=True,
    )
    ax[1, true_col].axis("off")

    show_img(
        ax[0, pred_col],
        pred_frame_context,
        f"Frame 4 + PRED cond\nt={visual_t_start}",
    )

    ax[1, pred_col].text(
        0.5,
        0.95,
        textwrap.fill(pred_text, width=35),
        ha="center",
        va="top",
        fontsize=9,
        wrap=True,
    )
    ax[1, pred_col].axis("off")

    if show_pure_noise_debug:
        pure_true_col = n_input + 3
        pure_pred_col = n_input + 4

        show_img(
            ax[0, pure_true_col],
            pure_true_frame,
            "Pure noise + TRUE cond",
        )

        ax[1, pure_true_col].text(
            0.5,
            0.95,
            "Debug only.",
            ha="center",
            va="top",
            fontsize=9,
            wrap=True,
        )
        ax[1, pure_true_col].axis("off")

        show_img(
            ax[0, pure_pred_col],
            pure_pred_frame,
            "Pure noise + PRED cond",
        )

        ax[1, pure_pred_col].text(
            0.5,
            0.95,
            "Debug only.",
            ha="center",
            va="top",
            fontsize=9,
            wrap=True,
        )
        ax[1, pure_pred_col].axis("off")

    plt.tight_layout()
    plt.show()


# =========================================================
# Text debugging
# =========================================================

@torch.no_grad()
def debug_text_prediction(
    predictor,
    dataloader,
    dec_tokenizer,
    device,
    sample_idx=0,
    max_text_len=80,
):
    predictor.eval()

    batch = next(iter(dataloader))

    frames, image_target, text_dict = _unpack(batch)

    frames = frames.to(device)

    enc_ids = text_dict["enc_input_ids"].to(device)
    enc_mask = text_dict["enc_attention_mask"].to(device)
    tgt_ids = text_dict["target_ids"].to(device)
    tgt_mask = text_dict["target_attention_mask"].to(device)

    sample_idx = min(sample_idx, frames.size(0) - 1)

    out = predictor(
        frames,
        enc_ids,
        enc_mask,
        tgt_ids,
        tgt_mask,
    )

    logits = out["pred_text_logits"]
    V = logits.size(-1)

    pad_id = dec_tokenizer.pad_token_id

    if pad_id is None:
        pad_id = dec_tokenizer.eos_token_id

    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, V),
        tgt_ids[:, 1:].reshape(-1),
        ignore_index=pad_id,
    )

    pred_tokens = logits[:, :-1].argmax(dim=-1)
    true_tokens = tgt_ids[:, 1:]

    valid = true_tokens != pad_id

    if valid.any():
        token_acc = (
            pred_tokens[valid] == true_tokens[valid]
        ).float().mean().item()
    else:
        token_acc = 0.0

    teacher_forced_pred_ids = pred_tokens[sample_idx]

    teacher_forced_pred_text = dec_tokenizer.decode(
        teacher_forced_pred_ids,
        skip_special_tokens=True,
    )

    target_text = dec_tokenizer.decode(
        tgt_ids[sample_idx],
        skip_special_tokens=True,
    )

    mem = out["text_memory"][sample_idx:sample_idx + 1]

    start_id = dec_tokenizer.bos_token_id

    if start_id is None:
        start_id = dec_tokenizer.eos_token_id

    if start_id is None:
        start_id = dec_tokenizer.pad_token_id

    ids = torch.full(
        (1, 1),
        start_id,
        dtype=torch.long,
        device=device,
    )

    eos_id = dec_tokenizer.eos_token_id

    for _ in range(max_text_len):
        attn = torch.ones_like(ids)

        dec_out = predictor.text_decoder(
            input_ids=ids,
            attention_mask=attn,
            encoder_hidden_states=mem,
        )

        next_id = dec_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        ids = torch.cat([ids, next_id], dim=1)

        if eos_id is not None and next_id.item() == eos_id:
            break

    free_generated_text = dec_tokenizer.decode(
        ids[0, 1:],
        skip_special_tokens=True,
    )

    print("\n================ TEXT DEBUG ================")
    print(f"text_loss/token CE : {loss.item():.4f}")
    print(f"teacher-forced token accuracy: {token_acc:.4f}")

    print("\nTARGET TEXT:")
    print(target_text)

    print("\nTEACHER-FORCED ARGMAX TEXT:")
    print(teacher_forced_pred_text)

    print("\nFREE-GENERATED TEXT:")
    print(free_generated_text)

    words = free_generated_text.lower().split()

    if len(words) >= 4:
        bigrams = list(zip(words[:-1], words[1:]))
        unique_bigram_ratio = len(set(bigrams)) / max(1, len(bigrams))
        print(f"\nunique_bigram_ratio: {unique_bigram_ratio:.4f}")

    print("===========================================\n")


# =========================================================
# Temporal dependency debugging
# =========================================================

@torch.no_grad()
def debug_temporal_dependency(
    predictor,
    dataloader,
    device,
    sample_idx=0,
):
    """
    Tests whether the predictor is really using frames/texts 1-3,
    or mostly relying on frame/text 4.
    """
    predictor.eval()

    batch = next(iter(dataloader))

    frames, image_target, text_dict = _unpack(batch)

    frames = frames.to(device)

    enc_ids = text_dict["enc_input_ids"].to(device)
    enc_mask = text_dict["enc_attention_mask"].to(device)
    tgt_ids = text_dict["target_ids"].to(device)
    tgt_mask = text_dict["target_attention_mask"].to(device)

    sample_idx = min(sample_idx, frames.size(0) - 1)

    def run_variant(frames_variant, enc_ids_variant, enc_mask_variant, name):
        out = predictor(
            frames_variant,
            enc_ids_variant,
            enc_mask_variant,
            tgt_ids,
            tgt_mask,
        )

        return {
            "name": name,
            "cond": out["pred_image_cond"],
            "latent": out["pred_image_latent"],
            "v_attn": out.get("visual_attn_weights", None),
            "t_attn": out.get("text_attn_weights", None),
        }

    original = run_variant(
        frames,
        enc_ids,
        enc_mask,
        "original",
    )

    frames_only_4 = frames.clone()
    frames_only_4[:, 0] = frames[:, 3]
    frames_only_4[:, 1] = frames[:, 3]
    frames_only_4[:, 2] = frames[:, 3]

    enc_ids_only_4 = enc_ids.clone()
    enc_mask_only_4 = enc_mask.clone()

    enc_ids_only_4[:, 0] = enc_ids[:, 3]
    enc_ids_only_4[:, 1] = enc_ids[:, 3]
    enc_ids_only_4[:, 2] = enc_ids[:, 3]

    enc_mask_only_4[:, 0] = enc_mask[:, 3]
    enc_mask_only_4[:, 1] = enc_mask[:, 3]
    enc_mask_only_4[:, 2] = enc_mask[:, 3]

    only_4 = run_variant(
        frames_only_4,
        enc_ids_only_4,
        enc_mask_only_4,
        "only_frame_text_4_repeated",
    )

    frames_no_4 = frames.clone()
    frames_no_4[:, 3] = frames[:, 2]

    enc_ids_no_4 = enc_ids.clone()
    enc_mask_no_4 = enc_mask.clone()

    enc_ids_no_4[:, 3] = enc_ids[:, 2]
    enc_mask_no_4[:, 3] = enc_mask[:, 2]

    no_4 = run_variant(
        frames_no_4,
        enc_ids_no_4,
        enc_mask_no_4,
        "no_frame_text_4",
    )

    perm = torch.tensor([2, 0, 1, 3], device=device)

    shuffled = run_variant(
        frames[:, perm],
        enc_ids[:, perm],
        enc_mask[:, perm],
        "shuffle_1_to_3_keep_4",
    )

    variants = [only_4, no_4, shuffled]

    base_cond = original["cond"].flatten(1)
    base_latent = original["latent"]

    print("\n================ TEMPORAL DEPENDENCY DEBUG ================")

    for v in variants:
        cond = v["cond"].flatten(1)
        latent = v["latent"]

        cond_cos = F.cosine_similarity(
            base_cond,
            cond,
            dim=-1,
        ).mean().item()

        cond_mse = F.mse_loss(
            original["cond"],
            v["cond"],
        ).item()

        latent_cos = F.cosine_similarity(
            F.normalize(base_latent, dim=-1),
            F.normalize(latent, dim=-1),
            dim=-1,
        ).mean().item()

        print(f"\nVariant: {v['name']}")
        print(f"cond cosine vs original  : {cond_cos:.6f}")
        print(f"cond MSE vs original     : {cond_mse:.6f}")
        print(f"latent cosine vs original: {latent_cos:.6f}")

    if original["v_attn"] is not None:
        print("\nVisual temporal attention for selected sample:")
        print(original["v_attn"][sample_idx].detach().cpu().numpy())

    if original["t_attn"] is not None:
        print("\nText temporal attention for selected sample:")
        print(original["t_attn"][sample_idx].detach().cpu().numpy())

    print("===========================================================\n")


# =========================================================
# Training loop
# =========================================================

def train_predictor(
    predictor,
    diffusion,
    train_loader,
    val_loader,
    dec_tokenizer,
    device,
    n_epochs=10,
    lr=2e-4,
    accum_steps=4,
    w_cond=1.0,
    w_latent=0.1,
    w_text=1.0,
    grad_clip=1.0,
    checkpoint_filename="sequence_predictor.pth",
    log_name="training_predictor.txt",
    resume=True,
    eval_n_visual=1,
    freeze_text_encoder=True,
    show_val_sample=True,
    val_sample_idx=0,
    val_max_text_len=80,
    enc_tokenizer=None,
    visual_t_start=250,
    show_text_debug=True,
    show_pure_noise_debug=False,
    run_temporal_debug_every=0,
):
    predictor.to(device)

    # Freeze visual CLIP feature extractor.
    for p in predictor.visual_clip.parameters():
        p.requires_grad = False

    # Freeze RoBERTa encoder if requested.
    if freeze_text_encoder:
        for p in predictor.text_encoder.parameters():
            p.requires_grad = False

    trainable = [p for p in predictor.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable,
        lr=lr,
    )

    use_cuda = (
        device.type == "cuda"
        if hasattr(device, "type")
        else str(device).startswith("cuda")
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_cuda,
    )

    pad_id = dec_tokenizer.pad_token_id

    if pad_id is None:
        pad_id = dec_tokenizer.eos_token_id

    # Optional LPIPS
    lpips_fn = None

    try:
        import lpips

        lpips_fn = lpips.LPIPS(net="alex").to(device).eval()

        for p in lpips_fn.parameters():
            p.requires_grad = False

        print("LPIPS enabled (net=alex).")

    except Exception:
        print("LPIPS unavailable - `pip install lpips` to enable. Using MSE/SSIM/PSNR.")

    # Optional HF metrics
    rouge_metric = None
    meteor_metric = None

    try:
        import evaluate

        rouge_metric = evaluate.load("rouge")
        meteor_metric = evaluate.load("meteor")

    except Exception:
        print("HF `evaluate` unavailable - METEOR/ROUGE-L will be skipped.")

    # Resume
    start_epoch = 0
    history = make_predictor_history()

    if resume:
        try:
            predictor, optimizer, start_epoch, _, history = load_predictor_checkpoint(
                predictor,
                optimizer,
                filename=checkpoint_filename,
            )

            print(f"Resuming from epoch {start_epoch + 1}")

        except FileNotFoundError:
            print("No predictor checkpoint - training from scratch.")

    for epoch in range(start_epoch, n_epochs):
        predictor.train()

        running = 0.0
        nb = 0

        last_cond = 0.0
        last_text = 0.0

        optimizer.zero_grad()

        pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{n_epochs}",
            leave=True,
        )

        for step, batch in pbar:
            frames, image_target, text_dict = _unpack(batch)

            frames = frames.to(device)
            image_target = image_target.to(device)

            enc_ids = text_dict["enc_input_ids"].to(device)
            enc_mask = text_dict["enc_attention_mask"].to(device)
            tgt_ids = text_dict["target_ids"].to(device)
            tgt_mask = text_dict["target_attention_mask"].to(device)

            with torch.amp.autocast("cuda", enabled=use_cuda):
                out = predictor(
                    frames,
                    enc_ids,
                    enc_mask,
                    tgt_ids,
                    tgt_mask,
                )

                with torch.no_grad():
                    true_cond = predictor.target_image_cond(image_target)
                    true_latent = predictor.target_image_latent(image_target)

                loss_cond = F.mse_loss(
                    out["pred_image_cond"],
                    true_cond,
                )

                pred_latent_norm = F.normalize(
                    out["pred_image_latent"],
                    dim=-1,
                )

                true_latent_norm = F.normalize(
                    true_latent,
                    dim=-1,
                )

                loss_latent = 1 - F.cosine_similarity(
                    pred_latent_norm,
                    true_latent_norm,
                    dim=-1,
                ).mean()

                logits = out["pred_text_logits"]
                V = logits.size(-1)

                loss_text = F.cross_entropy(
                    logits[:, :-1].reshape(-1, V),
                    tgt_ids[:, 1:].reshape(-1),
                    ignore_index=pad_id,
                )

                loss = (
                    w_cond * loss_cond
                    + w_latent * loss_latent
                    + w_text * loss_text
                ) / accum_steps

            scaler.scale(loss).backward()

            running += loss.item() * accum_steps
            nb += 1

            last_cond = float(loss_cond.item())
            last_text = float(loss_text.item())

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    grad_clip,
                )

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            pbar.set_postfix(
                {
                    "loss": f"{running / max(1, nb):.4f}",
                    "cond": f"{last_cond:.4f}",
                    "text": f"{last_text:.4f}",
                    "accum": f"{(step % accum_steps) + 1}/{accum_steps}",
                }
            )

        # Handle leftover gradients when dataloader length is not divisible by accum_steps.
        if len(train_loader) % accum_steps != 0:
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                trainable,
                grad_clip,
            )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss = running / max(1, nb)

        # Show validation sample before slower full metrics.
        if show_val_sample:
            print("\nShowing validation prediction...")

            show_validation_prediction_debug(
                predictor=predictor,
                diffusion=diffusion,
                dataloader=val_loader,
                dec_tokenizer=dec_tokenizer,
                device=device,
                max_text_len=val_max_text_len,
                sample_idx=val_sample_idx,
                enc_tokenizer=enc_tokenizer,
                visual_t_start=visual_t_start,
                show_pure_noise_debug=show_pure_noise_debug,
            )

            if show_text_debug:
                debug_text_prediction(
                    predictor=predictor,
                    dataloader=val_loader,
                    dec_tokenizer=dec_tokenizer,
                    device=device,
                    sample_idx=val_sample_idx,
                    max_text_len=val_max_text_len,
                )

            if run_temporal_debug_every and ((epoch + 1) % run_temporal_debug_every == 0):
                debug_temporal_dependency(
                    predictor=predictor,
                    dataloader=val_loader,
                    device=device,
                    sample_idx=val_sample_idx,
                )

        print("Running validation metrics...")

        metrics = evaluate_predictor(
            predictor=predictor,
            diffusion=diffusion,
            dataloader=val_loader,
            dec_tokenizer=dec_tokenizer,
            device=device,
            n_visual=eval_n_visual,
            lpips_fn=lpips_fn,
            rouge_metric=rouge_metric,
            meteor_metric=meteor_metric,
            visual_t_start=visual_t_start,
            use_context_generation=True,
        )

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["text_loss"].append(metrics.get("text_loss"))
        history["image_loss"].append(last_cond)

        for k in [
            "bleu",
            "meteor",
            "rougeL",
            "mse",
            "ssim",
            "lpips",
            "psnr",
        ]:
            history[k].append(metrics.get(k))

        def _f(v):
            return "n/a" if v is None else f"{v:.4f}"

        print(
            f"Epoch {epoch + 1}/{n_epochs} | "
            f"loss {train_loss:.4f} "
            f"(cond {last_cond:.4f}, text {last_text:.4f}) | "
            f"BLEU {_f(metrics.get('bleu'))} "
            f"ROUGE-L {_f(metrics.get('rougeL'))} "
            f"METEOR {_f(metrics.get('meteor'))} | "
            f"SSIM {_f(metrics.get('ssim'))} "
            f"PSNR {_f(metrics.get('psnr'))} "
            f"LPIPS {_f(metrics.get('lpips'))}"
        )

        save_predictor_checkpoint(
            predictor,
            optimizer,
            epoch + 1,
            train_loss,
            history,
            filename=checkpoint_filename,
        )

        export_predictor_log(
            history,
            filename=log_name,
        )

        plot_loss(history)
        plot_text_metrics(history)
        plot_visual_metrics(history)

    if history["epoch"]:
        last = {
            k: history[k][-1]
            for k in [
                "bleu",
                "rougeL",
                "meteor",
                "ssim",
                "lpips",
                "mse",
                "psnr",
            ]
            if history.get(k) and history[k][-1] is not None
        }

        plot_all_metrics_bar(last)

    return predictor, optimizer, history
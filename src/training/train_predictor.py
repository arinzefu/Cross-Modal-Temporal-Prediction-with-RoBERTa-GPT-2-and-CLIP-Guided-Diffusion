


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
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
                print(f"Loss improved from {self.best_loss:.4f} to {current_loss:.4f}. Resetting patience.")
            self.best_loss = current_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.verbose:
                print(f"Loss did not improve. Patience: {self.num_bad_epochs}/{self.patience}")
            if self.num_bad_epochs >= self.patience:
                self.stop = True
        return self.stop

import contextlib

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from train_predictor_logging import (
    make_predictor_history,
    save_predictor_checkpoint,
    load_predictor_checkpoint,
    export_predictor_log,
    plot_loss,
    plot_text_metrics,
    plot_visual_metrics,
    plot_all_metrics_bar,
)

# ---- optional metric libraries (all degrade gracefully) ----
try:
    from skimage.metrics import structural_similarity as _ssim
    from skimage.metrics import peak_signal_noise_ratio as _psnr
except Exception:
    _ssim = _psnr = None

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _SMOOTH = SmoothingFunction().method1
except Exception:
    sentence_bleu = None
    _SMOOTH = None

GEN_SIZE = 224   # diffusion was trained at 224; generate + score at this resolution


# =========================================================
# Batch unpacking (robust to the long CoT tuple or a short one)
# =========================================================
def _unpack(batch):
    frames = batch[0]
    image_target = batch[1]
    text_dict = next(x for x in batch if isinstance(x, dict))
    return frames, image_target, text_dict


# =========================================================
# Generation: feed the predicted condition into the FROZEN diffusion model
# =========================================================
@contextlib.contextmanager
def _external_condition(unet_model, cond_feat):
    """Make CLIPDiffusionUNet.forward use `cond_feat` for the duration of the block,
    via the `_external_cond` attribute the hook reads. Reuses GaussianDiffusion.sample
    unchanged."""
    prev = getattr(unet_model, "_external_cond", None)
    unet_model._external_cond = cond_feat
    try:
        yield
    finally:
        unet_model._external_cond = prev


@torch.no_grad()
def generate_frames(diffusion, pred_cond, image_size=GEN_SIZE):
    """Denoise from pure noise under the predicted condition. Returns [B,3,H,W] in [0,1]."""
    device = pred_cond.device
    x = torch.randn(pred_cond.size(0), 3, image_size, image_size, device=device)
    with _external_condition(diffusion.model, pred_cond):
        out = diffusion.sample(x_start=x, t_start=diffusion.T - 1)   # uses the hook
    return (out.clamp(-1, 1) + 1) / 2


@torch.no_grad()
def generate_text(predictor, image_seq, enc_ids, enc_mask, dec_tokenizer,
                  device, max_len=40):
    """Greedy GPT-2 decode conditioned on the predictor's predicted text memory."""
    predictor.eval()
    B = image_seq.size(0)
    start = dec_tokenizer.bos_token_id
    if start is None:
        start = dec_tokenizer.eos_token_id
    ids = torch.full((B, 1), start, dtype=torch.long, device=device)

    # one forward pass to obtain the predicted cross-attn memory
    out = predictor(image_seq, enc_ids, enc_mask, ids)
    mem = out["text_memory"]

    eos = dec_tokenizer.eos_token_id
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_len):
        dec = predictor.text_decoder(input_ids=ids, encoder_hidden_states=mem)
        nxt = dec.logits[:, -1, :].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        if eos is not None:
            finished = finished | (nxt.squeeze(1) == eos)
            if bool(finished.all()):
                break
    return [dec_tokenizer.decode(ids[b, 1:], skip_special_tokens=True) for b in range(B)]


# =========================================================
# Evaluation -> metrics dict consumed by the logging module
# =========================================================
@torch.no_grad()
def evaluate_predictor(predictor, diffusion, dataloader, dec_tokenizer, device,
                       n_visual=8, lpips_fn=None, max_text_len=40,
                       rouge_metric=None, meteor_metric=None):
    predictor.eval()
    pad_id = dec_tokenizer.pad_token_id
    if pad_id is None:
        pad_id = dec_tokenizer.eos_token_id

    preds_txt, gts_txt, bleu = [], [], []
    ssim_v, psnr_v, mse_v, lpips_v = [], [], [], []
    text_loss_sum, nb = 0.0, 0
    vis_done = 0

    for batch in dataloader:
        frames, image_target, text_dict = _unpack(batch)
        frames = frames.to(device)
        image_target = image_target.to(device)
        enc_ids = text_dict["input_ids"].to(device)
        enc_mask = text_dict["attention_mask"].to(device)
        tgt_ids = text_dict["target_ids"].to(device)

        out = predictor(frames, enc_ids, enc_mask, tgt_ids)

        # teacher-forced text loss (shifted)
        logits = out["pred_text_logits"]
        V = logits.size(-1)
        text_loss_sum += F.cross_entropy(
            logits[:, :-1].reshape(-1, V), tgt_ids[:, 1:].reshape(-1),
            ignore_index=pad_id,
        ).item()
        nb += 1

        # text generation metrics
        gen = generate_text(predictor, frames, enc_ids, enc_mask, dec_tokenizer, device, max_text_len)
        for b in range(frames.size(0)):
            gt = dec_tokenizer.decode(tgt_ids[b], skip_special_tokens=True)
            preds_txt.append(gen[b]); gts_txt.append(gt)
            if sentence_bleu is not None:
                ref = gt.split()
                bleu.append(sentence_bleu([ref], gen[b].split(), smoothing_function=_SMOOTH) if ref else 0.0)

        # visual generation metrics on a capped subset (sampling is expensive)
        if vis_done < n_visual:
            take = min(frames.size(0), n_visual - vis_done)
            gen_frames = generate_frames(diffusion, out["pred_image_cond"][:take], GEN_SIZE)
            tgt = F.interpolate(image_target[:take], size=(GEN_SIZE, GEN_SIZE),
                                mode="bilinear", align_corners=False)
            for b in range(take):
                g = gen_frames[b].cpu().permute(1, 2, 0).numpy()
                t = tgt[b].cpu().permute(1, 2, 0).numpy()
                mse_v.append(float(np.mean((g - t) ** 2)))
                if _ssim is not None:
                    try:
                        ssim_v.append(_ssim(t, g, channel_axis=2, data_range=1.0))
                    except TypeError:
                        ssim_v.append(_ssim(t, g, multichannel=True, data_range=1.0))
                    psnr_v.append(_psnr(t, g, data_range=1.0))
            if lpips_fn is not None:
                lpips_v.append(lpips_fn(gen_frames * 2 - 1, tgt * 2 - 1).mean().item())
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
            metrics["rougeL"] = rouge_metric.compute(predictions=preds_txt, references=gts_txt)["rougeL"]
        except Exception:
            pass
    if meteor_metric is not None and preds_txt:
        try:
            metrics["meteor"] = meteor_metric.compute(predictions=preds_txt, references=gts_txt)["meteor"]
        except Exception:
            pass
    return metrics


# =========================================================
# Training loop  (latent/cond + text losses, AMP, grad-accum, resume, logging)
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
    w_cond=1.0,           # drives generation quality (the real visual objective)
    w_latent=0.1,         # aux semantic latent (cosine)
    w_text=1.0,
    grad_clip=1.0,
    checkpoint_filename="sequence_predictor.pth",
    log_name="training_predictor.txt",
    resume=True,
    eval_n_visual=8,
    freeze_text_encoder=True,
):
    predictor.to(device)

    # ── freeze the frozen generator + (by default) the RoBERTa encoder ──
    for p in predictor.visual_clip.parameters():
        p.requires_grad = False
    if freeze_text_encoder:
        for p in predictor.text_encoder.parameters():
            p.requires_grad = False
    # everything else (projectors, temporal, fusion, dynamics, heads, GPT-2 decoder
    # incl. its randomly-initialised cross-attention) stays trainable.

    trainable = [p for p in predictor.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    scaler = GradScaler()

    pad_id = dec_tokenizer.pad_token_id
    if pad_id is None:
        pad_id = dec_tokenizer.eos_token_id

    # ── optional LPIPS + HF metrics, loaded once ──
    lpips_fn = None
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False
        print("LPIPS enabled (net=alex).")
    except Exception:
        print("LPIPS unavailable - `pip install lpips` to enable. Using MSE/SSIM/PSNR.")

    rouge_metric = meteor_metric = None
    try:
        import evaluate
        rouge_metric = evaluate.load("rouge")
        meteor_metric = evaluate.load("meteor")
    except Exception:
        print("HF `evaluate` unavailable - METEOR/ROUGE-L will be skipped (BLEU still computed).")

    # ── resume ──
    start_epoch = 0
    history = make_predictor_history()
    if resume:
        try:
            predictor, optimizer, start_epoch, _, history = load_predictor_checkpoint(
                predictor, optimizer, filename=checkpoint_filename)
            print(f"Resuming from epoch {start_epoch + 1}")
        except FileNotFoundError:
            print("No predictor checkpoint - training from scratch.")

    for epoch in range(start_epoch, n_epochs):
        predictor.train()
        running, nb = 0.0, 0
        last_cond = last_text = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            frames, image_target, text_dict = _unpack(batch)
            frames = frames.to(device)
            image_target = image_target.to(device)
            enc_ids = text_dict["input_ids"].to(device)
            enc_mask = text_dict["attention_mask"].to(device)
            tgt_ids = text_dict["target_ids"].to(device)

            with autocast():
                out = predictor(frames, enc_ids, enc_mask, tgt_ids)

                with torch.no_grad():
                    true_cond = predictor.target_image_cond(image_target)     # [B,Cc,hc,wc]
                    true_latent = predictor.target_image_latent(image_target)  # [B,512]

                loss_cond = F.mse_loss(out["pred_image_cond"], true_cond)
                loss_latent = 1 - F.cosine_similarity(
                    out["pred_image_latent"], true_latent, dim=-1).mean()

                logits = out["pred_text_logits"]
                V = logits.size(-1)
                loss_text = F.cross_entropy(
                    logits[:, :-1].reshape(-1, V), tgt_ids[:, 1:].reshape(-1),
                    ignore_index=pad_id)

                loss = (w_cond * loss_cond + w_latent * loss_latent + w_text * loss_text) / accum_steps

            scaler.scale(loss).backward()
            running += loss.item() * accum_steps
            nb += 1
            last_cond, last_text = float(loss_cond.item()), float(loss_text.item())

            if (step + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        train_loss = running / max(1, nb)

        # ── per-epoch evaluation (real pixel + text metrics) ──
        metrics = evaluate_predictor(
            predictor, diffusion, val_loader, dec_tokenizer, device,
            n_visual=eval_n_visual, lpips_fn=lpips_fn,
            rouge_metric=rouge_metric, meteor_metric=meteor_metric)

        # ── record history (resume-safe; drives the graphs) ──
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["text_loss"].append(metrics.get("text_loss"))
        history["image_loss"].append(last_cond)
        for k in ["bleu", "meteor", "rougeL", "mse", "ssim", "lpips", "psnr"]:
            history[k].append(metrics.get(k))

        def _f(v):
            return "n/a" if v is None else f"{v:.4f}"
        print(f"Epoch {epoch+1}/{n_epochs} | loss {train_loss:.4f} "
              f"(cond {last_cond:.4f}, text {last_text:.4f}) | "
              f"BLEU {_f(metrics.get('bleu'))} ROUGE-L {_f(metrics.get('rougeL'))} "
              f"METEOR {_f(metrics.get('meteor'))} | SSIM {_f(metrics.get('ssim'))} "
              f"PSNR {_f(metrics.get('psnr'))} LPIPS {_f(metrics.get('lpips'))}")

        # ── save + log + the three graphs (continue across resume) ──
        save_predictor_checkpoint(predictor, optimizer, epoch + 1, train_loss, history,
                                  filename=checkpoint_filename)
        export_predictor_log(history, filename=log_name)
        plot_loss(history)
        plot_text_metrics(history)
        plot_visual_metrics(history)

    # ── final all-metrics bar chart (latest epoch) ──
    if history["epoch"]:
        last = {k: history[k][-1] for k in
                ["bleu", "rougeL", "meteor", "ssim", "lpips", "mse", "psnr"]
                if history.get(k) and history[k][-1] is not None}
        plot_all_metrics_bar(last)

    return predictor, optimizer, history

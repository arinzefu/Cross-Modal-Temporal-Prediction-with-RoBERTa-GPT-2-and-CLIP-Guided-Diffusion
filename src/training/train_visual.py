# =========================================================
# pretrain_visual_autoencoder.py


import os
import csv
import math

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# tqdm with a no-op fallback so the loop still runs if tqdm isn't installed.
try:
    from tqdm.auto import tqdm
except Exception:
    class _NoTqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable if iterable is not None else []

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, *a, **k):
            pass

        def set_description(self, *a, **k):
            pass

        def close(self):
            pass

    def tqdm(iterable=None, **kwargs):
        return _NoTqdm(iterable, **kwargs)


# =========================================================
# Fallback utilities (override via args if you already have them)
# =========================================================

class EarlyStopping:
    """Minimal early stopping on a minimised metric. `.step(loss) -> stop?`"""

    def __init__(self, patience=10, min_delta=0.0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            return False
        self.counter += 1
        if self.verbose:
            print(f"  EarlyStopping: {self.counter}/{self.patience} epochs without improvement")
        return self.counter >= self.patience


def _default_save_checkpoint(model, optimizer, epoch, loss,
                             filename="visual_autoencoder.pth"):
    drive_folder = "/content/drive/MyDrive/DL_Checkpoints"
    os.makedirs(drive_folder, exist_ok=True)
    full_path = os.path.join(drive_folder, filename)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        full_path,
    )
    print(f"Checkpoint saved to Google Drive: {full_path} at epoch {epoch}")


def _default_load_checkpoint(model, optimizer=None,
                             filename="visual_autoencoder.pth"):
    drive_folder = "/content/drive/MyDrive/DL_Checkpoints"
    full_path = os.path.join(drive_folder, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Checkpoint file not found: {full_path}")
    checkpoint = torch.load(full_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", None)
    print(f"Checkpoint loaded from: {full_path} (epoch {epoch})")
    return model, optimizer, epoch, loss


def _build_default_criterion(device):
    try:
        from visual_autoencoder import ReconstructionLoss
    except Exception as e:
        raise ImportError(
            "Could not import ReconstructionLoss automatically. Pass criterion=...\n"
            "    criterion = ReconstructionLoss(pixel_weight=0.8, perceptual_weight=0.2)"
        ) from e
    return ReconstructionLoss(pixel_weight=0.8, perceptual_weight=0.2).to(device)


# =========================================================
# Metrics
# =========================================================

def _build_metrics(device):
    """Returns (ssim_metric, lpips_metric); either may be None if unavailable."""
    ssim_metric = None
    lpips_metric = None

    try:
        try:
            from torchmetrics import StructuralSimilarityIndexMeasure
        except Exception:
            from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    except Exception as e:
        print(f"[metrics] SSIM disabled (pip install torchmetrics): {e}")

    try:
        try:
            from torchmetrics import LearnedPerceptualImagePatchSimilarity
        except Exception:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(device)
    except Exception as e:
        print(f"[metrics] LPIPS disabled (pip install torchmetrics): {e}")

    return ssim_metric, lpips_metric


@torch.no_grad()
def _evaluate_metrics(model, loader, device, image_size,
                      ssim_metric, lpips_metric, max_batches):
    """Clean (eval-mode) reconstruction metrics over `loader`."""
    model.eval()
    total_mse, n = 0.0, 0
    if ssim_metric is not None:
        ssim_metric.reset()
    if lpips_metric is not None:
        lpips_metric.reset()

    total = max_batches if max_batches is not None else None
    for i, batch in enumerate(tqdm(loader, total=total, desc="  metrics", leave=False)):
        if max_batches is not None and i >= max_batches:
            break
        imgs = batch[0].to(device)
        imgs = F.interpolate(imgs, size=(image_size, image_size),
                             mode="bilinear", align_corners=False)
        recon = model(imgs).clamp(0.0, 1.0)  # eval => no latent noise

        bs = imgs.size(0)
        total_mse += F.mse_loss(recon, imgs).item() * bs
        n += bs
        if ssim_metric is not None:
            ssim_metric.update(recon, imgs)
        if lpips_metric is not None:
            lpips_metric.update(recon, imgs)

    mse = total_mse / max(n, 1)
    psnr = 10.0 * math.log10(1.0 / mse) if mse > 0 else float("inf")
    ssim = ssim_metric.compute().item() if ssim_metric is not None else float("nan")
    lpips_v = lpips_metric.compute().item() if lpips_metric is not None else float("nan")
    return {"mse": mse, "psnr": psnr, "ssim": ssim, "lpips": lpips_v}


# =========================================================
# Plotting + logging
# =========================================================

def _to_img(t):
    return t.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()


def _plot_progress(history, sample_orig_np, sample_recon_np, show_plots):
    if not show_plots:
        return
    try:
        from IPython.display import clear_output
        clear_output(wait=True)
    except Exception:
        pass

    # --- reconstruction (fixed sample, tracked across epochs) ---
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(sample_orig_np)
    ax[0].set_title("Original")
    ax[0].axis("off")
    ax[1].imshow(sample_recon_np)
    ax[1].set_title(f"Reconstruction (epoch {history['epoch'][-1]})")
    ax[1].axis("off")
    plt.tight_layout()
    plt.show()

    # --- metric curves ---
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    epochs = history["epoch"]
    panels = [("mse", "MSE (lower better)"),
              ("psnr", "PSNR / dB (higher better)"),
              ("ssim", "SSIM (higher better)"),
              ("lpips", "LPIPS (lower better)")]
    for axp, (key, title) in zip(axes.ravel(), panels):
        axp.plot(epochs, history[key], marker="o", markersize=3)
        axp.set_title(title)
        axp.set_xlabel("epoch")
        axp.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


_LOG_FIELDS = ["epoch", "train_loss", "mse", "psnr", "ssim", "lpips",
               "encoder_lr", "decoder_lr"]


def _append_log(log_path, row):
    if log_path is None:
        return
    folder = os.path.dirname(log_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    is_new = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _optimizer_to(optimizer, device):
    """Move loaded optimizer state tensors onto `device` (Colab resume safety)."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


# =========================================================
# Main entry point
# =========================================================

def pretrain_visual_autoencoder(
    model,
    dataloader,
    device,
    *,
    save_checkpoint_fn=None,
    load_checkpoint_fn=None,
    early_stopping_cls=EarlyStopping,
    criterion=None,
    n_epochs=25,
    encoder_lr=1e-5,
    decoder_lr=1e-4,
    weight_decay=1e-4,
    grad_clip=1.0,
    image_size=224,
    checkpoint_filename="visual_autoencoder.pth",
    log_path="/content/drive/MyDrive/DL_Checkpoints/visual_autoencoder_log.csv",
    resume=True,
    early_stopping_patience=10,
    scheduler_factor=0.5,
    scheduler_patience=4,
    scheduler_min_lr=1e-7,
    metric_dataloader=None,
    max_metric_batches=None,
    latent_noise=False,
    show_plots=True,
):
    """
    Pretrain `model` to reconstruct images.

    Args:
        model:              VisualAutoencoder (encoder + decoder).
        dataloader:         training loader; batch[0] = images in [0,1].
        device:             torch device.
        save_checkpoint_fn: your save_checkpoint_to_drive (defaults to a copy).
        load_checkpoint_fn: your load_checkpoint_from_drive (defaults to a copy).
        early_stopping_cls: class with .step(loss)->bool and .best_loss.
        criterion:          loss module; defaults to ReconstructionLoss(0.8, 0.2).
        encoder_lr/decoder_lr/weight_decay/grad_clip: optimisation settings.
        scheduler_*:        ReduceLROnPlateau (mode='min' on train loss).
        metric_dataloader:  loader for eval metrics (defaults to `dataloader`).
        max_metric_batches: cap metric eval cost (None = full pass).
        latent_noise:       keep the decoder's training-time latent noise on
                            (hardening) instead of disabling it for clean
                            pretraining. Metrics are always clean.
        show_plots:         render live plots (False for headless).

    Returns:
        history (dict of lists): epoch, train_loss, mse, psnr, ssim, lpips,
                                 encoder_lr, decoder_lr.
    """
    save_checkpoint_fn = save_checkpoint_fn or _default_save_checkpoint
    load_checkpoint_fn = load_checkpoint_fn or _default_load_checkpoint
    metric_dataloader = metric_dataloader or dataloader

    # ----- trainable params -----
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.decoder.parameters(), "lr": decoder_lr},
        ],
        weight_decay=weight_decay,
    )

    if criterion is None:
        criterion = _build_default_criterion(device)

    early_stopper = early_stopping_cls(patience=early_stopping_patience, verbose=True)

    # ----- resume -----
    start_epoch = 0
    if resume:
        try:
            model, optimizer, start_epoch, initial_loss = load_checkpoint_fn(
                model, optimizer, filename=checkpoint_filename
            )
            _optimizer_to(optimizer, device)
            if initial_loss is not None:
                early_stopper.best_loss = initial_loss
            print(f"Resuming from epoch {start_epoch + 1}")
            print("Note: scheduler state is not persisted in the checkpoint; "
                  "the LR scheduler restarts on resume.")
        except FileNotFoundError:
            print("No checkpoint found, training from scratch.")

    # Scheduler (created after any resume so it reads the resumed LRs).
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor,
        patience=scheduler_patience, min_lr=scheduler_min_lr,
    )

    # ----- metrics + fixed visualisation sample -----
    ssim_metric, lpips_metric = _build_metrics(device)

    sample_batch = next(iter(metric_dataloader))
    sample_orig = F.interpolate(
        sample_batch[0][0:1].to(device),
        size=(image_size, image_size), mode="bilinear", align_corners=False,
    )
    sample_orig_np = _to_img(sample_orig[0])

    # ----- temporarily disable latent noise for clean pretraining -----
    saved_noise = {}
    if not latent_noise:
        for attr in ("z_noise_std", "spatial_noise_std"):
            if hasattr(model, attr):
                saved_noise[attr] = getattr(model, attr)
                setattr(model, attr, 0.0)

    history = {k: [] for k in _LOG_FIELDS}

    try:
        for epoch in range(start_epoch, n_epochs):
            model.train()
            epoch_loss = 0.0

            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
            for step, batch in enumerate(pbar, start=1):
                images = batch[0].to(device)
                images = F.interpolate(images, size=(image_size, image_size),
                                       mode="bilinear", align_corners=False)

                optimizer.zero_grad()
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss
                pbar.set_postfix(loss=f"{batch_loss:.4f}",
                                 avg=f"{epoch_loss / step:.4f}")

            avg_train = epoch_loss / len(dataloader)

            # ----- clean eval metrics -----
            metrics = _evaluate_metrics(
                model, metric_dataloader, device, image_size,
                ssim_metric, lpips_metric, max_metric_batches,
            )

            # ----- fixed-sample reconstruction (eval) -----
            model.eval()
            with torch.no_grad():
                sample_recon = model(sample_orig).clamp(0, 1)
            sample_recon_np = _to_img(sample_recon[0])

            # ----- scheduler (detect + report LR drops) -----
            lrs_before = [g["lr"] for g in optimizer.param_groups]
            scheduler.step(avg_train)
            lrs_after = [g["lr"] for g in optimizer.param_groups]
            if lrs_after != lrs_before:
                print(f"  LR reduced: {lrs_before} -> {lrs_after}")
            enc_lr, dec_lr = lrs_after[0], lrs_after[1]

            # ----- record -----
            history["epoch"].append(epoch + 1)
            history["train_loss"].append(avg_train)
            history["mse"].append(metrics["mse"])
            history["psnr"].append(metrics["psnr"])
            history["ssim"].append(metrics["ssim"])
            history["lpips"].append(metrics["lpips"])
            history["encoder_lr"].append(enc_lr)
            history["decoder_lr"].append(dec_lr)

            print(
                f"Epoch {epoch+1}/{n_epochs} | loss {avg_train:.4f} "
                f"| mse {metrics['mse']:.4f} | psnr {metrics['psnr']:.2f}dB "
                f"| ssim {metrics['ssim']:.4f} | lpips {metrics['lpips']:.4f} "
                f"| lr(enc/dec) {enc_lr:.2e}/{dec_lr:.2e}"
            )

            _append_log(log_path, {
                "epoch": epoch + 1,
                "train_loss": round(avg_train, 6),
                "mse": round(metrics["mse"], 6),
                "psnr": round(metrics["psnr"], 4),
                "ssim": round(metrics["ssim"], 6),
                "lpips": round(metrics["lpips"], 6),
                "encoder_lr": enc_lr,
                "decoder_lr": dec_lr,
            })

            # ----- checkpoint -----
            save_checkpoint_fn(model, optimizer, epoch + 1, avg_train,
                               filename=checkpoint_filename)

            # ----- live plots -----
            _plot_progress(history, sample_orig_np, sample_recon_np, show_plots)

            # ----- early stopping -----
            if early_stopper.step(avg_train):
                print("Early stopping triggered")
                break

    finally:
        # Restore the model's latent-noise configuration.
        for attr, val in saved_noise.items():
            setattr(model, attr, val)

    return history
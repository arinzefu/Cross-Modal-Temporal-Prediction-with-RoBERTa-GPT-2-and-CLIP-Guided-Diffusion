import os
import math

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
# @title Early stopping
# =========================================================
# Early stopping
# =========================================================
class EarlyStopping:
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
                print(f"Loss improved {self.best_loss:.4f} -> {current_loss:.4f}. Resetting patience.")
            self.best_loss = current_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.verbose:
                print(f"No improvement. Patience: {self.num_bad_epochs}/{self.patience}")
            if self.num_bad_epochs >= self.patience:
                self.stop = True
        return self.stop


DRIVE_CHECKPOINT_FOLDER = "/content/drive/MyDrive/DL_Checkpoints"
EVAL_LEVELS = (50, 200, 500)


# =========================================================
# History helper  (now stores per-level eval metrics)
# =========================================================
def make_diffusion_history():
    return {"epoch": [], "eps_mse": [], "eval": []}   # eval[i] = {L: {mse,ssim,psnr,lpips?}}


# =========================================================
# Save / Load to Drive
# =========================================================
def save_checkpoint_to_drive(model, optimizer, epoch, loss, history,
                             filename="visual_autoencoder.pth"):
    os.makedirs(DRIVE_CHECKPOINT_FOLDER, exist_ok=True)
    full_path = os.path.join(DRIVE_CHECKPOINT_FOLDER, filename)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "history": history,
        },
        full_path,
    )
    print(f"\u2713 Checkpoint saved to Drive: {full_path}  (epoch {epoch})")


def load_checkpoint_from_drive(model, optimizer=None, filename="visual_autoencoder.pth"):
    full_path = os.path.join(DRIVE_CHECKPOINT_FOLDER, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Checkpoint not found: {full_path}")

    checkpoint = torch.load(full_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))
    history = checkpoint.get("history", make_diffusion_history())
    if "eval" not in history:                        # migrate old-format history
        print("  (old history format detected \u2013 starting fresh metric history)")
        history = make_diffusion_history()

    print(f"\u2713 Checkpoint loaded from Drive: {full_path}  (epoch {epoch})")
    return model, optimizer, epoch, loss, history


# =========================================================
# Image metrics
# =========================================================
def _to_numpy01(img):
    """[C,H,W] in [-1,1] -> HxWxC numpy in [0,1]."""
    return ((img.detach().cpu().float() + 1) / 2).clamp(0, 1).permute(1, 2, 0).numpy()


@torch.no_grad()
def image_metrics(recon, target, lpips_fn=None):
    """recon, target: [B,3,H,W] in [-1,1]. Returns dict of averaged metrics."""
    r01 = (recon.clamp(-1, 1) + 1) / 2
    t01 = (target.clamp(-1, 1) + 1) / 2
    mse = F.mse_loss(r01, t01).item()

    ssim_vals, psnr_vals = [], []
    for i in range(recon.size(0)):
        ri, ti = _to_numpy01(recon[i]), _to_numpy01(target[i])
        try:
            s = ssim(ti, ri, channel_axis=2, data_range=1.0)
        except TypeError:                            # older skimage
            s = ssim(ti, ri, multichannel=True, data_range=1.0)
        ssim_vals.append(s)
        psnr_vals.append(psnr(ti, ri, data_range=1.0))

    out = {"mse": mse, "ssim": float(np.mean(ssim_vals)), "psnr": float(np.mean(psnr_vals))}
    if lpips_fn is not None:
        out["lpips"] = lpips_fn(recon.clamp(-1, 1), target.clamp(-1, 1)).mean().item()
    return out


# =========================================================
# Evaluation: partial-noise to level L, denoise back, score vs target
# =========================================================
@torch.no_grad()
def evaluate_diffusion(diffusion, x0, levels=EVAL_LEVELS, lpips_fn=None):
    device = x0.device
    metrics, recons = {}, {}
    for L in levels:
        t = torch.full((x0.size(0),), L, device=device, dtype=torch.long)
        x_t = diffusion.q_sample(x0, t)
        recon = diffusion.sample(x_start=x_t, t_start=L)
        recons[L] = recon
        metrics[L] = image_metrics(recon, x0, lpips_fn)
    return metrics, recons


# =========================================================
# Per-epoch visual: target | denoised 50/200/500 | noisy 200
# =========================================================
@torch.no_grad()
def _viz_epoch(diffusion, x0, recons, epoch, levels=EVAL_LEVELS, noisy_level=200):
    device = x0.device
    t = torch.full((1,), noisy_level, device=device, dtype=torch.long)
    noisy = diffusion.q_sample(x0[:1], t)

    panels = [x0[:1]] + [recons[L][:1] for L in levels] + [noisy]
    titles = ["target"] + [f"denoised {L}" for L in levels] + [f"noise {noisy_level}"]

    fig, ax = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.6))
    for a, im, ti in zip(ax, panels, titles):
        a.imshow(((im[0].cpu() + 1) / 2).permute(1, 2, 0).clamp(0, 1))
        a.set_title(ti, fontsize=10)
        a.axis("off")
    plt.suptitle(f"Epoch {epoch}", fontsize=11)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# =========================================================
# Metrics + loss graph (redrawn every epoch)
# =========================================================
def plot_history(history, levels=EVAL_LEVELS):
    ep = history["epoch"]
    if not ep:
        return

    metric_names = ["mse", "ssim", "psnr"]
    if history["eval"] and "lpips" in history["eval"][0][levels[0]]:
        metric_names.append("lpips")

    n = 1 + len(metric_names)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    axes[0].plot(ep, history["eps_mse"], marker="o", markersize=3, color="black")
    axes[0].set_title("Training eps MSE (loss)")
    axes[0].set_xlabel("epoch")

    for k, m in enumerate(metric_names, start=1):
        for L in levels:
            vals = [history["eval"][i][L][m] for i in range(len(ep))]
            axes[k].plot(ep, vals, marker="o", markersize=3, label=f"t={L}")
        axes[k].set_title(m.upper())
        axes[k].set_xlabel("epoch")
        axes[k].legend(fontsize=8)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Diffusion metrics & loss", fontsize=11)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# =========================================================
# Log export  (full history rewritten each epoch -> never loses earlier epochs)
# =========================================================
def export_training_log(history, filename="training_diffusion_log.txt", levels=EVAL_LEVELS):
    os.makedirs(DRIVE_CHECKPOINT_FOLDER, exist_ok=True)
    filepath = os.path.join(DRIVE_CHECKPOINT_FOLDER, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=== DIFFUSION TRAINING LOG ===\n\n")
        for i in range(len(history["epoch"])):
            f.write(f"Epoch {history['epoch'][i]}\n")
            f.write(f"  eps_mse : {history['eps_mse'][i]:.6f}\n")
            ev = history["eval"][i]
            for L in levels:
                m = ev[L]
                line = f"  t={L:<4}| MSE {m['mse']:.5f} | SSIM {m['ssim']:.4f} | PSNR {m['psnr']:.2f}"
                if "lpips" in m:
                    line += f" | LPIPS {m['lpips']:.4f}"
                f.write(line + "\n")
            f.write("\n")

        if history["epoch"]:
            f.write("=== SUMMARY ===\n")
            f.write(f"Total epochs  : {len(history['epoch'])}\n")
            f.write(f"Final eps_mse : {history['eps_mse'][-1]:.6f}\n")
            best = min(history["eps_mse"])
            be = history["epoch"][history["eps_mse"].index(best)]
            f.write(f"Best eps_mse  : {best:.6f} (epoch {be})\n")
            mid = levels[len(levels) // 2]
            ssim_mid = [history["eval"][i][mid]["ssim"] for i in range(len(history["epoch"]))]
            bs = max(ssim_mid)
            bse = history["epoch"][ssim_mid.index(bs)]
            f.write(f"Best SSIM@{mid} : {bs:.4f} (epoch {bse})\n")

    print(f"\u2713 Training log saved to {filepath}")


# =========================================================
# Training loop
# =========================================================
def train_diffusion(
    model,
    diffusion,
    dataloader,
    device,
    n_epochs=50,
    checkpoint_filename_v="visual_autoencoder.pth",
    log_name="training_diffusion_log.txt",
    clip_lr=1e-5,
    unet_lr=1e-4,
    weight_decay=1e-4,
    grad_clip=1.0,
    resume=True,
    early_stopper=None,
    eval_levels=EVAL_LEVELS,
    n_eval=4,
    noisy_level=200,
    eval_every=1,
):
    clip_params = [p for n, p in model.named_parameters()
                   if n.startswith("clip.") and p.requires_grad]
    unet_params = [p for n, p in model.named_parameters()
                   if not n.startswith("clip.") and p.requires_grad]

    param_groups = []
    if clip_params:
        param_groups.append({"params": clip_params, "lr": clip_lr})
    if unet_params:
        param_groups.append({"params": unet_params, "lr": unet_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    # ---- optional LPIPS (the 4th metric) ----
    lpips_fn = None
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False
        print("LPIPS enabled (net=alex).")
    except Exception:
        print("LPIPS unavailable \u2013 run `pip install lpips` to enable it. Continuing with MSE/SSIM/PSNR.")

    # ---- fixed eval batch (so metrics are comparable across epochs) ----
    eval_batch = next(iter(dataloader))
    eval_imgs = (eval_batch[0] if isinstance(eval_batch, (list, tuple)) else eval_batch).to(device)
    eval_imgs = F.interpolate(eval_imgs, (224, 224), mode="bilinear", align_corners=False)
    eval_x0 = (eval_imgs * 2 - 1)[:n_eval]           # [-1,1], small subset

    # ---- resume ----
    start_epoch = 0
    history = make_diffusion_history()
    if resume:
        try:
            model, optimizer, start_epoch, init_loss, history = \
                load_checkpoint_from_drive(model, optimizer, filename=checkpoint_filename_v)
            print(f"Resuming from epoch {start_epoch + 1}")
            if early_stopper is not None:
                early_stopper.best_loss = init_loss
        except FileNotFoundError:
            print("No checkpoint, training from scratch.")

    # ---- loop ----
    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")

        for batch in pbar:
            images = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
            images = F.interpolate(images, (224, 224), mode="bilinear", align_corners=False)
            x0 = images * 2 - 1                       # [0,1] -> [-1,1]

            t = torch.randint(0, diffusion.T, (x0.size(0),), device=device).long()

            optimizer.zero_grad()
            loss, _ = diffusion.p_losses(x0, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"eps_mse": f"{loss.item():.4f}"})

        avg = epoch_loss / max(1, len(dataloader))

        # ---- per-epoch evaluation (denoise from each level, score vs target) ----
        do_eval = (eval_every and (epoch + 1) % eval_every == 0)
        if do_eval:
            model.eval()
            metrics, recons = evaluate_diffusion(diffusion, eval_x0, eval_levels, lpips_fn)
            model.train()
        else:
            metrics, recons = history["eval"][-1] if history["eval"] else {}, None

        history["epoch"].append(epoch + 1)
        history["eps_mse"].append(avg)
        history["eval"].append(metrics)

        # ---- print ----
        print(f"\nEpoch {epoch+1}/{n_epochs} | eps_mse: {avg:.6f}")
        if metrics:
            for L in eval_levels:
                m = metrics[L]
                line = f"   t={L:<4}| MSE {m['mse']:.5f} | SSIM {m['ssim']:.4f} | PSNR {m['psnr']:.2f}"
                if "lpips" in m:
                    line += f" | LPIPS {m['lpips']:.4f}"
                print(line)

        # ---- save, log, graph, visual ----
        save_checkpoint_to_drive(model, optimizer, epoch + 1, avg, history,
                                 filename=checkpoint_filename_v)
        export_training_log(history, filename=log_name, levels=eval_levels)
        plot_history(history, levels=eval_levels)
        if recons is not None:
            _viz_epoch(diffusion, eval_x0, recons, epoch + 1, eval_levels, noisy_level)

        if early_stopper is not None and early_stopper.step(avg):
            print("Early stopping triggered.")
            break

    return model, optimizer, history
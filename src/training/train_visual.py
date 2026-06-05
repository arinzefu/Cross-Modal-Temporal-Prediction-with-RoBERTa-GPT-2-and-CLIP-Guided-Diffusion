import os
import math
import copy
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt


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
# EMA (exponential moving average of weights)
# Continuation-safe: on resume it initialises from the loaded weights,
# and its own state is persisted in the checkpoint so it carries across runs.
# =========================================================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.is_floating_point():
                s.mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                s.copy_(v)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        for k in self.shadow:
            if k in sd:
                self.shadow[k].copy_(sd[k].to(self.shadow[k].device))

    @contextmanager
    def average_parameters(self, model):
        """Temporarily swap EMA weights into the model (for eval / sampling)."""
        backup = copy.deepcopy(model.state_dict())
        msd = model.state_dict()
        to_load = {}
        for k in msd:
            if k in self.shadow:
                to_load[k] = self.shadow[k].to(dtype=msd[k].dtype, device=msd[k].device)
            else:
                to_load[k] = msd[k]
        model.load_state_dict(to_load)
        try:
            yield
        finally:
            model.load_state_dict(backup)


# =========================================================
# History helpers
# =========================================================
def make_diffusion_history():
    return {"epoch": [], "eps_mse": [], "eval": []}   # eval[i] = {L: {mse,ssim,psnr,lpips?}} or None


def _has_levels(ev, levels):
    """True only if ev is a full metrics dict containing every level."""
    return isinstance(ev, dict) and all(L in ev for L in levels)


# =========================================================
# Save / Load to Drive
# =========================================================
def save_checkpoint_to_drive(model, optimizer, epoch, loss, history,
                             filename="visual_autoencoder.pth", ema_state=None):
    os.makedirs(DRIVE_CHECKPOINT_FOLDER, exist_ok=True)
    full_path = os.path.join(DRIVE_CHECKPOINT_FOLDER, filename)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "history": history,
    }
    if ema_state is not None:
        ckpt["ema_state_dict"] = ema_state          # EMA travels with the checkpoint
    torch.save(ckpt, full_path)
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
    ema_state = checkpoint.get("ema_state_dict", None)   # may be absent on old checkpoints

    # ---- normalize history so consumers never hit a bad/short eval list ----
    history.setdefault("epoch", [])
    history.setdefault("eps_mse", [])
    history.setdefault("eval", [])
    n = len(history["epoch"])
    ev = (history["eval"] + [None] * n)[:n]                         # align length to epochs
    history["eval"] = [e if isinstance(e, dict) else None for e in ev]  # malformed -> None

    print(f"\u2713 Checkpoint loaded from Drive: {full_path}  (epoch {epoch})")
    return model, optimizer, epoch, loss, history, ema_state


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
# Per-epoch visual: grid of [noisy input | denoised | target] for each t
# =========================================================
@torch.no_grad()
def _viz_epoch(diffusion, x0, recons, epoch, levels=EVAL_LEVELS, idx=0):
    device = x0.device

    def to_img(t):                                    # [1,3,H,W] in [-1,1] -> HxWxC [0,1]
        return ((t[0].cpu() + 1) / 2).permute(1, 2, 0).clamp(0, 1)

    n = len(levels)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    axes = np.array(axes).reshape(n, 3)
    col_titles = ["noisy input", "denoised", "target"]

    xi = x0[idx:idx + 1]
    for r, L in enumerate(levels):
        t = torch.full((1,), L, device=device, dtype=torch.long)
        noisy = diffusion.q_sample(xi, t)
        imgs = [noisy, recons[L][idx:idx + 1], xi]
        for c, im in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(to_img(im))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11)
        axes[r, 0].set_ylabel(f"t = {L}", fontsize=12)

    plt.suptitle(f"Epoch {epoch} \u2014 denoising at t = {list(levels)}", fontsize=12)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# =========================================================
# Metrics + loss graph (redrawn every epoch; skips epochs without eval)
# =========================================================
def plot_history(history, levels=EVAL_LEVELS):
    ep = history["epoch"]
    if not ep:
        return

    metric_names = ["mse", "ssim", "psnr"]
    first_valid = next((e for e in history["eval"] if _has_levels(e, levels)), None)
    if first_valid is not None and "lpips" in first_valid[levels[0]]:
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
            xs, ys = [], []
            for i in range(len(ep)):
                ev = history["eval"][i] if i < len(history["eval"]) else None
                if _has_levels(ev, levels):
                    xs.append(ep[i]); ys.append(ev[L][m])
            axes[k].plot(xs, ys, marker="o", markersize=3, label=f"t={L}")
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
# Log export  (full history rewritten each epoch; tolerates missing eval)
# =========================================================
def export_training_log(history, filename="training_diffusion_log.txt", levels=EVAL_LEVELS):
    os.makedirs(DRIVE_CHECKPOINT_FOLDER, exist_ok=True)
    filepath = os.path.join(DRIVE_CHECKPOINT_FOLDER, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=== DIFFUSION TRAINING LOG ===\n\n")
        for i in range(len(history["epoch"])):
            f.write(f"Epoch {history['epoch'][i]}\n")
            f.write(f"  eps_mse : {history['eps_mse'][i]:.6f}\n")
            ev = history["eval"][i] if i < len(history["eval"]) else None
            if _has_levels(ev, levels):
                for L in levels:
                    m = ev[L]
                    line = f"  t={L:<4}| MSE {m['mse']:.5f} | SSIM {m['ssim']:.4f} | PSNR {m['psnr']:.2f}"
                    if "lpips" in m:
                        line += f" | LPIPS {m['lpips']:.4f}"
                    f.write(line + "\n")
            else:
                f.write("  (no eval this epoch)\n")
            f.write("\n")

        if history["epoch"]:
            f.write("=== SUMMARY ===\n")
            f.write(f"Total epochs  : {len(history['epoch'])}\n")
            f.write(f"Final eps_mse : {history['eps_mse'][-1]:.6f}\n")
            best = min(history["eps_mse"])
            be = history["epoch"][history["eps_mse"].index(best)]
            f.write(f"Best eps_mse  : {best:.6f} (epoch {be})\n")
            mid = levels[len(levels) // 2]
            ssim_pts = [(history["epoch"][i], history["eval"][i][mid]["ssim"])
                        for i in range(len(history["epoch"]))
                        if i < len(history["eval"]) and _has_levels(history["eval"][i], levels)]
            if ssim_pts:
                be2, bs = max(ssim_pts, key=lambda p: p[1])
                f.write(f"Best SSIM@{mid} : {bs:.4f} (epoch {be2})\n")

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
    eval_every=1,
    use_ema=True,
    ema_decay=0.999,
    override_lr_on_resume=True,
    eval_with_ema=True,
):
    clip_params = [p for n, p in model.named_parameters()
                   if n.startswith("clip.") and p.requires_grad]
    unet_params = [p for n, p in model.named_parameters()
                   if not n.startswith("clip.") and p.requires_grad]

    # build groups, remembering the target LR for each so we can re-apply post-resume
    group_specs = []
    if clip_params:
        group_specs.append((clip_params, clip_lr))
    if unet_params:
        group_specs.append((unet_params, unet_lr))
    optimizer = torch.optim.AdamW(
        [{"params": p, "lr": lr} for p, lr in group_specs], weight_decay=weight_decay)
    target_lrs = [lr for _, lr in group_specs]

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

    # ---- fixed eval batch (so metrics/visuals are comparable across epochs) ----
    eval_batch = next(iter(dataloader))
    eval_imgs = (eval_batch[0] if isinstance(eval_batch, (list, tuple)) else eval_batch).to(device)
    eval_imgs = F.interpolate(eval_imgs, (224, 224), mode="bilinear", align_corners=False)
    eval_x0 = (eval_imgs * 2 - 1)[:n_eval]           # [-1,1], small subset

    # ---- resume ----
    start_epoch = 0
    history = make_diffusion_history()
    ema_state = None
    if resume:
        try:
            model, optimizer, start_epoch, init_loss, history, ema_state = \
                load_checkpoint_from_drive(model, optimizer, filename=checkpoint_filename_v)
            print(f"Resuming from epoch {start_epoch + 1}")
            if early_stopper is not None:
                early_stopper.best_loss = init_loss
        except FileNotFoundError:
            print("No checkpoint, training from scratch.")

    # ---- LR override (must come AFTER load: optimizer.load_state_dict restores old LRs) ----
    if override_lr_on_resume:
        for g, lr in zip(optimizer.param_groups, target_lrs):
            g["lr"] = lr
        print(f"Learning rates set to: {[g['lr'] for g in optimizer.param_groups]}")

    # ---- EMA: init from current (loaded) weights, then restore saved shadow if present ----
    ema = None
    if use_ema:
        ema = EMA(model, decay=ema_decay)
        if ema_state is not None:
            ema.load_state_dict(ema_state)
            print("EMA shadow restored from checkpoint.")
        else:
            print("EMA initialised from current weights.")

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
            if ema is not None:
                ema.update(model)

            epoch_loss += loss.item()
            pbar.set_postfix({"eps_mse": f"{loss.item():.4f}"})

        avg = epoch_loss / max(1, len(dataloader))

        # ---- per-epoch evaluation (denoise from each level, score vs target) ----
        do_eval = bool(eval_every) and ((epoch + 1) % eval_every == 0)
        if do_eval:
            model.eval()
            if ema is not None and eval_with_ema:
                with ema.average_parameters(model):       # eval/sample under EMA weights
                    metrics, recons = evaluate_diffusion(diffusion, eval_x0, eval_levels, lpips_fn)
            else:
                metrics, recons = evaluate_diffusion(diffusion, eval_x0, eval_levels, lpips_fn)
            model.train()
        else:
            metrics, recons = None, None              # mark this epoch as "no eval"

        history["epoch"].append(epoch + 1)
        history["eps_mse"].append(avg)
        history["eval"].append(metrics)               # may be None; consumers skip it

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
                                 filename=checkpoint_filename_v,
                                 ema_state=(ema.state_dict() if ema is not None else None))
        export_training_log(history, filename=log_name, levels=eval_levels)
        plot_history(history, levels=eval_levels)
        if recons is not None:
            _viz_epoch(diffusion, eval_x0, recons, epoch + 1, eval_levels)

        if early_stopper is not None and early_stopper.step(avg):
            print("Early stopping triggered.")
            break

    return model, optimizer, history, ema

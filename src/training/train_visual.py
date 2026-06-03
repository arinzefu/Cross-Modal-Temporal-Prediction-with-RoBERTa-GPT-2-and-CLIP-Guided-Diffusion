import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt

DRIVE_CHECKPOINT_FOLDER = "/content/drive/MyDrive/DL_Checkpoints"


# =========================================================
# History helper
# =========================================================

def make_diffusion_history():
    return {"epoch": [], "eps_mse": []}


# =========================================================
# Save / Load to Drive
# =========================================================

def save_checkpoint_to_drive(model, optimizer, epoch, loss, history,
                             filename="clip_diffusion.pth"):
    os.makedirs(DRIVE_CHECKPOINT_FOLDER, exist_ok=True)
    full_path = os.path.join(DRIVE_CHECKPOINT_FOLDER, filename)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "history": history,        # history travels with the checkpoint
        },
        full_path,
    )
    print(f"\u2713 Checkpoint saved to Drive: {full_path}  (epoch {epoch})")


def load_checkpoint_from_drive(model, optimizer=None, filename="clip_diffusion.pth"):
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

    print(f"\u2713 Checkpoint loaded from Drive: {full_path}  (epoch {epoch})")
    return model, optimizer, epoch, loss, history


# =========================================================
# Log export
# =========================================================

def export_training_log(history, filename="training_diffusion_log.txt"):
    os.makedirs(DRIVE_CHECKPOINT_FOLDER, exist_ok=True)
    filepath = os.path.join(DRIVE_CHECKPOINT_FOLDER, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=== DIFFUSION TRAINING LOG ===\n\n")
        n = len(history["epoch"])
        for i in range(n):
            f.write(f"Epoch {history['epoch'][i]}\n")
            f.write(f"  Eps MSE     : {history['eps_mse'][i]:.6f}\n\n")

        if n > 0:
            best = min(history["eps_mse"])
            best_epoch = history["epoch"][history["eps_mse"].index(best)]
            f.write("\n=== SUMMARY ===\n")
            f.write(f"Total Epochs     : {n}\n")
            f.write(f"Final Eps MSE    : {history['eps_mse'][-1]:.6f}\n")
            f.write(f"Best Eps MSE     : {best:.6f}\n")
            f.write(f"Best Epoch       : {best_epoch}\n")

    print(f"\u2713 Training log saved to {filepath}")


# =========================================================
# Visualization
# =========================================================

def _viz(diffusion, x0, x0_hat, t):
    with torch.no_grad():
        x_t = diffusion.q_sample(x0[:1], t[:1])
        imgs = [x_t, x0_hat[:1], x0[:1]]
        titles = ["x_t (noisy)", "predicted x0", "target"]
        fig, ax = plt.subplots(1, 3, figsize=(11, 4))
        for a, im, ti in zip(ax, imgs, titles):
            a.imshow(((im[0].cpu() + 1) / 2).permute(1, 2, 0).clamp(0, 1))
            a.set_title(ti); a.axis("off")
        plt.show()
        plt.close(fig)


# =========================================================
# Training loop
# =========================================================

def train_diffusion(
    model,
    diffusion,
    dataloader,
    device,
    n_epochs=50,
    ckpt_name="clip_diffusion.pth",
    log_name="training_diffusion_log.txt",
    clip_lr=1e-5,
    unet_lr=1e-4,
    weight_decay=1e-4,
    grad_clip=1.0,
    resume=True,
    early_stopper=None,
    viz_every=300,
):
    """Pretrain the CLIP diffusion model, checkpointing + logging to Drive each epoch.

    Returns (model, optimizer, history).
    """
    # ---- optimizer: CLIP (unfrozen only) slower than the U-Net ----
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

    # ---- resume ----
    start_epoch = 0
    history = make_diffusion_history()
    if resume:
        try:
            model, optimizer, start_epoch, init_loss, history = \
                load_checkpoint_from_drive(model, optimizer, filename=ckpt_name)
            print(f"Resuming from epoch {start_epoch + 1}")
            if early_stopper is not None:
                early_stopper.best_loss = init_loss
        except FileNotFoundError:
            print("No checkpoint, training from scratch.")

    # ---- loop ----
    global_step = 0
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
            loss, x0_hat = diffusion.p_losses(x0, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({"eps_mse": f"{loss.item():.4f}"})

            if viz_every and global_step % viz_every == 0:
                model.eval()
                _viz(diffusion, x0, x0_hat, t)
                model.train()
            global_step += 1

        avg = epoch_loss / max(1, len(dataloader))
        history["epoch"].append(epoch + 1)
        history["eps_mse"].append(avg)
        print(f"\nEpoch {epoch+1}/{n_epochs} | avg eps_mse: {avg:.6f}")

        save_checkpoint_to_drive(model, optimizer, epoch + 1, avg, history, filename=ckpt_name)
        export_training_log(history, filename=log_name)

        if early_stopper is not None and early_stopper.step(avg):
            print("Early stopping triggered.")
            break

    return model, optimizer, history

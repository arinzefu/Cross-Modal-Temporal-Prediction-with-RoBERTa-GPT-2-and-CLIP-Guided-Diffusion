import os

import numpy as np
import torch
import matplotlib.pyplot as plt


DRIVE_CHECKPOINT_FOLDER = "/content/drive/MyDrive/DL_Checkpoints"


def make_predictor_history():
    keys = ["epoch", "train_loss", "text_loss", "image_loss",
            "bleu", "meteor", "rougeL", "mse", "ssim", "lpips", "psnr"]
    return {k: [] for k in keys}


# =========================================================
# Checkpoint (carries full history so resume continues the graphs)
# =========================================================
def save_predictor_checkpoint(model, optimizer, epoch, loss, history,
                              filename="sequence_predictor.pth"):
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
    print(f"\u2713 Predictor checkpoint saved: {path} (epoch {epoch})")


def load_predictor_checkpoint(model, optimizer=None, filename="sequence_predictor.pth"):
    path = os.path.join(DRIVE_CHECKPOINT_FOLDER, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = ckpt.get("epoch", 0)
    loss = ckpt.get("loss", float("inf"))
    history = ckpt.get("history", make_predictor_history())
    # normalize so consumers never index past the end on a resume
    n = len(history.get("epoch", []))
    for k in make_predictor_history():
        history.setdefault(k, [])
        if k != "epoch":
            history[k] = (history[k] + [None] * n)[:n]
    print(f"\u2713 Predictor checkpoint loaded: {path} (epoch {epoch})")
    return model, optimizer, epoch, loss, history


# =========================================================
# Log file  (full history rewritten each epoch -> never loses earlier epochs)
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
            fh.write(f"  train_loss : {f(history['train_loss'][i])}"
                     f"   text_loss : {f(history['text_loss'][i])}"
                     f"   image_loss : {f(history['image_loss'][i])}\n")
            fh.write(f"  TEXT   | BLEU {f(history['bleu'][i])}"
                     f"  METEOR {f(history['meteor'][i])}"
                     f"  ROUGE-L {f(history['rougeL'][i])}\n")
            fh.write(f"  VISUAL | MSE {f(history['mse'][i])}"
                     f"  SSIM {f(history['ssim'][i])}"
                     f"  LPIPS {f(history['lpips'][i])}"
                     f"  PSNR {f(history['psnr'][i])}\n\n")

        if history["epoch"]:
            fh.write("=== SUMMARY ===\n")
            fh.write(f"Epochs run : {len(history['epoch'])}\n")
            tl = [x for x in history["train_loss"] if x is not None]
            if tl:
                best = min(tl)
                be = history["epoch"][history["train_loss"].index(best)]
                fh.write(f"Best train_loss : {best:.4f} (epoch {be})\n")

    print(f"\u2713 Predictor log saved: {path}")


# =========================================================
# Plot helpers
# =========================================================
def _xy(history, key):
    xs, ys = [], []
    for e, v in zip(history["epoch"], history.get(key, [])):
        if v is not None:
            xs.append(e); ys.append(v)
    return xs, ys


def plot_loss(history):
    """Loss curve(s) over epochs — continues across resume since history is full."""
    if not history["epoch"]:
        return
    plt.figure(figsize=(9, 4))
    for key, lab in [("train_loss", "total"), ("text_loss", "text"), ("image_loss", "image")]:
        xs, ys = _xy(history, key)
        if xs:
            plt.plot(xs, ys, marker="o", markersize=3, label=lab)
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Sequence predictor loss")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show(); plt.close()


def plot_text_metrics(history):
    """Single graph: BLEU / METEOR / ROUGE-L vs epoch."""
    if not history["epoch"]:
        return
    plt.figure(figsize=(9, 4))
    for key, lab in [("bleu", "BLEU"), ("meteor", "METEOR"), ("rougeL", "ROUGE-L")]:
        xs, ys = _xy(history, key)
        if xs:
            plt.plot(xs, ys, marker="o", markersize=3, label=lab)
    plt.xlabel("epoch"); plt.ylabel("score"); plt.title("Text metrics vs epoch")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show(); plt.close()


def plot_visual_metrics(history):
    """Single graph: SSIM / LPIPS (left axis, 0-1) and MSE (right axis) vs epoch."""
    if not history["epoch"]:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    for key, lab in [("ssim", "SSIM"), ("lpips", "LPIPS")]:
        xs, ys = _xy(history, key)
        if xs:
            ax.plot(xs, ys, marker="o", markersize=3, label=lab)
    ax.set_xlabel("epoch"); ax.set_ylabel("SSIM / LPIPS (0-1)")

    ax2 = ax.twinx()
    xs, ys = _xy(history, "mse")
    if xs:
        ax2.plot(xs, ys, marker="s", markersize=3, color="firebrick", label="MSE")
    ax2.set_ylabel("MSE")

    lines = ax.get_lines() + ax2.get_lines()
    if lines:
        ax.legend(lines, [l.get_label() for l in lines], loc="best", fontsize=8)
    ax.set_title("Visual metrics vs epoch"); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.show(); plt.close(fig)


def plot_all_metrics_bar(metrics):
    """Final bar chart of every metric. PSNR sits on its own right axis (dB scale)."""
    unit = [("BLEU", metrics.get("bleu")), ("ROUGE-L", metrics.get("rougeL")),
            ("METEOR", metrics.get("meteor")), ("SSIM", metrics.get("ssim")),
            ("LPIPS", metrics.get("lpips")), ("MSE", metrics.get("mse"))]
    unit = [(k, float(v)) for k, v in unit if v is not None]

    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [k for k, _ in unit]
    vals = [v for _, v in unit]
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color="steelblue")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("score (0-1 range)")

    tick_pos = list(x)
    tick_lab = list(labels)

    psnr = metrics.get("psnr")
    if psnr is not None:
        ax2 = ax.twinx()
        pos = len(labels)
        ax2.bar([pos], [float(psnr)], color="orange", width=0.6)
        ax2.text(pos, float(psnr), f"{float(psnr):.2f}", ha="center", va="bottom", fontsize=8)
        ax2.set_ylabel("PSNR (dB)")
        tick_pos.append(pos)
        tick_lab.append("PSNR")

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)
    ax.set_title("Final evaluation metrics")
    plt.tight_layout(); plt.show(); plt.close(fig)

import os, csv, math, textwrap
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def save_predictor_checkpoint(path, predictor, optimizer, epoch, history, best_val_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Saving checkpoint for epoch {epoch}: {path}")

    torch.save({
        "epoch": epoch,
        "model_state_dict": predictor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "best_val_loss": best_val_loss,
    }, path)

    print(f"Checkpoint saved: {path}")


def load_predictor_checkpoint(path, predictor, optimizer=None, device="cpu"):
    print(f"Looking for checkpoint: {path}")

    if not os.path.exists(path):
        print(f"No checkpoint found at {path}. Starting from scratch.")
        return 1, float("inf"), []

    print(f"Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    predictor.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print("Optimizer state loaded.")
    elif optimizer is not None:
        print("Checkpoint has no optimizer state. Optimizer will start fresh.")

    start_epoch = ckpt.get("epoch", 0) + 1
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    history = ckpt.get("history", [])

    print(
        f"Checkpoint loaded. Resuming from epoch {start_epoch}; "
        f"best_val_loss={best_val_loss:.4f}"
    )
    return start_epoch, best_val_loss, history


def write_metrics_log(path, history):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    keys = [
        "epoch", "train_loss", "train_image_loss", "train_text_loss",
        "val_loss", "val_text_loss", "mse", "psnr", "ssim", "lpips",
        "bleu", "meteor", "rougeL", "attn_loss", "copy_loss",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k, None) for k in keys})

    print(f"Metrics log saved: {path}")


def ssim_torch(pred, target):
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.avg_pool2d(pred, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)

    sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y

    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    )

    return ssim.mean()


def psnr_from_mse(mse):
    return 10.0 * math.log10(1.0 / max(mse, 1e-8))


def get_lpips_model(device):
    try:
        import lpips
        model = lpips.LPIPS(net="alex").to(device)
        model.eval()
        return model
    except Exception:
        print("LPIPS not available. Install with: pip install lpips")
        return None


def rouge_l_score(pred, ref):
    p = pred.lower().split()
    r = ref.lower().split()

    if not p or not r:
        return 0.0

    dp = [[0] * (len(r) + 1) for _ in range(len(p) + 1)]

    for i in range(len(p)):
        for j in range(len(r)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if p[i] == r[j] else max(
                dp[i][j + 1], dp[i + 1][j]
            )

    lcs = dp[-1][-1]
    prec = lcs / len(p)
    rec = lcs / len(r)

    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def text_metrics(pred_texts, target_texts):
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        smooth = SmoothingFunction().method1
        bleu = [
            sentence_bleu([t.split()], p.split(), smoothing_function=smooth)
            for p, t in zip(pred_texts, target_texts)
        ]
    except Exception:
        bleu = [0.0 for _ in pred_texts]

    meteor = []
    for p, t in zip(pred_texts, target_texts):
        p_set = set(p.lower().split())
        t_set = set(t.lower().split())
        overlap = len(p_set & t_set)

        if overlap == 0:
            meteor.append(0.0)
        else:
            prec = overlap / max(len(p_set), 1)
            rec = overlap / max(len(t_set), 1)
            meteor.append((10 * prec * rec) / max(rec + 9 * prec, 1e-8))

    rouge = [rouge_l_score(p, t) for p, t in zip(pred_texts, target_texts)]

    return {
        "bleu": sum(bleu) / max(len(bleu), 1),
        "meteor": sum(meteor) / max(len(meteor), 1),
        "rougeL": sum(rouge) / max(len(rouge), 1),
    }


def next_token_ce_loss(logits, target_ids, target_attention_mask):
    labels = target_ids[:, 1:].clone()
    labels[target_attention_mask[:, 1:] == 0] = -100

    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
    )


def attention_anti_collapse_loss(
    out,
    last_frame_cap=0.55,
    min_entropy_frac=0.65,
):
    losses = []
    device = out["pred_image_z"].device

    for key in [
        "visual_attn_weights",
        "text_attn_weights",
        "fusion_attn_weights",
    ]:
        weights = out.get(key, None)

        if weights is None:
            continue

        weights = weights.clamp_min(1e-8)

        last_weight = weights[:, -1]
        last_penalty = F.relu(last_weight - last_frame_cap).pow(2).mean()

        entropy = -(weights * weights.log()).sum(dim=-1)
        max_entropy = math.log(weights.size(1))
        entropy_floor = min_entropy_frac * max_entropy

        entropy_penalty = F.relu(entropy_floor - entropy).pow(2).mean()

        losses.append(last_penalty + 0.25 * entropy_penalty)

    if not losses:
        return torch.tensor(0.0, device=device)

    return torch.stack(losses).mean()


def _anti_copy_margin_loss(
    pred,
    target,
    last_input,
    margin_ratio=0.20,
    min_target_change=1e-5,
):
    pred = pred.flatten(start_dim=1)
    target = target.flatten(start_dim=1)
    last_input = last_input.flatten(start_dim=1)

    d_pred_target = F.mse_loss(pred, target, reduction="none").mean(dim=1)
    d_pred_last = F.mse_loss(pred, last_input, reduction="none").mean(dim=1)
    d_target_last = F.mse_loss(target, last_input, reduction="none").mean(dim=1)

    active = (d_target_last.detach() > min_target_change).float()
    margin = margin_ratio * d_target_last.detach()

    loss = F.relu(d_pred_target - d_pred_last + margin)

    return (loss * active).sum() / active.sum().clamp_min(1.0)


def anti_frame4_copy_loss(out, image_seq, image_target):
    last_image = image_seq[:, -1]

    image_copy_loss = _anti_copy_margin_loss(
        pred=out["pred_image"],
        target=image_target,
        last_input=last_image,
        margin_ratio=0.20,
        min_target_change=1e-4,
    )

    z_copy_loss = _anti_copy_margin_loss(
        pred=out["pred_image_z"],
        target=out["target_image_z"],
        last_input=out["input_visual_z"][:, -1],
        margin_ratio=0.25,
        min_target_change=1e-5,
    )

    spatial_copy_loss = _anti_copy_margin_loss(
        pred=out["pred_image_spatial"],
        target=out["target_image_spatial"],
        last_input=out["input_visual_spatial"][:, -1],
        margin_ratio=0.15,
        min_target_change=1e-5,
    )

    return image_copy_loss + z_copy_loss + 0.25 * spatial_copy_loss


def apply_temporal_context_dropout(
    image_seq,
    enc_ids,
    enc_mask,
    p_other=0.05,
    p_last=0.20,
    image_fill=0.5,
    pad_token_id=None,
):
    if p_other <= 0 and p_last <= 0:
        return image_seq, enc_ids, enc_mask

    b, t = image_seq.shape[:2]
    device = image_seq.device

    drop = torch.rand(b, t, device=device) < p_other
    drop[:, -1] = torch.rand(b, device=device) < p_last

    all_dropped = drop.all(dim=1)
    if all_dropped.any():
        drop[all_dropped, 0] = False

    if not drop.any():
        return image_seq, enc_ids, enc_mask

    image_seq = image_seq.clone()
    enc_ids = enc_ids.clone()
    enc_mask = enc_mask.clone()

    image_seq[drop] = image_fill
    enc_mask[drop] = 0

    if pad_token_id is not None:
        enc_ids[drop] = pad_token_id

    return image_seq, enc_ids, enc_mask


def print_latent_diagnostic(out):
    z_cos = F.cosine_similarity(
        out["pred_image_z"],
        out["target_image_z"],
        dim=-1,
    ).mean()

    spatial_mse = F.mse_loss(
        out["pred_image_spatial"],
        out["target_image_spatial"],
    )

    print("\nLatent diagnostic")
    print(
        "z pred mean/std:",
        f"{out['pred_image_z'].mean().item():.4f}",
        f"{out['pred_image_z'].std().item():.4f}",
    )
    print(
        "z target mean/std:",
        f"{out['target_image_z'].mean().item():.4f}",
        f"{out['target_image_z'].std().item():.4f}",
    )
    print(
        "spatial pred mean/std:",
        f"{out['pred_image_spatial'].mean().item():.4f}",
        f"{out['pred_image_spatial'].std().item():.4f}",
    )
    print(
        "spatial target mean/std:",
        f"{out['target_image_spatial'].mean().item():.4f}",
        f"{out['target_image_spatial'].std().item():.4f}",
    )
    print("z cosine:", f"{z_cos.item():.4f}")
    print("spatial mse:", f"{spatial_mse.item():.4f}")


@torch.no_grad()
def evaluate_predictor(
    predictor,
    val_loader,
    reconstruction_loss,
    dec_tokenizer,
    device,
    lpips_model=None,
    max_new_tokens=80,
    text_weight=1.0,
    image_weight=1.0
):
    predictor.eval()

    totals = {
        "val_loss": 0.0,
        "val_text_loss": 0.0,
        "mse": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "lpips": 0.0,
    }

    pred_texts_all = []
    target_texts_all = []

    n = 0

    for batch in val_loader:
        image_seq = batch[0].to(device)
        image_target = batch[1].to(device)
        text_dict = next(x for x in batch if isinstance(x, dict))

        enc_ids = text_dict["enc_input_ids"].to(device)
        enc_mask = text_dict["enc_attention_mask"].to(device)
        target_ids = text_dict["target_ids"].to(device)
        target_mask = text_dict["target_attention_mask"].to(device)

        decoder_input_ids = target_ids[:, :-1]
        decoder_attention_mask = target_mask[:, :-1]

        out = predictor(
            image_seq=image_seq,
            input_ids_text_encoder=enc_ids,
            attention_mask_text_encoder=enc_mask,
            target_seq_text_decoder=decoder_input_ids,
            target_attention_mask_text_decoder=decoder_attention_mask,
            image_target=image_target,
            decode_image=True,
        )

        if n == 0:
            print_latent_diagnostic(out)

        image_loss = reconstruction_loss(
            out["pred_image"],
            image_target,
            z_pred=out["pred_image_z"],
            z_target=out["target_image_z"],
            spatial_pred=out["pred_image_spatial"],
            spatial_target=out["target_image_spatial"],
        )

        text_loss = next_token_ce_loss(
            logits=out["pred_text_logits"],
            target_ids=target_ids,
            target_attention_mask=target_mask,
        )

        val_loss = image_weight * image_loss + text_weight * text_loss
        mse = F.mse_loss(out["pred_image"], image_target).item()

        totals["val_loss"] += val_loss.item()
        totals["val_text_loss"] += text_loss.item()
        totals["mse"] += mse
        totals["psnr"] += psnr_from_mse(mse)
        totals["ssim"] += ssim_torch(out["pred_image"], image_target).item()

        if lpips_model is not None:
            lp = lpips_model(
                out["pred_image"] * 2 - 1,
                image_target * 2 - 1,
            ).mean().item()
            totals["lpips"] += lp

        gen_ids = predictor.generate_text_ids(
            image_seq=image_seq,
            input_ids_text_encoder=enc_ids,
            attention_mask_text_encoder=enc_mask,
            max_new_tokens=max_new_tokens,
        )

        pred_texts_all += dec_tokenizer.batch_decode(
            gen_ids,
            skip_special_tokens=True,
        )

        target_texts_all += dec_tokenizer.batch_decode(
            target_ids,
            skip_special_tokens=True,
        )

        n += 1

    for k in totals:
        totals[k] /= max(n, 1)

    if lpips_model is None:
        totals["lpips"] = None

    totals.update(text_metrics(pred_texts_all, target_texts_all))
    return totals


@torch.no_grad()
def visualize_validation_epoch(
    predictor,
    val_loader,
    enc_tokenizer,
    dec_tokenizer,
    device,
    epoch,
    save_dir,
    max_samples=2,
    max_new_tokens=80,
):
    os.makedirs(save_dir, exist_ok=True)
    predictor.eval()

    batch = next(iter(val_loader))

    image_seq = batch[0].to(device)
    image_target = batch[1].to(device)
    text_dict = next(x for x in batch if isinstance(x, dict))

    enc_ids = text_dict["enc_input_ids"].to(device)
    enc_mask = text_dict["enc_attention_mask"].to(device)
    target_ids = text_dict["target_ids"].to(device)
    target_mask = text_dict["target_attention_mask"].to(device)

    out = predictor(
        image_seq=image_seq,
        input_ids_text_encoder=enc_ids,
        attention_mask_text_encoder=enc_mask,
        target_seq_text_decoder=target_ids[:, :-1],
        target_attention_mask_text_decoder=target_mask[:, :-1],
        image_target=image_target,
        decode_image=True,
    )

    gen_ids = predictor.generate_text_ids(
        image_seq,
        enc_ids,
        enc_mask,
        max_new_tokens=max_new_tokens,
    )

    pred_texts = dec_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    target_texts = dec_tokenizer.batch_decode(target_ids, skip_special_tokens=True)

    take = min(max_samples, image_seq.size(0))

    for b in range(take):
        fig, axes = plt.subplots(1, 6, figsize=(18, 4))

        for t in range(4):
            axes[t].imshow(image_seq[b, t].detach().cpu().permute(1, 2, 0).clamp(0, 1))
            axes[t].set_title(f"Input frame {t + 1}")
            axes[t].axis("off")

        axes[4].imshow(image_target[b].detach().cpu().permute(1, 2, 0).clamp(0, 1))
        axes[4].set_title("Target frame 5")
        axes[4].axis("off")

        axes[5].imshow(out["pred_image"][b].detach().cpu().permute(1, 2, 0).clamp(0, 1))
        axes[5].set_title("Predicted frame 5")
        axes[5].axis("off")

        input_texts = [
            enc_tokenizer.decode(enc_ids[b, t], skip_special_tokens=True)
            for t in range(4)
        ]

        text_block = "\n".join(
            [
                f"Input {i + 1}: {textwrap.shorten(txt, width=120)}"
                for i, txt in enumerate(input_texts)
            ]
        )

        text_block += "\n\n"
        text_block += "Target 5: " + textwrap.shorten(target_texts[b], width=160)
        text_block += "\n"
        text_block += "Predicted 5: " + textwrap.shorten(pred_texts[b], width=160)

        fig.suptitle(f"Validation sample {b + 1} after epoch {epoch}", fontsize=14)
        fig.text(0.01, -0.05, text_block, fontsize=10, va="top")
        plt.tight_layout()

        path = os.path.join(save_dir, f"epoch_{epoch:03d}_sample_{b + 1}.png")
        plt.savefig(path, bbox_inches="tight", dpi=140)
        print(f"Validation visualization saved: {path}")
        plt.show()
        plt.close(fig)


def train_sequence_predictor(
    predictor,
    train_loader,
    val_loader,
    optimizer,
    reconstruction_loss,
    enc_tokenizer,
    dec_tokenizer,
    device,
    epochs=20,
    checkpoint_dir="/content/drive/MyDrive/DL_Checkpoints",
    resume=False,
    text_weight=1.0,
    image_weight=1.0,
    attn_balance_weight=0.05,
    copy_margin_weight=0.10,
    last_frame_cap=0.55,
    use_context_dropout=True,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    ckpt_path = os.path.join(checkpoint_dir, "sequence_predictor_latest.pth")
    best_path = os.path.join(checkpoint_dir, "sequence_predictor_best.pth")
    log_path = os.path.join(checkpoint_dir, "sequence_predictor_metrics.csv")
    viz_dir = os.path.join(checkpoint_dir, "validation_visuals")

    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Latest checkpoint path: {ckpt_path}")
    print(f"Best checkpoint path: {best_path}")
    print(f"Metrics log path: {log_path}")

    lpips_model = get_lpips_model(device)

    if resume:
        start_epoch, best_val_loss, history = load_predictor_checkpoint(
            ckpt_path, predictor, optimizer, device
        )
    else:
        print("resume=False. Starting training from scratch.")
        start_epoch, best_val_loss, history = 1, float("inf"), []

    if start_epoch > epochs:
        print(
            f"Checkpoint is already past requested epochs "
            f"(start_epoch={start_epoch}, epochs={epochs}). Nothing to train."
        )
        return history

    for epoch in range(start_epoch, epochs + 1):
        predictor.train()
        print(f"\nStarting epoch {epoch}/{epochs}")

        train_loss = train_image_loss = train_text_loss = 0.0
        train_attn_loss = train_copy_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for batch in pbar:
            image_seq = batch[0].to(device)
            image_target = batch[1].to(device)
            text_dict = next(x for x in batch if isinstance(x, dict))

            enc_ids = text_dict["enc_input_ids"].to(device)
            enc_mask = text_dict["enc_attention_mask"].to(device)
            target_ids = text_dict["target_ids"].to(device)
            target_mask = text_dict["target_attention_mask"].to(device)

            train_image_seq, train_enc_ids, train_enc_mask = image_seq, enc_ids, enc_mask

            if use_context_dropout:
                train_image_seq, train_enc_ids, train_enc_mask = apply_temporal_context_dropout(
                    train_image_seq,
                    train_enc_ids,
                    train_enc_mask,
                    p_other=0.05,
                    p_last=0.20,
                    pad_token_id=getattr(enc_tokenizer, "pad_token_id", None),
                )

            optimizer.zero_grad(set_to_none=True)

            out = predictor(
                image_seq=train_image_seq,
                input_ids_text_encoder=train_enc_ids,
                attention_mask_text_encoder=train_enc_mask,
                target_seq_text_decoder=target_ids[:, :-1],
                target_attention_mask_text_decoder=target_mask[:, :-1],
                image_target=image_target,
                decode_image=True,
            )

            image_loss = reconstruction_loss(
                out["pred_image"],
                image_target,
                z_pred=out["pred_image_z"],
                z_target=out["target_image_z"],
                spatial_pred=out["pred_image_spatial"],
                spatial_target=out["target_image_spatial"],
            )

            text_loss = next_token_ce_loss(
                out["pred_text_logits"],
                target_ids,
                target_mask,
            )

            attn_loss = attention_anti_collapse_loss(
                out,
                last_frame_cap=last_frame_cap,
                min_entropy_frac=0.65,
            )

            copy_loss = anti_frame4_copy_loss(
                out,
                image_seq,
                image_target,
            )

            loss = (
                image_weight * image_loss
                + text_weight * text_loss
                + attn_balance_weight * attn_loss
                + copy_margin_weight * copy_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_image_loss += image_loss.item()
            train_text_loss += text_loss.item()
            train_attn_loss += attn_loss.item()
            train_copy_loss += copy_loss.item()

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                text=f"{text_loss.item():.4f}",
                img=f"{image_loss.item():.4f}",
            )

        n_train = max(len(train_loader), 1)

        print(f"Running validation after epoch {epoch}...")
        val_metrics = evaluate_predictor(
            predictor=predictor,
            val_loader=val_loader,
            reconstruction_loss=reconstruction_loss,
            dec_tokenizer=dec_tokenizer,
            device=device,
            lpips_model=lpips_model,
            text_weight=text_weight,
            image_weight=image_weight,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss / n_train,
            "train_image_loss": train_image_loss / n_train,
            "train_text_loss": train_text_loss / n_train,
            "attn_loss": train_attn_loss / n_train,
            "copy_loss": train_copy_loss / n_train,
            **val_metrics,
        }

        history.append(row)
        write_metrics_log(log_path, history)

        # Always save the latest checkpoint after every epoch.
        save_predictor_checkpoint(
            ckpt_path,
            predictor,
            optimizer,
            epoch,
            history,
            best_val_loss,
        )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            print(
                f"New best validation loss at epoch {epoch}: "
                f"{best_val_loss:.4f}"
            )
            save_predictor_checkpoint(
                best_path,
                predictor,
                optimizer,
                epoch,
                history,
                best_val_loss,
            )
        else:
            print(
                f"No best-checkpoint update. Current val_loss="
                f"{val_metrics['val_loss']:.4f}, best_val_loss={best_val_loss:.4f}"
            )

        visualize_validation_epoch(
            predictor=predictor,
            val_loader=val_loader,
            enc_tokenizer=enc_tokenizer,
            dec_tokenizer=dec_tokenizer,
            device=device,
            epoch=epoch,
            save_dir=viz_dir,
        )

        print(
            f"Epoch {epoch} | "
            f"train={row['train_loss']:.4f} | "
            f"val={row['val_loss']:.4f} | "
            f"text={row['val_text_loss']:.4f} | "
            f"mse={row['mse']:.4f} | "
            f"psnr={row['psnr']:.2f} | "
            f"ssim={row['ssim']:.4f} | "
            f"bleu={row['bleu']:.4f} | "
            f"meteor={row['meteor']:.4f} | "
            f"rougeL={row['rougeL']:.4f}"
        )

    print("Training complete.")
    print(f"Latest checkpoint: {ckpt_path}")
    print(f"Best checkpoint: {best_path}")
    print(f"Metrics log: {log_path}")
    print(f"Validation visualizations: {viz_dir}")

    return history

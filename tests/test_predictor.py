import math
import textwrap

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


GEN_SIZE = 224


# =========================================================
# Batch helpers
# =========================================================

def _unpack(batch):
    frames = batch[0]          # [B, 4, 3, H, W]
    image_target = batch[1]    # [B, 3, H, W]
    text_dict = next(x for x in batch if isinstance(x, dict))
    return frames, image_target, text_dict


def _get_text_tensors(text_dict, device):
    enc_ids = text_dict["enc_input_ids"].to(device)
    enc_mask = text_dict["enc_attention_mask"].to(device)
    tgt_ids = text_dict["target_ids"].to(device)
    tgt_mask = text_dict["target_attention_mask"].to(device)
    return enc_ids, enc_mask, tgt_ids, tgt_mask


def _decoder_inputs(target_ids, target_mask):
    return target_ids[:, :-1], target_mask[:, :-1]


# =========================================================
# Metrics
# =========================================================

def psnr_from_mse(mse):
    return 10.0 * math.log10(1.0 / max(mse, 1e-8))


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


def rouge_l_score(pred, ref):
    p = pred.lower().split()
    r = ref.lower().split()

    if not p or not r:
        return 0.0

    dp = [[0] * (len(r) + 1) for _ in range(len(p) + 1)]

    for i in range(len(p)):
        for j in range(len(r)):
            dp[i + 1][j + 1] = (
                dp[i][j] + 1
                if p[i] == r[j]
                else max(dp[i][j + 1], dp[i + 1][j])
            )

    lcs = dp[-1][-1]
    precision = lcs / len(p)
    recall = lcs / len(r)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


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

    for pred, target in zip(pred_texts, target_texts):
        p_set = set(pred.lower().split())
        t_set = set(target.lower().split())
        overlap = len(p_set & t_set)

        if overlap == 0:
            meteor.append(0.0)
        else:
            precision = overlap / max(len(p_set), 1)
            recall = overlap / max(len(t_set), 1)
            meteor.append((10 * precision * recall) / max(recall + 9 * precision, 1e-8))

    rouge = [rouge_l_score(p, t) for p, t in zip(pred_texts, target_texts)]

    return {
        "bleu": sum(bleu) / max(len(bleu), 1),
        "meteor": sum(meteor) / max(len(meteor), 1),
        "rougeL": sum(rouge) / max(len(rouge), 1),
    }


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
    predictor.eval()

    gen_ids = predictor.generate_text_ids(
        image_seq=image_seq.to(device),
        input_ids_text_encoder=enc_ids.to(device),
        attention_mask_text_encoder=enc_mask.to(device),
        max_new_tokens=max_len,
    )

    return dec_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)


# =========================================================
# Batch prediction
# =========================================================

@torch.no_grad()
def predict_batch(
    predictor,
    batch,
    device,
    max_new_tokens=80,
):
    predictor.eval()

    frames, image_target, text_dict = _unpack(batch)

    frames = frames.to(device)
    image_target = image_target.to(device)

    enc_ids, enc_mask, target_ids, target_mask = _get_text_tensors(text_dict, device)
    decoder_ids, decoder_mask = _decoder_inputs(target_ids, target_mask)

    out = predictor(
        image_seq=frames,
        input_ids_text_encoder=enc_ids,
        attention_mask_text_encoder=enc_mask,
        target_seq_text_decoder=decoder_ids,
        target_attention_mask_text_decoder=decoder_mask,
        image_target=image_target,
        decode_image=True,
    )

    gen_ids = predictor.generate_text_ids(
        image_seq=frames,
        input_ids_text_encoder=enc_ids,
        attention_mask_text_encoder=enc_mask,
        max_new_tokens=max_new_tokens,
    )

    out["generated_text_ids"] = gen_ids
    out["image_seq"] = frames
    out["image_target"] = image_target
    out["target_ids"] = target_ids
    out["target_attention_mask"] = target_mask
    out["enc_ids"] = enc_ids
    out["enc_mask"] = enc_mask

    return out


# =========================================================
# Shape / sanity test
# =========================================================

@torch.no_grad()
def test_predictor_shapes(
    predictor,
    dataloader,
    device,
):
    batch = next(iter(dataloader))
    out = predict_batch(predictor, batch, device)

    frames = out["image_seq"]
    target = out["image_target"]

    assert frames.dim() == 5, f"Expected image_seq [B,4,3,H,W], got {frames.shape}"
    assert frames.size(1) == 4, f"Expected 4 context frames, got {frames.size(1)}"

    assert out["pred_image"].shape == target.shape
    assert out["pred_image_z"].dim() == 2
    assert out["pred_image_spatial"].dim() == 4
    assert out["pred_text_logits"].dim() == 3

    assert out["visual_attn_weights"].shape[:2] == frames.shape[:2]
    assert out["text_attn_weights"].shape[:2] == frames.shape[:2]

    print("Shape test passed.")
    print("pred_image:", tuple(out["pred_image"].shape))
    print("pred_image_z:", tuple(out["pred_image_z"].shape))
    print("pred_image_spatial:", tuple(out["pred_image_spatial"].shape))
    print("pred_text_logits:", tuple(out["pred_text_logits"].shape))


# =========================================================
# Frame-4 copy diagnostic
# =========================================================

@torch.no_grad()
def frame4_copy_diagnostic(
    predictor,
    dataloader,
    device,
    num_batches=5,
):
    predictor.eval()

    pred_target_dist = []
    pred_last_dist = []
    target_last_dist = []
    last_attn = []

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        out = predict_batch(predictor, batch, device)

        pred = out["pred_image"].flatten(start_dim=1)
        target = out["image_target"].flatten(start_dim=1)
        last = out["image_seq"][:, -1].flatten(start_dim=1)

        pred_target_dist.append(F.mse_loss(pred, target).item())
        pred_last_dist.append(F.mse_loss(pred, last).item())
        target_last_dist.append(F.mse_loss(target, last).item())

        last_attn.append(out["visual_attn_weights"][:, -1].mean().item())

    result = {
        "pred_to_target_mse": sum(pred_target_dist) / max(len(pred_target_dist), 1),
        "pred_to_frame4_mse": sum(pred_last_dist) / max(len(pred_last_dist), 1),
        "target_to_frame4_mse": sum(target_last_dist) / max(len(target_last_dist), 1),
        "mean_visual_attention_on_frame4": sum(last_attn) / max(len(last_attn), 1),
    }

    print("Frame-4 copy diagnostic:")
    for k, v in result.items():
        print(f"  {k}: {v:.6f}")

    if result["mean_visual_attention_on_frame4"] > 0.70:
        print("Warning: visual attention is heavily concentrated on frame 4.")

    if result["pred_to_frame4_mse"] < result["pred_to_target_mse"]:
        print("Warning: prediction is closer to frame 4 than to target frame 5.")

    return result


# =========================================================
# Evaluation
# =========================================================

@torch.no_grad()
def evaluate_predictor_test(
    predictor,
    dataloader,
    dec_tokenizer,
    device,
    reconstruction_loss=None,
    lpips_model=None,
    max_batches=None,
):
    predictor.eval()

    totals = {
        "loss": 0.0,
        "text_loss": 0.0,
        "image_loss": 0.0,
        "mse": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "lpips": 0.0,
    }

    pred_texts = []
    target_texts = []
    n = 0

    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        out = predict_batch(predictor, batch, device)

        image_loss = F.mse_loss(out["pred_image"], out["image_target"])

        if reconstruction_loss is not None:
            image_loss = reconstruction_loss(
                out["pred_image"],
                out["image_target"],
                z_pred=out["pred_image_z"],
                z_target=out["target_image_z"],
                spatial_pred=out["pred_image_spatial"],
                spatial_target=out["target_image_spatial"],
            )

        labels = out["target_ids"][:, 1:].clone()
        labels[out["target_attention_mask"][:, 1:] == 0] = -100

        text_loss = F.cross_entropy(
            out["pred_text_logits"].reshape(-1, out["pred_text_logits"].size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

        loss = image_loss + text_loss

        mse = F.mse_loss(out["pred_image"], out["image_target"]).item()

        totals["loss"] += loss.item()
        totals["text_loss"] += text_loss.item()
        totals["image_loss"] += image_loss.item()
        totals["mse"] += mse
        totals["psnr"] += psnr_from_mse(mse)
        totals["ssim"] += ssim_torch(out["pred_image"], out["image_target"]).item()

        if lpips_model is not None:
            lp = lpips_model(
                out["pred_image"] * 2 - 1,
                out["image_target"] * 2 - 1,
            ).mean().item()
            totals["lpips"] += lp

        pred_texts += dec_tokenizer.batch_decode(
            out["generated_text_ids"],
            skip_special_tokens=True,
        )

        target_texts += dec_tokenizer.batch_decode(
            out["target_ids"],
            skip_special_tokens=True,
        )

        n += 1

    for k in totals:
        totals[k] /= max(n, 1)

    if lpips_model is None:
        totals["lpips"] = None

    totals.update(text_metrics(pred_texts, target_texts))

    print("Predictor test metrics:")
    for k, v in totals.items():
        if v is None:
            print(f"  {k}: unavailable")
        else:
            print(f"  {k}: {v:.4f}")

    return totals


# =========================================================
# Attention visualization
# =========================================================

@torch.no_grad()
def plot_temporal_attention(
    predictor,
    dataloader,
    device,
    max_samples=8,
):
    predictor.eval()

    batch = next(iter(dataloader))
    out = predict_batch(predictor, batch, device)

    items = [
        ("Visual", out["visual_attn_weights"]),
        ("Text", out["text_attn_weights"]),
    ]

    if "fusion_attn_weights" in out:
        items.append(("Fusion", out["fusion_attn_weights"]))

    n = min(max_samples, out["visual_attn_weights"].size(0))

    fig, axes = plt.subplots(1, len(items), figsize=(5 * len(items), 0.6 * n + 3))

    if len(items) == 1:
        axes = [axes]

    for ax, (title, weights) in zip(axes, items):
        mat = weights[:n].detach().cpu().numpy()

        im = ax.imshow(mat, cmap="viridis", aspect="auto")
        ax.set_title(f"{title} attention")
        ax.set_xlabel("Input frame/story")
        ax.set_ylabel("Sample")
        ax.set_xticks(range(mat.shape[1]))
        ax.set_xticklabels([str(i + 1) for i in range(mat.shape[1])])

        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=7,
                )

        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


# =========================================================
# Qualitative visualization
# =========================================================

@torch.no_grad()
def visualize_predictions(
    predictor,
    dataloader,
    enc_tokenizer,
    dec_tokenizer,
    device,
    max_samples=3,
    max_new_tokens=80,
):
    predictor.eval()

    batch = next(iter(dataloader))
    out = predict_batch(
        predictor,
        batch,
        device,
        max_new_tokens=max_new_tokens,
    )

    pred_texts = dec_tokenizer.batch_decode(
        out["generated_text_ids"],
        skip_special_tokens=True,
    )

    target_texts = dec_tokenizer.batch_decode(
        out["target_ids"],
        skip_special_tokens=True,
    )

    n = min(max_samples, out["image_seq"].size(0))

    for b in range(n):
        fig, axes = plt.subplots(1, 6, figsize=(18, 4))

        for t in range(4):
            axes[t].imshow(
                out["image_seq"][b, t].detach().cpu().permute(1, 2, 0).clamp(0, 1)
            )
            axes[t].set_title(f"Input frame {t + 1}")
            axes[t].axis("off")

        axes[4].imshow(
            out["image_target"][b].detach().cpu().permute(1, 2, 0).clamp(0, 1)
        )
        axes[4].set_title("Target frame 5")
        axes[4].axis("off")

        axes[5].imshow(
            out["pred_image"][b].detach().cpu().permute(1, 2, 0).clamp(0, 1)
        )
        axes[5].set_title("Predicted frame 5")
        axes[5].axis("off")

        input_texts = [
            enc_tokenizer.decode(
                out["enc_ids"][b, t],
                skip_special_tokens=True,
            )
            for t in range(4)
        ]

        text_block = "\n".join(
            [
                f"Input {i + 1}: {textwrap.shorten(txt, width=130)}"
                for i, txt in enumerate(input_texts)
            ]
        )

        text_block += "\n\n"
        text_block += "Target 5: " + textwrap.shorten(target_texts[b], width=170)
        text_block += "\n"
        text_block += "Predicted 5: " + textwrap.shorten(pred_texts[b], width=170)

        fig.suptitle(f"Prediction sample {b + 1}", fontsize=14)
        fig.text(0.01, -0.05, text_block, fontsize=10, va="top")

        plt.tight_layout()
        plt.show()
        plt.close(fig)
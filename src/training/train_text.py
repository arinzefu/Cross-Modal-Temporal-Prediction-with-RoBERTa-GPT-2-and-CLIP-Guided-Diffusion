# =========================================================
# src/training/train_text.py
# Training utilities for RoBERTa -> GPT-2 text autoencoder
# =========================================================

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup


# =========================================================
# GOOGLE DRIVE CHECKPOINT HELPERS
# =========================================================

DRIVE_CHECKPOINT_FOLDER = "/content/drive/MyDrive/DL_Checkpoints"
DRIVE_LOG_PATH = os.path.join(DRIVE_CHECKPOINT_FOLDER, "training_text_log.txt")


def move_optimizer_to_device(optimizer, device):
    """
    After loading optimizer state from checkpoint, move optimizer tensors
    to the current device.
    """
    if optimizer is None:
        return optimizer

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    return optimizer


def save_checkpoint_to_drive(
    model,
    optimizer,
    scheduler,
    epoch,
    current_loss,
    best_loss,
    history,
    filename="roberta2gpt2.pth",
):
    os.makedirs(DRIVE_CHECKPOINT_FOLDER, exist_ok=True)
    full_path = os.path.join(DRIVE_CHECKPOINT_FOLDER, filename)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "current_loss": current_loss,
        "best_loss": best_loss,
        "history": history,
    }

    torch.save(checkpoint, full_path)
    print(f"✓ Checkpoint saved to Drive: {full_path}  (epoch {epoch})")


def load_checkpoint_from_drive(
    model,
    optimizer=None,
    scheduler=None,
    filename="roberta2gpt2.pth",
    device="cpu",
):
    full_path = os.path.join(DRIVE_CHECKPOINT_FOLDER, filename)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Checkpoint not found: {full_path}")

    checkpoint = torch.load(full_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer = move_optimizer_to_device(optimizer, device)

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    best_loss = checkpoint.get(
        "best_loss",
        checkpoint.get("loss", checkpoint.get("current_loss", float("inf"))),
    )

    history = checkpoint.get(
        "history",
        {
            "epoch": [],
            "ce_loss": [],
            "div_loss": [],
            "total_loss": [],
            "bleu": [],
            "rouge1": [],
            "rouge2": [],
            "rougeL": [],
            "meteor": [],
            "avg_overlap": [],
        },
    )

    # Backward compatibility if older checkpoint history is missing keys.
    for key in [
        "epoch",
        "ce_loss",
        "div_loss",
        "total_loss",
        "bleu",
        "rouge1",
        "rouge2",
        "rougeL",
        "meteor",
        "avg_overlap",
    ]:
        history.setdefault(key, [])

    print(f"✓ Checkpoint loaded from Drive: {full_path}  (epoch {epoch})")

    return model, optimizer, scheduler, epoch, best_loss, history


def append_epoch_log(epoch, ce, div, total, metrics, filepath=DRIVE_LOG_PATH):
    os.makedirs(DRIVE_CHECKPOINT_FOLDER, exist_ok=True)
    new_file = not os.path.exists(filepath)

    with open(filepath, "a", encoding="utf-8") as f:
        if new_file:
            f.write("=== TEXT AUTOENCODER TRAINING LOG ===\n\n")

        f.write(f"Epoch {epoch}\n")
        f.write(f"  CE Loss      : {ce:.6f}\n")
        f.write(f"  Div Loss     : {div:.6f}\n")
        f.write(f"  Total Loss   : {total:.6f}\n")
        f.write(f"  BLEU         : {metrics.get('bleu', 0.0):.6f}\n")
        f.write(f"  ROUGE-1      : {metrics.get('rouge1', 0.0):.6f}\n")
        f.write(f"  ROUGE-2      : {metrics.get('rouge2', 0.0):.6f}\n")
        f.write(f"  ROUGE-L      : {metrics.get('rougeL', 0.0):.6f}\n")
        f.write(f"  METEOR       : {metrics.get('meteor', 0.0):.6f}\n")
        f.write(f"  Avg Overlap  : {metrics.get('avg_overlap', 0.0):.6f}\n")
        f.write("\n")

    print(f"✓ Epoch {epoch} log appended to Drive: {filepath}")


# =========================================================
# EARLY STOPPING
# =========================================================

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = None
        self.num_bad_epochs = 0
        self.stop = False

    def step(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
            return False

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.stop = True

        return self.stop


# =========================================================
# CONTRASTIVE DIVERSITY LOSS
# =========================================================

class ContrastiveDiversityLoss(nn.Module):
    """
    Optional latent diversity loss.

    For your current recovery stage, keep diversity_weight=0.0 first.
    Add this only after reconstruction becomes strong.
    """

    def __init__(self, margin: float = 0.85):
        super().__init__()
        self.margin = margin

    def forward(self, memory, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()

        pooled = (memory * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        pooled = F.normalize(pooled, dim=-1)

        sim = torch.matmul(pooled, pooled.T)
        batch_size = pooled.size(0)

        if batch_size < 2:
            return torch.tensor(0.0, device=memory.device)

        mask_off_diag = ~torch.eye(batch_size, dtype=torch.bool, device=memory.device)

        return F.relu(sim[mask_off_diag] - self.margin).mean()


# =========================================================
# OPTIMIZER
# =========================================================

def build_optimizer(
    model,
    lr_encoder=1e-5,
    lr_decoder=1e-4,
    weight_decay=0.01,
):
    """
    Low LR for RoBERTa encoder.
    Higher LR for GPT-2 decoder and fresh cross-attention layers.
    """

    encoder_param_ids = {id(p) for p in model.encoder.parameters()}

    enc_params = [
        p for p in model.parameters()
        if id(p) in encoder_param_ids and p.requires_grad
    ]

    dec_params = [
        p for p in model.parameters()
        if id(p) not in encoder_param_ids and p.requires_grad
    ]

    param_groups = []

    if len(enc_params) > 0:
        param_groups.append(
            {
                "params": enc_params,
                "lr": lr_encoder,
                "name": "encoder",
            }
        )

    if len(dec_params) > 0:
        param_groups.append(
            {
                "params": dec_params,
                "lr": lr_decoder,
                "name": "decoder",
            }
        )

    if len(param_groups) == 0:
        raise ValueError("No trainable parameters found. Check requires_grad settings.")

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=weight_decay,
    )

    return optimizer


def get_lr_by_name(optimizer, group_name):
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            return group["lr"]
    return None


# =========================================================
# SCHEDULER
# =========================================================

def build_scheduler(
    optimizer,
    dataloader_len,
    n_epochs,
    warmup_ratio=0.08,
    gradient_accumulation_steps=1,
):
    steps_per_epoch = math.ceil(dataloader_len / gradient_accumulation_steps)
    num_training_steps = steps_per_epoch * n_epochs
    num_warmup_steps = int(warmup_ratio * num_training_steps)

    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


# =========================================================
# LOSS
# =========================================================

def compute_ce_loss(logits, labels, label_smoothing=0.0):
    """
    Computes CE loss manually so label smoothing actually works.

    labels should contain -100 for ignored padding positions.
    """

    vocab_size = logits.size(-1)

    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
        label_smoothing=label_smoothing,
    )


# =========================================================
# SAFE TOKENIZATION FOR METRICS
# =========================================================

def safe_word_tokenize(text):
    try:
        return nltk.word_tokenize(str(text).lower())
    except LookupError:
        return str(text).lower().split()


# =========================================================
# GENERATION
# =========================================================

def get_decoder_start_token_id(model, dec_tokenizer):
    """
    GPT-2 often uses eos as the decoder start token.
    This prevents decoder_start_token_id=None errors.
    """

    if hasattr(model, "model"):
        config = model.model.config
    else:
        config = model.config

    decoder_start = getattr(config, "decoder_start_token_id", None)

    if decoder_start is not None:
        return decoder_start

    if dec_tokenizer.bos_token_id is not None:
        return dec_tokenizer.bos_token_id

    if dec_tokenizer.eos_token_id is not None:
        return dec_tokenizer.eos_token_id

    raise ValueError("Could not resolve decoder_start_token_id.")


@torch.no_grad()
def generate_text(
    model,
    enc_tokenizer,
    dec_tokenizer,
    prompt,
    device,
    max_new_tokens=160,
    num_beams=3,
    do_sample=False,
    temperature=0.8,
    top_p=0.9,
    top_k=50,
    no_repeat_ngram_size=3,
    repetition_penalty=1.15,
):
    model.eval()

    enc = enc_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    decoder_start_token_id = get_decoder_start_token_id(model, dec_tokenizer)

    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "decoder_start_token_id": decoder_start_token_id,
        "pad_token_id": dec_tokenizer.pad_token_id,
        "eos_token_id": dec_tokenizer.eos_token_id,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "repetition_penalty": repetition_penalty,
    }

    if do_sample:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
        )
    else:
        generation_kwargs.update(
            {
                "do_sample": False,
                "num_beams": num_beams,
                "early_stopping": True,
            }
        )

    gen = model.generate(**generation_kwargs)

    text = dec_tokenizer.decode(
        gen[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    model.train()
    return text


# =========================================================
# METRICS
# =========================================================

def compute_metrics(hypothesis, reference):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer as rs_mod

    hyp_tok = safe_word_tokenize(hypothesis)
    ref_tok = safe_word_tokenize(reference)

    if len(hyp_tok) == 0 or len(ref_tok) == 0:
        bleu = 0.0
        meteor = 0.0
    else:
        bleu = sentence_bleu(
            [ref_tok],
            hyp_tok,
            smoothing_function=SmoothingFunction().method1,
        )

        try:
            meteor = meteor_score([ref_tok], hyp_tok)
        except Exception:
            meteor = 0.0

    scorer = rs_mod.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    rscores = scorer.score(str(reference), str(hypothesis))

    return {
        "bleu": bleu,
        "meteor": meteor,
        "rouge1": rscores["rouge1"].fmeasure,
        "rouge2": rscores["rouge2"].fmeasure,
        "rougeL": rscores["rougeL"].fmeasure,
    }


def text_overlap(a, b):
    sa = set(safe_word_tokenize(a))
    sb = set(safe_word_tokenize(b))

    if not sa and not sb:
        return 1.0

    if not sa or not sb:
        return 0.0

    return len(sa & sb) / len(sa | sb)


# =========================================================
# EVALUATION
# =========================================================

def run_epoch_evaluation(
    model,
    enc_tokenizer,
    dec_tokenizer,
    device,
    epoch,
    eval_prompt,
    eval_reference=None,
    diversity_prompts=None,
    max_new_tokens=160,
):
    """
    For text autoencoder evaluation, the most meaningful setup is:

        eval_prompt == eval_reference

    because the model is reconstructing text, not predicting the next story yet.
    """

    print(f"\n{'=' * 70}")
    print(f"  EPOCH {epoch} EVALUATION")
    print(f"{'=' * 70}")

    if eval_reference is None:
        eval_reference = eval_prompt

    gen_main = generate_text(
        model=model,
        enc_tokenizer=enc_tokenizer,
        dec_tokenizer=dec_tokenizer,
        prompt=eval_prompt,
        device=device,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=3,
    )

    metrics = compute_metrics(gen_main, eval_reference)

    print(f"\nREFERENCE:\n{eval_reference[:500]}")
    print(f"\nGENERATED:\n{gen_main[:500]}")

    print("\nScores:")
    print(f"BLEU-4  : {metrics['bleu']:.4f}")
    print(f"METEOR  : {metrics['meteor']:.4f}")
    print(f"ROUGE-1 : {metrics['rouge1']:.4f}")
    print(f"ROUGE-2 : {metrics['rouge2']:.4f}")
    print(f"ROUGE-L : {metrics['rougeL']:.4f}")

    avg_overlap = 0.0

    if diversity_prompts:
        print("\nDIVERSITY / CONDITIONING TEST\n")

        gens = {}

        for name, prompt in diversity_prompts.items():
            g = generate_text(
                model=model,
                enc_tokenizer=enc_tokenizer,
                dec_tokenizer=dec_tokenizer,
                prompt=prompt,
                device=device,
                max_new_tokens=80,
                do_sample=False,
                num_beams=3,
            )

            gens[name] = g

            print(f"[{name}]")
            print(f"PROMPT : {prompt}")
            print(f"OUTPUT : {g[:250]}\n")

        names = list(gens.keys())
        overlap_vals = []

        print("Pairwise overlaps:\n")

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                ov = text_overlap(gens[names[i]], gens[names[j]])
                overlap_vals.append(ov)
                print(f"{names[i]} ↔ {names[j]} : {ov:.3f}")

        avg_overlap = sum(overlap_vals) / max(len(overlap_vals), 1)
        print(f"\nAverage overlap: {avg_overlap:.3f}")

    metrics["avg_overlap"] = avg_overlap

    print(f"{'=' * 70}\n")

    model.train()
    return metrics


# =========================================================
# PLOTTING
# =========================================================

def plot_history(history, epoch):
    ep = history["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(ep, history["ce_loss"], label="CE Loss", marker="o", markersize=3)
    axes[0].plot(ep, history["total_loss"], label="Total Loss", marker="s", markersize=3)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(fontsize=8)

    axes[1].plot(ep, history["bleu"], label="BLEU", marker="o", markersize=3)
    axes[1].plot(ep, history["meteor"], label="METEOR", marker="s", markersize=3)
    axes[1].set_title("Generation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=8)

    axes[2].plot(ep, history["rouge1"], label="ROUGE-1", marker="o", markersize=3)
    axes[2].plot(ep, history["rougeL"], label="ROUGE-L", marker="s", markersize=3)
    axes[2].set_title("ROUGE")
    axes[2].set_xlabel("Epoch")
    axes[2].legend(fontsize=8)

    plt.suptitle(
        f"RoBERTa → GPT-2 Text Autoencoder — after epoch {epoch}",
        fontsize=10,
    )

    plt.tight_layout()
    plt.show()


# =========================================================
# TRAINING LOOP
# =========================================================

def train_text_autoencoder(
    model,
    dataloader,
    enc_tokenizer,
    dec_tokenizer,
    optimizer,
    scheduler,
    device,
    n_epochs=4,
    checkpoint_filename="roberta2gpt2.pth",
    eval_prompt=None,
    eval_reference=None,
    diversity_prompts=None,
    max_grad_norm=1.0,
    label_smoothing=0.0,
    ce_weight=1.0,
    diversity_weight=0.0,
    gradient_accumulation_steps=1,
    resume=True,
    plot_every=1,
):
    """
    Compatible with batches containing:

        batch["input_ids"]
        batch["attention_mask"]
        batch["labels"]

    Optional but recommended:

        batch["target_attention_mask"]

    The dataset should already set label padding positions to -100.
    """

    if isinstance(device, str):
        device = torch.device(device)

    use_amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    contrastive_loss_fn = ContrastiveDiversityLoss()

    start_epoch = 0
    best_loss = float("inf")

    history = {
        "epoch": [],
        "ce_loss": [],
        "div_loss": [],
        "total_loss": [],
        "bleu": [],
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
        "meteor": [],
        "avg_overlap": [],
    }

    # -----------------------------------------------------
    # Resume checkpoint
    # -----------------------------------------------------
    if resume:
        try:
            model, optimizer, scheduler, start_epoch, best_loss, history = load_checkpoint_from_drive(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                filename=checkpoint_filename,
                device=device,
            )

            print(
                f"Resuming from epoch {start_epoch} | "
                f"best loss so far: {best_loss:.5f}"
            )

        except FileNotFoundError:
            print("No checkpoint found on Drive — training from scratch.")
    else:
        print("resume=False — training from scratch.")

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------
    for epoch in range(start_epoch, n_epochs):
        model.train()

        epoch_ce_loss = []
        epoch_div_loss = []
        epoch_total_loss = []

        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{n_epochs}",
        )

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            decoder_attention_mask = None
            if "target_attention_mask" in batch:
                decoder_attention_mask = batch["target_attention_mask"].to(device)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    decoder_attention_mask=decoder_attention_mask,
                )

                if label_smoothing > 0:
                    ce_loss = compute_ce_loss(
                        logits=out.logits,
                        labels=labels,
                        label_smoothing=label_smoothing,
                    )
                else:
                    ce_loss = out.loss

                div_loss = torch.tensor(0.0, device=device)

                if diversity_weight > 0:
                    memory = model.encode(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                    div_loss = contrastive_loss_fn(
                        memory=memory,
                        attention_mask=attention_mask,
                    )

                total_loss = (ce_weight * ce_loss) + (diversity_weight * div_loss)

                loss_for_backward = total_loss / gradient_accumulation_steps

            scaler.scale(loss_for_backward).backward()

            should_step = (
                (step + 1) % gradient_accumulation_steps == 0
                or (step + 1) == len(dataloader)
            )

            if should_step:
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm,
                )

                scaler.step(optimizer)
                scaler.update()

                if scheduler is not None:
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)

            epoch_ce_loss.append(ce_loss.detach().item())
            epoch_div_loss.append(div_loss.detach().item())
            epoch_total_loss.append(total_loss.detach().item())

            enc_lr = get_lr_by_name(optimizer, "encoder")
            dec_lr = get_lr_by_name(optimizer, "decoder")

            postfix = {
                "ce": f"{ce_loss.detach().item():.4f}",
                "total": f"{total_loss.detach().item():.4f}",
            }

            if enc_lr is not None:
                postfix["enc_lr"] = f"{enc_lr:.2e}"

            if dec_lr is not None:
                postfix["dec_lr"] = f"{dec_lr:.2e}"

            pbar.set_postfix(postfix)

        avg_ce = sum(epoch_ce_loss) / max(len(epoch_ce_loss), 1)
        avg_div = sum(epoch_div_loss) / max(len(epoch_div_loss), 1)
        avg_total = sum(epoch_total_loss) / max(len(epoch_total_loss), 1)

        print(f"\nEpoch {epoch + 1}")
        print(f"CE Loss     : {avg_ce:.5f}")
        print(f"Div Loss    : {avg_div:.5f}")
        print(f"Total Loss  : {avg_total:.5f}")

        # -------------------------------------------------
        # Evaluation
        # -------------------------------------------------
        if eval_prompt:
            metrics = run_epoch_evaluation(
                model=model,
                enc_tokenizer=enc_tokenizer,
                dec_tokenizer=dec_tokenizer,
                device=device,
                epoch=epoch + 1,
                eval_prompt=eval_prompt,
                eval_reference=eval_reference,
                diversity_prompts=diversity_prompts,
            )
        else:
            metrics = {
                "bleu": 0.0,
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "meteor": 0.0,
                "avg_overlap": 0.0,
            }

        # -------------------------------------------------
        # History
        # -------------------------------------------------
        history["epoch"].append(epoch + 1)
        history["ce_loss"].append(avg_ce)
        history["div_loss"].append(avg_div)
        history["total_loss"].append(avg_total)
        history["bleu"].append(metrics["bleu"])
        history["rouge1"].append(metrics["rouge1"])
        history["rouge2"].append(metrics["rouge2"])
        history["rougeL"].append(metrics["rougeL"])
        history["meteor"].append(metrics["meteor"])
        history["avg_overlap"].append(metrics["avg_overlap"])

        # -------------------------------------------------
        # Checkpoint
        # -------------------------------------------------
        improved = avg_total < best_loss

        if improved:
            best_loss = avg_total

        save_checkpoint_to_drive(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            current_loss=avg_total,
            best_loss=best_loss,
            history=history,
            filename=checkpoint_filename,
        )

        print(f"  {'✓ new best' if improved else '-'}")

        append_epoch_log(
            epoch=epoch + 1,
            ce=avg_ce,
            div=avg_div,
            total=avg_total,
            metrics=metrics,
        )

        # -------------------------------------------------
        # Plot
        # -------------------------------------------------
        if plot_every is not None and plot_every > 0:
            if (epoch + 1) % plot_every == 0:
                plot_history(history, epoch + 1)

    print("\n✓ Training complete.")
    return history
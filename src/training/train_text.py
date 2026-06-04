import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from src.training.train_visual import EarlyStopping, train_diffusion
from test.visual_test import single_image_diffusion_test

# =========================================================
# GOOGLE DRIVE CHECKPOINT HELPERS
# =========================================================
DRIVE_CHECKPOINT_FOLDER = "/content/drive/MyDrive/DL_Checkpoints"


def save_checkpoint_to_drive(model, optimizer, epoch, loss, history, filename="roberta2gpt2.pth"):
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


def load_checkpoint_from_drive(model, optimizer=None, filename="roberta2gpt2.pth"):
    full_path = os.path.join(DRIVE_CHECKPOINT_FOLDER, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Checkpoint not found: {full_path}")

    checkpoint = torch.load(full_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", float("inf"))
    history = checkpoint.get("history", {
        "epoch": [], "ce_loss": [], "div_loss": [], "total_loss": [],
        "bleu": [], "rouge1": [], "meteor": [],
    })
    print(f"\u2713 Checkpoint loaded from Drive: {full_path}  (epoch {epoch})")
    return model, optimizer, epoch, loss, history


DRIVE_LOG_PATH = os.path.join(DRIVE_CHECKPOINT_FOLDER, "training_text_log.txt")


def append_epoch_log(epoch, ce, div, tot, metrics, filepath=DRIVE_LOG_PATH):
    os.makedirs(DRIVE_CHECKPOINT_FOLDER, exist_ok=True)
    new_file = not os.path.exists(filepath)
    with open(filepath, "a", encoding="utf-8") as f:    # "a" = append, never truncates
        if new_file:
            f.write("=== TRAINING LOG ===\n\n")
        f.write(f"Epoch {epoch}\n")
        f.write(f"  CE Loss     : {ce:.6f}\n")
        f.write(f"  Div Loss    : {div:.6f}\n")
        f.write(f"  Total Loss  : {tot:.6f}\n")
        f.write(f"  BLEU        : {metrics['bleu']:.6f}\n")
        f.write(f"  ROUGE-1     : {metrics['rouge1']:.6f}\n")
        f.write(f"  METEOR      : {metrics['meteor']:.6f}\n")
        f.write("\n")
    print(f"\u2713 Epoch {epoch} log appended to Drive: {filepath}")


# =========================================================
# EARLY STOPPING  (kept for import compatibility)
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
            self.best_loss = current_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.stop = True
        return self.stop


# =========================================================
# CONTRASTIVE DIVERSITY LOSS  (kept for import compatibility)
# =========================================================
class ContrastiveDiversityLoss(nn.Module):
    def __init__(self, margin: float = 0.85):
        super().__init__()
        self.margin = margin

    def forward(self, memory, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (memory * mask).sum(1) / mask.sum(1).clamp(min=1)
        pooled = F.normalize(pooled, dim=-1)
        sim = torch.matmul(pooled, pooled.T)
        B = pooled.size(0)
        if B < 2:
            return torch.tensor(0.0, device=memory.device)
        mask_off = ~torch.eye(B, dtype=torch.bool, device=memory.device)
        return F.relu(sim[mask_off] - self.margin).mean()


# =========================================================
# OPTIMIZER  -- low LR on pretrained RoBERTa, higher LR on the GPT-2 decoder
# (the fresh cross-attention lives inside the decoder, so it gets the higher LR)
# =========================================================
def build_optimizer(model, lr_encoder=1e-5, lr_decoder=1e-4, weight_decay=0.01):
    encoder_param_ids = {id(p) for p in model.encoder.parameters()}

    enc_params = [p for p in model.parameters()
                  if id(p) in encoder_param_ids and p.requires_grad]
    dec_params = [p for p in model.parameters()
                  if id(p) not in encoder_param_ids and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": enc_params, "lr": lr_encoder, "name": "encoder"},
            {"params": dec_params, "lr": lr_decoder, "name": "decoder"},
        ],
        weight_decay=weight_decay,
    )
    return optimizer


# =========================================================
# SCHEDULER
# =========================================================
def build_scheduler(optimizer, dataloader_len, n_epochs, warmup_ratio=0.08):
    num_training_steps = dataloader_len * n_epochs
    num_warmup_steps = int(warmup_ratio * num_training_steps)
    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


# =========================================================
# GENERATION  -- nucleus sampling via HF .generate (two tokenizers)
# =========================================================
@torch.no_grad()
def generate_text(
    model,
    enc_tokenizer,
    dec_tokenizer,
    prompt,
    device,
    max_new_tokens=200,
    temperature=0.9,
    top_p=0.92,
    top_k=50,
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

    gen = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        no_repeat_ngram_size=3,
        decoder_start_token_id=dec_tokenizer.bos_token_id,
        pad_token_id=dec_tokenizer.pad_token_id,
        eos_token_id=dec_tokenizer.eos_token_id,
    )

    text = dec_tokenizer.decode(gen[0], skip_special_tokens=True)
    model.train()
    return text


# =========================================================
# METRICS
# =========================================================
def compute_metrics(hypothesis, reference):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer as rs_mod

    hyp_tok = nltk.word_tokenize(hypothesis.lower())
    ref_tok = nltk.word_tokenize(reference.lower())

    bleu = sentence_bleu([ref_tok], hyp_tok, smoothing_function=SmoothingFunction().method1)
    meteor = meteor_score([ref_tok], hyp_tok)

    scorer = rs_mod.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rscores = scorer.score(reference, hypothesis)

    return {
        "bleu": bleu,
        "meteor": meteor,
        "rouge1": rscores["rouge1"].fmeasure,
        "rouge2": rscores["rouge2"].fmeasure,
        "rougeL": rscores["rougeL"].fmeasure,
    }


# =========================================================
# TEXT OVERLAP
# =========================================================
def text_overlap(a, b):
    sa = set(nltk.word_tokenize(a.lower()))
    sb = set(nltk.word_tokenize(b.lower()))
    if not sa and not sb:
        return 1.0
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
    eval_reference,
    diversity_prompts,
):
    print(f"\n{'='*70}")
    print(f"  EPOCH {epoch} EVALUATION")
    print(f"{'='*70}")

    gen_main = generate_text(model, enc_tokenizer, dec_tokenizer, eval_prompt, device)
    metrics = compute_metrics(gen_main, eval_reference)

    print(f"\nREFERENCE:\n{eval_reference[:300]}")
    print(f"\nGENERATED:\n{gen_main[:400]}")
    print(f"\nScores:")
    print(f"BLEU-4  : {metrics['bleu']:.4f}")
    print(f"METEOR  : {metrics['meteor']:.4f}")
    print(f"ROUGE-1 : {metrics['rouge1']:.4f}")
    print(f"ROUGE-2 : {metrics['rouge2']:.4f}")
    print(f"ROUGE-L : {metrics['rougeL']:.4f}")

    print(f"\nDIVERSITY TEST\n")
    gens = {}
    for name, prompt in diversity_prompts.items():
        g = generate_text(model, enc_tokenizer, dec_tokenizer, prompt, device)
        gens[name] = g
        print(f"[{name}]")
        print(f"PROMPT : {prompt}")
        print(f"OUTPUT : {g[:200]}\n")

    names = list(gens.keys())
    overlap_vals = []
    print("Pairwise overlaps:\n")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ov = text_overlap(gens[names[i]], gens[names[j]])
            overlap_vals.append(ov)
            print(f"{names[i]} \u2194 {names[j]} : {ov:.3f}")

    avg_overlap = sum(overlap_vals) / max(len(overlap_vals), 1)
    print(f"\nAverage overlap: {avg_overlap:.3f}")
    print(f"{'='*70}\n")

    model.train()
    return metrics


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
):
    scaler = torch.amp.GradScaler("cuda")

    # ---- resume from Drive if a checkpoint exists ----
    start_epoch = 0
    best_loss = float("inf")
    history = {
        "epoch": [], "ce_loss": [], "div_loss": [], "total_loss": [],
        "bleu": [], "rouge1": [], "meteor": [],
    }

    try:
        model, optimizer, start_epoch, best_loss, history = load_checkpoint_from_drive(
            model, optimizer, filename=checkpoint_filename
        )
        print(f"  Resuming from epoch {start_epoch}  |  best loss so far: {best_loss:.5f}")
    except FileNotFoundError:
        print("No checkpoint found on Drive \u2013 training from scratch.")

    # ---- training ----
    for epoch in range(start_epoch, n_epochs):
        model.train()
        epoch_loss = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss.append(loss.item())
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "enc_lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                "dec_lr": f"{optimizer.param_groups[1]['lr']:.2e}",
            })

        avg_ce = sum(epoch_loss) / len(epoch_loss)

        print(f"\nEpoch {epoch+1}")
        print(f"CE Loss    : {avg_ce:.5f}")

        # ---- evaluation ----
        if eval_prompt and eval_reference and diversity_prompts:
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
            metrics = {"bleu": 0, "rouge1": 0, "rouge2": 0, "rougeL": 0, "meteor": 0}

        # ---- accumulate history ----
        history["epoch"].append(epoch + 1)
        history["ce_loss"].append(avg_ce)
        history["div_loss"].append(0.0)        # no diversity loss in this setup
        history["total_loss"].append(avg_ce)
        history["bleu"].append(metrics["bleu"])
        history["rouge1"].append(metrics["rouge1"])
        history["meteor"].append(metrics["meteor"])

        # ---- save to Drive after EVERY epoch ----
        improved = avg_ce < best_loss
        if improved:
            best_loss = avg_ce

        save_checkpoint_to_drive(
            model, optimizer, epoch + 1, best_loss, history, filename=checkpoint_filename
        )
        print(f"  {'\u2713 new best' if improved else '-'}")
        append_epoch_log(epoch + 1, avg_ce, 0.0, avg_ce, metrics)

        # ---- plot full history (includes all prior runs) ----
        ep = history["epoch"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(ep, history["ce_loss"], label="CE / Total", marker="o", markersize=3)
        axes[0].set_title("Training loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(fontsize=8)

        axes[1].plot(ep, history["bleu"], label="BLEU", marker="o", markersize=3, color="steelblue")
        axes[1].plot(ep, history["meteor"], label="METEOR", marker="s", markersize=3, color="darkorange")
        axes[1].set_title("Generation Metrics"); axes[1].set_xlabel("Epoch"); axes[1].legend(fontsize=8)

        axes[2].plot(ep, history["rouge1"], label="ROUGE-1", marker="o", markersize=3, color="mediumseagreen")
        axes[2].set_title("ROUGE-1 (vs fixed reference)"); axes[2].set_xlabel("Epoch"); axes[2].legend(fontsize=8)

        plt.suptitle(f"RoBERTa2GPT2 story continuation \u2013 after epoch {epoch + 1}", fontsize=10)
        plt.tight_layout()
        plt.show()

    print("\n\u2713 Training complete.")
    return history

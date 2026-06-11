# =========================================================
# text_autoencoder.py
# RoBERTa encoder + GPT-2 decoder text autoencoder
# =========================================================

import math
from typing import Optional

import torch
import torch.nn as nn

from transformers import RobertaModel, EncoderDecoderModel, AutoConfig

# =========================================================
# LEGACY COMPONENTS
# ---------------------------------------------------------
# =========================================================
# Imports
# =========================================================

import re
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from bs4 import BeautifulSoup
from transformers import RobertaTokenizer, GPT2Tokenizer


# =========================================================
# Text cleaning helper
# =========================================================

def clean_description(text):
    """
    Cleans extracted GDI text by removing excessive whitespace.
    """
    if text is None:
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================
# GDI parser
# =========================================================

def parse_gdi_text(text):
    """
    Parses a story containing multiple <gdi imageX>...</gdi> blocks.

    Returns a list of dictionaries:

        {
            "image_id": "1",
            "description": "...",
            "objects": [...],
            "actions": [...],
            "locations": [...],
            "raw_text": "<gdi image1>...</gdi>"
        }
    """

    if text is None:
        return []

    soup = BeautifulSoup(str(text), "html.parser")
    images = []

    for gdi in soup.find_all("gdi"):
        image_id = None

        # Case: <gdi image1>
        if gdi.attrs:
            for attr_name, _ in gdi.attrs.items():
                if "image" in attr_name.lower():
                    image_id = attr_name.lower().replace("image", "")
                    break

        # Fallback regex case
        if not image_id:
            tag_str = str(gdi)
            match = re.search(r"<gdi\s+image(\d+)", tag_str, flags=re.IGNORECASE)
            if match:
                image_id = match.group(1)

        # Final fallback
        if not image_id:
            image_id = str(len(images) + 1)

        # Full clean description without tags
        description = clean_description(gdi.get_text(" ", strip=True))

        # Extract tagged elements
        objects = [
            clean_description(obj.get_text(" ", strip=True))
            for obj in gdi.find_all("gdo")
        ]

        actions = [
            clean_description(act.get_text(" ", strip=True))
            for act in gdi.find_all("gda")
        ]

        locations = [
            clean_description(loc.get_text(" ", strip=True))
            for loc in gdi.find_all("gdl")
        ]

        images.append(
            {
                "image_id": image_id,
                "description": description,
                "objects": objects,
                "actions": actions,
                "locations": locations,
                "raw_text": str(gdi),
            }
        )

    return images


# =========================================================
# Text autoencoder dataset
# =========================================================

class TextTaskDataset(Dataset):
    """
    Dataset for RoBERTa encoder + GPT-2 decoder text autoencoder.

    It converts each full story row into separate frame-level examples.

    Example:
        3,500 stories × 6 GDI frames ≈ 21,000 text samples

    Returned keys:
        input_ids              -> RoBERTa encoder input
        attention_mask         -> RoBERTa encoder attention mask
        labels                 -> GPT-2 target ids with pad positions set to -100

        enc_input_ids          -> alias for input_ids
        enc_attention_mask     -> alias for attention_mask
        target_ids             -> GPT-2 target ids before -100 masking
        target_attention_mask  -> GPT-2 target attention mask
    """

    def __init__(
        self,
        dataset,
        enc_tokenizer,
        dec_tokenizer,
        max_len=256,
        target_max_len=None,
        label_pad_id=-100,
        add_eos_token=True,
        min_text_chars=10,
        story_key="story",
    ):
        self.dataset = dataset
        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer
        self.max_len = min(max_len, 512)
        self.target_max_len = target_max_len or self.max_len
        self.label_pad_id = label_pad_id
        self.add_eos_token = add_eos_token
        self.story_key = story_key

        # GPT-2 has no pad token by default.
        if self.dec_tokenizer.pad_token is None:
            self.dec_tokenizer.pad_token = self.dec_tokenizer.eos_token

        self.samples = []

        # -------------------------------------------------
        # Flatten dataset from story-level to frame-level.
        # -------------------------------------------------
        for row_idx in range(len(dataset)):
            sample = dataset[row_idx]

            if self.story_key not in sample:
                raise KeyError(
                    f"Expected key '{self.story_key}' in dataset sample, "
                    f"but got keys: {list(sample.keys())}"
                )

            story = sample[self.story_key]
            image_attributes = parse_gdi_text(story)

            for frame_idx, frame_data in enumerate(image_attributes):
                description = clean_description(frame_data.get("description", ""))

                if len(description) < min_text_chars:
                    continue

                self.samples.append(
                    {
                        "row_idx": row_idx,
                        "frame_idx": frame_idx,
                        "image_id": frame_data.get("image_id", str(frame_idx + 1)),
                        "description": description,
                        "objects": frame_data.get("objects", []),
                        "actions": frame_data.get("actions", []),
                        "locations": frame_data.get("locations", []),
                        "raw_text": frame_data.get("raw_text", ""),
                    }
                )

        if len(self.samples) == 0:
            raise ValueError(
                "TextTaskDataset found 0 usable text samples. "
                "Check that your dataset has a 'story' column and valid <gdi> blocks."
            )

        print(f"TextTaskDataset created with {len(self.samples):,} frame-level samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        description = item["description"]

        # -------------------------------------------------
        # RoBERTa encoder input
        # -------------------------------------------------
        enc = self.enc_tokenizer(
            description,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        enc_input_ids = enc["input_ids"].squeeze(0)
        enc_attention_mask = enc["attention_mask"].squeeze(0)

        # -------------------------------------------------
        # GPT-2 decoder target
        # -------------------------------------------------
        target_text = description

        if self.add_eos_token and self.dec_tokenizer.eos_token is not None:
            target_text = target_text + self.dec_tokenizer.eos_token

        dec = self.dec_tokenizer(
            target_text,
            padding="max_length",
            truncation=True,
            max_length=self.target_max_len,
            return_tensors="pt",
        )

        target_ids = dec["input_ids"].squeeze(0)
        target_attention_mask = dec["attention_mask"].squeeze(0)

        # labels are used for loss.
        # pad tokens must be ignored with -100.
        labels = target_ids.clone()
        labels[target_attention_mask == 0] = self.label_pad_id

        return {
            # Main Hugging Face-style keys
            "input_ids": enc_input_ids,
            "attention_mask": enc_attention_mask,
            "labels": labels,

            # Aliases for your notebook print/debug style
            "enc_input_ids": enc_input_ids,
            "enc_attention_mask": enc_attention_mask,
            "target_ids": target_ids,
            "target_attention_mask": target_attention_mask,

            # Metadata/debugging
            "row_idx": torch.tensor(item["row_idx"], dtype=torch.long),
            "frame_idx": torch.tensor(item["frame_idx"], dtype=torch.long),
            "image_id": item["image_id"],
            "description": description,
        }
# =========================================================


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class RobertaEncoder(nn.Module):
    def __init__(
        self,
        model_name="roberta-base",
        unfreeze_layers=8,
        memory_dropout=0.1,
        train_embeddings=False,
    ):
        super().__init__()

        self.roberta = RobertaModel.from_pretrained(model_name)

        # Freeze all RoBERTa parameters first.
        for param in self.roberta.parameters():
            param.requires_grad = False

        # Optionally train embeddings.
        if train_embeddings:
            for param in self.roberta.embeddings.parameters():
                param.requires_grad = True

        # Unfreeze top N encoder layers.
        if unfreeze_layers > 0:
            for layer in self.roberta.encoder.layer[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.memory_dropout = nn.Dropout(memory_dropout)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        memory = outputs.last_hidden_state
        memory = self.memory_dropout(memory)

        return memory


def generate_square_subsequent_mask(sz, device):
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
    return mask


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=768,
        nhead=8,
        num_layers=6,
        dropout=0.15,
        max_len=2048,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=max_len,
        )

        self.input_norm = nn.LayerNorm(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying.
        self.out.weight = self.embedding.weight

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = self.embedding(tgt) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.input_norm(x)
        x = self.embed_dropout(x)

        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        x = self.final_norm(x)
        logits = self.out(x)

        return logits


# =========================================================
# Seq2Seq
# ---------------------------------------------------------
# RoBERTa encoder + GPT-2 decoder.
#
# Keeps the class name Seq2Seq so your main notebook import
# does not break.
# =========================================================


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder_name="roberta-base",
        decoder_name="gpt2",
        unfreeze_encoder_layers=8,
        enc_tokenizer=None,
        dec_tokenizer=None,
        train_encoder_embeddings=False,
        decoder_start_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        max_generation_length: int = 256,
    ):
        super().__init__()

        self.enc_tokenizer = enc_tokenizer
        self.dec_tokenizer = dec_tokenizer
        self.max_generation_length = max_generation_length

        # -------------------------------------------------
        # GPT-2 has no pad token by default.
        # Usually we set:
        #     dec_tokenizer.pad_token = dec_tokenizer.eos_token
        # before passing it here.
        # -------------------------------------------------
        if self.dec_tokenizer is not None:
            if self.dec_tokenizer.pad_token is None:
                self.dec_tokenizer.pad_token = self.dec_tokenizer.eos_token

        # -------------------------------------------------
        # Build RoBERTa encoder + GPT-2 decoder.
        # decoder_is_decoder=True and decoder_add_cross_attention=True
        # make GPT-2 usable as a conditional decoder.
        # -------------------------------------------------
        decoder_config = AutoConfig.from_pretrained(decoder_name)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True

        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_name,
            decoder_name,
            decoder_config=decoder_config,
        )

        # -------------------------------------------------
        # Resize embeddings if tokenizers were changed outside this class.
        # This is safe even if no new tokens were added.
        # -------------------------------------------------
        if self.enc_tokenizer is not None:
            self.model.encoder.resize_token_embeddings(len(self.enc_tokenizer))

        if self.dec_tokenizer is not None:
            self.model.decoder.resize_token_embeddings(len(self.dec_tokenizer))

        # -------------------------------------------------
        # Resolve decoder token IDs.
        # -------------------------------------------------
        dec_config = self.model.config.decoder

        if self.dec_tokenizer is not None:
            resolved_eos = self.dec_tokenizer.eos_token_id
            resolved_pad = self.dec_tokenizer.pad_token_id
            resolved_bos = self.dec_tokenizer.bos_token_id

            # GPT-2 normally has no real BOS token.
            # EOS is commonly used as decoder start.
            if resolved_bos is None:
                resolved_bos = resolved_eos

        else:
            resolved_eos = dec_config.eos_token_id
            resolved_pad = dec_config.pad_token_id
            resolved_bos = dec_config.bos_token_id

            if resolved_bos is None:
                resolved_bos = resolved_eos

            if resolved_pad is None:
                resolved_pad = resolved_eos

        if decoder_start_token_id is not None:
            resolved_bos = decoder_start_token_id

        if eos_token_id is not None:
            resolved_eos = eos_token_id

        if pad_token_id is not None:
            resolved_pad = pad_token_id

        if resolved_bos is None:
            raise ValueError(
                "decoder_start_token_id could not be resolved. "
                "Pass dec_tokenizer or decoder_start_token_id explicitly."
            )

        if resolved_eos is None:
            raise ValueError(
                "eos_token_id could not be resolved. "
                "Pass dec_tokenizer or eos_token_id explicitly."
            )

        if resolved_pad is None:
            resolved_pad = resolved_eos

        # -------------------------------------------------
        # Main EncoderDecoderModel config.
        # -------------------------------------------------
        self.model.config.decoder_start_token_id = resolved_bos
        self.model.config.eos_token_id = resolved_eos
        self.model.config.pad_token_id = resolved_pad
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

        # Decoder config.
        self.model.config.decoder.decoder_start_token_id = resolved_bos
        self.model.config.decoder.eos_token_id = resolved_eos
        self.model.config.decoder.pad_token_id = resolved_pad
        self.model.config.decoder.is_decoder = True
        self.model.config.decoder.add_cross_attention = True

        # Generation config for newer Transformers versions.
        if hasattr(self.model, "generation_config"):
            self.model.generation_config.decoder_start_token_id = resolved_bos
            self.model.generation_config.eos_token_id = resolved_eos
            self.model.generation_config.pad_token_id = resolved_pad
            self.model.generation_config.max_length = max_generation_length

        # -------------------------------------------------
        # Freeze RoBERTa first, then unfreeze top N layers.
        # GPT-2 decoder remains trainable by default.
        # -------------------------------------------------
        roberta = self.model.encoder

        for param in roberta.parameters():
            param.requires_grad = False

        if train_encoder_embeddings:
            for param in roberta.embeddings.parameters():
                param.requires_grad = True

        if unfreeze_encoder_layers > 0:
            for layer in roberta.encoder.layer[-unfreeze_encoder_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    # -----------------------------------------------------
    # Compatibility properties.
    # -----------------------------------------------------
    @property
    def encoder(self):
        return self.model.encoder

    @property
    def decoder(self):
        return self.model.decoder

    # -----------------------------------------------------
    # Forward pass.
    # -----------------------------------------------------
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        decoder_attention_mask=None,
        **kwargs,
    ):
        """
        Training usage:

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        Important:
        - Prefer passing labels that already have padding replaced by -100.
        - If decoder_attention_mask is passed, this method will also mask
          padding positions in labels.
        """

        if labels is not None:
            labels = labels.clone()

            # This is safer than replacing every pad_token_id with -100,
            # because for GPT-2 we often set pad_token_id == eos_token_id.
            # Masking every pad/eos ID would remove real EOS targets too.
            if decoder_attention_mask is not None:
                labels[decoder_attention_mask == 0] = -100

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            **kwargs,
        )

    # -----------------------------------------------------
    # Encoder helper for later sequence predictor use.
    # -----------------------------------------------------
    def encode(self, input_ids, attention_mask=None):
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        return encoder_outputs.last_hidden_state

    # -----------------------------------------------------
    # Generation helper.
    # -----------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        **kwargs,
    ):
        generation_kwargs = {
            "max_length": self.max_generation_length,
            "num_beams": 3,
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.15,
        }

        generation_kwargs.update(kwargs)

        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )

    # -----------------------------------------------------
    # Save helper.
    # -----------------------------------------------------
    def save_pretrained(self, save_directory: str):
        self.model.save_pretrained(save_directory)

        if self.enc_tokenizer is not None:
            self.enc_tokenizer.save_pretrained(
                f"{save_directory}/encoder_tokenizer"
            )

        if self.dec_tokenizer is not None:
            self.dec_tokenizer.save_pretrained(
                f"{save_directory}/decoder_tokenizer"
            )
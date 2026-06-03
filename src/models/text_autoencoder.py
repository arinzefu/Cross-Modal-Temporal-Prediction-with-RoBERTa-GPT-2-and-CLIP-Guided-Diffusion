# @title Importing the necessary libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from transformers import CLIPProcessor, CLIPModel, RobertaModel, RobertaTokenizer

import math

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaModel, EncoderDecoderModel


# =========================================================
# LEGACY COMPONENTS
# ---------------------------------------------------------
# Kept so existing imports
#   from src.models.text_autoencoder import (
#       RobertaEncoder, TransformerDecoder, Seq2Seq, SinusoidalPositionalEncoding
#   )
# keep working. RobertaEncoder / TransformerDecoder are no longer used by the
# new Seq2Seq (which is a pretrained RoBERTa2GPT2 model), but are left intact in
# case you want the from-scratch path again.
# =========================================================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class RobertaEncoder(nn.Module):
    def __init__(self, model_name="roberta-base", unfreeze_layers=8, memory_dropout=0.1):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        for param in self.roberta.parameters():
            param.requires_grad = False
        if unfreeze_layers > 0:
            for layer in self.roberta.encoder.layer[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        self.memory_dropout = nn.Dropout(memory_dropout)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        memory = outputs.last_hidden_state
        memory = self.memory_dropout(memory)
        return memory


def generate_square_subsequent_mask(sz, device):
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
    return mask


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=8, num_layers=6, dropout=0.15, max_len=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len)
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
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.out.weight = self.embedding.weight

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
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
# NEW Seq2Seq  ── pretrained RoBERTa encoder + pretrained GPT-2 decoder
# ---------------------------------------------------------
# keep working. The cross-attention layers added to GPT-2 are randomly
# initialized and are trained during fine-tuning.
# =========================================================
class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder_name="roberta-base",
        decoder_name="gpt2",
        unfreeze_encoder_layers=8,
        enc_tokenizer=None,
        dec_tokenizer=None,
    ):
        super().__init__()

        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_name, decoder_name
        )

        # ---- required config plumbing (loss + generation break without this) ----
        if dec_tokenizer is not None:
            dec_bos = dec_tokenizer.bos_token_id
            dec_eos = dec_tokenizer.eos_token_id
            dec_pad = dec_tokenizer.pad_token_id if dec_tokenizer.pad_token_id is not None else dec_eos
        else:
            dec_bos = self.model.config.decoder.bos_token_id
            dec_eos = self.model.config.decoder.eos_token_id
            dec_pad = dec_eos

        self.model.config.decoder_start_token_id = dec_bos
        self.model.config.eos_token_id = dec_eos
        self.model.config.pad_token_id = dec_pad
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

        # ---- freeze RoBERTa, unfreeze the top N layers ----
        roberta = self.model.encoder  # a RobertaModel
        for p in roberta.parameters():
            p.requires_grad = False
        if unfreeze_encoder_layers > 0:
            for layer in roberta.encoder.layer[-unfreeze_encoder_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True

    # expose submodules so the optimizer split + freezing logic keep working
    @property
    def encoder(self):
        return self.model.encoder

    @property
    def decoder(self):
        return self.model.decoder

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # HF builds decoder_input_ids from labels and computes the CE loss itself.
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

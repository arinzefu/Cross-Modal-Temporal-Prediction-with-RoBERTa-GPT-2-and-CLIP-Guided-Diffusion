# @title Importing the necessary libraries

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt
import numpy as np
import os
from nltk.translate.bleu_score import sentence_bleu
import json
import pandas as pd
from torchinfo import summary
from transformers import CLIPProcessor, CLIPModel, RobertaModel, RobertaTokenizer
import evaluate
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import tqdm
from datasets.fingerprint import random
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.transforms.functional as FT
import math
from transformers import BertTokenizer
import gc
import random
import re
from typing import Dict, Any, List, Optional, Tuple
import textwrap
# This will prompt you to authorize Google Drive access
from google.colab import drive

def save_checkpoint_to_drive(model, optimizer, epoch, loss, filename="roberta_clip_checkpoint.pth"):
    """
    Saves the checkpoint directly to a specified folder in your mounted Google Drive.
    """
    # 1. Define the full Google Drive path
    # 'DL_Checkpoints' is the folder you want to save to inside your Drive
    drive_folder = '/content/drive/MyDrive/DL_Checkpoints'

    # Ensure the directory exists before attempting to save
    os.makedirs(drive_folder, exist_ok=True)

    # 2. Combine the folder and the filename
    full_path = os.path.join(drive_folder, filename)

    # 3. Create the checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    # 4. Save the dictionary to the Google Drive path
    torch.save(checkpoint, full_path)
    print(f"Checkpoint saved to Google Drive: {full_path} at epoch {epoch}")


def load_checkpoint_from_drive(model, optimizer=None, filename="roberta_clip_checkpoint.pth"):
    """
    Loads a checkpoint from your Google Drive folder into the model and optimizer (if provided).
    """
    # Define the same Google Drive folder path
    drive_folder = '/content/drive/MyDrive/DL_Checkpoints'
    full_path = os.path.join(drive_folder, filename)

    # Check if the checkpoint file exists
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Checkpoint file not found: {full_path}")

    # Load the checkpoint
    checkpoint = torch.load(full_path, map_location=torch.device('cpu'))  # use cuda if available

    # Restore model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Restore optimizer state (if provided)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Extract metadata
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', None)

    print(f"Checkpoint loaded from: {full_path} (epoch {epoch})")

    return model, optimizer, epoch, loss

emb_dim = 128
latent_dim = 128
num_layers = 6
max_seq_len = 120
batch_size = 16
dropout = 0.1


def parse_gdi_text(text):

    soup = BeautifulSoup(text, 'html.parser')
    images = []

    for gdi in soup.find_all('gdi'):



        image_id = None
        if gdi.attrs:

            for attr_name, attr_value in gdi.attrs.items():
                if 'image' in attr_name.lower():
                    image_id = attr_name.replace('image', '')
                    break

        x
        if not image_id:
            tag_str = str(gdi)
            match = re.search(r'<gdi\s+image(\d+)', tag_str)
            if match:
                image_id = match.group(1)


        if not image_id:
            image_id = str(len(images) + 1)

        content = gdi.get_text().strip()

        # Extract tagged elements using BeautifulSoup directly
        objects = [obj.get_text().strip() for obj in gdi.find_all('gdo')]
        actions = [act.get_text().strip() for act in gdi.find_all('gda')]
        locations = [loc.get_text().strip() for loc in gdi.find_all('gdl')]

        images.append({
            'image_id': image_id,
            'description': content,
            'objects': objects,
            'actions': actions,
            'locations': locations,
            'raw_text': str(gdi)
        })

    return images


def show_image(ax, image, de_normalize = False, img_mean = None, img_std = None):

  if de_normalize:
    new_mean = -img_mean/img_std
    new_std = 1/img_std

    image = transforms.Normalize(
        mean=new_mean,
        std=new_std
    )(image)
  ax.imshow(image.permute(1, 2, 0))

def _parse_markdown_table(block: str) -> List[Dict[str, str]]:
    lines = [l.rstrip() for l in block.splitlines()]
    table_lines = [l for l in lines if l.strip().startswith("|")]
    if len(table_lines) < 3:
        return []
    header_line = table_lines[0]
    data_lines = table_lines[2:]
    headers = [h.strip() for h in header_line.strip("|").split("|")]
    rows = []
    for line in data_lines:
        if not line.strip().startswith("|"):
            break
        cols = [c.strip() for c in line.strip("|").split("|")]
        if len(cols) != len(headers):
            continue
        rows.append(dict(zip(headers, cols)))
    return rows

def parse_cot_grounding(chain_of_thought: str) -> Dict[int, Dict[str, Any]]:
    """Parse StoryReasoning-style CoT markdown into per-frame bbox annotations."""
    frames: Dict[int, Dict[str, Any]] = {}
    img_pattern = re.compile(r"^##\s*Image\s+(\d+)", flags=re.MULTILINE)
    matches = list(img_pattern.finditer(chain_of_thought or ""))

    for i, m in enumerate(matches):
        img_idx = int(m.group(1)) - 1
        start = m.end()
        end = matches[i + 1].start() if (i + 1 < len(matches)) else len(chain_of_thought)
        section = (chain_of_thought or "")[start:end]

        frames[img_idx] = {"characters": [], "objects": []}

        char_match = re.search(r"###\s*Characters(.*?)(?=\n###|\n##|$)", section, flags=re.DOTALL)
        if char_match:
            for row in _parse_markdown_table(char_match.group(1)):
                cid = row.get("Character ID", "").strip()
                bbox_str = row.get("Bounding Box", "").strip()
                if cid and bbox_str:
                    try:
                        x1, y1, x2, y2 = [int(v) for v in bbox_str.split(",")]
                        frames[img_idx]["characters"].append({"id": cid, "bbox": [x1, y1, x2, y2]})
                    except Exception:
                        pass

        obj_match = re.search(r"###\s*Objects(.*?)(?=\n###|\n##|$)", section, flags=re.DOTALL)
        if obj_match:
            for row in _parse_markdown_table(obj_match.group(1)):
                oid = row.get("Object ID", "").strip()
                bbox_str = row.get("Bounding Box", "").strip()
                if oid and bbox_str:
                    try:
                        x1, y1, x2, y2 = [int(v) for v in bbox_str.split(",")]
                        frames[img_idx]["objects"].append({"id": oid, "bbox": [x1, y1, x2, y2]})
                    except Exception:
                        pass
    return frames

def _clamp_bbox(x1, y1, x2, y2, W, H):
    x1 = max(0, min(x1, W - 1))
    x2 = max(0, min(x2, W - 1))
    y1 = max(0, min(y1, H - 1))
    y2 = max(0, min(y2, H - 1))
    if x2 <= x1:
        x2 = min(W - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2

def crop_and_resize(pil_img, bbox, out_hw=(60, 125)):
    x1, y1, x2, y2 = bbox
    W, H = pil_img.size
    x1, y1, x2, y2 = _clamp_bbox(x1, y1, x2, y2, W, H)
    crop = pil_img.crop((x1, y1, x2, y2))
    crop = transforms.Resize(out_hw)(crop)
    crop = transforms.ToTensor()(crop)
    return crop

def pick_reid_pair(frames_cot: Dict[int, Dict[str, Any]]) -> Optional[Tuple[int, int, List[int], List[int], str]]:
    """Pick two detections of the same entity id across frames."""
    id_to_dets = {}
    for f_idx, content in frames_cot.items():
        for det in content.get("characters", []) + content.get("objects", []):
            ent_id = det.get("id")
            bbox = det.get("bbox")
            if ent_id and bbox:
                id_to_dets.setdefault(ent_id, []).append((f_idx, bbox))

    candidates = [ent_id for ent_id, dets in id_to_dets.items() if len(dets) >= 2]
    if not candidates:
        return None

    ent_id = random.choice(candidates)
    dets = id_to_dets[ent_id]
    (f1, b1), (f2, b2) = random.sample(dets, 2)
    return f1, f2, b1, b2, ent_id

def extract_cot_text_for_frame(chain_of_thought: str, frame_idx: int, max_chars: int = 600) -> str:
    """Option 4 helper: extract the 'Image N' section as plain text (best-effort)."""
    if not chain_of_thought:
        return ""
    img_pattern = re.compile(r"^##\s*Image\s+(\d+)", flags=re.MULTILINE)
    matches = list(img_pattern.finditer(chain_of_thought))
    target = None
    for i, m in enumerate(matches):
        if int(m.group(1)) - 1 == frame_idx:
            start = m.end()
            end = matches[i + 1].start() if (i + 1 < len(matches)) else len(chain_of_thought)
            target = chain_of_thought[start:end]
            break
    if target is None:
        return ""
    # Remove markdown tables (keep only non-table lines)
    lines = []
    for line in target.splitlines():
        if line.strip().startswith("|"):
            continue
        if set(line.strip()) <= set("-|:"):
            continue
        lines.append(line)
    text = " ".join([l.strip() for l in lines if l.strip()])
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


# @title Main dataset

class SequencePredictionDataset(Dataset):
    def __init__(self, original_dataset, tokenizer, K: int = 4, max_len: int = 120, image_hw=(224, 224)):
        super(SequencePredictionDataset, self).__init__()
        self.dataset = original_dataset
        self.tokenizer = tokenizer
        self.K = K
        self.max_len = max_len
        self.image_hw = image_hw

        self.transform = transforms.Compose([
            transforms.Resize(image_hw),  # Reasonable size based on our previous analysis
            transforms.ToTensor(),  # HxWxC -> CxHxW
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Selects a 5 frame sequence from the dataset. Sets 4 for training and the last one
        as a target.

        Returns:
          frames:        [K, C, H, W]
          descriptions:  [K, T]
          image_target:  [C, H, W]
          target_ids:    [1, T]
          roi1, roi2:    [C, H, W] (cropped from CoT bboxes, if available)
          roi_valid:     0/1
          roi_frame:     frame index for roi1 (0..K-1) if available else -1
          ent_id:        string id for the ROI entity (empty if none)
        """
        frames = self.dataset[idx]["images"]
        image_attributes = parse_gdi_text(self.dataset[idx]["story"])

        # CoT grounding annotations (may be missing / unparseable)
        cot = self.dataset[idx].get("chain_of_thought", "")
        cot_frames = parse_cot_grounding(cot)

        frame_tensors = []
        description_list = []

        for frame_idx in range(self.K):
            image = FT.equalize(frames[frame_idx])
            input_frame = self.transform(image)
            frame_tensors.append(input_frame)

            description = image_attributes[frame_idx]["description"]

            # Option 4: include CoT text snippet for this frame (best-effort)
            if USE_COT_TEXT:
                cot_txt = extract_cot_text_for_frame(cot, frame_idx)
                if cot_txt:
                    description = description + " [COT] " + cot_txt

            input_ids = self.tokenizer(
                description,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_len
            ).input_ids.squeeze(0)

            description_list.append(input_ids)

        image_target = FT.equalize(frames[self.K])
        image_target = self.transform(image_target)

        main_description = image_attributes[0]["description"]
        encoded = self.tokenizer(
            main_description,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )  # [1, T]

        text_dict = {"input_ids": encoded['input_ids'][0],
                     "attention_mask": encoded["attention_mask"][0],
                     "decoder_input_ids": encoded["input_ids"][0, :-1],
                     "target_ids": encoded["input_ids"][0, 1:]
                     }

        # ---- CoT ROI pair (Options 1-3 need these) ----
        roi_valid = torch.tensor(0, dtype=torch.long)
        roi1 = torch.zeros((3, self.image_hw[0], self.image_hw[1]))
        roi2 = torch.zeros((3, self.image_hw[0], self.image_hw[1]))
        roi_frame = torch.tensor(-1, dtype=torch.long)
        ent_id = ""

        pair = pick_reid_pair(cot_frames)
        if pair is not None:
            f1, f2, b1, b2, ent_id = pair
            # We only use ROIs that fall within the input window (0..K-1)
            if (0 <= f1 < self.K) and (0 <= f2 < self.K):
                try:
                    roi1 = crop_and_resize(frames[f1], b1, out_hw=self.image_hw)
                    roi2 = crop_and_resize(frames[f2], b2, out_hw=self.image_hw)
                    roi_valid = torch.tensor(1, dtype=torch.long)
                    roi_frame = torch.tensor(int(f1), dtype=torch.long)
                except Exception:
                    pass

        sequence_tensor = torch.stack(frame_tensors)  # [K, C, H, W]
        obj_labels = tuple(["bg"] * self.K + ["bg"])

        ent_id_tensor = torch.tensor([0], dtype=torch.long)  # placeholder

        return (
            sequence_tensor,  # 0: frames,
            image_target,
            roi1, roi2, roi_valid, roi_frame,
            ent_id_tensor,  # 6: ent_id as tensor
            text_dict,  # 7: Text Dict
            obj_labels  # 8: tuple
        )

class TextTaskDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_len=max_seq_len):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image_attributes = parse_gdi_text(sample["story"])
        frame_idx = np.random.randint(0, len(image_attributes))
        description = image_attributes[frame_idx]["description"]

        encoded = self.tokenizer(
            description,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        decoder_input_ids = input_ids[:-1]
        target_ids = input_ids[1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "target_ids": target_ids
        }






class AutoEncoderTaskDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([
          transforms.Resize((224, 224)),# Reasonable size based on our previous analysis
          transforms.ToTensor(), # HxWxC -> CxHxW
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
      num_frames = self.dataset[idx]["frame_count"]
      frames = self.dataset[idx]["images"]

      # Pick a frame at random
      frame_idx = torch.randint(0, 5, (1,)).item()
      input_frame = self.transform(frames[frame_idx]) # Input to the autoencoder

      return input_frame, # Returning the image

import random

class TextDataset(Dataset):
    """
    Story-continuation pairs for the RoBERTa2GPT2 Seq2Seq.

        encoder side (input_ids / attention_mask) = a PREFIX of the story  (RoBERTa tokenizer)
        decoder side (labels)                     = the CONTINUATION       (GPT-2 tokenizer)

    The split point is chosen RANDOMLY on every fetch, so across epochs the model
    learns to continue from many different amounts of context (10% -> 90%,
    75% -> 25%, ...), which is the "give X%, predict the rest" goal.

    HuggingFace's EncoderDecoderModel builds decoder_input_ids from `labels`
    internally, so we only emit `labels` (continuation tokens + EOS), with padded
    positions set to -100 so they are ignored by the loss.
    """

    def __init__(
        self,
        data,
        enc_tokenizer,                # RoBERTa tokenizer (source / prefix)
        dec_tokenizer,                # GPT-2 tokenizer   (target / continuation)
        max_len=512,                  # prefix cap  -- keep <= 512 (RoBERTa limit)
        max_target_len=256,           # continuation cap -- keep <= 1024 (GPT-2 limit)
        text_field="story",
        min_prefix_frac=0.10,
        max_prefix_frac=0.75,
        label_pad_id=-100,
    ):
        self.data = data
        self.enc_tok = enc_tokenizer
        self.dec_tok = dec_tokenizer
        self.max_len = min(max_len, 512)
        self.max_target_len = min(max_target_len, 1024)
        self.text_field = text_field
        self.min_prefix_frac = min_prefix_frac
        self.max_prefix_frac = max_prefix_frac
        self.label_pad_id = label_pad_id
        self.dec_eos = dec_tokenizer.eos_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx][self.text_field]
        words = text.split()
        if len(words) < 2:
            words = words + ["."]                     # guard tiny rows

        lo = max(1, int(self.min_prefix_frac * len(words)))
        hi = max(lo + 1, int(self.max_prefix_frac * len(words)))
        cut = random.randint(lo, hi)

        prefix = " ".join(words[:cut])
        continuation = " ".join(words[cut:])

        # ---- encoder input: PREFIX (RoBERTa, padded, with attention mask) ----
        enc = self.enc_tok(
            prefix,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # ---- decoder labels: CONTINUATION (GPT-2) + EOS, padded with -100 ----
        cont = self.dec_tok(continuation, add_special_tokens=False)["input_ids"]
        cont = cont[: self.max_target_len - 1]        # leave room for EOS
        labels = cont + [self.dec_eos]
        labels = labels[: self.max_target_len]
        labels = labels + [self.label_pad_id] * (self.max_target_len - len(labels))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long),
        }
import base64
import inspect
import io
import os
import threading

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms


# =========================================================
# Deployment config
# =========================================================


PREDICTOR_CKPT = os.environ.get("PREDICTOR_CKPT", "sequence_predictor_best.pth")
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "arinzefu/mosaic_predictor").strip()
HF_MODEL_REVISION = os.environ.get("HF_MODEL_REVISION", "").strip() or None

IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "224"))
MAX_TEXT_LEN = int(os.environ.get("MAX_TEXT_LEN", "256"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "80"))

# These match 03_multimodal_temporal_predictor.ipynb.
HIDDEN_DIM = 512
VISUAL_LATENT = 128
VISUAL_SPATIAL = 256
VISUAL_UNFREEZE_LAYERS = 2
TEXT_UNFREEZE_ENCODER_LAYERS = 6
TEXT_ENCODER_NAME = "roberta-base"
TEXT_DECODER_NAME = "gpt2"

# Keep strict loading on by default. If this fails, app.py does not match the
# checkpoint and should be fixed instead of silently ignoring missing weights.
STRICT_LOAD = os.environ.get("STRICT_LOAD", "1").lower() not in {"0", "false", "no"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_BUNDLE = None
MODEL_LOCK = threading.Lock()
INFERENCE_LOCK = threading.Lock()
SERVE_API = os.environ.get("SERVE_API", "0").lower() in {"1", "true", "yes"}


# =========================================================
# Model loading
# =========================================================

def _resolve_ckpt_path():
    if HF_MODEL_REPO:
        from huggingface_hub import hf_hub_download

        kwargs = {
            "repo_id": HF_MODEL_REPO,
            "filename": PREDICTOR_CKPT,
        }
        if HF_MODEL_REVISION:
            kwargs["revision"] = HF_MODEL_REVISION

        # Needed only for private model repos. Add HF_TOKEN as a Space secret.
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            kwargs["token"] = token

        return hf_hub_download(**kwargs)

    if not os.path.exists(PREDICTOR_CKPT):
        raise FileNotFoundError(
            f"Checkpoint not found: {PREDICTOR_CKPT!r}. "
            "Upload the checkpoint beside app.py or set HF_MODEL_REPO to the "
            "Hub model repo containing it."
        )

    return PREDICTOR_CKPT


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch versions do not have weights_only.
        return torch.load(path, map_location="cpu")


def build_model():
    from transformers import AutoTokenizer

    from src.models.multimodal_predictor import SequencePredictor
    from src.models.text_autoencoder import Seq2Seq
    from src.models.visual_autoencoder import VisualAutoencoder

    enc_tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_NAME)
    dec_tokenizer = AutoTokenizer.from_pretrained(TEXT_DECODER_NAME)
    if dec_tokenizer.pad_token is None:
        dec_tokenizer.pad_token = dec_tokenizer.eos_token

    visual_autoencoder = VisualAutoencoder(
        latent_dim=VISUAL_LATENT,
        spatial_dim=VISUAL_SPATIAL,
        unfreeze_layers=VISUAL_UNFREEZE_LAYERS,
    )

    text_autoencoder = Seq2Seq(
        encoder_name=TEXT_ENCODER_NAME,
        decoder_name=TEXT_DECODER_NAME,
        unfreeze_encoder_layers=TEXT_UNFREEZE_ENCODER_LAYERS,
        enc_tokenizer=enc_tokenizer,
        dec_tokenizer=dec_tokenizer,
    )

    predictor = SequencePredictor(
        visual_autoencoder,
        text_autoencoder,
        hidden_dim=HIDDEN_DIM,
        train_text_decoder=False,
    )

    ckpt_path = _resolve_ckpt_path()
    ckpt = _load_checkpoint(ckpt_path)
    state = ckpt.get("model_state_dict", ckpt)

    missing, unexpected = predictor.load_state_dict(state, strict=STRICT_LOAD)
    if missing:
        print(f"[load] missing keys ({len(missing)}): {missing[:8]}")
    if unexpected:
        print(f"[load] unexpected keys ({len(unexpected)}): {unexpected[:8]}")

    predictor.to(DEVICE).eval()
    return predictor, enc_tokenizer, dec_tokenizer


def get_model_bundle():
    global MODEL_BUNDLE

    if MODEL_BUNDLE is None:
        with MODEL_LOCK:
            if MODEL_BUNDLE is None:
                print(f"Loading model on {DEVICE}...")
                MODEL_BUNDLE = build_model()
                print("Model ready.")

    return MODEL_BUNDLE


# =========================================================
# Pre/post-processing
# =========================================================

_img_tf = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ]
)


def _preprocess_images(pil_list):
    tensors = [_img_tf(img.convert("RGB")) for img in pil_list]
    return torch.stack(tensors).unsqueeze(0)


def _preprocess_texts(stories, enc_tokenizer):
    enc = enc_tokenizer(
        list(stories),
        padding="max_length",
        truncation=True,
        max_length=MAX_TEXT_LEN,
        return_tensors="pt",
    )
    ids = enc["input_ids"].unsqueeze(0)
    mask = enc["attention_mask"].unsqueeze(0)
    return ids, mask


def _tensor_to_pil(t):
    arr = (t.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8")
    return Image.fromarray(arr)


def _weights_figure(weights):
    fig, ax = plt.subplots(figsize=(4, 3))
    labels = [f"f{i + 1}" for i in range(len(weights))]
    ax.bar(labels, weights, color="#4C72B0")
    ax.set_ylim(0, max(0.5, max(weights) * 1.25))
    ax.set_title("Frame blend weights")
    for i, weight in enumerate(weights):
        ax.text(i, weight, f"{weight:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig


# =========================================================
# Inference
# =========================================================

@torch.inference_mode()
def predict_core(pil_frames, stories):
    model, enc_tokenizer, dec_tokenizer = get_model_bundle()

    frames = _preprocess_images(pil_frames).to(DEVICE)
    ids, mask = _preprocess_texts(stories, enc_tokenizer)
    ids, mask = ids.to(DEVICE), mask.to(DEVICE)

    with INFERENCE_LOCK:
        out = model(
            image_seq=frames,
            input_ids_text_encoder=ids,
            attention_mask_text_encoder=mask,
            decode_image=True,
        )

        gen_ids = model.generate_text_ids(
            image_seq=frames,
            input_ids_text_encoder=ids,
            attention_mask_text_encoder=mask,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    pred_img = _tensor_to_pil(out["pred_image"][0])
    weights = out["frame_weights"][0].detach().cpu().tolist()
    story = dec_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

    return pred_img, story, weights


def run_prediction(img1, img2, img3, img4, s1, s2, s3, s4):
    frames = [img1, img2, img3, img4]
    if any(frame is None for frame in frames):
        return None, "Please provide all four input frames.", None

    stories = [s or "" for s in (s1, s2, s3, s4)]

    try:
        pred_img, story, weights = predict_core(frames, stories)
        return pred_img, story, _weights_figure(weights)
    except Exception as exc:
        print(f"[predict] {type(exc).__name__}: {exc}")
        return None, f"Prediction failed: {exc}", None


# =========================================================
# Gradio UI
# =========================================================

DESCRIPTION = """
# MOSAIC - Multimodal Next-Frame Predictor
Give the model four context frames and their four short stories.
It predicts the next frame and generates the next story.
"""

EXAMPLES = [
    # ["examples/s1_f1.png", "examples/s1_f2.png", "examples/s1_f3.png", "examples/s1_f4.png",
    #  "Story 1", "Story 2", "Story 3", "Story 4"],
]

with gr.Blocks(title="MOSAIC Next-Frame Predictor") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        img_inputs = [gr.Image(type="pil", label=f"Frame {i + 1}") for i in range(4)]

    with gr.Row():
        txt_inputs = [gr.Textbox(label=f"Story {i + 1}", lines=2) for i in range(4)]

    run_btn = gr.Button("Predict next frame + story", variant="primary")

    with gr.Row():
        out_image = gr.Image(label="Predicted frame 5")
        out_plot = gr.Plot(label="Per-frame contribution")

    out_story = gr.Textbox(label="Predicted story 5", lines=4)

    run_btn.click(
        run_prediction,
        inputs=img_inputs + txt_inputs,
        outputs=[out_image, out_story, out_plot],
    )

    if EXAMPLES:
        gr.Examples(examples=EXAMPLES, inputs=img_inputs + txt_inputs)

demo.queue(max_size=8, default_concurrency_limit=1)


# =========================================================
# Optional FastAPI app
# =========================================================

api_app = FastAPI(title="MOSAIC API")


@api_app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": MODEL_BUNDLE is not None,
        "checkpoint": PREDICTOR_CKPT,
        "hf_model_repo": HF_MODEL_REPO or None,
    }


class PredictRequest(BaseModel):
    frames: list[str]
    stories: list[str]


def _b64_to_pil(b64):
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@api_app.post("/api/predict")
def api_predict(req: PredictRequest):
    if len(req.frames) != 4 or len(req.stories) != 4:
        raise HTTPException(
            status_code=400,
            detail="Provide exactly 4 frames and 4 stories.",
        )

    try:
        pil_frames = [_b64_to_pil(frame) for frame in req.frames]
        pred_img, story, weights = predict_core(pil_frames, req.stories)
    except Exception as exc:
        print(f"[api] {type(exc).__name__}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "predicted_image": _pil_to_b64(pred_img),
        "predicted_story": story,
        "frame_weights": weights,
    }


if SERVE_API:

    app = gr.mount_gradio_app(api_app, demo, path="/", ssr_mode=False)
else:
    # Gradio Space mode: do not mount the demo into FastAPI. Gradio 6 Spaces may
    # start a Node SSR proxy during launch(), and mounting first can start a
    # second proxy on the same public port.
    app = demo


if __name__ == "__main__":
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860")))
    host = os.environ.get("HOST", "0.0.0.0")
    running_on_spaces = bool(os.environ.get("SPACE_ID"))
    share = os.environ.get("GRADIO_SHARE", os.environ.get("SHARE", "0")).lower() in {
        "1",
        "true",
        "yes",
    }

    # Default path for a normal Gradio SDK Space. If you run this file directly
    # and also want /api/predict, set SERVE_API=1 or use the Dockerfile/uvicorn.
    if SERVE_API:
        import uvicorn

        uvicorn.run(app, host=host, port=port)
    else:
        launch_kwargs = {
            "server_name": host,
            "share": share,
        }
        if not running_on_spaces:
            launch_kwargs["server_port"] = port
        if "ssr_mode" in inspect.signature(demo.launch).parameters:
            launch_kwargs["ssr_mode"] = False
        demo.launch(**launch_kwargs)

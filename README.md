# Cross-Modal Temporal Prediction with RoBERTa–GPT-2 and CLIP-Guided Diffusion

This work is still in progress but it is built on my school project work from this repo - https://github.com/arinzeuwazurike/dnnls_final_project.git

The diffusion was unable to generate from pure noise so this project became - Cross-Modal Temporal Story-Frame Prediction with RoBERTa-GPT-2 and CLIP Autoencoders.

I used the visual autoencoder for a latent prediction task instead

Currently still pretraining the visual and text autoencoder but this is the link to the pretrained so far.
- https://drive.google.com/drive/folders/1bKR48-lXS1MFUVnGXH8SPxhxtDFmo7CB?usp=sharing
Here is the structure of the notebook

# Repository Structure

```text
Cross-Modal Temporal Prediction with RoBERTa–GPT-2 and CLIP-Guided Diffusion
│
├── checkpoints/                     # Checkpoint download links
│   └── checkpoint_link.txt
│
├── Deploy/                          # Deployment-related files
│
├── Result/                          # Generated results and visualizations
│
├── src/
│   ├── inference/                   # Inference pipeline
│   │
│   ├── models/                      # Model architectures
│   │   ├── multimodal_predictor.py
│   │   ├── text_autoencoder.py
│   │   └── visual_autoencoder.py
│   │
│   ├── training/                    # Training scripts
│   │   ├── train_text.py
│   │   ├── train_visual.py
│   │   └── train_predictor.py
│   │
│   ├── utils/                       # Utility functions
│   │   └── helper.py
│   │
│   └── test/                        # Model evaluation scripts
│       ├── text_test.py
│       ├── test_predictor.py
│       └── visual_test.py
│
├── 01_text_autoencoder_pretraining.ipynb
├── 02_visual_autoencoder_pretraining.ipynb
├── 03_multimodal_temporal_predictor.ipynb
├── README.md
├── training_diffusion_log.txt
└── training_text_log.txt
```



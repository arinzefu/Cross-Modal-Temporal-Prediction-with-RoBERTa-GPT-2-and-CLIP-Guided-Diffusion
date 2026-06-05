# Cross-Modal Temporal Prediction with RoBERTa–GPT-2 and CLIP-Guided Diffusion

This work is still in progress but it is built on my school project work from this repo - https://github.com/arinzeuwazurike/dnnls_final_project.git

In my previous work I showed the progress in how changes in the text and visual autoencoder improved the sequence prediction but here I am implementing my suggessions by pairing the ROberta encoder with gpt2 transformer and turning the visual autoencoder to a complete CLIP denoising diffusion style autoencoder

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
│   │   ├── attention.py
│   │   ├── multimodal_predictor.py
│   │   ├── text_autoencoder.py
│   │   └── visual_autoencoder.py
│   │
│   ├── training/                    # Training scripts
│   │   ├── train_text.py
│   │   ├── train_visual.py
│   │   ├── train_predictor_logging.py 
│   │   └── train_predictor.py
│   │
│   ├── utils/                       # Utility functions
│   │   └── helper.py
│   │
│   └── test/                        # Model evaluation scripts
│       ├── text_test.py
│       ├── test_predictorpy
│       └── visual_test.py
│
├── 01_text_autoencoder_pretraining.ipynb
├── 02_visual_autoencoder_pretraining.ipynb
├── 03_multimodal_temporal_predictor.ipynb
├── README.md
├── training_diffusion_log.txt
└── training_text_log.txt
```



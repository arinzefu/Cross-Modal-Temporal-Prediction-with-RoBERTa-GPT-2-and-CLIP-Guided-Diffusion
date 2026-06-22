# Cross-Modal Temporal Prediction with RoBERTa–GPT-2 and CLIP-Autoencoder

This work is still in progress but it is built on my school project work from this repo - https://github.com/arinzeuwazurike/dnnls_final_project.git

The diffusion was unable to generate from pure noise so this project became - Cross-Modal Temporal Story-Frame Prediction with RoBERTa-GPT-2 and CLIP Autoencoders.

I used the visual autoencoder for a latent prediction task instead

The attention can be found in [src/models/sequence_predictor.py]

The main notebook is [03_multimodal_temporal_predictor.py]

Overall the model displayed positive results but was limited to the dataset provide, it was able to retrain the structural meaning of the sentences for the story dataset but the latent prediction for the visual task was not entirely solved.

The model was deployed on hugging face and I created a space for it - https://huggingface.co/spaces/arinzefu/MOSAIC-story-predictor


Here is the structure of the notebook

# Repository Structure

```text
Cross-Modal Temporal Prediction with RoBERTa–GPT-2 and CLIP-Guided Diffusion
│
├── checkpoints/                     # Checkpoint download links
│   └── checkpoint_link.txt
│
├── validation_results/ 
│
├── training_log/ 
│
├── Deploy/                            # Deployment-related files
│   │
│   ├── app.py       
│   ├── dockerfile     
│   └── requirement.txt                          
│
├── Result/                          # Generated results and visualizations
│
├── src/ 
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
│   └── utils/                       # Utility functions
│       └── helper.py
│   
├── test/                        # Model evaluation scripts
│   ├── text_test.py
│   ├── test_predictor.py
│   └── visual_test.py
│
├── 01_text_autoencoder_pretraining.ipynb
├── 02_visual_autoencoder_pretraining.ipynb
├── 03_multimodal_temporal_predictor.ipynb
├── README.md
└── clean_notebook.ipynb
```



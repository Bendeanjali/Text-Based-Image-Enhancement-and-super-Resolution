## Project Title

**Text Based Image Enhancement and Super Resolution using Diffusion Models**

## Description

This project explores the application of the DeepFloyd IF framework to upscale low-resolution images (64x64) to high-resolution outputs (256x256) using text-guided diffusion models. The method combines the T5-XXL text encoder with a UNet architecture, integrated through cross-attention mechanisms, to ensure semantic alignment between the image and the input text.

## Features

- Cascaded diffusion stages for progressive image enhancement
- T5-XXL encoder for rich, context-aware text embeddings
- Cross-attention in UNet for semantic consistency
- Supports positive and negative text prompts for controlled output
- Resource-efficient with memory-efficient attention and model offloading
- Compatible with consumer-grade GPUs

## Technologies Used

- Python
- PyTorch
- Hugging Face Diffusers
- CUDA (GPU Acceleration)
- DeepFloyd IF (IFPipeline & IFPSuperResolutionPipeline)

## Dataset

**CelebA-HQ Dataset**

- 30,000 high-quality celebrity face images
- Size: 256x256 pixels
- Used for training and evaluating super-resolution results
- [Kaggle Dataset Link](https://www.kaggle.com/datasets/badasstechie/celebahq-resized-256x256)

## Project Structure

```
text-to-image-superres/
├── models/
│   └── deepfloyd_if_pipeline.py
├── data/
│   └── celebA-HQ/
├── outputs/
│   └── generated_images/
├── notebooks/
│   └── analysis.ipynb
├── utils/
│   └── helpers.py
├── main.py
└── README.md
```

## How to Run

1. Clone the repository

```bash
git clone https://github.com/yourusername/text-to-image-superres.git
cd text-to-image-superres
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Download and prepare the dataset

```bash
# Place CelebA-HQ 256x256 images into data/celebA-HQ/
```

4. Run the inference pipeline

```bash
python main.py --prompt "A smiling girl wearing a red hat" --negative_prompt "blurry, distorted"
```

## Results

- **PSNR Improvement**: Demonstrated improvement in Peak Signal-to-Noise Ratio at each stage
- **FID Score**: Improved realism across 64x64 to 256x256 upscaling
- **Visual Results**: Enhanced facial details with preserved semantic features

## Future Work

- Extend pipeline to upscale from 256x256 to 1024x1024
- Introduce real-world image restoration benchmarks
- Add support for domain-specific applications (e.g., medical imaging, historical restoration)

## Acknowledgements

- DeepFloyd IF by Stability AI
- Hugging Face Transformers and Diffusers
- CelebA-HQ dataset providers

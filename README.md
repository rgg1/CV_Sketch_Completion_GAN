# Sketch Completion with Generative Adversarial Networks

6.S058 Computer Vision Final Project implementing and comparing different GAN architectures for automatic sketch completion and inpainting.

## Overview

This project explores sketch completion using various GAN techniques, comparing LSGAN, WGAN-GP, and WGAN-GP with attention mechanisms. The models learn to intelligently fill in missing portions of sketches while maintaining consistency and artistic style.

## Features

- **Multiple GAN Architectures**: LSGAN, WGAN-GP, and WGAN-GP with attention
- **U-Net Generator**: 4-stage encoder-decoder with skip connections (17M parameters)
- **Attention**: Bottleneck attention mechanism for focused completion
- **Multi-Dataset Evaluation**: QuickDraw, ImageNet-Sketch, and custom data for comprehensive testing
- **Multiple Loss Functions**: Adversarial, masked L1, and VGG perceptual losses

## Architecture

### Generator (U-Net + Attention)
- 4-stage encoder with progressive downsampling
- Spatial attention module at bottleneck
- 4-stage decoder with skip connections
- normalization and LeakyReLU activation

### Discriminator
- Convolutional layers with stride-2 downsampling
- Outputs patch-wise authenticity scores
- Gradient penalty for better WGAN-GP training stability

## Dataset

- **QuickDraw**: 8 categories (cat, dog, house, tree, bicycle, car, face, flower)
- **ImageNet-Sketch**: For generalization testing, contains more complex sketches
- **Custom**: Sketches we made ourselves to further test generalization
- **Line-aware masking**: Masks placed on actual sketch lines to get better learning during training
- **Mask size evaluation**: Mask sizes from 1/8 to 1/2 of image area were tested

## Results

### Quantitative Metrics (WGAN-GP + Attention)
- **QuickDraw Seen**: L1 = 0.064, LPIPS = 0.025
- **QuickDraw Unseen**: L1 = 0.064, LPIPS = 0.025  
- **ImageNet-Sketch**: L1 = 0.196, LPIPS = 0.261

### Findings
- WGAN-GP with attention outperforms LSGAN variants
- Consistent performance on seen vs unseen QuickDraw samples
- Successful generalization to ImageNet-Sketch dataset
- Attention mechanism improves completion quality

## File Structure

```
├── CV_GAN_Attention.ipynb           # Main training notebook
├── CV_metrics_postprocessing.ipynb  # Evaluation and analysis
├── images_to_use/                   # Sample results and figures
│   ├── QD_Seen_Samples/             # QuickDraw seen category results
│   ├── QD_Unseen_Samples/           # QuickDraw unseen category results
│   ├── ImageNet_Samples/            # ImageNet-Sketch results
│   ├── custom_sketch_completions/   # Custom sketch results
│   ├── metric_plots/                # Performance analysis plots
│   └── architecture_diagrams/       # Model architecture visualizations
└── latest_checkpoints_completions/  # Recent model outputs
```

### Training
1. In `CV_GAN_Attention.ipynb`
2. Configure hyperparameters in the configuration section
3. Run all cells to train the model
4. Checkpoints are saved every 5 epochs

### Evaluation
1. In `CV_metrics_postprocessing.ipynb`
2. Specify checkpoint path for evaluation
3. Run evaluation including:
   - Loss curve visualization
   - Quantitative metrics (L1, LPIPS)
   - Visual completion examples
   - Generalization tests

### Custom Sketches
1. Put custom sketches in `custom_sketches/` directory
2. Run the custom evaluation section in the metrics notebook
3. View completion results with mask highlighting

#### Note about requirements.txt

The actual training and evaluation of our models was done on Google Colab to take advantage of their GPUs. Thus, the virtual environment used in this repo did not actually install all of the libraries that we import at the top of the notebooks as we never did the full runs locally. So, pay attention to those import lines if you do actually want to run any of the files as the `requirements.txt` file does not have all the necessary imports.

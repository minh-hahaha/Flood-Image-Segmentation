# FearlessFlooders: FloodNet Segmentation Project

## Overview

FearlessFlooders is a deep learning project for flood segmentation using the FloodNet dataset. This project implements a semantic segmentation model to identify and classify different types of flood damage in aerial imagery, which is crucial for disaster response and damage assessment.


## Dataset

The project uses the **FloodNet-Supervised v1.0** dataset, which contains:

- **Training set**: 1,447 images with corresponding segmentation masks
- **Validation set**: 452 images with masks  
- **Test set**: 450 images with masks
- **10 semantic classes** including:
  - Background
  - Building (undamaged)
  - Building (flooded)
  - Road (undamaged)
  - Road (flooded)
  - Water
  - Tree
  - Vehicle
  - Pool
  - Grass

## Model Architecture

The project uses **DeepLabV3 with ResNet50 backbone** for semantic segmentation:

- **Backbone**: ResNet50 (pretrained on ImageNet)
- **Architecture**: DeepLabV3 with Atrous Spatial Pyramid Pooling (ASPP)
- **Input size**: 512x512 pixels
- **Output**: 10-class segmentation masks

## Key Features

### Data Processing
- Custom `FloodNetDataset` class for efficient data loading
- Data augmentation support for training
- RGB to class index conversion using color palette
- Weighted sampling for class imbalance handling

### Training Features
- **Loss function**: Combined CrossEntropy + Dice loss
- **Optimizer**: Adam with learning rate 1e-4
- **Batch size**: 4 (configurable)
- **Epochs**: 15 (configurable)
- **Class weighting**: Automatic handling of class imbalance

### Evaluation Metrics
- **mIoU (mean Intersection over Union)**: Primary metric
- **Per-class IoU**: Individual class performance
- **Accuracy**: Overall pixel accuracy
- **Confusion matrix**: Detailed class-wise analysis

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM


### Running the Notebook

1. **Start Jupyter Lab**:
   ```bash
   jupyter lab
   ```

2. **Open the notebook**: `FearlessFlooders_Final.ipynb`

3. **Configure paths**: Update the `CFG` class with your dataset paths:
   ```python
   @dataclass
   class CFG:
       ROOT_DIR: str = "/path/to/FloodNet-Supervised_v1.0"
       PALETTE_XLSX: str = "/path/to/ColorPalette-Values.xlsx"
   ```

4. **Run all cells**: Execute the notebook from top to bottom

### Configuration Options

The notebook provides several configuration options:

```python
@dataclass
class CFG:
    ROOT_DIR: str = "/path/to/dataset"
    NUM_CLASSES: int = 10
    INPUT_SIZE: int = 512
    BATCH_SIZE: int = 4
    EPOCHS: int = 25
    LR: float = 1e-4
    NUM_WORKERS: int = 0  # Set to 0 for Jupyter compatibility
    USE_PRETRAINED: bool = True
```

## Output Files

The training process generates several output files:

- **Model checkpoints**: Saved in `outputs_floodnet_supervised/`
- **Training logs**: Loss and metric curves
- **Validation results**: Detailed performance metrics
- **Sample predictions**: Visualized segmentation results


## Dependencies

```
torch>=1.9.0
torchvision>=0.10.0
torchaudio>=0.9.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
matplotlib>=3.3.0
pandas>=1.3.0
numpy>=1.21.0
torchmetrics>=0.6.0
```

## Citation

If you use this project in your research, please cite:

```bibtex
@article{floodnet2021,
  title={FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding},
  author={Rahnemoonfar, Maryam and others},
  journal={IEEE Access},
  year={2021}
}
```

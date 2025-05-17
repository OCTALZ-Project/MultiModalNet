# MultiModelNet
Multi model classification for OCT-A

This repository contains a multi-model architecture for classifying retinal diseases using both OCT/A projection maps and OCT/A b-scans. For OCT/A projection maps, ResNet50, ResNet101, and modified AlexNet (similar to [LPIPS](https://github.com/richzhang/PerceptualSimilarity)) architectures are used, while for OCT/A b-scans, Vision Transformer architectures (ViT-B, ViT-L, ViT-H) are implemented.

## Overview

The model is designed to classify retinal images into three categories:
- NORMAL
- AMD (Age-related Macular Degeneration)
- DR (Diabetic Retinopathy)

The approach uses multiple retinal scans per patient:
- OCT/A projection maps: FULL, ILM_OPL, OPL_BM (processed with ResNet50, ResNet101, and modified AlexNet)
- OCT/A b-scans at different positions: 185, 200, 215 (processed with Vision Transformer models)

### Dataset

This work uses the [OCTA-500 dataset](https://ieee-dataport.org/open-access/octa-500), which originally contains multiple classes including NORMAL, AMD, DR, CNV, CSC, RVO, and OTHERS. Due to class imbalance issues in the dataset, we focused only on three main classes: NORMAL, AMD, and DR for our experiments.

### Model Architecture
- **Projection Maps Processing**: ResNet50, ResNet101, and AlexNet (modified like LPIPS)
- **B-scans Processing**: Vision Transformer (ViT-B, ViT-L, ViT-H)
- **Combined Architecture**: Features from both processing streams are fused for final classification

### Validation Strategy
We implemented k-fold cross-validation (k=5) to ensure robust evaluation of the model performance. This approach helps mitigate potential biases from a single train/validation split, especially important when working with medical imaging datasets.

## Installation

```bash
# Clone the repository
git clone https://github.com/OCTALZ-Project/MultiModelNet.git
cd MultiModelNet

# Install requirements (versions are not mandatory, just my instant versions)
pip install -r requirements.txt
```

## Dataset Structure

The expected dataset structure is:

```
DATASET/
├── train/
│   ├── NORMAL/
│   │   ├── patient_id/
│   │   │   ├── OCTA(FULL)_*.bmp
│   │   │   ├── OCTA(ILM_OPL)_*.bmp
│   │   │   ├── OCTA(OPL_BM)_*.bmp
│   │   │   ├── 200.bmp (B-scan)
│   │   │   ├── 185.bmp (B-scan)
│   │   │   └── 215.bmp (B-scan)
│   ├── AMD/
│   └── DR/
├── val/
│   ├── NORMAL/
│   ├── AMD/
│   └── DR/
└── test/
    ├── NORMAL/
    ├── AMD/
    └── DR/
```

For k-fold cross-validation, the data should be structured as:

```
kfold_dataset/
├── fold_0/
│   ├── train/
│   ├── val/
│   └── test/
├── fold_1/
...
└── fold_4/
```

## Fine-tuning

To fine-tune the model, use the provided script:

```bash
python main_finetune.py \
    --batch_size 16 \
    --epochs 50 \
    --blr 5e-5 \
    --layer_decay 0.75 \
    --weight_decay 0.05 \
    --clip_grad 1.0 \ 
    --data_path /path/to/your/data \
    --nb_classes 3 \
    --vit_finetune finetune_model.pth \
    --vit_global_pool avg \
    --output_dir ./output/model \
    --log_dir ./output/logs \
    --num_workers 4 \
    --pin_mem
```

### K-fold Cross-validation

To perform k-fold cross-validation, use the run_kfold.py script:

```bash
python run_kfold.py
```

This script will automatically:
1. Train and evaluate the model on each fold
2. Save model checkpoints for each fold
3. Generate consolidated reports with overall performance metrics
4. Calculate mean accuracy and standard deviation across all folds

### Important Parameters:

- `--data_path`: Path to your dataset
- `--vit_finetune`: Path to pre-trained model
- `--nb_classes`: Number of classes (3 for NORMAL, AMD, DR)
- `--output_dir`: Directory to save model checkpoints
- `--log_dir`: Directory to save training logs


## Citation

```
@inproceedings{aydin2024retinal,
  title={Retinal Disease Classification Using Optical Coherence Tomography Angiography Images},
  author={Aydın, O. F. and Nazlı, M. S. and Tek, F. B. and Turkan, Y.},
  booktitle={2024 9th International Conference on Computer Science and Engineering (UBMK)},
  pages={884--889},
  year={2024},
  organization={IEEE},
  address={Antalya, Turkiye},
  doi={10.1109/UBMK63289.2024.10773610},
  keywords={Visualization; Retinopathy; Optical coherence tomography; Angiography; Transfer learning; Retina; Transformers; Monitoring; Diseases; Residual neural networks; OCTA; ResNet50; Deep Learning; Retinal Diseases; Image Classification; Optical Coherence Tomography Angiography; Class Imbalance; k-Fold Cross-Validation}
}
```

For the Masked Autoencoder (MAE) pre-training method:

```
@article{he2022masked,
  title={Masked Autoencoders Are Scalable Vision Learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll\'{a}r, Piotr and Girshick, Ross},
  journal={CVPR 2022},
  year={2022}
}
```

## Acknowledgements

This work builds upon several open-source projects:

1. [Masked Autoencoders (MAE)](https://github.com/facebookresearch/mae) - We use the MAE framework for pre-training our Vision Transformer models.
2. [timm](https://github.com/rwightman/pytorch-image-models) - Used for efficient implementations of various computer vision models.

We thank the authors of the [OCTA-500 dataset](https://ieee-dataport.org/open-access/octa-500) for making their data available for research purposes.
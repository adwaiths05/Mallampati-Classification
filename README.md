# Mallampati-Classification
## Overview
This project implements a deep learning-based system for automated Mallampati score classification to predict difficult intubation using oropharyngeal images. It addresses the subjectivity of manual assessments (60% inter-observer agreement) by employing MobileNetV2 and VGG16 models, trained on a dataset of 800 training, 100 validation, and 100 test images across four Mallampati classes, aiming to reduce intubation complications (5–15% failure rate) and improve airway management in anesthesia, emergency medicine, and intensive care.

## Features 
- Objective: Automates Mallampati score classification for consistent, objective airway assessment.

- Models: MobileNetV2 and two VGG16 variants, fine-tuned with the last 10 layers unfrozen, global average pooling, 128-unit dense layer, batch normalization, and dropout.

- Dataset: 800 training, 100 validation, 100 test images, balanced across four Mallampati classes (I–IV).

- Training: 3-fold cross-validation, test-time augmentation (TTA), early stopping, and learning rate scheduling.

- Evaluation: Generates classification report, confusion matrix, ROC, and precision-recall curves.

- Large File Support: Model weights (.pth) managed with Git LFS due to large file sizes.

## Requirements

- Python 3.8+



- PyTorch



- torchvision



- scikit-learn



- matplotlib



- seaborn



- numpy



- Pillow



- Git LFS

### Install dependencies:
```bash
pip install torch torchvision scikit-learn matplotlib seaborn numpy pillow
```

### Install Git LFS:
```bash
git lfs install
```

## Dataset Structure
```plain
augmented_data_output/
├── train/
│   ├── 1/  # Mallampati Class I
│   ├── 2/  # Mallampati Class II
│   ├── 3/  # Mallampati Class III
│   ├── 4/  # Mallampati Class IV
├── validation/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
├── test/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   ├── 4/
```
- Each class folder contains images (.jpg, .jpeg, .png).

-  Counts per class: 200 images per class in train (800 total), 25 per class in validation and test (100 each).

## Setup

1. Clone the repository:

```bash
git clone https://github.com/username/repo_name.git
cd repo_name
git lfs pull
```


2. Ensure dataset is in ./augmented_data_output/.



3. Run the pipeline:

- Open and run code.ipynb in Jupyter to execute the pipeline.


4. Outputs:





- Trained models saved in ./models/ (e.g., MobileNetV2_final.pth, VGG16_v1_final.pth, VGG16_v2_final.pth).



- Results (classification report, plots) saved in ./results/.

## Adding Large Models to GitHub

Model weights are managed with Git LFS:

```bash
git lfs track "*.pth"
git add .gitattributes
git add models/*.pth
git commit -m "Add model weights with LFS"
git push origin main
```

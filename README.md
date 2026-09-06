# Deep Learning Projects

A PyTorch notebook implementing multi-label image classification on the Pascal VOC 2012 dataset (20 object categories, one or more labels per image).

## Contents

`image_classification.ipynb`:

- Downloads Pascal VOC 2012 via `torchvision` and builds a multi-label dataset from the object-detection annotations.
- Uses an ImageNet-pretrained EfficientNet-B3 backbone with a replaced multi-label classification head.
- Training setup: mixed precision (`torch.amp`), AdamW optimizer with differential learning rates, OneCycleLR schedule, and label-smoothed binary cross-entropy loss.
- Evaluation via mean average precision (mAP), with helper code for training curves, per-class average precision, prediction visualization, and a label co-occurrence heatmap.
- Includes an optional (disabled by default) ResNet-50 baseline for comparison.

## Status

The saved notebook shows the data pipeline, model, and training loop defined and partially exercised (data loading and model construction ran successfully: 5,717 training / 5,823 validation images, 10.7M trainable parameters), but the training-loop cell in the saved output ends in a `NameError` before completing an epoch. No trained-model metrics (mAP, loss curves) are present in the saved outputs, so none are reported here. Running the notebook top to bottom in a fresh kernel should resolve the variable-ordering issue.

## Requirements

Python 3 with torch, torchvision, scikit-learn, matplotlib, numpy, and Pillow. Written for a GPU-backed Colab runtime.

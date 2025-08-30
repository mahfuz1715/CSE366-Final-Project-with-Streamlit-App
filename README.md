🍃 Bangladeshi Mango Leaf Classification

Author: Mahfuz Uddin Ahmed
ID: 2023-1-60-207
Course: CSE 366 – Artificial Intelligence (Summer 2025)
Project Type: Group Project


📌 Project Overview

This project focuses on building and explaining a complete deep learning pipeline for image classification on the Bangladeshi Mango Leaf Dataset. The goal is to move from a hand-crafted CNN baseline to state-of-the-art transfer learning and Vision Transformer models, complemented by Explainable AI (XAI) techniques for interpretability. The final system is deployed as an interactive Streamlit application where users can upload mango leaf images, obtain cultivar predictions, and visualize model explanations.


🎯 Objectives

Construct a balanced dataset of six Bangladeshi mango cultivars: Amrapali, Banana, Chaunsa, Fazli, Haribhanga, and Himsagar.

Implement a Custom CNN model designed and trained from scratch.

Fine-tune multiple transfer learning backbones for robust classification.

Train and evaluate a Vision Transformer (ViT) model.

Apply XAI techniques (Grad-CAM, Grad-CAM++, Eigen-CAM, Ablation-CAM, LIME) for model interpretability.

Deploy the trained model in a Streamlit web application for real-time predictions and visualizations.


📊 Dataset Description

Total Classes: 6 mango cultivars

Class Distribution:

Amrapali → 120 images

Banana → 67 images

Chaunsa → 150 images

Fazli → 160 images

Haribhanga → 120 images

Himsagar → 120 images

Dataset Parts: Original images + Augmented images

Train/Validation/Test split: 70% / 20% / 10%


⚙️ Methodology

Data Preparation

Applied resizing, normalization, and basic augmentations (random crop, flip).

Used PyTorch DataLoader for efficient batching (Batch Size = 32).

Custom CNN

Designed from scratch with 4 convolutional blocks + fully connected classifier.

Included BatchNorm, ReLU activations, Dropout, and Adaptive AvgPooling.

Transfer Learning

Fine-tuned multiple pre-trained architectures on ImageNet weights.

Compared performance across accuracy, precision, recall, and F1-score.

Vision Transformer (ViT)

Implemented ViT-based model for sequence-like image representation.

Explainable AI (XAI)

Applied Grad-CAM, Grad-CAM++, Eigen-CAM, Ablation-CAM, and LIME.

Found that ResNet-50 with Eigen-CAM gave the clearest interpretability.

Deployment

Built an interactive Streamlit app for real-time image upload, prediction, and XAI visualization overlays.


📈 Results

Best Performing Model: EfficientNet-B0 (highest accuracy).

Loss Curve: ResNet-50 showed the smoothest training/validation loss, EfficientNet-B0 ranked second.

XAI Analysis: Eigen-CAM on ResNet-50 provided the most reliable and interpretable heatmaps.

Custom CNN: Achieved a better fit (moderate accuracy) but outperformed by transfer learning models.


Streamlit Application

The Streamlit app allows users to:

Upload or choose a mango leaf image.

Get the predicted cultivar label.

Toggle XAI overlays to visualize decision-making.


📚 References

Original CNN papers (ResNet, DenseNet, EfficientNet, MobileNet, VGG).

Vision Transformer (ViT).

XAI techniques: Grad-CAM, Grad-CAM++, Eigen-CAM, Ablation-CAM, LIME.




✅ Acknowledgement

This project was completed individually by Mahfuz Uddin Ahmed (ID: 2023-1-60-207) as part of the CSE366: Artificial Intelligence term project.

# MNIST Digit Classifier

A fully connected neural network built in PyTorch to classify handwritten digits from the MNIST dataset, with data augmentation, training/validation monitoring, and error analysis.

Overview

This project implements an end-to-end digit classification pipeline: loading and augmenting the MNIST dataset, training a multilayer perceptron (MLP), tracking training/validation metrics across epochs, and visualizing both aggregate performance and individual misclassified examples.

Features
Data augmentation on the training set: random rotation (±10°) and random affine transforms (translation, scaling) to improve generalization<br>
Train / validation / test split: 50,000 training images, 10,000 validation images, and the standard 10,000-image MNIST test set<br>
Custom MLP architecture (Neural_numbers): input flattening → Linear(784→128) → ReLU → Linear(128→128) → ReLU → Linear(128→10)<br>
Training loop with tqdm progress bars, Cross-Entropy loss, and the Adam optimizer<br>
Device-agnostic training — automatically uses CUDA, Apple MPS, or CPU depending on availability<br>
Full training dashboard: plots of training/validation loss and accuracy across epochs<br>
Error analysis: visualization of misclassified test samples with ground-truth vs. predicted labels<br>
Parameter counting utility to inspect model size<br>

Tech Stack
Language: Python<br>
Framework: PyTorch, torchvision<br>
Other libraries: NumPy, Matplotlib, OpenCV, scikit-learn (confusion matrix), tqdm

Results

The model is trained for 7 epochs, with training/validation loss and accuracy tracked and plotted after training. Misclassified examples from the test set are visualized to inspect common failure patterns.

<img width="1214" height="1009" alt="image" src="https://github.com/user-attachments/assets/1e95fa6b-660a-4bea-935a-70fc537a4d52" />

Motivation

This project was built as a hands-on introduction to training neural networks in PyTorch — covering the full workflow from data loading and augmentation to model design, training, and error analysis on a classic benchmark dataset.


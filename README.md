# EEG-Design-task-cognitive-load-classification
This repository presents a complete pipeline for classifying cognitive states from EEG recordings using classical machine learning models. The project involves preprocessing EEG data, extracting interpretable signal features, and training supervised classifiers to identify brain states associated with resting conditions. This work has implications for neuroergonomics, brain-computer interfaces, and cognitive workload monitoring.

## Overview
The objective of this project is to build a reproducible pipeline to classify cognitive states using EEG recordings during resting-state tasks (e.g., RST1 and RST2). The approach integrates established signal processing techniques with robust machine learning workflows, facilitating accurate and interpretable brain state classification.

## The key steps in the project are:
- EEG Preprocessing: Raw EEG data is preprocessed using MNE, including filtering, artifact removal via ICA, and epoch extraction.
- Feature Extraction: From the clean epochs, a set of handcrafted features is computed for each trial. These include:
    - Band power features (delta, theta, alpha, beta, gamma)
    - Hjorth parameters (activity, mobility, complexity)
    - Spectral entropy
- Feature Engineering: All features are organized into a labeled DataFrame suitable for machine learning. Labels are extracted from session names to distinguish between RST1 and RST2.
- Model Training: Classical models such as SVM, Random Forest, and XGBoost are trained using 5-fold cross-validation. Feature scaling is applied to standardize the input space.
- Model Evaluation: Performance is evaluated using accuracy, precision, recall, F1-score, and confusion matrices. The best models are saved for later reuse without retraining.

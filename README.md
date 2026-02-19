# Deconvolution of BOLD fMRI Signals using Neural Networks

## Overview

Functional MRI (fMRI) measures brain activity indirectly through the Blood-Oxygen-Level Dependent (BOLD) signal. Recovering the underlying neural activity from BOLD measurements requires solving an ill-posed deconvolution problem.

Traditional approaches (e.g., Wiener filtering, Total Activation) rely on assumptions about:
- Noise distributions
- Hemodynamic Response Function (HRF)
- Sparsity priors

In this project, I explored **data-driven alternatives** using neural networks to perform BOLD signal deconvolution without explicit modeling assumptions.

Two architectures are investigated:

- **Convolutional Neural Networks (CNNs)**
- **Autoencoders (AEs) with latent deconvolution module**

The project studies:

- Supervised training on real and simulated data
- Two-stage training (simulation pretraining → real fine-tuning)
- Mixed-domain training for generalization
- Out-of-distribution (OOD) robustness
- Physically constrained lightweight models

## Problem Formulation

The BOLD signal can be modeled as:

$y(t) = (u * \text{HRF})(t) + n(t)$

Where:
- $u(t)$: underlying activity-inducing neural signal (piecewise constant)
- HRF: hemodynamic response function
- $n(t)$: noise

The objective is to estimate $u(t)$ from observed $y(t)$.

This is an **inverse problem** that is:
- Ill-posed
- Sensitive to noise
- Dependent on HRF assumptions

## Datasets

### 1️⃣ Real Data – Human Connectome Project (HCP)

Task-based fMRI data from 100 subjects:
- Motor
- Language
- Gambling
- Emotion (OOD)
- Working Memory (OOD)

Several curation steps were performed by another lab member prior to the project. They include:
- Preprocessing: Motion correction, slice timing correction, registration to MNI space, grey matter masking, smoothing
- Target estimation using the general linear model
- Selection of only the statistically significant voxels for ensuring high quality input/target pairs (98th percentile F-statistic, R² > 0.1).

I used the prior curated dataset and performed the following additional data processing steps:
- Subject-level train/val/test split to reduce bias and leakage
- Data augmentation (temporal shift, amplitude scaling, noise injection)
- Zero-padding to fixed sequence length

Final dataset:
- ~980k ID samples (70 / 15 / 15 split, motor - language - gambling tasks only)
- ~250k OOD samples (emotion - working memory tasks only)

### 2️⃣ Simulated Data

To overcome lack of ground truth and improve generalization:

- Generated piecewise-constant neural signals
- Convolved with variable SPM (Statistical Parametric Mapping) HRFs
- Added Gaussian noise matching real SNR levels

This allowed:
- Controlled HRF variability
- Larger training volume
- Explicit ground truth access

![simulated sample % HRFs](figures/sim_data.png)

## Methodology

### Loss Function

To encourage piecewise-constant reconstructions, the loss used was the Mean Squared Error (MSE) with a Gaussian version of Total Variation (TV) regularization, penalizing smooth transitions while preserving sharp changes.
The Gaussian TV formulation follows the approach described in:
G. L. Zeng, *Better than the total variation regularization* International Journal of Biomedical Research & Practice, 2024

### Models

#### 1️⃣ Convolutional Neural Networks

- 1–4 convolutional layers
- Kernel size search: 5–90
- Filters per layer: 1–10
- Zero-padding to preserve signal length

Hyperparameter search performed using Optuna (Tree-Structured Parzen Estimator).

### 2️⃣ Autoencoder with Latent Deconvolution Module

Architecture:

- Encoder: Conv1D → Pooling → FC → Dropout
- Decoder: Symmetric reconstruction
- Latent Deconvolution Module (LDM):
  - Fully connected expansion
  - 4-layer CNN

~200k parameters.

Sequential training:
1. Train encoder-decoder (reconstruction)
2. Freeze and train LDM (deconvolution)

📌 **Insert Figure:**  
`Figure 2 – Autoencoder architecture diagram`

## Training Strategy

Three main training paradigms were evaluated:

### 1️⃣ Simulated Pretraining → Real Fine-tuning

- Pretrain on synthetic data
- Fine-tune using real BOLD samples
- Multiple adaptation strategies (adapters, partial freezing)

### 2️⃣ Mixed-Domain Training

Train on varying proportions of real vs simulated data:
- 0%, 10%, 30%, 50%, 70%, 100% real

### 3️⃣ Physically Constrained CNN

- Field of view < 40 time points (HRF length)
- Only 585 parameters
- Enforces biologically realistic temporal receptive field

## Results

### Fine-Tuning Improves Real-Data Performance

Two-stage training significantly reduced MSE on real tasks:

- Strong improvement on ID tasks
- CNN showed better OOD generalization
- AE achieved lowest ID error but overfit more easily

📌 **Insert Figure:**  
`Figure 6 & 7 – Finetuning MSE comparison`

### Mixed Training Improves Generalization

Training with as little as **30% real data** significantly improved OOD performance while maintaining simulated performance.

📌 **Insert Figure:**  
`Figure 3 – MSE vs real/simulated proportion`

### CNN vs AE

| Model | ID Performance | OOD Performance | Generalization |
|-------|----------------|----------------|----------------|
| CNN   | Good          | Best           | Strong        |
| AE    | Best on ID    | Worse on OOD   | Overfits more |
| CNN-small | Lower ID accuracy | Most stable | Most generalizable |

Key insight:
- CNNs rely on **local temporal dependencies**, aiding robustness.
- AEs rely on **global latent structure**, increasing overfitting risk.

📌 **Insert Figure:**  
`Figure 8–11 – Qualitative inference comparisons`

## Key Technical Contributions

- Designed supervised deconvolution pipeline with GLM-based target estimation
- Built synthetic fMRI generator with HRF variability
- Implemented Gaussian-TV regularized loss
- Performed Bayesian hyperparameter optimization
- Systematic OOD evaluation
- Investigated domain shift between simulated and real fMRI
- Explored physically constrained architectures

## Tech Stack

- Python
- PyTorch
- Optuna
- NumPy / SciPy
- fMRI preprocessing pipelines (HCP data)

## Takeaways

- Simulation pretraining improves real-data performance.
- Domain gap between synthetic and real fMRI is significant.
- CNNs provide better OOD robustness than autoencoders.
- Physically constrained models improve stability but reduce peak accuracy.
- Mixed-domain training is promising for improving generalization.

## Future Work

- Improved HRF variability modeling
- End-to-end training of AE with single composite loss
- Physically informed architecture constraints
- Extension to resting-state fMRI
- Interpretability analysis of latent representations

## Repository Structure


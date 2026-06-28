# LinAlg-ImageCompare: Advanced Picture-to-Picture Comparison

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-green?logo=numpy)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📖 Project Overview

**LinAlg-ImageCompare** is a Python-based computational imaging project designed to quantify the similarity and differences between two images using rigorous **Linear Algebra** and **Algorithmic** techniques. 

Instead of relying on "black-box" machine learning models, this project treats images as mathematical constructs (matrices and tensors). By applying matrix factorization, vector space geometry, and algorithmic convolutions, we can extract deep structural, textural, and pixel-level comparisons. This approach is highly efficient, mathematically interpretable, and serves as a foundational stepping stone for understanding computer vision and deep learning.

---

## 📐 Mathematical & Algorithmic Foundations

This project implements comparisons across three mathematical dimensions: **Norm-based Distances**, **Vector Space Similarities**, and **Matrix Factorizations**.

### 1. Linear Algebra: Images as Matrices and Vectors
An image is fundamentally a matrix $A \in \mathbb{R}^{m \times n}$ (for grayscale) or a 3D tensor $T \in \mathbb{R}^{m \times n \times c}$ (for RGB). We flatten these into vectors $\mathbf{a}, \mathbf{b} \in \mathbb{R}^N$ for specific calculations.

#### A. Matrix Norms (Pixel-wise Differences)
We measure the absolute divergence between image matrices $A$ and $B$ using matrix norms:
*   **Frobenius Norm (L2 Norm):** Measures the Euclidean distance between the two image matrices. Highly sensitive to absolute pixel changes.
    $$ ||A - B||_F = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} |a_{ij} - b_{ij}|^2} $$
*   **Nuclear Norm (L1 Norm / Manhattan):** The sum of absolute differences. More robust to outliers (e.g., salt-and-pepper noise) than the Frobenius norm.
    $$ ||A - B|_1 = \sum_{i=1}^{m} \sum_{j=1}^{n} |a_{ij} - b_{ij}| $$

#### B. Vector Space Geometry (Structural Similarity)
By treating images as vectors in an $N$-dimensional space, we can measure the angle between them, which indicates structural similarity regardless of brightness/contrast scaling.
*   **Cosine Similarity:** Calculates the cosine of the angle $\theta$ between flattened image vectors $\mathbf{a}$ and $\mathbf{b}$.
    $$ \text{Cosine Similarity} = \cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}||_2 ||\mathbf{b}||_2} = \frac{\sum a_i b_i}{\sqrt{\sum a_i^2} \sqrt{\sum b_i^2}} $$
    *Values range from -1 to 1, where 1 means identical structure.*

#### C. Matrix Factorization (Feature Extraction)
*   **Singular Value Decomposition (SVD):** Decomposes the image matrix $A$ into $U \Sigma V^T$. 
    *   $U$ and $V$ represent the spatial and frequency bases.
    *   $\Sigma$ (diagonal matrix of singular values) represents the "energy" or importance of each feature.
    *   **Comparison Metric:** We compare the distribution of singular values ($\Sigma_A$ vs $\Sigma_B$) to determine if the core structural features of the images match, even if pixel values differ slightly.

### 2. Algorithmic Approaches
*   **Mean Squared Error (MSE) & Peak Signal-to-Noise Ratio (PSNR):** Standard algorithmic baselines for image quality assessment.
*   **2D Cross-Correlation / Convolution:** Using algorithmic sliding windows (kernels) to detect if one image is a translated/shifted version of another.
*   **Histogram Intersection:** Algorithmically comparing the color/intensity distributions of the two images.

---

## ✨ Features

- [x] **Image Preprocessing:** Automatic resizing, grayscale conversion, and normalization to ensure matrix dimensions match.
- [x] **Norm Calculations:** Compute L1, L2 (Frobenius), and Infinity norms between images.
- [x] **Vector Similarity:** Calculate Cosine Similarity and Dot Products.
- [x] **SVD Analysis:** Perform Singular Value Decomposition and compare the energy spectra (singular values) of images.
- [x] **Difference Heatmaps:** Generate visual matrices highlighting exactly where the algorithmic differences occur.
- [x] **Comprehensive Reporting:** Outputs a structured JSON/Dictionary report of all mathematical metrics.

---

## 📂 Project Structure

```text
picpiccomparison-python/
│
├── src/
│   ├── __init__.py
│   ├── loader.py          # Image loading, resizing, and matrix conversion
│   ├── linalg_metrics.py  # Linear algebra calculations (Norms, SVD, Cosine)
│   ├── algo_metrics.py    # Algorithmic calculations (MSE, PSNR, Histograms)
│   ├── visualizer.py      # Plotting heatmaps, SVD spectrums, and differences
│   └── comparator.py      # Main orchestrator class
│
├── tests/
│   ├── test_linalg.py     # Unit tests for linear algebra functions
│   └── test_algo.py       # Unit tests for algorithmic functions
│
├── data/                  # Sample images for testing
│   ├── image_A.png
│   └── image_B.png
│
├── notebooks/             # Jupyter notebooks for mathematical exploration
│   └── math_exploration.ipynb
│
├── requirements.txt
├── Project.md             # This file
└── main.py                # Entry point for CLI execution
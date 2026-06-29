# PicPicComparison

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-green?logo=numpy)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow)

Advanced Picture-to-Picture Comparison using **Linear Algebra** and **Algorithmic** techniques from Linear Algebra Classes.

Instead of relying on "black-box" ML models, PicPicComparison treats images as mathematical constructs (matrices and tensors) and applies matrix factorization, vector space geometry, and algorithmic analysis to quantify similarity.

## Features

- **Norm-Based Distances:** Frobenius (L2), Manhattan (L1), Chebyshev (L∞)
- **Vector Space Geometry:** Cosine Similarity, Dot Products
- **SVD Analysis:** Singular Value Decomposition energy spectra comparison
- **Algorithmic Metrics:** MSE, PSNR, Histogram Intersection
- **Advanced Metrics:** Patch-wise Cosine, PCA Projection, NCC, Sobel Edge Detection
- **Interactive Web UI:** Streamlit-based with Basic and Advanced modes
- **Advanced Mode:** Step-by-step pipeline with LaTeX formulas and intermediate visualizations
- **Export:** JSON reports, PNG plots, ZIP bundles

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Web UI

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload two images and compare!

### 3. Run from CLI

```bash
python main.py image_A.png image_B.png
python main.py image_A.png image_B.png --output-dir results/ --size 512 512
python main.py img1.png img2.png --json-only --no-plots
```

## Project Structure

```
picpiccomparison-python/
├── app.py                          # Streamlit home page
├── main.py                         # CLI entry point
├── pages/
│   ├── 1_📊_Basic_Comparison.py    # Basic comparison UI
│   ├── 2_🔬_Advanced_Analysis.py   # Advanced step-through UI
│   └── 3_📥_Export_Results.py      # Export/download page
├── ui/
│   ├── components.py               # Reusable Streamlit widgets
│   ├── step_runner.py              # Advanced mode pipeline engine
│   └── styles.py                   # Custom CSS
├── src/
│   ├── loader.py                   # Image loading, resizing, masking
│   ├── linalg_metrics.py           # Linear algebra metrics
│   ├── algo_metrics.py             # Algorithmic metrics
│   ├── visualizer.py               # Matplotlib visualizations
│   └── comparator.py               # Main orchestrator class
├── tests/
│   ├── conftest.py                 # Pytest fixtures
│   ├── test_loader.py              # Loader tests
│   ├── test_linalg.py              # Linear algebra tests
│   ├── test_algo.py                # Algorithmic tests
│   ├── test_visualizer.py          # Visualizer tests
│   └── test_comparator.py          # Integration tests
├── requirements.txt
└── Project.md
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

## Usage Modes

### Basic Mode
- Side-by-side image comparison
- 8 key metric cards (Frobenius, Cosine, MSE, PSNR, L1, L∞, Histogram, SVD)
- Difference heatmap, SVD spectrum, histogram overlay

### Advanced Mode
- Interactive 8-step pipeline walkthrough
- LaTeX formulas for every computation
- Intermediate visualizations at each step
- Detailed matrix statistics and timing

## Mathematical Foundations

| Metric | Formula | Description |
|--------|---------|-------------|
| Frobenius (L2) | `‖A-B‖_F = √(Σ|a_ij - b_ij|²)` | Euclidean distance between matrices |
| Manhattan (L1) | `‖A-B‖₁ = Σ|a_ij - b_ij|` | Sum of absolute differences |
| Cosine Similarity | `cos(θ) = (a·b)/(‖a‖‖b‖)` | Angle between image vectors |
| SVD Energy | `Energy = Σσ_i²` | Structural feature importance |
| MSE | `(1/mn)Σ(a_ij - b_ij)²` | Mean squared error |
| PSNR | `10·log₁₀(MAX²/MSE)` | Peak signal-to-noise ratio |
| Histogram Intersection | `Σ min(h_A, h_B)` | Intensity distribution overlap |

## License

MIT

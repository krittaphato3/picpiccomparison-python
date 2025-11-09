# Linear-Algebra Image Similarity (NumPy + Pillow)

Single-file GUI that compares two images using classic linear-algebra metrics with robust preprocessing (Sobel edges, foreground masking, auto-crop, zero-mean). Works on Python 3.14 without OpenCV.

## Methods
- **Cosine Similarity (Centered, Masked)**
- **Frobenius Norm Distance (Masked)**
- **Patch-wise Cosine Similarity (Masked, Z-Score)**
- **PCA Projection Cosine Similarity**
- **SVD Spectral + Directional Similarity**
- **Normalized Cross-Correlation (NCC)**

## Install
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

# gui_similarity.py
# Linear-Algebra Image Similarity (NumPy + Pillow)
# Methods (explicit, searchable):
#   - Cosine Similarity (Centered, Masked)
#   - Frobenius Norm Distance (Masked)
#   - Patch-wise Cosine Similarity (Masked, Z-Score)
#   - PCA Projection Cosine Similarity
#   - SVD Spectral + Directional Similarity
#   - Normalized Cross-Correlation (NCC)
#
# Always-on defaults (no auto heuristics, no hidden weights):
#   - Sobel edge images
#   - Zero-mean centering
#   - Foreground mask (threshold) + auto-crop to mask bbox

import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

EPS = 1e-9

# ---------------- LA helpers ----------------
def l2(v): return float(np.sqrt((v*v).sum() + EPS))
def dot(a,b): return float(a.dot(b))

def cos_to_pct(s):
    # Map cosine [-1,1] -> [0,100] with 0 correlation -> 0%.
    return 100.0 * max(0.0, float(s))

def zscore(v):
    m = float(v.mean()); s = float(v.std() + EPS)
    return (v - m)/s

# ---------------- Image ops ----------------
def pil_read_gray(path):
    im = Image.open(path).convert("L")
    return np.asarray(im, dtype=np.float32)/255.0

def resize_np(img, size_px):
    H,W = img.shape
    if H==size_px and W==size_px: return img
    y = np.linspace(0, H-1, size_px, dtype=np.float32)
    x = np.linspace(0, W-1, size_px, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    x0 = np.floor(xv).astype(int); x1 = np.clip(x0+1, 0, W-1)
    y0 = np.floor(yv).astype(int); y1 = np.clip(y0+1, 0, H-1)
    wa = (x1 - xv) * (y1 - yv)
    wb = (xv - x0) * (y1 - yv)
    wc = (x1 - xv) * (yv - y0)
    wd = (xv - x0) * (yv - y0)
    return (wa*img[y0,x0] + wb*img[y0,x1] + wc*img[y1,x0] + wd*img[y1,x1]).astype(np.float32)

def sobel_mag(img):
    Kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=np.float32)
    Ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32)
    def conv_pad(a, k):
        kh, kw = k.shape; ph, pw = kh//2, kw//2
        ap = np.pad(a, ((ph,ph),(pw,pw)), mode="edge")
        H,W = a.shape; out = np.empty_like(a, dtype=np.float32)
        for i in range(H):
            for j in range(W):
                out[i,j] = float((ap[i:i+kh, j:j+kw] * k).sum())
        return out
    gx, gy = conv_pad(img, Kx), conv_pad(img, Ky)
    mag = np.sqrt(gx*gx + gy*gy)
    denom = (mag.mean() + 3*mag.std() + EPS)
    return (mag/denom).astype(np.float32)

# ---------------- Masking & cropping ----------------
def auto_mask(img, thr=0.95):
    return (img < float(thr))

def bbox_from_mask(mask, margin=3):
    ys, xs = np.where(mask)
    if ys.size == 0: return (0,0,mask.shape[1],mask.shape[0])
    y0,y1 = max(0,int(ys.min()-margin)), min(mask.shape[0], int(ys.max()+1+margin))
    x0,x1 = max(0,int(xs.min()-margin)), min(mask.shape[1], int(xs.max()+1+margin))
    return (x0,y0,x1,y1)

def crop_to_bbox(img, bbox):
    x0,y0,x1,y1 = bbox
    return img[y0:y1, x0:x1]

def apply_mask_flat(img, mask, center=True):
    v = img[mask].astype(np.float32)
    if v.size == 0: return np.zeros(1, dtype=np.float32)
    return zscore(v) if center else v

# ---------------- Metrics (explicit) ----------------
def cosine_masked(A,B,M):
    va, vb = apply_mask_flat(A,M,True), apply_mask_flat(B,M,True)
    na, nb = l2(va), l2(vb)
    if na<EPS or nb<EPS: return 0.0
    return cos_to_pct(dot(va,vb)/(na*nb))

def frob_masked(A,B,M):
    W = M.astype(np.float32)
    num = float(((A-B)**2 * W).sum())
    den = float(((A*A + B*B) * W).sum() + EPS)
    dist = math.sqrt(num)/math.sqrt(den)
    return 100.0*(1.0 - min(1.0, dist))

def patchcos_masked(A,B,M, patch=16, stride=8, min_cov=0.1):
    H,W = A.shape
    if patch>H or patch>W: return 0.0
    S=[]
    for i in range(0, H-patch+1, stride):
        for j in range(0, W-patch+1, stride):
            m = M[i:i+patch, j:j+patch]
            if m.mean() < min_cov: continue
            va = apply_mask_flat(A[i:i+patch, j:j+patch], m, True)
            vb = apply_mask_flat(B[i:i+patch, j:j+patch], m, True)
            na, nb = l2(va), l2(vb)
            if na<EPS or nb<EPS: continue
            S.append(dot(va,vb)/(na*nb))
    if not S: return 0.0
    return cos_to_pct(float(np.mean(S)))

def pca_cosine(A,B,M,k=20):
    va, vb = apply_mask_flat(A,M,True), apply_mask_flat(B,M,True)
    if va.shape == vb.shape and float(np.linalg.norm(va - vb)) < 1e-8:
        return 100.0
    X = np.stack([va, vb], axis=1)
    if float(np.linalg.norm(X[:,0] - X[:,1])) < 1e-8:
        na, nb = l2(va), l2(vb)
        if na<EPS or nb<EPS: return 0.0
        return cos_to_pct(dot(va,vb)/(na*nb))
    mu = X.mean(axis=1, keepdims=True)
    Xc = X - mu
    U,S,Vt = np.linalg.svd(Xc, full_matrices=False)
    r = int((S>1e-8).sum())
    if r==0:
        na, nb = l2(va), l2(vb)
        if na<EPS or nb<EPS: return 0.0
        return cos_to_pct(dot(va,vb)/(na*nb))
    k = int(max(1, min(k, r)))
    Uk = U[:,:k]
    ak = Uk.T @ (va - mu[:,0]); bk = Uk.T @ (vb - mu[:,0])
    na, nb = l2(ak), l2(bk)
    if na<EPS or nb<EPS:
        na0, nb0 = l2(va), l2(vb)
        if na0<EPS or nb0<EPS: return 0.0
        return cos_to_pct(dot(va,vb)/(na0*nb0))
    return cos_to_pct(dot(ak,bk)/(na*nb))

def svd_energy(A,B,M,k=20):
    # crop to mask bbox so background doesn't dominate
    bbox = bbox_from_mask(M); A = crop_to_bbox(A,bbox); B = crop_to_bbox(B,bbox)
    Ua,Sa,Va = np.linalg.svd(A, full_matrices=False)
    Ub,Sb,Vb = np.linalg.svd(B, full_matrices=False)
    k = int(max(1, min(k, len(Sa), len(Sb))))
    sa, sb = Sa[:k].astype(np.float32), Sb[:k].astype(np.float32)
    # spectra similarity
    num = float(np.linalg.norm(sa - sb))
    den = float(np.linalg.norm(sa) + np.linalg.norm(sb) + EPS)
    s_sigma = max(0.0, 1.0 - num/den)
    # principal directions agreement
    cu = abs(float(Ua[:,0].dot(Ub[:,0]))/(np.linalg.norm(Ua[:,0])*np.linalg.norm(Ub[:,0])+EPS))
    cv = abs(float(Va[0,:].dot(Vb[0,:]))/(np.linalg.norm(Va[0,:])*np.linalg.norm(Vb[0,:])+EPS))
    s_vec = 0.5*(cu+cv)
    return 100.0*max(0.0, s_sigma*s_vec)

def ncc_shift(A,B,M,max_shift=8):
    H,W = A.shape; best = -1.0
    for dy in range(-max_shift, max_shift+1):
        for dx in range(-max_shift, max_shift+1):
            y0a=max(0,dy); y1a=min(H,H+dy); x0a=max(0,dx); x1a=min(W,W+dx)
            y0b=max(0,-dy);y1b=min(H,H-dy);x0b=max(0,-dx);x1b=min(W,W-dx)
            if y1a-y0a<=0 or x1a-x0a<=0: continue
            Aov = A[y0a:y1a, x0a:x1a]; Bov = B[y0b:y1b, x0b:x1b]
            Mov = M[y0a:y1a, x0a:x1a] | M[y0b:y1b, x0b:x1b]
            va = apply_mask_flat(Aov, Mov, True); vb = apply_mask_flat(Bov, Mov, True)
            na, nb = l2(va), l2(vb)
            if na<EPS or nb<EPS: continue
            s = dot(va,vb)/(na*nb)
            if s>best: best = s
    return cos_to_pct(best if best>-2 else -1)

# ---------------- Preprocess ----------------
def preprocess(path, size_px, mask_thr):
    img = pil_read_gray(path)
    mask = auto_mask(img, mask_thr)
    # crop to content, then resize, then recompute mask
    bbox = bbox_from_mask(mask)
    img  = crop_to_bbox(img, bbox)
    img  = resize_np(img, size_px)
    mask = auto_mask(img, mask_thr)
    # edges always on
    img = sobel_mag(img)
    return img.astype(np.float32), mask

def pil_preview(path, max_side=380):
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    return ImageTk.PhotoImage(im)

# ---------------- GUI ----------------
METHOD_LABELS = [
    "Cosine Similarity (Centered, Masked)",
    "Frobenius Norm Distance (Masked)",
    "Patch-wise Cosine Similarity (Masked, Z-Score)",
    "PCA Projection Cosine Similarity",
    "SVD Spectral + Directional Similarity",
    "Normalized Cross-Correlation (NCC)"
]
LABEL_TO_KEY = {
    METHOD_LABELS[0]: "cosine",
    METHOD_LABELS[1]: "frob",
    METHOD_LABELS[2]: "patchcos",
    METHOD_LABELS[3]: "pca",
    METHOD_LABELS[4]: "svd",
    METHOD_LABELS[5]: "ncc",
}

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Linear Algebra Image Similarity")
        self.geometry("1020x720"); self.resizable(True, True)

        self.path1 = None; self.path2 = None
        self.photo1 = None; self.photo2 = None

        top = ttk.Frame(self, padding=10); top.pack(fill="both", expand=True)
        left = ttk.Frame(top); right = ttk.Frame(top)
        left.pack(side="left", fill="both", expand=True, padx=(0,6))
        right.pack(side="left", fill="both", expand=True, padx=(6,0))

        ttk.Button(left, text="Select Image 1", command=self.pick1).pack(anchor="w")
        self.lbl1 = ttk.Label(left, text="No file selected", wraplength=460); self.lbl1.pack(anchor="w", pady=(4,8))
        self.canvas1 = tk.Label(left, bd=1, relief="sunken", bg="#222"); self.canvas1.pack(fill="both", expand=True)

        ttk.Button(right, text="Select Image 2", command=self.pick2).pack(anchor="w")
        self.lbl2 = ttk.Label(right, text="No file selected", wraplength=460); self.lbl2.pack(anchor="w", pady=(4,8))
        self.canvas2 = tk.Label(right, bd=1, relief="sunken", bg="#222"); self.canvas2.pack(fill="both", expand=True)

        ctl = ttk.Frame(self, padding=10); ctl.pack(fill="x")
        ttk.Label(ctl, text="Method:").grid(row=0, column=0, sticky="e")
        self.method_label = tk.StringVar(value=METHOD_LABELS[2])
        ttk.Combobox(ctl, textvariable=self.method_label, width=40,
                     values=METHOD_LABELS).grid(row=0, column=1, padx=6, sticky="w")

        ttk.Label(ctl, text="Resize(px):").grid(row=0, column=2, sticky="e")
        self.resize_px = tk.IntVar(value=256)
        ttk.Entry(ctl, textvariable=self.resize_px, width=6).grid(row=0, column=3, sticky="w")

        ttk.Label(ctl, text="Mask thr:").grid(row=0, column=4, sticky="e")
        self.mask_thr = tk.DoubleVar(value=0.95)
        ttk.Entry(ctl, textvariable=self.mask_thr, width=6).grid(row=0, column=5, sticky="w")

        # method params
        ttk.Label(ctl, text="Patch:").grid(row=1, column=0, sticky="e")
        self.patch = tk.IntVar(value=16); ttk.Entry(ctl, textvariable=self.patch, width=6).grid(row=1, column=1, sticky="w")
        ttk.Label(ctl, text="Stride:").grid(row=1, column=2, sticky="e")
        self.stride = tk.IntVar(value=8); ttk.Entry(ctl, textvariable=self.stride, width=6).grid(row=1, column=3, sticky="w")
        ttk.Label(ctl, text="Min cov:").grid(row=1, column=4, sticky="e")
        self.min_cov = tk.DoubleVar(value=0.10); ttk.Entry(ctl, textvariable=self.min_cov, width=6).grid(row=1, column=5, sticky="w")
        ttk.Label(ctl, text="k (PCA/SVD):").grid(row=1, column=6, sticky="e")
        self.k = tk.IntVar(value=20); ttk.Entry(ctl, textvariable=self.k, width=6).grid(row=1, column=7, sticky="w")
        ttk.Label(ctl, text="Shift (NCC):").grid(row=1, column=8, sticky="e")
        self.shift = tk.IntVar(value=8); ttk.Entry(ctl, textvariable=self.shift, width=6).grid(row=1, column=9, sticky="w")

        # Run + results
        bottom = ttk.Frame(self, padding=(10,0,10,10)); bottom.pack(fill="x")
        ttk.Button(bottom, text="Compare", command=self.compare).pack(side="left")
        self.lbl_score = ttk.Label(bottom, text="Score: —", font=("Segoe UI",12,"bold"))
        self.lbl_score.pack(side="left", padx=16)
        self.lbl_parts = ttk.Label(bottom, text="", justify="left")
        self.lbl_parts.pack(side="left", padx=8)

    def pick1(self):
        p = filedialog.askopenfilename(title="Select Image 1",
            filetypes=[("Image files","*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.tif;*.tiff"),("All files","*.*")])
        if not p: return
        self.path1 = p; self.lbl1.configure(text=p)
        try: self.photo1 = pil_preview(p); self.canvas1.configure(image=self.photo1)
        except Exception as e: messagebox.showerror("Error", f"Preview failed: {e}")

    def pick2(self):
        p = filedialog.askopenfilename(title="Select Image 2",
            filetypes=[("Image files","*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.tif;*.tiff"),("All files","*.*")])
        if not p: return
        self.path2 = p; self.lbl2.configure(text=p)
        try: self.photo2 = pil_preview(p); self.canvas2.configure(image=self.photo2)
        except Exception as e: messagebox.showerror("Error", f"Preview failed: {e}")

    def compare(self):
        if not self.path1 or not self.path2:
            messagebox.showwarning("Missing images","Please select both images."); return
        try:
            size = max(8, int(self.resize_px.get()))
            thr  = float(self.mask_thr.get())

            A, MA = preprocess(self.path1, size, thr)   # edges+centering+mask+crop are always on
            B, MB = preprocess(self.path2, size, thr)
            M = (MA | MB)

            label = self.method_label.get()
            key = LABEL_TO_KEY[label]
            if key == "cosine":
                s = cosine_masked(A,B,M)
            elif key == "frob":
                s = frob_masked(A,B,M)
            elif key == "patchcos":
                s = patchcos_masked(A,B,M, patch=int(self.patch.get()), stride=int(self.stride.get()),
                                    min_cov=float(self.min_cov.get()))
            elif key == "pca":
                s = pca_cosine(A,B,M, k=int(self.k.get()))
            elif key == "svd":
                s = svd_energy(A,B,M, k=int(self.k.get()))
            elif key == "ncc":
                s = ncc_shift(A,B,M, max_shift=int(self.shift.get()))
            else:
                raise ValueError("Unknown method")
            self.lbl_score.configure(text=f"Score: {s:.2f}%")
            self.lbl_parts.configure(text=label)
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    App().mainloop()

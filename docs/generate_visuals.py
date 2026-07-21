"""Generate per-motif GLCM/LBP visualizations for the docs.

Reuses the project's own feature extractors so the figures match exactly
what the SVM/KNN sees during training.
Run: python docs/generate_visuals.py
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

from src.glcm_svm import extract_glcm_features
from src.lbp_extractor import extract_lbp_features

MOTIFS = ["parang", "kawung", "megamendung"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "img")
GLCM_PROPS = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']


def sample_path(motif):
    d = os.path.join("data", "test", motif)
    files = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    return os.path.join(d, files[0])


def glcm_props_d1a0(gray):
    """Single GLCM (distance=1, angle=0) + its 6 scalar properties, for display."""
    q = (gray / 8).astype(np.uint8)
    glcm = graycomatrix(q, distances=[1], angles=[0], levels=32, symmetric=True, normed=True)
    props = {p: float(graycoprops(glcm, p)[0, 0]) for p in GLCM_PROPS}
    return glcm[:, :, 0, 0], props


def make_figure(motif):
    gray = cv2.cvtColor(cv2.imread(sample_path(motif)), cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist = extract_lbp_features(gray)
    glcm_img, props = glcm_props_d1a0(gray)
    full_glcm = extract_glcm_features(gray)  # 72 dims (6 props x 3 dist x 4 angle)

    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"Analisis Tekstur Motif: {motif.upper()}", fontsize=16, fontweight='bold')

    ax[0, 0].imshow(gray, cmap='gray'); ax[0, 0].set_title("1. Gambar Grayscale (input)")
    ax[0, 0].axis('off')

    ax[0, 1].imshow(lbp, cmap='gray'); ax[0, 1].set_title("2. Peta LBP (pola lokal per piksel)")
    ax[0, 1].axis('off')

    ax[0, 2].bar(range(len(lbp_hist)), lbp_hist, color='#8C6239')
    ax[0, 2].set_title("3. Histogram LBP (10 bin fitur)")
    ax[0, 2].set_xlabel("bin pola"); ax[0, 2].set_ylabel("frekuensi")

    im = ax[1, 0].imshow(glcm_img, cmap='viridis')
    ax[1, 0].set_title("4. Matriks GLCM (d=1, 0°)")
    ax[1, 0].set_xlabel("level abu tetangga"); ax[1, 0].set_ylabel("level abu piksel")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046)

    ax[1, 1].barh(GLCM_PROPS, [props[p] for p in GLCM_PROPS], color='#4A6B82')
    ax[1, 1].set_title("5. Properti GLCM (nilai skalar)")
    for i, p in enumerate(GLCM_PROPS):
        ax[1, 1].text(props[p], i, f" {props[p]:.3f}", va='center', fontsize=9)

    ax[1, 2].axis('off')
    txt = (f"Vektor fitur akhir motif ini:\n\n"
           f"  GLCM : {len(full_glcm)} dimensi\n"
           f"  LBP  : {len(lbp_hist)} dimensi\n"
           f"  ---------------------------\n"
           f"  Total: {len(full_glcm) + len(lbp_hist)} dimensi\n\n"
           f"Vektor inilah yang di-scale (StandardScaler)\n"
           f"lalu diklasifikasi oleh SVM (RBF) / KNN.")
    ax[1, 2].text(0.02, 0.95, txt, va='top', ha='left', fontsize=11, family='monospace')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT_DIR, f"analysis_{motif}.png")
    plt.savefig(out, dpi=130); plt.close()
    print(f"  saved: {out}  (GLCM+LBP = {len(full_glcm) + len(lbp_hist)} dim)")
    return {motif: props}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Membuat visualisasi GLCM + LBP per motif...")
    for m in MOTIFS:
        make_figure(m)
    print("Selesai. Lihat docs/img/ dan docs/glcm_lbp_svm.md")


if __name__ == "__main__":
    main()

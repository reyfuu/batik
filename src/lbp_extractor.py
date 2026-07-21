import numpy as np
from skimage.feature import local_binary_pattern

def extract_lbp_features(img_gray, P=8, R=1, method='uniform'):
    """Ekstrak histogram Local Binary Pattern (LBP) dari gambar grayscale."""
    lbp = local_binary_pattern(img_gray, P, R, method=method)

    # Jumlah bin HARUS tetap (P + 2 untuk 'uniform'), bukan dari lbp.max(): gambar yang
    # tak memuat nilai LBP tertinggi -> bin lebih sedikit -> panjang fitur beda antar
    # gambar -> scaler.transform error saat predict.
    n_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return hist

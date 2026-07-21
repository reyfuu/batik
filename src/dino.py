"""DINOv2 image embedding — backbone terbaik untuk motif batik di DATASETv7.

Terukur 87% test (probe MLP) vs CLIP 81% vs SVM GLCM+LBP 42%. Dipakai di
src/compare_models.py (benchmark) dan web/app.py (inferensi upload).

CATATAN: jangan import faiss di proses yang sama dengan modul ini — torch dan
faiss sama-sama bawa libomp -> SIGABRT di macOS. Lihat src/clip_search.py.
"""
import os
import numpy as np

MODEL_ID = "facebook/dinov2-base"
V7_DIR = "data/Contoh foto batik/DATASETv7"

_model = _proc = None


def _load():
    """Muat model sekali per proses. Wajib di-cache: 86M param, load per request = server lumpuh."""
    global _model, _proc
    if _model is None:
        from transformers import AutoModel, AutoImageProcessor
        from src.clip_search import get_device
        _proc = AutoImageProcessor.from_pretrained(MODEL_ID)
        _model = AutoModel.from_pretrained(MODEL_ID).to(get_device()).eval()
    return _model, _proc


def embed(items, batch_size=32):
    """items = list path (str) atau PIL.Image -> (N, 768) float32 L2-normalized.

    Pakai CLS token (praktik standar linear-probe DINOv2). Satu-satunya jalur
    embedding: dipakai saat latih probe maupun inferensi web, supaya preprocessing
    training == produksi. Beda skala/warna = akurasi jatuh diam-diam tanpa error.
    """
    import torch
    from PIL import Image
    from src.clip_search import get_device

    model, proc = _load()
    dev = get_device()
    out = []
    for i in range(0, len(items), batch_size):
        imgs = [Image.open(x).convert("RGB") if isinstance(x, str) else x.convert("RGB")
                for x in items[i:i + batch_size]]
        inp = proc(images=imgs, return_tensors="pt").to(dev)
        with torch.no_grad():
            cls = model(**inp).last_hidden_state[:, 0]
        out.append((cls / cls.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def index(split, model_path="models", base_dir=V7_DIR):
    """Embed satu split (subfolder = kelas), cache ke npz. Return (vectors, classes)."""
    from torchvision.datasets import ImageFolder

    cache = os.path.join(model_path, f"dino_index_{split.lower()}.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d["vectors"], d["classes"]
    ds = ImageFolder(os.path.join(base_dir, split))
    vecs = embed([p for p, _ in ds.samples])
    classes = np.array([ds.classes[t] for t in ds.targets])
    os.makedirs(model_path, exist_ok=True)
    np.savez(cache, vectors=vecs, classes=classes)
    print(f"[DINOv2] {split}: {len(vecs)} vektor -> {cache}")
    return vecs, classes


def load_probe(model_path="models"):
    """Muat classifier di atas embedding DINOv2; latih dari index kalau belum ada.

    Hyperparameter (MLP 256, alpha=0.01) dipilih di validation split (compare_models.py),
    bukan TEST. Jangan diubah tanpa ukur ulang.
    """
    import pickle
    from sklearn.neural_network import MLPClassifier

    path = os.path.join(model_path, "dino_probe.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    print("[DINOv2] probe belum ada — melatih dari index TRAIN (sekali saja)...")
    X, y = index("TRAIN", model_path)
    clf = MLPClassifier((256,), alpha=0.01, max_iter=800, random_state=42).fit(X, y)
    pickle.dump(clf, open(path, "wb"))
    return clf

"""CLIP ViT-B/32 sebagai image embedder + similarity search untuk motif batik.

Bukan zero-shot text-label: gambar -> vektor 512-dim, cari tetangga terdekat pakai
cosine distance (1 - dot, vektor L2-normalized). Klasifikasi = voting k tetangga,
jadi bisa diadu dengan CNN & SVM di TEST set yang sama.
"""
import os
import sys
import numpy as np

MODEL_ID = "openai/clip-vit-base-patch32"
V7_DIR = "data/Contoh foto batik/DATASETv7"
EXTS = (".jpg", ".jpeg", ".png")

# torch SENGAJA tidak di-import di level modul: faiss + torch sama-sama bawa libomp,
# dua-duanya di satu proses macOS = SIGABRT (OMP Error #15), urutan import apa pun.
# Embedding sudah di-cache ke npz -> search murni (npz + faiss) tak menyentuh torch.
# Import torch di dalam _load() saja.
_model = _proc = None


def _load():
    global _model, _proc
    if _model is None:
        import torch
        from transformers import CLIPModel, CLIPImageProcessor
        _proc = CLIPImageProcessor.from_pretrained(MODEL_ID)
        _model = CLIPModel.from_pretrained(MODEL_ID).to(get_device()).eval()
    return _model, _proc


def get_device():
    import torch
    return torch.device("mps" if torch.backends.mps.is_available() else
                        ("cuda" if torch.cuda.is_available() else "cpu"))


def embed_images(paths, batch_size=32):
    """List path gambar -> array (N, 512) float32, L2-normalized.

    Satu-satunya jalur embedding: index TRAIN maupun query TEST, supaya preprocessing
    yang diindeks == yang dicari.
    """
    import torch
    from PIL import Image

    model, proc = _load()
    device = get_device()
    out = []
    for i in range(0, len(paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
        inputs = proc(images=imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        # transformers 5.x balikin objek (pooler_output = embed 512-dim); 4.x balikin tensor.
        feats = getattr(feats, "pooler_output", feats)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        out.append(feats.cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def embed_dir(root):
    """Subfolder = nama kelas. Return (vectors, rel_paths, classes)."""
    names = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    rel = [os.path.join(c, f) for c in names
           for f in sorted(os.listdir(os.path.join(root, c))) if f.lower().endswith(EXTS)]
    vecs = embed_images([os.path.join(root, r) for r in rel])
    return vecs, np.array(rel), np.array([r.split(os.sep)[0] for r in rel])


def build_index(split="TRAIN", base_dir=V7_DIR, model_path="models", rebuild=False):
    """Embed satu split lalu cache ke npz. Return dict {vectors, paths, classes}."""
    cache = os.path.join(model_path, f"clip_index{'' if split == 'TRAIN' else '_' + split.lower()}.npz")
    if os.path.exists(cache) and not rebuild:
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in ("vectors", "paths", "classes")}
    vecs, paths, classes = embed_dir(os.path.join(base_dir, split))
    os.makedirs(model_path, exist_ok=True)
    np.savez(cache, vectors=vecs, paths=paths, classes=classes)
    print(f"[CLIP] {split}: {len(vecs)} vektor -> {cache}")
    return {"vectors": vecs, "paths": paths, "classes": classes}


def _faiss_safe():
    """faiss hanya kalau terpasang DAN torch belum masuk proses ini.

    Kalau torch sudah ada, jatuh ke numpy: hasil identik (dua-duanya exact), cuma
    lebih lambat -- jauh lebih baik daripada proses mati tanpa pesan.
    """
    if "torch" in sys.modules:
        return False
    try:
        import faiss  # noqa: F401
        return True
    except ImportError:
        return False


def faiss_index(index):
    """Bangun IndexFlatIP dari vektor yang ada, di-cache di dict index.

    IndexFlatIP = inner product exact; vektor L2-normalized -> == cosine similarity,
    hasil sama dengan jalur numpy.

    ponytail: di 1050 vektor faiss TIDAK membantu (numpy 0.9ms vs faiss 1.1ms;
    IndexFlatIP juga brute-force, numpy pakai Accelerate BLAS di Apple Silicon).
    Default tetap numpy. faiss baru menang lewat index aproksimasi di korpus besar:
    di 100k vektor CLIP, IndexHNSWFlat(512, 32, METRIC_INNER_PRODUCT) 23x lebih cepat
    (85->3.6ms), recall@10 99.2%. Ganti ke HNSW kalau korpus tembus ~100k.
    """
    import faiss
    if "_faiss" not in index:
        fi = faiss.IndexFlatIP(index["vectors"].shape[1])
        fi.add(index["vectors"])
        index["_faiss"] = fi
    return index["_faiss"]


def search(query_vecs, index, k=10, backend="auto"):
    """Return (distances, neighbor_idx) top-k. Cosine distance, kecil = mirip.

    backend: "faiss" | "numpy" | "auto". Keduanya exact -> hasil identik; beda cuma
    kecepatan di dataset besar.
    """
    q = np.atleast_2d(query_vecs).astype(np.float32)
    if backend == "auto":
        backend = "faiss" if _faiss_safe() else "numpy"

    if backend == "faiss":
        if "torch" in sys.modules:
            # Error jelas > SIGABRT tanpa pesan.
            raise RuntimeError(
                "faiss tidak aman dipakai satu proses dengan torch (konflik libomp -> SIGABRT). "
                "Embed di proses terpisah lalu search dari npz, atau pakai backend='numpy'.")
        sims, idx = faiss_index(index).search(q, k)
        return 1.0 - sims, idx

    sims = q @ index["vectors"].T
    # argpartition (O(n)) lalu sort k kandidat saja; argsort penuh buang-buang.
    k = min(k, sims.shape[1])
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1)
    idx = np.take_along_axis(part, order, axis=1)
    dist = 1.0 - np.take_along_axis(sims, idx, axis=1)
    return dist, idx


def knn_predict(query_vecs, index, k=5, backend="auto"):
    """Voting kelas dari k tetangga terdekat, bobot = similarity."""
    dist, idx = search(query_vecs, index, k=k, backend=backend)
    neigh = index["classes"][idx]
    preds = []
    for row_cls, row_dist in zip(neigh, dist):
        score = {}
        for c, d in zip(row_cls, row_dist):
            score[c] = score.get(c, 0.0) + (1.0 - d)
        preds.append(max(score, key=score.get))
    return np.array(preds)


def evaluate(ks=(1, 3, 5, 10), base_dir=V7_DIR, model_path="models"):
    """Akurasi CLIP+kNN di DATASETv7/TEST, index dari TRAIN."""
    train = build_index("TRAIN", base_dir, model_path)
    test = build_index("TEST", base_dir, model_path)
    results = {}
    for k in ks:
        acc = (knn_predict(test["vectors"], train, k=k) == test["classes"]).mean() * 100
        results[k] = acc
        print(f"CLIP + kNN (k={k}): {acc:.2f}%")
    return results


def train_probe(base_dir=V7_DIR, model_path="models", C=100.0):
    """Linear probe: logistic regression di atas embedding CLIP beku.

    C=100 dipilih lewat 5-fold CV di TRAIN (bukan TEST). Latih <1 detik di CPU:
    CLIP tidak ikut dilatih, cuma 1050x512 vektor.
    """
    import pickle
    from sklearn.linear_model import LogisticRegression

    train = build_index("TRAIN", base_dir, model_path)
    clf = LogisticRegression(max_iter=3000, C=C).fit(train["vectors"], train["classes"])
    out = os.path.join(model_path, "clip_probe.pkl")
    pickle.dump(clf, open(out, "wb"))
    print(f"[CLIP] linear probe -> {out}")
    return clf


def predict_probe(image_paths, model_path="models"):
    """Klasifikasi gambar pakai CLIP embedding + linear probe tersimpan."""
    import pickle
    clf = pickle.load(open(os.path.join(model_path, "clip_probe.pkl"), "rb"))
    return clf.predict(embed_images(list(image_paths)))


def demo():
    """Self-check: search konsisten, dan faiss == numpy."""
    index = build_index("TRAIN")
    dist, idx = search(index["vectors"][:5], index, k=1)
    assert (idx[:, 0] == np.arange(5)).all(), "tetangga terdekat bukan diri sendiri"
    assert (dist[:, 0] < 1e-4).all(), f"jarak ke diri sendiri harus ~0, dapat {dist[:, 0]}"
    assert np.allclose(np.linalg.norm(index["vectors"], axis=1), 1.0, atol=1e-3), "vektor tidak L2-normalized"

    # faiss IndexFlatIP exact -> JARAK wajib sama dengan numpy. Urutan tetangga
    # berjarak identik boleh beda (tie-break faiss != argsort numpy, dua-duanya benar).
    # Cek jaraknya, bukan urutan tie.
    test = build_index("TEST")
    q = test["vectors"][:20]
    d_np, i_np = search(q, index, k=10, backend="numpy")
    d_fa, i_fa = search(q, index, k=10, backend="faiss")
    assert np.allclose(d_np, d_fa, atol=1e-5), f"jarak beda, max {np.abs(d_np - d_fa).max():.2e}"
    tied = np.isclose(d_np, d_fa, atol=1e-5) & (i_np != i_fa)
    assert ((i_np == i_fa) | tied).all(), "tetangga beda di posisi yang jaraknya tidak sama"

    # Yang penting: pilihan backend tidak boleh menggeser akurasi.
    acc_np = (knn_predict(test["vectors"], index, k=5, backend="numpy") == test["classes"]).mean()
    acc_fa = (knn_predict(test["vectors"], index, k=5, backend="faiss") == test["classes"]).mean()
    assert acc_np == acc_fa, f"akurasi beda antar backend: numpy {acc_np} vs faiss {acc_fa}"
    print(f"demo OK: vektor ternormalisasi, faiss == numpy (akurasi sama {acc_np*100:.2f}%)")


if __name__ == "__main__":
    demo()
    evaluate()

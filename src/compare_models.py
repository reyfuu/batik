"""Adu 7 pendekatan klasifikasi motif batik di DATASETv7, protokol identik.

Protokol (penting, jangan diubah):
  TRAIN 1050 --stratified--> fit 840 / val 210.  TEST 300 disentuh SEKALI di akhir.
  Semua hyperparameter & pemilihan epoch diputuskan di VAL, tidak pernah TEST.
  Memperbaiki bug train_v7() yang simpan best-epoch dari akurasi TEST -- itu bikin
  TEST jadi validation set dan angkanya terlalu optimis (kebocoran).

Kelompok:
  A. Embedding beku + classifier klasik : CLIP/DINOv2 x (LinearSVC, LogReg, SVM-RBF, MLP)
  B. Finetune CNN penuh                 : EfficientNet-B0, MobileNetV3-L, MobileNetV2
"""
import os
import pickle
import numpy as np

V7_DIR = "data/Contoh foto batik/DATASETv7"
SEED = 42


# ---------- embedding ----------

from src.dino import index as dino_index  # noqa: E402  (embedding DINOv2 -> src/dino.py)


def clip_ordered(split):
    """Embedding CLIP diurutkan sama dengan ImageFolder, supaya split-nya identik."""
    from torchvision.datasets import ImageFolder
    from src.clip_search import build_index

    idx = build_index(split)
    root = os.path.join(V7_DIR, split)
    ds = ImageFolder(root)
    pos = {p: i for i, p in enumerate(idx["paths"])}
    order = [pos[os.path.relpath(p, root)] for p, _ in ds.samples]
    return idx["vectors"][order], np.array([ds.classes[t] for t in ds.targets])


# ---------- kelompok A: embedding beku + classifier ----------

def probe_grid():
    """Kandidat classifier + grid-nya. Semua dari sklearn yang sudah terpasang."""
    from sklearn.svm import LinearSVC, SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    return {
        "LinearSVC": [(f"C={c}", LinearSVC(C=c, max_iter=5000)) for c in (0.01, 0.1, 1, 10)],
        "LogReg": [(f"C={c}", LogisticRegression(C=c, max_iter=3000)) for c in (1, 10, 100, 1000)],
        "SVM-RBF": [(f"C={c}", SVC(C=c, kernel="rbf", gamma="scale")) for c in (1, 10, 100, 1000)],
        "MLP-256": [(f"a={a}", MLPClassifier((256,), alpha=a, max_iter=800, random_state=SEED))
                    for a in (1e-4, 1e-2)],
    }


def run_probes(Xf, yf, Xv, yv, Xt, yt, tag, rows):
    """Untuk tiap classifier: pilih hyperparameter di VAL, lalu ukur TEST sekali."""
    for name, cands in probe_grid().items():
        best, best_val = None, -1.0
        for label, clf in cands:
            clf.fit(Xf, yf)
            v = (clf.predict(Xv) == yv).mean()
            if v > best_val:
                best, best_val, best_lbl = clf, v, label
        test = (best.predict(Xt) == yt).mean() * 100
        rows.append((f"{tag} + {name}", best_val * 100, test, best_lbl))
        print(f"  {tag} + {name:<10} val {best_val*100:5.2f}%  test {test:5.2f}%  ({best_lbl})")


# ---------- kelompok B: finetune CNN ----------

def finetune(arch, fit_idx, val_idx, epochs=12, batch_size=32, lr=1e-4):
    """Finetune penuh satu arsitektur torchvision. Best epoch dipilih di VAL, bukan TEST."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms, models
    from torchvision.datasets import ImageFolder
    from src.clip_search import get_device

    dev = get_device()
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])

    root = os.path.join(V7_DIR, "TRAIN")
    fit_ds = Subset(ImageFolder(root, transform=train_tf), fit_idx)
    val_ds = Subset(ImageFolder(root, transform=eval_tf), val_idx)
    test_ds = ImageFolder(os.path.join(V7_DIR, "TEST"), transform=eval_tf)
    n_cls = len(test_ds.classes)

    builders = {
        "EfficientNet-B0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, "classifier"),
        "MobileNetV3-L": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT, "classifier"),
        "MobileNetV2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT, "classifier"),
    }
    fn, weights, _ = builders[arch]
    model = fn(weights=weights)
    # Ganti Linear terakhir; tiap arsitektur beda struktur head.
    head = model.classifier
    last = len(head) - 1
    head[last] = nn.Linear(head[last].in_features, n_cls)
    model = model.to(dev)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    fit_ld = DataLoader(fit_ds, batch_size=batch_size, shuffle=True)
    val_ld = DataLoader(val_ds, batch_size=batch_size)
    test_ld = DataLoader(test_ds, batch_size=batch_size)

    def acc(loader):
        model.eval()
        c = t = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(dev), y.to(dev)
                c += model(x).argmax(1).eq(y).sum().item()
                t += y.size(0)
        return c / t

    best_val, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        for x, y in fit_ld:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        v = acc(val_ld)
        if v > best_val:
            best_val = v
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        print(f"    {arch} ep{ep+1:>2}/{epochs} val {v*100:5.2f}%")
    model.load_state_dict(best_state)
    return best_val * 100, acc(test_ld) * 100


def main():
    from sklearn.model_selection import train_test_split

    X_clip_tr, y_tr = clip_ordered("TRAIN")
    X_clip_te, y_te = clip_ordered("TEST")
    X_dino_tr, y_dino_tr = dino_index("TRAIN")
    X_dino_te, y_dino_te = dino_index("TEST")
    assert (y_tr == y_dino_tr).all() and (y_te == y_dino_te).all(), "urutan label CLIP != DINOv2"

    # Satu split untuk SEMUA model -> perbandingan terkontrol, bukan beda undian.
    fit_i, val_i = train_test_split(np.arange(len(y_tr)), test_size=0.2,
                                    stratify=y_tr, random_state=SEED)
    print(f"split: fit {len(fit_i)} / val {len(val_i)} / test {len(y_te)}\n")

    rows = []
    print("Kelompok A - embedding beku + classifier:")
    for tag, Xtr, Xte in (("CLIP", X_clip_tr, X_clip_te), ("DINOv2", X_dino_tr, X_dino_te)):
        run_probes(Xtr[fit_i], y_tr[fit_i], Xtr[val_i], y_tr[val_i], Xte, y_te, tag, rows)

    print("\nKelompok B - finetune CNN penuh:")
    for arch in ("EfficientNet-B0", "MobileNetV3-L", "MobileNetV2"):
        v, t = finetune(arch, fit_i, val_i)
        rows.append((f"{arch} finetune", v, t, "best-epoch@val"))
        print(f"  {arch:<16} val {v:5.2f}%  test {t:5.2f}%")

    n = len(y_te)
    print("\n" + "=" * 74)
    print("HASIL (peringkat menurut VAL -- TEST hanya dilaporkan, tidak dipakai memilih)")
    print("=" * 74)
    print(f"{'model':<28} {'val':>7} {'test':>7}  {'pilihan':<16}")
    for name, v, t, lbl in sorted(rows, key=lambda r: -r[1]):
        print(f"{name:<28} {v:6.2f}% {t:6.2f}%  {lbl:<16}")
    se = (0.82 * 0.18 / n) ** 0.5 * 100
    print("-" * 74)
    print(f"n_test={n}, standard error ~±{se:.1f}% -> selisih di bawah ~{2*se:.0f}% tidak berarti apa-apa.")
    pickle.dump(rows, open("models/compare_rows.pkl", "wb"))
    return rows


if __name__ == "__main__":
    main()

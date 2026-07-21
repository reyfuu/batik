"""Latih + evaluasi classifier motif batik di raw_batik_v2 (20 kelas Daerah_Motif).

Pipeline: DINOv2 (backbone beku) -> head MLP. Backbone tidak ikut dilatih -- terbukti
terbaik (src/compare_models.py) dan cepat.

WAJIB dedup dulu: raw_batik_v2 hasil scrape dan BOCOR train->test. Tanpa dedup, akurasi
test palsu tinggi. Dua tahap:
  1) buang file test identik byte-per-byte dgn train (md5)
  2) buang file test near-dupe (cosine >= 0.98 ke train mana pun)
"""
import os
import hashlib
import numpy as np
import torch
import torch.nn as nn

from src.dino import embed
from src.clip_search import get_device

DATA = "data/raw_batik_v2"
SEED = 42


def _list(split):
    """(paths, labels) untuk satu split, subfolder = kelas."""
    root = os.path.join(DATA, split)
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    paths, labels = [], []
    for c in classes:
        for f in sorted(os.listdir(os.path.join(root, c))):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(root, c, f))
                labels.append(c)
    return paths, np.array(labels), classes


def _md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def load_clean():
    """Embed train+test, buang kebocoran test, kembalikan array bersih + laporan dedup."""
    ptr, ytr, classes = _list("train")
    pte, yte, _ = _list("test")

    # Tahap 1: identik byte-per-byte (md5). Kebocoran tak terbantahkan.
    train_md5 = {_md5(p) for p in ptr}
    keep1 = np.array([_md5(p) not in train_md5 for p in pte])
    n_exact = int((~keep1).sum())

    Xtr = embed(ptr)
    Xte = embed(pte)

    # Tahap 2: near-dupe (cosine >= 0.98). DINOv2 L2-normalized -> dot = cosine.
    sim_max = (Xte @ Xtr.T).max(axis=1)
    keep2 = sim_max < 0.98
    keep = keep1 & keep2
    n_near = int((keep1 & ~keep2).sum())

    report = {"n_test_awal": len(pte), "n_exact": n_exact, "n_near": n_near,
              "n_test_bersih": int(keep.sum())}
    return Xtr, ytr, Xte[keep], yte[keep], classes, report


class Head(nn.Module):
    """MLP kecil di atas embedding beku: 768 -> 256 -> n_kelas."""
    def __init__(self, dim, n):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n))

    def forward(self, x):
        return self.net(x)


def _fit(Xtr_t, ytr_t, n_cls, dim, dev, epochs, Xval_t=None, yval_t=None, verbose=False):
    """Latih head. Ada val: kembalikan bobot epoch val-acc terbaik (early stop jujur,
    tanpa mengintip test). Tidak ada: bobot epoch terakhir."""
    torch.manual_seed(SEED)
    model = Head(dim, n_cls).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    bs = 32
    best_val, best_state, best_ep = -1.0, None, 0
    hist = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=dev)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            crit(model(Xtr_t[b]), ytr_t[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            tr = (model(Xtr_t).argmax(1) == ytr_t).float().mean().item() * 100
            val = ((model(Xval_t).argmax(1) == yval_t).float().mean().item() * 100
                   if Xval_t is not None else float("nan"))
        hist.append((ep + 1, tr, val))
        if verbose:
            print(f"Epoch [{ep+1:>2}/{epochs}]  train {tr:5.1f}%  val {val:5.1f}%")
        if Xval_t is not None and val > best_val:
            best_val, best_ep = val, ep + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist, best_ep, best_val


def train(epochs=40):
    from sklearn.model_selection import train_test_split
    dev = get_device()
    Xtr, ytr, Xte, yte, classes, rep = load_clean()
    cls_idx = {c: i for i, c in enumerate(classes)}
    ytr_i = np.array([cls_idx[c] for c in ytr])
    yte_i = np.array([cls_idx[c] for c in yte])

    # Val split dari TRAIN saja -> epoch dipilih di val, TEST diukur sekali.
    fit_i, val_i = train_test_split(np.arange(len(Xtr)), test_size=0.2,
                                    stratify=ytr_i, random_state=SEED)
    t = lambda a: torch.tensor(a, device=dev)
    Xf, yf = t(Xtr[fit_i]), t(ytr_i[fit_i])
    Xv, yv = t(Xtr[val_i]), t(ytr_i[val_i])
    Xte_t = t(Xte)

    print(f"raw_batik_v2: {len(fit_i)} fit / {len(val_i)} val / {len(yte)} test bersih, "
          f"{len(classes)} kelas")
    print(f"dedup: buang {rep['n_exact']} identik + {rep['n_near']} near-dupe "
          f"dari {rep['n_test_awal']} test\n")

    model, hist, best_ep, best_val = _fit(Xf, yf, len(classes), Xtr.shape[1], dev,
                                          epochs, Xv, yv, verbose=True)
    model.eval()
    with torch.no_grad():
        te_pred = model(Xte_t).argmax(1).cpu().numpy()
        tr_acc = (model(t(Xtr)).argmax(1).cpu().numpy() == ytr_i).mean() * 100
    te_acc = (te_pred == yte_i).mean() * 100
    print(f"\nEpoch terbaik di val: {best_ep} (val {best_val:.1f}%)")

    # Laporan akhir
    print("\n" + "=" * 60)
    print("LAPORAN AKHIR — DINOv2 + MLP head @ raw_batik_v2")
    print("=" * 60)
    final_te = te_acc
    n = len(yte_i)
    se = (final_te / 100 * (1 - final_te / 100) / n) ** 0.5 * 100
    print(f"Akurasi test (bersih) : {final_te:.2f}%  (n={n}, SE ±{se:.1f}%)")
    print(f"Akurasi train         : {tr_acc:.2f}%")

    print("\nAkurasi per kelas (urut terburuk):")
    rows = [(c, (te_pred[yte_i == cls_idx[c]] == cls_idx[c]).mean() * 100
             if (yte_i == cls_idx[c]).any() else float("nan"),
             int((yte_i == cls_idx[c]).sum())) for c in classes]
    for c, a, k in sorted(rows, key=lambda r: r[1]):
        print(f"  {c:<30} {a:5.1f}%  (n={k})")

    print("\nPasangan paling sering tertukar:")
    from collections import Counter
    errs = Counter((classes[t], classes[p]) for t, p in zip(yte_i, te_pred) if t != p)
    for (t, p), c in errs.most_common(8):
        print(f"  {t}  ->  {p}   ({c}x)")

    os.makedirs("models", exist_ok=True)
    torch.save({"state": model.state_dict(), "classes": classes, "dim": Xtr.shape[1]},
               "models/rawv2_head.pth")
    print(f"\nModel -> models/rawv2_head.pth")
    return te_acc, te_pred, yte_i, classes


def learning_curve(epochs=40):
    """Jawab 'butuh data lagi?' secara empiris: latih pakai 25/50/75/100% data
    (per-kelas, seimbang), pilih epoch di val, ukur test. Masih naik di 100% = data
    tambahan membantu. Datar = bottleneck bukan jumlah data."""
    from sklearn.model_selection import train_test_split
    dev = get_device()
    Xtr, ytr, Xte, yte, classes, _ = load_clean()
    cls_idx = {c: i for i, c in enumerate(classes)}
    ytr_i = np.array([cls_idx[c] for c in ytr])
    yte_i = np.array([cls_idx[c] for c in yte])
    fit_i, val_i = train_test_split(np.arange(len(Xtr)), test_size=0.2,
                                    stratify=ytr_i, random_state=SEED)
    t = lambda a: torch.tensor(a, device=dev)
    Xv, yv, Xte_t = t(Xtr[val_i]), t(ytr_i[val_i]), t(Xte)

    rng = np.random.default_rng(SEED)
    print("\n" + "=" * 60)
    print("KURVA BELAJAR — apakah menambah data akan menaikkan akurasi?")
    print("=" * 60)
    print(f"{'porsi data':>12} {'n train':>8} {'test acc':>10}")
    accs = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        # subsample seimbang per kelas dari fit_i
        sub = []
        for c in range(len(classes)):
            idx = fit_i[ytr_i[fit_i] == c]
            k = max(1, int(round(len(idx) * frac)))
            sub.extend(rng.choice(idx, k, replace=False))
        sub = np.array(sub)
        model, _, _, _ = _fit(t(Xtr[sub]), t(ytr_i[sub]), len(classes), Xtr.shape[1],
                              dev, epochs, Xv, yv)
        model.eval()
        with torch.no_grad():
            acc = (model(Xte_t).argmax(1).cpu().numpy() == yte_i).mean() * 100
        accs.append(acc)
        print(f"{frac*100:>11.0f}% {len(sub):>8} {acc:>9.1f}%")
    slope = accs[-1] - accs[-2]  # kenaikan dari 75%->100%
    print(f"\nKenaikan 75%->100% data: {slope:+.1f} poin")
    if slope >= 3:
        print("VERDICT: kurva masih menanjak -> menambah data KEMUNGKINAN membantu.")
    elif slope >= 1:
        print("VERDICT: kurva mulai landai -> data tambahan sedikit membantu, bukan prioritas.")
    else:
        print("VERDICT: kurva datar -> menambah data TIDAK akan banyak menolong;")
        print("         bottleneck-nya label/taksonomi, bukan jumlah data.")
    return accs


if __name__ == "__main__":
    train(epochs=40)
    learning_curve(epochs=40)

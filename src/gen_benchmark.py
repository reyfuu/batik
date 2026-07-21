"""Benchmark 4 image libraries generating a batik PNG from an SVM result.

Mirrors src/benchmark.py: each library turns a motif label (pred_idx from SVM)
into a 512x512 batik-style pattern and saves a .png. We time the full
generate->encode->save workload, print a table, and plot a comparison chart.

Libraries (all already installed): Pillow, OpenCV, scikit-image, matplotlib.
Run: python src/gen_benchmark.py
"""
import os
import sys
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

SIZE = 512
MOTIFS = ["parang", "kawung", "megamendung"]  # pred_idx 0,1,2 from SVM
BG = (245, 239, 230)      # cream
FG = (140, 98, 57)        # batik brown
OUT_IMG = "public/img"
SAMPLE_DIR = os.path.join(OUT_IMG, "gen_samples")


# ---------- 2. Pillow / PIL ----------
def gen_pillow(pred_idx, path):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (SIZE, SIZE), BG)
    d = ImageDraw.Draw(img)
    if pred_idx == 0:  # parang: diagonal stripes
        for off in range(-SIZE, SIZE, 40):
            d.line([(off, 0), (off + SIZE, SIZE)], fill=FG, width=14)
    elif pred_idx == 1:  # kawung: grid of circles
        for y in range(0, SIZE + 60, 60):
            for x in range(0, SIZE + 60, 60):
                d.ellipse([x - 26, y - 26, x + 26, y + 26], outline=FG, width=6)
    else:  # megamendung: concentric arcs
        for cx in range(60, SIZE, 130):
            for r in range(20, 90, 18):
                d.arc([cx - r, 200 - r, cx + r, 200 + r], 180, 360, fill=FG, width=6)
    img.save(path)


# ---------- 3. OpenCV (cv2) ----------
def gen_opencv(pred_idx, path):
    import cv2
    img = np.full((SIZE, SIZE, 3), BG[::-1], dtype=np.uint8)  # BGR
    c = FG[::-1]
    if pred_idx == 0:
        for off in range(-SIZE, SIZE, 40):
            cv2.line(img, (off, 0), (off + SIZE, SIZE), c, 14)
    elif pred_idx == 1:
        for y in range(0, SIZE + 60, 60):
            for x in range(0, SIZE + 60, 60):
                cv2.circle(img, (x, y), 26, c, 6)
    else:
        for cx in range(60, SIZE, 130):
            for r in range(20, 90, 18):
                cv2.ellipse(img, (cx, 200), (r, r), 0, 180, 360, c, 6)
    cv2.imwrite(path, img)


# ---------- 4. scikit-image ----------
def gen_skimage(pred_idx, path):
    from skimage import draw, io
    img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    img[:] = BG
    def stroke(rr, cc):
        rr = np.clip(rr, 0, SIZE - 1); cc = np.clip(cc, 0, SIZE - 1)
        img[rr, cc] = FG
    if pred_idx == 0:
        for off in range(-SIZE, SIZE, 40):
            rr, cc = draw.line(0, max(off, 0), min(SIZE - 1, SIZE - 1), min(off + SIZE, SIZE - 1))
            stroke(rr, cc)
    elif pred_idx == 1:
        for y in range(0, SIZE, 60):
            for x in range(0, SIZE, 60):
                rr, cc = draw.circle_perimeter(y, x, 26, shape=img.shape)
                stroke(rr, cc)
    else:
        for cx in range(60, SIZE, 130):
            for r in range(20, 90, 18):
                rr, cc = draw.circle_perimeter(200, cx, r, shape=img.shape)
                keep = rr >= 200  # lower half = arc
                stroke(rr[keep], cc[keep])
    io.imsave(path, img)


# ---------- 5. matplotlib ----------
def gen_matplotlib(pred_idx, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Arc
    fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)
    ax.set_xlim(0, SIZE); ax.set_ylim(0, SIZE); ax.axis('off')
    fig.patch.set_facecolor(np.array(BG) / 255)
    ax.set_facecolor(np.array(BG) / 255)
    col = np.array(FG) / 255
    if pred_idx == 0:
        for off in range(-SIZE, SIZE, 40):
            ax.plot([off, off + SIZE], [0, SIZE], color=col, lw=5)
    elif pred_idx == 1:
        for y in range(0, SIZE + 60, 60):
            for x in range(0, SIZE + 60, 60):
                ax.add_patch(Circle((x, y), 26, fill=False, ec=col, lw=2))
    else:
        for cx in range(60, SIZE, 130):
            for r in range(20, 90, 18):
                ax.add_patch(Arc((cx, SIZE - 200), 2 * r, 2 * r, theta1=180, theta2=360, ec=col, lw=2))
    plt.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path); plt.close(fig)


LIBS = {
    "Pillow (PIL)": gen_pillow,
    "OpenCV (cv2)": gen_opencv,
    "scikit-image": gen_skimage,
    "matplotlib": gen_matplotlib,
}


def benchmark_lib(name, fn, repeats=10):
    """Warmup once, then time generate+save over all motifs x repeats."""
    tmp = os.path.join(SAMPLE_DIR, "_warmup.png")
    fn(0, tmp)  # warmup (import + first-call cost excluded)
    latencies = []
    for _ in range(repeats):
        for idx in range(3):
            t0 = time.perf_counter()
            fn(idx, os.path.join(SAMPLE_DIR, "_bench.png"))
            latencies.append((time.perf_counter() - t0) * 1000)
    # save one clean sample per motif for inspection
    for idx, m in enumerate(MOTIFS):
        fn(idx, os.path.join(SAMPLE_DIR, f"{name.split()[0].lower()}_{m}.png"))
    return float(np.mean(latencies))


def run_benchmark():
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    print(f"Benchmark generasi batik PNG ({SIZE}x{SIZE}) dari label SVM...")
    print(f"Setiap library generate 3 motif x 10 ulangan.\n")

    results = {}
    for i, (name, fn) in enumerate(LIBS.items(), 1):
        print(f"[{i}/4] Menguji {name}...")
        results[name] = benchmark_lib(name, fn)

    print("\n" + "=" * 60)
    print("     BENCHMARK KECEPATAN GENERASI GAMBAR (per PNG)")
    print("=" * 60)
    for name, lat in sorted(results.items(), key=lambda x: x[1]):
        print(f"- {name:<20}: {lat:>8.2f} ms")
    print("=" * 60)

    # --- chart ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(results.keys())
    lats = list(results.values())
    colors = ['#8C6239', '#C29B70', '#4A6B82', '#A3B5C3']
    plt.figure(figsize=(9, 6))
    bars = plt.bar(names, lats, color=colors, edgecolor='black', width=0.6)
    plt.title('Benchmark Kecepatan Generasi Batik PNG dari Hasil SVM\n(Lebih Pendek = Lebih Cepat)',
              fontweight='bold', pad=15)
    plt.ylabel('Waktu Generasi per PNG (milidetik - ms)')
    plt.xlabel('Library')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for b in bars:
        plt.annotate(f'{b.get_height():.2f} ms', xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold', fontsize=9)
    plt.tight_layout()
    chart = os.path.join(OUT_IMG, "gen_benchmark_speed.png")
    plt.savefig(chart, dpi=150); plt.close()
    print(f"\nGrafik disimpan ke: {chart}")
    print(f"Sampel gambar per motif ada di: {SAMPLE_DIR}/")
    return results


if __name__ == "__main__":
    run_benchmark()

"""Flask backend untuk Batik Motif Finder (DINOv2 + MLP probe).

Dua fitur:
  1) IDENTIFIKASI (POST /api/identify): gambar -> {corak, warna, motif} + contoh.
       - motif : DINOv2 + probe MLP, DATASETv7 (15 kelas), 87.00% test (GLCM+LBP+SVM cuma 42.00%).
       - warna : warna dominan dari pixel (HSV, bukan ML).
       - corak : dari tabel per-motif (CORAK).
  2) GENERATE (POST /api/generate): prompt -> batik via SD-Turbo, simpan ke public/generated/.

Jalankan:  .venv/bin/python web/app.py     (default http://127.0.0.1:8791)
Ganti port: PORT=9000 .venv/bin/python web/app.py
"""
import os
import sys
import uuid
import subprocess
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_file, send_from_directory
from dotenv import load_dotenv

# ROOT (folder di atas web/) masuk sys.path supaya `import src.*` jalan.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

# Konvensi Aiprenur: key di ~/api/.env. Project .env (kalau ada) menimpa nilai yang sama.
load_dotenv(os.path.expanduser("~/api/.env"))
load_dotenv(os.path.join(ROOT, ".env"), override=True)

from PIL import Image

# Probe DINOv2 (87.00% test) menggantikan GLCM+LBP+SVM (42.00%); detail di src/compare_models.py.
from src.dino import embed, load_probe

PUBLIC = os.path.join(ROOT, "public")
REF_DIR = os.path.join(ROOT, "data", "Contoh foto batik", "DATASETv7", "TRAIN")  # sumber training + gambar contoh
RAWV2_DIR = os.path.join(ROOT, "data", "raw_batik_v2", "train")                  # dataset kedua untuk katalog
MODELS = os.path.join(ROOT, "models")
GEN_DIR = os.path.join(PUBLIC, "generated")
os.makedirs(GEN_DIR, exist_ok=True)

# Corak per motif — isi field 'corak' hasil identifikasi (probe tak keluarkan corak langsung).
CORAK = {
    "Batik Bali": "Ornamen flora-fauna",
    "Batik Betawi": "Flora & ondel-ondel",
    "Batik Cendrawasih": "Flora-fauna (burung)",
    "Batik Dayak": "Etnik geometris",
    "Batik Geblek Renteng": "Geometris berulang",
    "Batik Ikat Celup": "Tie-dye / abstrak",
    "Batik Insang": "Geometris (insang ikan)",
    "Batik Kawung": "Geometris (irisan buah aren)",
    "Batik Lasem": "Flora pesisir",
    "Batik Megamendung": "Awan melengkung",
    "Batik Pala": "Flora (buah pala)",
    "Batik Parang": "Lereng diagonal",
    "Batik Poleng": "Kotak-kotak (catur)",
    "Batik Sekar Jagad": "Peta / patchwork",
    "Batik Tambal": "Tambal / patchwork",
}


def dominant_color(bgr_img):
    """Tebak warna dominan dari gambar BGR via rata-rata HSV.

    ponytail: heuristik kasar (rata-rata), cukup untuk label. Ganti k-means kalau perlu presisi.
    """
    small = cv2.resize(bgr_img, (64, 64))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = hsv[:, 0].mean(), hsv[:, 1].mean(), hsv[:, 2].mean()

    # Value rendah -> hitam; saturasi rendah -> putih/abu.
    if v < 50:
        return "Hitam"
    if s < 40:
        return "Putih" if v > 180 else "Abu-abu"

    # Hue di OpenCV rentang 0..179.
    hue = h
    if hue < 10 or hue >= 170:
        return "Merah"
    if hue < 20:
        return "Coklat" if v < 160 else "Oranye"   # oranye kalau terang, coklat kalau gelap
    if hue < 33:
        return "Kuning"
    if hue < 78:
        return "Hijau"
    if hue < 100:
        return "Cyan"
    if hue < 131:
        return "Biru"
    if hue < 155:
        return "Ungu"
    return "Merah muda"


# Probe dimuat sekali saat start (backbone di-cache di src.dino._load() saat embed pertama).
PROBE = load_probe(MODELS)
print(f"[DINOv2] Siap: {len(PROBE.classes_)} kelas")

# static_url_path="" -> file public/ dilayani di root (mis. /css/tailwind.css, /generated/xxx.png).
app = Flask(__name__, static_folder=PUBLIC, static_url_path="")


@app.after_request
def cors(resp):
    """Header CORS di semua response, supaya frontend origin lain (mis. Live Server) bisa memanggil.

    ponytail: izinkan semua origin — cukup untuk lokal/dev.
    """
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/")
def home():
    """Halaman utama = frontend (public/index.html)."""
    return send_from_directory(PUBLIC, "index.html")


@app.route("/ref/<path:rel>")
def ref_image(rel):
    """Serve gambar referensi/contoh dari folder TRAIN (mis. /ref/Batik Kawung/1.jpg)."""
    return send_file(os.path.join(REF_DIR, rel))


@app.route("/rawv2/<path:rel>")
def rawv2_image(rel):
    """Serve gambar raw_batik_v2 (mis. /rawv2/Yogyakarta_Kawung/150001.jpg)."""
    return send_file(os.path.join(RAWV2_DIR, rel))


# Nama daerah di raw_batik_v2 mengandung underscore ("Sumatera_Barat_Rumah_Minang"),
# jadi tidak bisa dipecah di underscore terakhir — cocokkan dengan daftar ini dulu.
REGIONS = ["Sumatera_Barat", "Sumatera_Utara", "Sulawesi_Selatan", "Jawa_Barat", "Jawa_Timur",
           "Yogyakarta", "Kalimantan", "Lampung", "Madura", "Maluku", "Papua", "Solo",
           "Bali", "Aceh", "DKI", "NTB"]


def _split_rawv2(cls):
    """'Yogyakarta_Kawung' -> ('Yogyakarta', 'Kawung'). Return (None, cls) kalau tak cocok."""
    for r in sorted(REGIONS, key=len, reverse=True):
        if cls.startswith(r + "_"):
            return r.replace("_", " "), cls[len(r) + 1:].replace("_", " ")
    return None, cls.replace("_", " ")


def examples_for(cls, n=6):
    """Ambil sampai n gambar contoh dari folder motif hasil prediksi (untuk grid)."""
    cdir = os.path.join(REF_DIR, cls)
    if not os.path.isdir(cdir):
        return []
    files = [f for f in sorted(os.listdir(cdir))
             if f.lower().endswith((".jpg", ".jpeg", ".png"))][:n]
    # URL web (route /ref/...), bukan path filesystem.
    return [{"name": f, "image_url": f"/ref/{cls}/{f}"} for f in files]


def _scan(root, url_prefix, source):
    """Baca folder dataset -> list (motif, daerah, nama_file, url). Subfolder = kelas."""
    out = []
    if not os.path.isdir(root):
        return out
    for cls in sorted(os.listdir(root)):
        cdir = os.path.join(root, cls)
        if not os.path.isdir(cdir):
            continue
        if source == "rawv2":
            region, motif = _split_rawv2(cls)
        else:
            motif = cls[len("Batik "):] if cls.startswith("Batik ") else cls
            region = None  # nama kelas DATASETv7 campur daerah & motif, jadi tak dipisah
        for f in sorted(os.listdir(cdir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                out.append((motif, region, f, f"{url_prefix}/{cls}/{f}"))
    return out


@app.route("/api/collection")
def collection():
    """Katalog motif gabungan DATASETv7 + raw_batik_v2, di-dedup per nama motif.

    Motif yang sama muncul di dua dataset (Kawung, Parang, Megamendung, Cendrawasih,
    Dayak, Pala) digabung jadi satu entri; daerah asalnya dikumpulkan ke `regions`.
    Dibaca langsung dari folder, jadi selalu sinkron dengan isi dataset.
    `per` = maksimum preview per motif (default 12).

    CATATAN: dataset TIDAK punya data perancang/studio, jadi tidak ada field `designer` —
    subtitle kartu memakai daerah asal. Jangan mengarang nama studio.
    """
    per = max(1, min(int(request.args.get("per", 12)), 70))
    rows = (_scan(REF_DIR, "/ref", "v7") + _scan(RAWV2_DIR, "/rawv2", "rawv2"))

    # Kelompokkan per motif, lalu per sumber (daerah / DATASETv7).
    motifs = {}
    for motif, region, fname, url in rows:
        e = motifs.setdefault(motif, {"motif": motif, "regions": [], "count": 0, "by_src": {}})
        e["count"] += 1
        if region and region not in e["regions"]:
            e["regions"].append(region)
        e["by_src"].setdefault(region or "DATASETv7", []).append(
            {"name": fname, "image_url": url, "region": region})

    items = []
    for m in sorted(motifs.values(), key=lambda x: x["motif"]):
        # Preview diambil bergiliran antar sumber, bukan berurutan — kalau berurutan,
        # motif gabungan (mis. Parang: v7 + Solo + Yogyakarta) previewnya habis oleh
        # sumber pertama saja dan katalog tidak mewakili gabungannya.
        buckets = list(m.pop("by_src").values())
        picked = []
        for i in range(max(len(b) for b in buckets)):
            for b in buckets:
                if i < len(b) and len(picked) < per:
                    picked.append(b[i])
            if len(picked) >= per:
                break
        m["images"] = picked
        m["corak"] = CORAK.get("Batik " + m["motif"], CORAK.get(m["motif"], "Tekstur batik"))
        items.append(m)
    return jsonify(total=sum(i["count"] for i in items), motifs=items)


@app.route("/api/identify", methods=["POST", "OPTIONS"])
def identify():
    """Identifikasi kategori {corak, warna, motif} dari gambar yang diupload."""
    if request.method == "OPTIONS":
        return ("", 204)                          # preflight CORS

    file = request.files.get("image")
    if file is None:
        return jsonify(error="field 'image' wajib ada"), 400

    data = np.frombuffer(file.stream.read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify(error="gagal baca gambar"), 400

    # cv2 baca BGR, DINOv2 dilatih RGB -> konversi WAJIB. Tanpa ini kanal merah/biru
    # tertukar: tanpa error, prediksinya saja yang diam-diam ngawur.
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    vec = embed([pil])                                      # (1, 768) L2-normalized
    motif = str(PROBE.predict(vec)[0])
    proba = PROBE.predict_proba(vec)[0]
    conf = float(proba[list(PROBE.classes_).index(motif)])

    return jsonify(
        classifier="dinov2+mlp",
        source={"motif_name": file.filename},
        kategori={
            "corak": CORAK.get(motif, "Tekstur batik"),   # fallback kalau motif tak ada di tabel
            "warna": dominant_color(bgr),
            "motif": motif,
        },
        confidence=round(conf, 4) if conf is not None else None,
        examples=examples_for(motif),
    )


@app.route("/api/generate", methods=["POST", "OPTIONS"])
def generate():
    """Generate variasi batik: gambar upload + prompt -> SD-Turbo img2img LOKAL.

    Butuh field 'image' (referensi) + 'prompt'. Load pertama ~10 dtk, lalu ~2 dtk/gambar.
    'count' opsional (default 5).
    """
    if request.method == "OPTIONS":
        return ("", 204)

    file = request.files.get("image")
    if file is None:
        return jsonify(error="upload gambar batik dulu (field 'image')"), 400
    prompt = (request.form.get("prompt") or "").strip()
    if not prompt:
        return jsonify(error="prompt kosong"), 400
    # Batasi jumlah per klik biar browser tidak menggantung. 300 lewat CLI, bukan web.
    count = max(1, min(int(request.form.get("count", 5)), 20))

    data = file.stream.read()
    if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is None:
        return jsonify(error="gagal baca gambar"), 400

    # Simpan potongan acuan yang dipakai model supaya user tahu motif bagian mana yang
    # jadi sumber. Aman di proses server: _texture_patch cuma butuh PIL/numpy, torch dan
    # diffusers baru di-import saat pipeline dimuat (di subprocess).
    import io
    from src.gen_local import _texture_patch
    patch_name = "_patch_" + uuid.uuid4().hex[:8] + ".png"
    _texture_patch(Image.open(io.BytesIO(data)).convert("RGB")).save(
        os.path.join(GEN_DIR, patch_name))

    # SUBPROCESS terpisah: SD-Turbo butuh MPS sendiri; satu proses dengan DINOv2 (sudah
    # dimuat di server) bikin VAE fp16 keluar gambar hitam. Subprocess = standalone yang
    # stabil, plus memorinya bebas setelah selesai (penting di 8GB).
    tmp = os.path.join(GEN_DIR, f"_ref_{os.getpid()}_{len(data)}.png")
    with open(tmp, "wb") as f:
        f.write(data)
    before = set(os.listdir(GEN_DIR))
    try:
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "gen_batik_local.py"),
             tmp, prompt, str(count)],
            cwd=ROOT, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            return jsonify(error=f"gagal generate: {r.stderr.strip()[-200:]}"), 500
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    new = [f for f in os.listdir(GEN_DIR) if f not in before and f.startswith("batik_")]
    images = ["/generated/" + n for n in sorted(new)]
    return jsonify(model="sd-turbo", prompt=prompt, images=images,
                   patch_url="/generated/" + patch_name)


if __name__ == "__main__":
    # Port default 8791 (jarang bentrok); override: PORT=9000 .venv/bin/python web/app.py
    port = int(os.environ.get("PORT", 8791))
    app.run(host="127.0.0.1", port=port, debug=False)

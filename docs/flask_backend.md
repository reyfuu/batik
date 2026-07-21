# Backend Flask — Batik Motif Finder

## Apa itu Flask?

**Flask** adalah *web framework* mikro untuk Python. Tugasnya: menerima
permintaan HTTP dari browser (mis. upload gambar), menjalankan kode Python
(di sini: model CLIP), lalu mengirim balik respons (JSON hasil pencarian).

Disebut "mikro" karena inti-nya ringan — cuma routing + request/response.
Tidak memaksa struktur folder atau database seperti framework besar (Django,
Laravel). Cocok untuk 1–beberapa endpoint seperti kasus kita.

Kenapa Flask di sini, bukan Express/Laravel? Karena modelnya **Python**
(scikit-learn SVM + OpenCV/skimage untuk fitur GLCM+LBP). Flask jalan di bahasa
yang sama dengan model, jadi tidak perlu jembatan antar bahasa. Express (Node)
atau Laravel (PHP) tetap harus memanggil Python — 1 lompatan ekstra yang tidak perlu.

## Alur kerja

```
Browser (public/index.html)
   │  upload .png/.jpeg  (POST /api/identify)
   ▼
Flask (web/app.py)
   │  1. ekstrak fitur GLCM + LBP (grayscale)
   │  2. SVM prediksi MOTIF (15 kelas DATASETv7)
   │  3. WARNA = warna dominan dari pixel (HSV, bukan model)
   │  4. CORAK = tabel per-motif
   ▼
JSON { kategori:{corak,warna,motif}, confidence, examples[] }
   -> kartu Kategori + grid gambar contoh dari motif
```

Model SVM dilatih **sekali** saat pertama start (dari `DATASETv7/TRAIN`), lalu
di-cache ke `models/svm_batik.pkl` (+ scaler & daftar kelas). Start berikutnya
langsung baca cache.

> Akurasi SVM GLCM+LBP saat ini **±42%** pada 15 kelas (data uji DATASETv7).
> Menengah — cukup untuk demo, belum untuk produksi.

## Cara menjalankan

Dari root project (`/Users/wrapxbake/batik`):

```bash
# 1. Pastikan dependency ada (sekali saja)
.venv/bin/pip install -r requirements.txt

# 2. Jalankan server
.venv/bin/python web/app.py
```

Start pertama akan melatih SVM dari 1050 gambar TRAIN (~beberapa detik).
Start berikutnya langsung jalan karena pakai model tersimpan.

Setelah muncul `Running on http://127.0.0.1:8080`, buka di browser:

```
http://127.0.0.1:8080
```

Upload gambar batik (.png / .jpeg) → klik **Identifikasi Motif** → muncul 10
motif termirip beserta `distance` (makin kecil makin mirip).

## Pakai Live Server (opsional)

Bisa. Buka `public/index.html` lewat Live Server untuk frontend, sementara
Flask jalan terpisah di port 8000. CORS sudah diaktifkan di `web/app.py`, jadi
frontend beda origin tetap bisa memanggil backend.

> Catatan: kalau Flask **tidak** jalan, halaman otomatis jatuh ke **mode demo**
> (hasil placeholder). Untuk memproses gambar sungguhan, Flask harus hidup.

## Endpoint

| Method | Path | Fungsi |
|---|---|---|
| GET | `/` | Halaman frontend |
| GET | `/ref/<kelas>/<file>` | Serve gambar contoh motif |
| POST | `/api/identify` | Terima field `image`, balikan JSON kategori {corak, warna, motif} |
| POST | `/api/generate` | Terima `{prompt}`, generate gambar via Replicate (Flux), simpan ke `public/generated/`, balikan `{images:[url]}` |

### Generate gambar (uji API)

Butuh token Replicate (gratis saat daftar):

```bash
export REPLICATE_API_TOKEN=r8_xxxxx
.venv/bin/python web/app.py
```

Tanpa token, endpoint balas 503 dengan pesan jelas. Hasil generate tersimpan di
`public/generated/` dan tampil di panel "Generate Batik" pada halaman.

Contoh uji lewat terminal:

```bash
curl -s -X POST \
  -F "image=@data/Contoh foto batik/DATASETv7/TEST/Batik Kawung/70.jpg" \
  http://127.0.0.1:8080/api/identify
```

## Ganti port / hentikan

- Ganti port: set env `PORT` (default 8080), mis. `PORT=9000 .venv/bin/python web/app.py`.
  Kalau ubah port, sesuaikan juga `API_ENDPOINT` di `public/index.html`.
- Latih ulang model (mis. setelah data TRAIN berubah): hapus `models/svm_batik.pkl`,
  `models/scaler_batik.pkl`, `models/classes.pkl`, lalu start ulang.
- Hentikan server: `Ctrl+C` di terminal, atau `pkill -f "web/app.py"`.

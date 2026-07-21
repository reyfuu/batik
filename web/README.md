# Menjalankan Web Batik Motif (Flask)

Backend Flask untuk **identifikasi** (upload batik → motif/warna/corak) dan
**generate** (upload batik + prompt → 5 variasi batik dibuat lokal).

## Prasyarat (sekali saja)

- Sudah ada virtualenv di `../.venv` (Python 3.14) dengan `flask`, `torch`,
  `diffusers`, `transformers`, `opencv-python`.
- Koneksi internet **saat pertama** jalan: model DINOv2 (~350 MB) dan SD-Turbo
  (~2.5 GB) diunduh sekali dari HuggingFace, lalu tersimpan di cache dan jalan offline.

## Menjalankan

Dari folder project (`/Users/wrapxbake/batik`):

```bash
.venv/bin/python web/app.py
```

Lalu buka di browser: **http://127.0.0.1:8791**

Ganti port kalau 8791 dipakai:

```bash
PORT=9000 .venv/bin/python web/app.py
```

Hentikan server: `Ctrl+C` di terminal yang sama.

> Tidak perlu `source .venv/bin/activate` — memanggil `.venv/bin/python`
> langsung sudah memakai virtualenv-nya. `app.py` juga menambah root project ke
> `sys.path` sendiri, jadi bisa dijalankan dari folder mana pun dengan path absolut:
> `/Users/wrapxbake/batik/.venv/bin/python /Users/wrapxbake/batik/web/app.py`
>
> .venv/bin/python web/app.py

## Cara pakai di browser

1. **Upload batik** di dropzone atas (drag-drop atau klik). Muncul hasil
   identifikasi: motif, warna, corak + contoh gambar.
2. **Generate**: dengan batik yang sama masih ter-upload, ketik deskripsi di kotak
   Generate lalu klik **Generate** → 5 variasi batik dari acuan Anda muncul di halaman.
   Hasil tersimpan di `public/generated/`.

## Yang perlu diketahui

- **Generate pertama ~20–30 detik** (model SD-Turbo dimuat di proses terpisah),
  berikutnya lebih cepat. Model generate sengaja dijalankan sebagai **subprocess**
  terpisah dari server: di Mac 8 GB, DINOv2 (identify) dan SD-Turbo (generate) tidak
  boleh berebut memori GPU bersamaan — kalau dipaksa, hasil generate jadi gambar hitam.
- **Batas 5–20 gambar per klik** di web (biar browser tidak menggantung). Untuk
  batch besar (mis. 300) pakai CLI, bukan web:
  ```bash
  .venv/bin/python scripts/gen_batik_local.py "<batik.jpg>" "<prompt>" 300
  ```
- **Endpoint** (kalau mau tes tanpa browser):
  - `POST /api/identify` — field `image` (multipart) → JSON motif/warna/corak.
  - `POST /api/generate` — field `image` + `prompt` (+ `count` opsional) → JSON `images: [url]`.

## Kalau bermasalah

- **Generate hasil kosong / gambar hitam** — sudah ditangani lewat subprocess; kalau
  muncul lagi, kemungkinan memori penuh. Tutup aplikasi lain, atau kecilkan `count`.
- **Port sudah dipakai** — `PORT=9000 .venv/bin/python web/app.py`.
- **Model lama diunduh ulang** — pastikan cache `~/.cache/huggingface` tidak terhapus.

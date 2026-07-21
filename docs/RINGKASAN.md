# Ringkasan Proyek Batik

Status per 2026-07-20.

## 1. Model klasifikasi — 11 pendekatan diadu

Protokol: split validation dari TRAIN, hyperparameter & epoch dipilih di situ, TEST disentuh sekali.

| Model | Akurasi TEST |
|---|---|
| DINOv2 + MLP (dipakai) | 87.00% (v7) · 88.89% (raw_batik_v2) |
| CLIP + linear probe | 81.33% |
| MobileNetV2 / V3 / EfficientNet-B0 finetune | 78–80% |
| GLCM + LBP + SVM (baseline lama) | 42.00% |

Temuan utama: yang menentukan adalah **embedding**-nya, bukan classifier-nya. Empat classifier di atas DINOv2 cuma berbeda 85.7–88.1% (dalam noise). DINOv2 unggul 7–8% dari CLIP di ukuran model yang sama — DINOv2 self-supervised dari gambar (kuat untuk tekstur), CLIP dilatih untuk teks.

Backbone beku + classifier 1 detik di CPU mengalahkan semua finetune GPU.

## 2. Masalah data

- **Taksonomi campur aduk** — DATASETv7 mencampur label daerah (Bali, Betawi, Lasem) dan motif (Parang, Kawung). Ada pasangan yang tidak saling eksklusif: batik Bali bermotif burung sah disebut "Bali" dan "Cendrawasih". Batik Bali jadi kelas terburuk (60%).
- **Kebocoran train→test di kedua dataset** — raw_batik_v2 punya 7 file identik byte-per-byte. Setelah dibersihkan, v7 turun dari 87% ke ~85–86%.
- **Semua dataset hasil scrape, bukan foto pabrik** — 378–587 resolusi berbeda, ukuran thumbnail Google, watermark artikel tercetak di gambar.
- **"data pabrik" bukan data pabrik** — 1033 gambar unduhan web, 20 kode huruf tanpa legenda. 6 kode terbaca otomatis (k=Megamendung 92%, h=Kawung 73%, L=Parang 65%…); 11 kelompok tidak cocok kelas mana pun = bahan ekspansi yang sudah ada di disk.
- **Kebutuhan data** — kurva belajar (3 seed): 25%→70.6%, 50%→84.5%, 75%→88.2%, 100%→90.5%. Masih naik tipis, diminishing return. Yang perlu diminta ke perusahaan bukan lebih banyak gambar internet, tapi **foto produk asli** — satu-satunya distribusi yang belum pernah diuji.

## 3. Web app (Flask, port 8791)

- **Katalog Batik** — 28 motif unik / 1690 gambar, gabungan dua dataset dengan dedup (Parang 134 gambar dari 3 sumber), auto-load, preview bergiliran antar sumber.
- **Generate Batik** — upload acuan + prompt → Pratinjau (1 gambar + potongan acuan) → Generate (5 variasi).
- Dropzone identifikasi sudah dihapus; endpoint `/api/identify` tetap hidup di backend.

## 4. Generate gambar

Berjalan lokal & gratis di Mac 8GB, setelah beberapa kegagalan yang informatif:

- SD1.5 float32 → ~5,5 jam/gambar (RAM habis, swap). Mati.
- SD-Turbo float16 + slicing → ~2 detik/gambar. 300 gambar ≈ 10 menit.
- Gambar hitam di server (DINOv2 + SD-Turbo rebutan MPS) → diperbaiki: generate dijalankan di subprocess terpisah.
- Hasil aneh (orang/pintu/poster) → akarnya img2img mewarisi komposisi acuan, bukan model kurang latih. Diperbaiki dengan memilih patch paling bertekstur (zoom 0.25) + strength 0.75.

Kenapa tidak dilatih: LoRA tidak akan memperbaiki cacat itu (masalahnya komposisi input, bukan pengetahuan batik), dan tidak muat di RAM 8GB.

## 5. Yang masih terbuka

- **Kontrol prompt lemah** — SD-Turbo jalan di `guidance_scale=0`, prompt nyaris tidak menyetir. Perlu SD biasa + CFG (lambat) atau cloud SDXL (~$1–3 untuk 300).
- Dua riset berhenti (jumlah motif batik Indonesia; taksonomi corak kain non-batik) — bisa di-resume dari cache.
- Spec similarity search manager menunggu metadata `id`/`designer`/`image_url` dari perusahaan. Katalog sengaja pakai daerah asal sebagai subtitle — nama studio tidak dikarang.
- **Sebelum angka masuk skripsi**: dedup md5 dulu, dan catat bahwa `train_v7()` memilih best-epoch pakai TEST (bias optimis).

# GLCM, LBP, dan SVM dari Dasar — dan Kenapa Lebih Kencang dari CLIP ViT-B/32

Dokumen ini menjelaskan tiga komponen pipeline tekstur tradisional **dari nol**, lalu
membedah **kenapa pipeline ini ~10× lebih cepat** daripada grouping berbasis CLIP ViT-B/32.

Untuk penjelasan hasil per motif (Parang/Kawung/Megamendung), lihat
[`glcm_lbp_svm.md`](glcm_lbp_svm.md). Dokumen ini soal **konsep + biaya komputasi**.

Kode terkait:
- [`src/glcm_svm.py`](../src/glcm_svm.py) — GLCM + gabung fitur + SVM/KNN
- [`src/lbp_extractor.py`](../src/lbp_extractor.py) — LBP
- [`src/benchmark.py`](../src/benchmark.py) — pengukuran kecepatan

---

## 1. GLCM — Gray-Level Co-occurrence Matrix

### Ide dasar
Tekstur = pola **hubungan antar piksel bertetangga**, bukan warna satu piksel. GLCM
menghitung: *"seberapa sering piksel dengan tingkat abu `i` muncul bersebelahan dengan
piksel tingkat abu `j`?"*

### Langkah
1. **Grayscale + kuantisasi.** Gambar diubah ke abu-abu, lalu 256 level dipadatkan jadi
   **32 level** (`nilai // 8`). Tujuannya: matriks jadi 32×32 (ringkas), bukan 256×256.
2. **Hitung ko-okurensi.** Untuk sebuah arah (misal "tetangga kanan, jarak 1"), telusuri
   seluruh gambar dan hitung tiap pasangan `(i, j)`. Hasilnya matriks 32×32.
3. **Ulang untuk beberapa jarak & sudut.** Di kode: jarak `[1, 2, 4]` × sudut
   `[0°, 45°, 90°, 135°]` = **12 matriks**.
4. **Ambil ringkasan statistik.** Dari tiap matriks diambil 6 angka:

   | Properti | Menangkap |
   | --- | --- |
   | `contrast` | perbedaan abu antar tetangga (kasar vs halus) |
   | `dissimilarity` | mirip contrast tapi linear |
   | `homogeneity` | seberapa "rata" teksturnya |
   | `energy` / `ASM` | keteraturan / pengulangan pola |
   | `correlation` | ketergantungan linear antar tetangga |

   **6 properti × 12 matriks = 72 angka.**

Intinya: GLCM mengubah gambar jadi **72 angka statistik**. Operasinya cuma
menghitung dan menjumlah — tidak ada perkalian matriks besar.

---

## 2. LBP — Local Binary Pattern

### Ide dasar
LBP menandai **bentuk lokal** di sekitar tiap piksel: apakah tetangganya lebih terang
atau lebih gelap dari titik tengah.

### Langkah
1. Untuk tiap piksel, lihat **8 tetangga** (radius 1).
2. Bandingkan tiap tetangga dengan piksel tengah → tulis `1` kalau ≥ tengah, `0` kalau
   kurang. Susun jadi angka biner 8-bit → satu kode LBP per piksel.
3. Mode `uniform`: pola dikelompokkan (garis, sudut, titik, tepi) supaya tahan rotasi dan
   jumlah kategori kecil.
4. Hitung **histogram** frekuensi tiap kode → ~10 angka. Itulah fitur LBP.

Hasil: **~10 angka** yang meringkas "distribusi micro-pattern" di gambar. Sekali sapu
piksel, tanpa operasi berat.

> Catatan: panjang histogram LBP di kode saat ini bisa bergeser 1 bin antar gambar
> (`n_bins = lbp.max()+1`). Idealnya dipatok `P+2 = 10` bin tetap. Lihat
> [`src/lbp_extractor.py`](../src/lbp_extractor.py#L11).

### Gabungan
```
72 (GLCM) + 10 (LBP) = 82 angka per gambar
```
Vektor 82-dimensi inilah "sidik jari tekstur" yang dipakai classifier.

---

## 3. SVM — Support Vector Machine

### Ide dasar
Setelah tiap gambar jadi titik di ruang 82-dimensi, kita butuh **garis pemisah** antar
kelas motif. SVM mencari batas yang **jaraknya (margin) semaksimal mungkin** ke titik-titik
terdekat dari tiap kelas ("support vectors").

### Kernel RBF
Batas antar motif jarang bisa lurus. **Kernel RBF** memetakan data ke ruang lebih tinggi
secara implisit sehingga batas melengkung bisa dibuat — tanpa benar-benar menghitung
koordinat baru (kernel trick).

### Biaya
- **Latih:** sekali saja, cepat untuk ratusan sampel.
- **Prediksi:** cuma menghitung kernel antara gambar baru dan sejumlah support vector —
  beberapa perkalian kecil. Nyaris instan.

`StandardScaler` menyamakan skala tiap fitur dulu (biar contrast yang besar tak menutupi
homogeneity yang kecil), baru masuk SVM. KNN dipakai sebagai pembanding.

---

## 4. Kenapa Lebih Kencang dari CLIP ViT-B/32 Grouping?

Hasil benchmark nyata di mesin ini ([`src/benchmark.py`](../src/benchmark.py),
`run_benchmark()`):

| Jalur | Embedding | Grouping/klasifikasi | Total/gambar |
| --- | --- | --- | --- |
| **GLCM + LBP + SVM** | ~1.3 ms | <0.1 ms | **~2.3 ms** |
| **CLIP ViT-B/32 Grouping** | ~18.5 ms | ~0.2 ms | **~19 ms** |

Selisihnya **~8–14×**, dan **hampir seluruhnya di langkah embedding**. Cosine search vs
SVM sama-sama <0.2 ms — bukan itu penentunya. Alasannya:

**a. Jumlah operasi.**
GLCM+LBP = beberapa kali sapu piksel + penjumlahan → menghasilkan 82 angka. CLIP ViT-B/32
menjalankan **transformer ~88 juta parameter**: puluhan layer attention + matmul besar untuk
menghasilkan 512 angka. Bedanya bukan konstanta kecil, tapi **ordo besaran**.

**b. Tanpa akselerator, gap makin lebar.**
GLCM/LBP jalan efisien di CPU. Transformer butuh GPU/MPS biar wajar; di CPU murni CLIP bisa
puluhan-ratusan ms. Untuk deploy hemat (server kecil / tanpa GPU), tradisional menang telak.

**c. Fitur ringkas, classifier ringan.**
Vektor 82-dim → SVM cukup hitung kernel ke segelintir support vector. Tidak ada beban model
besar yang harus di-load & di-forward tiap gambar.

**d. Preprocessing lebih murah.**
GLCM cukup grayscale + kuantisasi. CLIP butuh resize 224×224, normalisasi, tensor, dan
pipeline processor-nya sendiri.

### Tapi — cepat bukan berarti selalu lebih baik
Kecepatan ini ada harganya:

| Aspek | GLCM+LBP+SVM | CLIP ViT-B/32 |
| --- | --- | --- |
| Kecepatan | ✅ ~2 ms | ~19 ms |
| Butuh training + label | ✅ ya (supervised) | ❌ tidak (zero-shot / embedding siap pakai) |
| Kaya makna semantik | terbatas (tekstur saja) | ✅ 512-dim, general |
| Motif baru tanpa retrain | ❌ perlu latih ulang | ✅ tinggal tambah embedding |
| Cocok untuk | klasifikasi tekstur cepat, hemat resource | pencarian "lookalike", dataset terbuka |

**Ringkas:** GLCM+LBP+SVM cepat karena ia **model kecil dengan fitur tangan yang ringkas**;
CLIP lambat karena ia **jaringan raksasa serba-bisa**. Pilih tradisional saat kelas tetap
dan resource terbatas; pilih CLIP saat butuh grouping/similarity terbuka seperti
mahatex-lookalike.

---

**Regenerate angka benchmark:**
```bash
.venv/bin/python -c "from src.benchmark import run_benchmark; run_benchmark()"
.venv/bin/python -m src.benchmark grouping   # rincian embed vs grouping terpisah
```

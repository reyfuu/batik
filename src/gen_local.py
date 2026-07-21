"""Generate variasi batik lokal (SD-Turbo img2img), dipakai CLI dan web.

Model di-load sekali per proses (lazy) — load tiap request bikin server 8GB swap.
float16 dipilih karena SD1.5 float32 = ~5.5 jam/gambar di Mac ini (lihat
[[local-diffusion-unviable]]).
"""
import os
import time
import uuid
import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "public", "generated")
MODEL = "stabilityai/sd-turbo"

_pipe = None


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load():
    """Muat pipeline sekali (float16, ~2.5GB, muat di 8GB berkat slicing).

    PENTING: fp16 andal HANYA kalau SD-Turbo punya MPS sendiri. Satu proses dengan
    DINOv2 (mis. server web) -> rebutan memori 8GB, VAE fp16 keluar gambar HITAM (NaN).
    Karena itu web memanggil lewat SUBPROCESS (lihat web/app.py), bukan import langsung.
    """
    global _pipe
    if _pipe is None:
        import torch
        from diffusers import AutoPipelineForImage2Image
        p = AutoPipelineForImage2Image.from_pretrained(MODEL, torch_dtype=torch.float16)
        p = p.to(_device())
        p.set_progress_bar_config(disable=True)
        p.enable_attention_slicing()
        p.enable_vae_slicing()
        if hasattr(p, "safety_checker"):
            p.safety_checker = None  # motif batik kadang salah-trigger -> hitam palsu
        _pipe = p
    return _pipe


def _texture_patch(img, zoom=0.25):
    """Ambil potongan PALING BERTEKSTUR dari gambar, lalu perbesar ke 512.

    WAJIB untuk foto produk (baju di manekin/orang): img2img mempertahankan komposisi
    acuan, jadi foto baju utuh -> keluarannya ikut jadi foto baju/katalog, bukan motif.
    Crop tengah saja tidak cukup — masih kena latar putih & tepi baju, siluetnya
    menembus ke hasil. Jadi dipilih kandidat dengan energi tepi tertinggi = kain
    bermotif, bukan latar polos.
    """
    w, h = img.size
    s = int(min(w, h) * zoom)
    best, best_score = None, -1.0
    for fx in (0.15, 0.5, 0.85):
        for fy in (0.15, 0.5, 0.85):
            left, top = int((w - s) * fx), int((h - s) * fy)
            c = img.crop((left, top, left + s, top + s))
            g = np.asarray(c.convert("L"), dtype=np.float32)
            score = float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean())
            if score > best_score:
                best, best_score = c, score
    return best.resize((512, 512), Image.LANCZOS)


def generate(ref, prompt, n=5, strength=0.75, steps=4, out_dir=OUT_DIR, zoom=0.25):
    """n variasi batik dari `ref` (path str atau PIL.Image) dipandu prompt.

    strength: 0=identik referensi, 1=abaikan referensi. 0.75 dipilih dari sapuan uji —
    di bawah ~0.7 komposisi acuan (siluet baju, tata letak katalog) ikut tercetak.
    zoom: ukuran patch tekstur; 0.25 supaya foto produk tidak menyeret latar & tepi baju.
    Return: list nama .png.
    """
    import torch
    pipe = _load()
    os.makedirs(out_dir, exist_ok=True)
    src = (Image.open(ref) if isinstance(ref, str) else ref).convert("RGB")
    init = _texture_patch(src, zoom) if zoom else src.resize((512, 512))

    saved = []
    # ponytail: kena gambar hitam -> ulang dengan seed lain, jangan `continue` — dulu
    # minta 5 bisa pulang 2 tanpa keterangan. Batas 2n percobaan supaya tidak loop
    # selamanya kalau MPS-nya memang lagi rusak.
    seed = int(time.time() * 1000) % 2**31
    for attempt in range(2 * n):
        if len(saved) >= n:
            break
        g = torch.Generator(device=_device()).manual_seed(seed + attempt)
        img = pipe(prompt=prompt, image=init, strength=strength,
                   num_inference_steps=steps, guidance_scale=0.0, generator=g).images[0]
        if np.asarray(img).mean() < 8:  # guard hitam (bug float16 VAE di MPS)
            continue
        fname = "batik_" + uuid.uuid4().hex[:10] + ".png"
        img.save(os.path.join(out_dir, fname))
        saved.append(fname)
    return saved

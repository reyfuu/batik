"""Generate variasi batik dari 1 referensi + prompt, via Replicate SDXL img2img (cloud).

Cloud, bukan diffusers lokal: Mac ini tanpa GPU besar, lokal di CPU = berjam-jam.
Bayar per gambar (~$0.003-0.01), jadi TES 5 dulu sebelum 300.

Pakai:  .venv/bin/python scripts/gen_batik.py <ref.jpg> "<prompt>" <n>
Default (tes): 5 gambar dari batik Megamendung contoh.
"""
import os
import sys
import uuid
import requests
import replicate
from dotenv import load_dotenv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.expanduser("~/api/.env"))
load_dotenv(os.path.join(ROOT, ".env"), override=True)

OUT_DIR = os.path.join(ROOT, "public", "generated")
MODEL = "stability-ai/sdxl"  # img2img: input image + prompt + prompt_strength


def _model_ref():
    """Resolve 'owner/name' -> 'owner/name:version' (replicate.run butuh versi eksplisit)."""
    t = os.environ["REPLICATE_API_TOKEN"].strip()
    r = requests.get(f"https://api.replicate.com/v1/models/{MODEL}",
                     headers={"Authorization": f"Bearer {t}"}, timeout=20)
    r.raise_for_status()
    return f"{MODEL}:{r.json()['latest_version']['id']}"


def _save(item):
    """Simpan 1 output (FileOutput atau URL) -> path lokal, kembalikan nama file."""
    fname = "batik_" + uuid.uuid4().hex[:10] + ".png"
    data = item.read() if hasattr(item, "read") else requests.get(str(item), timeout=120).content
    with open(os.path.join(OUT_DIR, fname), "wb") as f:
        f.write(data)
    return fname


def generate(ref_path, prompt, n, strength=0.65):
    """n variasi batik dari ref_path yang dipandu prompt.

    strength (prompt_strength SDXL): 0=identik referensi, 1=abaikan referensi. 0.65 =
    tetap batik tapi ikut prompt; naikkan kalau mau lebih jauh dari referensi.
    """
    if not os.environ.get("REPLICATE_API_TOKEN"):
        sys.exit("REPLICATE_API_TOKEN belum di-set di .env")
    os.makedirs(OUT_DIR, exist_ok=True)
    ref = _model_ref()
    print(f"model: {ref}")
    print(f"referensi: {ref_path}\nprompt: {prompt}\nstrength: {strength} | target: {n} gambar\n")

    saved = []
    # SDXL num_outputs maks 4/panggilan -> pecah jadi batch.
    while len(saved) < n:
        batch = min(4, n - len(saved))
        with open(ref_path, "rb") as fh:
            out = replicate.run(ref, input={
                "image": fh, "prompt": prompt, "prompt_strength": strength,
                "num_outputs": batch, "num_inference_steps": 25,
            })
        for it in (out if isinstance(out, (list, tuple)) else [out]):
            saved.append(_save(it))
            print(f"  [{len(saved)}/{n}] {saved[-1]}")
    print(f"\nselesai: {len(saved)} gambar -> {OUT_DIR}")
    return saved


if __name__ == "__main__":
    ref = sys.argv[1] if len(sys.argv) > 1 else \
        "data/raw_batik_v2/train/Jawa_Barat_Megamendung/" + \
        sorted(os.listdir("data/raw_batik_v2/train/Jawa_Barat_Megamendung"))[0]
    prompt = sys.argv[2] if len(sys.argv) > 2 else \
        "Indonesian batik megamendung motif, traditional cloud pattern, indigo blue, " \
        "intricate hand-drawn textile, seamless pattern, high detail"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    generate(ref, prompt, n)

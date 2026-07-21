"""CLI generate variasi batik lokal (SD-Turbo img2img); wrapper src/gen_local.py.

Batch besar (mis. 300) lewat CLI ini, bukan web (browser tidak menggantung).
Pakai:  .venv/bin/python scripts/gen_batik_local.py <ref.jpg> "<prompt>" <n>
"""
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.gen_local import generate, OUT_DIR

if __name__ == "__main__":
    ref = sys.argv[1] if len(sys.argv) > 1 else \
        "data/raw_batik_v2/train/Jawa_Barat_Megamendung/" + \
        sorted(os.listdir("data/raw_batik_v2/train/Jawa_Barat_Megamendung"))[0]
    prompt = sys.argv[2] if len(sys.argv) > 2 else \
        "Indonesian batik megamendung motif, cloud pattern, indigo blue, seamless"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    print(f"referensi: {ref}\nprompt: {prompt}\ntarget: {n} gambar")
    t0 = time.time()
    names = generate(ref, prompt, n=n)
    print(f"selesai: {len(names)} gambar dalam {time.time()-t0:.0f}s -> {OUT_DIR}")
    for nm in names:
        print("  " + nm)

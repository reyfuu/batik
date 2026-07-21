"""Builder prompt batik untuk generator teks->gambar (Flux/SDXL).

Isi motif + warna, keluar prompt natural. Flux lebih suka kalimat utuh daripada tag.

    from web.prompt_batik import build_prompt, MOTIFS
    build_prompt("parang", palette="sogan brown and cream")
"""

# Deskripsi visual + warna tradisional tiap motif, untuk memperkaya prompt.
MOTIFS = {
    "parang":       ("diagonal repeating S-shaped 'parang' blade motif in parallel rows", "sogan brown and cream"),
    "kawung":       ("geometric 'kawung' pattern of intersecting circles and quatrefoils", "deep brown and ivory"),
    "megamendung":  ("layered wavy 'megamendung' cloud motif with graduated outlines", "indigo blue and white"),
    "sekar jagad":  ("'sekar jagad' patchwork of varied floral islands like a map", "earthy brown, blue and red"),
    "cendrawasih":  ("'cendrawasih' bird-of-paradise motif among flowing florals", "warm brown, gold and teal"),
    "insang":       ("'insang' fish-gill geometric bands motif", "red, cream and black"),
    "poleng":       ("'poleng' black-and-white checkerboard motif", "black and white"),
    "lasem":        ("coastal 'lasem' floral and phoenix motif", "vivid red, cream and blue"),
    "tambal":       ("'tambal' patchwork of many small differing motif tiles", "multicolor earthy tones"),
}

# Template inti (natural language) untuk Flux/SDXL.
_TEMPLATE = (
    "A seamless traditional Indonesian batik textile pattern, {desc}, "
    "hand-drawn wax-resist (batik tulis) with fine intricate canting lines, "
    "{palette} color palette, flat top-down view of fabric, evenly repeating, "
    "highly detailed, authentic cultural craftsmanship, soft natural lighting"
)


def build_prompt(motif, palette=None, extra=None):
    """Bangun prompt batik.

    motif   : kunci MOTIFS (mis. 'parang'), atau teks bebas untuk motif kustom.
    palette : override warna; default = warna tradisional motif.
    extra   : detail opsional (mis. 'for a formal dress', 'pesisir style').
    """
    key = motif.strip().lower()
    if key in MOTIFS:
        desc, default_palette = MOTIFS[key]
    else:
        desc = f"'{motif}' motif"          # motif kustom di luar daftar
        default_palette = "traditional earthy tones"
    prompt = _TEMPLATE.format(desc=desc, palette=palette or default_palette)
    if extra:
        prompt += f", {extra}"
    return prompt


if __name__ == "__main__":
    # self-check
    p = build_prompt("parang")
    assert "parang" in p and "sogan brown" in p and "seamless" in p
    assert "indigo blue" in build_prompt("megamendung")
    assert "royal purple" in build_prompt("kawung", palette="royal purple and gold")
    assert "'gurda' motif" in build_prompt("gurda")   # motif kustom
    print("OK — contoh output:\n")
    for m in ["parang", "megamendung", "kawung"]:
        print(f"[{m}]\n{build_prompt(m)}\n")

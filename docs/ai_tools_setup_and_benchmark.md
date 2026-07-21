# Setup AI Tools dan Benchmark GLCM + LBP + SVM

Kode tool dipasang di `external_tools/` dengan environment Python terpisah:

| Tool | Folder | Command launcher |
| --- | --- | --- |
| Hugging Face | `.venv` project utama | `.venv/bin/hf --help` |
| ComfyUI | `external_tools/ComfyUI` | `bash scripts/launch_ai_tool.sh comfyui` |
| Fooocus | `external_tools/Fooocus` | `bash scripts/launch_ai_tool.sh fooocus` |
| InvokeAI | `external_tools/InvokeAI` | `bash scripts/launch_ai_tool.sh invokeai` |
| AUTOMATIC1111 | `external_tools/stable-diffusion-webui` | `bash scripts/launch_ai_tool.sh automatic1111` |

Port default:

| Tool | URL lokal |
| --- | --- |
| ComfyUI | `http://127.0.0.1:8188` |
| Fooocus | `http://127.0.0.1:7865` |
| InvokeAI | lihat output terminal InvokeAI saat start |
| AUTOMATIC1111 | `http://127.0.0.1:7860` |

## Model Gambar

UI Stable Diffusion biasanya belum siap generate sampai checkpoint/model ditempatkan sesuai folder tool:

| Tool | Folder model umum |
| --- | --- |
| ComfyUI | `external_tools/ComfyUI/models/checkpoints/` |
| Fooocus | otomatis mengunduh SDXL saat run pertama |
| InvokeAI | lewat model manager/launcher InvokeAI |
| AUTOMATIC1111 | `external_tools/stable-diffusion-webui/models/Stable-diffusion/` |

## Struktur Output Benchmark

Simpan gambar hasil tiap tool ke:

```text
data/tool_outputs/
  huggingface/
    parang/
    kawung/
    megamendung/
  comfyui/
    parang/
    kawung/
    megamendung/
  fooocus/
    parang/
    kawung/
    megamendung/
  invokeai/
    parang/
    kawung/
    megamendung/
  automatic1111/
    parang/
    kawung/
    megamendung/
```

Jika gambar tidak dipisah dalam folder label, benchmark tetap memprediksi kelasnya tetapi akurasi tidak dihitung.

Run:

```bash
.venv/bin/python scripts/benchmark_ai_tool_outputs.py
```

Output laporan:

- `reports/ai_tools_glcm_lbp_svm_benchmark.csv`
- `reports/ai_tools_glcm_lbp_svm_summary.json`
- `reports/ai_tools_glcm_lbp_svm_benchmark.md`

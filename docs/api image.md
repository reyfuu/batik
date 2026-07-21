# API Image Generator

Panduan ini untuk workflow image generator di project ini:

- **Diffusers/Hugging Face** bisa dipakai langsung dari terminal/Python.
- **ComfyUI**, **InvokeAI**, dan **AUTOMATIC1111** sebaiknya dijalankan sebagai server lalu dipanggil lewat API.
- **Fooocus** berbasis Gradio; bisa dipanggil lewat Gradio client, tetapi bukan API image-generation resmi yang stabil seperti ComfyUI/A1111.

Jangan commit API key ke repo. Simpan di environment variable atau file lokal yang masuk `.gitignore`.

## Langkah Cepat: API Key + Browser

Kalau tujuanmu hanya **menjalankan UI sampai bisa dibuka di browser**, gunakan ini:

| Tool | Ambil API key? | Jalankan dari terminal | Buka di browser |
| --- | --- | --- | --- |
| ComfyUI local | Tidak perlu | `python image.py comfyui` | `http://127.0.0.1:8188` |
| InvokeAI local | Tidak perlu untuk single-user local | `python image.py invokeai` | lihat URL yang muncul di terminal |
| AUTOMATIC1111 local | Tidak perlu, kecuali mau proteksi API | `python image.py automatic` | `http://127.0.0.1:7860` |
| Hugging Face / Diffusers | Perlu kalau model gated/private atau pakai HF API | terminal/Python, bukan browser UI utama | tidak ada browser UI default |
| Comfy Cloud / Partner Nodes | Perlu Comfy API Key | lewat Comfy Platform/API | platform Comfy |

Urutan paling simpel:

```bash
python image.py comfyui
```

Lalu buka:

```text
http://127.0.0.1:8188
```

Untuk AUTOMATIC1111:

```bash
python image.py automatic
```

Lalu buka:

```text
http://127.0.0.1:7860
```

Untuk InvokeAI:

```bash
python image.py invokeai
```

Lalu lihat URL yang dicetak di terminal InvokeAI, kemudian buka URL itu di browser.

Kalau mau jalankan tiga server sekaligus:

```bash
python image.py all
```

Tekan `Ctrl+C` di terminal untuk menghentikan server.

## Ringkasan

| Tool | Butuh API key? | Bentuk auth | Kapan dipakai |
| --- | --- | --- | --- |
| Hugging Face / Diffusers | Ya, kalau download gated/private model atau pakai HF Inference/API | `HF_TOKEN=hf_...` | Terminal/Python, model dari Hugging Face |
| ComfyUI local server | Tidak untuk `/prompt` lokal standar | Tanpa key di `127.0.0.1` | Submit workflow lokal |
| Comfy Platform / Comfy Cloud / Partner Nodes | Ya | Comfy API Key | Cloud API atau paid Partner Nodes |
| AUTOMATIC1111 local API | Tidak wajib | Optional Basic Auth via `--api-auth user:pass` | Endpoint `/sdapi/v1/...` |
| InvokeAI local | Umumnya tidak wajib di single-user mode | Bisa user auth jika multi-user diaktifkan | Invoke web/server lokal |
| Fooocus | Tidak punya API key resmi | Optional Basic Auth via `auth.json` | Gradio UI/client lokal |

## 1. Hugging Face / Diffusers

Diffusers lokal bisa jalan tanpa API key jika modelnya sudah lokal dan tidak gated. API key/token diperlukan untuk:

- download model private/gated dari Hugging Face,
- login Hugging Face CLI,
- memakai Hugging Face Inference Providers/API,
- akses model dengan rate limit akun.

Cara ambil token:

1. Buka <https://huggingface.co/settings/tokens>.
2. Login ke akun Hugging Face.
3. Klik **New token**.
4. Pilih role:
   - `read` untuk download model dan inference biasa.
   - `write` hanya jika perlu upload/push model.
   - `fine-grained` untuk production atau akses terbatas ke model tertentu.
5. Copy token yang formatnya seperti `hf_...`.

Set di terminal:

```bash
export HF_TOKEN="hf_xxx"
.venv/bin/hf auth login
```

Untuk cek token sudah aktif:

```bash
.venv/bin/hf auth whoami
```

Contoh Diffusers terminal/Python:

```bash
export HF_TOKEN="hf_xxx"
.venv/bin/python - <<'PY'
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    token=True,
)
pipe = pipe.to("cpu")
image = pipe("batik parang textile pattern").images[0]
image.save("data/tool_outputs/huggingface/parang/parang_hf.png")
PY
```

Sumber: Hugging Face menjelaskan token dibuat dari Settings → Access Tokens, token bisa dipakai sebagai bearer token atau di library Python, dan sebaiknya memakai token per app serta fine-grained untuk production: <https://huggingface.co/docs/hub/security-tokens>.

## 2. ComfyUI

Ada dua jenis API:

### A. ComfyUI Local Server

Untuk server lokal yang sudah di-install di project ini, biasanya **tidak perlu API key**.

Jalankan:

```bash
python image.py comfyui
```

Buka browser:

```text
http://127.0.0.1:8188
```

Kalau browser menampilkan halaman ComfyUI, server sudah berhasil jalan. Untuk stop, kembali ke terminal lalu tekan `Ctrl+C`.

Endpoint penting:

- `POST /prompt` untuk submit workflow.
- `GET /history/{prompt_id}` untuk hasil.
- `GET /view` untuk mengambil file image.
- `GET /object_info` untuk daftar node.

Sumber: dokumentasi ComfyUI Server API mencantumkan `/prompt`, `/history`, `/view`, `/queue`, `/object_info`, dan route lain: <https://docs.comfy.org/development/comfyui-server/comms_routes>.

### B. Comfy Platform / Cloud API / Partner Nodes

Ini yang membutuhkan **Comfy API Key**.

Cara ambil API key:

1. Buka <https://platform.comfy.org/login>.
2. Login.
3. Masuk ke bagian **API Keys**.
4. Klik **+ New**.
5. Isi nama key.
6. Klik **Generate**.
7. Copy key saat itu juga. Key hanya ditampilkan sekali.

Simpan:

```bash
export COMFY_API_KEY="comfyui_xxx"
```

Untuk workflow local yang memakai paid Partner Nodes, key biasanya dikirim di `extra_data.api_key_comfy_org` pada payload workflow. Untuk Cloud API, key dikirim di request header sesuai dokumentasi endpoint cloud.

Sumber: ComfyUI menjelaskan API key dibuat di platform.comfy.org, hanya terlihat saat dibuat, dan dipakai untuk Cloud API, Partner Nodes, MCP, serta login lokal: <https://docs.comfy.org/development/api-development/getting-an-api-key>.

Catatan penting: **ComfyUI local di `http://127.0.0.1:8188` tidak membutuhkan Comfy API Key** untuk membuka UI dan menjalankan workflow lokal biasa. Key Comfy hanya untuk Cloud/Platform/Partner Nodes.

## 3. AUTOMATIC1111

AUTOMATIC1111 local API tidak memakai API key cloud. API diaktifkan dengan flag `--api`, lalu endpoint berada di:

```text
http://127.0.0.1:7860/sdapi/v1/
```

Jalankan dengan API:

```bash
python image.py automatic
```

Buka browser:

```text
http://127.0.0.1:7860
```

`image.py` sudah menambahkan `--api`, jadi endpoint API juga aktif:

```text
http://127.0.0.1:7860/sdapi/v1/
```

Jika ingin “API key style”, gunakan Basic Auth:

```bash
COMMANDLINE_ARGS="--server-name 127.0.0.1 --port 7860 --api --api-auth batik:passwordku" \
python image.py automatic
```

Contoh call:

```bash
curl -u batik:passwordku \
  -H "Content-Type: application/json" \
  -d '{"prompt":"batik kawung textile pattern","steps":20,"width":512,"height":512}' \
  http://127.0.0.1:7860/sdapi/v1/txt2img
```

Catatan:

- `--api` menjalankan API bersama WebUI.
- `--nowebui` menjalankan API tanpa UI.
- `--api-auth user:pass` menambahkan Basic Auth.

Sumber: wiki API AUTOMATIC1111: <https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API>. Flag `--api` dan `--api-auth` juga ada di `external_tools/stable-diffusion-webui/modules/cmd_args.py`.

## 4. InvokeAI

InvokeAI yang terpasang di project ini adalah server/web app lokal. Untuk pemakaian lokal single-user, biasanya **tidak perlu API key eksternal**.

Jalankan:

```bash
python image.py invokeai
```

Lihat URL di output terminal InvokeAI, lalu buka URL itu di browser. Untuk OpenAPI docs, setelah server hidup biasanya cek:

```text
http://127.0.0.1:<port>/docs
http://127.0.0.1:<port>/openapi.json
```

Auth yang mungkin relevan:

- **Single-user local mode**: tidak perlu API key.
- **Multi-user mode**: perlu user login; buat user dengan command InvokeAI jika mode ini aktif.
- **External generation providers** di InvokeAI: beberapa provider butuh API key sendiri, misalnya OpenAI, Gemini, Alibaba Cloud DashScope, atau Seedream.

Nama setting yang ditemukan di versi repo ini:

- `openai_api_key`
- `gemini_api_key`
- `alibabacloud_api_key`
- `seedream_api_key`
- `remote_api_tokens`

Cara umum mengisi:

1. Buka InvokeAI Web UI.
2. Masuk ke Settings atau konfigurasi provider.
3. Masukkan API key provider yang dipakai.
4. Jangan simpan key ke file repo publik.

Contoh environment lokal jika ingin dikelola dari terminal:

```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
```

Untuk membuka InvokeAI di browser, **jangan cari API key dulu** kecuali kamu memang memakai provider eksternal. Jalankan server, buka URL terminal, lalu masukkan key provider dari Settings kalau fitur itu diminta.

Catatan: nama environment variable final mengikuti provider/node InvokeAI yang dipakai. Cek UI/config InvokeAI setelah server berjalan.

Sumber lokal versi yang terpasang:

- `external_tools/InvokeAI/docs/src/generated/settings.json`
- `external_tools/InvokeAI/invokeai/app/services/external_generation/providers/openai.py`
- `external_tools/InvokeAI/invokeai/app/services/external_generation/providers/gemini.py`
- `external_tools/InvokeAI/invokeai/app/services/external_generation/providers/alibabacloud.py`
- `external_tools/InvokeAI/invokeai/app/services/external_generation/providers/seedream.py`

## 5. Fooocus

Fooocus tidak punya API key image generator resmi. Secara default akses UI lokal tidak memakai auth. Jika dibuka untuk network/share, pakai Basic Auth lewat `auth.json`.

Cara aktifkan auth:

1. Buat file:

```text
external_tools/Fooocus/auth.json
```

2. Isi:

```json
[
  {
    "user": "batik",
    "pass": "passwordku"
  }
]
```

3. Jalankan Fooocus:

```bash
bash scripts/launch_ai_tool.sh fooocus
```

Buka browser:

```text
http://127.0.0.1:7865
```

Fooocus memakai Gradio. Jika ingin dipanggil dari script, biasanya gunakan `gradio_client`, tapi endpoint/fungsi bisa berubah mengikuti versi UI. Untuk pipeline API stabil, lebih disarankan ComfyUI atau AUTOMATIC1111.

Sumber lokal versi yang terpasang:

- `external_tools/Fooocus/readme.md`
- `external_tools/Fooocus/modules/auth.py`
- `external_tools/Fooocus/auth-example.json`

## Rekomendasi untuk Project Batik

Untuk generate gambar motif lalu benchmark GLCM + LBP + SVM:

1. Pakai **Diffusers/Hugging Face** dari terminal untuk baseline scriptable.
2. Pakai **ComfyUI API** untuk workflow node yang kompleks.
3. Pakai **AUTOMATIC1111 API** untuk text-to-image standar dan batch.
4. Pakai **InvokeAI** bila butuh canvas/workflow UI dan endpoint lokalnya.
5. Pakai **Fooocus** untuk hasil cepat dari UI, lalu simpan output ke folder benchmark.

Folder output benchmark:

```text
data/tool_outputs/
  huggingface/parang/
  comfyui/parang/
  fooocus/parang/
  invokeai/parang/
  automatic1111/parang/
```

Setelah gambar masuk, jalankan:

```bash
.venv/bin/python scripts/benchmark_ai_tool_outputs.py
```

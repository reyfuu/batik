#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL="${1:-}"

export MPLCONFIGDIR="${ROOT_DIR}/.cache/matplotlib"
export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p "${MPLCONFIGDIR}" "${HF_HOME}"

case "${TOOL}" in
  comfyui)
    cd "${ROOT_DIR}/external_tools/ComfyUI"
    exec .venv/bin/python main.py --listen 127.0.0.1 --port 8188
    ;;
  fooocus)
    cd "${ROOT_DIR}/external_tools/Fooocus"
    exec .venv/bin/python entry_with_update.py --listen 127.0.0.1 --port 7865
    ;;
  invokeai)
    cd "${ROOT_DIR}/external_tools/InvokeAI"
    exec .venv/bin/invokeai-web
    ;;
  automatic1111|a1111)
    cd "${ROOT_DIR}/external_tools/stable-diffusion-webui"
    export COMMANDLINE_ARGS="${COMMANDLINE_ARGS:---server-name 127.0.0.1 --port 7860 --skip-torch-cuda-test --upcast-sampling --no-half-vae --use-cpu interrogate}"
    exec ./webui.sh
    ;;
  *)
    echo "Usage: $0 {comfyui|fooocus|invokeai|automatic1111}" >&2
    exit 2
    ;;
esac

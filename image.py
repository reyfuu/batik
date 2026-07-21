#!/usr/bin/env python3
"""Launcher untuk image generator lokal: ComfyUI, InvokeAI, dan AUTOMATIC1111.

Contoh:
  python image.py comfyui
  python image.py invokeai
  python image.py automatic
  python image.py all
"""
import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


TOOLS = {
    "comfyui": {
        "name": "ComfyUI",
        "cwd": ROOT / "external_tools" / "ComfyUI",
        "cmd": [".venv/bin/python", "main.py", "--listen", "127.0.0.1", "--port", "8188"],
        "url": "http://127.0.0.1:8188",
    },
    "invokeai": {
        "name": "InvokeAI",
        "cwd": ROOT / "external_tools" / "InvokeAI",
        "cmd": [".venv/bin/invokeai-web"],
        "url": "lihat URL di output terminal InvokeAI",
    },
    "automatic": {
        "name": "AUTOMATIC1111",
        "cwd": ROOT / "external_tools" / "stable-diffusion-webui",
        "cmd": ["./webui.sh"],
        "url": "http://127.0.0.1:7860",
        "commandline_args": (
            "--server-name 127.0.0.1 "
            "--port 7860 "
            "--api "
            "--skip-torch-cuda-test "
            "--upcast-sampling "
            "--no-half-vae "
            "--use-cpu interrogate"
        ),
    },
}

ALIASES = {
    "a1111": "automatic",
    "automatic1111": "automatic",
    "auto": "automatic",
}


def build_env(tool_key: str) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    env.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["HF_HOME"]).mkdir(parents=True, exist_ok=True)

    tool = TOOLS[tool_key]
    if "commandline_args" in tool:
        env.setdefault("COMMANDLINE_ARGS", tool["commandline_args"])
    return env


def normalize_tool(name: str) -> str:
    return ALIASES.get(name.lower(), name.lower())


def validate_tool(tool_key: str) -> None:
    tool = TOOLS[tool_key]
    if not tool["cwd"].exists():
        raise FileNotFoundError(f"Folder belum ada: {tool['cwd']}")

    executable = tool["cwd"] / tool["cmd"][0]
    if not executable.exists():
        raise FileNotFoundError(f"Command belum ada: {executable}")


def start_tool(tool_key: str) -> subprocess.Popen:
    validate_tool(tool_key)
    tool = TOOLS[tool_key]
    print(f"\n==> Menjalankan {tool['name']}")
    print(f"    URL: {tool['url']}")
    print(f"    Folder: {tool['cwd']}")
    sys.stdout.flush()

    return subprocess.Popen(
        tool["cmd"],
        cwd=tool["cwd"],
        env=build_env(tool_key),
    )


def run_single(tool_key: str) -> int:
    proc = start_tool(tool_key)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        stop_process(proc)
        return 130


def stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def run_all() -> int:
    processes: list[subprocess.Popen] = []
    for tool_key in ("comfyui", "invokeai", "automatic"):
        processes.append(start_tool(tool_key))

    print("\nSemua server dimulai. Tekan Ctrl+C untuk menghentikan semuanya.")
    sys.stdout.flush()

    try:
        while True:
            for proc in processes:
                code = proc.poll()
                if code is not None and code != 0:
                    print(f"\nSalah satu proses berhenti dengan exit code {code}.")
                    return code
            signal.pause()
    except KeyboardInterrupt:
        print("\nMenghentikan semua server...")
        for proc in processes:
            stop_process(proc)
        return 130


def main() -> int:
    parser = argparse.ArgumentParser(description="Jalankan image generator lokal.")
    parser.add_argument(
        "tool",
        choices=["comfyui", "invokeai", "automatic", "automatic1111", "a1111", "auto", "all"],
        help="Tool yang ingin dijalankan.",
    )
    args = parser.parse_args()

    tool_key = normalize_tool(args.tool)
    if tool_key == "all":
        return run_all()
    return run_single(tool_key)


if __name__ == "__main__":
    raise SystemExit(main())

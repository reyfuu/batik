import argparse
import csv
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.glcm_svm import extract_combined_features, train_and_evaluate

TOOLS = ["huggingface", "comfyui", "fooocus", "invokeai", "automatic1111"]
LABELS = ["parang", "kawung", "megamendung"]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def load_or_train(model_dir: Path):
    scaler_path = model_dir / "scaler.pkl"
    svm_path = model_dir / "svm_model.pkl"
    needs_train = True

    if scaler_path.exists() and svm_path.exists():
        with scaler_path.open("rb") as f:
            scaler = pickle.load(f)
        with svm_path.open("rb") as f:
            svm = pickle.load(f)
        needs_train = not hasattr(scaler, "n_features_in_") or scaler.n_features_in_ != 82
    if needs_train:
        result = train_and_evaluate()
        if result[0] is None:
            raise RuntimeError("Model SVM belum tersedia dan training gagal.")
        with scaler_path.open("rb") as f:
            scaler = pickle.load(f)
        with svm_path.open("rb") as f:
            svm = pickle.load(f)
    return scaler, svm


def iter_images(root: Path):
    for tool in TOOLS:
        tool_dir = root / tool
        if not tool_dir.exists():
            continue

        labelled_found = False
        for label in LABELS:
            label_dir = tool_dir / label
            if not label_dir.exists():
                continue
            labelled_found = True
            for path in sorted(label_dir.rglob("*")):
                if path.suffix.lower() in IMAGE_EXTS:
                    yield tool, label, path

        if labelled_found:
            continue
        for path in sorted(tool_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTS:
                yield tool, None, path


def predict_image(path: Path, scaler, svm):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Gagal membaca gambar: {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    feat = extract_combined_features(gray).reshape(1, -1)
    feat_scaled = scaler.transform(feat)
    pred_idx = int(svm.predict(feat_scaled)[0])
    confidence = None
    if hasattr(svm, "predict_proba"):
        confidence = float(svm.predict_proba(feat_scaled)[0][pred_idx])
    latency_ms = (time.perf_counter() - t0) * 1000
    return LABELS[pred_idx], confidence, latency_ms


def summarize(rows):
    summary = {}
    for tool in TOOLS:
        tool_rows = [r for r in rows if r["tool"] == tool]
        if not tool_rows:
            summary[tool] = {"count": 0, "accuracy": None, "avg_latency_ms": None}
            continue
        labelled = [r for r in tool_rows if r["expected_label"]]
        correct = [r for r in labelled if r["expected_label"] == r["predicted_label"]]
        avg_latency = sum(r["latency_ms"] for r in tool_rows) / len(tool_rows)
        summary[tool] = {
            "count": len(tool_rows),
            "labelled_count": len(labelled),
            "accuracy": (len(correct) / len(labelled)) if labelled else None,
            "avg_latency_ms": avg_latency,
        }
    return summary


def write_markdown(summary, out_path: Path):
    lines = [
        "# Benchmark Output AI Tools dengan GLCM + LBP + SVM",
        "",
        "| Tool | Jumlah gambar | Labelled | Akurasi | Latency rata-rata |",
        "| --- | --: | --: | --: | --: |",
    ]
    for tool, item in summary.items():
        accuracy = "-" if item["accuracy"] is None else f"{item['accuracy'] * 100:.2f}%"
        latency = "-" if item["avg_latency_ms"] is None else f"{item['avg_latency_ms']:.2f} ms"
        lines.append(
            f"| {tool} | {item['count']} | {item.get('labelled_count', 0)} | {accuracy} | {latency} |"
        )
    lines.append("")
    lines.append("Catatan: akurasi dihitung hanya jika gambar diletakkan di subfolder label: `parang`, `kawung`, atau `megamendung`.")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Benchmark output 5 AI tools memakai GLCM + LBP + SVM.")
    parser.add_argument("--input", default="data/tool_outputs", help="Folder output tool.")
    parser.add_argument("--reports", default="reports", help="Folder laporan.")
    parser.add_argument("--model-dir", default="models", help="Folder model SVM/scaler.")
    args = parser.parse_args()

    root = Path(args.input)
    reports = Path(args.reports)
    reports.mkdir(parents=True, exist_ok=True)

    scaler, svm = load_or_train(Path(args.model_dir))
    rows = []
    for tool, expected, path in iter_images(root):
        predicted, confidence, latency_ms = predict_image(path, scaler, svm)
        rows.append(
            {
                "tool": tool,
                "expected_label": expected,
                "predicted_label": predicted,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "path": str(path),
            }
        )

    csv_path = reports / "ai_tools_glcm_lbp_svm_benchmark.csv"
    json_path = reports / "ai_tools_glcm_lbp_svm_summary.json"
    md_path = reports / "ai_tools_glcm_lbp_svm_benchmark.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["tool", "expected_label", "predicted_label", "confidence", "latency_ms", "path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, md_path)

    print(f"Processed {len(rows)} image(s).")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()

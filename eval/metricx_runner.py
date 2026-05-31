#!/usr/bin/env python3
"""Direct MetricX-23-QE inference.

Bypasses MetricX's predict.py / transformers.Trainer because that path
fights us on Apple Silicon (Trainer/Accelerate routes inputs to MPS
while predict.py places the model on CPU, causing device mismatch
errors). Here we keep complete control over the device.

Same input/output JSONL contract as predict.py:
  input:  {"source": str, "hypothesis": str}  per line
  output: {"source": str, "hypothesis": str, "prediction": float}  per line

Imports MetricX's `MT5ForRegression` model class from `.metricx/`,
so the harness must add that to PYTHONPATH before invoking this runner
(see harness.py::metricx_env).
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch
from transformers import AutoTokenizer

from metricx23.models import MT5ForRegression


def format_input(source: str, hypothesis: str, is_qe: bool) -> str:
    """Match MetricX's input formatting (see metricx23/predict.py::get_dataset)."""
    if is_qe:
        return f"candidate: {hypothesis} source: {source}"
    return f"candidate: {hypothesis} reference: {source}"


def pick_device() -> torch.device:
    """Prefer CUDA when available; otherwise CPU.

    We deliberately do NOT use MPS — MetricX's MT5ForRegression hardcodes
    a CUDA-or-default device assumption in its forward pass that breaks
    on MPS.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--max_input_length", type=int, default=1024)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument(
        "--qe",
        action="store_true",
        help="Reference-free mode (uses source as the comparison anchor).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Forward-pass batch size with proper padding.",
    )
    args = parser.parse_args()

    device = pick_device()
    print(f"[metricx_runner] device: {device}", file=sys.stderr, flush=True)
    print(
        f"[metricx_runner] loading tokenizer: {args.tokenizer}",
        file=sys.stderr,
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(
        f"[metricx_runner] loading model: {args.model_name_or_path}",
        file=sys.stderr,
        flush=True,
    )
    model = MT5ForRegression.from_pretrained(args.model_name_or_path)
    model.to(device)
    model.eval()

    # Read all inputs first so we can batch them.
    inputs: list[dict] = []
    with open(args.input_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            inputs.append(json.loads(line))

    print(
        f"[metricx_runner] scoring {len(inputs)} pairs (batch_size={args.batch_size})",
        file=sys.stderr,
        flush=True,
    )
    t0 = time.time()
    n_done = 0
    with open(args.output_file, "w") as out:
        for start in range(0, len(inputs), args.batch_size):
            batch = inputs[start : start + args.batch_size]
            texts = [format_input(o["source"], o["hypothesis"], args.qe) for o in batch]
            enc = tokenizer(
                texts,
                max_length=args.max_input_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            input_ids = enc.input_ids.to(device)
            attention_mask = enc.attention_mask.to(device)
            with torch.no_grad():
                output = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
            scores = output.predictions.detach().cpu().tolist()
            for obj, score in zip(batch, scores):
                obj_with_pred = dict(obj)
                obj_with_pred["prediction"] = float(score)
                out.write(json.dumps(obj_with_pred) + "\n")
            n_done += len(batch)
            if n_done % 25 == 0 or n_done == len(inputs):
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0.0
                print(
                    f"[metricx_runner]   {n_done}/{len(inputs)} "
                    f"({elapsed:.1f}s, {rate:.2f}/s)",
                    file=sys.stderr,
                    flush=True,
                )

    print(
        f"[metricx_runner] done: {n_done} scores in {time.time() - t0:.1f}s",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

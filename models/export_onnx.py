#!/usr/bin/env python3
"""
Export MADLAD-400-3B-MT to ONNX format via HuggingFace Optimum.

Produces three ONNX files in the output directory:
  encoder_model.onnx          (~4 GB, fp32)
  decoder_model.onnx          (~4 GB, fp32)
  decoder_with_past_model.onnx (~4 GB, fp32)

Total disk: ~12 GB.  Use --quantize for int8 CPU-optimized export (~7 GB).

Requires:
  pip install "optimum[exporters]>=1.19" transformers sentencepiece

Usage:
  python3 models/export_onnx.py --output /path/to/madlad400-3b-mt-onnx
  python3 models/export_onnx.py --output /path/to/madlad400-3b-mt-onnx --quantize
"""
import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MADLAD-400-3B-MT to ONNX")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory to write ONNX files into",
    )
    parser.add_argument(
        "--model",
        default="google/madlad400-3b-mt",
        help="HuggingFace model ID (default: google/madlad400-3b-mt)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply dynamic int8 quantization after FP32 export (CPU-optimized, ~7 GB; breaks CoreML MLProgram)",
    )
    args = parser.parse_args()

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Check dependencies ──────────────────────────────────────────────────
    try:
        from optimum.exporters.onnx import main_export  # type: ignore[import]
    except ImportError:
        print(
            "ERROR: optimum not found.\n"
            '  Install with: pip install "optimum[exporters]>=1.19" transformers sentencepiece',
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Export ──────────────────────────────────────────────────────────────
    print(f"Exporting {args.model} → {output_dir} (fp32)")
    print("This requires ~12 GB free disk space and ~8 GB RAM; expect 10–30 min.")
    print()

    main_export(
        model_name_or_path=args.model,
        output=output_dir,
        task="seq2seq-lm-with-past",   # exports all 3 graphs (encoder, decoder, decoder_with_past)
        opset=17,
        # Note: no_post_process=True prevents optimum from merging the two decoder
        # graphs into decoder_model_merged.onnx. ORT applies equivalent graph
        # optimizations at inference time anyway.
        no_post_process=True,
        # Note: do NOT pass optimize="O2" — ORT's graph optimizer strips ir_version
        # and type annotations from ONNX files, breaking both post-processing
        # (onnx.checker.check_model) and quantization.
    )

    if args.quantize:
        _quantize(output_dir)

    print()
    print("Done. Files written to:", output_dir)
    for f in sorted(output_dir.glob("*.onnx")):
        size_gb = f.stat().st_size / 1e9
        print(f"  {f.name:50s}  {size_gb:.2f} GB")


def _quantize(output_dir: Path) -> None:
    """Apply dynamic int8 quantization to all ONNX files in output_dir."""
    try:
        import onnx  # type: ignore[import]
        from onnxruntime.quantization import quantize_dynamic, QuantType  # type: ignore[import]
    except ImportError:
        print(
            "WARNING: onnxruntime not found — skipping quantization.\n"
            "  Install with: pip install onnxruntime",
            file=sys.stderr,
        )
        return

    import shutil

    for onnx_file in sorted(output_dir.glob("*.onnx")):
        if "_quant" in onnx_file.name:
            continue
        quant_path = onnx_file.with_name(onnx_file.stem + "_quant.onnx")
        print(f"Quantizing {onnx_file.name} → {quant_path.name} …")
        quantize_dynamic(
            model_input=str(onnx_file),
            model_output=str(quant_path),
            weight_type=QuantType.QInt8,
            extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
        )
        # Replace original with quantized version.
        shutil.move(str(quant_path), str(onnx_file))
        print(f"  replaced {onnx_file.name}")


if __name__ == "__main__":
    main()

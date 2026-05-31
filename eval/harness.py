#!/usr/bin/env python3
"""
Locale evaluation harness for the universal-translator engine.

Scores candidate locale codes on two dimensions:

  Phase 1 — Output-language consistency (free; local API)
    For each source sentence, translate to the candidate locale, then ask
    our own detector what language the output is. Compute the fraction of
    outputs whose detected base matches the candidate's base.

  Phase 2 — MetricX-23-QE quality estimation (free; local model)
    Reference-free quality estimation using Google's MetricX-23 QE model
    (Apache 2.0). Returns a single score per (source, hypothesis) pair on
    a roughly 0–25 scale where LOWER is better. Mean across all sentences
    is the per-candidate score.

Results CSV columns:

  code, name, tier, n_sources, n_translated, target_lang_consistency,
  metricx_n, metricx_mean, recommendation, notes

`recommendation` is one of PASS / BORDERLINE / FAIL based on configurable
thresholds (defaults: consistency ≥ 0.90 and metricx_mean ≤ 7.0 → PASS;
consistency < 0.85 or metricx_mean ≥ 10.0 → FAIL; otherwise BORDERLINE).

Required: a running translator-api on --api-url. MetricX is invoked as a
subprocess; install via `make setup` from the eval/ directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
METRICX_DIR = SCRIPT_DIR / ".metricx"

DEFAULT_API_URL = "http://localhost:3000"
DEFAULT_METRICX_MODEL = "google/metricx-23-qe-large-v2p0"
DEFAULT_METRICX_TOKENIZER = "google/mt5-xl"
DEFAULT_METRICX_BATCH = 4  # metricx_runner.py pads properly, batching is safe.
DEFAULT_METRICX_MAX_INPUT = 1024

# MetricX-23 scores: lower is better, scale roughly 0–25.
# Calibrated against TranslateGemma 4B's WMT24++-validated locales
# (fr/de/ja/pt-BR/zh-CN), which scored 0.63–1.00 — solidly in the
# "near-human" band per the MetricX paper. Thresholds below give ~3×
# headroom over that worst case for PASS, and a clear FAIL beyond ~5.
PASS_CONSISTENCY = 0.90
PASS_METRICX_MAX = 2.5
FAIL_CONSISTENCY = 0.85
FAIL_METRICX_MIN = 5.0


@dataclass
class Candidate:
    code: str
    name: str
    tier: str
    source_lang: str = "en"
    notes: str = ""


@dataclass
class CandidateResult:
    candidate: Candidate
    n_sources: int = 0
    n_translated: int = 0
    target_lang_consistency: float | None = None
    metricx_n: int = 0
    metricx_mean: float | None = None
    recommendation: str = "ERROR"
    notes: str = ""
    translations: list[dict[str, Any]] = field(default_factory=list)


def base_code(code: str) -> str:
    """Return the base language portion of a BCP 47 code (lowercase)."""
    return code.split("-")[0].split("_")[0].lower()


def metricx_quality(score: float | None) -> str:
    """Human-readable bucket for a MetricX-23 score (lower is better, 0-25 scale).

    Bands from the MetricX-23 paper, where < 2 is near-human, < 5 is good,
    < 10 is still acceptable, and ≥ 10 indicates major errors.
    """
    if score is None:
        return ""
    if score < 2.0:
        return "near-human"
    if score < 5.0:
        return "good"
    if score < 10.0:
        return "acceptable"
    return "poor"


def load_candidates(path: Path) -> list[Candidate]:
    out: list[Candidate] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get("code") or "").strip()
            if not code:
                continue
            source_lang = (row.get("source_lang") or "").strip() or "en"
            out.append(
                Candidate(
                    code=code,
                    name=(row.get("name") or "").strip(),
                    tier=(row.get("tier") or "").strip(),
                    source_lang=source_lang,
                    notes=(row.get("notes") or "").strip(),
                )
            )
    return out


def load_sources(path: Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    return [ln for ln in lines if ln]


def sources_for_lang(lang: str, default_path: Path) -> list[str]:
    """Load `sources-{lang}.txt` from the script dir; fall back to default_path
    for `en`. Cached via the @lru_cache wrapper attached at runtime."""
    if lang == "en":
        return load_sources(default_path)
    candidate_path = SCRIPT_DIR / f"sources-{lang}.txt"
    if not candidate_path.exists():
        raise FileNotFoundError(
            f"Source file for language '{lang}' not found at {candidate_path}. "
            f"Create it with sentences parallel to the English set."
        )
    return load_sources(candidate_path)


def translate_batch(
    api_url: str, sources: list[str], target: str, source_lang: str = "en"
) -> dict[str, Any]:
    """Call POST /translate. Returns the parsed response."""
    body = {
        "texts": sources,
        "target_languages": [target],
        "source_language": source_lang,
    }
    resp = requests.post(f"{api_url}/translate", json=body, timeout=600)
    resp.raise_for_status()
    return resp.json()


def detect(api_url: str, text: str) -> dict[str, Any]:
    """Call POST /detect-language. Returns the parsed response."""
    resp = requests.post(
        f"{api_url}/detect-language", json={"text": text}, timeout=60
    )
    resp.raise_for_status()
    return resp.json()


def health_check(api_url: str) -> bool:
    try:
        resp = requests.get(f"{api_url}/health", timeout=5)
        return resp.ok
    except requests.RequestException:
        return False


def metricx_available() -> bool:
    """Return True if the cloned MetricX repo + predict.py is in place.

    The MetricX repo isn't pip-installable; we run it via PYTHONPATH.
    """
    return (METRICX_DIR / "metricx23" / "predict.py").is_file()


def metricx_env() -> dict[str, str]:
    """Subprocess env with .metricx/ on PYTHONPATH so `python -m metricx23.predict` works.

    Also forces Accelerate (used by transformers Trainer) to CPU on Apple
    Silicon — `use_mps_device=False` in TrainingArguments alone doesn't stick
    because Accelerate has its own device-detection layer that ignores it.
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    if existing:
        env["PYTHONPATH"] = f"{METRICX_DIR}{os.pathsep}{existing}"
    else:
        env["PYTHONPATH"] = str(METRICX_DIR)
    # Accelerate picks MPS by default on Apple Silicon; predict.py places the
    # model on CPU explicitly, so we need to keep inputs on CPU too.
    env.setdefault("ACCELERATE_USE_CPU", "1")
    return env


def run_metricx_qe(
    pairs: list[tuple[str, str]],
    model_name: str,
    tokenizer: str,
    batch_size: int,
    max_input_length: int,
) -> list[float]:
    """Run MetricX-23-QE on all (source, hypothesis) pairs.

    Returns one score per pair, in order. Lower is better (~0–25 scale).
    Spawns a single subprocess so the model loads once for the whole batch.
    """
    if not pairs:
        return []

    # Write inputs to a temp JSONL.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as in_f:
        for src, hyp in pairs:
            in_f.write(json.dumps({"source": src, "hypothesis": hyp}) + "\n")
        in_path = in_f.name

    out_path = in_path + ".out"

    runner = SCRIPT_DIR / "metricx_runner.py"
    cmd = [
        sys.executable,
        str(runner),
        "--tokenizer",
        tokenizer,
        "--model_name_or_path",
        model_name,
        "--max_input_length",
        str(max_input_length),
        "--batch_size",
        str(batch_size),
        "--input_file",
        in_path,
        "--output_file",
        out_path,
        "--qe",
    ]
    print(
        f"\nMetricX: scoring {len(pairs)} (source, hypothesis) pairs "
        f"with {model_name}…",
        flush=True,
    )
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True, env=metricx_env())
    except subprocess.CalledProcessError as e:
        os.unlink(in_path)
        if os.path.exists(out_path):
            os.unlink(out_path)
        raise RuntimeError(f"MetricX failed: {e}") from e

    scores: list[float] = []
    with open(out_path) as f:
        for line in f:
            obj = json.loads(line)
            scores.append(float(obj["prediction"]))

    os.unlink(in_path)
    os.unlink(out_path)

    if len(scores) != len(pairs):
        raise RuntimeError(
            f"MetricX returned {len(scores)} scores for {len(pairs)} pairs"
        )
    print(f"MetricX: {len(scores)} scores in {time.time() - t0:.1f}s")
    return scores


def run_phase1_consistency(
    api_url: str, target: str, sources: list[str], translations: list[str]
) -> tuple[float, list[str]]:
    """Detect each translation, return (consistency_fraction, detected_codes)."""
    target_base = base_code(target)
    detected_codes: list[str] = []
    matches = 0
    n = 0
    for hyp in translations:
        if not hyp.strip():
            detected_codes.append("")
            continue
        try:
            det = detect(api_url, hyp)
        except requests.RequestException as e:
            detected_codes.append(f"<err:{e!s:.40}>")
            continue
        det_code = det.get("language", "")
        detected_codes.append(det_code)
        n += 1
        if base_code(det_code) == target_base:
            matches += 1
    consistency = matches / n if n else 0.0
    return consistency, detected_codes


def classify(
    consistency: float | None, metricx_mean: float | None
) -> tuple[str, str]:
    """Return (recommendation, notes)."""
    notes: list[str] = []
    if consistency is None:
        return "ERROR", "phase 1 failed"
    if metricx_mean is None:
        # Phase 2 skipped — be conservative.
        if consistency >= PASS_CONSISTENCY:
            return "BORDERLINE", "metricx skipped; consistency-only"
        if consistency < FAIL_CONSISTENCY:
            return "FAIL", f"low consistency ({consistency:.2f})"
        return "BORDERLINE", f"consistency {consistency:.2f}; metricx skipped"

    if consistency >= PASS_CONSISTENCY and metricx_mean <= PASS_METRICX_MAX:
        return "PASS", ""
    if consistency < FAIL_CONSISTENCY:
        notes.append(f"consistency {consistency:.2f} below {FAIL_CONSISTENCY}")
    if metricx_mean >= FAIL_METRICX_MIN:
        notes.append(f"metricx {metricx_mean:.2f} above {FAIL_METRICX_MIN}")
    if notes:
        return "FAIL", "; ".join(notes)
    return "BORDERLINE", f"consistency {consistency:.2f}, metricx {metricx_mean:.2f}"


def evaluate_candidate(
    candidate: Candidate,
    sources: list[str],
    api_url: str,
) -> CandidateResult:
    """Translate and run Phase 1 only. MetricX is batched after the loop."""
    result = CandidateResult(candidate=candidate, n_sources=len(sources))
    src_note = (
        f", source={candidate.source_lang}"
        if candidate.source_lang != "en"
        else ""
    )
    print(f"\n→ {candidate.code} ({candidate.name}) [tier {candidate.tier}{src_note}]")

    try:
        resp = translate_batch(
            api_url, sources, candidate.code, source_lang=candidate.source_lang
        )
    except requests.RequestException as e:
        result.notes = f"translate failed: {e}"
        print(f"  translate error: {e}", file=sys.stderr)
        return result

    translations: list[str] = []
    pairs: list[tuple[str, str]] = []
    items = resp.get("results", [])
    for src, item in zip(sources, items):
        translation_map = item.get("translations", {})
        hyp = translation_map.get(candidate.code, "")
        translations.append(hyp)
        pairs.append((src, hyp))

    result.translations = [
        {"source": s, "translation": h} for s, h in pairs
    ]
    result.n_translated = sum(1 for h in translations if h.strip())
    print(
        f"  translated {result.n_translated}/{result.n_sources} sentences"
    )

    if result.n_translated == 0:
        result.recommendation = "FAIL"
        result.notes = "no translations produced"
        return result

    try:
        consistency, detected_codes = run_phase1_consistency(
            api_url, candidate.code, sources, translations
        )
    except Exception as e:  # noqa: BLE001
        result.notes = f"phase 1 error: {e}"
        return result
    result.target_lang_consistency = consistency
    target_base = base_code(candidate.code)
    for entry, det in zip(result.translations, detected_codes):
        entry["detected_code"] = det
        entry["matches_target"] = (
            "yes" if det and base_code(det) == target_base else "no"
        )
    print(f"  phase 1 — target-language consistency: {consistency:.2%}")
    return result


def write_results(path: Path, results: list[CandidateResult]) -> None:
    fields = [
        "code",
        "name",
        "tier",
        "n_sources",
        "n_translated",
        "target_lang_consistency",
        "metricx_n",
        "metricx_mean",
        "metricx_quality",
        "recommendation",
        "notes",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "code": r.candidate.code,
                    "name": r.candidate.name,
                    "tier": r.candidate.tier,
                    "n_sources": r.n_sources,
                    "n_translated": r.n_translated,
                    "target_lang_consistency": (
                        f"{r.target_lang_consistency:.4f}"
                        if r.target_lang_consistency is not None
                        else ""
                    ),
                    "metricx_n": r.metricx_n,
                    "metricx_mean": (
                        f"{r.metricx_mean:.3f}"
                        if r.metricx_mean is not None
                        else ""
                    ),
                    "metricx_quality": metricx_quality(r.metricx_mean),
                    "recommendation": r.recommendation,
                    "notes": r.notes,
                }
            )


def write_translations(path: Path, results: list[CandidateResult]) -> None:
    """Companion artifact: per-sentence detail for manual review.

    Columns: code, name, source, translation, detected_code,
    matches_target, metricx_score. Filter `matches_target = no` rows to
    inspect Phase 1 failures; sort by `metricx_score` to find the worst
    translations Phase 2 flagged.
    """
    fields = [
        "code",
        "name",
        "source",
        "translation",
        "detected_code",
        "matches_target",
        "metricx_score",
        "metricx_quality",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            for entry in r.translations:
                metricx_score = entry.get("metricx_score")
                writer.writerow(
                    {
                        "code": r.candidate.code,
                        "name": r.candidate.name,
                        "source": entry["source"],
                        "translation": entry["translation"],
                        "detected_code": entry.get("detected_code", ""),
                        "matches_target": entry.get("matches_target", ""),
                        "metricx_score": (
                            f"{metricx_score:.3f}"
                            if metricx_score is not None
                            else ""
                        ),
                        "metricx_quality": metricx_quality(metricx_score),
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--candidates",
        type=Path,
        default=SCRIPT_DIR / "candidates.csv",
        help="CSV of candidates (columns: code, name, tier, notes). "
        "Defaults to <script_dir>/candidates.csv (gitignored).",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=SCRIPT_DIR / "sources.txt",
        help="Source sentences, one per line (default: <script_dir>/sources.txt).",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("TRANSLATOR_API_URL", DEFAULT_API_URL),
        help=f"Translator API base URL (default {DEFAULT_API_URL}).",
    )
    parser.add_argument(
        "--metricx-model",
        default=os.environ.get("METRICX_MODEL", DEFAULT_METRICX_MODEL),
        help=f"MetricX QE model name (default {DEFAULT_METRICX_MODEL}). "
        "Use the xl/xxl variants for higher fidelity at the cost of GB.",
    )
    parser.add_argument(
        "--metricx-tokenizer",
        default=os.environ.get("METRICX_TOKENIZER", DEFAULT_METRICX_TOKENIZER),
        help=f"MetricX tokenizer (default {DEFAULT_METRICX_TOKENIZER}).",
    )
    parser.add_argument(
        "--metricx-batch-size",
        type=int,
        default=DEFAULT_METRICX_BATCH,
        help=f"MetricX batch size (default {DEFAULT_METRICX_BATCH}).",
    )
    parser.add_argument(
        "--metricx-max-input",
        type=int,
        default=DEFAULT_METRICX_MAX_INPUT,
        help=f"MetricX max input length tokens (default {DEFAULT_METRICX_MAX_INPUT}).",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip Phase 2 (MetricX). Recommendation will be BORDERLINE for "
        "any code that passes Phase 1.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N candidates (useful for quick smoke runs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <script_dir>/results/results-<timestamp>.csv).",
    )
    args = parser.parse_args()

    if not args.candidates.exists():
        print(
            f"ERROR: candidates file not found: {args.candidates}\n"
            f"Hint: copy {SCRIPT_DIR / 'candidates.example.csv'} "
            f"to {SCRIPT_DIR / 'candidates.csv'} and edit.",
            file=sys.stderr,
        )
        return 2

    # English source: validate and seed the per-language cache.
    sources_en = load_sources(args.sources)
    if not sources_en:
        print(f"ERROR: no source sentences found in {args.sources}", file=sys.stderr)
        return 2
    sources_cache: dict[str, list[str]] = {"en": sources_en}

    candidates = load_candidates(args.candidates)
    if args.limit is not None:
        candidates = candidates[: args.limit]
    if not candidates:
        print("ERROR: no candidates loaded.", file=sys.stderr)
        return 2

    # Pre-load any non-en source files referenced by candidates; fail early.
    needed_langs = {c.source_lang for c in candidates if c.source_lang != "en"}
    for lang in needed_langs:
        try:
            sources_cache[lang] = sources_for_lang(lang, args.sources)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2

    if not health_check(args.api_url):
        print(
            f"ERROR: translator API not reachable at {args.api_url}.\n"
            "Start it with: cargo run -p translator-api",
            file=sys.stderr,
        )
        return 2

    if not args.skip_judge and not metricx_available():
        print(
            f"ERROR: MetricX not found at {METRICX_DIR}/metricx23/predict.py.\n"
            "Run `make setup` from the eval/ directory, or pass --skip-judge.",
            file=sys.stderr,
        )
        return 2

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output = args.output or (
        SCRIPT_DIR / "results" / f"results-{timestamp}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    src_summary = (
        f"{len(sources_en)} en"
        + "".join(
            f" + {len(sources_cache[lg])} {lg}" for lg in sorted(needed_langs)
        )
    )
    print(
        f"Candidates: {len(candidates)} | Sources: {src_summary} | "
        f"API: {args.api_url}"
    )
    if args.skip_judge:
        print("MetricX: SKIPPED")
    else:
        print(
            f"MetricX: {args.metricx_model} "
            f"(tokenizer {args.metricx_tokenizer}, batch {args.metricx_batch_size})"
        )

    # Phase 0+1: translate + detect per candidate.
    results: list[CandidateResult] = []
    for cand in candidates:
        cand_sources = sources_cache[cand.source_lang]
        try:
            res = evaluate_candidate(cand, cand_sources, args.api_url)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Writing partial results.", file=sys.stderr)
            break
        except Exception as e:  # noqa: BLE001
            res = CandidateResult(candidate=cand, n_sources=len(cand_sources))
            res.recommendation = "ERROR"
            res.notes = f"unhandled: {e}"
            print(f"  unhandled error: {e}", file=sys.stderr)
        results.append(res)

    # Phase 2: batch MetricX-23-QE on all (source, hypothesis) pairs.
    if not args.skip_judge:
        all_pairs: list[tuple[str, str]] = []
        pair_index: list[tuple[int, int]] = []  # (result_idx, entry_idx)
        for ri, r in enumerate(results):
            for ei, entry in enumerate(r.translations):
                if entry["translation"].strip():
                    all_pairs.append((entry["source"], entry["translation"]))
                    pair_index.append((ri, ei))

        if all_pairs:
            try:
                scores = run_metricx_qe(
                    all_pairs,
                    args.metricx_model,
                    args.metricx_tokenizer,
                    args.metricx_batch_size,
                    args.metricx_max_input,
                )
            except Exception as e:  # noqa: BLE001
                print(f"MetricX failed: {e}", file=sys.stderr)
                scores = []

            if scores:
                # Distribute scores back to per-sentence entries and aggregate.
                per_candidate: dict[int, list[float]] = {}
                for (ri, ei), score in zip(pair_index, scores):
                    results[ri].translations[ei]["metricx_score"] = score
                    per_candidate.setdefault(ri, []).append(score)
                for ri, scs in per_candidate.items():
                    results[ri].metricx_n = len(scs)
                    results[ri].metricx_mean = sum(scs) / len(scs)

    # Classify each candidate.
    for r in results:
        rec, notes = classify(r.target_lang_consistency, r.metricx_mean)
        r.recommendation = rec
        r.notes = notes
        line = f"  → {r.candidate.code}: {rec}"
        details: list[str] = []
        if r.target_lang_consistency is not None:
            details.append(f"consistency={r.target_lang_consistency:.2%}")
        if r.metricx_mean is not None:
            details.append(
                f"metricx={r.metricx_mean:.2f} ({metricx_quality(r.metricx_mean)})"
            )
        if details:
            line += f" — {', '.join(details)}"
        if notes:
            line += f" ({notes})"
        print(line)

    write_results(output, results)
    translations_path = output.with_name(
        output.stem.replace("results-", "translations-") + ".csv"
    )
    write_translations(translations_path, results)

    print(f"\nWrote: {output}")
    print(f"Wrote: {translations_path}")

    counts: dict[str, int] = {}
    for r in results:
        counts[r.recommendation] = counts.get(r.recommendation, 0) + 1
    print("\nSummary:")
    for k in ("PASS", "BORDERLINE", "FAIL", "ERROR"):
        if k in counts:
            print(f"  {k}: {counts[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

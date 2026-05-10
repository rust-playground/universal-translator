#!/usr/bin/env python3
"""
Locale evaluation harness for the universal-translator engine.

Scores candidate locale codes on two dimensions:

  Phase 1 — Output-language consistency (free; local API)
    For each source sentence, translate to the candidate locale, then ask
    our own detector what language the output is. Compute the fraction of
    outputs whose detected base matches the candidate's base.

  Phase 4 — LLM-as-judge sample (paid; Anthropic API)
    Sample N translations and ask Claude to score fluency and adequacy on
    a 1–5 scale. Mean across the sample is the per-candidate score.

Results CSV columns:

  code, name, tier, n_sources, n_translated, target_lang_consistency,
  judge_n, judge_mean_fluency, judge_mean_adequacy, judge_mean_overall,
  recommendation, notes

`recommendation` is one of PASS / BORDERLINE / FAIL based on configurable
thresholds (defaults: consistency ≥ 0.90, judge mean ≥ 3.5 for PASS;
consistency < 0.85 or judge < 3.0 for FAIL; otherwise BORDERLINE).

Required: a running translator-api on --api-url and ANTHROPIC_API_KEY in
the environment (unless --skip-judge is passed).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

DEFAULT_API_URL = "http://localhost:3000"
DEFAULT_JUDGE_MODEL = "claude-haiku-4-5"
DEFAULT_SAMPLE_JUDGE = 10

PASS_CONSISTENCY = 0.90
PASS_JUDGE_MEAN = 3.5
FAIL_CONSISTENCY = 0.85
FAIL_JUDGE_MEAN = 3.0


@dataclass
class Candidate:
    code: str
    name: str
    tier: str
    notes: str = ""


@dataclass
class CandidateResult:
    candidate: Candidate
    n_sources: int = 0
    n_translated: int = 0
    target_lang_consistency: float | None = None
    judge_n: int = 0
    judge_mean_fluency: float | None = None
    judge_mean_adequacy: float | None = None
    judge_mean_overall: float | None = None
    recommendation: str = "ERROR"
    notes: str = ""
    translations: list[dict[str, str]] = field(default_factory=list)


def base_code(code: str) -> str:
    """Return the base language portion of a BCP 47 code (lowercase)."""
    return code.split("-")[0].split("_")[0].lower()


def load_candidates(path: Path) -> list[Candidate]:
    out: list[Candidate] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get("code") or "").strip()
            if not code:
                continue
            out.append(
                Candidate(
                    code=code,
                    name=(row.get("name") or "").strip(),
                    tier=(row.get("tier") or "").strip(),
                    notes=(row.get("notes") or "").strip(),
                )
            )
    return out


def load_sources(path: Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    return [ln for ln in lines if ln]


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


# ── Judge ─────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are an expert linguist evaluating machine translations from English "
    "into other languages. You score on a strict rubric and respond only with "
    "JSON. Never include any prose outside the JSON object."
)

JUDGE_USER_TEMPLATE = """Evaluate this translation.

Source language: en
Target locale: {target_code} ({target_name})
Source text: {source}
Translation: {hypothesis}

Score on a 1–5 integer scale:
- fluency: how natural the output reads in the target language (1=broken/garbled, 3=understandable but awkward, 5=native-quality)
- adequacy: how completely and accurately the translation conveys the source meaning (1=wrong/missing, 3=core meaning preserved, 5=fully equivalent)

Also note any obvious issues briefly.

Respond with ONLY this JSON object, no other text:
{{"fluency": <int 1-5>, "adequacy": <int 1-5>, "issues": "<short note or empty string>"}}"""


def make_anthropic_client():
    try:
        import anthropic  # type: ignore
    except ImportError:
        print(
            "ERROR: 'anthropic' package not installed. Run "
            "`pip install -r eval/requirements.txt` "
            "or pass --skip-judge.",
            file=sys.stderr,
        )
        sys.exit(2)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "ERROR: ANTHROPIC_API_KEY not set. Export it or pass --skip-judge.",
            file=sys.stderr,
        )
        sys.exit(2)
    return anthropic.Anthropic(api_key=api_key)


def judge_one(
    client: Any,
    model: str,
    target_code: str,
    target_name: str,
    source: str,
    hypothesis: str,
) -> dict[str, Any] | None:
    """Returns parsed judge JSON or None on failure."""
    user = JUDGE_USER_TEMPLATE.format(
        target_code=target_code,
        target_name=target_name,
        source=source,
        hypothesis=hypothesis,
    )
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=200,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
    except Exception as e:  # noqa: BLE001
        print(f"  judge error: {e}", file=sys.stderr)
        return None
    if not msg.content:
        return None
    text = msg.content[0].text.strip()
    # Trim accidental markdown fences.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        print(f"  judge returned non-JSON: {text[:120]!r}", file=sys.stderr)
        return None
    fluency = parsed.get("fluency")
    adequacy = parsed.get("adequacy")
    if not isinstance(fluency, int) or not isinstance(adequacy, int):
        return None
    if not (1 <= fluency <= 5 and 1 <= adequacy <= 5):
        return None
    return {
        "fluency": fluency,
        "adequacy": adequacy,
        "issues": parsed.get("issues", ""),
    }


# ── Phase orchestration ───────────────────────────────────────────────────────


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


def run_phase4_judge(
    client: Any,
    model: str,
    candidate: Candidate,
    pairs: list[tuple[str, str]],
    sample_size: int,
) -> dict[str, Any]:
    """Sample pairs and judge each. Returns aggregate scores."""
    sample = pairs[:sample_size]
    scores_f: list[int] = []
    scores_a: list[int] = []
    issues: list[str] = []
    for src, hyp in sample:
        if not hyp.strip():
            continue
        result = judge_one(
            client, model, candidate.code, candidate.name, src, hyp
        )
        if result is None:
            continue
        scores_f.append(result["fluency"])
        scores_a.append(result["adequacy"])
        if result["issues"]:
            issues.append(result["issues"])
    n = len(scores_f)
    if n == 0:
        return {"n": 0, "fluency": None, "adequacy": None, "overall": None, "issues": []}
    mean_f = sum(scores_f) / n
    mean_a = sum(scores_a) / n
    return {
        "n": n,
        "fluency": mean_f,
        "adequacy": mean_a,
        "overall": (mean_f + mean_a) / 2,
        "issues": issues[:5],
    }


def classify(
    consistency: float | None, judge_overall: float | None
) -> tuple[str, str]:
    """Return (recommendation, notes)."""
    notes: list[str] = []
    if consistency is None:
        return "ERROR", "phase 1 failed"
    if judge_overall is None:
        # Phase 1 only — be conservative.
        if consistency >= PASS_CONSISTENCY:
            return "BORDERLINE", "judge skipped; consistency-only"
        if consistency < FAIL_CONSISTENCY:
            return "FAIL", f"low consistency ({consistency:.2f})"
        return "BORDERLINE", f"consistency {consistency:.2f}; judge skipped"

    if consistency >= PASS_CONSISTENCY and judge_overall >= PASS_JUDGE_MEAN:
        return "PASS", ""
    if consistency < FAIL_CONSISTENCY:
        notes.append(f"consistency {consistency:.2f} below {FAIL_CONSISTENCY}")
    if judge_overall < FAIL_JUDGE_MEAN:
        notes.append(f"judge {judge_overall:.2f} below {FAIL_JUDGE_MEAN}")
    if notes:
        return "FAIL", "; ".join(notes)
    return "BORDERLINE", f"consistency {consistency:.2f}, judge {judge_overall:.2f}"


def evaluate_candidate(
    candidate: Candidate,
    sources: list[str],
    api_url: str,
    judge_client: Any | None,
    judge_model: str,
    sample_size: int,
) -> CandidateResult:
    result = CandidateResult(candidate=candidate, n_sources=len(sources))
    print(f"\n→ {candidate.code} ({candidate.name}) [tier {candidate.tier}]")

    # Translate.
    try:
        resp = translate_batch(api_url, sources, candidate.code)
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
        # Some aliases (zh-Hans → zh-CN) may surface under a different key.
        if not hyp and translation_map:
            # Take the first non-empty translation if the map has only one.
            for k, v in translation_map.items():
                if v:
                    hyp = v
                    break
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

    # Phase 1.
    try:
        consistency, _ = run_phase1_consistency(
            api_url, candidate.code, sources, translations
        )
    except Exception as e:  # noqa: BLE001
        result.notes = f"phase 1 error: {e}"
        return result
    result.target_lang_consistency = consistency
    print(f"  phase 1 — target-language consistency: {consistency:.2%}")

    # Phase 4.
    if judge_client is not None:
        judged = run_phase4_judge(
            judge_client, judge_model, candidate, pairs, sample_size
        )
        result.judge_n = judged["n"]
        result.judge_mean_fluency = judged["fluency"]
        result.judge_mean_adequacy = judged["adequacy"]
        result.judge_mean_overall = judged["overall"]
        if judged["overall"] is not None:
            print(
                f"  phase 4 — judge n={judged['n']}: fluency={judged['fluency']:.2f}, "
                f"adequacy={judged['adequacy']:.2f}, overall={judged['overall']:.2f}"
            )
        else:
            print("  phase 4 — judge produced no usable scores")

    rec, notes = classify(result.target_lang_consistency, result.judge_mean_overall)
    result.recommendation = rec
    result.notes = notes
    print(f"  → {rec}{(' — ' + notes) if notes else ''}")
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────


def write_results(path: Path, results: list[CandidateResult]) -> None:
    fields = [
        "code",
        "name",
        "tier",
        "n_sources",
        "n_translated",
        "target_lang_consistency",
        "judge_n",
        "judge_mean_fluency",
        "judge_mean_adequacy",
        "judge_mean_overall",
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
                    "judge_n": r.judge_n,
                    "judge_mean_fluency": (
                        f"{r.judge_mean_fluency:.2f}"
                        if r.judge_mean_fluency is not None
                        else ""
                    ),
                    "judge_mean_adequacy": (
                        f"{r.judge_mean_adequacy:.2f}"
                        if r.judge_mean_adequacy is not None
                        else ""
                    ),
                    "judge_mean_overall": (
                        f"{r.judge_mean_overall:.2f}"
                        if r.judge_mean_overall is not None
                        else ""
                    ),
                    "recommendation": r.recommendation,
                    "notes": r.notes,
                }
            )


def write_translations(path: Path, results: list[CandidateResult]) -> None:
    """Companion artifact: full source/translation pairs for manual review."""
    fields = ["code", "name", "source", "translation"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            for entry in r.translations:
                writer.writerow(
                    {
                        "code": r.candidate.code,
                        "name": r.candidate.name,
                        "source": entry["source"],
                        "translation": entry["translation"],
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("eval/candidates.csv"),
        help="CSV of candidates (columns: code, name, tier, notes). "
        "Defaults to eval/candidates.csv (gitignored).",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=Path("eval/sources.txt"),
        help="Source sentences, one per line (default: eval/sources.txt).",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("TRANSLATOR_API_URL", DEFAULT_API_URL),
        help=f"Translator API base URL (default {DEFAULT_API_URL}).",
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        help=f"Anthropic model for judge (default {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--sample-judge",
        type=int,
        default=DEFAULT_SAMPLE_JUDGE,
        help=f"How many translations to send to the judge (default {DEFAULT_SAMPLE_JUDGE}).",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip Phase 4 (no Anthropic API calls). Recommendation will be BORDERLINE for any code that passes Phase 1.",
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
        help="Output CSV path (default: eval/results/results-<timestamp>.csv).",
    )
    args = parser.parse_args()

    if not args.candidates.exists():
        print(
            f"ERROR: candidates file not found: {args.candidates}\n"
            f"Hint: copy eval/candidates.example.csv to eval/candidates.csv "
            "and edit.",
            file=sys.stderr,
        )
        return 2

    sources = load_sources(args.sources)
    if not sources:
        print(f"ERROR: no source sentences found in {args.sources}", file=sys.stderr)
        return 2
    candidates = load_candidates(args.candidates)
    if args.limit is not None:
        candidates = candidates[: args.limit]
    if not candidates:
        print("ERROR: no candidates loaded.", file=sys.stderr)
        return 2

    if not health_check(args.api_url):
        print(
            f"ERROR: translator API not reachable at {args.api_url}.\n"
            "Start it with: cargo run -p translator-api",
            file=sys.stderr,
        )
        return 2

    judge_client = None if args.skip_judge else make_anthropic_client()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output = args.output or Path(f"eval/results/results-{timestamp}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Candidates: {len(candidates)} | Sources: {len(sources)} | API: {args.api_url}")
    if judge_client is None:
        print("Judge: SKIPPED")
    else:
        print(f"Judge: {args.judge_model} (sample size {args.sample_judge})")

    results: list[CandidateResult] = []
    for cand in candidates:
        try:
            res = evaluate_candidate(
                cand, sources, args.api_url, judge_client, args.judge_model, args.sample_judge
            )
        except KeyboardInterrupt:
            print("\nInterrupted by user. Writing partial results.", file=sys.stderr)
            break
        except Exception as e:  # noqa: BLE001
            res = CandidateResult(candidate=cand, n_sources=len(sources))
            res.recommendation = "ERROR"
            res.notes = f"unhandled: {e}"
            print(f"  unhandled error: {e}", file=sys.stderr)
        results.append(res)

    write_results(output, results)
    translations_path = output.with_name(output.stem.replace("results-", "translations-") + ".csv")
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

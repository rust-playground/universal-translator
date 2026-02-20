#!/usr/bin/env python3
"""
Integration test suite for the universal-translator CLI.

Modes:
  python tests/integration.py             # run tests against golden CSV
  python tests/integration.py --seed      # (re)generate golden CSV from current CLI output
  python tests/integration.py --verbose   # show actual vs expected on failures
  python tests/integration.py --binary PATH --models-dir PATH  # custom paths
"""

import argparse
import csv
import json
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LANGUAGES = [
    "af", "ar", "bg", "ca", "cs", "cy", "da", "de", "el", "en", "eo", "es",
    "et", "eu", "fi", "fr", "he", "hi", "hu", "hy", "id", "is", "it", "ja",
    "lt", "lv", "mk", "ml", "mr", "nl", "pt", "ro", "ru", "sk", "sq", "sv",
    "sw", "tl", "tr", "uk", "ur", "vi", "zh",
]

CSV_COLUMNS = ["input_lang", "input_text"] + LANGUAGES

# 10 diverse test inputs covering text + date + currency in multiple source languages.
TEST_INPUTS = [
    ("en", "The conference fee is $250.00 and the event is on March 3rd, 2026."),
    ("en", "Invoices over \u20ac500.00 must be paid by 31 December 2025."),
    ("fr", "Les frais de conf\u00e9rence sont de 250,00 \u20ac et l'\u00e9v\u00e9nement a lieu le 3 mars 2026."),
    ("de", "Die Konferenzgeb\u00fchr betr\u00e4gt 250,00 \u20ac und die Veranstaltung findet am 3. M\u00e4rz 2026 statt."),
    ("es", "La tarifa de la conferencia es de 250,00 \u20ac y el evento es el 3 de marzo de 2026."),
    ("ru", "\u0421\u0442\u043e\u0438\u043c\u043e\u0441\u0442\u044c \u043a\u043e\u043d\u0444\u0435\u0440\u0435\u043d\u0446\u0438\u0438 \u0441\u043e\u0441\u0442\u0430\u0432\u043b\u044f\u0435\u0442 250 \u0434\u043e\u043b\u043b\u0430\u0440\u043e\u0432, \u043c\u0435\u0440\u043e\u043f\u0440\u0438\u044f\u0442\u0438\u0435 \u043f\u0440\u043e\u0439\u0434\u0451\u0442 3 \u043c\u0430\u0440\u0442\u0430 2026 \u0433\u043e\u0434\u0430."),
    ("ar", "\u0631\u0633\u0648\u0645 \u0627\u0644\u0645\u0624\u062a\u0645\u0631 250.00 \u062f\u0648\u0644\u0627\u0631 \u0648\u0627\u0644\u062d\u062f\u062b \u0641\u064a 3 \u0645\u0627\u0631\u0633 2026."),
    ("zh", "\u4f1a\u8bae\u8d39\u7528\u4e3a250.00\u7f8e\u5143\uff0c\u6d3b\u52a8\u4e8e2026\u5e743\u67083\u65e5\u4e3e\u884c\u3002"),
    ("ja", "\u4f1a\u8b70\u306e\u53c2\u52a0\u8cbb\u306f250\u30c9\u30eb\u3067\u30012026\u5e743\u67083\u65e5\u306b\u958b\u50ac\u3055\u308c\u307e\u3059\u3002"),
    ("hi", "\u0938\u092e\u094d\u092e\u0947\u0932\u0928 \u0936\u0941\u0932\u094d\u0915 $250.00 \u0939\u0948 \u0914\u0930 \u092f\u0939 3 \u092e\u093e\u0930\u094d\u091a 2026 \u0915\u094b \u0906\u092f\u094b\u091c\u093f\u0924 \u0939\u094b\u0917\u093e\u0964"),
]

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
CSV_PATH = os.path.join(FIXTURES_DIR, "translations.csv")

DEFAULT_BINARY = os.path.join(
    os.path.dirname(__file__), "..", "target", "release", "translator"
)
DEFAULT_MODELS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "models", "opus-mt"
)


# ---------------------------------------------------------------------------
# CLI execution
# ---------------------------------------------------------------------------

def run_cli(binary: str, models_dir: str, text: str) -> dict:
    cmd = [
        binary,
        "--models-dir", models_dir,
        "translate",
        "--text", text,
        "--language", "all",
        "--output", "json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"CLI exited {result.returncode}:\n{result.stderr}"
        )
    data = json.loads(result.stdout)
    return data["results"][0]


# ---------------------------------------------------------------------------
# Seed mode
# ---------------------------------------------------------------------------

def seed(binary: str, models_dir: str) -> None:
    os.makedirs(FIXTURES_DIR, exist_ok=True)

    rows = []
    total = len(TEST_INPUTS)
    for i, (input_lang, input_text) in enumerate(TEST_INPUTS, 1):
        short = input_text[:60] + ("..." if len(input_text) > 60 else "")
        print(f"[{i}/{total}] {input_lang}: \"{short}\"")

        result = run_cli(binary, models_dir, input_text)

        detected = result.get("detected_language", "")
        if detected != input_lang:
            print(f"  WARNING: detected language '{detected}' != expected '{input_lang}'")

        errors = result.get("errors", {})
        if errors:
            failed = ", ".join(errors.keys())
            print(f"  WARNING: translation errors for: {failed}")

        translations = result.get("translations", {})
        row = {"input_lang": input_lang, "input_text": input_text}
        for lang in LANGUAGES:
            row[lang] = translations.get(lang, "")

        rows.append(row)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSeeded {len(rows)} rows → {CSV_PATH}")
    print("Review the CSV (especially ja, mr, vi, ur, cy columns) and blank out garbage cells.")


# ---------------------------------------------------------------------------
# Test mode
# ---------------------------------------------------------------------------

def test(binary: str, models_dir: str, verbose: bool) -> bool:
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: golden CSV not found at {CSV_PATH}")
        print("Run with --seed first to generate it.")
        return False

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        golden_rows = list(reader)

    total = len(golden_rows)
    passed = 0

    for i, row in enumerate(golden_rows, 1):
        input_lang = row["input_lang"]
        input_text = row["input_text"]
        short = input_text[:60] + ("..." if len(input_text) > 60 else "")
        print(f"[{i}/{total}] {input_lang}: \"{short}\"")

        result = run_cli(binary, models_dir, input_text)

        row_ok = True

        # Detection check
        detected = result.get("detected_language", "")
        if detected == input_lang:
            print(f"  \u2713 detected: {detected}")
        else:
            print(f"  \u2717 detected: expected '{input_lang}' got '{detected}'")
            row_ok = False

        # Translation checks
        translations = result.get("translations", {})
        mismatches = []
        checked = 0
        matched = 0
        for lang in LANGUAGES:
            expected = row.get(lang, "")
            if not expected:
                # Empty cell = skip
                continue
            checked += 1
            actual = translations.get(lang, "")
            if actual == expected:
                matched += 1
            else:
                mismatches.append((lang, expected, actual))

        if not mismatches:
            print(f"  \u2713 translations: {matched}/{checked} matched")
        else:
            print(f"  \u2717 translations: {matched}/{checked} matched")
            row_ok = False
            if verbose:
                for lang, expected, actual in mismatches:
                    print(f"    [{lang}] expected: {expected!r}")
                    print(f"          got:      {actual!r}")
            else:
                for lang, expected, actual in mismatches[:3]:
                    exp_short = expected[:50] + ("..." if len(expected) > 50 else "")
                    got_short = actual[:50] + ("..." if len(actual) > 50 else "")
                    print(f"    [{lang}] expected: \"{exp_short}\"  got: \"{got_short}\"")
                if len(mismatches) > 3:
                    print(f"    ... and {len(mismatches) - 3} more (use --verbose to see all)")

        if row_ok:
            passed += 1

    print(f"\nResults: {passed}/{total} passed")
    return passed == total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integration tests for universal-translator CLI."
    )
    parser.add_argument("--seed", action="store_true",
                        help="(Re)generate golden CSV from current CLI output")
    parser.add_argument("--verbose", action="store_true",
                        help="Show all actual vs expected on failures")
    parser.add_argument("--binary", default=DEFAULT_BINARY,
                        help=f"Path to translator binary (default: {DEFAULT_BINARY})")
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR,
                        help=f"Path to models directory (default: {DEFAULT_MODELS_DIR})")
    args = parser.parse_args()

    binary = os.path.realpath(args.binary)
    models_dir = os.path.realpath(args.models_dir)

    if not os.path.isfile(binary):
        print(f"ERROR: binary not found: {binary}")
        print("Build it first with: cargo build --release")
        sys.exit(1)

    if not os.path.isdir(models_dir):
        print(f"ERROR: models directory not found: {models_dir}")
        sys.exit(1)

    if args.seed:
        seed(binary, models_dir)
    else:
        ok = test(binary, models_dir, verbose=args.verbose)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

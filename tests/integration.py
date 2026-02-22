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
import platform
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

TEST_INPUTS = [
    ("en", "The sun rises in the east and sets in the west."),
    ("en", "The coffee costs $3.50 and the newspaper costs \u20ac2.00."),
    ("en", "The meeting is on Wednesday, 1 April 2026 at 10:30 AM."),
    ("en", "Name"),
    ("en", "Username"),
    ("en", "Location"),
]

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
CSV_PATH = os.path.join(FIXTURES_DIR, "translations.csv")

DEFAULT_BINARY = os.path.join(
    os.path.dirname(__file__), "..", "target", "debug", "ut"
)
if platform.system() == "Darwin":
    _cache_base = os.path.expanduser("~/Library/Caches")
else:
    _cache_base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
DEFAULT_MODELS_DIR = os.path.join(_cache_base, "ut", "models")


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

        detected = result.get("detected_language", input_lang)
        if detected != input_lang:
            print(f"  WARNING: detected language '{detected}' != expected '{input_lang}'")

        errors = result.get("errors", {})
        if errors:
            failed = ", ".join(errors.keys())
            print(f"  WARNING: translation errors for: {failed}")

        translations = result.get("translations", {})
        row = {"input_lang": detected, "input_text": input_text}
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

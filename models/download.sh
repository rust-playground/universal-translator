#!/usr/bin/env bash
# Download and convert Helsinki-NLP OPUS-MT models to CTranslate2 format.
#
# Usage:
#   bash models/download.sh                              # → ~/.cache/ut/models/ (Linux)
#                                                        # → ~/Library/Caches/ut/models/ (macOS)
#   MODELS_DIR=/custom/path bash models/download.sh      # override output directory
#   bash models/download.sh en-fr fr-en                  # specific pairs only
#
# Prerequisites: cmake, pip install ctranslate2 transformers sentencepiece torch
#
# License policy: only add models with commercially permissive licenses.
#   Standard Helsinki-NLP/opus-mt-*     → Apache-2.0
#   Helsinki-NLP/opus-mt-tc-big-*       → CC-BY-4.0
#   gsarti/opus-mt-tc-base-en-ja        → CC-BY-4.0
#   google/madlad400-3b-mt              → Apache-2.0
# Both Apache-2.0 and CC-BY-4.0 permit commercial use (attribution required).
#
# Disk requirements for MADLAD-400-3B-MT:
#   ~12 GB download from HuggingFace → ~3 GB model.bin after int8 quantization.

set -euo pipefail

# Resolve platform-appropriate cache dir (mirrors dirs::cache_dir() in Rust)
case "$(uname -s)" in
  Darwin) _cache_base="${HOME}/Library/Caches" ;;
  *)      _cache_base="${XDG_CACHE_HOME:-${HOME}/.cache}" ;;
esac
DEFAULT_MODELS_DIR="${_cache_base}/ut/models"
MODELS_DIR="${MODELS_DIR:-${DEFAULT_MODELS_DIR}}"

mkdir -p "${MODELS_DIR}"


# All known Helsinki-NLP standard simple-pair OPUS-MT models.
# Format: "src-tgt"  (maps to Helsinki-NLP/opus-mt-src-tgt)
MODELS=(
  # English → X
  en-af   # Afrikaans
  en-ar   # Arabic
  en-bg   # Bulgarian
  en-ca   # Catalan
  en-cs   # Czech
  en-cy   # Welsh
  en-da   # Danish
  en-de   # German
  en-el   # Greek
  en-eo   # Esperanto
  en-es   # Spanish
  en-et   # Estonian
  en-eu   # Basque
  en-fi   # Finnish
  en-fr   # French
  en-he   # Hebrew
  en-hi   # Hindi
  en-hu   # Hungarian
  en-hy   # Armenian
  en-id   # Indonesian
  en-is   # Icelandic
  en-it   # Italian
  en-ja   # Japanese  (HuggingFace: gsarti/opus-mt-tc-base-en-ja → overridden below)
  en-lt   # Lithuanian (HuggingFace: opus-mt-tc-big-en-lt → overridden below)
  en-lv   # Latvian    (HuggingFace: opus-mt-tc-big-en-lv → overridden below)
  en-mk   # Macedonian
  en-ml   # Malayalam
  en-mr   # Marathi
  en-mul  # Multilingual fallback (120 targets including Thai)
  en-nl   # Dutch
  en-pt   # Portuguese (HuggingFace: opus-mt-tc-big-en-pt → overridden below)
  en-ro   # Romanian
  en-ru   # Russian
  en-sk   # Slovak
  en-sq   # Albanian
  en-sv   # Swedish
  en-sw   # Swahili
  en-tl   # Filipino/Tagalog
  en-tr   # Turkish   (HuggingFace: opus-mt-tc-big-en-tr → overridden below)
  en-uk   # Ukrainian
  en-ur   # Urdu
  en-vi   # Vietnamese
  en-zh   # Chinese

  # X → English
  af-en   # Afrikaans
  ar-en   # Arabic
  bg-en   # Bulgarian
  ca-en   # Catalan
  cs-en   # Czech
  cy-en   # Welsh
  da-en   # Danish
  de-en   # German
  eo-en   # Esperanto
  es-en   # Spanish
  et-en   # Estonian
  eu-en   # Basque
  fi-en   # Finnish
  fr-en   # French
  hi-en   # Hindi
  hu-en   # Hungarian
  id-en   # Indonesian
  is-en   # Icelandic
  it-en   # Italian
  ja-en   # Japanese
  lv-en   # Latvian
  mr-en   # Marathi
  mul-en  # Multilingual→English pivot fallback (100+ source languages)
  nl-en   # Dutch
  ru-en   # Russian
  sk-en   # Slovak
  sv-en   # Swedish
  sw-en   # Swahili  (HuggingFace: opus-mt-swc-en → overridden below)
  tl-en   # Filipino/Tagalog
  tr-en   # Turkish
  uk-en   # Ukrainian
  ur-en   # Urdu
  vi-en   # Vietnamese
  zh-en   # Chinese
)

# If specific pairs are given as arguments, use those instead
if [[ $# -gt 0 ]]; then
  MODELS=("$@")
fi

PASS=()
SKIP=()
FAIL=()

for pair in "${MODELS[@]}"; do
  src="${pair%-*}"
  tgt="${pair#*-}"
  # Some pairs have non-standard HuggingFace model IDs; others follow Helsinki-NLP/opus-mt-{src}-{tgt}.
  case "${pair}" in
    en-ja) model="gsarti/opus-mt-tc-base-en-ja" ;;      # Helsinki tc-base; tc-big-en-ja does not exist publicly
    en-lt) model="Helsinki-NLP/opus-mt-tc-big-en-lt" ;;
    en-lv) model="Helsinki-NLP/opus-mt-tc-big-en-lv" ;;    # confirmed tc-big variant
    en-pt) model="Helsinki-NLP/opus-mt-tc-big-en-pt" ;;
    en-tr) model="Helsinki-NLP/opus-mt-tc-big-en-tr" ;;
    sw-en) model="Helsinki-NLP/opus-mt-swc-en" ;;          # HF uses "swc" code
    *)     model="Helsinki-NLP/opus-mt-${src}-${tgt}" ;;
  esac
  out_dir="${MODELS_DIR}/${pair}"

  # Skip if already converted (model.bin present)
  if [[ -f "${out_dir}/model.bin" ]]; then
    echo "SKIP  ${pair}  (already exists)"
    SKIP+=("${pair}")
    continue
  fi

  echo "━━━  ${pair}  (${model})"
  # Capture combined stdout+stderr, suppressing noisy warnings
  convert_output=$(ct2-transformers-converter \
      --model "${model}" \
      --output_dir "${out_dir}" \
      --quantization float32 \
      --copy_files source.spm target.spm \
      --force 2>&1) && convert_ok=true || convert_ok=false
  # Print output with warnings stripped
  echo "${convert_output}" \
    | grep -v "NotOpenSSLWarning" \
    | grep -v "torch_dtype.*deprecated" \
    | grep -v "Recommended: pip install sacremoses" \
    | grep -v "warnings.warn" \
    || true
  if $convert_ok; then
    echo "OK    ${pair}"
    PASS+=("${pair}")
  else
    echo "FAIL  ${pair}"
    FAIL+=("${pair}")
    # Remove incomplete output dir so it won't look valid later
    rm -rf "${out_dir}"
  fi
  echo ""
done

# ── MADLAD-400-3B-MT ─────────────────────────────────────────────────────────
# Single model covering 400+ languages; replaces all per-pair en-X models.
# Disk: ~12 GB download, ~3 GB output (int8). Runtime RAM: ~3 GB (always hot).
MADLAD_DIR="${MODELS_DIR}/madlad400-3b-mt"
if [[ ! -f "${MADLAD_DIR}/model.bin" ]]; then
  echo "━━━  madlad400-3b-mt  (google/madlad400-3b-mt)"
  ct2-transformers-converter \
      --model google/madlad400-3b-mt \
      --output_dir "${MADLAD_DIR}" \
      --quantization int8 \
      --copy_files spiece.model \
      --force
  # SpmTokenizer expects source.spm / target.spm; MADLAD uses one shared model
  cp "${MADLAD_DIR}/spiece.model" "${MADLAD_DIR}/source.spm"
  cp "${MADLAD_DIR}/spiece.model" "${MADLAD_DIR}/target.spm"
  echo "OK    madlad400-3b-mt"
else
  echo "SKIP  madlad400-3b-mt  (already exists)"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Results: ${#PASS[@]} converted, ${#SKIP[@]} skipped, ${#FAIL[@]} failed"
[[ ${#PASS[@]} -gt 0 ]] && echo "  converted: ${PASS[*]}"
[[ ${#SKIP[@]} -gt 0 ]] && echo "  skipped:   ${SKIP[*]}"
[[ ${#FAIL[@]} -gt 0 ]] && echo "  failed:    ${FAIL[*]}"

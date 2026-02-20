#!/usr/bin/env bash
# Download and convert Helsinki-NLP OPUS-MT models to CTranslate2 format.
#
# Usage:
#   ./models/download.sh              # convert all pairs in MODELS list
#   ./models/download.sh en-fr fr-en  # convert specific pairs only
#
# Run from the project root:
#   bash models/download.sh
#
# Prerequisites: pip install ctranslate2 transformers sentencepiece torch
#
# License policy: only add models with commercially permissive licenses.
#   Standard Helsinki-NLP/opus-mt-*     → Apache-2.0
#   Helsinki-NLP/opus-mt-tc-big-*       → CC-BY-4.0
#   gsarti/opus-mt-tc-base-en-ja        → CC-BY-4.0
# Both Apache-2.0 and CC-BY-4.0 permit commercial use (attribution required).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}"


# All known Helsinki-NLP standard simple-pair OPUS-MT models.
# Format: "src-tgt"  (maps to Helsinki-NLP/opus-mt-src-tgt)
MODELS=(
  # English → X
  en-af   # Afrikaans
  en-ar   # Arabic
  en-bg   # Bulgarian
  en-ca   # Catalan
  en-cs   # Czech
  en-da   # Danish
  en-de   # German
  en-el   # Greek
  en-es   # Spanish
  en-et   # Estonian
  en-fi   # Finnish
  en-fr   # French
  en-he   # Hebrew
  en-hi   # Hindi
  en-hu   # Hungarian
  en-id   # Indonesian
  en-it   # Italian
  en-ja   # Japanese  (HuggingFace: opus-mt-tc-big-en-ja → overridden below)
  en-ko   # Korean    (HuggingFace: opus-mt-tc-big-en-ko → overridden below)
  en-lt   # Lithuanian (HuggingFace: opus-mt-tc-big-en-lt → overridden below)
  en-ml   # Malayalam
  en-mr   # Marathi
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
  en-mul  # Multilingual fallback (120 targets including Thai)
  en-lv   # Latvian    (HuggingFace: opus-mt-tc-big-en-lv → overridden below)
  en-mk   # Macedonian
  en-is   # Icelandic
  en-cy   # Welsh
  en-mt   # Maltese     (confirmed: Helsinki-NLP/opus-mt-en-mt)
  en-gl   # Galician
  en-eu   # Basque
  en-eo   # Esperanto
  en-hy   # Armenian

  # X → English
  af-en   # Afrikaans
  ar-en   # Arabic
  bg-en   # Bulgarian
  bn-en   # Bengali
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
  ko-en   # Korean
  lv-en   # Latvian
  mr-en   # Marathi
  mt-en   # Maltese
  nl-en   # Dutch
  pl-en   # Polish
  ru-en   # Russian
  sk-en   # Slovak
  sv-en   # Swedish
  th-en   # Thai
  tl-en   # Filipino/Tagalog
  tr-en   # Turkish
  uk-en   # Ukrainian
  ur-en   # Urdu
  vi-en   # Vietnamese
  zh-en   # Chinese
  mul-en  # Multilingual→English pivot fallback (100+ source languages)
  sw-en   # Swahili  (HuggingFace: opus-mt-swc-en → overridden below)
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
    en-ko) model="Helsinki-NLP/opus-mt-tc-big-en-ko" ;;
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

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Results: ${#PASS[@]} converted, ${#SKIP[@]} skipped, ${#FAIL[@]} failed"
[[ ${#PASS[@]} -gt 0 ]] && echo "  converted: ${PASS[*]}"
[[ ${#SKIP[@]} -gt 0 ]] && echo "  skipped:   ${SKIP[*]}"
[[ ${#FAIL[@]} -gt 0 ]] && echo "  failed:    ${FAIL[*]}"

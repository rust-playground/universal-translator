# Third-Party Model Attributions

This project uses pre-trained machine translation models and libraries at runtime.
The model weights are not distributed with this source code — they are downloaded
separately via `models/download.sh`. Licenses and attribution requirements for
each component are listed below.

---

## google/translategemma-4b-it

**License:** Gemma Terms of Use (not Apache/MIT)
**Source:** https://huggingface.co/google/translategemma-4b-it
**Used for:** Tokenizer configuration (tokenizer.json, config.json) — downloaded at
model setup. The model is an instruction-tuned Gemma 3 4B fine-tuned for translation.

> See [LICENSE-GEMMA](LICENSE-GEMMA) and [NOTICE](NOTICE) for the full terms.
> You must accept the Gemma Terms of Use before downloading model files.

---

## mradermacher/translategemma-4b-it-GGUF

**License:** Gemma Terms of Use (inherited from source; GGUF conversion by mradermacher)
**Source:** https://huggingface.co/mradermacher/translategemma-4b-it-GGUF
**Used for:** The `model-q4k.gguf` file (~2.6 GB) downloaded at model setup —
community Q4_K_M GGUF quantisation enabling Candle inference without Python tooling.

---

## Candle (inference framework)

**License:** MIT OR Apache-2.0
**Source:** https://github.com/huggingface/candle
**Used for:** All model inference at runtime (candle-core, candle-transformers,
candle-nn crates)

No additional attribution action is required by the MIT/Apache-2.0 licenses.

---

## HuggingFace Tokenizers

**License:** Apache 2.0
**Source:** https://github.com/huggingface/tokenizers
**Used for:** Fast tokenisation via the `tokenizers` Rust crate and `tokenizer.json`

---

## Lingua

**License:** Apache 2.0
**Source:** https://github.com/pemistahl/lingua-rs
**Used for:** Automatic source-language detection, covering 75+ languages entirely
offline

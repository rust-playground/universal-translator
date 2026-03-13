# Third-Party Model Attributions

This project uses pre-trained machine translation models and libraries at runtime.
The model weights are not distributed with this source code — they are downloaded
separately via `ut setup`. Licenses and attribution requirements for
each component are listed below.

---

## google/translategemma-4b-it

**License:** Gemma Terms of Use (not Apache/MIT)
**Source:** https://huggingface.co/google/translategemma-4b-it
**Used for:** GGUF quantised model weights and config files — downloaded at model setup.
The model is an instruction-tuned Gemma 3 4B fine-tuned for translation.

> See [LICENSE-GEMMA](LICENSE-GEMMA) and [NOTICE](NOTICE) for the full terms.
> You must accept the Gemma Terms of Use before downloading model files.

---

## mradermacher/translategemma-4b-it-GGUF

**License:** Gemma Terms of Use (inherited from source; GGUF conversion by mradermacher)
**Source:** https://huggingface.co/mradermacher/translategemma-4b-it-GGUF
**Used for:** Community GGUF quantisations (Q8_0 and Q4_K_M) for llama.cpp inference.

---

## llama.cpp (inference framework)

**License:** MIT
**Source:** https://github.com/ggerganov/llama.cpp
**Used for:** All model inference at runtime via the `llama-cpp-2` Rust crate

---

## Lingua

**License:** Apache 2.0
**Source:** https://github.com/pemistahl/lingua-rs
**Used for:** Automatic source-language detection, covering 75+ languages entirely
offline

# Third-Party Model Attributions

This project uses pre-trained machine translation models and libraries at runtime.
The model weights are not distributed with this source code — they are downloaded
separately via `models/download.sh`. Licenses and attribution requirements for
each component are listed below.

---

## google/madlad400-3b-mt

**License:** Apache 2.0
**Source:** https://huggingface.co/google/madlad400-3b-mt
**Used for:** All translation inference — the underlying model weights

> Kudugunta, S. et al. (2024). MADLAD-400: A Multilingual And Document-Level Large
> Audited Dataset. *Advances in Neural Information Processing Systems 36*.

---

## jbochi/madlad400-3b-mt (GGUF conversion)

**License:** Apache 2.0 (inherits from source)
**Source:** https://huggingface.co/jbochi/madlad400-3b-mt
**Used for:** The `model-q4k.gguf` file downloaded at runtime — community int4
quantisation of the above model in GGUF format, enabling Candle inference without
Python tooling.

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

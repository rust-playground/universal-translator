# Third-Party Model Attributions

This project uses pre-trained machine translation models at runtime. The models
themselves are not distributed with this source code — they are downloaded
separately via `models/download.sh`. Licenses and attribution requirements for
each model family are listed below.

---

## Helsinki-NLP OPUS-MT (standard models)

**License:** Apache 2.0
**Source:** https://huggingface.co/Helsinki-NLP
**Used for:** Most language pairs (af, ar, bg, ca, cs, cy, da, de, el, eo, es,
et, eu, fi, fr, gl, he, hi, hu, hy, id, is, it, lt, mk, ml, mr, mt, nl, pt,
ro, ru, sk, sq, sv, sw, tl, tr, uk, ur, vi, zh, and corresponding X→en models)

> Tiedemann, J., & Thottingal, S. (2020). OPUS-MT — Building open translation
> services for the World. In *Proceedings of the 22nd Annual Conference of the
> European Association for Machine Translation*, pp. 479–480.

---

## Helsinki-NLP OPUS-MT TC-Big models

**License:** CC-BY 4.0
**Source:** https://huggingface.co/Helsinki-NLP
**Used for:** en-ko (Korean), en-lt (Lithuanian), en-lv (Latvian),
en-pt (Portuguese), en-tr (Turkish)

> Tiedemann, J. (2020). The Tatoeba Translation Challenge — Realistic Data Sets
> for Low Resource and Multilingual MT. In *Proceedings of the Fifth Conference
> on Machine Translation (WMT20)*.

Attribution is required under CC-BY 4.0. Credit: Language Technology Research
Group at the University of Helsinki.

---

## gsarti/opus-mt-tc-base-en-ja

**License:** CC-BY 4.0
**Source:** https://huggingface.co/gsarti/opus-mt-tc-base-en-ja
**Used for:** en-ja (Japanese)
**Original model:** Helsinki-NLP MarianMT tc-base, converted by Gabriele Sarti

> Tiedemann, J. (2020). The Tatoeba Translation Challenge — Realistic Data Sets
> for Low Resource and Multilingual MT. In *Proceedings of the Fifth Conference
> on Machine Translation (WMT20)*.

Attribution is required under CC-BY 4.0. Credit: Language Technology Research
Group at the University of Helsinki; conversion by Gabriele Sarti.

---

## Helsinki-NLP/opus-mt-swc-en

**License:** Apache 2.0
**Source:** https://huggingface.co/Helsinki-NLP/opus-mt-swc-en
**Used for:** sw-en (Swahili → English pivot)

---

## CTranslate2 (runtime inference engine)

**License:** MIT
**Source:** https://github.com/OpenNMT/CTranslate2
**Used for:** All model inference at runtime via the `ct2rs` Rust bindings

No additional attribution action is required by the MIT license.

# models/

This directory holds converted CTranslate2 model directories. Each subdirectory is one language pair.

For full instructions on converting and installing models see [docs/models.md](../docs/models.md).

---

## Currently available models

| Directory | Language Pair |
|-----------|---------------|
| `en-fr/` | English -> French |

---

## Expected directory layout

```
models/
└── en-fr/
    ├── model.bin              # CTranslate2 weights
    ├── source.spm             # Source SentencePiece model
    ├── target.spm             # Target SentencePiece model
    ├── config.json
    └── shared_vocabulary.json
```

Model directories are not checked into version control. See [docs/models.md](../docs/models.md) to generate them.

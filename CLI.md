# CLI Reference

The `ut` binary is built from the `translator-cli` crate.

```bash
cargo build -p translator-cli
# binary at ./target/debug/ut (or ./target/release/ut with --release)
```

---

## Global flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path PATH` | `<cache>/ut/models/translategemma-4b/model-q8_0.gguf`¹ | Full path to the GGUF model file |

¹ `~/.cache/ut/models/…` on Linux, `~/Library/Caches/ut/models/…` on macOS.
Override with the `MODEL_PATH` environment variable or this flag.

---

## Subcommands

### translate

Translate one or more texts into one or more target languages.

```
ut translate -t TEXT [-t TEXT ...] -l LANG[,LANG...] [-s SOURCE] [--output pretty|json]
```

| Flag | Description |
|------|-------------|
| `-t`, `--text TEXT` | Text to translate. Required. Repeat for multiple inputs. |
| `-l`, `--language LANG` | Target BCP 47 code (`fr`, `pt-BR`, `zh-Hant`). Required. Comma-separated or repeated. Dash and underscore both accepted; case-insensitive. |
| `-s`, `--source LANG` | Source BCP 47 code. Skips auto-detection when supplied. All texts in the batch are assumed to be in this language. |
| `--output pretty\|json` | Output format (default: `pretty`) |

**Pretty output** (default):

```
Source [en]: Hello world
  [fr] Bonjour le monde
  [de] Hallo Welt
```

**JSON output** (`--output json`) — `TranslationResultSet`:

```json
[
  {
    "source_text": "Hello world",
    "detected_language": "en",
    "translations": {
      "fr": "Bonjour le monde",
      "de": "Hallo Welt"
    },
    "errors": {}
  }
]
```

**Examples**

```bash
# Translate to French
ut translate -t "Hello world" -l fr

# Multiple target languages — comma-separated or repeated flag
ut translate -t "Hello world" -l fr,de,ja
ut translate -t "Hello world" -l fr -l de -l ja

# Regional variants — pt-BR vs pt-PT, zh-Hant vs zh-Hans
ut translate -t "Hello world" -l pt-BR,pt-PT
ut translate -t "Hello world" -l zh-Hant,zh-Hans

# Underscore form is also accepted
ut translate -t "Hello world" -l pt_BR,fr_CA

# Multiple input texts
ut translate -t "Hello world" -t "Good morning" -l fr

# Supply known source language (skips auto-detection)
ut translate -t "Bonjour le monde" -s fr -l en,de

# JSON output (useful for scripting)
ut translate -t "Hello world" -l fr --output json
```

---

### detect-language

Detect the language of a text and return full details including confidence and whether
the language is supported for translation.

```
ut detect-language TEXT [--output pretty|json]
```

| Argument / Flag | Description |
|----------------|-------------|
| `TEXT` | Positional — text whose language to detect |
| `--output pretty\|json` | Output format (default: `pretty`) |

**Pretty output** (default):

```
Language: French (fr) — confidence: 87.3%
```

When the detector emits an alias whose translate-side equivalent is a
different code (e.g. Bokmål `nb` → `no`, Tagalog `tl` → `fil`), the line
includes a `translate as` clause:

```
Language: Bokmal (nb) — translate as: no — confidence: 92.0%
```

For text the detector recognizes but translate can't accept (e.g. Welsh):

```
Language: Welsh (cy) — confidence: 91.0% — translation supported: no
```

**JSON output** (`--output json`) — `LanguageDetectionResult`:

```json
{
  "language": "fr",
  "translate_language": "fr",
  "confidence": 0.873
}
```

`language` is the raw BCP 47 code from the detector (may include script
subtags like `zh-CN` or heuristic regional tags like `pt-BR`).
`translate_language` is the same code mapped into the translate-side
`Language` enum via the standard FromStr aliases (`nb`/`nn` → `no`,
`tl` → `fil`, `iw` → `he`, etc.); `null` for lingua-only languages the
engine can't translate from. Pass either form to `ut translate -s …` —
both work via FromStr.

`confidence` is a relative score in `[0, 1]`. See
[API.md — confidence score semantics](API.md#confidence-score-semantics) for the
interpretation table.

**Examples**

```bash
ut detect-language "Bonjour le monde"
ut detect-language "مرحبا بالعالم" --output json
```

---

### detect

Detect the language of a text and print only the BCP 47 code. Lightweight
alternative to `detect-language` when you only need the code. Returned codes
may include script tags (`zh-Hant`, `sr-Cyrl`) or heuristic regional tags
(`pt-BR`, `en-US`, `fr-CA`).

```
ut detect TEXT [--output pretty|json]
```

| Argument / Flag | Description |
|----------------|-------------|
| `TEXT` | Positional — text whose language to detect |
| `--output pretty\|json` | Output format (default: `pretty`) |

**Pretty output** (default):

```
Detected language: fr
```

**JSON output** (`--output json`):

```json
{
  "text": "Bonjour le monde",
  "detected_language": "fr"
}
```

**Examples**

```bash
ut detect "Bonjour le monde"
ut detect "Hello world" --output json
```

---

### languages

List supported languages for either translate or detect.

```
ut languages [--for translate|detect] [--filter STRING] [--output pretty|json]
```

| Flag | Description |
|------|-------------|
| `--for translate\|detect` | Which side's list to print (default: `translate`). The detect list is broader: lingua's 75 base languages plus deterministic script refinements (`zh-Hant`, `sr-Cyrl`) and best-effort heuristic regional codes (`pt-BR`, `en-US`). Codes from the detect list may not round-trip into translation. |
| `--filter STRING` | Case-insensitive substring filter on code or language name |
| `--output pretty\|json` | Output format (default: `pretty`) |

**Pretty output** (default):

```
af         Afrikaans
ar         Arabic
ar-EG      Egyptian Arabic
ar-SA      Saudi Arabic
...
pt-BR      Brazilian Portuguese
pt-PT      European Portuguese
...
70 language(s)
```

**JSON output** (`--output json`):

```json
[
  { "code": "af", "name": "Afrikaans" },
  { "code": "ar", "name": "Arabic" },
  { "code": "ar-EG", "name": "Egyptian Arabic" },
  ...
]
```

**Examples**

```bash
# List translate-supported languages (default)
ut languages

# List detect-supported codes (broader; includes zh-Hant, sr-Cyrl, cy, ka, etc.)
ut languages --for detect

# Filter by name or code
ut languages --filter french
ut languages --filter zh

# JSON output
ut languages --output json
```

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
| `--models-dir PATH` | Platform cache dir¹ | Directory containing model files |

¹ `~/.cache/ut/models` on Linux, `~/Library/Caches/ut/models` on macOS.
Override with the `MODELS_DIR` environment variable or this flag.

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
| `-l`, `--language LANG` | Target language ISO 639-1 code. Required. Comma-separated or repeated. |
| `-s`, `--source LANG` | Source language ISO 639-1 code. Skips auto-detection when supplied. All texts in the batch are assumed to be in this language. |
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
Language: French (fr) — confidence: 87.3% — translation supported: yes
```

**JSON output** (`--output json`) — `LanguageDetectionResult`:

```json
{
  "language_code": "fr",
  "language": "French",
  "confidence": 0.873,
  "translation_supported": true
}
```

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

Detect the language of a text and print only the ISO 639-1 language code. Lightweight
alternative to `detect-language` when you only need the code.

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

List all supported languages.

```
ut languages [--filter STRING] [--output pretty|json]
```

| Flag | Description |
|------|-------------|
| `--filter STRING` | Case-insensitive substring filter on code or language name |
| `--output pretty\|json` | Output format (default: `pretty`) |

**Pretty output** (default):

```
af     afrikaans
am     amharic
ar     arabic
...
55 language(s)
```

**JSON output** (`--output json`):

```json
[
  { "code": "af", "name": "afrikaans" },
  { "code": "am", "name": "amharic" },
  ...
]
```

**Examples**

```bash
# List all languages
ut languages

# Filter by name or code
ut languages --filter french
ut languages --filter zh

# JSON output
ut languages --output json
```

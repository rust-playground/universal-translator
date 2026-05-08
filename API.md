# API Reference

## Base URL / Content-Type

Default base URL: `http://localhost:3000`

All request and response bodies use `Content-Type: application/json`.

---

## Endpoints

### GET /health

Returns `200 OK` with a plain-text body when the server is running. Use this for
liveness checks.

**Response**

```
HTTP/1.1 200 OK
Content-Type: text/plain

OK
```

**Example**

```bash
curl http://localhost:3000/health
```

---

### GET /languages

Returns the list of supported BCP 47 language/locale codes with English names.

**Query parameters**

| Param | Default | Description |
|-------|---------|-------------|
| `for` | `translate` | `translate` returns the 70-entry translate-supported set (the `Language` enum). `detect` returns the broader detect-supported list (lingua's 75 base languages plus deterministic script refinements like `zh-Hant`, `sr-Cyrl`, and best-effort heuristic dialect codes like `pt-BR`, `en-US`). |

**Response**

```json
{
  "languages": [
    {"code": "af", "name": "Afrikaans"},
    {"code": "ar", "name": "Arabic"},
    {"code": "ar-EG", "name": "Egyptian Arabic"},
    {"code": "ar-SA", "name": "Saudi Arabic"},
    {"code": "...", "name": "..."},
    {"code": "pt-BR", "name": "Brazilian Portuguese"},
    {"code": "pt-PT", "name": "European Portuguese"},
    {"code": "zh-CN", "name": "Simplified Chinese"},
    {"code": "zh-TW", "name": "Traditional Chinese"}
  ]
}
```

Codes returned by `?for=detect` may include entries the translate side rejects
(e.g. `cy` Welsh, `ka` Georgian, `eu` Basque). Use `?for=translate` to know
what `/translate` will accept.

**Example**

```bash
curl http://localhost:3000/languages
curl 'http://localhost:3000/languages?for=detect'
```

---

### POST /translate

Translate one or more texts into one or more target languages.

**Request body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `texts` | `string[]` | Yes | Non-empty array of source texts to translate |
| `target_languages` | `string[]` | Yes | BCP 47 codes (e.g. `fr`, `pt-BR`, `zh-Hant`). Dash and underscore both accepted; case-insensitive. Unknown region tags fall back to the base language. |
| `source_language` | `string` | No | BCP 47 code of the source — skips auto-detection. |

```json
{
  "texts": ["Hello world", "How are you?"],
  "target_languages": ["fr", "pt-BR", "zh-Hant"],
  "source_language": "en"
}
```

**Response**

`TranslationResultSet` — one result per input text, in the same order.

```json
{
  "results": [
    {
      "source_text": "Hello world",
      "detected_language": "en",
      "translations": {
        "fr": "Bonjour le monde",
        "de": "Hallo Welt"
      },
      "errors": {}
    },
    {
      "source_text": "How are you?",
      "detected_language": "en",
      "translations": {
        "fr": "Comment allez-vous ?",
        "de": "Wie geht es Ihnen?"
      },
      "errors": {}
    }
  ]
}
```

The `errors` key maps target language codes to structured error objects for any
translations that failed individually. Each error is a JSON object with `type` and
`message` fields, e.g. `{"type":"TranslationFailed","message":"inference timeout"}`.
Possible types: `DetectionFailed`, `UnsupportedLanguage`, `TranslationFailed`.
The key is omitted from the response when empty.

**Example**

```bash
curl -X POST http://localhost:3000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello world"],
    "target_languages": ["fr", "de", "ja"]
  }'
```

---

### POST /detect-language

Detect the language of a piece of text.

**Request body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | `string` | Yes | The text whose language to detect |

```json
{
  "text": "Bonjour le monde"
}
```

**Response**

`LanguageDetectionResult`

```json
{
  "language": "pt-BR",
  "translate_language": "pt-BR",
  "confidence": 0.94
}
```

The two language fields exist because the detector's output universe is
broader than the translate-side `Language` enum. They're equal in most
cases. They differ when the detector emits an alias of a translate-supported
code (e.g. Bokmål/Tagalog), or when the detector identifies a language the
engine can't translate from:

```json
{ "language": "nb",   "translate_language": "no",   "confidence": 0.92 }
{ "language": "tl",   "translate_language": "fil",  "confidence": 0.94 }
{ "language": "cy",   "translate_language": null,   "confidence": 0.88 }
```

| Field | Type | Description |
|-------|------|-------------|
| `language` | `string` | Raw BCP 47 code from the detector. May include script subtags (`zh-CN`, `sr-Cyrl`), heuristic regional tags (`pt-BR`, `en-US`), or lingua-only codes outside the translate set (`cy`, `ka`, `nb`, `tl`). The most precise signal we have about the input. |
| `translate_language` | `string \| null` | Translate-side enum equivalent of `language`, applying FromStr aliases (`nb`/`nn` → `no`, `tl` → `fil`, `iw` → `he`, `zh-Hans` → `zh-CN`, `pt-AO` → `pt`, etc.). `null` for lingua-only languages the engine can't translate from. Use this when you need a code `/translate` accepts as `source_language`. |
| `confidence` | `number` | Relative confidence score in `[0, 1]` — see below |

> **Breaking change in this release:** the previous `translation_supported`
> boolean has been removed. Check `translate_language !== null` instead.

**Confidence score semantics**

The `confidence` field is a **relative** score: `top / (top + second)`, where `top`
and `second` are the raw Lingua probability scores for the first- and second-ranked
candidate languages.

This answers "how clearly does the top language beat its nearest competitor?" rather
than reporting an absolute probability. Because Lingua's raw scores are spread across
~75 languages and sum to 1.0, a raw absolute score of 17% is actually a strong signal
(~13× the average); the relative formula converts that to ~73%, which is more
human-readable.

| Range | Meaning |
|-------|---------|
| > 0.90 | Strong, unambiguous signal |
| 0.70–0.90 | Confident — short or common phrases often land here |
| 0.50–0.70 | Moderate — treat as a best guess |
| < 0.50 | Weak — text is very short or genuinely ambiguous |

Short common phrases (e.g. "Hello, how are you?") may score in the 70–80% range even
when correctly detected; this is expected. Longer or script-distinctive text (e.g.
Japanese, Arabic, Malayalam) will score 95%+.

**Example**

```bash
curl -X POST http://localhost:3000/detect-language \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour le monde"}'
```

---

## Error responses

On failure, the API returns a non-2xx HTTP status with a JSON body:

```json
{
  "error": "description of what went wrong"
}
```

Common status codes:

| Status | Meaning |
|--------|---------|
| 400 | Bad request — missing or invalid fields in the request body |
| 422 | Unprocessable entity — request parsed but semantically invalid (e.g. unknown language code) |
| 500 | Internal server error — inference or model error |

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

Returns the list of supported ISO 639-1 language codes.

**Response**

```json
{
  "languages": ["af", "am", "ar", "bg", "bn", "ca", "cs", "da", "de", "el",
                 "en", "es", "et", "fa", "fi", "fr", "gu", "ha", "hi", "hr",
                 "hu", "id", "it", "ja", "kn", "ko", "lt", "lv", "ml", "mr",
                 "ms", "mt", "ne", "nl", "no", "pa", "pl", "pt", "ro", "ru",
                 "si", "sk", "sl", "sr", "sv", "sw", "ta", "te", "th", "tr",
                 "uk", "ur", "vi", "yi", "zh"]
}
```

**Example**

```bash
curl http://localhost:3000/languages
```

---

### POST /translate

Translate one or more texts into one or more target languages.

**Request body**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `texts` | `string[]` | Yes | Non-empty array of source texts to translate |
| `target_languages` | `string[]` | Yes | ISO 639-1 codes of target languages |
| `source_language` | `string` | No | ISO 639-1 code of the source language — skips auto-detection when already known |

```json
{
  "texts": ["Hello world", "How are you?"],
  "target_languages": ["fr", "de"],
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

The `errors` key maps target language codes to error messages for any translations
that failed individually. It is omitted from the response when empty.

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
  "language_code": "fr",
  "language": "French",
  "confidence": 0.87,
  "translation_supported": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `language_code` | `string` | ISO 639-1 code of the detected language |
| `language` | `string` | Full English name of the detected language |
| `confidence` | `number` | Relative confidence score in `[0, 1]` — see below |
| `translation_supported` | `bool` | Whether the detected language can be used as a source or target for `/translate` |

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

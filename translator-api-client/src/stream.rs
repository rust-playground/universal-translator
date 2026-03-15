use futures::stream::Stream;
use reqwest::Response;
use translator_core::types::TranslationResult;

use crate::error::ClientError;

/// Parse an SSE stream from the `/translate/stream` endpoint.
///
/// Yields one `TranslationResult` per `event: translation` SSE event.
/// Ends on `event: done` or stream EOF.
pub fn parse_sse_stream(
    response: Response,
) -> impl Stream<Item = Result<TranslationResult, ClientError>> {
    let byte_stream = response.bytes_stream();

    futures::stream::unfold(
        (byte_stream, String::new()),
        |(mut byte_stream, mut buffer)| async move {
            loop {
                // Try to extract a complete SSE event from the buffer.
                if let Some(pos) = buffer.find("\n\n") {
                    let block = buffer[..pos].to_string();
                    buffer = buffer[pos + 2..].to_string();

                    if let Some(result) = parse_sse_block(&block) {
                        return Some((result, (byte_stream, buffer)));
                    }
                    // Non-yielding event (e.g. `done`), continue parsing.
                    continue;
                }

                // Need more data from the stream.
                use futures::StreamExt as _;
                match byte_stream.next().await {
                    Some(Ok(bytes)) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));
                    }
                    Some(Err(e)) => {
                        return Some((
                            Err(ClientError::Stream(format!("stream read error: {e}"))),
                            (byte_stream, buffer),
                        ));
                    }
                    None => {
                        // Stream ended. Try to parse any remaining data.
                        if !buffer.trim().is_empty()
                            && let Some(result) = parse_sse_block(&buffer)
                        {
                            buffer.clear();
                            return Some((result, (byte_stream, buffer)));
                        }
                        return None;
                    }
                }
            }
        },
    )
}

/// Parse a single SSE block (text between `\n\n` boundaries).
///
/// Returns `Some(Ok(result))` for `event: translation`,
/// `Some(Err(_))` for `event: error`,
/// `None` for `event: done` or unrecognized events.
fn parse_sse_block(block: &str) -> Option<Result<TranslationResult, ClientError>> {
    let mut event_type = None;
    let mut data = None;

    for line in block.lines() {
        if let Some(rest) = line.strip_prefix("event:") {
            event_type = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("data:") {
            data = Some(rest.trim().to_string());
        }
    }

    let event_type = event_type.as_deref()?;
    let data = data?;

    match event_type {
        "translation" => {
            let result = serde_json::from_str::<TranslationResult>(&data)
                .map_err(|e| ClientError::Parse(format!("failed to parse translation event: {e}")));
            Some(result)
        }
        "error" => Some(Err(ClientError::Stream(data))),
        "done" => None,
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use translator_core::Language;

    #[test]
    fn parse_translation_event() {
        let block = "event: translation\ndata: {\"source_text\":\"hi\",\"detected_language\":\"en\",\"translations\":{\"fr\":\"salut\"}}";
        let result = parse_sse_block(block);
        assert!(result.is_some());
        let result = result.unwrap().unwrap();
        assert_eq!(result.source_text, "hi");
        assert_eq!(result.translations.get(&Language::Fr).unwrap(), "salut");
    }

    #[test]
    fn parse_error_event() {
        let block = "event: error\ndata: something went wrong";
        let result = parse_sse_block(block);
        assert!(result.is_some());
        assert!(result.unwrap().is_err());
    }

    #[test]
    fn parse_done_event() {
        let block = "event: done\ndata: [DONE]";
        let result = parse_sse_block(block);
        assert!(result.is_none());
    }

    #[test]
    fn parse_empty_block() {
        let result = parse_sse_block("");
        assert!(result.is_none());
    }
}

use unicode_segmentation::UnicodeSegmentation;

/// A chunk of text with metadata for reassembly.
pub struct TextChunk {
    pub text: String,
    /// Separator to insert before this chunk's translation when reassembling.
    /// First chunk always has "". Paragraph boundaries use "\n\n".
    /// Sentence boundaries use "" (sentences carry trailing whitespace).
    pub join_separator: &'static str,
}

/// Split text at paragraph boundaries (`\n\n`) first, falling back to sentence
/// boundaries for oversized paragraphs. Keeps related sentences together for
/// better translation context.
///
/// - `paragraph_target` — flush paragraph accumulator when adding next paragraph
///   would exceed this (quality limit)
/// - `max_chars` — hard ceiling for sentence fallback (capacity limit); also used
///   for the initial "skip chunking" check
pub fn chunk_text(text: &str, paragraph_target: usize, max_chars: usize) -> Vec<TextChunk> {
    if text.len() <= max_chars {
        return vec![TextChunk { text: text.to_string(), join_separator: "" }];
    }

    let mut chunks: Vec<TextChunk> = Vec::new();
    let mut current = String::new();
    // Tracks the separator to use when `current` is eventually flushed as a chunk.
    let mut current_sep: &'static str = "";

    for para in text.split("\n\n") {
        if !current.is_empty() && current.len() + "\n\n".len() + para.len() > paragraph_target {
            chunks.push(TextChunk { text: std::mem::take(&mut current), join_separator: current_sep });
            current_sep = "\n\n";
        }

        if current.is_empty() && para.len() > max_chars {
            // Paragraph too large — fall back to sentence splitting
            let para_sep = if chunks.is_empty() { "" } else { "\n\n" };
            let chunks_before = chunks.len();
            chunk_by_sentences(para, max_chars, &mut chunks, &mut current, para_sep);
            if !current.is_empty() {
                // Sentence chunks were flushed → leftover is a continuation ("")
                // No chunks flushed → whole paragraph in current, use para_sep
                current_sep = if chunks.len() > chunks_before { "" } else { para_sep };
            }
        } else {
            if current.is_empty() {
                current_sep = if chunks.is_empty() { "" } else { "\n\n" };
            }
            if !current.is_empty() {
                current.push_str("\n\n");
            }
            current.push_str(para);
        }
    }

    if !current.is_empty() {
        chunks.push(TextChunk { text: current, join_separator: current_sep });
    }
    if chunks.is_empty() {
        chunks.push(TextChunk { text: text.to_string(), join_separator: "" });
    }
    chunks
}

/// Pack sentences greedily into chunks up to `max_chars`. If no sentence
/// boundaries are found, the text is left in `current` as-is.
fn chunk_by_sentences(
    text: &str,
    max_chars: usize,
    chunks: &mut Vec<TextChunk>,
    current: &mut String,
    para_separator: &'static str,
) {
    let mut first_flush = true;
    for sentence in text.unicode_sentences() {
        if !current.is_empty() && current.len() + sentence.len() > max_chars {
            let sep = if first_flush { para_separator } else { "" };
            first_flush = false;
            chunks.push(TextChunk { text: std::mem::take(current), join_separator: sep });
        }
        current.push_str(sentence);
    }
}

#[cfg(test)]
mod tests {
    use super::{chunk_text, TextChunk};

    /// Reassemble chunks using their stored separators (mirrors reassembly logic).
    fn reassemble(chunks: &[TextChunk]) -> String {
        let mut out = String::new();
        for chunk in chunks {
            out.push_str(chunk.join_separator);
            out.push_str(&chunk.text);
        }
        out
    }

    /// Extract just the text strings for simpler assertions.
    fn texts(chunks: &[TextChunk]) -> Vec<&str> {
        chunks.iter().map(|c| c.text.as_str()).collect()
    }

    #[test]
    fn short_text_no_chunking() {
        let result = chunk_text("Hello world.", 100, 100);
        assert_eq!(texts(&result), vec!["Hello world."]);
        assert_eq!(result[0].join_separator, "");
    }

    #[test]
    fn exact_boundary() {
        let text = "Hello world.";
        let result = chunk_text(text, text.len(), text.len());
        assert_eq!(texts(&result), vec!["Hello world."]);
    }

    #[test]
    fn empty_string() {
        let result = chunk_text("", 100, 100);
        assert_eq!(texts(&result), vec![""]);
    }

    // --- Paragraph splitting ---

    #[test]
    fn two_paragraphs_split_at_boundary() {
        let text = "First paragraph.\n\nSecond paragraph.";
        let result = chunk_text(text, 20, 20);
        assert_eq!(texts(&result), vec!["First paragraph.", "Second paragraph."]);
        assert_eq!(result[0].join_separator, "");
        assert_eq!(result[1].join_separator, "\n\n");
        assert_eq!(reassemble(&result), text);
    }

    #[test]
    fn small_paragraphs_greedy_packed() {
        let text = "A.\n\nB.\n\nC.";
        let result = chunk_text(text, 100, 100);
        assert_eq!(texts(&result), vec!["A.\n\nB.\n\nC."]);
    }

    #[test]
    fn paragraph_packing_respects_limit() {
        let text = "AA\n\nBB\n\nCC";
        let result = chunk_text(text, 8, 8);
        assert_eq!(texts(&result), vec!["AA\n\nBB", "CC"]);
        assert_eq!(result[0].join_separator, "");
        assert_eq!(result[1].join_separator, "\n\n");
        assert_eq!(reassemble(&result), text);
    }

    // --- Sentence fallback for oversized paragraphs ---

    #[test]
    fn large_paragraph_falls_back_to_sentences() {
        let text = "First sentence. Second sentence. Third sentence.";
        let result = chunk_text(text, 20, 20);
        assert!(result.len() >= 2, "expected sentence split, got {} chunks", result.len());
        // Sentence chunks use "" separators (sentences carry trailing whitespace)
        for chunk in &result {
            assert_eq!(chunk.join_separator, "");
        }
        assert_eq!(reassemble(&result), text);
    }

    #[test]
    fn mixed_paragraphs_and_sentence_fallback() {
        let para1 = "Short.";
        let para2 = "First sentence. Second sentence. Third sentence.";
        let para3 = "End.";
        let text = format!("{}\n\n{}\n\n{}", para1, para2, para3);
        let result = chunk_text(&text, 25, 25);
        assert!(result.len() >= 3, "expected at least 3 chunks, got {}: {:?}", result.len(), texts(&result));
        // First chunk separator is always ""
        assert_eq!(result[0].join_separator, "");
        // Content is preserved via reassembly
        assert_eq!(reassemble(&result), text);
    }

    // --- Sentence-level tests (still valid for fallback path) ---

    #[test]
    fn two_sentences_split() {
        let text = "First sentence. Second sentence.";
        let result = chunk_text(text, 16, 16);
        assert_eq!(result.len(), 2);
        assert!(result[0].text.contains("First"));
        assert!(result[1].text.contains("Second"));
    }

    #[test]
    fn multiple_sentences_greedy_packing() {
        let text = "One. Two. Three. Four. Five. Six.";
        let result = chunk_text(text, 12, 12);
        assert!(result.len() < 6, "expected greedy packing, got {} chunks", result.len());
        assert_eq!(reassemble(&result), text);
    }

    #[test]
    fn unicode_sentences_japanese() {
        let text = "これは文です。もう一つの文です。";
        let result = chunk_text(text, 24, 24);
        assert!(result.len() >= 2, "expected split at Japanese sentence boundary, got {} chunks", result.len());
        assert_eq!(reassemble(&result), text);
    }

    #[test]
    fn labels_without_sentence_boundaries() {
        let r = chunk_text("Title:", 10, 10);
        assert_eq!(texts(&r), vec!["Title:"]);

        let r = chunk_text("First Name:", 20, 20);
        assert_eq!(texts(&r), vec!["First Name:"]);

        let r = chunk_text("Last Name:", 20, 20);
        assert_eq!(texts(&r), vec!["Last Name:"]);
    }

    #[test]
    fn no_sentence_boundaries() {
        let text = "word ".repeat(50);
        let text = text.trim();
        let result = chunk_text(text, 20, 20);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, text);
    }

    #[test]
    fn single_long_sentence_no_boundary() {
        let text = "This is one very long sentence without any terminating punctuation that goes on and on";
        let result = chunk_text(text, 20, 20);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text, text);
    }

    // --- Long document integration test ---

    #[test]
    fn long_document_paragraph_splitting() {
        let para1 = "The quick brown fox jumps over the lazy dog. It was a sunny day.";
        let para2 = "Meanwhile, in another part of the forest, birds were singing. The wind rustled through the leaves.";
        let para3 = "Short ending.";
        let text = format!("{}\n\n{}\n\n{}", para1, para2, para3);

        let max_chars = 100;
        let result = chunk_text(&text, max_chars, max_chars);

        assert!(result.len() >= 2, "expected paragraph-level splits, got {}: {:?}", result.len(), texts(&result));

        for chunk in &result {
            assert!(
                chunk.text.len() <= max_chars || !chunk.text.contains(". "),
                "chunk exceeds max_chars and has sentence boundaries: {:?}",
                chunk.text
            );
        }

        // Reassembled content preserves original text using separators
        assert_eq!(reassemble(&result), text);
    }

    // --- Separator correctness ---

    #[test]
    fn paragraph_chunks_have_newline_separators() {
        let text = "Para one.\n\nPara two.\n\nPara three.";
        let result = chunk_text(text, 15, 15);
        assert_eq!(result[0].join_separator, "");
        for chunk in &result[1..] {
            assert_eq!(chunk.join_separator, "\n\n");
        }
        assert_eq!(reassemble(&result), text);
    }

    #[test]
    fn sentence_chunks_have_empty_separators() {
        let text = "First sentence. Second sentence. Third sentence.";
        let result = chunk_text(text, 20, 20);
        assert!(result.len() >= 2);
        for chunk in &result {
            assert_eq!(chunk.join_separator, "");
        }
        assert_eq!(reassemble(&result), text);
    }

    #[test]
    fn mixed_paragraph_and_sentence_separators() {
        // para1 fits, para2 needs sentence splitting, para3 fits
        let para1 = "Short.";
        let para2 = "Sentence A. Sentence B. Sentence C.";
        let para3 = "End.";
        let text = format!("{}\n\n{}\n\n{}", para1, para2, para3);
        let result = chunk_text(&text, 20, 20);

        // First chunk always ""
        assert_eq!(result[0].join_separator, "");
        // Verify reassembly preserves original
        assert_eq!(reassemble(&result), text);
    }

    // --- Two-tier behavior tests ---

    #[test]
    fn paragraph_target_smaller_than_max_chars() {
        // paragraph_target=30 flushes paragraphs early, max_chars=60 for sentence fallback
        let text = "Para one is short.\n\nPara two is also short.\n\nPara three here.";
        // Total len > max_chars=60 so chunking is triggered.
        // Each paragraph < 30 but packing two would exceed paragraph_target=30.
        let result = chunk_text(text, 30, 60);
        // Should split at paragraph boundaries due to paragraph_target, not pack everything
        assert!(result.len() >= 2, "expected paragraph splits at target, got {}: {:?}", result.len(), texts(&result));
        assert_eq!(reassemble(&result), text);
    }

    #[test]
    fn sentence_fallback_uses_max_chars_not_target() {
        // A single paragraph with multiple sentences, longer than paragraph_target but
        // sentences fit within max_chars. Sentence fallback should use max_chars.
        let text = "Sentence one is here. Sentence two is here. Sentence three is here too.";
        // paragraph_target=20 (smaller), max_chars=50 (sentence fallback ceiling)
        // The single paragraph exceeds max_chars so sentence fallback kicks in.
        let result = chunk_text(text, 20, 50);
        // Sentences should be packed up to max_chars=50, not paragraph_target=20
        for chunk in &result {
            assert!(
                chunk.text.len() <= 50 || !chunk.text.contains(". "),
                "chunk exceeds max_chars: {:?} (len={})",
                chunk.text, chunk.text.len()
            );
        }
        assert_eq!(reassemble(&result), text);
    }
}

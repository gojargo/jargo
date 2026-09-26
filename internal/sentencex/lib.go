// Package sentencex splits text into sentences, with rules for many
// languages. It is a Go port of sentencex v1.0.31
// (https://github.com/wikimedia/sentencex), the sentence segmentation library
// by Santhosh Thottingal, released under the MIT License. The algorithm, the
// per-language rules and the bundled word lists are carried over unchanged,
// and the upstream test corpus runs against this package.
//
// Segment returns the sentences as substrings of the input, in order, with
// the whitespace that follows a sentence kept at its end and each paragraph
// separator returned as a span of its own. Offsets are UTF-8 byte offsets and
// the input is expected to be valid UTF-8.
package sentencex

// chunkSize is the text length, in bytes, above which Segment splits the text
// at paragraph boundaries before segmenting it.
const chunkSize = 10 * 1024

// languageFactory returns the rules for languageCode, following the fallback
// chains for a code without rules of its own and settling on English when a
// chain ends or loops.
func languageFactory(languageCode string) *language {
	current := languageCode
	visited := make(map[string]struct{})

	for {
		if _, seen := visited[current]; seen {
			current = "en" // Default to English when a cycle is detected.
		} else {
			visited[current] = struct{}{}
		}

		if build, ok := languages[current]; ok {
			return build()
		}

		fallbacks, ok := getFallbacks(current)
		if !ok {
			current = "en" // Default to English when there are no fallbacks.
			continue
		}
		for _, next := range fallbacks {
			if _, seen := visited[next]; !seen {
				current = next
				break
			}
		}
	}
}

// paragraphSpans returns the [start, end) byte ranges of the paragraphs of
// text, the text between paragraph separators.
func paragraphSpans(text string) [][2]int {
	var spans [][2]int
	nextStart := 0

	for _, sep := range paragraphBreaks(text) {
		spans = append(spans, [2]int{nextStart, sep[0]})
		nextStart = sep[1]
	}

	if nextStart < len(text) {
		spans = append(spans, [2]int{nextStart, len(text)})
	}

	return spans
}

// textChunk is a chunk of a text and its byte offset in the text.
type textChunk struct {
	offset int
	text   string
}

// chunkText splits text into chunks of at most size bytes at paragraph
// boundaries. A paragraph longer than size stays whole.
func chunkText(text string, size int) []textChunk {
	if size == 0 || len(text) <= size {
		return []textChunk{{offset: 0, text: text}}
	}

	chunks := make([]textChunk, 0, len(text)/size+1)
	chunkStart, chunkEnd, open := 0, 0, false

	for _, span := range paragraphSpans(text) {
		if open && span[1]-chunkStart <= size {
			chunkEnd = span[1]
			continue
		}
		if open {
			chunks = append(chunks, textChunk{offset: chunkStart, text: text[chunkStart:chunkEnd]})
		}
		chunkStart, chunkEnd, open = span[0], span[1], true
	}

	if open {
		chunks = append(chunks, textChunk{offset: chunkStart, text: text[chunkStart:chunkEnd]})
	}

	return chunks
}

// Segment splits text into sentences using the rules of the language named by
// languageCode (an ISO 639 code such as "en" or "fr"). A code without rules of
// its own falls back through related languages, and finally to English.
//
// The spans are substrings of text, in order: each sentence keeps the
// whitespace that follows it, and each paragraph separator (a blank line) is
// a span of its own.
//
// A text longer than 10 KiB is split at paragraph boundaries into chunks that
// are segmented one by one, and the paragraph separators between two chunks
// are not returned. A paragraph longer than that with no paragraph break
// inside is processed whole, so no sentence or word is ever split across
// chunks.
func Segment(languageCode, text string) []string {
	lang := languageFactory(languageCode)

	if len(text) <= chunkSize {
		return lang.segment(text)
	}

	var sentences []string
	for _, chunk := range chunkText(text, chunkSize) {
		sentences = append(sentences, lang.segment(chunk.text)...)
	}
	return sentences
}

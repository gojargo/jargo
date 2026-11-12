package text

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// TaggedChunkKind is the kind of a TaggedChunk.
type TaggedChunkKind int

// The kinds of chunk a TaggedLLMTextParser produces.
const (
	// TaggedText is text outside the thought.
	TaggedText TaggedChunkKind = iota
	// TaggedThoughtStart is the thought beginning. Its text is empty.
	TaggedThoughtStart
	// TaggedThoughtText is text inside the thought.
	TaggedThoughtText
	// TaggedThoughtEnd is the thought ending. Its text is empty.
	TaggedThoughtEnd
)

// TaggedChunk is a categorized piece of text a TaggedLLMTextParser produced.
type TaggedChunk struct {
	Kind TaggedChunkKind
	// Text is the piece's text, empty for a thought's start and end.
	Text string
}

// taggedState is where a TaggedLLMTextParser is in the text.
type taggedState int

const (
	taggedDetecting taggedState = iota
	taggedInThought
	taggedAfterThought
	taggedContent
)

// TaggedLLMTextParser parses a tagged thought, such as <think>...</think>, out
// of streamed LLM text. Some models write their reasoning into the answer text,
// wrapped in tags, ahead of the answer itself; the parser separates that
// reasoning from the text that follows, so a service can emit it as thoughts
// rather than as text to speak.
//
// Feed the text chunk by chunk to ParseNextChunk, which returns the pieces each
// chunk completes: answer text, or a thought's start, text and end.
//
// A thought is recognized only at the start of the text, optionally after
// whitespace; tags later in the text pass on as text. Tags may be split across
// chunks, so the start of the text is held back until it is clear whether it
// opens a thought, and inside a thought a possible start of the closing tag is
// held back. Text that does not open with a thought passes through from the
// first chunk that rules one out, and whitespace between a thought and the text
// after it is dropped.
//
// Call Flush when the text ends, or Reset when it is cut short. Either readies
// the parser for the next text.
type TaggedLLMTextParser struct {
	openTag, closeTag string
	state             taggedState
	buffer            string
}

// NewTaggedLLMTextParser builds a parser for the thought between openTag and
// closeTag, such as "<think>" and "</think>".
func NewTaggedLLMTextParser(openTag, closeTag string) *TaggedLLMTextParser {
	return &TaggedLLMTextParser{openTag: openTag, closeTag: closeTag}
}

// ParseNextChunk parses the next chunk of streamed text, returning the pieces
// it completes, in order.
func (p *TaggedLLMTextParser) ParseNextChunk(text string) []TaggedChunk {
	if p.state == taggedContent {
		return []TaggedChunk{{Kind: TaggedText, Text: text}}
	}
	var out []TaggedChunk

	if p.state == taggedDetecting {
		p.buffer += text
		head := strings.TrimLeftFunc(p.buffer, unicode.IsSpace)
		if len(head) < len(p.openTag) && strings.HasPrefix(p.openTag, head) {
			return out
		}
		if !strings.HasPrefix(head, p.openTag) {
			passthrough := p.buffer
			p.buffer = ""
			p.state = taggedContent
			return []TaggedChunk{{Kind: TaggedText, Text: passthrough}}
		}
		out = append(out, TaggedChunk{Kind: TaggedThoughtStart})
		p.buffer = ""
		p.state = taggedInThought
		text = head[len(p.openTag):]
	}

	if p.state == taggedInThought {
		p.buffer += text
		idx := strings.Index(p.buffer, p.closeTag)
		if idx < 0 {
			// What could be the start of the closing tag is held back: the
			// tag's length less one character.
			hold := utf8.RuneCountInString(p.closeTag) - 1
			runes := []rune(p.buffer)
			if safeEnd := len(runes) - hold; safeEnd > 0 {
				out = append(out, TaggedChunk{Kind: TaggedThoughtText, Text: string(runes[:safeEnd])})
				p.buffer = string(runes[safeEnd:])
			}
			return out
		}
		if thought := p.buffer[:idx]; thought != "" {
			out = append(out, TaggedChunk{Kind: TaggedThoughtText, Text: thought})
		}
		out = append(out, TaggedChunk{Kind: TaggedThoughtEnd})
		text = p.buffer[idx+len(p.closeTag):]
		p.buffer = ""
		p.state = taggedAfterThought
	}

	// After the thought, whitespace is dropped until the text resumes.
	if text = strings.TrimLeftFunc(text, unicode.IsSpace); text != "" {
		p.state = taggedContent
		out = append(out, TaggedChunk{Kind: TaggedText, Text: text})
	}
	return out
}

// Flush ends the text, releasing what was held back and closing a thought left
// open.
func (p *TaggedLLMTextParser) Flush() []TaggedChunk { return p.end(true) }

// Reset ends text that was cut short, by an interruption or an error, say.
// What was held back is discarded; a thought left open is closed, so the caller
// can end it.
func (p *TaggedLLMTextParser) Reset() []TaggedChunk { return p.end(false) }

func (p *TaggedLLMTextParser) end(release bool) []TaggedChunk {
	var out []TaggedChunk
	switch {
	case p.state == taggedInThought:
		if p.buffer != "" && release {
			out = append(out, TaggedChunk{Kind: TaggedThoughtText, Text: p.buffer})
		}
		out = append(out, TaggedChunk{Kind: TaggedThoughtEnd})
	case p.state == taggedDetecting && p.buffer != "" && release:
		out = append(out, TaggedChunk{Kind: TaggedText, Text: p.buffer})
	}
	p.buffer = ""
	p.state = taggedDetecting
	return out
}

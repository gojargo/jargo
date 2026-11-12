package text_test

import (
	"slices"
	"testing"

	"github.com/gojargo/jargo/utils/text"
)

const (
	tStart   = text.TaggedThoughtStart
	tThought = text.TaggedThoughtText
	tEnd     = text.TaggedThoughtEnd
	tText    = text.TaggedText
)

// parseTagged feeds texts through a parser, merging consecutive text of the
// same kind.
func parseTagged(texts []string, flush bool) []text.TaggedChunk {
	p := text.NewTaggedLLMTextParser("<think>", "</think>")
	var chunks []text.TaggedChunk
	for _, s := range texts {
		chunks = append(chunks, p.ParseNextChunk(s)...)
	}
	if flush {
		chunks = append(chunks, p.Flush()...)
	} else {
		chunks = append(chunks, p.Reset()...)
	}
	var merged []text.TaggedChunk
	for _, c := range chunks {
		if n := len(merged); n > 0 && (c.Kind == tThought || c.Kind == tText) && merged[n-1].Kind == c.Kind {
			merged[n-1].Text += c.Text
			continue
		}
		merged = append(merged, c)
	}
	return merged
}

func wantChunks(t *testing.T, got, want []text.TaggedChunk) {
	t.Helper()
	if !slices.Equal(got, want) {
		t.Errorf("got %+v, want %+v", got, want)
	}
}

func thoughtThenHi() []text.TaggedChunk {
	return []text.TaggedChunk{{Kind: tStart}, {Kind: tThought, Text: "Hmm."}, {Kind: tEnd}, {Kind: tText, Text: "Hi"}}
}

func TestTaggedThoughtSplitAcrossChunks(t *testing.T) {
	wantChunks(t, parseTagged([]string{"<th", "ink", ">Let me", " check.</", "think", ">", "Hello", "!"}, true),
		[]text.TaggedChunk{
			{Kind: tStart}, {Kind: tThought, Text: "Let me check."}, {Kind: tEnd}, {Kind: tText, Text: "Hello!"},
		})
}

func TestTaggedEverySplitPointOfTheTags(t *testing.T) {
	s := "<think>Hmm.</think>Hi"
	for i := 1; i < len(s); i++ {
		wantChunks(t, parseTagged([]string{s[:i], s[i:]}, true), thoughtThenHi())
	}
}

func TestTaggedWhitespaceBeforeTheThoughtIsAllowed(t *testing.T) {
	wantChunks(t, parseTagged([]string{"\n", "<think>Hmm.</think>Hi"}, true), thoughtThenHi())
}

func TestTaggedWhitespaceAfterTheThoughtIsDroppedAcrossChunks(t *testing.T) {
	wantChunks(t, parseTagged([]string{"<think>Hmm.</think>", "\n", "\n ", "Hi there", "\n"}, true),
		[]text.TaggedChunk{{Kind: tStart}, {Kind: tThought, Text: "Hmm."}, {Kind: tEnd}, {Kind: tText, Text: "Hi there\n"}})
}

func TestTaggedTextWithoutAThoughtPassesThroughUnchanged(t *testing.T) {
	wantChunks(t, parseTagged([]string{" Hel", "lo <think>", "</think>"}, true),
		[]text.TaggedChunk{{Kind: tText, Text: " Hello <think></think>"}})
}

func TestTaggedTextThatRulesOutTheTagIsNotHeldBack(t *testing.T) {
	p := text.NewTaggedLLMTextParser("<think>", "</think>")
	if got := p.ParseNextChunk("<th"); len(got) != 0 {
		t.Fatalf("got %+v, want the possible tag held back", got)
	}
	wantChunks(t, p.ParseNextChunk("e"), []text.TaggedChunk{{Kind: tText, Text: "<the"}})
	wantChunks(t, p.ParseNextChunk("re"), []text.TaggedChunk{{Kind: tText, Text: "re"}})
}

func TestTaggedPartialTagIsFlushedAtTheEnd(t *testing.T) {
	wantChunks(t, parseTagged([]string{"<"}, true), []text.TaggedChunk{{Kind: tText, Text: "<"}})
}

func TestTaggedThoughtLeftOpenIsClosedAtTheEnd(t *testing.T) {
	wantChunks(t, parseTagged([]string{"<think>Hmm", "</thi"}, true),
		[]text.TaggedChunk{{Kind: tStart}, {Kind: tThought, Text: "Hmm</thi"}, {Kind: tEnd}})
}

func TestTaggedTextCutShortIsNotFlushed(t *testing.T) {
	// Inside a thought, the closing tag's length less one character is held back.
	wantChunks(t, parseTagged([]string{"<think>Let me check.", "</thi"}, false),
		[]text.TaggedChunk{{Kind: tStart}, {Kind: tThought, Text: "Let me chec"}, {Kind: tEnd}})
	if got := parseTagged([]string{"<thi"}, false); len(got) != 0 {
		t.Errorf("got %+v, want nothing", got)
	}
}

func TestTaggedFlushReadiesTheParserForTheNextText(t *testing.T) {
	p := text.NewTaggedLLMTextParser("<thinking>", "</thinking>")
	for _, s := range []string{"<thinking>One.</thinking>", "<thinking>Two.</thinking>"} {
		var kinds []text.TaggedChunkKind
		for _, c := range append(p.ParseNextChunk(s), p.Flush()...) {
			kinds = append(kinds, c.Kind)
		}
		if !slices.Equal(kinds, []text.TaggedChunkKind{tStart, tThought, tEnd}) {
			t.Errorf("%q: kinds %v, want start, thought, end", s, kinds)
		}
	}
	if got := p.Flush(); len(got) != 0 {
		t.Errorf("a second flush gave %+v, want nothing", got)
	}
}

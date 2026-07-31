package context

import (
	"strings"

	"github.com/gojargo/jargo/frames"
	ttstext "github.com/gojargo/jargo/utils/text"
)

// parallelAggregation is one completed sentence, in each of the three channels
// the sequencer tracks.
type parallelAggregation struct {
	// tts is the sentence as sent to the synthesizer, after any transform.
	tts string
	// llm is the sentence in the original model output, with any delimiters.
	llm string
	// userFacing is the sentence as shown, without synthesis tags or transforms.
	userFacing string
}

// parallelSentenceAggregator groups streamed tokens back into sentences while
// keeping three channels of the same text in step.
//
// It is used when the synthesizer is fed token by token but whole sentences are
// still needed, for word tracking and for reporting progress. Each token adds a
// part to all three channels, and completion is driven by the text sent to the
// synthesizer, which completes all three together.
//
// A boundary is only confirmed by lookahead, so a sentence is emitted once the
// first non-whitespace character of the next one has arrived. Tokens are not
// one word each: a chunk can carry the tail of one sentence and the head of the
// next. When every token of the pending sentence reads identically in all three
// channels, which is the case whenever no transform is rewriting the text, the
// boundary is cut inside the token at its confirmed offset. Once a transform has
// made the channels differ in length there is no shared offset to cut at, so the
// cut falls at the token boundary instead: what was buffered before the token is
// emitted, and the whole token starts the next sentence.
type parallelSentenceAggregator struct {
	inner     *ttstext.SimpleAggregator
	tokenizer SentenceTokenizer

	tts        string
	llm        string
	userFacing string
	// aligned records whether every token of the pending sentence read the same
	// in all three channels. While it holds, the tts buffer mirrors the inner
	// aggregator's and a boundary can be cut inside a token.
	aligned bool
}

// newParallelSentenceAggregator builds an aggregator over tokenizer.
func newParallelSentenceAggregator(tokenizer SentenceTokenizer) *parallelSentenceAggregator {
	return &parallelSentenceAggregator{
		inner:     ttstext.NewSimpleAggregator(frames.AggregationSentence, tokenizer),
		tokenizer: tokenizer,
		aligned:   true,
	}
}

// aggregate folds one token into all three channels and returns the sentences it
// completes. Usually none or one, but a coarse chunk can complete several.
func (p *parallelSentenceAggregator) aggregate(ttsText, llmText, userFacingText string) []parallelAggregation {
	identical := ttsText == llmText && llmText == userFacingText

	// The inner aggregator decides when and how many boundaries are confirmed,
	// applying the lookahead rule character by character on the synthesized text.
	boundaries := len(p.inner.Aggregate(ttsText))

	if boundaries > 0 && p.aligned && identical {
		return p.sliceInsideToken(ttsText, boundaries)
	}

	var out []parallelAggregation
	if boundaries > 0 && (p.tts != "" || p.llm != "" || p.userFacing != "") {
		out = append(out, parallelAggregation{tts: p.tts, llm: p.llm, userFacing: p.userFacing})
		p.tts, p.llm, p.userFacing = "", "", ""
	}

	// Plain concatenation: model tokens already carry their own spacing.
	p.tts += ttsText
	p.llm += llmText
	p.userFacing += userFacingText
	// Re-derive alignment rather than assume a token-boundary cut restored it.
	p.aligned = p.tts == p.llm && p.llm == p.userFacing && p.tts == p.inner.Buffer()
	return out
}

// sliceInsideToken cuts the confirmed boundaries inside the incoming token,
// which is possible only while the channels read identically.
func (p *parallelSentenceAggregator) sliceInsideToken(ttsText string, boundaries int) []parallelAggregation {
	combined := p.tts + ttsText
	var out []parallelAggregation
	idx := 0
	for range boundaries {
		boundary := p.tokenizer.MatchEndOfSentence(combined[idx:])
		if boundary <= 0 {
			break
		}
		sentence := combined[idx : idx+boundary]
		out = append(out, parallelAggregation{tts: sentence, llm: sentence, userFacing: sentence})
		idx += boundary
	}
	// The remainder came from identical channels, so the next pending sentence
	// starts aligned and its buffer still mirrors the inner one.
	rest := combined[idx:]
	p.tts, p.llm, p.userFacing = rest, rest, rest
	p.aligned = true
	return out
}

// flush emits the trailing partial sentence at the end of a turn.
func (p *parallelSentenceAggregator) flush() (parallelAggregation, bool) {
	p.inner.Flush()
	if strings.TrimSpace(p.userFacing) == "" {
		return parallelAggregation{}, false
	}
	out := parallelAggregation{tts: p.tts, llm: p.llm, userFacing: p.userFacing}
	p.reset()
	return out, true
}

// reset discards everything buffered.
func (p *parallelSentenceAggregator) reset() {
	p.tts, p.llm, p.userFacing = "", "", ""
	p.aligned = true
}

// trimSpace is strings.TrimSpace, named here so the sequencer reads the same as
// the rest of the package.
func trimSpace(s string) string { return strings.TrimSpace(s) }

// contains reports whether outer holds inner.
func contains(outer, inner string) bool { return strings.Contains(outer, inner) }

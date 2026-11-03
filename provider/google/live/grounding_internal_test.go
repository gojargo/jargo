package live

import (
	"context"
	"reflect"
	"testing"

	googleframes "github.com/gojargo/jargo/provider/google/frames"
)

// These tests are not ports of upstream tests: upstream has none for the
// grounding Gemini Live reports.

const groundingJSON = `{
	"searchEntryPoint": {"renderedContent": "<div>suggestions</div>"},
	"groundingChunks": [
		{"web": {"uri": "https://example.com/a", "title": "A"}},
		{"web": {"uri": "https://example.com/b", "title": "B"}}
	],
	"groundingSupports": [
		{"segment": {"text": "It is sunny."}, "groundingChunkIndices": [0, 1], "confidenceScores": [0.9, 0.7]}
	]
}`

// searchFrames are the search responses the service pushed.
func searchFrames(c *capture) []*googleframes.LLMSearchResponseFrame {
	var out []*googleframes.LLMSearchResponseFrame
	for _, f := range c.frames() {
		if sf, ok := f.(*googleframes.LLMSearchResponseFrame); ok {
			out = append(out, sf)
		}
	}
	return out
}

// Grounding that comes with the answer is reported once the turn completes,
// against the whole answer, with each page and the parts of the answer it
// supports.
func TestGroundingWithTheAnswerIsReportedAtTheEndOfTheTurn(t *testing.T) {
	s, sink := newTestService(t)
	ctx := context.Background()

	s.handle(ctx, decode(t,
		`{"serverContent":{"outputTranscription":{"text":"It is "},"groundingMetadata":`+groundingJSON+`}}`))
	s.handle(ctx, decode(t, `{"serverContent":{"outputTranscription":{"text":"sunny."}}}`))
	if got := searchFrames(sink); len(got) != 0 {
		t.Fatalf("search responses = %d before the turn completed, want none", len(got))
	}

	s.handle(ctx, decode(t, `{"serverContent":{"turnComplete":true}}`))
	got := searchFrames(sink)
	if len(got) != 1 {
		t.Fatalf("search responses = %d, want one", len(got))
	}
	f := got[0]
	if f.SearchResult != "It is sunny." || f.RenderedContent != "<div>suggestions</div>" {
		t.Errorf("frame = %+v, want the answer and the suggestions", f)
	}
	want := []googleframes.LLMSearchOrigin{
		{SiteURI: "https://example.com/a", SiteTitle: "A", Results: []googleframes.LLMSearchResult{
			{Text: "It is sunny.", Confidence: []float64{0.9, 0.7}},
		}},
		{SiteURI: "https://example.com/b", SiteTitle: "B", Results: []googleframes.LLMSearchResult{
			{Text: "It is sunny.", Confidence: []float64{0.9, 0.7}},
		}},
	}
	if !reflect.DeepEqual(f.Origins, want) {
		t.Errorf("origins = %+v, want %+v", f.Origins, want)
	}

	// The next turn starts afresh.
	s.handle(ctx, decode(t, `{"serverContent":{"turnComplete":true}}`))
	if n := len(searchFrames(sink)); n != 1 {
		t.Errorf("search responses = %d, want no second one", n)
	}
}

// Grounding on its own is reported at once.
func TestGroundingOnItsOwnIsReportedAtOnce(t *testing.T) {
	s, sink := newTestService(t)
	s.handle(context.Background(), decode(t, `{"serverContent":{"groundingMetadata":`+groundingJSON+`}}`))
	if n := len(searchFrames(sink)); n != 1 {
		t.Errorf("search responses = %d, want one", n)
	}
}

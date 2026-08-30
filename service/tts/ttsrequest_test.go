package tts_test

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/utils/events"
	ttstext "github.com/gojargo/jargo/utils/text"
)

// requestRecorder collects what each announced request carried.
type requestRecorder struct {
	mu  sync.Mutex
	got []tts.TTSRequest
}

func (r *requestRecorder) attach(b *tts.Base) {
	events.On(b.Events(), tts.EventTTSRequest, func(_ context.Context, req tts.TTSRequest) {
		r.mu.Lock()
		r.got = append(r.got, req)
		r.mu.Unlock()
	})
}

func (r *requestRecorder) requests() []tts.TTSRequest {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]tts.TTSRequest(nil), r.got...)
}

// TestTTSRequestAnnouncesEachUnit checks every unit handed to the provider is
// announced, in order and grouped under the synthesis context it belongs to.
func TestTTSRequestAnnouncesEachUnit(t *testing.T) {
	h := newTransformHarness(t, nil)
	var rec requestRecorder
	rec.attach(h.base)

	h.runTurn(t, "First sentence. ", "Second sentence.")

	got := rec.requests()
	if len(got) != 2 {
		t.Fatalf("announced %d requests, want one per unit: %+v", len(got), got)
	}
	if !strings.Contains(got[0].Text, "First sentence") ||
		!strings.Contains(got[1].Text, "Second sentence") {
		t.Errorf("announced %q and %q, want the two sentences in order", got[0].Text, got[1].Text)
	}
	if got[0].ContextID == "" {
		t.Error("announced request carries no context id")
	}
	if got[0].ContextID != got[1].ContextID {
		t.Errorf("context ids %q and %q differ, want one turn to be one context",
			got[0].ContextID, got[1].ContextID)
	}
}

// TestTTSRequestCarriesWhatTheProviderIsGiven checks the announced text is the
// text as sent, transforms included. Announcing the text before it was reshaped
// would describe something the synthesizer never received, which is the whole
// reason for looking at this point rather than at the model's output.
func TestTTSRequestCarriesWhatTheProviderIsGiven(t *testing.T) {
	h := newTransformHarness(t, nil)
	h.base.SetTextTransformers(tts.TextTransformer{
		AggregatedBy: frames.AnyAggregation,
		Transform: func(_ context.Context, text string, _ frames.AggregationType) (string, error) {
			return strings.ReplaceAll(text, "@", " at "), nil
		},
	})
	var rec requestRecorder
	rec.attach(h.base)

	h.runTurn(t, "Mail me at me@example.com.")

	got := rec.requests()
	if len(got) != 1 {
		t.Fatalf("announced %d requests, want 1: %+v", len(got), got)
	}
	if !strings.Contains(got[0].Text, "me at example.com") {
		t.Errorf("announced %q, want the transformed text the provider was given", got[0].Text)
	}
	if spoken := h.syn.texts(); len(spoken) != 1 || spoken[0] != got[0].Text {
		t.Errorf("announced %q but spoke %q, want them identical", got[0].Text, spoken)
	}
}

// TestTTSRequestIsNotAnnouncedForUnspokenText checks a unit that is passed
// downstream unsynthesized draws no announcement: nothing was requested of the
// provider, so there is no request to report.
func TestTTSRequestIsNotAnnouncedForUnspokenText(t *testing.T) {
	const codeType frames.AggregationType = "code"
	agg := ttstext.NewPatternPairAggregator(frames.AggregationSentence, newTokenizer(t))
	if err := agg.AddPattern(codeType, "<code>", "</code>", ttstext.MatchAggregate); err != nil {
		t.Fatalf("add pattern: %v", err)
	}

	h := newTransformHarness(t, agg)
	h.base.SetSkipAggregatorTypes(codeType)
	var rec requestRecorder
	rec.attach(h.base)

	h.runTurn(t, "Here it is. <code>print(1)</code> Done.")

	for _, req := range rec.requests() {
		if strings.Contains(req.Text, "print(1)") {
			t.Errorf("announced %q, want the skipped unit never requested", req.Text)
		}
	}
}

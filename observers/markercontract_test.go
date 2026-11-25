package observers_test

import (
	"context"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
)

// markersPushedFor runs one response through the turn-completion extension and
// returns the markers it pushed.
func markersPushedFor(t *testing.T, text string) []*frames.LLMMarkerFrame {
	t.Helper()
	extension := llm.NewTurnCompletionExtension(llm.UserTurnCompletionConfig{})
	pushed, err := extension.FilterPushFrame(context.Background(), frames.NewLLMTextFrame(text), processor.Downstream)
	if err != nil {
		t.Fatal(err)
	}
	var markers []*frames.LLMMarkerFrame
	for _, f := range pushed {
		if m, ok := f.(*frames.LLMMarkerFrame); ok {
			markers = append(markers, m)
		}
	}
	return markers
}

// The latency observer reads an incomplete verdict as a marker standing alone
// and a complete one as a marker prefixing the reply, which is the way round
// the turn-completion extension pushes them.
func TestTurnCompletionPushesMarkersThatWayRound(t *testing.T) {
	incomplete := markersPushedFor(t, llm.MarkerIncompleteShort)
	if len(incomplete) != 1 || incomplete[0].Marker != llm.MarkerIncompleteShort ||
		!incomplete[0].AppendToContextImmediately {
		t.Errorf("an incomplete verdict pushed %+v, want the short marker standing alone", incomplete)
	}

	complete := markersPushedFor(t, llm.MarkerComplete+" Hello!")
	if len(complete) != 1 || complete[0].Marker != llm.MarkerComplete || complete[0].AppendToContextImmediately {
		t.Errorf("a complete verdict pushed %+v, want the complete marker prefixing the reply", complete)
	}
}

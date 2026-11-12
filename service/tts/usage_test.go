package tts_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/tts"
	ttstext "github.com/gojargo/jargo/utils/text"
)

// usageSynth reports each sentence it is sent as usage, as a provider does once
// the text is sent, then speaks it as toneSynth does.
type usageSynth struct{ toneSynth }

func (s *usageSynth) RunTTS(ctx context.Context, text, contextID string, yield func(f frames.Frame) error) error {
	tts.StartUsageMetrics(ctx, text)
	return s.toneSynth.RunTTS(ctx, text, contextID, yield)
}

// usageOf speaks a reply through svc with the given metrics switches, and
// returns the character counts it reported.
func usageOf(t *testing.T, svc *tts.Base, metrics, usage bool, tokens ...string) []int {
	t.Helper()
	w, r, stop := runRecorded(t, svc, metrics, usage)
	w.QueueFrame(frames.NewLLMFullResponseStartFrame())
	for _, tok := range tokens {
		w.QueueFrame(frames.NewLLMTextFrame(tok))
	}
	w.QueueFrame(frames.NewLLMFullResponseEndFrame())
	waitFor(t, "the stop frame", func() bool { _, ok := firstArrived[*frames.TTSStoppedFrame](r); return ok })
	time.Sleep(100 * time.Millisecond)
	stop()
	got, _ := metricsArrived[frames.TTSUsageMetricsData](r)
	counts := make([]int, 0, len(got))
	for _, u := range got {
		counts = append(counts, u.Value)
	}
	return counts
}

// TestTTSUsageFollowsTheUsageMetricsSwitch checks TTS usage is reported when
// usage metrics are on, whatever the metrics switch says, and not otherwise.
func TestTTSUsageFollowsTheUsageMetricsSwitch(t *testing.T) {
	for _, tc := range []struct {
		name           string
		metrics, usage bool
		want           bool
	}{
		{"usage on, metrics off", false, true, true},
		{"usage off, metrics on", true, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := usageOf(t, tts.New("FakeTTS", &usageSynth{}), tc.metrics, tc.usage, "Hello there.")
			if (len(got) > 0) != tc.want {
				t.Errorf("usage reported = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestTTSUsageIsReportedPerSentence checks each sentence sent to the provider
// reports its own characters, rather than the reply reporting them once.
func TestTTSUsageIsReportedPerSentence(t *testing.T) {
	got := usageOf(t, tts.New("FakeTTS", &usageSynth{}), false, true, "One. ", "Three.")
	if len(got) != 2 || got[0] != 4 || got[1] != 6 {
		t.Errorf("usage = %v, want [4 6]: one report per sentence sent", got)
	}
}

// TestTTSUsageIsNotReportedForAProviderThatDoesNot checks the base reports no
// usage of its own: a provider that does not report what it sends, reports
// nothing.
func TestTTSUsageIsNotReportedForAProviderThatDoesNot(t *testing.T) {
	if got := usageOf(t, tts.New("FakeTTS", &toneSynth{}), false, true, "Hello there."); len(got) != 0 {
		t.Errorf("usage = %v, want none", got)
	}
}

// TestTTSUsageStreamingTokensIsReportedOncePerReply checks that, streaming
// tokens, the provider's per-call reports are skipped, and the reply's whole
// text is reported once it ends.
func TestTTSUsageStreamingTokensIsReportedOncePerReply(t *testing.T) {
	svc := tts.New("FakeTTS", &usageSynth{})
	svc.SetTextAggregator(ttstext.NewTokenAggregator())
	got := usageOf(t, svc, false, true, "Hello", " there", ".")
	if len(got) != 1 || got[0] != len("Hello there.") {
		t.Errorf("usage = %v, want [%d]: the whole reply, once", got, len("Hello there."))
	}
}

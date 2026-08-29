package tts_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/tts"
	errs "github.com/gojargo/jargo/utils/errors"
	"github.com/gojargo/jargo/utils/events"
	ttstext "github.com/gojargo/jargo/utils/text"
)

// errTransform is what a failing transform reports.
//
//nolint:gochecknoglobals // sentinel error for the tests below
var errTransform = errors.New("the translation service refused the request")

// transformHarness runs a turn through a base with transforms set and records
// what the provider was asked to speak, what went upstream, and what the
// conversation was told was spoken.
type transformHarness struct {
	syn  *spacedSynth
	base *tts.Base
	task *pipeline.Worker

	mu      sync.Mutex
	errs    []*frames.ErrorFrame
	spokenT []string
}

func newTransformHarness(t *testing.T, agg ttstext.Aggregator) *transformHarness {
	t.Helper()
	h := &transformHarness{syn: &spacedSynth{}}
	h.base = tts.New("TransformTTS", h.syn)
	if agg != nil {
		h.base.SetTextAggregator(agg)
	}
	h.task = pipeline.NewWorker(pipeline.New(h.base), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
	})
	events.On(&h.task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		if ef, ok := f.(*frames.ErrorFrame); ok {
			h.mu.Lock()
			h.errs = append(h.errs, ef)
			h.mu.Unlock()
		}
	})
	events.On(&h.task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if tf, ok := f.(*frames.TTSTextFrame); ok {
			h.mu.Lock()
			h.spokenT = append(h.spokenT, tf.Text)
			h.mu.Unlock()
		}
	})
	return h
}

// runTurn feeds one model turn and waits for the pipeline to drain.
func (h *transformHarness) runTurn(t *testing.T, texts ...string) {
	t.Helper()
	runDone := make(chan error, 1)
	go func() { runDone <- h.task.Run(context.Background()) }()

	h.task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	for _, text := range texts {
		h.task.QueueFrame(frames.NewLLMTextFrame(text))
	}
	h.task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	h.task.StopWhenDone()

	select {
	case <-runDone:
	case <-time.After(5 * time.Second):
		t.Fatal("task did not finish")
	}
}

func (h *transformHarness) errorFrames() []*frames.ErrorFrame {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]*frames.ErrorFrame(nil), h.errs...)
}

func (h *transformHarness) recorded() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]string(nil), h.spokenT...)
}

// A transform reshapes what the provider is given, so a tag the provider
// understands can be added without the model having written it.
func TestTextTransformReshapesWhatTheProviderIsGiven(t *testing.T) {
	h := newTransformHarness(t, nil)
	h.base.SetTextTransformers(tts.TextTransformer{
		AggregatedBy: frames.AnyAggregation,
		Transform: func(_ context.Context, text string, _ frames.AggregationType) (string, error) {
			return strings.ReplaceAll(text, "@", " at "), nil
		},
	})

	h.runTurn(t, "Mail me at me@example.com.")

	got := h.syn.texts()
	if len(got) != 1 || !strings.Contains(got[0], "me at example.com") {
		t.Errorf("spoken = %q, want the address read out", got)
	}
}

// The conversation records what the model wrote, not what the provider had to
// be told, so a tag added for synthesis never reaches the transcript.
func TestTextTransformDoesNotReachTheConversation(t *testing.T) {
	h := newTransformHarness(t, nil)
	h.base.SetTextTransformers(tts.TextTransformer{
		AggregatedBy: frames.AnyAggregation,
		Transform: func(_ context.Context, text string, _ frames.AggregationType) (string, error) {
			return "<spell>" + text + "</spell>", nil
		},
	})

	h.runTurn(t, "Hello there.")

	for _, got := range h.recorded() {
		if strings.Contains(got, "<spell>") {
			t.Errorf("the conversation recorded %q, want the text as written", got)
		}
	}
}

// A transform registered against one way of grouping text is not applied to
// units grouped another way.
func TestTextTransformOnlyRunsForItsAggregationType(t *testing.T) {
	h := newTransformHarness(t, nil)
	var ran int
	h.base.SetTextTransformers(tts.TextTransformer{
		AggregatedBy: frames.AggregationToken,
		Transform: func(_ context.Context, text string, _ frames.AggregationType) (string, error) {
			ran++
			return "changed", nil
		},
	})

	h.runTurn(t, "Hello there.")

	if ran != 0 {
		t.Errorf("a token transform ran %d times on sentences", ran)
	}
	got := h.syn.texts()
	if len(got) != 1 || got[0] != "Hello there." {
		t.Errorf("spoken = %q, want the sentence untouched", got)
	}
}

// Transforms run in the order they were registered, each seeing what the one
// before it produced.
func TestTextTransformsRunInOrder(t *testing.T) {
	h := newTransformHarness(t, nil)
	h.base.AddTextTransformer(tts.TextTransformer{
		AggregatedBy: frames.AnyAggregation,
		Transform: func(_ context.Context, text string, _ frames.AggregationType) (string, error) {
			return text + " one", nil
		},
	})
	h.base.AddTextTransformer(tts.TextTransformer{
		AggregatedBy: frames.AggregationSentence,
		Transform: func(_ context.Context, text string, _ frames.AggregationType) (string, error) {
			return text + " two", nil
		},
	})

	h.runTurn(t, "Start.")

	got := h.syn.texts()
	if len(got) != 1 || got[0] != "Start. one two" {
		t.Errorf("spoken = %q, want [\"Start. one two\"]", got)
	}
}

// A transform that fails is application code failing, not the provider: the
// error says so, the service stays usable, and the unit is left unspoken.
// Speaking the untransformed text would defeat a transform that exists to take
// something out.
func TestFailingTextTransformLeavesTheServiceUsable(t *testing.T) {
	h := newTransformHarness(t, nil)
	h.base.SetTextTransformers(tts.TextTransformer{
		AggregatedBy: frames.AnyAggregation,
		Transform: func(_ context.Context, _ string, _ frames.AggregationType) (string, error) {
			return "", errTransform
		},
	})

	h.runTurn(t, "Hello there.")

	errFrames := h.errorFrames()
	if len(errFrames) == 0 {
		t.Fatal("a failing transform reported nothing")
	}
	if errFrames[0].Category != errs.Application {
		t.Errorf("category = %q, want %q", errFrames[0].Category, errs.Application)
	}
	if !h.base.Usable() {
		t.Error("application code failing cost the service its usability")
	}
	if got := h.syn.texts(); len(got) != 0 {
		t.Errorf("spoken = %q, want nothing said", got)
	}
}

package rtvi_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/utils/events"
)

func off() *bool { v := false; return &v }

// types reports the type of each message, in order.
func types(msgs []rtvi.Message) []string {
	out := make([]string, len(msgs))
	for i, m := range msgs {
		out[i] = m.Type
	}
	return out
}

// sent reports whether any message of that type was sent.
func sent(msgs []rtvi.Message, msgType string) bool {
	for _, m := range msgs {
		if m.Type == msgType {
			return true
		}
	}
	return false
}

// conversation is one small exchange touching every category a client can turn
// off, so one feed can be run against several configurations.
func conversation() []frames.Frame {
	return []frames.Frame{
		frames.NewUserStartedSpeakingFrame(),
		frames.NewTranscriptionFrame("hello", "user-1", "2026-08-29T00:00:00Z"),
		frames.NewUserStoppedSpeakingFrame(),
		frames.NewLLMFullResponseStartFrame(),
		frames.NewLLMTextFrame("Hi there."),
		frames.NewLLMFullResponseEndFrame(),
		frames.NewTTSStartedFrame(),
		frames.NewTTSTextFrame("Hi there."),
		frames.NewTTSStoppedFrame(),
		frames.NewBotStartedSpeakingFrame(),
		frames.NewBotStoppedSpeakingFrame(),
	}
}

// Left alone, every category a client normally wants is reported: the defaults
// are what an unconfigured observer sends.
func TestEveryCategoryIsReportedByDefault(t *testing.T) {
	msgs := observerHarness(t, rtvi.DefaultObserverParams(), conversation()...)

	for _, want := range []string{
		rtvi.TypeUserStartedSpeaking,
		rtvi.TypeUserTranscription,
		rtvi.TypeBotLLMStarted,
		rtvi.TypeBotLLMText,
		rtvi.TypeBotTTSStarted,
		rtvi.TypeBotTTSText,
		rtvi.TypeBotStartedSpeaking,
	} {
		if !sent(msgs, want) {
			t.Errorf("no %s message, got %v", want, types(msgs))
		}
	}
}

// Turning off what the model produced silences the response brackets and the
// text inside them, and nothing else.
func TestBotLLMCategoryCanBeTurnedOff(t *testing.T) {
	params := rtvi.DefaultObserverParams()
	params.BotLLMEnabled = off()
	msgs := observerHarness(t, params, conversation()...)

	for _, gone := range []string{rtvi.TypeBotLLMStarted, rtvi.TypeBotLLMStopped, rtvi.TypeBotLLMText} {
		if sent(msgs, gone) {
			t.Errorf("%s was reported with the category off, got %v", gone, types(msgs))
		}
	}
	if !sent(msgs, rtvi.TypeBotTTSText) {
		t.Errorf("turning off the model's text silenced the voice too, got %v", types(msgs))
	}
}

// Turning off the voice silences the synthesis brackets and the spoken caption.
func TestBotTTSCategoryCanBeTurnedOff(t *testing.T) {
	params := rtvi.DefaultObserverParams()
	params.BotTTSEnabled = off()
	msgs := observerHarness(t, params, conversation()...)

	for _, gone := range []string{rtvi.TypeBotTTSStarted, rtvi.TypeBotTTSStopped, rtvi.TypeBotTTSText} {
		if sent(msgs, gone) {
			t.Errorf("%s was reported with the category off, got %v", gone, types(msgs))
		}
	}
	if !sent(msgs, rtvi.TypeBotLLMText) {
		t.Errorf("turning off the voice silenced the model too, got %v", types(msgs))
	}
}

// Turning off the transcription silences both the interim guesses and the final
// transcript, leaving the turn events that bracket them.
func TestUserTranscriptionCategoryCanBeTurnedOff(t *testing.T) {
	params := rtvi.DefaultObserverParams()
	params.UserTranscriptionEnabled = off()
	msgs := observerHarness(t, params, conversation()...)

	if sent(msgs, rtvi.TypeUserTranscription) {
		t.Errorf("a transcription was reported with the category off, got %v", types(msgs))
	}
	if !sent(msgs, rtvi.TypeUserStartedSpeaking) {
		t.Errorf("turning off the transcript silenced the turn too, got %v", types(msgs))
	}
}

// Turning off the user's turn silences its start and end, leaving what was said.
func TestUserSpeakingCategoryCanBeTurnedOff(t *testing.T) {
	params := rtvi.DefaultObserverParams()
	params.UserSpeakingEnabled = off()
	msgs := observerHarness(t, params, conversation()...)

	for _, gone := range []string{rtvi.TypeUserStartedSpeaking, rtvi.TypeUserStoppedSpeaking} {
		if sent(msgs, gone) {
			t.Errorf("%s was reported with the category off, got %v", gone, types(msgs))
		}
	}
	if !sent(msgs, rtvi.TypeUserTranscription) {
		t.Errorf("turning off the turn silenced the transcript too, got %v", types(msgs))
	}
}

// Turning off the bot's playback silences its start and stop.
func TestBotSpeakingCategoryCanBeTurnedOff(t *testing.T) {
	params := rtvi.DefaultObserverParams()
	params.BotSpeakingEnabled = off()
	msgs := observerHarness(t, params, conversation()...)

	for _, gone := range []string{rtvi.TypeBotStartedSpeaking, rtvi.TypeBotStoppedSpeaking} {
		if sent(msgs, gone) {
			t.Errorf("%s was reported with the category off, got %v", gone, types(msgs))
		}
	}
}

// Metrics are reported by default and can be turned off on their own.
func TestMetricsCategoryCanBeTurnedOff(t *testing.T) {
	metrics := frames.NewMetricsFrame(frames.TTFBMetricsData{
		BaseMetricsData: frames.BaseMetricsData{Processor: "TTS"},
		Value:           time.Second,
	})
	if msgs := observerHarness(t, rtvi.DefaultObserverParams(), metrics); !sent(msgs, rtvi.TypeMetrics) {
		t.Fatalf("metrics are not reported by default, got %v", types(msgs))
	}

	params := rtvi.DefaultObserverParams()
	params.MetricsEnabled = off()
	metrics2 := frames.NewMetricsFrame(frames.TTFBMetricsData{
		BaseMetricsData: frames.BaseMetricsData{Processor: "TTS"},
		Value:           time.Second,
	})
	if msgs := observerHarness(t, params, metrics2); sent(msgs, rtvi.TypeMetrics) {
		t.Errorf("metrics were reported with the category off, got %v", types(msgs))
	}
}

// An error explains a conversation that stopped working, so it is reported
// whatever else has been turned off.
func TestErrorsAreReportedWhateverIsTurnedOff(t *testing.T) {
	params := rtvi.ObserverParams{
		BotLLMEnabled:            off(),
		BotTTSEnabled:            off(),
		BotSpeakingEnabled:       off(),
		UserSpeakingEnabled:      off(),
		UserTranscriptionEnabled: off(),
		MetricsEnabled:           off(),
	}
	msgs := observerHarness(t, params, frames.NewErrorFrame("it broke"))

	if !sent(msgs, rtvi.TypeError) {
		t.Errorf("the error was not reported, got %v", types(msgs))
	}
}

// A branch of the pipeline the client is not meant to see says nothing at all.
// An evaluation model answering alongside the real one has a conversation of its
// own, and none of it is the client's business.
func TestIgnoredSourceIsSilent(t *testing.T) {
	msgs, ignoredMsgs := runTwoBranches(t)

	if !sent(msgs, rtvi.TypeBotLLMText) {
		t.Fatalf("the branch said nothing even before it was ignored, got %v", types(msgs))
	}
	if sent(ignoredMsgs, rtvi.TypeBotLLMText) {
		t.Errorf("an ignored branch was still reported, got %v", types(ignoredMsgs))
	}
}

// evalBranch stands for a secondary branch of the pipeline: it answers on its
// own account, so the frames it pushes have it as their source.
type evalBranch struct {
	*processor.Base
}

func newEvalBranch() *evalBranch {
	b := &evalBranch{}
	b.Base = processor.New("EvalBranch", b)
	return b
}

func (b *evalBranch) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := b.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if err := b.PushFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.StartFrame); !ok {
		return nil
	}
	// Its own answer, which no client asked for.
	return b.PushFrame(ctx, frames.NewLLMTextFrame("only the evaluator hears this"),
		processor.Downstream)
}

// runTwoBranches runs the same secondary branch twice, once reported and once
// ignored.
func runTwoBranches(t *testing.T) (reported, ignored []rtvi.Message) {
	t.Helper()
	return collectFrom(t, false), collectFrom(t, true)
}

// collectFrom runs a pipeline whose first processor answers on its own account,
// optionally telling the observer to ignore that processor.
func collectFrom(t *testing.T, ignore bool) []rtvi.Message {
	t.Helper()

	var (
		mu   sync.Mutex
		msgs []rtvi.Message
	)
	branch := newEvalBranch()
	proc := rtvi.NewProcessor()
	obs := rtvi.NewObserverWithParams(proc, rtvi.DefaultObserverParams())
	if ignore {
		obs.AddIgnoredSource(branch)
	}
	// The branch sits at the end, so what it answers on its own account goes no
	// further and the only handover of that frame has the branch as its source.
	task := pipeline.NewWorker(pipeline.New(proc, branch), pipeline.WorkerConfig{
		Observers:               []pipeline.Observer{obs},
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if m, ok := f.(*frames.OutputTransportMessageUrgentFrame); ok {
			if msg, ok := m.Message.(rtvi.Message); ok {
				mu.Lock()
				msgs = append(msgs, msg)
				mu.Unlock()
			}
		}
	})

	done := make(chan error, 1)
	go func() { done <- task.Run(t.Context()) }()

	// Let the observer's own goroutine catch up before the run is ended, since
	// whatever is still queued for it when the pipeline stops is dropped.
	time.Sleep(300 * time.Millisecond)
	task.StopWhenDone()
	if err := <-done; err != nil {
		t.Fatal(err)
	}

	mu.Lock()
	defer mu.Unlock()
	return append([]rtvi.Message(nil), msgs...)
}

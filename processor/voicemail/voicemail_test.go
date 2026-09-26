package voicemail_test

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/voicemail"
	llmservice "github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/utils/events"
)

const (
	// verdictSettle is long enough for a verdict to reach the delayed voicemail
	// handler, which fires voicemailDelay after the verdict.
	verdictSettle   = time.Second
	voicemailDelay  = 100 * time.Millisecond
	decisionTimeout = 200 * time.Millisecond
)

// answer is one scripted classifier answer: a label and a confidence, or an
// error.
type answer struct {
	label      string
	confidence float64
	err        error
}

// fakeClassifier answers the voicemail question from a scripted list.
type fakeClassifier struct {
	*classifier.Base

	mu        sync.Mutex
	answers   []answer
	asked     []string
	setUp     bool
	cleanedUp bool
}

func newFakeClassifier(answers ...answer) *fakeClassifier {
	c := &fakeClassifier{answers: answers}
	c.Base = classifier.NewBase("FakeClassifier", "", c)
	return c
}

func (c *fakeClassifier) Model() string { return "" }

func (c *fakeClassifier) Setup(context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.setUp = true
	return nil
}

func (c *fakeClassifier) Cleanup(ctx context.Context) error {
	c.mu.Lock()
	c.cleanedUp = true
	c.mu.Unlock()
	return c.Base.Cleanup(ctx)
}

func (c *fakeClassifier) AskQuestions(
	_ context.Context, state any, questions []classifier.Named[classifier.Question],
) (map[string]classifier.Result, *frames.LLMTokenUsage, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.asked = append(c.asked, fmt.Sprint(state))
	a := c.answers[0]
	c.answers = c.answers[1:]
	if a.err != nil {
		return nil, nil, a.err
	}
	results := map[string]classifier.Result{}
	for _, q := range questions {
		cq, _ := q.Question.(classifier.ChoiceQuestion)
		probabilities := map[string]float64{}
		for _, o := range cq.Options {
			if o.Name == a.label {
				probabilities[o.Name] = 1
			} else {
				probabilities[o.Name] = 0
			}
		}
		results[q.Name] = classifier.ChoiceResult{Choice: a.label, Probabilities: probabilities, Confidence: a.confidence}
	}
	return results, nil, nil
}

func (c *fakeClassifier) askedSoFar() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return slices.Clone(c.asked)
}

func newDetector(t *testing.T, cfg voicemail.Config, answers ...answer) (*voicemail.Detector, *fakeClassifier) {
	t.Helper()
	c := newFakeClassifier(answers...)
	cfg.Classifier = c
	if cfg.VoicemailResponseDelay == 0 {
		cfg.VoicemailResponseDelay = voicemailDelay
	}
	if cfg.DecisionTimeout == 0 {
		cfg.DecisionTimeout = decisionTimeout
	}
	d, err := voicemail.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return d, c
}

// fired counts the times an event of the detector fired, and what it fired with.
type fired struct {
	mu      sync.Mutex
	sources []any
}

func (f *fired) record(d *voicemail.Detector, event string) {
	d.Events().Add(event, func(_ context.Context, source any, _ ...any) {
		f.mu.Lock()
		f.sources = append(f.sources, source)
		f.mu.Unlock()
	})
}

func (f *fired) count() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.sources)
}

func onVoicemail(d *voicemail.Detector) *fired {
	f := &fired{}
	f.record(d, voicemail.EventVoicemailDetected)
	return f
}

func onConversation(d *voicemail.Detector) *fired {
	f := &fired{}
	f.record(d, voicemail.EventConversationDetected)
	return f
}

// wantFiredWith checks the event fired once, with the detector as its source.
func wantFiredWith(t *testing.T, f *fired, d *voicemail.Detector) {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.sources) != 1 || f.sources[0] != d {
		t.Fatalf("fired with %v, want the detector once", f.sources)
	}
}

func said(text string) frames.Frame { return frames.NewTranscriptionFrame(text, "", "") }

// run sends the script through a pipeline of procs, where a time.Duration in the
// script is a pause, and returns what reached the end of it and whether the
// pipeline ended on its own.
func run(t *testing.T, procs []processor.Processor, script ...any) ([]frames.Frame, bool) {
	t.Helper()
	off := false
	w := pipeline.NewWorker(pipeline.New(procs...), pipeline.WorkerConfig{
		IdleTimeout:             -1,
		EnableRTVI:              &off,
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	var (
		mu   sync.Mutex
		down []frames.Frame
	)
	events.On(&w.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		down = append(down, f)
		mu.Unlock()
	})
	done := make(chan error, 1)
	go func() { done <- w.Run(context.Background()) }()
	time.Sleep(50 * time.Millisecond)
	ended := false
	for _, step := range script {
		switch s := step.(type) {
		case time.Duration:
			select {
			case err := <-done:
				if err != nil {
					t.Fatal(err)
				}
				ended = true
			case <-time.After(s):
			}
		case frames.Frame:
			w.QueueFrame(s)
		}
		if ended {
			break
		}
	}
	if !ended {
		w.StopWhenDone()
		select {
		case err := <-done:
			if err != nil {
				t.Fatal(err)
			}
		case <-time.After(5 * time.Second):
			t.Fatal("the pipeline did not end")
		}
	}
	mu.Lock()
	defer mu.Unlock()
	return slices.Clone(down), ended
}

func names(fs []frames.Frame) []string {
	out := make([]string, 0, len(fs))
	for _, f := range fs {
		n, _, _ := strings.Cut(f.Name(), "#")
		out = append(out, n)
	}
	return out
}

// Verdicts

func TestVoicemailVerdictFiresHandlerWithTheTranscript(t *testing.T) {
	d, c := newDetector(t, voicemail.Config{}, answer{label: "voicemail", confidence: 0.95})
	f := onVoicemail(d)
	run(t, []processor.Processor{d}, said("Hi, you've reached Sam."), verdictSettle)
	wantFiredWith(t, f, d)
	if got := c.askedSoFar(); !slices.Equal(got, []string{"Hi, you've reached Sam."}) {
		t.Fatalf("asked %q", got)
	}
}

func TestEachClassificationPushesItsMetrics(t *testing.T) {
	d, c := newDetector(t, voicemail.Config{}, answer{label: "conversation", confidence: 0.9})
	down, _ := run(t, []processor.Processor{d}, said("Hello?"), verdictSettle)
	var metrics []*frames.MetricsFrame
	for _, f := range down {
		if m, ok := f.(*frames.MetricsFrame); ok {
			metrics = append(metrics, m)
		}
	}
	if len(metrics) != 1 || len(metrics[0].Data) != 1 {
		t.Fatalf("got %v, want one metrics frame with the processing time", metrics)
	}
	processing, ok := metrics[0].Data[0].(frames.ProcessingMetricsData)
	if !ok || processing.Processor != c.Name() {
		t.Fatalf("got %+v, want processing metrics under the classifier's name", metrics[0].Data[0])
	}
}

func TestConversationVerdictFiresHandler(t *testing.T) {
	d, _ := newDetector(t, voicemail.Config{}, answer{label: "conversation", confidence: 0.9})
	f := onConversation(d)
	run(t, []processor.Processor{d}, said("Hello?"), verdictSettle)
	wantFiredWith(t, f, d)
}

func TestALaterAnswerReplacesAnEarlierOne(t *testing.T) {
	d, c := newDetector(t, voicemail.Config{},
		answer{label: "voicemail", confidence: 0.4}, answer{label: "voicemail", confidence: 0.95})
	f := onVoicemail(d)
	run(t, []processor.Processor{d},
		frames.NewUserStartedSpeakingFrame(),
		said("Hi,"),
		200*time.Millisecond,
		said("you've reached Sam. Leave a message."),
		frames.NewUserStoppedSpeakingFrame(),
		verdictSettle,
	)
	if f.count() != 1 {
		t.Fatalf("fired %d times, want once", f.count())
	}
	if got := c.askedSoFar(); !slices.Equal(got, []string{"Hi,", "Hi, you've reached Sam. Leave a message."}) {
		t.Fatalf("asked %q", got)
	}
}

func testAnErrorDoesNotDecide(t *testing.T, err error) {
	t.Helper()
	d, c := newDetector(t, voicemail.Config{}, answer{err: err}, answer{label: "conversation", confidence: 0.9})
	f := onConversation(d)
	run(t, []processor.Processor{d},
		frames.NewUserStartedSpeakingFrame(),
		said("Hello?"),
		200*time.Millisecond,
		said("Anyone there?"),
		frames.NewUserStoppedSpeakingFrame(),
		verdictSettle,
	)
	if f.count() != 1 || len(c.askedSoFar()) != 2 {
		t.Fatalf("fired %d times after %d questions, want once after two", f.count(), len(c.askedSoFar()))
	}
}

func TestClassifierErrorDoesNotDecide(t *testing.T) {
	testAnErrorDoesNotDecide(t, fmt.Errorf("%w: down", classifier.ErrClassifier))
}

//nolint:err113 // an error of the kind a classifier does not wrap
func TestAnUnexpectedClassifierErrorDoesNotHangTheDecision(t *testing.T) {
	testAnErrorDoesNotDecide(t, errors.New("connection reset"))
}

func TestSetupAndCleanupReachTheClassifier(t *testing.T) {
	d, c := newDetector(t, voicemail.Config{}, answer{label: "conversation", confidence: 0.9})
	run(t, []processor.Processor{d}, said("Hello?"))
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.setUp || !c.cleanedUp {
		t.Fatalf("set up %v, cleaned up %v; want both", c.setUp, c.cleanedUp)
	}
}

// Silence decides with the latest answer; more speech restarts the wait.

func TestSilenceDecidesWithTheBestAnswerSoFar(t *testing.T) {
	d, _ := newDetector(t, voicemail.Config{}, answer{label: "voicemail", confidence: 0.3})
	f := onVoicemail(d)
	run(t, []processor.Processor{d},
		said("Hi, this is Sam."), frames.NewUserStoppedSpeakingFrame(), 500*time.Millisecond+verdictSettle)
	wantFiredWith(t, f, d)
}

func TestTheMessageIsLeftOnce(t *testing.T) {
	// A beep or a prompt heard after the message started must not leave it
	// again.
	d, _ := newDetector(t, voicemail.Config{}, answer{label: "voicemail", confidence: 0.9})
	f := onVoicemail(d)
	run(t, []processor.Processor{d},
		frames.NewUserStartedSpeakingFrame(),
		said("Hi, you've reached Sam. Leave a message after the tone."),
		frames.NewUserStoppedSpeakingFrame(),
		verdictSettle,
		frames.NewUserStartedSpeakingFrame(),
		100*time.Millisecond,
		frames.NewUserStoppedSpeakingFrame(),
		verdictSettle,
	)
	if f.count() != 1 {
		t.Fatalf("fired %d times, want once", f.count())
	}
}

func TestMoreSpeechCancelsTheSilenceDecision(t *testing.T) {
	d, _ := newDetector(t, voicemail.Config{DecisionTimeout: 300 * time.Millisecond},
		answer{label: "voicemail", confidence: 0.3}, answer{label: "conversation", confidence: 0.3})
	v, c := onVoicemail(d), onConversation(d)
	run(t, []processor.Processor{d},
		said("Hi, this is Sam."),
		frames.NewUserStoppedSpeakingFrame(),
		100*time.Millisecond,
		frames.NewUserStartedSpeakingFrame(),
		400*time.Millisecond,
	)
	if v.count() != 0 || c.count() != 0 {
		t.Fatalf("decided while the caller was speaking: %d voicemail, %d conversation", v.count(), c.count())
	}
}

func TestAVoicemailVerdictWaitsForTheCallerToStop(t *testing.T) {
	// The caller is still speaking: "Hi, this is Sam" looks like a person, then
	// the greeting goes on and turns out to be a voicemail. Neither verdict acts
	// until the caller stops.
	answers := []answer{{label: "conversation", confidence: 0.99}, {label: "voicemail", confidence: 0.98}}
	speaking := []any{
		frames.NewUserStartedSpeakingFrame(),
		said("Hi, this is Sam."),
		200 * time.Millisecond,
		said("Sorry I missed your call, leave a message."),
		verdictSettle,
	}

	d, _ := newDetector(t, voicemail.Config{}, answers...)
	v, c := onVoicemail(d), onConversation(d)
	run(t, []processor.Processor{d}, speaking...)
	if v.count() != 0 || c.count() != 0 {
		t.Fatalf("decided while the caller was speaking: %d voicemail, %d conversation", v.count(), c.count())
	}

	d, _ = newDetector(t, voicemail.Config{}, answers...)
	v, c = onVoicemail(d), onConversation(d)
	speaking[0] = frames.NewUserStartedSpeakingFrame()
	speaking[1] = said("Hi, this is Sam.")
	speaking[3] = said("Sorry I missed your call, leave a message.")
	run(t, []processor.Processor{d}, append(speaking, frames.NewUserStoppedSpeakingFrame(), verdictSettle)...)
	if c.count() != 0 {
		t.Fatalf("a conversation was decided %d times, want none", c.count())
	}
	wantFiredWith(t, v, d)
}

func TestAConfidentConversationDecidesAfterTheSilence(t *testing.T) {
	d, _ := newDetector(t, voicemail.Config{DecisionTimeout: 400 * time.Millisecond},
		answer{label: "conversation", confidence: 0.99})
	c := onConversation(d)
	run(t, []processor.Processor{d},
		frames.NewUserStartedSpeakingFrame(), said("Hello?"), 100*time.Millisecond,
		frames.NewUserStoppedSpeakingFrame(), 200*time.Millisecond)
	if c.count() != 0 {
		t.Fatal("a conversation verdict did not wait out the silence")
	}

	d, _ = newDetector(t, voicemail.Config{DecisionTimeout: 200 * time.Millisecond},
		answer{label: "conversation", confidence: 0.99})
	c = onConversation(d)
	run(t, []processor.Processor{d},
		frames.NewUserStartedSpeakingFrame(), said("Hello?"), 100*time.Millisecond,
		frames.NewUserStoppedSpeakingFrame(), 500*time.Millisecond)
	wantFiredWith(t, c, d)
}

func TestSilenceWithoutAnyAnswerAssumesAConversation(t *testing.T) {
	d, _ := newDetector(t, voicemail.Config{}, answer{err: fmt.Errorf("%w: down", classifier.ErrClassifier)})
	c := onConversation(d)
	run(t, []processor.Processor{d},
		said("Hello?"), frames.NewUserStoppedSpeakingFrame(), 500*time.Millisecond+verdictSettle)
	wantFiredWith(t, c, d)
}

// Gating

func TestConversationReleasesHeldSpeech(t *testing.T) {
	d, _ := newDetector(t, voicemail.Config{}, answer{label: "conversation", confidence: 0.9})
	down, _ := run(t, []processor.Processor{d, d.Gate()},
		frames.NewTTSStartedFrame(),
		frames.NewTTSTextFrame("Hi, this is Alex.", frames.AggregationSentence),
		200*time.Millisecond,
		said("Hello?"),
		verdictSettle,
	)
	got := names(down)
	started, text := slices.Index(got, "TTSStartedFrame"), slices.Index(got, "TTSTextFrame")
	if text == -1 || started == -1 || started > text {
		t.Fatalf("got %v, want the held speech released in order", got)
	}
}

func TestVoicemailDropsHeldSpeechAndBlocksLaterInput(t *testing.T) {
	d, _ := newDetector(t, voicemail.Config{}, answer{label: "voicemail", confidence: 0.95})
	down, _ := run(t, []processor.Processor{d, d.Gate()},
		frames.NewTTSStartedFrame(),
		frames.NewTTSTextFrame("Hi, this is Alex.", frames.AggregationSentence),
		200*time.Millisecond,
		said("Please leave a message."),
		verdictSettle,
		frames.NewLLMTextFrame("should not reach the conversation"),
		200*time.Millisecond,
	)
	got := names(down)
	if slices.Contains(got, "TTSTextFrame") || slices.Contains(got, "LLMTextFrame") {
		t.Fatalf("got %v, want the held speech dropped and later input blocked", got)
	}
}

func TestTranscriptionsPassThroughBeforeAVerdict(t *testing.T) {
	d, _ := newDetector(t, voicemail.Config{}, answer{label: "conversation", confidence: 0.9})
	down, _ := run(t, []processor.Processor{d}, said("Hello?"), verdictSettle)
	if !slices.Contains(names(down), "TranscriptionFrame") {
		t.Fatalf("got %v, want the transcription passed on", names(down))
	}
}

// A handler can end the call, whichever way it pushes the request.

func TestHandlerEndsTheWorker(t *testing.T) {
	for _, dir := range []processor.Direction{processor.Upstream, processor.Downstream} {
		t.Run(dir.String(), func(t *testing.T) {
			d, _ := newDetector(t, voicemail.Config{}, answer{label: "voicemail", confidence: 0.95})
			events.OnSignal(d.Events(), voicemail.EventVoicemailDetected, func(ctx context.Context) {
				end := frames.NewEndWorkerFrame()
				end.Reason = "Voicemail detected."
				_ = d.PushFrame(ctx, end, dir)
			})
			if _, ended := run(t, []processor.Processor{d}, said("Please leave a message."), 3*verdictSettle); !ended {
				t.Fatal("the EndWorkerFrame did not end the worker")
			}
		})
	}
}

// Deprecated LLM path

type silentLLM struct{}

func (silentLLM) RunInference(context.Context, *frames.LLMContext, llmservice.InferenceOptions) (string, error) {
	return "", nil
}

func TestNeedsAClassifierOrAnLLM(t *testing.T) {
	if _, err := voicemail.New(voicemail.Config{}); err == nil {
		t.Fatal("a detector with nothing to decide with was built")
	}
}

func TestLLMBuildsAnLLMClassifier(t *testing.T) {
	//nolint:staticcheck // the deprecated path is what is under test
	d, err := voicemail.New(voicemail.Config{LLM: silentLLM{}, CustomSystemPrompt: "Extra"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(d.Classifier().Name(), "LLMClassifier#") {
		t.Fatalf("classifier = %s, want an LLM classifier", d.Classifier().Name())
	}
}

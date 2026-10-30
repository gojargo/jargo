package turns

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

const settle = 100 * time.Millisecond

// fakeAsker answers each classification with the next scripted answer: an
// option, or an error.
type fakeAsker struct {
	mu      sync.Mutex
	answers []any
	delay   time.Duration
	asked   []map[string]string
}

func (a *fakeAsker) AskQuestions(
	ctx context.Context, state any, questions []classifier.Named[classifier.Question],
) (map[string]classifier.Result, *frames.LLMTokenUsage, error) {
	a.mu.Lock()
	asked, _ := state.(map[string]string)
	a.asked = append(a.asked, asked)
	answer := a.answers[0]
	a.answers = a.answers[1:]
	a.mu.Unlock()
	if a.delay > 0 {
		select {
		case <-time.After(a.delay):
		case <-ctx.Done():
			return nil, nil, fmt.Errorf("%w: %w", classifier.ErrClassifier, ctx.Err())
		}
	}
	if err, ok := answer.(error); ok {
		return nil, nil, err
	}
	choice, _ := answer.(string)
	results := map[string]classifier.Result{}
	for _, q := range questions {
		probs := map[string]float64{}
		question, _ := q.Question.(classifier.ChoiceQuestion)
		for _, name := range question.OptionNames() {
			if name == choice {
				probs[name] = 1
			} else {
				probs[name] = 0
			}
		}
		results[q.Name] = classifier.ChoiceResult{Choice: choice, Probabilities: probs, Confidence: 0.9}
	}
	return results, nil, nil
}

func (a *fakeAsker) Model() string { return "" }

func (a *fakeAsker) askedStates() []map[string]string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return append([]map[string]string(nil), a.asked...)
}

// fakeClassifier is a classifier answering through a fakeAsker, recording how
// it was set up and cleaned up.
type fakeClassifier struct {
	*classifier.Base
	asker     *fakeAsker
	mu        sync.Mutex
	setUp     bool
	cleanedUp bool
}

func newFakeClassifier(delay time.Duration, answers ...any) *fakeClassifier {
	a := &fakeAsker{answers: answers, delay: delay}
	return &fakeClassifier{Base: classifier.NewBase("FakeClassifier", "", a), asker: a}
}

func (c *fakeClassifier) Model() string { return "" }

func (c *fakeClassifier) Setup(ctx context.Context) error {
	c.mu.Lock()
	c.setUp = true
	c.mu.Unlock()
	return c.Base.Setup(ctx)
}

func (c *fakeClassifier) Cleanup(ctx context.Context) error {
	c.mu.Lock()
	c.cleanedUp = true
	c.mu.Unlock()
	return c.Base.Cleanup(ctx)
}

// vadStop decides the user is done as soon as the VAD stops, like a quick
// Smart Turn.
type vadStopStrategy struct{ StopStrategyBase }

func newVADStopStrategy() *vadStopStrategy {
	s := &vadStopStrategy{}
	s.EnableUserSpeakingFrames = true
	return s
}

func (s *vadStopStrategy) Process(f frames.Frame) ProcessFrameResult {
	if _, ok := f.(*frames.VADUserStoppedSpeakingFrame); ok {
		s.ReportStopped()
	}
	return Continue
}

func said(text string, finalized bool) *frames.TranscriptionFrame {
	f := frames.NewTranscriptionFrame(text, "", "")
	f.Finalized = finalized
	return f
}

func vadStarted() frames.Frame { return frames.NewVADUserStartedSpeakingFrame(0, time.Now()) }
func vadStopped() frames.Frame { return frames.NewVADUserStoppedSpeakingFrame(0, time.Now()) }

type classifierHarness struct {
	t          *testing.T
	classifier *fakeClassifier
	s          *ClassifierStop
	spy        *spy
}

func newClassifierHarness(
	t *testing.T, cfg ClassifierStopConfig, sttLatency, delay time.Duration, answers ...any,
) *classifierHarness {
	t.Helper()
	c := newFakeClassifier(delay, answers...)
	cfg.Classifier = c
	s, err := NewClassifierStop(cfg)
	if err != nil {
		t.Fatal(err)
	}
	h := &classifierHarness{t: t, classifier: c, s: s, spy: attachStop(s)}
	if err := s.Setup(processor.Setup{}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		h.spy.mu.Lock()
		s.Cleanup()
		h.spy.mu.Unlock()
	})
	md := frames.NewSTTMetadataFrame(sttLatency)
	h.send(md)
	h.spy.mu.Lock()
	s.TurnStarted()
	h.spy.mu.Unlock()
	return h
}

func (h *classifierHarness) send(fs ...frames.Frame) {
	for _, f := range fs {
		h.spy.sendStop(h.s, f)
	}
}

func (h *classifierHarness) say(texts ...string) {
	h.send(vadStarted())
	for _, text := range texts {
		h.send(said(text, false))
	}
	h.send(vadStopped())
	time.Sleep(settle)
}

func (h *classifierHarness) asked() []map[string]string { return h.classifier.asker.askedStates() }

func user(text string) map[string]string { return map[string]string{"user": text} }

func TestClassifierStopCompleteReportsThatTheUserIsDone(t *testing.T) {
	h := newClassifierHarness(t, ClassifierStopConfig{}, 0, 0, "complete")
	h.say("I'd like to", "book a table.")

	if got := h.asked(); !reflect.DeepEqual(got, []map[string]string{user("I'd like to book a table.")}) {
		t.Fatalf("asked = %v", got)
	}
	if h.spy.stops() != 1 {
		t.Fatalf("stops = %d, want 1", h.spy.stops())
	}
}

func TestClassifierStopNothingIsAskedWhileTheUserSpeaks(t *testing.T) {
	h := newClassifierHarness(t, ClassifierStopConfig{}, 0, 0, "complete")
	h.send(vadStarted(), said("I'd like to", false))
	time.Sleep(settle)
	if got := h.asked(); len(got) != 0 {
		t.Fatalf("asked = %v while the user speaks", got)
	}

	h.send(vadStopped())
	time.Sleep(settle)
	if got := h.asked(); len(got) != 1 {
		t.Fatalf("asked %d times, want 1", len(got))
	}
}

func TestClassifierStopATranscriptionAfterThePauseIsClassified(t *testing.T) {
	h := newClassifierHarness(t, ClassifierStopConfig{}, 0, 0, "complete")
	h.say()
	if got := h.asked(); len(got) != 0 {
		t.Fatalf("asked = %v about nothing", got)
	}

	h.send(said("book a table.", false))
	time.Sleep(settle)
	if got := h.asked(); !reflect.DeepEqual(got, []map[string]string{user("book a table.")}) {
		t.Fatalf("asked = %v", got)
	}
	if h.spy.stops() != 1 {
		t.Fatalf("stops = %d, want 1", h.spy.stops())
	}
}

func TestClassifierStopAPauseWaitsForTheSTTLatency(t *testing.T) {
	h := newClassifierHarness(t, ClassifierStopConfig{}, 300*time.Millisecond, 0, "complete")
	h.say("book a table.")
	if got := h.asked(); len(got) != 0 {
		t.Fatalf("asked = %v before the STT latency passed", got)
	}

	time.Sleep(300 * time.Millisecond)
	if h.spy.stops() != 1 {
		t.Fatalf("stops = %d, want 1", h.spy.stops())
	}
}

func TestClassifierStopTheSTTLatencyCountsFromWhenTheUserStopped(t *testing.T) {
	h := newClassifierHarness(t, ClassifierStopConfig{}, 300*time.Millisecond, 0, "complete")
	h.send(vadStarted(), said("book a table.", false), frames.NewVADUserStoppedSpeakingFrame(0.3, time.Now()))
	time.Sleep(settle)

	if h.spy.stops() != 1 {
		t.Fatalf("stops = %d, want 1", h.spy.stops())
	}
}

func TestClassifierStopAFinalizedTranscriptionDoesNotWaitForTheSTTLatency(t *testing.T) {
	h := newClassifierHarness(t, ClassifierStopConfig{}, time.Second, 0, "complete")
	h.say()
	h.send(said("book a table.", true))
	time.Sleep(settle)

	if h.spy.stops() != 1 {
		t.Fatalf("stops = %d, want 1", h.spy.stops())
	}
}

// The new words' transcription usually lands after the next pause.
func TestClassifierStopSpeakingAgainWaitsForTheNewWords(t *testing.T) {
	h := newClassifierHarness(t, ClassifierStopConfig{ShortTimeout: 5 * time.Second}, time.Second, 0,
		"short", "complete")
	h.say()
	h.send(said("I'd like to", true))
	time.Sleep(settle)
	h.say()
	if got := h.asked(); len(got) != 1 {
		t.Fatalf("asked %d times, want 1 until the new words are in", len(got))
	}

	h.send(said("book a table.", true))
	time.Sleep(settle)
	if got := h.asked(); len(got) != 2 || !reflect.DeepEqual(got[1], user("I'd like to book a table.")) {
		t.Fatalf("asked = %v, want the whole turn second", got)
	}
	if h.spy.stops() != 1 {
		t.Fatalf("stops = %d, want 1", h.spy.stops())
	}
}

// A transcription service can finalize the user's words before the VAD
// reports the stop.
func TestClassifierStopATranscriptionFinalizedBeforeTheStopDoesNotWait(t *testing.T) {
	h := newClassifierHarness(t, ClassifierStopConfig{}, time.Second, 0, "complete")
	h.send(vadStarted(), said("book a table.", true), vadStopped())
	time.Sleep(settle)

	if h.spy.stops() != 1 {
		t.Fatalf("stops = %d, want 1", h.spy.stops())
	}
}

// An interim transcription means more words are on their way.
func TestClassifierStopAnInterimTranscriptionHoldsThePause(t *testing.T) {
	h := newClassifierHarness(t, ClassifierStopConfig{ShortTimeout: 5 * time.Second}, time.Second, 0,
		"short", "complete")
	h.say()
	h.send(said("book a", true), frames.NewInterimTranscriptionFrame("table", "", ""), said("table", false))
	time.Sleep(settle)

	if got := h.asked(); !reflect.DeepEqual(got, []map[string]string{user("book a")}) {
		t.Fatalf("asked = %v", got)
	}
}

// A service proposes the stop after pushing the transcription.
func TestClassifierStopAProposedStopDoesNotWaitForTheSTTLatency(t *testing.T) {
	h := newClassifierHarness(t, ClassifierStopConfig{}, time.Second, 0, "complete")
	h.send(frames.NewProposedUserStartedSpeakingFrame(), said("book a table.", false),
		frames.NewProposedUserStoppedSpeakingFrame())
	time.Sleep(settle)

	if h.spy.stops() != 1 {
		t.Fatalf("stops = %d, want 1", h.spy.stops())
	}
}

func TestClassifierStopTheSameWordsAreAskedAboutOnce(t *testing.T) {
	h := newClassifierHarness(t, ClassifierStopConfig{}, 0, 0, "complete")
	h.say("book a table.")
	h.send(vadStopped())
	time.Sleep(settle)

	if got := h.asked(); len(got) != 1 {
		t.Fatalf("asked %d times, want 1", len(got))
	}
}

// The controller clears the stop decisions on any speech, such as a cough.
func TestClassifierStopSpeechWithoutNewWordsAsksAgain(t *testing.T) {
	h := newClassifierHarness(t, ClassifierStopConfig{}, 0, 0, "complete", "complete")
	h.say("book a table.")
	h.say()

	if got := h.asked(); len(got) != 2 {
		t.Fatalf("asked %d times, want 2", len(got))
	}
	if h.spy.stops() != 2 {
		t.Fatalf("stops = %d, want 2", h.spy.stops())
	}
}

func (h *classifierHarness) scheduled() bool {
	h.spy.mu.Lock()
	defer h.spy.mu.Unlock()
	return h.s.DecisionScheduled()
}

func TestClassifierStopShortWaitsBeforeReporting(t *testing.T) {
	h := newClassifierHarness(t, ClassifierStopConfig{ShortTimeout: 300 * time.Millisecond}, 0, 0, "short")
	h.say("I'd like to")
	if h.spy.stops() != 0 || !h.scheduled() {
		t.Fatalf("stops = %d, scheduled = %v, want 0 and a decision scheduled", h.spy.stops(), h.scheduled())
	}

	time.Sleep(300 * time.Millisecond)
	if h.spy.stops() != 1 || h.scheduled() {
		t.Fatalf("stops = %d, scheduled = %v, want 1 and none", h.spy.stops(), h.scheduled())
	}
}

func TestClassifierStopLongWaitsBeforeReporting(t *testing.T) {
	h := newClassifierHarness(t, ClassifierStopConfig{LongTimeout: 300 * time.Millisecond}, 0, 0, "long")
	h.say("let me think")
	if h.spy.stops() != 0 {
		t.Fatalf("stops = %d, want 0", h.spy.stops())
	}

	time.Sleep(300 * time.Millisecond)
	if h.spy.stops() != 1 {
		t.Fatalf("stops = %d, want 1", h.spy.stops())
	}
}

func TestClassifierStopSpeakingAgainEndsTheWaitAndClassifiesTheWholeTurn(t *testing.T) {
	h := newClassifierHarness(t, ClassifierStopConfig{ShortTimeout: 300 * time.Millisecond}, 0, 0,
		"short", "complete")
	h.say("I'd like to")
	h.say("book a table.")
	time.Sleep(300 * time.Millisecond)

	if got := h.asked(); len(got) != 2 || !reflect.DeepEqual(got[1], user("I'd like to book a table.")) {
		t.Fatalf("asked = %v, want the whole turn second", got)
	}
	if h.spy.stops() != 1 {
		t.Fatalf("stops = %d, want 1", h.spy.stops())
	}
}

func TestClassifierStopSpeakingAgainDropsAVerdictOnItsWay(t *testing.T) {
	h := newClassifierHarness(t, ClassifierStopConfig{ShortTimeout: 5 * time.Second}, 0,
		200*time.Millisecond, "complete", "short")
	h.send(said("I'd like to", false), vadStarted())
	time.Sleep(300 * time.Millisecond)

	if h.spy.stops() != 0 {
		t.Fatalf("stops = %d, want the verdict dropped", h.spy.stops())
	}
}

func TestClassifierStopAClassifierErrorLeavesTheTurnToTheOthers(t *testing.T) {
	h := newClassifierHarness(t, ClassifierStopConfig{}, 0, 0,
		fmt.Errorf("%w: down", classifier.ErrClassifier))
	h.say("hello")

	if h.spy.stops() != 1 {
		t.Fatalf("stops = %d, want 1", h.spy.stops())
	}
}

func TestClassifierStopASlowClassifierLeavesTheTurnToTheOthers(t *testing.T) {
	h := newClassifierHarness(t, ClassifierStopConfig{ClassificationTimeout: 100 * time.Millisecond}, 0,
		500*time.Millisecond, "short")
	h.say("hello")
	time.Sleep(100 * time.Millisecond)

	if h.spy.stops() != 1 {
		t.Fatalf("stops = %d, want 1", h.spy.stops())
	}
}

func TestClassifierStopNothingIsAskedOutsideATurn(t *testing.T) {
	h := newClassifierHarness(t, ClassifierStopConfig{}, 0, 0, "complete")
	h.spy.mu.Lock()
	h.s.TurnStopped()
	h.spy.mu.Unlock()
	h.say("hello")

	if got := h.asked(); len(got) != 0 {
		t.Fatalf("asked = %v outside a turn", got)
	}
}

func TestClassifierStopContextAddsTheAssistantsLastMessage(t *testing.T) {
	convo := frames.NewLLMContext("Be brief.")
	convo.AddMessage(frames.Message{Role: frames.RoleAssistant, Text: "Where would you go?"})
	convo.AddUserMessage("hmm")
	h := newClassifierHarness(t, ClassifierStopConfig{Context: convo, LongTimeout: 5 * time.Second}, 0, 0, "long")
	h.say("that's interesting")

	want := []map[string]string{{"assistant": "Where would you go?", "user": "that's interesting"}}
	if got := h.asked(); !reflect.DeepEqual(got, want) {
		t.Fatalf("asked = %v, want %v", got, want)
	}
}

func TestClassifierStopEachClassificationPushesItsMetrics(t *testing.T) {
	h := newClassifierHarness(t, ClassifierStopConfig{}, 0, 0, "complete")
	h.say("book a table.")

	var metrics []*frames.MetricsFrame
	eventually(t, func() bool {
		h.spy.mu.Lock()
		defer h.spy.mu.Unlock()
		metrics = nil
		for _, f := range h.spy.pushedFrame {
			if m, ok := f.(*frames.MetricsFrame); ok {
				metrics = append(metrics, m)
			}
		}
		return len(metrics) == 1
	}, time.Second, "the classification pushed no metrics")
	data := metrics[0].Data
	if len(data) != 1 {
		t.Fatalf("metrics = %v, want the processing time only", data)
	}
	processing, ok := data[0].(frames.ProcessingMetricsData)
	if !ok || processing.Processor != h.classifier.Name() {
		t.Fatalf("metrics = %v, want the classifier's processing time", data[0])
	}
}

func TestClassifierStopTheClassifierIsSetUpAndCleanedUp(t *testing.T) {
	h := newClassifierHarness(t, ClassifierStopConfig{}, 0, 0, "complete")
	h.classifier.mu.Lock()
	setUp := h.classifier.setUp
	h.classifier.mu.Unlock()
	if !setUp {
		t.Fatal("the classifier was not set up")
	}

	h.spy.mu.Lock()
	h.s.Cleanup()
	h.spy.mu.Unlock()
	h.classifier.mu.Lock()
	defer h.classifier.mu.Unlock()
	if !h.classifier.cleanedUp {
		t.Fatal("the classifier was not cleaned up")
	}
}

// ClassifierStop in the turn controller.

type classifierControllerHarness struct {
	c     *UserTurnController
	mu    sync.Mutex
	stops []StopStrategy
}

func newClassifierController(
	t *testing.T, withVAD bool, cfg ClassifierStopConfig, answers ...any,
) *classifierControllerHarness {
	t.Helper()
	cfg.Classifier = newFakeClassifier(0, answers...)
	cs, err := NewClassifierStop(cfg)
	if err != nil {
		t.Fatal(err)
	}
	var stop []StopStrategy
	if withVAD {
		stop = append(stop, newVADStopStrategy())
	}
	stop = append(stop, cs)
	h := &classifierControllerHarness{}
	h.c = NewUserTurnController(UserTurnStrategies{Start: []StartStrategy{NewVADStart()}, Stop: stop},
		200*time.Millisecond)
	h.c.SetHooks(ControllerHooks{
		Stopped: func(_ context.Context, s StopStrategy, _ UserTurnStoppedParams) {
			h.mu.Lock()
			h.stops = append(h.stops, s)
			h.mu.Unlock()
		},
	})
	if err := h.c.Setup(t.Context(), processor.Setup{}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(h.c.Cleanup)
	return h
}

func (h *classifierControllerHarness) say(text string) {
	h.c.Process(vadStarted())
	h.c.Process(said(text, false))
	h.c.Process(vadStopped())
	time.Sleep(settle)
}

func (h *classifierControllerHarness) stopped() []StopStrategy {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]StopStrategy(nil), h.stops...)
}

func TestTheClassifierAloneEndsACompleteTurn(t *testing.T) {
	h := newClassifierController(t, false, ClassifierStopConfig{}, "complete")
	h.say("I think I would like to see Kyoto.")

	if n := len(h.stopped()); n != 1 {
		t.Fatalf("stops = %d, want 1", n)
	}
}

func TestTheClassifierAloneHoldsAnUnfinishedTurn(t *testing.T) {
	h := newClassifierController(t, false, ClassifierStopConfig{ShortTimeout: 500 * time.Millisecond}, "short")
	h.say("I'd like to")
	time.Sleep(300 * time.Millisecond)
	if n := len(h.stopped()); n != 0 {
		t.Fatalf("stops = %d, want the turn held", n)
	}

	time.Sleep(300 * time.Millisecond)
	if n := len(h.stopped()); n != 1 {
		t.Fatalf("stops = %d, want 1", n)
	}
}

func TestTheTurnEndsOnceTheVADStrategyAndTheClassifierDecide(t *testing.T) {
	h := newClassifierController(t, true, ClassifierStopConfig{}, "complete")
	h.say("book a table.")

	stops := h.stopped()
	if len(stops) != 1 {
		t.Fatalf("stops = %d, want 1", len(stops))
	}
	if _, ok := stops[0].(*ClassifierStop); !ok {
		t.Fatalf("stopped by %T, want the classifier strategy", stops[0])
	}
}

func TestShortKeepsTheTurnOpenPastTheVADStrategyAndTheStopTimeout(t *testing.T) {
	h := newClassifierController(t, true, ClassifierStopConfig{ShortTimeout: 500 * time.Millisecond}, "short")
	h.say("I'd like to")
	time.Sleep(300 * time.Millisecond)
	if n := len(h.stopped()); n != 0 {
		t.Fatalf("stops = %d, want the turn held", n)
	}

	time.Sleep(300 * time.Millisecond)
	stops := h.stopped()
	if len(stops) != 1 {
		t.Fatalf("stops = %d, want 1", len(stops))
	}
	if _, ok := stops[0].(*ClassifierStop); !ok {
		t.Fatalf("stopped by %T, want the classifier strategy", stops[0])
	}
}

// ClassifierUserTurnStrategies.

func TestClassifierStrategiesAddTheClassifierAfterTheDefaultStopStrategies(t *testing.T) {
	strategies, err := ClassifierUserTurnStrategies(nil, ClassifierStopConfig{Classifier: newFakeClassifier(0)})
	if err != nil {
		t.Fatal(err)
	}
	if len(strategies.Stop) != 2 {
		t.Fatalf("stop = %d strategies, want 2", len(strategies.Stop))
	}
	if _, ok := strategies.Stop[0].(*TurnAnalyzerStop); !ok {
		t.Errorf("first = %T, want the default turn analyzer", strategies.Stop[0])
	}
	if _, ok := strategies.Stop[1].(*ClassifierStop); !ok {
		t.Errorf("second = %T, want the classifier strategy", strategies.Stop[1])
	}
}

func TestClassifierStrategiesAddTheClassifierToTheStopStrategiesGiven(t *testing.T) {
	c := newFakeClassifier(0)
	vad := newVADStopStrategy()
	strategies, err := ClassifierUserTurnStrategies([]StopStrategy{vad}, ClassifierStopConfig{Classifier: c})
	if err != nil {
		t.Fatal(err)
	}
	if strategies.Stop[0] != StopStrategy(vad) {
		t.Errorf("first = %T, want the one given", strategies.Stop[0])
	}
	cs, ok := strategies.Stop[1].(*ClassifierStop)
	if !ok || cs.Classifier() != classifier.Classifier(c) {
		t.Errorf("second = %T, want the classifier strategy with the classifier given", strategies.Stop[1])
	}
}

func TestClassifierStrategiesExtendingAServiceAddsOnlyTheClassifierAndGivenStrategies(t *testing.T) {
	strategies, err := ClassifierUserTurnStrategies(nil, ClassifierStopConfig{Classifier: newFakeClassifier(0)})
	if err != nil {
		t.Fatal(err)
	}
	if len(strategies.stopExtension) != 1 {
		t.Fatalf("extension = %d strategies, want 1", len(strategies.stopExtension))
	}
	if _, ok := strategies.stopExtension[0].(*ClassifierStop); !ok {
		t.Fatalf("extension = %T, want the classifier strategy", strategies.stopExtension[0])
	}
}

func TestClassifierStrategiesNeedAClassifier(t *testing.T) {
	_, err := ClassifierUserTurnStrategies([]StopStrategy{newVADStopStrategy()}, ClassifierStopConfig{})
	if err == nil {
		t.Fatal("built strategies without a classifier")
	}
	if errors.Is(err, classifier.ErrClassifier) {
		t.Fatal("a configuration error is not a classifier failure")
	}
}

func TestTheTurnCompletionQuestionHasTheThreeVerdicts(t *testing.T) {
	got := TurnCompletionQuestion.OptionNames()
	if !reflect.DeepEqual(got, []string{"complete", "short", "long"}) {
		t.Fatalf("options = %v", got)
	}
}

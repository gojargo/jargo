package perception_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/expressive"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/perception"
	"github.com/gojargo/jargo/utils/events"
)

// sampleRate is the sample rate of the pipeline's input audio in these tests.
const sampleRate = 16000

var (
	errDown    = fmt.Errorf("%w: down", classifier.ErrClassifier)
	errNoModel = fmt.Errorf("%w: no model", classifier.ErrClassifier)
	errNoKey   = fmt.Errorf("%w: no key", classifier.ErrClassifier)
)

func question(name string) perception.NamedQuestion {
	switch name {
	case "tone":
		return perception.NamedQuestion{Name: name, Question: perception.Question{
			Question: classifier.ChoiceQuestion{
				Instructions: "How does the user sound?",
				Options:      []classifier.Option{{Name: "calm"}, {Name: "frustrated"}, {Name: "confused"}},
			},
			Description: "How the user sounded.",
		}}
	case "hesitation":
		return perception.NamedQuestion{Name: name, Question: perception.Question{
			Question:    classifier.YesNoQuestion{Instructions: "Does the user sound unsure?"},
			Description: "The chance the user sounded unsure, from 0 to 1.",
		}}
	default:
		return perception.NamedQuestion{Name: name, Question: perception.Question{
			Question:    classifier.ScoreQuestion{Instructions: "How energetic is the user?", Levels: []any{"low", "high"}},
			Description: "From 0, low, to 1, high.",
		}}
	}
}

func questions(names ...string) []perception.NamedQuestion {
	out := make([]perception.NamedQuestion, 0, len(names))
	for _, name := range names {
		out = append(out, question(name))
	}
	return out
}

// allQuestions is the three questions every test asks.
func allQuestions() []perception.NamedQuestion { return questions("tone", "hesitation", "energy") }

func results() map[string]classifier.Result {
	return map[string]classifier.Result{
		"tone": classifier.ChoiceResult{
			Choice:        "frustrated",
			Probabilities: map[string]float64{"calm": 0.1, "frustrated": 0.7, "confused": 0.2},
			Confidence:    0.6,
		},
		"hesitation": classifier.YesNoResult{Probability: 0.6432},
		"energy": classifier.ScoreResult{
			Score:      0.678,
			Levels:     []classifier.ScoreLevel{{Level: "low", Probability: 0.32}, {Level: "high", Probability: 0.68}},
			Confidence: 0.4,
		},
	}
}

func value(name string, v float64) expressive.PerceptionSignal {
	return expressive.PerceptionSignal{Name: name, Value: &v}
}

func labeled(name string, labels ...expressive.PerceptionLabel) expressive.PerceptionSignal {
	return expressive.PerceptionSignal{Name: name, Labels: labels}
}

func label(l string, c float64) expressive.PerceptionLabel {
	return expressive.PerceptionLabel{Label: l, Confidence: c}
}

//nolint:gochecknoglobals // test fixtures
var (
	tone       = labeled("tone", label("frustrated", 0.7), label("confused", 0.2))
	hesitation = value("hesitation", 0.64)
	energy     = value("energy", 0.68)
	speechRate = value("speech_rate", 4.1)
)

// fakeClassifier reads text, answering with results().
type fakeClassifier struct {
	*classifier.Base
	err, setupErr error
	delay         time.Duration
	// other, when set, is a classifier this one answers only once it is asked
	// too, so only when both are asked at once.
	other    *fakeClassifier
	askedNow chan struct{}
	nowOnce  sync.Once

	mu    sync.Mutex
	asked []any
}

func newFakeClassifier() *fakeClassifier {
	c := &fakeClassifier{askedNow: make(chan struct{})}
	c.Base = classifier.NewBase("FakeClassifier", "", c)
	return c
}

func (c *fakeClassifier) Model() string { return "" }

func (c *fakeClassifier) Setup(ctx context.Context) error {
	if err := c.Base.Setup(ctx); err != nil {
		return err
	}
	return c.setupErr
}

func (c *fakeClassifier) AskQuestions(
	ctx context.Context, state any, qs []classifier.Named[classifier.Question], _ []classifier.Input,
) (map[string]classifier.Result, *frames.LLMTokenUsage, error) {
	c.nowOnce.Do(func() { close(c.askedNow) })
	if c.other != nil {
		select {
		case <-c.other.askedNow:
		case <-ctx.Done():
			return nil, nil, ctx.Err()
		}
	}
	c.mu.Lock()
	c.asked = append(c.asked, state)
	c.mu.Unlock()
	if !sleep(ctx, c.delay) {
		return nil, nil, ctx.Err()
	}
	if c.err != nil {
		return nil, nil, c.err
	}
	all := results()
	out := map[string]classifier.Result{}
	for _, q := range qs {
		out[q.Name] = all[q.Name]
	}
	return out, nil, nil
}

func (c *fakeClassifier) states() []any {
	c.mu.Lock()
	defer c.mu.Unlock()
	return slices.Clone(c.asked)
}

// fakeAudioClassifier hears clips, answering with results(), or the nth clip's
// hesitation with probabilities[n].
type fakeAudioClassifier struct {
	*classifier.Base
	min, max, preferred time.Duration
	delays              []time.Duration
	probabilities       []float64
	setupErr            error

	mu     sync.Mutex
	clips  []classifier.Input
	states []any
}

func newFakeAudioClassifier() *fakeAudioClassifier {
	c := &fakeAudioClassifier{min: time.Second}
	c.Base = classifier.NewBase("FakeAudioClassifier", "", c)
	return c
}

func (c *fakeAudioClassifier) Model() string           { return "" }
func (c *fakeAudioClassifier) SupportsAudio() bool     { return true }
func (c *fakeAudioClassifier) MinAudio() time.Duration { return c.min }
func (c *fakeAudioClassifier) MaxAudio() time.Duration { return c.max }
func (c *fakeAudioClassifier) PreferredAudio() time.Duration {
	if c.preferred == 0 {
		return c.min
	}
	return c.preferred
}

func (c *fakeAudioClassifier) Setup(ctx context.Context) error {
	if err := c.Base.Setup(ctx); err != nil {
		return err
	}
	return c.setupErr
}

func (c *fakeAudioClassifier) AskQuestions(
	ctx context.Context, state any, qs []classifier.Named[classifier.Question], inputs []classifier.Input,
) (map[string]classifier.Result, *frames.LLMTokenUsage, error) {
	c.mu.Lock()
	call := len(c.clips)
	var clip classifier.Input
	if len(inputs) > 0 {
		clip = inputs[0]
	}
	c.clips = append(c.clips, clip)
	c.states = append(c.states, state)
	c.mu.Unlock()
	if call < len(c.delays) && !sleep(ctx, c.delays[call]) {
		return nil, nil, ctx.Err()
	}
	all := results()
	if len(c.probabilities) > 0 {
		all["hesitation"] = classifier.YesNoResult{Probability: c.probabilities[call]}
	}
	out := map[string]classifier.Result{}
	for _, q := range qs {
		out[q.Name] = all[q.Name]
	}
	return out, nil, nil
}

func (c *fakeAudioClassifier) heard() []classifier.Input {
	c.mu.Lock()
	defer c.mu.Unlock()
	return slices.Clone(c.clips)
}

func (c *fakeAudioClassifier) asked() []any {
	c.mu.Lock()
	defer c.mu.Unlock()
	return slices.Clone(c.states)
}

func sleep(ctx context.Context, d time.Duration) bool {
	if d <= 0 {
		return true
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-ctx.Done():
		return false
	}
}

func source(t *testing.T, c classifier.Classifier, timeout *time.Duration, names ...string) perception.Source {
	t.Helper()
	s, err := perception.NewSource(perception.SourceConfig{
		Classifier: c, Questions: questions(names...), Timeout: timeout,
	})
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func newReader(t *testing.T, cfg perception.Config) *perception.Reader {
	t.Helper()
	r, err := perception.New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// reader is a reader with one source, which asks c these questions.
func reader(
	t *testing.T, c classifier.Classifier, qs []perception.NamedQuestion, timeout *time.Duration,
) *perception.Reader {
	t.Helper()
	s, err := perception.NewSource(perception.SourceConfig{Classifier: c, Questions: qs, Timeout: timeout})
	if err != nil {
		t.Fatal(err)
	}
	return newReader(t, perception.Config{Sources: []perception.Source{s}})
}

func secs(s float64) *time.Duration {
	d := time.Duration(s * float64(time.Second))
	return &d
}

func transcription(text string) *frames.TranscriptionFrame {
	return frames.NewTranscriptionFrame(text, "", "now")
}

func finalizedTranscription(text string) *frames.TranscriptionFrame {
	f := transcription(text)
	f.Finalized = true
	return f
}

func readByTheSTT(f *frames.TranscriptionFrame) *frames.TranscriptionFrame {
	f.Perception = []expressive.PerceptionSignal{speechRate}
	return f
}

func pcm(v byte, s float64) []byte {
	return bytes.Repeat([]byte{v}, int(sampleRate*s)*2)
}

func audioFrame(v byte, s float64) *frames.InputAudioRawFrame {
	return frames.NewInputAudioRawFrame(pcm(v, s), sampleRate, 1)
}

func clipOf(parts ...[]byte) classifier.Input {
	return classifier.Audio{Audio: bytes.Join(parts, nil), SampleRate: sampleRate}
}

func vadStarted() frames.Frame { return frames.NewVADUserStartedSpeakingFrame(0, time.Now()) }

func vadStopped() frames.Frame { return frames.NewVADUserStoppedSpeakingFrame(0, time.Now()) }

func assistantTurn(text string) frames.Frame {
	return frames.NewLLMContextAssistantTurnFrame(text, "now")
}

func ms(n int) time.Duration { return time.Duration(n) * time.Millisecond }

// run runs p alone in a pipeline started with params, sends it each frame in
// order (a time.Duration sleeps that long instead), lets it finish, and returns
// the frames that came out of it downstream and upstream, without the
// StartFrame and EndFrame.
func run(t *testing.T, p processor.Processor, params pipeline.Params, send ...any) (down, up []frames.Frame) {
	t.Helper()
	var mu sync.Mutex
	noRTVI := false
	task := pipeline.NewWorker(pipeline.New(p), pipeline.WorkerConfig{
		Params:                  params,
		EnableRTVI:              &noRTVI,
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		switch f.(type) {
		case *frames.StartFrame, *frames.EndFrame:
			return
		}
		mu.Lock()
		down = append(down, f)
		mu.Unlock()
	})
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		up = append(up, f)
		mu.Unlock()
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	for _, s := range send {
		switch v := s.(type) {
		case time.Duration:
			time.Sleep(v)
		case frames.Frame:
			task.QueueFrame(v, processor.Downstream)
		default:
			t.Fatalf("cannot send %T", s)
		}
	}
	task.StopWhenDone()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	return down, up
}

func of[T frames.Frame](fs []frames.Frame) []T {
	var out []T
	for _, f := range fs {
		if v, ok := f.(T); ok {
			out = append(out, v)
		}
	}
	return out
}

func names(fs []frames.Frame) []string {
	out := make([]string, len(fs))
	for i, f := range fs {
		out[i], _, _ = strings.Cut(f.Name(), "#")
	}
	return out
}

// kinds is the names of the frames, leaving out the ones the pipeline adds
// besides the reader's own.
func kinds(fs []frames.Frame, want ...string) []string {
	var out []string
	for _, n := range names(fs) {
		if slices.Contains(want, n) {
			out = append(out, n)
		}
	}
	return out
}

func perceptions(fs []frames.Frame) [][]expressive.PerceptionSignal {
	var out [][]expressive.PerceptionSignal
	for _, f := range of[*frames.TranscriptionFrame](fs) {
		out = append(out, f.Perception)
	}
	return out
}

func readings(fs []frames.Frame) [][]expressive.PerceptionSignal {
	var out [][]expressive.PerceptionSignal
	for _, f := range of[*frames.PerceptionFrame](fs) {
		out = append(out, f.Perception)
	}
	return out
}

func finalized(fs []frames.Frame) []bool {
	var out []bool
	for _, f := range of[*frames.TranscriptionFrame](fs) {
		out = append(out, f.Finalized)
	}
	return out
}

func sttLatencies(fs []frames.Frame) []time.Duration {
	var out []time.Duration
	for _, f := range of[*frames.STTMetadataFrame](fs) {
		out = append(out, f.TTFSP99Latency)
	}
	return out
}

func declared(fs []frames.Frame) [][]string {
	var out [][]string
	for _, f := range of[*frames.PerceptionMetadataFrame](fs) {
		var n []string
		for _, term := range f.Perception.Signals {
			n = append(n, term.Name)
		}
		out = append(out, n)
	}
	return out
}

func errorsIn(fs []frames.Frame) []*frames.ErrorFrame { return of[*frames.ErrorFrame](fs) }

func wantPerceptions(t *testing.T, down []frames.Frame, want ...[]expressive.PerceptionSignal) {
	t.Helper()
	if got := perceptions(down); !reflect.DeepEqual(got, want) {
		t.Errorf("perceptions = %s, want %s", formatAll(got), formatAll(want))
	}
}

func formatAll(readings [][]expressive.PerceptionSignal) string {
	parts := make([]string, 0, len(readings))
	for _, r := range readings {
		if r == nil {
			parts = append(parts, "None")
			continue
		}
		parts = append(parts, expressive.FormatPerception(r))
	}
	return "[" + strings.Join(parts, " ") + "]"
}

func wantClips(t *testing.T, c *fakeAudioClassifier, want ...classifier.Input) {
	t.Helper()
	if got := c.heard(); !reflect.DeepEqual(got, want) {
		t.Errorf("clips = %s, want %s", describeClips(got), describeClips(want))
	}
}

func describeClips(clips []classifier.Input) string {
	parts := make([]string, 0, len(clips))
	for _, c := range clips {
		a, ok := c.(classifier.Audio)
		if !ok {
			parts = append(parts, "None")
			continue
		}
		parts = append(parts, fmt.Sprintf("%v@%d", a.Duration(), a.SampleRate))
	}
	return "[" + strings.Join(parts, " ") + "]"
}

// reading is one reading, for wantPerceptions.
func reading(signals ...expressive.PerceptionSignal) []expressive.PerceptionSignal { return signals }

func TestTheSignalsAreDeclaredAtStart(t *testing.T) {
	down, _ := run(t, reader(t, newFakeClassifier(), allQuestions(), nil), pipeline.Params{})

	metadata := of[*frames.PerceptionMetadataFrame](down)
	if len(metadata) == 0 {
		t.Fatalf("down = %v, want a PerceptionMetadataFrame", names(down))
	}
	want := []expressive.PerceptionTerm{
		{Name: "tone", Description: "How the user sounded.", Labels: []string{"calm", "frustrated", "confused"}},
		{Name: "hesitation", Description: "The chance the user sounded unsure, from 0 to 1."},
		{Name: "energy", Description: "From 0, low, to 1, high."},
	}
	if got := metadata[0].Perception.Signals; !reflect.DeepEqual(got, want) {
		t.Errorf("signals = %+v, want %+v", got, want)
	}
}

func TestEachAnswerIsSetAsASignalOnItsTranscription(t *testing.T) {
	down, _ := run(t, reader(t, newFakeClassifier(), allQuestions(), nil), pipeline.Params{},
		transcription("I already gave you my order number."))

	want := []string{"PerceptionMetadataFrame", "PerceptionFrame", "TranscriptionFrame"}
	if got := names(down); !slices.Equal(got, want) {
		t.Fatalf("down = %v", got)
	}
	wantPerceptions(t, down, reading(tone, hesitation, energy))
}

func TestAChoiceReportsAtMostItsMaxLabels(t *testing.T) {
	q := question("tone")
	q.Question.MaxLabels = 1

	down, _ := run(t, reader(t, newFakeClassifier(), []perception.NamedQuestion{q}, nil), pipeline.Params{},
		transcription("Hello."))

	want := []string{"PerceptionMetadataFrame", "PerceptionFrame", "TranscriptionFrame"}
	if got := names(down); !slices.Equal(got, want) {
		t.Fatalf("down = %v", got)
	}
	wantPerceptions(t, down, reading(labeled("tone", label("frustrated", 0.7))))
}

func TestATurnIsAskedAboutWithTheBotsWordsBeforeIt(t *testing.T) {
	c := newFakeClassifier()

	run(t, reader(t, c, allQuestions(), nil), pipeline.Params{},
		transcription("Hi."),
		assistantTurn("What's your order number?"),
		transcription("I already gave it to you."))

	want := []any{
		map[string]any{"user": "Hi."},
		map[string]any{"assistant": "What's your order number?", "user": "I already gave it to you."},
	}
	if got := c.states(); !reflect.DeepEqual(got, want) {
		t.Errorf("states = %v, want %v", got, want)
	}
}

func TestALateReadingLetsTheTranscriptionPassUnread(t *testing.T) {
	c := newFakeClassifier()
	c.delay = time.Second

	down, up := run(t, reader(t, c, allQuestions(), secs(0.05)), pipeline.Params{}, transcription("Hello."))

	if got := names(down); !slices.Equal(got, []string{"PerceptionMetadataFrame", "TranscriptionFrame"}) {
		t.Fatalf("down = %v", got)
	}
	wantPerceptions(t, down, nil)
	if len(errorsIn(up)) != 0 {
		t.Errorf("errors = %v, want none", errorsIn(up))
	}
}

func TestALateReadingIsStillBroadcastOnceMade(t *testing.T) {
	c := newFakeClassifier()
	c.delay = ms(200)

	down, _ := run(t, reader(t, c, allQuestions(), secs(0.05)), pipeline.Params{}, transcription("Hello."), ms(300))

	want := []string{"PerceptionMetadataFrame", "TranscriptionFrame", "PerceptionFrame"}
	if got := names(down); !slices.Equal(got, want) {
		t.Fatalf("down = %v", got)
	}
	wantPerceptions(t, down, nil)
}

func TestAFailedReadingIsReportedAndTheTranscriptionStillPasses(t *testing.T) {
	c := newFakeClassifier()
	c.err = errDown

	down, up := run(t, reader(t, c, allQuestions(), nil), pipeline.Params{}, transcription("Hello."))

	if got := names(down); !slices.Equal(got, []string{"PerceptionMetadataFrame", "TranscriptionFrame"}) {
		t.Fatalf("down = %v", got)
	}
	wantPerceptions(t, down, nil)
	if len(errorsIn(up)) == 0 {
		t.Error("the failed reading was not reported")
	}
}

func TestAClassifierThatCantBeSetUpLeavesTheReaderUnusable(t *testing.T) {
	c := newFakeClassifier()
	c.setupErr = errNoModel
	r := reader(t, c, allQuestions(), nil)

	down, up := run(t, r, pipeline.Params{}, transcription("Hello."))

	if r.Usable() {
		t.Error("the reader is still usable")
	}
	if len(errorsIn(up)) == 0 {
		t.Error("the failed setup was not reported")
	}
	if len(of[*frames.PerceptionMetadataFrame](down)) != 0 {
		t.Error("signals were declared")
	}
	wantPerceptions(t, down, nil)
	if len(c.states()) != 0 {
		t.Errorf("asked %v, want nothing", c.states())
	}
}

func TestTheClassifiersMetricsArePushedWhenMetricsAreOn(t *testing.T) {
	withMetrics, _ := run(t, reader(t, newFakeClassifier(), allQuestions(), nil),
		pipeline.Params{EnableMetrics: true}, transcription("Hello."), ms(200))
	withoutMetrics, _ := run(t, reader(t, newFakeClassifier(), allQuestions(), nil),
		pipeline.Params{}, transcription("Hello."), ms(200))

	if len(of[*frames.MetricsFrame](withMetrics)) == 0 {
		t.Error("no metrics with metrics on")
	}
	if len(of[*frames.MetricsFrame](withoutMetrics)) != 0 {
		t.Error("metrics with metrics off")
	}
}

func TestATextReadingGoesOnAsSoonAsItsRead(t *testing.T) {
	down, _ := run(t, reader(t, newFakeClassifier(), allQuestions(), nil), pipeline.Params{},
		frames.NewSTTMetadataFrame(ms(300)),
		vadStarted(),
		finalizedTranscription("I already gave you my order number."),
		ms(200),
		assistantTurn("Let me check."),
		vadStopped())

	// It doesn't wait for the user to stop speaking, and leaves the STT's
	// finalized and latency as they are.
	if got := kinds(down, "TranscriptionFrame", "LLMContextAssistantTurnFrame"); !slices.Equal(
		got, []string{"TranscriptionFrame", "LLMContextAssistantTurnFrame"}) {
		t.Errorf("released %v", got)
	}
	if got := finalized(down); !slices.Equal(got, []bool{true}) {
		t.Errorf("finalized = %v, want [true]", got)
	}
	if got := sttLatencies(down); !slices.Equal(got, []time.Duration{ms(300)}) {
		t.Errorf("latencies = %v, want [300ms]", got)
	}
}

func audioReader(t *testing.T, c *fakeAudioClassifier, timeout *time.Duration) *perception.Reader {
	t.Helper()
	return reader(t, c, questions("hesitation"), timeout)
}

func TestAnAudioClassifierHearsTheUsersSpeech(t *testing.T) {
	c := newFakeAudioClassifier()

	down, _ := run(t, audioReader(t, c, nil), pipeline.Params{},
		assistantTurn("What's your order number?"),
		ms(50),
		audioFrame(1, 2),
		vadStarted(),
		audioFrame(2, 2),
		vadStopped(),
		transcription("I already gave it to you."))

	// The speech, with the second heard before the VAD reported it.
	wantClips(t, c, clipOf(pcm(1, 1), pcm(2, 2)))
	want := []any{map[string]any{"assistant": "What's your order number?"}}
	if got := c.asked(); !reflect.DeepEqual(got, want) {
		t.Errorf("states = %v, want %v", got, want)
	}
	wantPerceptions(t, down, reading(hesitation))
}

func TestAShortStopIsHeardWithTheUsersSpeechBeforeIt(t *testing.T) {
	c := newFakeAudioClassifier()
	c.min = 1500 * time.Millisecond

	down, _ := run(t, audioReader(t, c, nil), pipeline.Params{},
		vadStarted(), audioFrame(1, 0.5), vadStopped(),
		vadStarted(), audioFrame(2, 1), vadStopped(),
		transcription("I want to... cancel my order."))

	// The first stop's half second is shorter than the classifier takes.
	wantClips(t, c, clipOf(pcm(1, 0.5), pcm(2, 1)))
	wantPerceptions(t, down, reading(hesitation))
}

func TestAShortTurnAtTheStartOfACallIsHeardWithoutSilence(t *testing.T) {
	c := newFakeAudioClassifier()
	c.preferred = 5 * time.Second

	down, _ := run(t, audioReader(t, c, nil), pipeline.Params{},
		vadStarted(), audioFrame(1, 1), vadStopped(), transcription("Fine."))

	wantClips(t, c, clipOf(pcm(1, 1)))
	wantPerceptions(t, down, reading(hesitation))
}

func TestSpeechShorterThanTheClassifierTakesIsntRead(t *testing.T) {
	c := newFakeAudioClassifier()

	down, _ := run(t, audioReader(t, c, nil), pipeline.Params{},
		vadStarted(), audioFrame(1, 0.5), vadStopped(), transcription("No."))

	wantClips(t, c)
	wantPerceptions(t, down, nil)
}

// twoTurns is two user turns, the first read before the second, with the
// second's transcription.
func twoTurns(first, second frames.Frame) []any {
	return []any{
		vadStarted(), first, vadStopped(),
		ms(100),
		frames.NewUserStoppedSpeakingFrame(),
		vadStarted(), second, vadStopped(),
		transcription("No!"),
	}
}

func TestAShortTurnIsHeardWithTheUsersSpeechBeforeIt(t *testing.T) {
	c := newFakeAudioClassifier()
	c.probabilities = []float64{0.2, 0.9}

	down, _ := run(t, audioReader(t, c, nil), pipeline.Params{}, twoTurns(audioFrame(1, 2), audioFrame(2, 0.2))...)

	clips := c.heard()
	if len(clips) != 2 || !reflect.DeepEqual(clips[1], clipOf(pcm(1, 0.8), pcm(2, 0.2))) {
		t.Errorf("clips = %s, want the second 0.8s then 0.2s", describeClips(clips))
	}
	wantPerceptions(t, down, reading(value("hesitation", 0.9)))
}

func TestANewTurnDoesntGetTheLastTurnsReading(t *testing.T) {
	c := newFakeAudioClassifier()
	c.delays = []time.Duration{0, time.Second}
	c.probabilities = []float64{0.2, 0.9}

	down, _ := run(t, audioReader(t, c, secs(0.05)), pipeline.Params{}, twoTurns(audioFrame(1, 2), audioFrame(2, 0.2))...)

	wantPerceptions(t, down, nil)
}

func TestTheQuietBetweenTheUsersSpeechIsShortened(t *testing.T) {
	c := newFakeAudioClassifier()
	c.min = 3 * time.Second

	run(t, audioReader(t, c, nil), pipeline.Params{},
		vadStarted(), audioFrame(1, 1), vadStopped(),
		audioFrame(9, 3),
		vadStarted(), audioFrame(2, 1), vadStopped(),
		transcription("Hello?"))

	// The reply keeps a second of the quiet before it, and the speech before
	// that.
	wantClips(t, c, clipOf(pcm(1, 1), pcm(9, 1), pcm(2, 1)))
}

func TestSpeechBeforeAMuteIsntJoinedToSpeechAfterIt(t *testing.T) {
	c := newFakeAudioClassifier()
	c.preferred = 5 * time.Second

	run(t, audioReader(t, c, nil), pipeline.Params{},
		vadStarted(), audioFrame(1, 2), vadStopped(),
		ms(100),
		frames.NewUserMuteStartedFrame(), audioFrame(2, 0.5), frames.NewUserMuteStoppedFrame(),
		vadStarted(), audioFrame(3, 1), vadStopped(),
		transcription("Hello?"))

	// Nor is the audio heard while the user was muted.
	wantClips(t, c, clipOf(pcm(1, 2)), clipOf(pcm(3, 1)))
}

func TestSpeechBeforeTheBotsTurnIsntJoinedToSpeechAfterIt(t *testing.T) {
	c := newFakeAudioClassifier()
	c.preferred = 5 * time.Second

	run(t, audioReader(t, c, nil), pipeline.Params{},
		vadStarted(), audioFrame(1, 2), vadStopped(),
		frames.NewBotStartedSpeakingFrame(), frames.NewBotStoppedSpeakingFrame(),
		vadStarted(), audioFrame(2, 1), vadStopped(),
		transcription("Yes."))

	wantClips(t, c, clipOf(pcm(1, 2)), clipOf(pcm(2, 1)))
}

func TestSpeechDuringTheBotsTurnIsntJoinedToTheReplyAfterIt(t *testing.T) {
	c := newFakeAudioClassifier()
	c.preferred = 5 * time.Second

	run(t, audioReader(t, c, nil), pipeline.Params{},
		frames.NewBotStartedSpeakingFrame(),
		vadStarted(), audioFrame(1, 1.5), vadStopped(),
		frames.NewBotStoppedSpeakingFrame(),
		vadStarted(), audioFrame(2, 1), vadStopped(),
		transcription("Yes."))

	wantClips(t, c, clipOf(pcm(1, 1.5)), clipOf(pcm(2, 1)))
}

func TestSpeechTheUserIsInWhenTheBotStopsIsKept(t *testing.T) {
	c := newFakeAudioClassifier()
	c.preferred = 5 * time.Second

	run(t, audioReader(t, c, nil), pipeline.Params{},
		vadStarted(), audioFrame(1, 2), vadStopped(),
		vadStarted(), audioFrame(2, 1), frames.NewBotStoppedSpeakingFrame(), audioFrame(2, 1), vadStopped(),
		transcription("Wait, no."))

	wantClips(t, c, clipOf(pcm(1, 2)), clipOf(pcm(2, 2)))
}

func TestSpeechWhileTheUserIsMutedIsntRead(t *testing.T) {
	c := newFakeAudioClassifier()

	down, _ := run(t, audioReader(t, c, nil), pipeline.Params{},
		frames.NewUserMuteStartedFrame(),
		vadStarted(), audioFrame(1, 2), vadStopped(),
		transcription("Hello?"))

	wantClips(t, c)
	wantPerceptions(t, down, nil)
}

func TestALateAudioReadingLetsTheTranscriptionPassUnread(t *testing.T) {
	c := newFakeAudioClassifier()
	c.delays = []time.Duration{time.Second}

	down, up := run(t, audioReader(t, c, secs(0.05)), pipeline.Params{},
		vadStarted(), audioFrame(1, 2), vadStopped(), transcription("Hello."))

	wantPerceptions(t, down, nil)
	if len(errorsIn(up)) != 0 {
		t.Errorf("errors = %v, want none", errorsIn(up))
	}
}

// twoPauses is a turn with a pause after each stretch of speech, the first
// read before the second.
func twoPauses(first, second frames.Frame) []any {
	return []any{
		vadStarted(), first, vadStopped(),
		ms(100),
		vadStarted(), second, vadStopped(),
		transcription("I already told you. No!"),
	}
}

func TestEachPauseSendsTheAudioSinceTheTurnsLastClip(t *testing.T) {
	c := newFakeAudioClassifier()

	run(t, audioReader(t, c, nil), pipeline.Params{}, twoPauses(audioFrame(1, 2), audioFrame(2, 3))...)

	wantClips(t, c, clipOf(pcm(1, 2)), clipOf(pcm(2, 3)))
}

func TestAShortLastStretchReachesBackIntoTheTurn(t *testing.T) {
	c := newFakeAudioClassifier()
	c.min = 2 * time.Second

	run(t, audioReader(t, c, nil), pipeline.Params{}, twoPauses(audioFrame(1, 3), audioFrame(2, 0.5))...)

	// Only as much of the earlier speech as the clip needs, from its end.
	clips := c.heard()
	if len(clips) != 2 || !reflect.DeepEqual(clips[1], clipOf(pcm(1, 1.5), pcm(2, 0.5))) {
		t.Errorf("clips = %s, want the second 1.5s then 0.5s", describeClips(clips))
	}
}

func TestATranscriptionGetsTheAnswersAboutTheTurnsLatestClip(t *testing.T) {
	c := newFakeAudioClassifier()
	c.probabilities = []float64{0.2, 0.9}

	down, _ := run(t, audioReader(t, c, nil), pipeline.Params{}, twoPauses(audioFrame(1, 2), audioFrame(2, 2))...)

	wantPerceptions(t, down, reading(value("hesitation", 0.9)))
}

func TestALateClipLeavesItsSpeechUnread(t *testing.T) {
	c := newFakeAudioClassifier()
	c.delays = []time.Duration{0, time.Second}
	c.probabilities = []float64{0.2, 0.9}

	down, _ := run(t, audioReader(t, c, secs(0.05)), pipeline.Params{}, twoPauses(audioFrame(1, 2), audioFrame(2, 2))...)

	// The earlier clip's reading is the earlier speech's, not this one's.
	wantPerceptions(t, down, nil)
}

func TestAStopDuringAnEarlierReadIsReadOnceItsDone(t *testing.T) {
	c := newFakeAudioClassifier()
	c.delays = []time.Duration{ms(300)}

	run(t, audioReader(t, c, nil), pipeline.Params{},
		vadStarted(), audioFrame(1, 2), vadStopped(),
		vadStarted(), audioFrame(2, 2), vadStopped(),
		transcription("Hello?"))

	wantClips(t, c, clipOf(pcm(1, 2)), clipOf(pcm(2, 2)))
}

func TestOfTheStopsDuringAnEarlierReadOnlyTheNewestIsRead(t *testing.T) {
	c := newFakeAudioClassifier()
	c.delays = []time.Duration{ms(300)}
	c.probabilities = []float64{0.2, 0.9}

	down, _ := run(t, audioReader(t, c, nil), pipeline.Params{},
		vadStarted(), audioFrame(1, 1), vadStopped(),
		vadStarted(), audioFrame(2, 1), vadStopped(),
		vadStarted(), audioFrame(3, 1), vadStopped(),
		transcription("Hello?"))

	// The classifier hears one clip at a time, so the second stop's is dropped.
	wantClips(t, c, clipOf(pcm(1, 1)), clipOf(pcm(3, 1)))
	wantPerceptions(t, down, reading(value("hesitation", 0.9)))
}

func TestAStopDuringAnEarlierReadGetsItsOwnClipsReading(t *testing.T) {
	c := newFakeAudioClassifier()
	c.delays = []time.Duration{ms(500), 0}
	c.probabilities = []float64{0.2, 0.9}

	down, _ := run(t, audioReader(t, c, nil), pipeline.Params{},
		vadStarted(), audioFrame(1, 2), vadStopped(),
		vadStarted(), audioFrame(2, 2), vadStopped(),
		transcription("I already told you. No!"),
		ms(600))

	wantPerceptions(t, down, reading(value("hesitation", 0.9)))
}

func TestATranscriptionOfNoNewSpeechGetsNoReading(t *testing.T) {
	down, _ := run(t, audioReader(t, newFakeAudioClassifier(), nil), pipeline.Params{},
		vadStarted(), audioFrame(1, 2), vadStopped(),
		transcription("Can you check my order?"),
		ms(100),
		// A soft reply the VAD missed: the earlier speech's reading isn't its.
		transcription("Yes."))

	wantPerceptions(t, down, reading(hesitation), nil)
}

func TestATextReadingIsBroadcastBothWays(t *testing.T) {
	down, up := run(t, reader(t, newFakeClassifier(), questions("hesitation"), nil), pipeline.Params{},
		transcription("Hello."))

	want := [][]expressive.PerceptionSignal{{hesitation}}
	if got := readings(down); !reflect.DeepEqual(got, want) {
		t.Errorf("down readings = %s, want %s", formatAll(got), formatAll(want))
	}
	if got := readings(up); !reflect.DeepEqual(got, want) {
		t.Errorf("up readings = %s, want %s", formatAll(got), formatAll(want))
	}
}

func TestEveryClipReadingIsBroadcast(t *testing.T) {
	c := newFakeAudioClassifier()
	c.probabilities = []float64{0.2, 0.9}

	down, up := run(t, audioReader(t, c, nil), pipeline.Params{}, twoPauses(audioFrame(1, 2), audioFrame(2, 2))...)

	want := [][]expressive.PerceptionSignal{{value("hesitation", 0.2)}, {value("hesitation", 0.9)}}
	if got := readings(down); !reflect.DeepEqual(got, want) {
		t.Errorf("down readings = %s, want %s", formatAll(got), formatAll(want))
	}
	if got := readings(up); !reflect.DeepEqual(got, want) {
		t.Errorf("up readings = %s, want %s", formatAll(got), formatAll(want))
	}
}

func TestATranscriptionWaitsForTheUserToStopSpeaking(t *testing.T) {
	down, _ := run(t, audioReader(t, newFakeAudioClassifier(), nil), pipeline.Params{},
		vadStarted(), audioFrame(1, 2),
		transcription("I already told you."),
		ms(100),
		vadStopped())

	wantPerceptions(t, down, reading(hesitation))
}

func TestTheTimeoutCountsFromWhenTheUserStopsSpeaking(t *testing.T) {
	down, _ := run(t, audioReader(t, newFakeAudioClassifier(), secs(0.2)), pipeline.Params{},
		vadStarted(), audioFrame(1, 2),
		transcription("I already told you."),
		ms(500),
		vadStopped())

	wantPerceptions(t, down, reading(hesitation))
}

func TestTheTimeoutCountsFromStopSecsBeforeTheVADStop(t *testing.T) {
	for _, tc := range []struct {
		stopSecs float64
		read     bool
	}{{0, true}, {0.3, false}} {
		t.Run(fmt.Sprint(tc.stopSecs), func(t *testing.T) {
			c := newFakeAudioClassifier()
			c.delays = []time.Duration{ms(300)}

			down, _ := run(t, audioReader(t, c, secs(0.45)), pipeline.Params{},
				vadStarted(), audioFrame(1, 2),
				frames.NewVADUserStoppedSpeakingFrame(tc.stopSecs, time.Now()),
				transcription("I already told you."),
				ms(600))

			if tc.read {
				wantPerceptions(t, down, reading(hesitation))
			} else {
				wantPerceptions(t, down, nil)
			}
		})
	}
}

func TestTranscriptionsOfTheSameSpeechShareOneDeadline(t *testing.T) {
	c := newFakeAudioClassifier()
	c.delays = []time.Duration{ms(300)}

	down, _ := run(t, audioReader(t, c, secs(0.2)), pipeline.Params{},
		vadStarted(), audioFrame(1, 2),
		transcription("Just perfect."),
		transcription("Exactly what I needed today."),
		ms(50),
		vadStopped(),
		ms(600))

	// The second waits no longer than the first, so neither gets the late clip.
	wantPerceptions(t, down, nil, nil)
}

func TestATranscriptionStopsWaitingForAUserWhoKeepsSpeaking(t *testing.T) {
	perception.SetSpeechEndWait(t, ms(100))

	down, _ := run(t, audioReader(t, newFakeAudioClassifier(), nil), pipeline.Params{},
		vadStarted(), audioFrame(1, 2),
		transcription("I already told you."),
		ms(500),
		vadStopped())

	wantPerceptions(t, down, nil)
}

func TestAUserWhoKeepsSpeakingDoesntGetTheirEarlierSpeechsReading(t *testing.T) {
	perception.SetSpeechEndWait(t, ms(100))

	down, _ := run(t, audioReader(t, newFakeAudioClassifier(), nil), pipeline.Params{},
		vadStarted(), audioFrame(1, 2), vadStopped(),
		ms(100),
		vadStarted(), audioFrame(2, 2),
		transcription("And another thing."),
		ms(500),
		vadStopped())

	wantPerceptions(t, down, nil)
}

func TestAClipReadAfterItsTurnIsStillBroadcast(t *testing.T) {
	c := newFakeAudioClassifier()
	c.delays = []time.Duration{ms(300)}

	down, _ := run(t, audioReader(t, c, secs(0.05)), pipeline.Params{},
		vadStarted(), audioFrame(1, 2), vadStopped(),
		transcription("Hello."),
		frames.NewUserStoppedSpeakingFrame(),
		ms(500))

	wantPerceptions(t, down, nil)
	if got, want := readings(down), [][]expressive.PerceptionSignal{{hesitation}}; !reflect.DeepEqual(got, want) {
		t.Errorf("readings = %s, want %s", formatAll(got), formatAll(want))
	}
}

func TestAnInterruptionDoesntDropATranscriptionWaitingForItsReading(t *testing.T) {
	c := newFakeAudioClassifier()
	c.delays = []time.Duration{ms(200)}

	down, _ := run(t, audioReader(t, c, nil), pipeline.Params{},
		vadStarted(), audioFrame(1, 2), vadStopped(),
		transcription("I've been on hold for an hour."),
		ms(50),
		frames.NewInterruptionFrame(),
		ms(400))

	wantPerceptions(t, down, reading(hesitation))
}

func TestOnlyTheLastTranscriptionOfTheUsersSpeechIsFinalized(t *testing.T) {
	down, _ := run(t, audioReader(t, newFakeAudioClassifier(), nil), pipeline.Params{},
		vadStarted(), audioFrame(1, 2),
		finalizedTranscription("Just perfect."),
		transcription("Exactly what I needed today."),
		ms(50),
		vadStopped(),
		ms(100))

	if got := finalized(down); !slices.Equal(got, []bool{false, true}) {
		t.Errorf("finalized = %v, want [false true]", got)
	}
}

func TestATranscriptionSentWithinTheSTTsLatencyGoesWithTheRest(t *testing.T) {
	down, _ := run(t, audioReader(t, newFakeAudioClassifier(), nil), pipeline.Params{},
		frames.NewSTTMetadataFrame(ms(300)),
		vadStarted(), audioFrame(1, 2),
		transcription("This is ridiculous."),
		vadStopped(),
		ms(150),
		transcription("Nobody can tell me anything."),
		ms(400))

	if got := finalized(down); !slices.Equal(got, []bool{false, true}) {
		t.Errorf("finalized = %v, want [false true]", got)
	}
}

func TestNoTranscriptionIsFinalizedWhenTheUserSpeaksAgainBeforeTheReading(t *testing.T) {
	c := newFakeAudioClassifier()
	c.delays = []time.Duration{ms(200)}

	down, _ := run(t, audioReader(t, c, nil), pipeline.Params{},
		vadStarted(), audioFrame(1, 2), vadStopped(),
		transcription("This is ridiculous."),
		ms(50),
		vadStarted(),
		ms(300),
		vadStopped())

	if got := finalized(down); !slices.Equal(got, []bool{false}) {
		t.Errorf("finalized = %v, want [false]", got)
	}
}

func TestAnAudioReaderAddsItsWaitToTheSTTLatencyItPassesOn(t *testing.T) {
	down, _ := run(t, audioReader(t, newFakeAudioClassifier(), secs(0.5)), pipeline.Params{},
		frames.NewSTTMetadataFrame(ms(300)))

	if got := sttLatencies(down); !slices.Equal(got, []time.Duration{ms(800)}) {
		t.Errorf("latencies = %v, want [800ms]", got)
	}
}

func TestAnUnusableAudioReaderPassesTheSTTLatencyOnAsItIs(t *testing.T) {
	c := newFakeAudioClassifier()
	c.setupErr = errNoKey

	down, _ := run(t, audioReader(t, c, nil), pipeline.Params{}, frames.NewSTTMetadataFrame(ms(300)))

	if got := sttLatencies(down); !slices.Equal(got, []time.Duration{ms(300)}) {
		t.Errorf("latencies = %v, want [300ms]", got)
	}
}

// speech is two seconds of the user's speech, then its transcription.
func speech(text string) []any {
	return []any{vadStarted(), audioFrame(1, 2), vadStopped(), transcription(text)}
}

func TestTheSourcesSignalsAreDeclaredTogether(t *testing.T) {
	r := newReader(t, perception.Config{Sources: []perception.Source{
		source(t, newFakeClassifier(), nil, "tone"),
		source(t, newFakeAudioClassifier(), nil, "hesitation", "energy"),
	}})

	down, _ := run(t, r, pipeline.Params{})

	if got, want := declared(down), [][]string{{"tone", "hesitation", "energy"}}; !reflect.DeepEqual(got, want) {
		t.Errorf("declared %v, want %v", got, want)
	}
}

func TestTextSourcesWriteOneReadingInTheirOrder(t *testing.T) {
	r := newReader(t, perception.Config{Sources: []perception.Source{
		source(t, newFakeClassifier(), nil, "hesitation"),
		source(t, newFakeClassifier(), nil, "tone", "energy"),
	}})

	down, _ := run(t, r, pipeline.Params{}, transcription("I already gave you my order number."))

	wantPerceptions(t, down, reading(hesitation, tone, energy))
}

func TestTextSourcesAreAskedAtTheSameTime(t *testing.T) {
	first, second := newFakeClassifier(), newFakeClassifier()
	first.other, second.other = second, first
	r := newReader(t, perception.Config{Sources: []perception.Source{
		source(t, first, nil, "tone"), source(t, second, nil, "hesitation"),
	}})

	down, _ := run(t, r, pipeline.Params{}, transcription("Hello."))

	wantPerceptions(t, down, reading(tone, hesitation))
}

func TestASourceReadsTheWordsWhileAnotherHearsThem(t *testing.T) {
	words, voice := newFakeClassifier(), newFakeAudioClassifier()
	r := newReader(t, perception.Config{Sources: []perception.Source{
		source(t, words, nil, "tone"), source(t, voice, nil, "hesitation"),
	}})

	down, _ := run(t, r, pipeline.Params{},
		append([]any{assistantTurn("What's your order number?"), ms(50)}, speech("I already gave it to you.")...)...)

	want := []any{map[string]any{"assistant": "What's your order number?", "user": "I already gave it to you."}}
	if got := words.states(); !reflect.DeepEqual(got, want) {
		t.Errorf("states = %v, want %v", got, want)
	}
	wantClips(t, voice, clipOf(pcm(1, 2)))
	wantPerceptions(t, down, reading(tone, hesitation))
}

func TestAReadingFollowsTheSourcesOrderNotTheOrderTheyReadIn(t *testing.T) {
	voice := newFakeAudioClassifier()
	voice.delays = []time.Duration{ms(200)}
	r := newReader(t, perception.Config{Sources: []perception.Source{
		source(t, voice, nil, "hesitation"), source(t, newFakeClassifier(), nil, "tone"),
	}})

	down, _ := run(t, r, pipeline.Params{}, speech("I already told you.")...)

	wantPerceptions(t, down, reading(hesitation, tone))
}

func TestAHeldTranscriptionWaitsForItsWordsToBeRead(t *testing.T) {
	words := newFakeClassifier()
	words.delay = ms(200)
	r := newReader(t, perception.Config{Sources: []perception.Source{
		source(t, words, nil, "tone"), source(t, newFakeAudioClassifier(), nil, "hesitation"),
	}})

	down, _ := run(t, r, pipeline.Params{}, speech("I already told you.")...)

	wantPerceptions(t, down, reading(tone, hesitation))
}

func TestEachTranscriptionOfASpeechGetsItsOwnWordsReadingAndTheClips(t *testing.T) {
	words := newFakeClassifier()
	r := newReader(t, perception.Config{Sources: []perception.Source{
		source(t, words, nil, "tone"), source(t, newFakeAudioClassifier(), nil, "hesitation"),
	}})

	down, _ := run(t, r, pipeline.Params{},
		vadStarted(), audioFrame(1, 2),
		transcription("I already told you."),
		transcription("Twice."),
		vadStopped())

	// Each read runs on a goroutine of its own, so the two may ask in either
	// order.
	want := []any{map[string]any{"user": "I already told you."}, map[string]any{"user": "Twice."}}
	got := words.states()
	slices.SortFunc(got, func(a, b any) int { return strings.Compare(fmt.Sprint(a), fmt.Sprint(b)) })
	if !reflect.DeepEqual(got, want) {
		t.Errorf("states = %v, want %v", got, want)
	}
	wantPerceptions(t, down, reading(tone, hesitation), reading(tone, hesitation))
}

func TestALateSourceLeavesOnlyItsSignalsOut(t *testing.T) {
	late := newFakeClassifier()
	late.delay = time.Second
	r := newReader(t, perception.Config{Sources: []perception.Source{
		source(t, late, secs(0.05), "tone"), source(t, newFakeClassifier(), nil, "hesitation"),
	}})

	down, up := run(t, r, pipeline.Params{}, transcription("Hello."))

	wantPerceptions(t, down, reading(hesitation))
	if len(errorsIn(up)) != 0 {
		t.Errorf("errors = %v, want none", errorsIn(up))
	}
}

func TestAFailedSourceIsReportedAndLeavesOnlyItsSignalsOut(t *testing.T) {
	failing := newFakeClassifier()
	failing.err = errDown
	r := newReader(t, perception.Config{Sources: []perception.Source{
		source(t, failing, nil, "tone"), source(t, newFakeClassifier(), nil, "hesitation"),
	}})

	down, up := run(t, r, pipeline.Params{}, transcription("Hello."))

	wantPerceptions(t, down, reading(hesitation))
	if len(errorsIn(up)) == 0 {
		t.Error("the failed source was not reported")
	}
}

func TestALateClipReadingLeavesTheWordsReading(t *testing.T) {
	voice := newFakeAudioClassifier()
	voice.delays = []time.Duration{time.Second}
	r := newReader(t, perception.Config{Sources: []perception.Source{
		source(t, newFakeClassifier(), nil, "tone"), source(t, voice, secs(0.05), "hesitation"),
	}})

	down, _ := run(t, r, pipeline.Params{}, speech("Hello.")...)

	wantPerceptions(t, down, reading(tone))
}

func TestALateWordsReadingLeavesTheClipsReading(t *testing.T) {
	words := newFakeClassifier()
	words.delay = time.Second
	r := newReader(t, perception.Config{Sources: []perception.Source{
		source(t, words, secs(0.05), "tone"), source(t, newFakeAudioClassifier(), nil, "hesitation"),
	}})

	down, _ := run(t, r, pipeline.Params{}, speech("Hello.")...)

	wantPerceptions(t, down, reading(hesitation))
}

func TestATranscriptionNoSourceReadPassesUnread(t *testing.T) {
	words, voice := newFakeClassifier(), newFakeAudioClassifier()
	words.delay = time.Second
	voice.delays = []time.Duration{time.Second}
	r := newReader(t, perception.Config{Sources: []perception.Source{
		source(t, words, secs(0.05), "tone"), source(t, voice, secs(0.05), "hesitation"),
	}})

	down, _ := run(t, r, pipeline.Params{}, speech("Hello.")...)

	wantPerceptions(t, down, nil)
}

func TestEachAudioSourceHearsAClipOfTheLengthItsClassifierTakes(t *testing.T) {
	whole, short, long := newFakeAudioClassifier(), newFakeAudioClassifier(), newFakeAudioClassifier()
	whole.min = ms(500)
	short.min, short.max = ms(500), time.Second
	long.min = 3 * time.Second
	r := newReader(t, perception.Config{Sources: []perception.Source{
		source(t, whole, nil, "tone"), source(t, short, nil, "hesitation"), source(t, long, nil, "energy"),
	}})

	run(t, r, pipeline.Params{},
		vadStarted(), audioFrame(1, 1), vadStopped(),
		vadStarted(), audioFrame(2, 1), audioFrame(3, 1), vadStopped(),
		transcription("I already told you."))

	// The latest speech, its last second, and the latest with the speech
	// before it.
	if clips := whole.heard(); len(clips) == 0 || !reflect.DeepEqual(clips[len(clips)-1], clipOf(pcm(2, 1), pcm(3, 1))) {
		t.Errorf("whole heard %s", describeClips(clips))
	}
	if clips := short.heard(); len(clips) == 0 || !reflect.DeepEqual(clips[len(clips)-1], clipOf(pcm(3, 1))) {
		t.Errorf("short heard %s", describeClips(clips))
	}
	wantClips(t, long, clipOf(pcm(1, 1), pcm(2, 1), pcm(3, 1)))
}

func TestEachSourcesReadingIsBroadcastAsItsOwn(t *testing.T) {
	r := newReader(t, perception.Config{Sources: []perception.Source{
		source(t, newFakeClassifier(), nil, "tone"), source(t, newFakeAudioClassifier(), nil, "hesitation"),
	}})

	down, up := run(t, r, pipeline.Params{}, speech("Hello.")...)

	want := [][]expressive.PerceptionSignal{{hesitation}, {tone}}
	for dir, got := range map[string][][]expressive.PerceptionSignal{"down": readings(down), "up": readings(up)} {
		slices.SortFunc(got, func(a, b []expressive.PerceptionSignal) int { return strings.Compare(a[0].Name, b[0].Name) })
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%s readings = %s, want %s", dir, formatAll(got), formatAll(want))
		}
	}
}

func TestEveryClassifiersMetricsArePushed(t *testing.T) {
	words, voice := newFakeClassifier(), newFakeAudioClassifier()
	r := newReader(t, perception.Config{Sources: []perception.Source{
		source(t, words, nil, "tone"), source(t, voice, nil, "hesitation"),
	}})

	down, _ := run(t, r, pipeline.Params{EnableMetrics: true}, append(speech("Hello."), ms(200))...)

	measured := map[string]bool{}
	for _, f := range of[*frames.MetricsFrame](down) {
		for _, d := range f.Data {
			if p, ok := d.(frames.ProcessingMetricsData); ok {
				measured[p.Processor] = true
			}
		}
	}
	if !measured[words.Name()] || !measured[voice.Name()] {
		t.Errorf("measured %v, want %s and %s", measured, words.Name(), voice.Name())
	}
}

func TestAReaderThatHearsAddsItsLongestWaitToTheSTTLatency(t *testing.T) {
	for _, tc := range []struct {
		words, want time.Duration
	}{{ms(100), ms(1000)}, {ms(900), ms(1200)}} {
		t.Run(tc.words.String(), func(t *testing.T) {
			words := tc.words
			r := newReader(t, perception.Config{Sources: []perception.Source{
				source(t, newFakeClassifier(), &words, "tone"),
				source(t, newFakeAudioClassifier(), secs(0.2), "hesitation"),
				source(t, newFakeAudioClassifier(), secs(0.7), "energy"),
			}})

			down, _ := run(t, r, pipeline.Params{}, frames.NewSTTMetadataFrame(ms(300)))

			if got := sttLatencies(down); !slices.Equal(got, []time.Duration{tc.want}) {
				t.Errorf("latencies = %v, want [%v]", got, tc.want)
			}
		})
	}
}

func TestASourceThatCantBeSetUpLeavesTheReaderUnusable(t *testing.T) {
	words, voice := newFakeClassifier(), newFakeAudioClassifier()
	voice.setupErr = errNoKey
	r := newReader(t, perception.Config{Sources: []perception.Source{
		source(t, words, nil, "tone"), source(t, voice, nil, "hesitation"),
	}})

	down, up := run(t, r, pipeline.Params{}, speech("Hello.")...)

	if r.Usable() {
		t.Error("the reader is still usable")
	}
	if len(errorsIn(up)) == 0 {
		t.Error("the failed setup was not reported")
	}
	if len(of[*frames.PerceptionMetadataFrame](down)) != 0 {
		t.Error("signals were declared")
	}
	wantPerceptions(t, down, nil)
	if len(words.states()) != 0 {
		t.Errorf("asked %v, want nothing", words.states())
	}
}

func TestTwoSourcesCantReportASignalOfTheSameName(t *testing.T) {
	_, err := perception.New(perception.Config{Sources: []perception.Source{
		source(t, newFakeClassifier(), nil, "tone", "hesitation"),
		source(t, newFakeAudioClassifier(), nil, "hesitation"),
	}})

	if !errors.Is(err, perception.ErrInvalidConfig) || !strings.Contains(err.Error(), "hesitation") {
		t.Errorf("err = %v", err)
	}
}

func TestAClassifierGoesInOneSource(t *testing.T) {
	c := newFakeClassifier()

	_, err := perception.New(perception.Config{Sources: []perception.Source{
		source(t, c, nil, "tone"), source(t, c, nil, "energy"),
	}})

	if !errors.Is(err, perception.ErrInvalidConfig) {
		t.Errorf("err = %v", err)
	}
}

func TestAReaderNeedsASource(t *testing.T) {
	if _, err := perception.New(perception.Config{}); !errors.Is(err, perception.ErrInvalidConfig) {
		t.Errorf("err = %v", err)
	}
}

func TestASourceTakesItsClassifiersDefaultTimeout(t *testing.T) {
	if got := source(t, newFakeClassifier(), nil, "tone").Timeout(); got != ms(500) {
		t.Errorf("text timeout = %v, want 500ms", got)
	}
	if got := source(t, newFakeAudioClassifier(), nil, "tone").Timeout(); got != 2*time.Second {
		t.Errorf("audio timeout = %v, want 2s", got)
	}
	if got := source(t, newFakeAudioClassifier(), secs(0.3), "tone").Timeout(); got != ms(300) {
		t.Errorf("timeout given = %v, want 300ms", got)
	}
}

func TestASourcesNameThatCantBeWrittenInAReadingIsRefused(t *testing.T) {
	q := question("energy")
	q.Name = "speech rate"

	_, err := perception.NewSource(perception.SourceConfig{
		Classifier: newFakeClassifier(), Questions: []perception.NamedQuestion{q},
	})

	if !errors.Is(err, perception.ErrInvalidConfig) {
		t.Errorf("err = %v", err)
	}
}

func sttSignals() expressive.PerceptionCapabilities {
	return expressive.PerceptionCapabilities{Signals: []expressive.PerceptionTerm{
		{Name: "speech_rate", Description: "Words per second."},
	}}
}

func TestAReadingIsAddedToTheOneTheTranscriptionCarries(t *testing.T) {
	r := newReader(t, perception.Config{Sources: []perception.Source{source(t, newFakeClassifier(), nil, "hesitation")}})

	down, _ := run(t, r, pipeline.Params{}, readByTheSTT(transcription("Hello.")))

	wantPerceptions(t, down, reading(speechRate, hesitation))
}

func TestASignalReadBeforeTheReaderTooKeepsTheHigherOfEachLabel(t *testing.T) {
	f := transcription("Hello.")
	f.Perception = []expressive.PerceptionSignal{labeled("tone", label("calm", 0.9), label("frustrated", 0.1))}
	r := newReader(t, perception.Config{Sources: []perception.Source{source(t, newFakeClassifier(), nil, "tone")}})

	down, _ := run(t, r, pipeline.Params{}, f)

	wantPerceptions(t, down,
		reading(labeled("tone", label("calm", 0.9), label("frustrated", 0.7), label("confused", 0.2))))
}

func TestATranscriptionTheReaderDoesntReadKeepsTheReadingItCarries(t *testing.T) {
	late := newFakeClassifier()
	late.delay = time.Second
	r := newReader(t, perception.Config{Sources: []perception.Source{source(t, late, secs(0.05), "hesitation")}})

	down, _ := run(t, r, pipeline.Params{}, readByTheSTT(transcription("Hello.")))

	wantPerceptions(t, down, reading(speechRate))
}

func TestAHeldTranscriptionGetsTheHeardReadingAddedToItsOwn(t *testing.T) {
	send := speech("Hello.")
	send[len(send)-1] = readByTheSTT(transcription("Hello."))
	r := newReader(t, perception.Config{Sources: []perception.Source{
		source(t, newFakeAudioClassifier(), nil, "hesitation"),
	}})

	down, _ := run(t, r, pipeline.Params{}, send...)

	wantPerceptions(t, down, reading(speechRate, hesitation))
}

func TestProvisionalReadingsAndReadingsMadeBeforeTheReaderPassOn(t *testing.T) {
	interim := frames.NewInterimTranscriptionFrame("Hel", "", "now")
	interim.Perception = []expressive.PerceptionSignal{speechRate}
	eager := frames.NewEagerTranscriptionFrame("Hello.", "", "now")
	eager.Perception = []expressive.PerceptionSignal{speechRate}
	r := newReader(t, perception.Config{Sources: []perception.Source{source(t, newFakeClassifier(), nil, "hesitation")}})

	down, _ := run(t, r, pipeline.Params{},
		frames.NewPerceptionFrame([]expressive.PerceptionSignal{speechRate}), interim, eager)

	if got, want := readings(down), [][]expressive.PerceptionSignal{{speechRate}}; !reflect.DeepEqual(got, want) {
		t.Errorf("readings = %s, want %s", formatAll(got), formatAll(want))
	}
	var provisional [][]expressive.PerceptionSignal
	for _, f := range down {
		switch f := f.(type) {
		case *frames.InterimTranscriptionFrame:
			provisional = append(provisional, f.Perception)
		case *frames.EagerTranscriptionFrame:
			provisional = append(provisional, f.Perception)
		}
	}
	if want := [][]expressive.PerceptionSignal{{speechRate}, {speechRate}}; !reflect.DeepEqual(provisional, want) {
		t.Errorf("provisional = %s, want %s", formatAll(provisional), formatAll(want))
	}
}

func TestTheSignalsDeclaredBeforeTheReaderAreDeclaredWithItsOwn(t *testing.T) {
	r := newReader(t, perception.Config{Sources: []perception.Source{source(t, newFakeClassifier(), nil, "hesitation")}})

	down, _ := run(t, r, pipeline.Params{}, frames.NewPerceptionMetadataFrame("stt", sttSignals()))

	want := [][]string{{"hesitation"}, {"speech_rate", "hesitation"}}
	if got := declared(down); !reflect.DeepEqual(got, want) {
		t.Errorf("declared %v, want %v", got, want)
	}
	for _, f := range of[*frames.PerceptionMetadataFrame](down) {
		if f.ServiceName != r.Name() {
			t.Errorf("declared by %q, want %q", f.ServiceName, r.Name())
		}
	}
}

func TestAnUnusableReaderDeclaresTheSignalsDeclaredBeforeIt(t *testing.T) {
	c := newFakeClassifier()
	c.setupErr = errNoModel
	r := newReader(t, perception.Config{Sources: []perception.Source{source(t, c, nil, "tone")}})

	down, _ := run(t, r, pipeline.Params{},
		frames.NewPerceptionMetadataFrame("stt", sttSignals()), readByTheSTT(transcription("Hello.")))

	if got, want := declared(down), [][]string{{"speech_rate"}}; !reflect.DeepEqual(got, want) {
		t.Errorf("declared %v, want %v", got, want)
	}
	wantPerceptions(t, down, reading(speechRate))
}

func TestAReplacingReaderReplacesTheReadingTheTranscriptionCarries(t *testing.T) {
	r := newReader(t, perception.Config{
		Sources:           []perception.Source{source(t, newFakeClassifier(), nil, "hesitation")},
		ReplacePerception: true,
	})

	down, _ := run(t, r, pipeline.Params{},
		frames.NewPerceptionMetadataFrame("stt", sttSignals()), readByTheSTT(transcription("Hello.")))

	// A turn it doesn't read keeps the STT's reading, so both are still
	// declared.
	want := [][]string{{"hesitation"}, {"speech_rate", "hesitation"}}
	if got := declared(down); !reflect.DeepEqual(got, want) {
		t.Errorf("declared %v, want %v", got, want)
	}
	wantPerceptions(t, down, reading(hesitation))
}

func TestAReplacingReaderKeepsTheReadingOfATurnItDoesntRead(t *testing.T) {
	late := newFakeClassifier()
	late.delay = time.Second
	r := newReader(t, perception.Config{
		Sources:           []perception.Source{source(t, late, secs(0.05), "hesitation")},
		ReplacePerception: true,
	})

	down, _ := run(t, r, pipeline.Params{}, readByTheSTT(transcription("Hello.")))

	wantPerceptions(t, down, reading(speechRate))
}

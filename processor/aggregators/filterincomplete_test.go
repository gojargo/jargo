package aggregators_test

import (
	"context"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/turn"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/utils/events"
)

// speechTimeout is how long the speech-timeout stop strategy these tests run
// waits after the user stops.
const speechTimeout = 100 * time.Millisecond

func filterIncomplete(cfg llm.UserTurnCompletionConfig) turns.UserTurnStrategies {
	return turns.FilterIncompleteUserTurnStrategies([]turns.StopStrategy{
		turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{UserSpeechTimeout: speechTimeout}),
	}, cfg)
}

// pause stands in for the time between two frames a test sends.
type pause time.Duration

// sendAndCollect queues the frames into a pipeline around ps, flushing on a
// flush and sleeping on a pause, and returns every frame that reached its end.
// A flush after the user starts speaking lets the interruption it raises go by,
// since that drops the frames queued behind it.
func sendAndCollect(t *testing.T, ps []processor.Processor, send ...any) []frames.Frame {
	t.Helper()
	var mu sync.Mutex
	var got []frames.Frame
	task := pipeline.NewWorker(pipeline.New(ps...), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		got = append(got, f)
		mu.Unlock()
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	flushPipeline(t, task)
	for _, s := range send {
		switch v := s.(type) {
		case flush:
			flushPipeline(t, task)
		case pause:
			time.Sleep(time.Duration(v))
		case frames.Frame:
			task.QueueFrame(v)
		}
	}
	time.Sleep(300 * time.Millisecond)
	task.StopWhenDone()
	<-runDone
	mu.Lock()
	defer mu.Unlock()
	return append([]frames.Frame(nil), got...)
}

func vadStart() frames.Frame { return frames.NewVADUserStartedSpeakingFrame(0.2, time.Now()) }
func vadStop() frames.Frame  { return frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now()) }

func transcriptOf(text string) frames.Frame { return frames.NewTranscriptionFrame(text, "", "now") }

// firstIndex is where the first frame of a kind is, or -1.
func firstIndex[T frames.Frame](fs []frames.Frame) int {
	for i, f := range fs {
		if _, ok := f.(T); ok {
			return i
		}
	}
	return -1
}

// resumeAfterTheAnswerStarted runs a turn whose inference starts, then has the
// user speak again.
func resumeAfterTheAnswerStarted(t *testing.T, strategies turns.UserTurnStrategies) []frames.Frame {
	t.Helper()
	return resumeAfterTheAnswerStartedThen(t, turns.Config{Strategies: strategies})
}

// resumeAfterTheAnswerStartedThen is resumeAfterTheAnswerStarted under cfg,
// sending after once the user spoke again.
func resumeAfterTheAnswerStartedThen(t *testing.T, cfg turns.Config, after ...any) []frames.Frame {
	t.Helper()
	pair := aggregators.New(frames.NewLLMContext(""), aggregators.WithTurns(cfg))
	send := append([]any{
		vadStart(),
		flush{},
		transcriptOf("Hi"), pause(50 * time.Millisecond), vadStop(),
		pause(speechTimeout + 300*time.Millisecond), vadStart(),
		flush{},
		pause(50 * time.Millisecond),
	}, after...)
	return sendAndCollect(t, []processor.Processor{pair.User()}, send...)
}

// contextFramesIn are the inferences that went downstream.
func contextFramesIn(fs []frames.Frame) []*frames.LLMContextFrame {
	var out []*frames.LLMContextFrame
	for _, f := range fs {
		if cf, ok := f.(*frames.LLMContextFrame); ok {
			out = append(out, cf)
		}
	}
	return out
}

// The voided answer is recovered like an empty user turn that interrupted the
// bot.
func TestSpeechWithoutWordsRecoversTheVoidedAnswer(t *testing.T) {
	down := resumeAfterTheAnswerStartedThen(t,
		turns.Config{Strategies: filterIncomplete(llm.UserTurnCompletionConfig{})},
		vadStop(), pause(speechTimeout+300*time.Millisecond))

	answers := contextFramesIn(down)
	if len(answers) != 2 {
		t.Fatalf("inferences = %d, want the turn's and the recovery", len(answers))
	}
	msgs := answers[1].Context.Messages()
	last := msgs[len(msgs)-1]
	if last.Role != frames.RoleDeveloper || last.Text != turns.DefaultEmptyUserTurnInterruptedPrompt {
		t.Fatalf("last message = %+v, want the interrupted empty-turn prompt", last)
	}
}

func TestSpeechWithoutWordsLeavesTheVoidedAnswerWithoutRecovery(t *testing.T) {
	off := ""
	down := resumeAfterTheAnswerStartedThen(t, turns.Config{
		Strategies:    filterIncomplete(llm.UserTurnCompletionConfig{}),
		EmptyUserTurn: &turns.EmptyUserTurnConfig{InterruptedPrompt: &off},
	}, vadStop(), pause(speechTimeout+300*time.Millisecond))

	if n := len(contextFramesIn(down)); n != 1 {
		t.Fatalf("inferences = %d, want only the turn's", n)
	}
}

// The session ending in the turn that voided the answer does not recover it.
func TestSessionEndDoesNotRecoverTheVoidedAnswer(t *testing.T) {
	down := resumeAfterTheAnswerStarted(t, filterIncomplete(llm.UserTurnCompletionConfig{}))

	answers := contextFramesIn(down)
	if len(answers) == 0 {
		t.Fatal("the turn's answer never started")
	}
	for _, m := range answers[0].Context.Messages() {
		if m.Text == turns.DefaultEmptyUserTurnInterruptedPrompt {
			t.Fatal("the session ending recovered the voided answer")
		}
	}
}

func TestSpeechWithoutWordsDoesNotRunAnAnswerItDidNotInterrupt(t *testing.T) {
	strategies := filterIncomplete(llm.UserTurnCompletionConfig{})
	quiet := turns.NewVADStart()
	quiet.EnableInterruptions = false
	strategies.Start = []turns.StartStrategy{quiet}

	down := resumeAfterTheAnswerStartedThen(t, turns.Config{Strategies: strategies},
		vadStop(), pause(speechTimeout+300*time.Millisecond))

	if n := len(contextFramesIn(down)); n != 1 {
		t.Fatalf("inferences = %d, want only the turn's", n)
	}
}

// The user speaking again after the answer started voids it.
func TestNewSpeechInterruptsTheTurnsAnswer(t *testing.T) {
	down := resumeAfterTheAnswerStarted(t, filterIncomplete(llm.UserTurnCompletionConfig{}))

	answer := firstIndex[*frames.LLMContextFrame](down)
	if answer < 0 {
		t.Fatal("the turn's answer never started")
	}
	if firstIndex[*frames.InterruptionFrame](down[answer:]) < 0 {
		t.Fatal("speaking again did not interrupt the answer")
	}
}

func TestATurnThatDoesNotInterruptOnlyCancelsItsAnswer(t *testing.T) {
	strategies := filterIncomplete(llm.UserTurnCompletionConfig{})
	quiet := turns.NewVADStart()
	quiet.EnableInterruptions = false
	strategies.Start = []turns.StartStrategy{quiet}

	down := resumeAfterTheAnswerStarted(t, strategies)

	answer := firstIndex[*frames.LLMContextFrame](down)
	if answer < 0 {
		t.Fatal("the turn's answer never started")
	}
	if firstIndex[*frames.InterruptionFrame](down) >= 0 {
		t.Fatal("a turn that does not interrupt the bot interrupted it")
	}
	if firstIndex[*frames.UserTurnResponseCancelFrame](down[answer:]) < 0 {
		t.Fatal("speaking again did not cancel the answer")
	}
}

func TestSpeakingAgainBeforeTheAnswerStartedInterruptsNothing(t *testing.T) {
	pair := aggregators.New(frames.NewLLMContext(""), aggregators.WithTurns(turns.Config{
		Strategies: filterIncomplete(llm.UserTurnCompletionConfig{}),
	}))
	down := sendAndCollect(t, []processor.Processor{pair.User()},
		vadStart(), flush{}, transcriptOf("Hi"), vadStop(), vadStart(), pause(50*time.Millisecond))

	resumed := -1
	for i, f := range down {
		if _, ok := f.(*frames.VADUserStartedSpeakingFrame); ok {
			resumed = i
		}
	}
	if resumed < 0 {
		t.Fatal("the speech never went downstream")
	}
	if firstIndex[*frames.InterruptionFrame](down[resumed:]) >= 0 {
		t.Fatal("speaking again before the answer started interrupted something")
	}
}

// Two inferences before the turn ends keep the whole transcript.
//
// When nothing has decided on the first inference and the user keeps speaking,
// the stop strategies' next decisions run a second inference. Both the report
// of the turn ending and the conversation reflect the full utterance, not just
// the segment from the last inference.
func TestMultipleInferencesInOneTurnPreserveTheAggregation(t *testing.T) {
	convo := frames.NewLLMContext("")
	pair := aggregators.New(convo, aggregators.WithTurns(turns.Config{
		Strategies: turns.UserTurnStrategies{Stop: []turns.StopStrategy{
			turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{UserSpeechTimeout: speechTimeout}),
			newCompletionStop(),
		}},
	}))
	var mu sync.Mutex
	inferences := 0
	var stopped *aggregators.UserTurnStopped
	pair.User().Events().Add(aggregators.EventUserTurnInferenceTriggered, func(context.Context, any, ...any) {
		mu.Lock()
		inferences++
		mu.Unlock()
	})
	events.On(pair.User().Events(), aggregators.EventUserTurnStopped,
		func(_ context.Context, s aggregators.UserTurnStopped) {
			mu.Lock()
			stopped = &s
			mu.Unlock()
		})

	down := sendAndCollect(t, []processor.Processor{pair.User()},
		vadStart(), flush{}, transcriptOf("I'm thinking"), pause(50*time.Millisecond), vadStop(),
		pause(speechTimeout+300*time.Millisecond),
		// The first inference ran here. Nothing decided on it, so the turn stays
		// open and the user keeps talking.
		vadStart(), flush{}, transcriptOf("about pizza"), pause(50*time.Millisecond), vadStop(),
		pause(speechTimeout+300*time.Millisecond),
		// The second inference ran here, and the decision on it ends the turn.
		frames.NewUserTurnInferenceCompletedFrame(), pause(50*time.Millisecond))

	mu.Lock()
	defer mu.Unlock()
	if inferences != 2 {
		t.Fatalf("inferences = %d, want 2", inferences)
	}
	// The decision on the inference ended the turn, not the session ending.
	if firstIndex[*frames.UserStoppedSpeakingFrame](down) < 0 {
		t.Fatal("the decision on the inference did not end the turn")
	}
	if stopped == nil || stopped.Content != "I'm thinking about pizza" {
		t.Fatalf("turn stopped = %+v, want the full transcript", stopped)
	}
	var said []string
	for _, m := range convo.Messages() {
		if m.Role == frames.RoleUser {
			said = append(said, m.Text)
		}
	}
	if !slices.Equal(said, []string{"I'm thinking", "about pizza"}) {
		t.Fatalf("user messages = %v, want one per inference", said)
	}
}

// scriptedMarkerLLM answers each inference with the next scripted reply, read
// through the turn-completion markers.
type scriptedMarkerLLM struct {
	mu         sync.Mutex
	replies    []string
	delays     []time.Duration
	inferences int
}

func (g *scriptedMarkerLLM) Generate(ctx context.Context, _ *frames.LLMContext, emit llm.Emit) error {
	g.mu.Lock()
	g.inferences++
	if len(g.replies) == 0 {
		g.mu.Unlock()
		return nil
	}
	reply := g.replies[0]
	g.replies = g.replies[1:]
	var delay time.Duration
	if len(g.delays) > 0 {
		delay, g.delays = g.delays[0], g.delays[1:]
	}
	g.mu.Unlock()
	select {
	case <-time.After(delay):
	case <-ctx.Done():
		return nil
	}
	return emit(reply)
}

func (g *scriptedMarkerLLM) count() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.inferences
}

// outputRecorder records the spoken text and the marker kinds that reach the
// output.
type outputRecorder struct {
	*processor.Base
	mu     sync.Mutex
	spoken []string
	kinds  []string
}

func newOutputRecorder() *outputRecorder {
	r := &outputRecorder{}
	r.Base = processor.New("OutputRecorder", r)
	return r
}

func (r *outputRecorder) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := r.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	r.mu.Lock()
	switch fr := f.(type) {
	case *frames.LLMTextFrame:
		r.spoken = append(r.spoken, fr.Text)
	case *frames.LLMMarkerResponseFrame:
		r.kinds = append(r.kinds, fr.Kind)
	}
	r.mu.Unlock()
	return r.PushFrame(ctx, f, dir)
}

func (r *outputRecorder) spokenText() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.spoken...)
}

// filteringRun is one conversation through aggregators, a marker-reading LLM
// and the output.
type filteringRun struct {
	recorder *outputRecorder
	context  *frames.LLMContext
	model    *scriptedMarkerLLM
}

func runFiltering(t *testing.T, cfg turns.Config, model *scriptedMarkerLLM, send ...any) filteringRun {
	t.Helper()
	convo := frames.NewLLMContext("")
	pair := aggregators.New(convo, aggregators.WithTurns(cfg))
	recorder := newOutputRecorder()
	service := llm.New("ScriptedMarkerLLM", model)
	sendAndCollect(t, []processor.Processor{pair.User(), service, recorder, pair.Assistant()}, send...)
	return filteringRun{recorder: recorder, context: convo, model: model}
}

// utterance is the user saying text and going quiet for the speech timeout.
func utterance(text string) []any {
	return []any{
		vadStart(),
		flush{},
		transcriptOf(text), pause(50 * time.Millisecond), vadStop(),
		pause(speechTimeout + 400*time.Millisecond),
	}
}

// openings are each message's role and first character.
func openings(c *frames.LLMContext) []string {
	var out []string
	for _, m := range c.Messages() {
		first := ""
		if r := []rune(m.Text); len(r) > 0 {
			first = string(r[0])
		}
		out = append(out, string(m.Role)+":"+first)
	}
	return out
}

func filteringConfig(stopMode turns.StrategyMode, cfg llm.UserTurnCompletionConfig) turns.Config {
	return turns.Config{
		Strategies:       filterIncomplete(cfg),
		StrategiesParams: turns.UserTurnStrategiesParams{StopMode: stopMode},
	}
}

func TestFilteringACompleteTurnIsAnswered(t *testing.T) {
	run := runFiltering(t, filteringConfig(turns.StrategyModeDefault, llm.UserTurnCompletionConfig{}),
		&scriptedMarkerLLM{replies: []string{"● Sure, Boston it is."}},
		utterance("Book a flight to Boston")...)

	if got := run.recorder.spokenText(); !slices.Equal(got, []string{"Sure, Boston it is."}) {
		t.Fatalf("spoken = %v, want the answer", got)
	}
	msgs := run.context.Messages()
	if len(msgs) != 2 || msgs[0].Role != frames.RoleUser || msgs[1].Role != frames.RoleAssistant ||
		!strings.HasPrefix(msgs[1].Text, "●") {
		t.Fatalf("messages = %v, want the user and the marked answer", msgs)
	}
}

func TestFilteringAnIncompleteTurnWaitsForTheRest(t *testing.T) {
	run := runFiltering(t, filteringConfig(turns.StrategyModeDefault, llm.UserTurnCompletionConfig{}),
		&scriptedMarkerLLM{replies: []string{"◐", "● Sure, Boston it is."}},
		append(utterance("Book a flight"), utterance("to Boston")...)...)

	if got := run.recorder.spokenText(); !slices.Equal(got, []string{"Sure, Boston it is."}) {
		t.Fatalf("spoken = %v, want only the answer to the whole turn", got)
	}
	want := []string{"user:B", "assistant:◐", "user:t", "assistant:●"}
	if got := openings(run.context); !slices.Equal(got, want) {
		t.Fatalf("messages = %v, want %v", got, want)
	}
}

func TestFilteringAQuietUserIsRepromptedInEitherStopMode(t *testing.T) {
	for _, mode := range []turns.StrategyMode{turns.StrategyModeAll, turns.StrategyModeFirst} {
		run := runFiltering(t,
			filteringConfig(mode, llm.UserTurnCompletionConfig{IncompleteShortTimeout: 200 * time.Millisecond}),
			&scriptedMarkerLLM{replies: []string{"◐", "● Are you still there?"}},
			append(utterance("Book a flight"), pause(500*time.Millisecond))...)

		if got := run.recorder.spokenText(); !slices.Equal(got, []string{"Are you still there?"}) {
			t.Fatalf("mode %v: spoken = %v, want the re-prompt's answer", mode, got)
		}
	}
}

func TestFilteringWordsAfterTheInferenceAreAnsweredWithoutAReprompt(t *testing.T) {
	run := runFiltering(t,
		filteringConfig(turns.StrategyModeDefault,
			llm.UserTurnCompletionConfig{IncompleteShortTimeout: 300 * time.Millisecond}),
		&scriptedMarkerLLM{replies: []string{"◐", "● Sure, Boston it is.", "● Are you still there?"}},
		vadStart(), flush{}, transcriptOf("Book a flight"), pause(50*time.Millisecond), vadStop(),
		pause(speechTimeout+50*time.Millisecond), transcriptOf("to Boston"), pause(600*time.Millisecond))

	if got := run.recorder.spokenText(); !slices.Equal(got, []string{"Sure, Boston it is."}) {
		t.Fatalf("spoken = %v, want the answer to the late words only", got)
	}
}

func TestFilteringEveryResponseReportsItsMarker(t *testing.T) {
	run := runFiltering(t, filteringConfig(turns.StrategyModeDefault, llm.UserTurnCompletionConfig{}),
		&scriptedMarkerLLM{replies: []string{"◐", "● Sure, Boston it is."}},
		append(utterance("Book a flight"), utterance("to Boston")...)...)

	run.recorder.mu.Lock()
	defer run.recorder.mu.Unlock()
	if !slices.Equal(run.recorder.kinds, []string{"short", "complete"}) {
		t.Fatalf("marker kinds = %v, want [short complete]", run.recorder.kinds)
	}
}

// The marker strategy decides on an eager answer while the turn commits, in
// either order.

func runEagerFiltering(t *testing.T, model *scriptedMarkerLLM) filteringRun {
	t.Helper()
	// Until the service recommends its strategies, the extending list joins
	// the default ones, whose end-of-turn model needs the ONNX runtime.
	if _, err := turn.NewSmartTurnV3(); err != nil {
		t.Skipf("the default stop strategies need the ONNX runtime: %v", err)
	}
	recommendation := frames.NewSTTMetadataFrame(0)
	recommendation.ServiceName = "eager-stt"
	recommendation.UserTurnStrategies = turns.EagerStrategies(turns.EagerStrategiesConfig{})
	text := "Book a flight to Boston"
	return runFiltering(t, turns.Config{
		Strategies: turns.FilterIncompleteUserTurnStrategies(nil, llm.UserTurnCompletionConfig{}),
		StrategiesParams: turns.UserTurnStrategiesParams{
			StartMergeMode: turns.MergeModeExtend, StopMergeMode: turns.MergeModeExtend,
		},
	}, model,
		recommendation, pause(50*time.Millisecond),
		frames.NewProposedUserStartedSpeakingFrame(), flush{},
		frames.NewEagerTranscriptionFrame(text, "user", "now"), pause(100*time.Millisecond),
		frames.NewTranscriptionFrame(text, "user", "now"), pause(50*time.Millisecond),
		frames.NewProposedUserStoppedSpeakingFrame(), pause(800*time.Millisecond))
}

func assertAnsweredOnce(t *testing.T, run filteringRun) {
	t.Helper()
	if got := run.recorder.spokenText(); !slices.Equal(got, []string{"Sure, Boston it is."}) {
		t.Fatalf("spoken = %v, want the answer once", got)
	}
	if got := openings(run.context); !slices.Equal(got, []string{"user:B", "assistant:●"}) {
		t.Fatalf("messages = %v, want the turn and its marked answer", got)
	}
}

func TestFilteringACompleteEagerAnswerIsSpokenOnceTheTurnCommits(t *testing.T) {
	assertAnsweredOnce(t, runEagerFiltering(t, &scriptedMarkerLLM{replies: []string{"● Sure, Boston it is."}}))
}

func TestFilteringASlowCompleteEagerAnswerIsGeneratedOnce(t *testing.T) {
	run := runEagerFiltering(t, &scriptedMarkerLLM{
		replies: []string{"● Sure, Boston it is."},
		delays:  []time.Duration{800 * time.Millisecond},
	})

	assertAnsweredOnce(t, run)
	if n := run.model.count(); n != 1 {
		t.Fatalf("inferences = %d, want 1", n)
	}
}

func TestFilteringAnIncompleteEagerAnswerIsDroppedAndTheCommittedTurnAnswered(t *testing.T) {
	assertAnsweredOnce(t, runEagerFiltering(t, &scriptedMarkerLLM{replies: []string{"◐", "● Sure, Boston it is."}}))
}

func TestFilteringAnEagerAnswerMarkedIncompleteAfterTheCommitIsAnsweredAgain(t *testing.T) {
	assertAnsweredOnce(t, runEagerFiltering(t, &scriptedMarkerLLM{
		replies: []string{"◐", "● Sure, Boston it is."},
		delays:  []time.Duration{450 * time.Millisecond, 0},
	}))
}

// The marker arrives as the turn commits, so the speculation is withdrawn while
// the eager strategy confirms it.
func TestFilteringAnEagerAnswerMarkedIncompleteAsTheTurnCommitsIsAnsweredAgain(t *testing.T) {
	assertAnsweredOnce(t, runEagerFiltering(t, &scriptedMarkerLLM{
		replies: []string{"◐", "● Sure, Boston it is."},
		delays:  []time.Duration{300 * time.Millisecond, 0},
	}))
}

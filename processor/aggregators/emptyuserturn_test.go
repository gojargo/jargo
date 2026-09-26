package aggregators_test

// Tests for the user turns that end with no transcript. Ported from upstream's
// empty user turn tests.

import (
	"context"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/utils/events"
)

const (
	emptyTurnStopTimeout = 200 * time.Millisecond
	emptyTurnSpeech      = 100 * time.Millisecond

	interruptedPrompt = "interrupted, nothing recognized"
	idlePrompt        = "idle, nothing recognized"
)

// contextFrameCounter counts the inferences the user aggregator requests.
type contextFrameCounter struct {
	*processor.Base
	n atomic.Int32
}

func newContextFrameCounter() *contextFrameCounter {
	c := &contextFrameCounter{}
	c.Base = processor.New("ContextFrameCounter", c)
	return c
}

func (c *contextFrameCounter) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := c.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.LLMContextFrame); ok && dir == processor.Downstream {
		c.n.Add(1)
	}
	return c.PushFrame(ctx, f, dir)
}

// emptyTurnConfig is the turn taking these tests run with: VAD or a transcript
// start a turn, silence ends it, and the watchdog closes a turn left open.
func emptyTurnConfig(empty *turns.EmptyUserTurnConfig, idleTimeout time.Duration) turns.Config {
	return turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart(), turns.NewTranscriptionStart(turns.TranscriptionStartConfig{})},
			Stop: []turns.StopStrategy{turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{
				UserSpeechTimeout: emptyTurnSpeech,
			})},
		},
		StopTimeout:   emptyTurnStopTimeout,
		IdleTimeout:   idleTimeout,
		EmptyUserTurn: empty,
	}
}

func withPrompts(idle string) *turns.EmptyUserTurnConfig {
	p := interruptedPrompt
	return &turns.EmptyUserTurnConfig{InterruptedPrompt: &p, IdlePrompt: idle}
}

func disabled() *turns.EmptyUserTurnConfig {
	none := ""
	return &turns.EmptyUserTurnConfig{InterruptedPrompt: &none}
}

// script is what is sent: frames, and pauses between them.
type script []any

func emptyTurn() script {
	return script{
		frames.NewVADUserStartedSpeakingFrame(0, time.Now()),
		frames.NewVADUserStoppedSpeakingFrame(0, time.Now()),
		emptyTurnStopTimeout + 200*time.Millisecond,
	}
}

func transcribedTurn(text string) script {
	return script{
		frames.NewVADUserStartedSpeakingFrame(0, time.Now()),
		frames.NewTranscriptionFrame(text, "", "now"),
		frames.NewVADUserStoppedSpeakingFrame(0, time.Now()),
		emptyTurnSpeech + 100*time.Millisecond,
	}
}

// botIdle is the bot having finished speaking and waiting for the user.
func botIdle() script {
	return script{frames.NewBotStoppedSpeakingFrame(), 50 * time.Millisecond}
}

// botSpeaking is a requested bot response that is still being spoken.
func botSpeaking() script {
	return script{
		frames.NewLLMRunFrame(),
		50 * time.Millisecond,
		frames.NewLLMFullResponseStartFrame(),
		frames.NewBotStartedSpeakingFrame(),
		frames.NewTTSTextFrame("Where would", frames.AggregationWord),
		50 * time.Millisecond,
	}
}

func join(parts ...script) script {
	var out script
	for _, p := range parts {
		out = append(out, p...)
	}
	return out
}

// runScript sends the script through the processors and stops the pipeline once
// it has played out.
func runScript(t *testing.T, s script, ps ...processor.Processor) {
	t.Helper()
	task := pipeline.NewWorker(pipeline.New(ps...), pipeline.WorkerConfig{IdleTimeout: -1})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	time.Sleep(50 * time.Millisecond)
	for _, step := range s {
		switch v := step.(type) {
		case time.Duration:
			time.Sleep(v)
		case frames.Frame:
			task.QueueFrame(v)
		}
	}
	time.Sleep(50 * time.Millisecond)
	task.StopWhenDone()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not end")
	}
}

// runEmptyTurn plays the script through a user/assistant pair and returns the
// conversation and the inferences requested.
func runEmptyTurn(t *testing.T, s script, cfg turns.Config) (*frames.LLMContext, int) {
	t.Helper()
	convo := frames.NewLLMContext("")
	pair := aggregators.New(convo, aggregators.WithTurns(cfg))
	counter := newContextFrameCounter()
	runScript(t, s, pair.User(), counter, pair.Assistant())
	return convo, int(counter.n.Load())
}

func wantDeveloper(t *testing.T, convo *frames.LLMContext, want ...string) {
	t.Helper()
	if got := developerMessages(convo); !slices.Equal(got, want) {
		t.Fatalf("developer messages = %q, want %q", got, want)
	}
}

func wantInferences(t *testing.T, got, want int) {
	t.Helper()
	if got != want {
		t.Fatalf("requested %d inferences, want %d", got, want)
	}
}

func TestEmptyTurnInterruptedWhileSpeaking(t *testing.T) {
	convo, inferences := runEmptyTurn(t, join(botSpeaking(), emptyTurn()), emptyTurnConfig(withPrompts(""), 0))
	wantDeveloper(t, convo, interruptedPrompt)
	wantInferences(t, inferences, 2)
	// The recovery comes after what the user heard of the interrupted response.
	msgs := convo.Messages()
	if len(msgs) < 2 || msgs[len(msgs)-2].Role != frames.RoleAssistant || msgs[len(msgs)-2].Text != "Where would" {
		t.Fatalf("messages = %+v, want the heard part of the response before the recovery", msgs)
	}
}

func TestEmptyTurnInterruptedBeforeResponseStarted(t *testing.T) {
	// The previous turn's response had not started when the user spoke: nothing
	// was heard, and the interruption canceled it.
	convo, inferences := runEmptyTurn(t, join(botIdle(), transcribedTurn("Tell me a story."), emptyTurn()),
		emptyTurnConfig(withPrompts(""), 0))
	wantDeveloper(t, convo, interruptedPrompt)
	wantInferences(t, inferences, 2)
}

func TestEmptyTurnBeforeBotSpoke(t *testing.T) {
	// Until the bot first finishes speaking it is not waiting for the user: its
	// greeting may still be on the way.
	convo, _ := runEmptyTurn(t, emptyTurn(), emptyTurnConfig(withPrompts(""), 0))
	wantDeveloper(t, convo, interruptedPrompt)
}

func TestEmptyTurnIdleIgnoredByDefault(t *testing.T) {
	convo, inferences := runEmptyTurn(t, join(botIdle(), emptyTurn()), emptyTurnConfig(withPrompts(""), 0))
	wantDeveloper(t, convo)
	wantInferences(t, inferences, 0)
}

func TestEmptyTurnIdlePrompt(t *testing.T) {
	convo, inferences := runEmptyTurn(t, join(botIdle(), emptyTurn()), emptyTurnConfig(withPrompts(idlePrompt), 0))
	wantDeveloper(t, convo, idlePrompt)
	wantInferences(t, inferences, 1)
}

func TestEmptyTurnIdleAfterResponseFinished(t *testing.T) {
	convo, _ := runEmptyTurn(t,
		join(botSpeaking(), script{frames.NewLLMFullResponseEndFrame()}, botIdle(), emptyTurn()),
		emptyTurnConfig(withPrompts(idlePrompt), 0))
	wantDeveloper(t, convo, idlePrompt)
}

func TestEmptyTurnEnabledByDefault(t *testing.T) {
	convo, inferences := runEmptyTurn(t, join(botSpeaking(), emptyTurn()), emptyTurnConfig(nil, 0))
	wantDeveloper(t, convo, turns.DefaultEmptyUserTurnInterruptedPrompt)
	wantInferences(t, inferences, 2)
}

func TestEmptyTurnDisabled(t *testing.T) {
	convo, inferences := runEmptyTurn(t, join(botSpeaking(), emptyTurn()), emptyTurnConfig(disabled(), 0))
	wantDeveloper(t, convo)
	wantInferences(t, inferences, 1)
}

func TestEmptyTurnTranscribedTurnNotRecovered(t *testing.T) {
	convo, inferences := runEmptyTurn(t, join(botSpeaking(), transcribedTurn("Hello!")),
		emptyTurnConfig(withPrompts(""), 0))
	wantDeveloper(t, convo)
	wantInferences(t, inferences, 2)
}

func TestEmptyTurnConsecutiveRecoveriesAreBounded(t *testing.T) {
	// The second empty turn interrupts the recovery's own pending response, but
	// only one recovery in a row is allowed. A transcribed turn resets the count.
	convo, _ := runEmptyTurn(t,
		join(botSpeaking(), emptyTurn(), emptyTurn(), transcribedTurn("Sorry, what?"), emptyTurn()),
		emptyTurnConfig(withPrompts(""), 0))
	wantDeveloper(t, convo, interruptedPrompt, interruptedPrompt)
}

func TestEmptyTurnFunctionCallInProgressNotRecovered(t *testing.T) {
	// The function call's result will run the LLM on its own.
	convo, inferences := runEmptyTurn(t, join(
		script{frames.NewLLMRunFrame()},
		botIdle(),
		script{
			frames.NewFunctionCallsStartedFrame([]frames.ToolCall{{ID: "1", Name: "get_weather"}}),
			50 * time.Millisecond,
		},
		emptyTurn(),
	), emptyTurnConfig(withPrompts(""), 0))
	wantDeveloper(t, convo)
	wantInferences(t, inferences, 1)
}

func TestEmptyTurnWithoutThePair(t *testing.T) {
	convo := frames.NewLLMContext("")
	pair := aggregators.New(convo, aggregators.WithTurns(emptyTurnConfig(withPrompts(""), 0)))
	runScript(t, join(botSpeaking(), emptyTurn()), pair.User())
	wantDeveloper(t, convo, interruptedPrompt)
}

// idleFired plays the script through a pair with a short idle timeout and
// reports whether the user was found idle.
func idleFired(t *testing.T, s script, empty *turns.EmptyUserTurnConfig) bool {
	t.Helper()
	convo := frames.NewLLMContext("")
	pair := aggregators.New(convo, aggregators.WithTurns(emptyTurnConfig(empty, 200*time.Millisecond)))
	var (
		mu   sync.Mutex
		idle bool
	)
	events.OnSignal(pair.User().Events(), aggregators.EventUserTurnIdle, func(context.Context) {
		mu.Lock()
		idle = true
		mu.Unlock()
	})
	runScript(t, append(s, 400*time.Millisecond), pair.User(), pair.Assistant())
	mu.Lock()
	defer mu.Unlock()
	return idle
}

func TestEmptyTurnIdleTimerRearmedAfterAnEmptyTurn(t *testing.T) {
	// The output transport stops the bot when the user interrupts it, while the
	// user turn is in progress, so that does not start the timer.
	s := join(botSpeaking(), script{
		frames.NewVADUserStartedSpeakingFrame(0, time.Now()),
		frames.NewBotStoppedSpeakingFrame(),
		frames.NewVADUserStoppedSpeakingFrame(0, time.Now()),
		emptyTurnStopTimeout + 200*time.Millisecond,
	})
	if !idleFired(t, s, disabled()) {
		t.Fatal("the idle timer was not restarted after an empty turn")
	}
}

func TestEmptyTurnIdleTimerNotRearmedAfterRecovery(t *testing.T) {
	// The recovery response is on its way, so the user is not idle yet.
	if idleFired(t, join(botSpeaking(), emptyTurn()), nil) {
		t.Fatal("the user was found idle while the recovery response was on its way")
	}
}

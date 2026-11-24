package eval

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/utils/events"
)

// speechTimeoutStop is a speech timeout standing in for the default turn
// model, which hears an end at every sentence the bot speaks.
func speechTimeoutStop(timeout time.Duration) []turns.StopStrategy {
	return []turns.StopStrategy{turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{UserSpeechTimeout: timeout})}
}

// A persona's LLM also judges whether the bot finished; a scripted run's turns
// are the turn model's.
func TestBotTurnStrategies(t *testing.T) {
	audioClient := func() *Client {
		scenario := &ScriptScenario{Name: "t"}
		scenario.BotAudio = true
		return clientFor(t, scenario, "ws://localhost:0", SessionParams{})
	}
	t.Run("a simulation waits for the persona to judge the bot done", func(t *testing.T) {
		// The turn model hears a finished turn at the end of each sentence the
		// bot's TTS speaks; the turn ends once the persona's LLM marks it
		// complete.
		c := audioClient()
		c.botTurnStop = speechTimeoutStop(100 * time.Millisecond)
		strategies, set := c.botTurnStrategies(true)
		if !set || len(strategies.Stop) != 2 {
			t.Fatalf("strategies = %+v (set %v), want the stop strategy and the completion gate", strategies, set)
		}
		if _, ok := strategies.Stop[1].(*turns.LLMTurnCompletionStop); !ok {
			t.Fatalf("last stop strategy = %T, want the LLM's turn completion", strategies.Stop[1])
		}
	})
	t.Run("a scripted run keeps the default strategies", func(t *testing.T) {
		if strategies, set := audioClient().botTurnStrategies(false); set {
			t.Fatalf("strategies = %+v, want the defaults", strategies)
		}
	})
}

// closedTurn is a turn the bot-turn aggregator ended: the strategy that decided
// it and what the bot said in it.
type closedTurn struct {
	strategy turns.StopStrategy
	text     string
}

// step is one frame of a closing turn, queued after a pause.
type step struct {
	after time.Duration
	frame frames.Frame
}

// closingTurns is each turn the aggregator ends, after the persona hung up,
// when the steps are played to it and the sink behind it.
func closingTurns(t *testing.T, steps func(t *testing.T) []step) []closedTurn {
	t.Helper()
	persona := stubPersona()
	persona.HangUp()
	// A speech timeout stands in for the turn model, hearing an end at every
	// sentence.
	strategies := turns.FilterIncompleteUserTurnStrategies(speechTimeoutStop(100*time.Millisecond),
		llm.UserTurnCompletionConfig{})
	pair := aggregators.New(persona.Context(), aggregators.WithTurns(turns.Config{Strategies: strategies}))
	var (
		mu     sync.Mutex
		closed []closedTurn
	)
	events.On(pair.User().Events(), aggregators.EventUserTurnStopped,
		func(_ context.Context, stopped aggregators.UserTurnStopped) {
			mu.Lock()
			closed = append(closed, closedTurn{strategy: stopped.Strategy, text: stopped.Content})
			mu.Unlock()
		})
	ended := func() []closedTurn {
		mu.Lock()
		defer mu.Unlock()
		return append([]closedTurn(nil), closed...)
	}

	sink := newBotFrameSink(newStream(true), persona, false)
	noRTVI := false
	worker := pipeline.NewWorker(pipeline.New(pair.User(), sink), pipeline.WorkerConfig{EnableRTVI: &noRTVI})
	started := make(chan struct{})
	var once sync.Once
	events.On(worker.Events(), pipeline.EventPipelineStarted,
		func(context.Context, *frames.StartFrame) { once.Do(func() { close(started) }) })
	done := make(chan error, 1)
	go func() { done <- worker.Run(context.WithoutCancel(t.Context())) }()
	select {
	case <-started:
	case <-time.After(10 * time.Second):
		t.Fatal("the pipeline did not start")
	}

	for _, s := range steps(t) {
		time.Sleep(s.after)
		worker.QueueFrame(s.frame)
	}
	// Well short of the controller's 5 s stop timeout.
	deadline := time.Now().Add(2 * time.Second)
	for len(ended()) == 0 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	turnsBeforeTheEnd := ended()
	worker.StopWhenDone()
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("the pipeline did not finish")
	}
	if got := ended(); len(got) != len(turnsBeforeTheEnd) {
		t.Fatalf("the pipeline ending closed a turn: %+v", got)
	}
	return turnsBeforeTheEnd
}

// The bot's turn after the persona hangs up ends once the bot stopped speaking.
func TestClosingBotTurn(t *testing.T) {
	// Frames of different kinds travel on separate paths, so each waits a
	// moment for the one before it to land.
	const beat = 30 * time.Millisecond
	said := func(text string) []step {
		return []step{
			{beat, frames.NewVADUserStartedSpeakingFrame(0, time.Now())},
			{beat, frames.NewTranscriptionFrame(text, "bot", "t")},
			{beat, frames.NewVADUserStoppedSpeakingFrame(0, time.Now())},
		}
	}
	t.Run("the closing turn ends when the bot stopped speaking", func(t *testing.T) {
		closed := closingTurns(t, func(t *testing.T) []step {
			t.Helper()
			steps := []step{{0, messageFrame(t, rtvi.TypeBotStartedSpeaking, nil)}}
			steps = append(steps, said("Goodbye!")...)
			return append(steps, step{beat, messageFrame(t, rtvi.TypeBotStoppedSpeaking, nil)})
		})
		// The completion gate ended the turn, not the stop timeout or the
		// pipeline ending.
		if len(closed) != 1 {
			t.Fatalf("turns = %+v, want one", closed)
		}
		if _, ok := closed[0].strategy.(*turns.LLMTurnCompletionStop); !ok {
			t.Fatalf("ended by %T, want the completion gate", closed[0].strategy)
		}
	})
	t.Run("a pause between sentences does not end the closing turn", func(t *testing.T) {
		closed := closingTurns(t, func(t *testing.T) []step {
			t.Helper()
			steps := []step{{0, messageFrame(t, rtvi.TypeBotStartedSpeaking, nil)}}
			steps = append(steps, said("Your table for two is booked.")...)
			// Past the speech timeout, with the bot still speaking.
			second := said("See you Friday at seven.")
			second[0].after = 300 * time.Millisecond
			steps = append(steps, second...)
			return append(steps, step{300 * time.Millisecond, messageFrame(t, rtvi.TypeBotStoppedSpeaking, nil)})
		})
		if len(closed) != 1 {
			t.Fatalf("turns = %+v, want one", closed)
		}
		if _, ok := closed[0].strategy.(*turns.LLMTurnCompletionStop); !ok {
			t.Fatalf("ended by %T, want the completion gate", closed[0].strategy)
		}
		if want := "Your table for two is booked. See you Friday at seven."; closed[0].text != want {
			t.Fatalf("turn = %q, want %q", closed[0].text, want)
		}
	})
}

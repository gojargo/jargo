package turns

// Unit tests for the individual start, stop and mute strategies.
//
// These are in-package because a strategy only signals through the unexported
// strategyEnv the controller attaches; driving them from turns_test would mean
// standing up a whole pipeline to observe a switch statement. The pipeline-level
// behavior is covered by turns_test.go — this file covers the decision logic.

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/turn"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// spy records everything a strategy signals through its environment.
type spy struct {
	mu sync.Mutex

	started    []UserTurnStartedParams
	stopped    []UserTurnStoppedParams
	resets     int
	inferences int
	pushed     []processor.Direction
	broadcast  []frames.Frame
}

func newSpy() *spy { return &spy{} }

func (s *spy) env() strategyEnv {
	return strategyEnv{
		mu:                 &s.mu,
		started:            func(p UserTurnStartedParams) { s.started = append(s.started, p) },
		stopped:            func(p UserTurnStoppedParams) { s.stopped = append(s.stopped, p) },
		resetAggregation:   func() { s.resets++ },
		inferenceTriggered: func() { s.inferences++ },
		push:               func(_ frames.Frame, d processor.Direction) { s.pushed = append(s.pushed, d) },
		broadcast:          func(build func() frames.Frame) { s.broadcast = append(s.broadcast, build()) },
	}
}

// send drives one frame through a start strategy the way the controller does:
// under the shared mutex.
func (s *spy) send(str StartStrategy, f frames.Frame) ProcessFrameResult {
	s.mu.Lock()
	defer s.mu.Unlock()
	return str.Process(f)
}

// sendStop is send for stop strategies.
func (s *spy) sendStop(str StopStrategy, f frames.Frame) ProcessFrameResult {
	s.mu.Lock()
	defer s.mu.Unlock()
	return str.Process(f)
}

// starts returns how many turn-starts have been signaled so far.
func (s *spy) starts() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.started)
}

// stops returns how many turn-ends have been signaled so far.
func (s *spy) stops() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.stopped)
}

// attachStart wires a start strategy to a fresh spy.
func attachStart(str StartStrategy) *spy {
	s := newSpy()
	str.attach(s.env())
	return s
}

// attachStop wires a stop strategy to a fresh spy.
func attachStop(str StopStrategy) *spy {
	s := newSpy()
	str.attach(s.env())
	return s
}

func transcript(text string) *frames.TranscriptionFrame {
	return frames.NewTranscriptionFrame(text, "user", "")
}

func interim(text string) *frames.InterimTranscriptionFrame {
	return frames.NewInterimTranscriptionFrame(text, "user", "")
}

// eventually polls cond until it holds or the deadline passes. Strategies arm
// real timers, so the wake-phrase and external-stop timeouts have to be awaited
// rather than asserted synchronously.
func eventually(t *testing.T, cond func() bool, within time.Duration, msg string) {
	t.Helper()
	deadline := time.Now().Add(within)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("condition never held: %s", msg)
}

func TestVADStart(t *testing.T) {
	s := NewVADStart()
	spy := attachStart(s)

	if got := spy.send(s, transcript("hello")); got != Continue {
		t.Errorf("Process(transcript) = %v, want Continue", got)
	}
	if spy.starts() != 0 {
		t.Error("a transcript must not open a VAD-gated turn")
	}

	if got := spy.send(s, frames.NewVADUserStartedSpeakingFrame(0)); got != Stop {
		t.Errorf("Process(vad start) = %v, want Stop", got)
	}
	if spy.starts() != 1 {
		t.Fatalf("starts = %d, want 1", spy.starts())
	}
	got := spy.started[0]
	if !got.EnableInterruptions || !got.EnableUserSpeakingFrames {
		t.Errorf("params = %+v, want both enabled", got)
	}
}

func TestTranscriptionStart(t *testing.T) {
	tests := []struct {
		name       string
		useInterim *bool
		frame      frames.Frame
		wantStart  bool
	}{
		{"final transcript", nil, transcript("hi"), true},
		{"interim by default", nil, interim("hi"), true},
		{"interim disabled", new(false), interim("hi"), false},
		{"interim enabled", new(true), interim("hi"), true},
		{"unrelated frame", nil, frames.NewBotSpeakingFrame(), false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewTranscriptionStart(TranscriptionStartConfig{UseInterim: tt.useInterim})
			spy := attachStart(s)

			got := spy.send(s, tt.frame)
			if started := spy.starts() == 1; started != tt.wantStart {
				t.Errorf("started = %v, want %v", started, tt.wantStart)
			}
			want := Continue
			if tt.wantStart {
				want = Stop
			}
			if got != want {
				t.Errorf("Process = %v, want %v", got, want)
			}
		})
	}
}

// TestMinWordsStartThreshold covers the barge-in gate: one word is enough while
// the bot is silent, but MinWords are required to interrupt it.
func TestMinWordsStartThreshold(t *testing.T) {
	tests := []struct {
		name        string
		botSpeaking bool
		text        string
		wantStart   bool
	}{
		{"bot silent, one word", false, "hey", true},
		{"bot silent, empty", false, "", false},
		{"bot speaking, too few", true, "uh huh", false},
		{"bot speaking, enough", true, "wait stop for a second", true},
		{"bot speaking, exactly enough", true, "one two three", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewMinWordsStart(MinWordsStartConfig{MinWords: 3})
			spy := attachStart(s)

			if tt.botSpeaking {
				spy.send(s, frames.NewBotStartedSpeakingFrame())
			}
			got := spy.send(s, transcript(tt.text))

			if started := spy.starts() == 1; started != tt.wantStart {
				t.Errorf("started = %v, want %v", started, tt.wantStart)
			}
			if tt.wantStart {
				if got != Stop {
					t.Errorf("Process = %v, want Stop", got)
				}
			} else {
				if got != Continue {
					t.Errorf("Process = %v, want Continue", got)
				}
				// Speech below the bar is dropped so it can't pollute the context.
				if spy.resets != 1 {
					t.Errorf("resetAggregation calls = %d, want 1", spy.resets)
				}
			}
		})
	}
}

// TestMinWordsStartBotStopsSpeaking checks the threshold drops back to one word
// once the bot falls silent.
func TestMinWordsStartBotStopsSpeaking(t *testing.T) {
	s := NewMinWordsStart(MinWordsStartConfig{MinWords: 5})
	spy := attachStart(s)

	spy.send(s, frames.NewBotStartedSpeakingFrame())
	spy.send(s, transcript("no"))
	if spy.starts() != 0 {
		t.Fatal("one word must not interrupt a speaking bot with MinWords=5")
	}

	spy.send(s, frames.NewBotStoppedSpeakingFrame())
	spy.send(s, transcript("no"))
	if spy.starts() != 1 {
		t.Errorf("starts = %d, want 1 once the bot is silent", spy.starts())
	}
}

// TestMinWordsStartTurnStartedClearsBotSpeaking checks the threshold drops to one
// word as soon as a turn opens, rather than waiting for the bot-stopped frame the
// interruption will produce.
func TestMinWordsStartTurnStartedClearsBotSpeaking(t *testing.T) {
	s := NewMinWordsStart(MinWordsStartConfig{MinWords: 3})
	spy := attachStart(s)

	spy.send(s, frames.NewBotStartedSpeakingFrame())
	spy.send(s, transcript("three words now"))
	if spy.starts() != 1 {
		t.Fatalf("starts = %d, want 1 once three words interrupt the bot", spy.starts())
	}
	s.TurnStarted()

	// No BotStoppedSpeakingFrame yet — the flag must already be clear.
	spy.send(s, transcript("yes"))
	if spy.starts() != 2 {
		t.Errorf("starts = %d, want 2: one word should suffice after a turn opened", spy.starts())
	}
}

func TestMinWordsStartInterim(t *testing.T) {
	t.Run("counted by default", func(t *testing.T) {
		s := NewMinWordsStart(MinWordsStartConfig{MinWords: 2})
		spy := attachStart(s)
		if got := spy.send(s, interim("hello")); got != Stop {
			t.Errorf("Process(interim) = %v, want Stop", got)
		}
		if spy.starts() != 1 {
			t.Error("interim transcript should open the turn by default")
		}
	})

	t.Run("ignored when disabled", func(t *testing.T) {
		s := NewMinWordsStart(MinWordsStartConfig{MinWords: 2, UseInterim: new(false)})
		spy := attachStart(s)
		if got := spy.send(s, interim("hello")); got != Continue {
			t.Errorf("Process(interim) = %v, want Continue", got)
		}
		if spy.starts() != 0 || spy.resets != 0 {
			t.Error("a disabled interim must be ignored entirely, not reset")
		}
	})
}

// TestWakePhraseStartGates is the core wake-phrase contract: asleep it blocks the
// rest of the start chain, the phrase wakes it, and it stays awake afterwards.
func TestWakePhraseStartGates(t *testing.T) {
	s := NewWakePhraseStart(WakePhraseStartConfig{Phrases: []string{"hey jargo"}, Timeout: time.Minute})
	spy := attachStart(s)
	defer s.Cleanup()

	if got := spy.send(s, transcript("what is the weather")); got != Stop {
		t.Errorf("asleep: Process = %v, want Stop (blocks the chain)", got)
	}
	if spy.starts() != 0 {
		t.Fatal("non-wake speech must not open a turn")
	}
	if spy.resets != 1 {
		t.Errorf("pre-wake speech should be dropped: resets = %d, want 1", spy.resets)
	}

	if got := spy.send(s, transcript("ok Hey   Jargo please")); got != Stop {
		t.Errorf("waking: Process = %v, want Stop", got)
	}
	if spy.starts() != 1 {
		t.Fatalf("the wake phrase should open a turn: starts = %d", spy.starts())
	}

	// Awake, the strategy defers to the rest of the chain.
	if got := spy.send(s, transcript("and the traffic")); got != Continue {
		t.Errorf("awake: Process = %v, want Continue", got)
	}
	if got := spy.send(s, frames.NewUserSpeakingFrame()); got != Continue {
		t.Errorf("awake: Process(user speaking) = %v, want Continue", got)
	}
	if spy.starts() != 1 {
		t.Errorf("awake speech must not re-trigger a start: starts = %d", spy.starts())
	}
}

// TestWakePhraseStartMatchesAcrossTranscripts checks the accumulator: a phrase
// split over two transcripts still matches.
func TestWakePhraseStartMatchesAcrossTranscripts(t *testing.T) {
	s := NewWakePhraseStart(WakePhraseStartConfig{Phrases: []string{"hey jargo"}, Timeout: time.Minute})
	spy := attachStart(s)
	defer s.Cleanup()

	spy.send(s, transcript("hey"))
	if spy.starts() != 0 {
		t.Fatal("half a phrase must not wake the session")
	}
	spy.send(s, transcript("jargo are you there"))
	if spy.starts() != 1 {
		t.Errorf("phrase split across transcripts should match: starts = %d", spy.starts())
	}
}

// TestWakePhraseStartAccumCap checks the accumulator stays bounded and that a
// phrase still matches once older speech has been trimmed away.
func TestWakePhraseStartAccumCap(t *testing.T) {
	s := NewWakePhraseStart(WakePhraseStartConfig{Phrases: []string{"hey jargo"}, Timeout: time.Minute})
	spy := attachStart(s)
	defer s.Cleanup()

	for range 20 {
		spy.send(s, transcript("filler words that will overflow the accumulator"))
	}
	if len(s.accum) > wakeAccumLimit {
		t.Errorf("accum grew to %d bytes, want <= %d", len(s.accum), wakeAccumLimit)
	}
	if spy.starts() != 0 {
		t.Fatal("filler must not wake the session")
	}
	spy.send(s, transcript("hey jargo"))
	if spy.starts() != 1 {
		t.Error("wake phrase should still match after the accumulator was trimmed")
	}
}

func TestWakePhraseStartSingleActivation(t *testing.T) {
	s := NewWakePhraseStart(WakePhraseStartConfig{
		Phrases:          []string{"hey jargo"},
		Timeout:          time.Minute,
		SingleActivation: true,
	})
	spy := attachStart(s)
	defer s.Cleanup()

	spy.send(s, transcript("hey jargo"))
	if spy.starts() != 1 {
		t.Fatal("wake phrase should open the first turn")
	}

	// The controller calls TurnStarted once the turn opens; single-activation
	// puts the session straight back to sleep.
	spy.mu.Lock()
	s.TurnStarted()
	spy.mu.Unlock()

	if got := spy.send(s, transcript("follow up question")); got != Stop {
		t.Errorf("after single activation: Process = %v, want Stop (asleep again)", got)
	}
	if spy.starts() != 1 {
		t.Errorf("single-activation must require the phrase again: starts = %d", spy.starts())
	}
}

// TestWakePhraseStartTimeout checks the inactivity timer puts the session back
// to sleep.
func TestWakePhraseStartTimeout(t *testing.T) {
	s := NewWakePhraseStart(WakePhraseStartConfig{Phrases: []string{"hey jargo"}, Timeout: 20 * time.Millisecond})
	spy := attachStart(s)
	defer s.Cleanup()

	spy.send(s, transcript("hey jargo"))
	if spy.starts() != 1 {
		t.Fatal("wake phrase should wake the session")
	}

	eventually(t, func() bool {
		spy.mu.Lock()
		defer spy.mu.Unlock()
		return !s.awake
	}, 2*time.Second, "session should fall asleep after the inactivity timeout")

	if got := spy.send(s, transcript("still there")); got != Stop {
		t.Errorf("after timeout: Process = %v, want Stop", got)
	}
}

// TestWakePhraseStartDefaultTimeout pins the documented 10s default.
func TestWakePhraseStartDefaultTimeout(t *testing.T) {
	s := NewWakePhraseStart(WakePhraseStartConfig{Phrases: []string{"hey"}})
	if s.timeout != wakePhraseDefaultTimeout {
		t.Errorf("timeout = %v, want %v", s.timeout, wakePhraseDefaultTimeout)
	}
}

func TestCompileWakePatterns(t *testing.T) {
	tests := []struct {
		name     string
		phrases  []string
		text     string
		want     bool
		wantPats int
	}{
		{"case insensitive", []string{"hey jargo"}, "HEY JARGO", true, 1},
		{"flexible whitespace", []string{"hey jargo"}, "hey    jargo", true, 1},
		{"word boundary", []string{"hey jargo"}, "theyjargo", false, 1},
		{"regex metacharacters are literal", []string{"a.b"}, "axb", false, 1},
		{"regex metacharacters match literally", []string{"a.b"}, "a.b", true, 1},
		{"blank phrases are skipped", []string{"", "   "}, "anything", false, 0},
		{"any phrase matches", []string{"hey jargo", "ok jargo"}, "ok jargo", true, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pats := compileWakePatterns(tt.phrases)
			if len(pats) != tt.wantPats {
				t.Fatalf("compiled %d patterns, want %d", len(pats), tt.wantPats)
			}
			s := &WakePhraseStart{patterns: pats}
			if got := s.matches(tt.text); got != tt.want {
				t.Errorf("matches(%q) = %v, want %v", tt.text, got, tt.want)
			}
		})
	}
}

// TestCompileWakePatternsDoesNotMutateInput guards the caller's slice: the
// compiler quotes words in place while building each pattern.
func TestCompileWakePatternsDoesNotMutateInput(t *testing.T) {
	phrases := []string{"a.b c"}
	compileWakePatterns(phrases)
	if phrases[0] != "a.b c" {
		t.Errorf("caller's phrase was mutated to %q", phrases[0])
	}
}

// TestExternalStart covers the relay case: another processor already decided the
// turn, so no interruption or speaking frames are re-emitted.
func TestExternalStart(t *testing.T) {
	s := NewExternalStart()
	spy := attachStart(s)

	if got := spy.send(s, frames.NewVADUserStartedSpeakingFrame(0)); got != Continue {
		t.Errorf("Process(vad) = %v, want Continue", got)
	}
	if spy.starts() != 0 {
		t.Error("ExternalStart must ignore VAD")
	}

	if got := spy.send(s, frames.NewUserStartedSpeakingFrame()); got != Stop {
		t.Errorf("Process(user started) = %v, want Stop", got)
	}
	if spy.starts() != 1 {
		t.Fatalf("starts = %d, want 1", spy.starts())
	}
	if p := spy.started[0]; p.EnableInterruptions || p.EnableUserSpeakingFrames {
		t.Errorf("params = %+v, want both disabled (the external processor emits them)", p)
	}
}

// TestExternalStopDebounce checks the stop is held back for a late transcript
// and then fires once one arrives.
func TestExternalStopDebounce(t *testing.T) {
	s := NewExternalStop(ExternalStopConfig{Timeout: 20 * time.Millisecond})
	spy := attachStop(s)
	defer s.Cleanup()

	spy.sendStop(s, frames.NewUserStartedSpeakingFrame())
	spy.sendStop(s, transcript("book me a table"))
	spy.sendStop(s, frames.NewUserStoppedSpeakingFrame())

	if spy.stops() != 0 {
		t.Fatal("stop must be debounced, not immediate")
	}
	eventually(t, func() bool { return spy.stops() == 1 }, 2*time.Second, "debounced stop should fire")

	spy.mu.Lock()
	defer spy.mu.Unlock()
	if spy.inferences != 1 {
		t.Errorf("inference triggers = %d, want 1", spy.inferences)
	}
	if p := spy.stopped[0]; p.EnableUserSpeakingFrames {
		t.Error("ExternalStop must not re-emit user speaking frames")
	}
}

// TestExternalStopWaitsForTranscript checks the default: no text yet means no
// turn end, and a pending interim keeps the turn open.
func TestExternalStopWaitsForTranscript(t *testing.T) {
	t.Run("no transcript", func(t *testing.T) {
		s := NewExternalStop(ExternalStopConfig{Timeout: 10 * time.Millisecond})
		spy := attachStop(s)
		defer s.Cleanup()

		spy.sendStop(s, frames.NewUserStoppedSpeakingFrame())
		time.Sleep(100 * time.Millisecond)
		if spy.stops() != 0 {
			t.Error("must not end the turn without a transcript")
		}
	})

	t.Run("interim still pending", func(t *testing.T) {
		s := NewExternalStop(ExternalStopConfig{Timeout: 10 * time.Millisecond})
		spy := attachStop(s)
		defer s.Cleanup()

		spy.sendStop(s, transcript("hello"))
		spy.sendStop(s, interim("and also"))
		spy.sendStop(s, frames.NewUserStoppedSpeakingFrame())
		time.Sleep(100 * time.Millisecond)
		if spy.stops() != 0 {
			t.Error("a pending interim means more speech is coming; the turn must stay open")
		}

		// The final transcript clears the interim and lets the next stop through.
		spy.sendStop(s, transcript("and also this"))
		spy.sendStop(s, frames.NewUserStoppedSpeakingFrame())
		eventually(t, func() bool { return spy.stops() == 1 }, 2*time.Second, "stop after the final transcript")
	})

	t.Run("empty transcript does not count", func(t *testing.T) {
		s := NewExternalStop(ExternalStopConfig{Timeout: 10 * time.Millisecond})
		spy := attachStop(s)
		defer s.Cleanup()

		spy.sendStop(s, transcript(""))
		spy.sendStop(s, frames.NewUserStoppedSpeakingFrame())
		time.Sleep(100 * time.Millisecond)
		if spy.stops() != 0 {
			t.Error("an empty transcript is not text")
		}
	})

	t.Run("disabled", func(t *testing.T) {
		s := NewExternalStop(ExternalStopConfig{
			Timeout:           10 * time.Millisecond,
			WaitForTranscript: new(false),
		})
		spy := attachStop(s)
		defer s.Cleanup()

		spy.sendStop(s, frames.NewUserStoppedSpeakingFrame())
		eventually(t, func() bool { return spy.stops() == 1 }, 2*time.Second,
			"with WaitForTranscript off the stop should fire regardless")
	})
}

// TestExternalStopUserResumed checks that speech resuming inside the debounce
// window cancels the pending stop.
func TestExternalStopUserResumed(t *testing.T) {
	s := NewExternalStop(ExternalStopConfig{Timeout: 30 * time.Millisecond})
	spy := attachStop(s)
	defer s.Cleanup()

	spy.sendStop(s, transcript("hold on"))
	spy.sendStop(s, frames.NewUserStoppedSpeakingFrame())
	spy.sendStop(s, frames.NewUserStartedSpeakingFrame()) // resumed before the timer

	time.Sleep(150 * time.Millisecond)
	if spy.stops() != 0 {
		t.Error("resumed speech must cancel the pending stop")
	}
}

// TestExternalStopResetsPerTurn checks TurnStarted/TurnStopped clear the
// transcript evidence so it can't leak into the next turn.
func TestExternalStopResetsPerTurn(t *testing.T) {
	for _, tt := range []struct {
		name  string
		reset func(s *ExternalStop)
	}{
		{"TurnStarted", func(s *ExternalStop) { s.TurnStarted() }},
		{"TurnStopped", func(s *ExternalStop) { s.TurnStopped() }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			s := NewExternalStop(ExternalStopConfig{Timeout: 10 * time.Millisecond})
			spy := attachStop(s)
			defer s.Cleanup()

			spy.sendStop(s, transcript("previous turn"))
			spy.mu.Lock()
			tt.reset(s)
			spy.mu.Unlock()

			spy.sendStop(s, frames.NewUserStoppedSpeakingFrame())
			time.Sleep(100 * time.Millisecond)
			if spy.stops() != 0 {
				t.Error("transcript evidence from the previous turn must not end this one")
			}
		})
	}
}

// TestExternalStopDefaultTimeout pins the documented 500ms default.
func TestExternalStopDefaultTimeout(t *testing.T) {
	s := NewExternalStop(ExternalStopConfig{})
	if s.timeout != 500*time.Millisecond {
		t.Errorf("timeout = %v, want 500ms", s.timeout)
	}
	if !s.waitForTx {
		t.Error("WaitForTranscript should default to true")
	}
}

// TestMuteStrategies drives each strategy over a frame sequence and checks the
// muted state it reports at every step.
func TestMuteStrategies(t *testing.T) {
	botStart := func() frames.Frame { return frames.NewBotStartedSpeakingFrame() }
	botStop := func() frames.Frame { return frames.NewBotStoppedSpeakingFrame() }
	other := func() frames.Frame { return frames.NewUserSpeakingFrame() }

	type step struct {
		frame frames.Frame
		want  bool
	}
	tests := []struct {
		name  string
		build func() MuteStrategy
		steps []step
	}{
		{
			name:  "AlwaysUserMute mutes for every bot turn",
			build: func() MuteStrategy { return NewAlwaysUserMute() },
			steps: []step{
				{other(), false},
				{botStart(), true},
				{other(), true},
				{botStop(), false},
				{botStart(), true}, // and again on the second turn
				{botStop(), false},
			},
		},
		{
			name:  "FirstSpeechUserMute mutes only the first bot turn",
			build: func() MuteStrategy { return NewFirstSpeechUserMute() },
			steps: []step{
				{other(), false}, // pre-speech input is allowed
				{botStart(), true},
				{other(), true},
				{botStop(), false},
				{botStart(), false}, // never again
				{other(), false},
				{botStop(), false},
			},
		},
		{
			name:  "MuteUntilFirstBotComplete mutes from session start",
			build: func() MuteStrategy { return NewMuteUntilFirstBotComplete() },
			steps: []step{
				{other(), true}, // muted before the bot has even spoken
				{botStart(), true},
				{botStop(), false},
				{other(), false},
				{botStart(), false},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := tt.build()
			for i, st := range tt.steps {
				if got := s.ShouldMute(st.frame); got != st.want {
					t.Errorf("step %d (%s): ShouldMute = %v, want %v", i, st.frame.Name(), got, st.want)
				}
			}
		})
	}
}

// TestFunctionCallUserMute checks the strategy stays muted until every in-flight
// call has settled, whether by result or cancellation.
func TestFunctionCallUserMute(t *testing.T) {
	s := NewFunctionCallUserMute()

	if s.ShouldMute(frames.NewUserSpeakingFrame()) {
		t.Error("no calls in flight: want unmuted")
	}

	started := frames.NewFunctionCallsStartedFrame("", []frames.ToolCall{
		{ID: "a", Name: "lookup"},
		{ID: "b", Name: "book"},
	})
	if !s.ShouldMute(started) {
		t.Error("calls in flight: want muted")
	}

	if !s.ShouldMute(frames.NewFunctionCallResultFrame("a", "lookup", "ok", false)) {
		t.Error("one call still in flight: want muted")
	}
	if s.ShouldMute(frames.NewFunctionCallCancelFrame("b", "book")) {
		t.Error("all calls settled: want unmuted")
	}

	// An unknown ID must not unmute anything or corrupt the set.
	s.ShouldMute(started)
	if s.ShouldMute(frames.NewFunctionCallResultFrame("zzz", "other", "ok", false)) != true {
		t.Error("an unrelated result must leave the in-flight calls muted")
	}
}

func TestDefaultStrategyChains(t *testing.T) {
	t.Run("start defaults", func(t *testing.T) {
		got := DefaultStartStrategies()
		if len(got) != 2 {
			t.Fatalf("len = %d, want 2", len(got))
		}
		if _, ok := got[0].(*VADStart); !ok {
			t.Errorf("first = %T, want *VADStart", got[0])
		}
		if _, ok := got[1].(*TranscriptionStart); !ok {
			t.Errorf("second = %T, want *TranscriptionStart", got[1])
		}
	})

	t.Run("stop defaults", func(t *testing.T) {
		got := DefaultStopStrategies()
		if len(got) != 1 {
			t.Fatalf("len = %d, want 1", len(got))
		}
		if _, ok := got[0].(*SpeechTimeoutStop); !ok {
			t.Errorf("first = %T, want *SpeechTimeoutStop", got[0])
		}
	})

	t.Run("fillDefaults only fills empty chains", func(t *testing.T) {
		custom := NewExternalStart()
		s := UserTurnStrategies{Start: []StartStrategy{custom}}
		s.fillDefaults()
		if len(s.Start) != 1 || s.Start[0] != custom {
			t.Error("a provided start chain must be left alone")
		}
		if len(s.Stop) == 0 {
			t.Error("an empty stop chain must be filled")
		}
	})
}

func TestExternalStrategies(t *testing.T) {
	s := ExternalStrategies()
	if len(s.Start) != 1 {
		t.Fatalf("start len = %d, want 1", len(s.Start))
	}
	if _, ok := s.Start[0].(*ExternalStart); !ok {
		t.Errorf("start = %T, want *ExternalStart", s.Start[0])
	}
	if len(s.Stop) != 1 {
		t.Fatalf("stop len = %d, want 1", len(s.Stop))
	}
	if _, ok := s.Stop[0].(*ExternalStop); !ok {
		t.Errorf("stop = %T, want *ExternalStop", s.Stop[0])
	}
}

// TestFilterIncompleteUserTurnStrategies checks the detectors are wrapped as
// deferred (they may only trigger inference) and the LLM decides the turn end.
func TestFilterIncompleteUserTurnStrategies(t *testing.T) {
	t.Run("wraps supplied detectors", func(t *testing.T) {
		got := FilterIncompleteUserTurnStrategies([]StopStrategy{NewExternalStop(ExternalStopConfig{})})
		if len(got.Stop) != 2 {
			t.Fatalf("stop len = %d, want detector + completion", len(got.Stop))
		}
		if _, ok := got.Stop[0].(*deferredStop); !ok {
			t.Errorf("detector = %T, want *deferredStop", got.Stop[0])
		}
		if _, ok := got.Stop[1].(*ExternalCompletionStop); !ok {
			t.Errorf("finalizer = %T, want *ExternalCompletionStop", got.Stop[1])
		}
		if len(got.Start) != 0 {
			t.Error("the start chain should be left to fillDefaults")
		}
	})

	t.Run("empty detectors use the defaults", func(t *testing.T) {
		got := FilterIncompleteUserTurnStrategies(nil)
		if len(got.Stop) != len(DefaultStopStrategies())+1 {
			t.Errorf("stop len = %d, want defaults + completion", len(got.Stop))
		}
	})
}

// TestDeferredStopSuppressesFinalization is the point of Deferred: the inner
// strategy may start inference but must not end the turn itself.
func TestDeferredStopSuppressesFinalization(t *testing.T) {
	inner := NewExternalStop(ExternalStopConfig{Timeout: 10 * time.Millisecond})
	d := Deferred(inner)
	spy := attachStop(d)
	defer d.Cleanup()

	spy.sendStop(d, transcript("done"))
	spy.sendStop(d, frames.NewUserStoppedSpeakingFrame())

	eventually(t, func() bool {
		spy.mu.Lock()
		defer spy.mu.Unlock()
		return spy.inferences == 1
	}, 2*time.Second, "deferred strategy should still trigger inference")

	spy.mu.Lock()
	defer spy.mu.Unlock()
	if len(spy.stopped) != 0 {
		t.Errorf("deferred strategy must not finalize the turn: stops = %d", len(spy.stopped))
	}
}

// TestDeferredStopDelegates checks the per-turn hooks reach the inner strategy.
func TestDeferredStopDelegates(t *testing.T) {
	inner := NewExternalStop(ExternalStopConfig{Timeout: time.Minute})
	d := Deferred(inner)
	spy := attachStop(d)

	spy.sendStop(d, transcript("text"))
	spy.mu.Lock()
	if !inner.haveText {
		spy.mu.Unlock()
		t.Fatal("Process should reach the inner strategy")
	}
	d.TurnStarted()
	if inner.haveText {
		spy.mu.Unlock()
		t.Error("TurnStarted should reach the inner strategy")
	}

	inner.haveText = true
	d.TurnStopped()
	if inner.haveText {
		spy.mu.Unlock()
		t.Error("TurnStopped should reach the inner strategy")
	}
	spy.mu.Unlock()
	d.Cleanup()
}

func TestNewLLMTurnCompletionStop(t *testing.T) {
	if _, ok := NewLLMTurnCompletionStop().(*ExternalCompletionStop); !ok {
		t.Error("NewLLMTurnCompletionStop should build an *ExternalCompletionStop")
	}
}

func TestCompletionInstructions(t *testing.T) {
	if got := CompletionInstructions(UserTurnCompletionConfig{}); got != defaultCompletionInstructions {
		t.Error("an empty config should yield the default marker-protocol instructions")
	}
	const custom = "answer only in haiku"
	if got := CompletionInstructions(UserTurnCompletionConfig{Instructions: custom}); got != custom {
		t.Errorf("CompletionInstructions = %q, want the override", got)
	}
}

func TestDefaultTurnParams(t *testing.T) {
	if got := DefaultStartedParams(); !got.EnableInterruptions || !got.EnableUserSpeakingFrames {
		t.Errorf("DefaultStartedParams = %+v, want both enabled", got)
	}
	if got := DefaultStoppedParams(); !got.EnableUserSpeakingFrames {
		t.Errorf("DefaultStoppedParams = %+v, want speaking frames enabled", got)
	}
}

// TestStrategyBaseNoOps checks the embedded defaults are safe to call, including
// on a strategy that was never attached to a controller.
func TestStrategyBaseNoOps(t *testing.T) {
	t.Run("start base without an env", func(t *testing.T) {
		var b StartStrategyBase
		b.TurnStarted()
		b.TurnStopped()
		b.Cleanup()
		b.TriggerStarted()
		b.TriggerResetAggregation()
	})

	t.Run("stop base without an env", func(t *testing.T) {
		var b StopStrategyBase
		b.TurnStarted()
		b.TurnStopped()
		b.Cleanup()
		b.TriggerStopped()
		b.Push(frames.NewUserSpeakingFrame(), processor.Downstream)
		b.Broadcast(func() frames.Frame { return frames.NewUserSpeakingFrame() })
	})
}

// TestStopStrategyBaseEmits covers the frame-emitting helpers a stop strategy
// uses to reach the pipeline.
func TestStopStrategyBaseEmits(t *testing.T) {
	var b StopStrategyBase
	b.EnableUserSpeakingFrames = true
	spy := newSpy()
	b.attach(spy.env())

	b.Push(frames.NewUserSpeakingFrame(), processor.Upstream)
	b.Broadcast(func() frames.Frame { return frames.NewBotSpeakingFrame() })
	b.TriggerStopped()

	if len(spy.pushed) != 1 || spy.pushed[0] != processor.Upstream {
		t.Errorf("pushed = %v, want one Upstream push", spy.pushed)
	}
	if len(spy.broadcast) != 1 {
		t.Errorf("broadcast = %d frames, want 1", len(spy.broadcast))
	}
	// TriggerStopped is inference-then-finalize, in that order.
	if spy.inferences != 1 || len(spy.stopped) != 1 {
		t.Errorf("inferences = %d, stops = %d, want 1 and 1", spy.inferences, len(spy.stopped))
	}
	if !spy.stopped[0].EnableUserSpeakingFrames {
		t.Error("the base's EnableUserSpeakingFrames should reach the stop params")
	}
}

// TestStrategyEnvAfterCancel checks a canceled timeout never runs its callback.
func TestStrategyEnvAfterCancel(t *testing.T) {
	spy := newSpy()
	env := spy.env()

	fired := false
	spy.mu.Lock()
	cancel := env.after(10*time.Millisecond, func() { fired = true })
	cancel()
	spy.mu.Unlock()

	time.Sleep(100 * time.Millisecond)
	spy.mu.Lock()
	defer spy.mu.Unlock()
	if fired {
		t.Error("a canceled timeout must not run")
	}
}

// TestStrategyEnvAfterRuns checks the callback runs with the shared mutex held,
// which is the contract strategies rely on instead of locking themselves.
func TestStrategyEnvAfterRuns(t *testing.T) {
	spy := newSpy()
	env := spy.env()

	done := make(chan struct{})
	env.after(5*time.Millisecond, func() {
		if spy.mu.TryLock() {
			spy.mu.Unlock()
			t.Error("callback should run with the shared mutex already held")
		}
		close(done)
	})

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("timeout callback never ran")
	}
}

// errEmit is the failure a fakeEmitter reports.
//
//nolint:gochecknoglobals // sentinel error
var errEmit = errors.New("emit failed")

// fakeEmitter records what the idle controller's callback sends into the
// pipeline.
type fakeEmitter struct {
	pushed    []processor.Direction
	broadcast int
	err       error
}

func (e *fakeEmitter) Push(_ context.Context, _ frames.Frame, dir processor.Direction) error {
	e.pushed = append(e.pushed, dir)
	return e.err
}

func (e *fakeEmitter) Broadcast(_ context.Context, _ func() frames.Frame) error {
	e.broadcast++
	return e.err
}

// TestUserIdleControllerEmit covers the frame-sending helpers an idle callback
// uses, including the guard for a controller that was never set up.
func TestUserIdleControllerEmit(t *testing.T) {
	t.Run("without an emitter", func(t *testing.T) {
		c := NewUserIdleController(IdleConfig{})
		if err := c.Push(t.Context(), frames.NewUserSpeakingFrame(), processor.Downstream); err != nil {
			t.Errorf("Push before Setup: %v", err)
		}
		if err := c.Broadcast(t.Context(), func() frames.Frame { return frames.NewUserSpeakingFrame() }); err != nil {
			t.Errorf("Broadcast before Setup: %v", err)
		}
	})

	t.Run("with an emitter", func(t *testing.T) {
		emit := &fakeEmitter{}
		c := NewUserIdleController(IdleConfig{})
		c.Setup(t.Context(), emit)

		if err := c.Push(t.Context(), frames.NewUserSpeakingFrame(), processor.Upstream); err != nil {
			t.Errorf("Push: %v", err)
		}
		if err := c.Broadcast(t.Context(), func() frames.Frame { return frames.NewUserSpeakingFrame() }); err != nil {
			t.Errorf("Broadcast: %v", err)
		}
		if len(emit.pushed) != 1 || emit.pushed[0] != processor.Upstream {
			t.Errorf("pushed = %v, want one Upstream push", emit.pushed)
		}
		if emit.broadcast != 1 {
			t.Errorf("broadcast = %d, want 1", emit.broadcast)
		}
		c.Cleanup()
	})

	t.Run("emitter errors propagate", func(t *testing.T) {
		emit := &fakeEmitter{err: errEmit}
		c := NewUserIdleController(IdleConfig{})
		c.Setup(t.Context(), emit)

		if err := c.Push(t.Context(), frames.NewUserSpeakingFrame(), processor.Downstream); !errors.Is(err, errEmit) {
			t.Errorf("Push error = %v, want errEmit", err)
		}
		build := func() frames.Frame { return frames.NewUserSpeakingFrame() }
		if err := c.Broadcast(t.Context(), build); !errors.Is(err, errEmit) {
			t.Errorf("Broadcast error = %v, want errEmit", err)
		}
	})
}

// slowAnalyzer takes a fixed time to judge the end of a turn, standing in for a
// model whose inference is not free.
type slowAnalyzer struct{ took time.Duration }

func (slowAnalyzer) SetSampleRate(int)                            {}
func (slowAnalyzer) AppendAudio([]byte, bool) turn.EndOfTurnState { return turn.Incomplete }
func (a slowAnalyzer) AnalyzeEndOfTurn() (turn.EndOfTurnState, float64, error) {
	time.Sleep(a.took)
	return turn.Complete, 0.9, nil
}
func (slowAnalyzer) Clear()                     {}
func (slowAnalyzer) Close() error               { return nil }
func (slowAnalyzer) UpdateVADStartSecs(float64) {}

// The STT safety net is anchored to the end of the user's speech, so the time
// the analyzer spends judging the turn comes out of the budget rather than
// extending it.
func TestTurnAnalyzerSTTTimeoutAnchoredToSpeechEnd(t *testing.T) {
	const (
		inference = 150 * time.Millisecond
		stopSecs  = 0.2
		sttP99    = 600 * time.Millisecond
	)

	s := NewTurnAnalyzerStop(TurnAnalyzerConfig{Analyzer: slowAnalyzer{took: inference}})
	spy := attachStop(s)

	spy.sendStop(s, frames.NewSTTMetadataFrame(sttP99))
	spy.sendStop(s, frames.NewVADUserStartedSpeakingFrame(stopSecs))

	// A transcript that is not final: only the safety net can release the turn.
	spy.sendStop(s, transcript("hello"))

	start := time.Now()
	spy.sendStop(s, frames.NewVADUserStoppedSpeakingFrame(stopSecs, time.Now()))

	// Anchored, the turn is released sttP99 minus the VAD's own stop window after
	// the speech ended, whatever the analyzer cost. Unanchored it would be that
	// plus the inference time.
	want := sttP99 - time.Duration(stopSecs*float64(time.Second))
	eventually(t, func() bool { return spy.stops() == 1 }, 2*time.Second, "turn never released")
	elapsed := time.Since(start)

	if elapsed < inference {
		t.Errorf("released after %v, before the analyzer even finished (%v)", elapsed, inference)
	}
	if slack := want + inference/2; elapsed > slack {
		t.Errorf("released after %v, want about %v: the inference time was added to "+
			"the budget rather than taken out of it", elapsed, want)
	}
}

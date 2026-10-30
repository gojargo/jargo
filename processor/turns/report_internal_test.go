package turns

import (
	"bytes"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Strategies report their decisions, and the controller combines them. These
// cover what the external and eager strategies report, and the deprecated
// names the reports had.

// openExternalTurn attaches s to a spy and opens a turn on it.
func openExternalTurn(t *testing.T, s StopStrategy) *spy {
	t.Helper()
	sp := attachStop(s)
	sp.mu.Lock()
	s.TurnStarted()
	sp.mu.Unlock()
	t.Cleanup(func() {
		sp.mu.Lock()
		s.Cleanup()
		sp.mu.Unlock()
	})
	return sp
}

// say is the service hearing the user say text.
func say(sp *spy, s StopStrategy, text string) {
	sp.sendStop(s, frames.NewProposedUserStartedSpeakingFrame())
	sp.sendStop(s, transcript(text))
	sp.sendStop(s, frames.NewProposedUserStoppedSpeakingFrame())
}

func TestExternalStopNoDecisionsWhileNoTurnIsOpen(t *testing.T) {
	s := NewExternalStop(ExternalStopConfig{Timeout: 50 * time.Millisecond, WaitForTranscript: new(bool)})
	sp := attachStop(s)
	defer s.Cleanup()

	// Never started a turn: nothing decides.
	time.Sleep(300 * time.Millisecond)
	if sp.stops() != 0 {
		t.Fatalf("stops = %d between turns, want 0", sp.stops())
	}

	// Once a turn opens, the stop signal ends it.
	sp.mu.Lock()
	s.TurnStarted()
	sp.mu.Unlock()
	sp.sendStop(s, frames.NewProposedUserStoppedSpeakingFrame())
	if sp.stops() != 1 {
		t.Fatalf("stops = %d, want 1", sp.stops())
	}
}

func TestExternalStopDecidesOnceWhileTheTurnStaysOpen(t *testing.T) {
	s := NewExternalStop(ExternalStopConfig{Timeout: 50 * time.Millisecond})
	sp := openExternalTurn(t, s)

	say(sp, s, "Hello!")
	time.Sleep(300 * time.Millisecond)

	if sp.stops() != 1 {
		t.Fatalf("decisions = %d, want 1", sp.stops())
	}
}

// The VAD hearing a cough clears the controller's decisions, so the strategy
// decides again after it.
func TestExternalStopSpeechTheServiceDidNotHearLetsItDecideAgain(t *testing.T) {
	s := NewExternalStop(ExternalStopConfig{Timeout: 50 * time.Millisecond})
	sp := openExternalTurn(t, s)
	say(sp, s, "Hello!")

	sp.sendStop(s, frames.NewVADUserStartedSpeakingFrame(0, time.Now()))
	sp.sendStop(s, frames.NewVADUserStoppedSpeakingFrame(0, time.Now()))

	if sp.stops() != 2 {
		t.Fatalf("decisions = %d, want 2", sp.stops())
	}
}

func TestExternalStopSpeechTheServiceHeardWaitsForItsStop(t *testing.T) {
	s := NewExternalStop(ExternalStopConfig{Timeout: 50 * time.Millisecond})
	sp := openExternalTurn(t, s)
	say(sp, s, "Hello!")

	sp.sendStop(s, frames.NewProposedUserStartedSpeakingFrame())
	sp.sendStop(s, frames.NewVADUserStartedSpeakingFrame(0, time.Now()))
	sp.sendStop(s, frames.NewVADUserStoppedSpeakingFrame(0, time.Now()))
	if sp.stops() != 1 {
		t.Fatalf("decisions = %d, want 1 until the service hears the user stop", sp.stops())
	}

	sp.sendStop(s, frames.NewProposedUserStoppedSpeakingFrame())
	if sp.stops() != 2 {
		t.Fatalf("decisions = %d, want 2", sp.stops())
	}
}

func TestExternalStopNewTextAfterTheDecisionGetsADecisionOfItsOwn(t *testing.T) {
	s := NewExternalStop(ExternalStopConfig{Timeout: 50 * time.Millisecond})
	sp := openExternalTurn(t, s)
	say(sp, s, "Book a flight")

	sp.sendStop(s, transcript(" to Boston"))
	time.Sleep(150 * time.Millisecond)

	if sp.stops() != 2 {
		t.Fatalf("decisions = %d, want 2", sp.stops())
	}
}

func TestExternalStopATranscriptionInPiecesAfterTheStopGetsOneDecision(t *testing.T) {
	s := NewExternalStop(ExternalStopConfig{Timeout: 100 * time.Millisecond})
	sp := openExternalTurn(t, s)
	sp.sendStop(s, frames.NewProposedUserStartedSpeakingFrame())
	sp.sendStop(s, frames.NewProposedUserStoppedSpeakingFrame())

	sp.sendStop(s, transcript("Book a flight"))
	time.Sleep(50 * time.Millisecond)
	sp.sendStop(s, transcript(" to Boston"))
	if sp.stops() != 0 {
		t.Fatalf("decisions = %d before the text settled, want 0", sp.stops())
	}

	time.Sleep(200 * time.Millisecond)
	if sp.stops() != 1 {
		t.Fatalf("decisions = %d, want 1", sp.stops())
	}
}

// Speech after a confirmation voids the turn's answer, and the confirmation
// with it.
func TestEagerStopADecisionAfterSpeechConfirmsNothing(t *testing.T) {
	s := NewEagerStop(EagerStopConfig{})
	sp := openExternalTurn(t, s)

	sp.sendStop(s, frames.NewProposedUserStartedSpeakingFrame())
	sp.sendStop(s, frames.NewEagerTranscriptionFrame("Book a flight", "user", ""))
	sp.sendStop(s, transcript("Book a flight"))
	sp.sendStop(s, frames.NewProposedUserStoppedSpeakingFrame())

	sp.sendStop(s, frames.NewVADUserStartedSpeakingFrame(0, time.Now()))
	sp.sendStop(s, frames.NewVADUserStoppedSpeakingFrame(0, time.Now()))

	sp.mu.Lock()
	defer sp.mu.Unlock()
	var confirms []bool
	for _, p := range sp.stopped {
		confirms = append(confirms, p.ConfirmsSpeculation)
	}
	if len(confirms) != 2 || !confirms[0] || confirms[1] {
		t.Fatalf("confirmations = %v, want [true false]", confirms)
	}
}

// specSpy records the speculations and withdrawals a stop strategy signals.
type specSpy struct {
	speculations []*UserTurnSpeculation
	cancels      int
}

func (s *specSpy) env() strategyEnv {
	return strategyEnv{
		inferenceTriggered:  func(_ StopStrategy, sp *UserTurnSpeculation) { s.speculations = append(s.speculations, sp) },
		speculationCanceled: func(StopStrategy) { s.cancels++ },
		push:                func(frames.Frame, processor.Direction) {},
	}
}

//nolint:staticcheck // the deprecated names are what is tested
func TestTheDeprecatedSpeculationMethodsStillFire(t *testing.T) {
	b := &plainStop{}
	spy := &specSpy{}
	b.attach(b, spy.env())

	b.TriggerInferenceTriggeredSpeculating(UserTurnSpeculation{Text: "Hello"})
	b.TriggerSpeculationCanceled()

	if len(spy.speculations) != 1 || spy.speculations[0].Text != "Hello" || spy.cancels != 1 {
		t.Fatalf("speculations = %v, cancels = %d, want one of each", spy.speculations, spy.cancels)
	}
}

//nolint:staticcheck // the deprecated name is what is tested
func TestAnInferenceWithoutASpeculationWarnsOnceAndIsIgnored(t *testing.T) {
	var buf bytes.Buffer
	previous := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	t.Cleanup(func() { slog.SetDefault(previous) })

	b := &plainStop{}
	spy := &specSpy{}
	b.attach(b, spy.env())

	b.TriggerInferenceTriggered()
	b.TriggerInferenceTriggered()

	if len(spy.speculations) != 0 {
		t.Fatalf("speculations = %d, want the call ignored", len(spy.speculations))
	}
	if n := strings.Count(buf.String(), "without a speculation"); n != 1 {
		t.Fatalf("warnings = %d, want 1:\n%s", n, buf.String())
	}
}

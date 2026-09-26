package observers_test

import (
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/observers"
	"github.com/gojargo/jargo/processor"
)

// speakingRecorder collects everything a Speaking observer reports, against a
// clock a test advances rather than waits on.
type speakingRecorder struct {
	mu     sync.Mutex
	clock  time.Time
	events []observers.SpeechEvent
}

// newSpeakingRecorder starts the clock at a round moment, so an offset read off
// a failure is legible.
func newSpeakingRecorder() *speakingRecorder {
	return &speakingRecorder{clock: time.Date(2026, time.September, 13, 12, 0, 0, 0, time.UTC)}
}

func (r *speakingRecorder) now() time.Time {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.clock
}

// wait advances the clock without sleeping.
func (r *speakingRecorder) wait(d time.Duration) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.clock = r.clock.Add(d)
}

func (r *speakingRecorder) record(e observers.SpeechEvent) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, e)
}

func (r *speakingRecorder) all() []observers.SpeechEvent {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]observers.SpeechEvent(nil), r.events...)
}

// kinds is the sequence of moments reported, which is what most of these tests
// are about.
func (r *speakingRecorder) kinds() []observers.SpeechEventKind {
	all := r.all()
	out := make([]observers.SpeechEventKind, 0, len(all))
	for _, e := range all {
		out = append(out, e.Kind)
	}
	return out
}

// of returns the single event of the given kind.
func (r *speakingRecorder) of(t *testing.T, kind observers.SpeechEventKind) observers.SpeechEvent {
	t.Helper()
	for _, e := range r.all() {
		if e.Kind == kind {
			return e
		}
	}
	t.Fatalf("no %s event was reported, got %v", kind, r.kinds())
	return observers.SpeechEvent{}
}

// newSpeaking builds an observer reporting into the recorder, on its clock.
func newSpeaking(r *speakingRecorder) *observers.Speaking {
	return observers.NewSpeaking(observers.SpeakingConfig{
		Now:           r.now,
		OnSpeechEvent: r.record,
	})
}

// sameKinds compares two sequences of moments.
func sameKinds(got, want []observers.SpeechEventKind) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

// TestSpeechIsTimedToSpeechNotToTheDetector covers the reason the VAD frames
// carry their own timestamps: a bar drawn from confirmation times is fat at both
// ends, because the detector needs to hear speech before it says so and silence
// before it says that.
func TestSpeechIsTimedToSpeechNotToTheDetector(t *testing.T) {
	r := newSpeakingRecorder()
	o := newSpeaking(r)

	push(o, frames.NewVADUserStartedSpeakingFrame(0.2, r.now()), processor.Downstream)
	r.wait(2500 * time.Millisecond)
	push(o, frames.NewVADUserStoppedSpeakingFrame(0.8, r.now()), processor.Downstream)

	started := r.of(t, observers.UserSpeechStarted)
	stopped := r.of(t, observers.UserSpeechStopped)

	// The detector confirmed 0.2s after speech began, and 0.8s after it ended.
	if want := r.now().Add(-2500*time.Millisecond - 200*time.Millisecond); !started.Timestamp.Equal(want) {
		t.Errorf("started at %v, want %v", started.Timestamp, want)
	}
	if want := r.now().Add(-800 * time.Millisecond); !stopped.Timestamp.Equal(want) {
		t.Errorf("stopped at %v, want %v", stopped.Timestamp, want)
	}
	// 2.5s of wall clock, less the stop window, plus the start window.
	if got, want := stopped.Timestamp.Sub(stopped.StartedAt), 1900*time.Millisecond; got != want {
		t.Errorf("speech lasted %v, want %v", got, want)
	}
}

// TestClosingMomentNamesTheStretchItEnds covers an interval reading whole from
// one record, without pairing it with the moment that opened it.
func TestClosingMomentNamesTheStretchItEnds(t *testing.T) {
	r := newSpeakingRecorder()
	o := newSpeaking(r)

	startedAt := r.now()
	push(o, frames.NewBotStartedSpeakingFrame(), processor.Downstream)
	r.wait(6300 * time.Millisecond)
	push(o, frames.NewBotStoppedSpeakingFrame(), processor.Downstream)

	stopped := r.of(t, observers.BotSpeechStopped)
	if !stopped.StartedAt.Equal(startedAt) {
		t.Errorf("started at %v, want %v", stopped.StartedAt, startedAt)
	}
	if got, want := stopped.Timestamp.Sub(stopped.StartedAt), 6300*time.Millisecond; got != want {
		t.Errorf("the bot spoke for %v, want %v", got, want)
	}
}

// TestMicrophoneAndStrategyAreReportedApart covers the two layers the user
// appears at: the pipeline acts on the strategy's ruling, and only the detector
// hears a false start.
func TestMicrophoneAndStrategyAreReportedApart(t *testing.T) {
	r := newSpeakingRecorder()
	o := newSpeaking(r)

	push(o, frames.NewVADUserStartedSpeakingFrame(0, r.now()), processor.Downstream)
	push(o, frames.NewUserStartedSpeakingFrame(), processor.Downstream)
	r.wait(time.Second)
	push(o, frames.NewVADUserStoppedSpeakingFrame(0, r.now()), processor.Downstream)
	r.wait(300 * time.Millisecond) // the strategy deliberating
	push(o, frames.NewUserStoppedSpeakingFrame(), processor.Downstream)

	want := []observers.SpeechEventKind{
		observers.UserSpeechStarted,
		observers.UserTurnStarted,
		observers.UserSpeechStopped,
		observers.UserTurnStopped,
	}
	if got := r.kinds(); !sameKinds(got, want) {
		t.Fatalf("kinds = %v, want %v", got, want)
	}

	vad := r.of(t, observers.UserSpeechStopped)
	turn := r.of(t, observers.UserTurnStopped)
	if got, want := turn.Timestamp.Sub(vad.Timestamp), 300*time.Millisecond; got != want {
		t.Errorf("the ruling took %v, want %v", got, want)
	}
}

// TestInterruptionNamesNoSpeech covers the one moment that is not a stretch: any
// processor can call for an interruption, and it ends nothing.
func TestInterruptionNamesNoSpeech(t *testing.T) {
	r := newSpeakingRecorder()
	o := newSpeaking(r)

	push(o, frames.NewInterruptionFrame(), processor.Downstream)

	e := r.of(t, observers.Interruption)
	if !e.StartedAt.IsZero() {
		t.Errorf("started at %v, want the zero time: an interruption ends no stretch", e.StartedAt)
	}
}

// TestBroadcastInterruptionReportedOnce covers the pair a broadcast builds: two
// frames with two ids, so an id alone cannot tell them apart.
func TestBroadcastInterruptionReportedOnce(t *testing.T) {
	r := newSpeakingRecorder()
	o := newSpeaking(r)

	down := frames.NewInterruptionFrame()
	up := frames.NewInterruptionFrame()
	down.SetBroadcastSiblingID(up.ID())
	up.SetBroadcastSiblingID(down.ID())

	push(o, down, processor.Downstream)
	push(o, up, processor.Upstream)

	if got := r.all(); len(got) != 1 {
		t.Errorf("events = %d, want 1: the two halves are one interruption", len(got))
	}
}

// TestStretchWhoseStartWasMissedClosesWithoutOne covers the observer attached
// mid-conversation: the stretch closes without a start rather than borrowing one
// from another stretch.
func TestStretchWhoseStartWasMissedClosesWithoutOne(t *testing.T) {
	r := newSpeakingRecorder()
	o := newSpeaking(r)

	push(o, frames.NewBotStoppedSpeakingFrame(), processor.Downstream)

	if got := r.of(t, observers.BotSpeechStopped); !got.StartedAt.IsZero() {
		t.Errorf("started at %v, want the zero time", got.StartedAt)
	}
}

// TestOtherFramesAreLeftAlone covers only the speaking lifecycle being reported.
func TestOtherFramesAreLeftAlone(t *testing.T) {
	r := newSpeakingRecorder()
	o := newSpeaking(r)

	push(o, frames.NewTextFrame("hello"), processor.Downstream)

	if got := r.all(); len(got) != 0 {
		t.Errorf("events = %v, want none", got)
	}
}

// TestBargeInReadsAsAnOverlap covers what makes a timeline show one voice
// cutting into another: the user's speech begins while the bot still has the
// floor.
func TestBargeInReadsAsAnOverlap(t *testing.T) {
	r := newSpeakingRecorder()
	o := newSpeaking(r)

	push(o, frames.NewBotStartedSpeakingFrame(), processor.Downstream)
	r.wait(5 * time.Second)
	push(o, frames.NewVADUserStartedSpeakingFrame(0, r.now()), processor.Downstream)
	push(o, frames.NewInterruptionFrame(), processor.Downstream)
	r.wait(200 * time.Millisecond)
	push(o, frames.NewBotStoppedSpeakingFrame(), processor.Downstream)

	botStopped := r.of(t, observers.BotSpeechStopped)
	userStarted := r.of(t, observers.UserSpeechStarted)

	if !userStarted.Timestamp.Before(botStopped.Timestamp) {
		t.Error("the user started after the bot stopped, want during")
	}
	if !userStarted.Timestamp.After(botStopped.StartedAt) {
		t.Error("the user started before the bot did, want during")
	}
}

// TestSpeechThatNeverBecomesATurnIsStillReported covers a cough or a false
// start: it reaches the microphone and stops there, so the strategy never rules
// and the pipeline never takes a turn from it.
func TestSpeechThatNeverBecomesATurnIsStillReported(t *testing.T) {
	r := newSpeakingRecorder()
	o := newSpeaking(r)

	push(o, frames.NewVADUserStartedSpeakingFrame(0, r.now()), processor.Downstream)
	r.wait(300 * time.Millisecond)
	push(o, frames.NewVADUserStoppedSpeakingFrame(0, r.now()), processor.Downstream)

	want := []observers.SpeechEventKind{observers.UserSpeechStarted, observers.UserSpeechStopped}
	if got := r.kinds(); !sameKinds(got, want) {
		t.Errorf("kinds = %v, want %v", got, want)
	}
}

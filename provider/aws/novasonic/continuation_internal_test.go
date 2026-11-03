package novasonic

import (
	"context"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// (Not ports of upstream tests: upstream has none for session continuation or
// the assistant response trigger.)

// continuingHarness is a session set up with continuation looking to hand over
// at once, its monitor ticking fast.
func continuingHarness(t *testing.T, params SessionContinuationParams) (*harness, *fakeStream) {
	t.Helper()
	h := newHarness(t, Config{Instructions: "Be brief.", SessionContinuation: &params})
	h.s.sc.tick = 5 * time.Millisecond
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Hello")
	h.seed(t, convo)
	return h, h.client.stream(t, 0)
}

func (c *sessionContinuation) waitingForAudio() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.waitingForAudioStart
}

// The handover: past the threshold, the assistant starting to speak opens the
// next session and buffers the user's audio; the assistant finishing hands the
// history and the audio over and swaps the new session in; the old one is
// closed.
func TestASessionIsHandedOver(t *testing.T) {
	h, old := continuingHarness(t, SessionContinuationParams{
		Enabled: true, AudioBufferDuration: time.Second,
	})
	waitFor(t, h.s.sc.waitingForAudio)

	h.drive(t, "contentStart", contentStartBody("TEXT", "ASSISTANT", "SPECULATIVE"))
	h.drive(t, "textOutput", textOutputBody("Hi there."))
	h.drive(t, "contentEnd", contentEndBody("PARTIAL_TURN"))
	h.drive(t, "contentStart", contentStartBody("AUDIO", "ASSISTANT", ""))
	next := h.client.stream(t, 1)
	if got := next.names(); !reflect.DeepEqual(got,
		[]string{"sessionStart", "promptStart", "contentStart", "textInput", "contentEnd"}) {
		t.Fatalf("next session = %v, want it set up as far as the system instruction", got)
	}
	if !h.s.sc.isBuffering() {
		t.Fatal("the user's audio should be buffered")
	}
	ctx := context.Background()
	if err := h.s.ProcessFrame(ctx, frames.NewInputAudioRawFrame([]byte{9, 9}, 16000, 1),
		processor.Downstream); err != nil {
		t.Fatal(err)
	}

	h.drive(t, "contentEnd", contentEndBody("END_TURN"))
	h.drive(t, "contentStart", contentStartBody("TEXT", "ASSISTANT", "FINAL"))
	h.drive(t, "textOutput", textOutputBody("Hi there."))
	h.drive(t, "contentEnd", contentEndBody("END_TURN"))

	waitFor(t, func() bool {
		h.s.mu.Lock()
		defer h.s.mu.Unlock()
		return h.s.stream == sessionStream(next)
	})
	texts := next.texts()
	want := []sentText{
		{role: "SYSTEM", text: "Be brief."},
		{role: "USER", text: "Hello"},
		{role: "ASSISTANT", text: "Hi there."},
	}
	if !reflect.DeepEqual(texts, want) {
		t.Errorf("handed-over texts = %+v, want %+v", texts, want)
	}
	names := next.names()
	if n := len(names); n < 2 || names[n-2] != "contentStart" || names[n-1] != "audioInput" {
		t.Errorf("next session = %v, want the audio block opened and the buffered audio sent", names)
	}

	// The old session is closed: the audio block, the prompt and the session.
	waitFor(t, func() bool { return slices.Contains(old.names(), "sessionEnd") })
	old.end()
	got := old.names()
	if tail := got[len(got)-3:]; !reflect.DeepEqual(tail, []string{"contentEnd", "promptEnd", "sessionEnd"}) {
		t.Errorf("old session's last events = %v", tail)
	}

	// The user's audio now goes to the new session.
	before := len(next.names())
	if err := h.s.ProcessFrame(ctx, frames.NewInputAudioRawFrame([]byte{7, 7}, 16000, 1),
		processor.Downstream); err != nil {
		t.Fatal(err)
	}
	if got := next.names(); len(got) != before+1 || got[before] != "audioInput" {
		t.Errorf("next session = %v, want the user's audio", got)
	}
}

// When the assistant does not speak within the timeout, the handover is forced,
// and the service swaps the new session in.
func TestAForcedTransitionSwapsTheSession(t *testing.T) {
	h, old := continuingHarness(t, SessionContinuationParams{
		Enabled: true, AudioBufferDuration: time.Second, AudioStartTimeout: time.Millisecond,
	})
	next := h.client.stream(t, 1)
	waitFor(t, func() bool {
		h.s.mu.Lock()
		defer h.s.mu.Unlock()
		return h.s.stream == sessionStream(next)
	})
	waitFor(t, func() bool { return slices.Contains(old.names(), "sessionEnd") })
	old.end()
}

// Continuation turned off never opens another session.
func TestContinuationCanBeTurnedOff(t *testing.T) {
	h, _ := continuingHarness(t, SessionContinuationParams{Enabled: false})
	time.Sleep(30 * time.Millisecond)
	if n := h.client.opened(); n != 1 {
		t.Errorf("opened %d sessions, want one", n)
	}
}

// The history handed over is the most recent, within the limits, starting with
// the user's.
func TestTheHistoryIsTrimmed(t *testing.T) {
	long := strings.Repeat("é", maxHistoryMessageBytes) // two bytes a rune
	got, total := trimHistory([]historyMessage{
		{role: "ASSISTANT", text: "dropped: it leads"},
		{role: "USER", text: long},
		{role: "ASSISTANT", text: "ok"},
	})
	if len(got) != 2 || got[0].role != "USER" || len(got[0].text) != maxHistoryMessageBytes {
		t.Fatalf("history = %d messages (first %d bytes), want the user's cut to the limit", len(got), len(got[0].text))
	}
	// The total counts a leading message dropped afterwards; it is only logged.
	if total != maxHistoryMessageBytes+2+len("dropped: it leads") {
		t.Errorf("total = %d", total)
	}

	var many []historyMessage
	for range 5 {
		many = append(many, historyMessage{role: "USER", text: strings.Repeat("a", maxHistoryMessageBytes)})
	}
	if got, _ := trimHistory(many); len(got) != 4 {
		t.Errorf("history = %d messages, want what fits the total", len(got))
	}
}

// The trigger recording is the PCM of the embedded WAV.
func TestTheTriggerRecordingIsLoaded(t *testing.T) {
	if got := len(readyAudio()); got != 11703*2 {
		t.Errorf("trigger audio = %d bytes, want 23406", got)
	}
}

// On the first-generation model the trigger plays the recording to the session,
// and its transcript is wrapped in the user's turn.
func TestTheTriggerPlaysTheRecording(t *testing.T) {
	h := newHarness(t, Config{Model: firstGenerationModel})
	h.s.triggerAudio = make([]byte, 1280)
	h.seed(t, frames.NewLLMContext(AwaitTriggerAssistantResponseInstruction))
	h.s.mu.Lock()
	h.s.connectedTime = time.Now().Add(-time.Second)
	h.s.mu.Unlock()
	st := h.client.stream(t, 0)
	before := len(st.names())
	h.s.TriggerAssistantResponse(context.Background())
	if got := st.names()[before:]; !reflect.DeepEqual(got, []string{"audioInput", "audioInput"}) {
		t.Errorf("events = %v, want the recording in two chunks", got)
	}

	h.drive(t, "contentStart", contentStartBody("TEXT", "USER", "FINAL"))
	h.drive(t, "textOutput", textOutputBody(" Ready "))
	h.drive(t, "contentEnd", contentEndBody("END_TURN"))
	h.drive(t, "contentStart", contentStartBody("TEXT", "ASSISTANT", "SPECULATIVE"))
	var up []string
	for _, f := range h.up.frames() {
		switch f.(type) {
		case *frames.UserStartedSpeakingFrame, *frames.TranscriptionFrame, *frames.UserStoppedSpeakingFrame:
			name, _, _ := strings.Cut(f.Name(), "#")
			up = append(up, name)
		}
	}
	want := []string{"UserStartedSpeakingFrame", "TranscriptionFrame", "UserStoppedSpeakingFrame"}
	if !reflect.DeepEqual(up, want) {
		t.Errorf("upstream = %v, want the transcript wrapped in the user's turn", up)
	}
}

// Later models need no trigger, so none is played.
func TestTheTriggerIsSkippedForLaterModels(t *testing.T) {
	h, st := liveStream(t)
	before := len(st.names())
	h.s.TriggerAssistantResponse(context.Background())
	if got := st.names()[before:]; len(got) != 0 {
		t.Errorf("events = %v, want nothing", got)
	}
}

package eval

import (
	"encoding/json"
	"testing"

	"github.com/gojargo/jargo/processor/rtvi"
)

// msg builds an inbound RTVI server message with the given payload.
func msg(t *testing.T, msgType string, data any) rtvi.Incoming {
	t.Helper()
	in := rtvi.Incoming{Label: rtvi.MessageLabel, Type: msgType}
	if data != nil {
		encoded, err := json.Marshal(data)
		if err != nil {
			t.Fatal(err)
		}
		in.Data = encoded
	}
	return in
}

// wantEvent asserts translate produced exactly the given event.
func wantEvent(t *testing.T, got *event, kind, text string) {
	t.Helper()
	if got == nil {
		t.Fatalf("expected a %s event, got none", kind)
	}
	if got.kind != kind || got.text != text {
		t.Fatalf("got %+v, want kind %q text %q", *got, kind, text)
	}
}

// wantNoEvent asserts translate produced nothing.
func wantNoEvent(t *testing.T, got *event) {
	t.Helper()
	if got != nil {
		t.Fatalf("expected no event, got %+v", *got)
	}
}

// TestTranslateAccumulatesLLMText checks the bot's text is joined across the
// response and emitted once, at the boundary.
func TestTranslateAccumulatesLLMText(t *testing.T) {
	s := &session{events: make(chan event, eventBuffer)}

	wantEvent(t, s.translate(msg(t, rtvi.TypeBotLLMStarted, nil)), EventLLMStarted, "")
	wantNoEvent(t, s.translate(msg(t, rtvi.TypeBotLLMText, rtvi.TextData{Text: "Hello "})))
	wantNoEvent(t, s.translate(msg(t, rtvi.TypeBotLLMText, rtvi.TextData{Text: "world"})))
	wantEvent(t, s.translate(msg(t, rtvi.TypeBotLLMStopped, nil)), EventLLMResponse, "Hello world")
}

// TestTranslateEmptyResponseStillEmitted checks an interrupted response with no
// text still emits an llm_response. The matcher's aggregation decides whether
// that should pass or fail.
func TestTranslateEmptyResponseStillEmitted(t *testing.T) {
	s := &session{events: make(chan event, eventBuffer)}

	wantEvent(t, s.translate(msg(t, rtvi.TypeBotLLMStarted, nil)), EventLLMStarted, "")
	wantEvent(t, s.translate(msg(t, rtvi.TypeBotLLMStopped, nil)), EventLLMResponse, "")
}

// TestTranslateInterruptionSuppressesStraggler checks that after an
// interruption the interrupted response can still flush a trailing token, and
// that it is dropped rather than attributed to the new turn. The genuinely new
// response begins at the next bot-llm-started.
func TestTranslateInterruptionSuppressesStraggler(t *testing.T) {
	s := &session{events: make(chan event, eventBuffer)}

	s.translate(msg(t, rtvi.TypeBotLLMStarted, nil))
	s.translate(msg(t, rtvi.TypeBotLLMText, rtvi.TextData{Text: "Tell me about Paris"}))

	// The user barges in: the bot is interrupted.
	got := s.translate(msg(t, rtvi.TypeBotInterrupted, nil))
	wantEvent(t, got, EventBotInterrupted, "")

	// A straggler from the interrupted response arrives just after the interrupt.
	wantNoEvent(t, s.translate(msg(t, rtvi.TypeBotLLMText, rtvi.TextData{Text: " what would"})))
	wantNoEvent(t, s.translate(msg(t, rtvi.TypeBotLLMStopped, nil)))

	// The genuinely new response.
	wantEvent(t, s.translate(msg(t, rtvi.TypeBotLLMStarted, nil)), EventLLMStarted, "")
	wantNoEvent(t, s.translate(msg(t, rtvi.TypeBotLLMText, rtvi.TextData{Text: "Tokyo"})))
	wantEvent(t, s.translate(msg(t, rtvi.TypeBotLLMStopped, nil)), EventLLMResponse, "Tokyo")
}

// TestTranslateUserStartedSpeakingDiscardsOutput checks a new user turn drops
// the bot output queued from a prior one, so a greeting the user interrupted
// cannot be matched against this turn.
func TestTranslateUserStartedSpeakingDiscardsOutput(t *testing.T) {
	s := &session{events: make(chan event, eventBuffer)}
	s.events <- event{kind: EventLLMResponse, text: "the greeting"}

	wantEvent(t, s.translate(msg(t, rtvi.TypeUserStartedSpeaking, nil)), EventUserStartedSpeaking, "")

	if len(s.events) != 0 {
		t.Fatalf("expected the queued bot output to be dropped, %d event(s) left", len(s.events))
	}
}

// TestTranslateDiscardPreservesUserTranscription checks the discard keeps a user
// transcription: a keypress emits its transcription immediately before the
// turn-start interruption, and that transcription is the turn's input, not the
// stale bot output the discard is meant to clear.
func TestTranslateDiscardPreservesUserTranscription(t *testing.T) {
	s := &session{events: make(chan event, eventBuffer)}
	s.events <- event{kind: EventLLMResponse, text: "the greeting"}
	s.events <- event{kind: EventUserTranscription, text: "DTMF: 123#"}

	s.translate(msg(t, rtvi.TypeUserStartedSpeaking, nil))

	if len(s.events) != 1 {
		t.Fatalf("expected only the transcription to survive, %d event(s) left", len(s.events))
	}
	if ev := <-s.events; ev.kind != EventUserTranscription || ev.text != "DTMF: 123#" {
		t.Fatalf("unexpected surviving event: %+v", ev)
	}
}

// TestTranslateUserTranscriptionFinalOnly checks an interim transcription is not
// a turn-level event.
func TestTranslateUserTranscriptionFinalOnly(t *testing.T) {
	s := &session{events: make(chan event, eventBuffer)}

	interim := rtvi.UserTranscriptionData{Text: "hel", Final: false}
	wantNoEvent(t, s.translate(msg(t, rtvi.TypeUserTranscription, interim)))

	final := rtvi.UserTranscriptionData{Text: "hello", Final: true}
	wantEvent(t, s.translate(msg(t, rtvi.TypeUserTranscription, final)), EventUserTranscription, "hello")
}

// TestTranslateFunctionCall checks the call's name and arguments both reach the
// matcher.
func TestTranslateFunctionCall(t *testing.T) {
	s := &session{events: make(chan event, eventBuffer)}

	data := rtvi.LLMFunctionCallData{
		ToolCallID:   "call-1",
		FunctionName: "get_weather",
		Arguments:    json.RawMessage(`{"city":"Paris"}`),
	}
	ev := s.translate(msg(t, rtvi.TypeLLMFunctionCall, data))
	if ev == nil || ev.kind != EventFunctionCall || ev.function != "get_weather" {
		t.Fatalf("unexpected event: %+v", ev)
	}
	if ev.args["city"] != "Paris" {
		t.Fatalf("unexpected args: %+v", ev.args)
	}
}

// TestTranslateUnmappedMessageIgnored checks a message the scenario format has
// no event for carries none.
func TestTranslateUnmappedMessageIgnored(t *testing.T) {
	s := &session{events: make(chan event, eventBuffer)}
	wantNoEvent(t, s.translate(msg(t, rtvi.TypeMetrics, nil)))
}

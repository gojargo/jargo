package eval

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
)

// newStream is a stream with a trace of its own.
func newStream(botAudio bool) *EventStream { return NewEventStream(botAudio, NewTrace()) }

// messageFrame is an RTVI server message as the client serializer keeps it: a
// raw transport message.
func messageFrame(t *testing.T, msgType string, data any) *frames.InputTransportMessageFrame {
	t.Helper()
	raw, err := json.Marshal(map[string]any{"label": rtvi.MessageLabel, "type": msgType, "data": data})
	if err != nil {
		t.Fatal(err)
	}
	return frames.NewInputTransportMessageFrame(raw)
}

// bare is an event without the timing it carries for the measures.
func bare(t *testing.T, event map[string]any) map[string]any {
	t.Helper()
	if event == nil {
		t.Fatal("no event")
	}
	out := map[string]any{}
	for k, v := range event {
		if k != "started_at" && k != "at" {
			out[k] = v
		}
	}
	return out
}

func wantEvent(t *testing.T, got, want map[string]any) {
	t.Helper()
	if !reflect.DeepEqual(bare(t, got), want) {
		t.Fatalf("event = %#v, want %#v", got, want)
	}
}

func wantNoEvent(t *testing.T, got map[string]any) {
	t.Helper()
	if got != nil {
		t.Fatalf("event = %#v, want none", got)
	}
}

// queued is every event still queued, in order, without taking them.
func queued(s *EventStream) []map[string]any {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]map[string]any(nil), s.queue...)
}

// A reply carries the time of its first token; a response with no text none.
func TestAReplyCarriesTheTimeOfItsFirstToken(t *testing.T) {
	s := newStream(false)
	s.FrameToEvent(frames.NewLLMFullResponseStartFrame())
	s.FrameToEvent(frames.NewLLMTextFrame("Hello "))
	s.FrameToEvent(frames.NewLLMTextFrame("world"))
	event := s.FrameToEvent(frames.NewLLMFullResponseEndFrame())
	startedAt, ok := event["started_at"].(float64)
	if !ok || startedAt > s.Elapsed() {
		t.Fatalf("started_at = %v, want the time of the first token", event["started_at"])
	}
	s.FrameToEvent(frames.NewLLMFullResponseStartFrame())
	if event := s.FrameToEvent(frames.NewLLMFullResponseEndFrame()); event["started_at"] != nil {
		t.Fatalf("a response with no text carries a time: %v", event)
	}
}

func TestLLMMarkerIsItsOwnEvent(t *testing.T) {
	s := newStream(false)
	s.FrameToEvent(frames.NewLLMFullResponseStartFrame())
	s.FrameToEvent(frames.NewLLMTextFrame("Hello"))
	markers := []string{"●", "◐", "○"}
	wantEvent(t, s.FrameToEvent(frames.NewLLMMarkerResponseFrame("● Hello", "●", "complete", markers)),
		map[string]any{keyType: EventLLMMarker, "text": "●", "kind": "complete", "raw": "● Hello", "markers": markers})
	// The marker is not part of the reply's text.
	wantEvent(t, s.FrameToEvent(frames.NewLLMFullResponseEndFrame()),
		map[string]any{keyType: EventLLMResponse, "text": "Hello"})
}

func TestLLMMarkerBeforeTheReplyStartsIsDropped(t *testing.T) {
	s := newStream(false)
	s.InputSent()
	wantNoEvent(t, s.FrameToEvent(frames.NewLLMMarkerResponseFrame("◐", "◐", "", nil)))
	s.FrameToEvent(frames.NewLLMFullResponseStartFrame())
	wantEvent(t, s.FrameToEvent(frames.NewLLMMarkerResponseFrame("◐", "◐", "short", nil)),
		map[string]any{keyType: EventLLMMarker, "text": "◐", "kind": "short", "raw": "◐", "markers": []string{}})
}

func TestLLMLifecycleAggregatesText(t *testing.T) {
	s := newStream(false)
	wantEvent(t, s.FrameToEvent(frames.NewLLMFullResponseStartFrame()), map[string]any{keyType: EventLLMStarted})
	wantNoEvent(t, s.FrameToEvent(frames.NewLLMTextFrame("Hello ")))
	wantNoEvent(t, s.FrameToEvent(frames.NewLLMTextFrame("world")))
	wantEvent(t, s.FrameToEvent(frames.NewLLMFullResponseEndFrame()),
		map[string]any{keyType: EventLLMResponse, "text": "Hello world"})
}

func TestInterruptionSuppressesStraggler(t *testing.T) {
	s := newStream(false)
	s.FrameToEvent(frames.NewLLMFullResponseStartFrame())
	s.FrameToEvent(frames.NewLLMTextFrame("Tell me about Paris"))
	wantEvent(t, s.FrameToEvent(messageFrame(t, rtvi.TypeBotInterrupted, nil)),
		map[string]any{keyType: EventBotInterrupted})
	// Straggler from the interrupted response is dropped.
	wantNoEvent(t, s.FrameToEvent(frames.NewLLMTextFrame(" what would")))
	wantNoEvent(t, s.FrameToEvent(frames.NewLLMFullResponseEndFrame()))
	// The genuinely new response.
	s.FrameToEvent(frames.NewLLMFullResponseStartFrame())
	s.FrameToEvent(frames.NewLLMTextFrame("Tokyo"))
	wantEvent(t, s.FrameToEvent(frames.NewLLMFullResponseEndFrame()),
		map[string]any{keyType: EventLLMResponse, "text": "Tokyo"})
}

// The bot's reported user-input arrives as a raw message, not a
// TranscriptionFrame, which the harness keeps for the bot's own response.
func TestUserTranscriptionFromMessage(t *testing.T) {
	s := newStream(false)
	input := func(data map[string]any) map[string]any {
		return s.FrameToEvent(messageFrame(t, rtvi.TypeUserInput, data))
	}
	wantEvent(t, input(map[string]any{"text": "hello", "input_type": "transcription", "final": true}),
		map[string]any{keyType: EventUserTranscription, "transcript": "hello"})
	// Interim transcriptions and chat text are ignored.
	wantNoEvent(t, input(map[string]any{"text": "hel", "input_type": "transcription", "final": false}))
	wantNoEvent(t, input(map[string]any{"text": "hello", "input_type": "chat", "final": true}))
}

// The bot's backchannel counts once it was spoken to the end; the user's counts
// as soon as the bot reports it.
func TestBackchannelsFromMessages(t *testing.T) {
	s := newStream(true)
	output := func(outputType, status, accumulated string) map[string]any {
		return s.FrameToEvent(messageFrame(t, rtvi.TypeBotOutput, map[string]any{
			"text": "Mm-hmm.", "output_type": outputType, "spoken_status": status,
			"spoken_progress": map[string]any{"accumulated_text": accumulated, "remaining_text": ""},
		}))
	}
	wantNoEvent(t, output("backchannel", "new", ""))
	wantEvent(t, output("backchannel", "completed", "Mm-hmm."),
		map[string]any{keyType: EventBotBackchannel, "text": "Mm-hmm."})
	wantNoEvent(t, output("sentence", "completed", "Mm-hmm."))
	wantEvent(t, s.FrameToEvent(messageFrame(t, rtvi.TypeUserInput,
		map[string]any{"text": "mhm", "input_type": "backchannel"})),
		map[string]any{keyType: EventUserBackchannel, "text": "mhm"})
	wantNoEvent(t, s.FrameToEvent(messageFrame(t, rtvi.TypeUserInput,
		map[string]any{"text": "hello", "input_type": "transcription"})))
}

// Expressive conversation reports what the TTS performed: where the delivery of
// the spoken text changes, and each sound and pause. The messages are the ones
// the bot's RTVI observer sends.
func TestExpressionFromMessages(t *testing.T) {
	s := newStream(true)
	output := func(segmentID uint64, textType frames.TextType, text string, delivery []string,
		status rtvi.SpokenStatus,
	) any {
		willBeSpoken := true
		data, _ := json.Marshal(rtvi.BotOutputData{
			Text: text, OutputType: textType, TextType: textType, SegmentID: &segmentID, //nolint:staticcheck // sent
			WillBeSpoken: &willBeSpoken, SpokenStatus: status, Delivery: delivery,
		})
		var decoded map[string]any
		_ = json.Unmarshal(data, &decoded)
		event := s.FrameToEvent(messageFrame(t, rtvi.TypeBotOutput, decoded))
		if event == nil {
			return nil
		}
		return event["text"]
	}
	words := func(segmentID uint64, delivery []string, status rtvi.SpokenStatus) any {
		return output(segmentID, frames.TextTypeSentence, "words", delivery, status)
	}
	want := func(got any, text string) {
		t.Helper()
		if text == "" && got != nil || text != "" && got != text {
			t.Fatalf("expression = %v, want %q", got, text)
		}
	}

	s.FrameToEvent(frames.NewLLMFullResponseStartFrame())
	// Spoken text counts as it is spoken, not when it is announced.
	want(words(1, []string{"excited"}, rtvi.SpokenNew), "")
	want(words(1, []string{"excited"}, rtvi.SpokenInProgress), "<<excited>>")
	want(words(1, []string{"excited"}, rtvi.SpokenCompleted), "")
	want(words(2, []string{"excited", "emphasis"}, rtvi.SpokenInProgress), "<<excited, emphasis>>")
	// Delivery can change within a segment, word by word.
	want(words(2, []string{"calm"}, rtvi.SpokenInProgress), "<<calm>>")
	// A sound counts when it is heard, once.
	want(output(3, frames.TextTypeExpressionSound, "laughter", nil, rtvi.SpokenNew), "")
	want(output(3, frames.TextTypeExpressionSound, "laughter", nil, rtvi.SpokenCompleted), "[[laughter]]")
	want(output(3, frames.TextTypeExpressionSound, "laughter", nil, rtvi.SpokenCompleted), "")
	want(output(4, frames.TextTypeExpressionPause, "600", nil, rtvi.SpokenCompleted), "[[pause 600]]")
	want(words(5, []string{"excited"}, rtvi.SpokenInProgress), "<<excited>>")
	want(words(6, []string{"excited"}, rtvi.SpokenInProgress), "")
	want(words(7, []string{}, rtvi.SpokenInProgress), "<<neutral>>")
	// Without expressive conversation, spoken text carries no delivery.
	want(words(8, nil, rtvi.SpokenInProgress), "")
	// Each response starts from the baseline.
	s.FrameToEvent(frames.NewLLMFullResponseStartFrame())
	want(words(9, []string{}, rtvi.SpokenInProgress), "")
}

// The user aggregator consumes the STT's TranscriptionFrames to build the bot's
// turn; the response is appended at the turn's end, not here.
func TestTranscriptionFrameIsIgnoredByTheSink(t *testing.T) {
	wantNoEvent(t, newStream(true).FrameToEvent(frames.NewTranscriptionFrame("Paris", "bot", "t")))
}

// The bot's reports about the harness (its raw VAD and turn-level speaking)
// arrive as raw messages and map to the matching scenario events.
func TestReportedSpeakingAndVADEventsFromMessages(t *testing.T) {
	s := newStream(true)
	for msgType, event := range map[string]string{
		rtvi.TypeUserStartedSpeaking: EventUserStartedSpeaking,
		rtvi.TypeUserStoppedSpeaking: EventUserStoppedSpeaking,
		rtvi.TypeVADUserStarted:      EventVADUserStartedSpeaking,
		rtvi.TypeVADUserStopped:      EventVADUserStoppedSpeaking,
		rtvi.TypeBotInterrupted:      EventBotInterrupted,
	} {
		wantEvent(t, s.FrameToEvent(messageFrame(t, msgType, nil)), map[string]any{keyType: event})
	}
}

// The harness's own VAD, speaking and interruption frames are internal plumbing,
// not scenario events.
func TestComputedVADAndSpeakingFramesAreIgnored(t *testing.T) {
	s := newStream(true)
	for _, f := range []frames.Frame{
		frames.NewVADUserStartedSpeakingFrame(0, time.Now()), frames.NewVADUserStoppedSpeakingFrame(0, time.Now()),
		frames.NewUserStartedSpeakingFrame(), frames.NewUserStoppedSpeakingFrame(),
		frames.NewInterruptionFrame(),
	} {
		wantNoEvent(t, s.FrameToEvent(f))
	}
}

// A new user turn drops the bot's leftover output, but keeps a queued
// user_transcription, which is the turn's input.
func TestUserStartedSpeakingMessageDiscardsInterruptedOutput(t *testing.T) {
	s := newStream(true)
	s.textBuffer = []string{"greeting"}
	s.queue = []map[string]any{
		{keyType: EventResponse, "text": "greeting"},
		{keyType: EventUserTranscription, "transcript": "hi"},
	}
	wantEvent(t, s.FrameToEvent(messageFrame(t, rtvi.TypeUserStartedSpeaking, nil)),
		map[string]any{keyType: EventUserStartedSpeaking})
	if len(s.textBuffer) != 0 {
		t.Fatalf("text buffer = %v, want it dropped", s.textBuffer)
	}
	if got := queued(s); !reflect.DeepEqual(got, []map[string]any{{keyType: EventUserTranscription, "transcript": "hi"}}) {
		t.Fatalf("queue = %v, want only the transcription", got)
	}
}

func TestFunctionCallEvent(t *testing.T) {
	s := newStream(false)
	f := frames.NewFunctionCallInProgressFrame("c", "get_weather", json.RawMessage(`{"city":"Paris"}`), false, "")
	wantEvent(t, s.FrameToEvent(f),
		map[string]any{keyType: EventFunctionCall, "name": "get_weather", "args": map[string]any{"city": "Paris"}})
}

func TestTTSTextOnlyInAudioMode(t *testing.T) {
	wantNoEvent(t, newStream(false).FrameToEvent(frames.NewTTSTextFrame("x", frames.TextTypeSentence)))
	wantEvent(t, newStream(true).FrameToEvent(frames.NewTTSTextFrame("spoken", frames.TextTypeSentence)),
		map[string]any{keyType: EventTTSResponse, "text": "spoken"})
}

// An interrupted response, with no text, still emits an empty llm_response; the
// matcher's aggregation decides whether that passes.
func TestEmptyResponseStillEmitted(t *testing.T) {
	s := newStream(false)
	wantEvent(t, s.FrameToEvent(frames.NewLLMFullResponseStartFrame()), map[string]any{keyType: EventLLMStarted})
	wantEvent(t, s.FrameToEvent(frames.NewLLMFullResponseEndFrame()),
		map[string]any{keyType: EventLLMResponse, "text": ""})
}

func TestUnmappedMessageIgnored(t *testing.T) {
	wantNoEvent(t, newStream(false).FrameToEvent(messageFrame(t, rtvi.TypeBotReady, map[string]any{})))
}

// metricsEvent is a metrics event reporting kinds.
func metricsEvent(kinds map[string]any) map[string]any {
	return map[string]any{keyType: EventMetrics, keyData: kinds}
}

// measurement is one entry of a metrics report.
func measurement(processor string, value float64) []any {
	return []any{map[string]any{"processor": processor, "value": value}}
}

// The bot's metrics reach the harness as sent, kept apart from its other
// events.
func TestMetricsEvent(t *testing.T) {
	t.Run("a metrics message becomes an event with its data", func(t *testing.T) {
		data := map[string]any{"ttfb": measurement("DeepgramSTTService#0", 0.31)}
		event := newStream(false).FrameToEvent(messageFrame(t, rtvi.TypeMetrics, data))
		wantEvent(t, event, metricsEvent(data))
	})
	t.Run("metrics are recorded apart from the events seen", func(t *testing.T) {
		s := newStream(false)
		s.Append(map[string]any{keyType: EventLLMStarted})
		s.Append(metricsEvent(map[string]any{"ttfb": measurement("LLM#0", 0.2)}))
		seen := s.EventsSeen()
		if len(seen) != 1 || seen[0][keyType] != EventLLMStarted {
			t.Fatalf("events seen = %v, want the llm_started alone", seen)
		}
		metrics := s.Metrics()
		if len(metrics) != 1 || metrics[0][keyType] != EventMetrics {
			t.Fatalf("metrics = %v, want the report", metrics)
		}
		if _, ok := metrics[0]["at"]; !ok {
			t.Fatal("the report has no time")
		}
	})
	t.Run("the result carries every report with its time", func(t *testing.T) {
		stream := newStream(false)
		driver := &ScriptDriver{scenario: &ScriptScenario{Name: "t"}}
		driver.stream, driver.trace = stream, NewTrace()
		data := map[string]any{"ttfb": measurement("STT#0", 0.3)}
		stream.Append(map[string]any{keyType: EventMetrics, keyData: data, "at": 1.5})
		result := driver.result(nil, 0, nil, nil, "")
		want := []BotMetricsReport{{At: 1.5, Data: data}}
		if !reflect.DeepEqual(result.BotMetrics, want) {
			t.Fatalf("bot metrics = %v, want %v", result.BotMetrics, want)
		}
	})
}

// A message of the bot's own making is an event a scenario can read.
func TestServerMessageEvent(t *testing.T) {
	t.Run("it becomes an event with its data and its text", func(t *testing.T) {
		data := map[string]any{"done": true, "transcript": "héllo"}
		event := newStream(false).FrameToEvent(messageFrame(t, rtvi.TypeServerMessage, data))
		wantEvent(t, event, map[string]any{
			keyType: EventServerMessage, keyData: data, keyText: `{"done": true, "transcript": "héllo"}`,
		})
	})
	t.Run("its text keeps the keys in the order the bot sent them", func(t *testing.T) {
		raw := []byte(`{"label":"rtvi-ai","type":"server-message","data":{"z":[1,2.5,null],"a":{"b":"x\"y"}}}`)
		event := newStream(false).FrameToEvent(frames.NewInputTransportMessageFrame(raw))
		if got, want := event[keyText], `{"z": [1, 2.5, null], "a": {"b": "x\"y"}}`; got != want {
			t.Fatalf("text = %v, want %v", got, want)
		}
	})
	t.Run("string data is its own text", func(t *testing.T) {
		event := newStream(false).FrameToEvent(messageFrame(t, rtvi.TypeServerMessage, "finished"))
		if event[keyData] != "finished" || event[keyText] != "finished" {
			t.Fatalf("event = %v, want the string as data and text", event)
		}
	})
}

// An image the bot reports becomes an image event.
func TestBotImageBecomesImage(t *testing.T) {
	event := newStream(false).FrameToEvent(messageFrame(t, evalws.BotImageType,
		map[string]any{"width": 64, "height": 32, "format": "RGB"}))
	wantEvent(t, event, map[string]any{keyType: EventImage, "width": 64.0, "height": 32.0, "format": "RGB"})
}

// A call made outside the pipeline and reported into it is an event like the
// pipeline's own.
func TestAnExternalCallStartsAndStopsLikeThePipelinesOwn(t *testing.T) {
	s := newStream(false)
	started := s.FrameToEvent(frames.NewExternalFunctionCallInProgressFrame("toolu_1", "run_tests",
		json.RawMessage(`{"path":"x"}`)))
	interim := frames.NewExternalFunctionCallResultFrame("toolu_1", "run_tests", json.RawMessage(`{}`), "1 of 3")
	interim.IsFinal = false
	progress := s.FrameToEvent(interim)
	canceled := s.FrameToEvent(frames.NewExternalFunctionCallCancelFrame("toolu_1", "run_tests"))
	done := s.FrameToEvent(frames.NewExternalFunctionCallResultFrame("toolu_2", "run_tests", json.RawMessage(`{}`), "ok"))

	wantEvent(t, started,
		map[string]any{keyType: EventFunctionCall, "name": "run_tests", "args": map[string]any{"path": "x"}})
	wantNoEvent(t, progress)
	wantEvent(t, canceled, map[string]any{
		keyType: EventFunctionCallStopped, "name": "run_tests",
		"args": map[string]any{"tool_call_id": "toolu_1", keyCanceled: true},
	})
	if want := map[string]any{"tool_call_id": "toolu_2", keyCanceled: false}; !reflect.DeepEqual(done["args"], want) {
		t.Fatalf("args = %v, want %v", done["args"], want)
	}
}

// The bot's speaking frames become events, and a wait for it to go quiet.
func TestTheBotsSpeakingFramesBecomeEventsAndAWait(t *testing.T) {
	ctx := t.Context()
	s := newStream(false)
	if s.BotSpeaking() || !s.WaitBotQuiet(ctx, 10*time.Millisecond) {
		t.Fatal("a fresh stream's bot is not quiet")
	}
	wantEvent(t, s.FrameToEvent(frames.NewBotStartedSpeakingFrame()), map[string]any{keyType: EventBotStartedSpeaking})
	if !s.BotSpeaking() || s.WaitBotQuiet(ctx, 50*time.Millisecond) {
		t.Fatal("the bot is not speaking after it started")
	}
	wantEvent(t, s.FrameToEvent(frames.NewBotStoppedSpeakingFrame()), map[string]any{keyType: EventBotStoppedSpeaking})
	if s.BotSpeaking() || !s.WaitBotQuiet(ctx, 10*time.Millisecond) {
		t.Fatal("the bot is speaking after it stopped")
	}
}

func TestAnInterruptionCountsAsTheBotGoingQuiet(t *testing.T) {
	s := newStream(false)
	s.FrameToEvent(frames.NewBotStartedSpeakingFrame())
	s.mu.Lock()
	s.interruptedLocked()
	s.mu.Unlock()
	if s.BotSpeaking() {
		t.Fatal("the bot is speaking after an interruption")
	}
}

// texts takes every queued event's text.
func texts(s *EventStream) []string {
	var out []string
	for _, event := range queued(s) {
		text, _ := event["text"].(string)
		out = append(out, text)
	}
	return out
}

// The bot's finished spoken turn is the reply only once its LLM has answered the
// input.
func TestBotTurn(t *testing.T) {
	t.Run("a turn after the input and the LLM restarting is the reply", func(t *testing.T) {
		s := newStream(true)
		s.InputSent()
		s.FrameToEvent(frames.NewLLMFullResponseStartFrame()) // the bot's new response
		s.BotTurnStopped("Tokyo.")
		if got := texts(s); !reflect.DeepEqual(got, []string{"Tokyo."}) {
			t.Fatalf("responses = %v", got)
		}
	})
	t.Run("a turn before the LLM restarts is dropped", func(t *testing.T) {
		s := newStream(true)
		s.InputSent()
		s.BotTurnStopped("straggler")
		if got := texts(s); len(got) != 0 {
			t.Fatalf("responses = %v, want none", got)
		}
	})
	t.Run("a bot-first turn needs no input", func(t *testing.T) {
		s := newStream(true)
		s.BotTurnStopped("Hello there!")
		if got := texts(s); !reflect.DeepEqual(got, []string{"Hello there!"}) {
			t.Fatalf("responses = %v", got)
		}
	})
	t.Run("an empty turn is not a response", func(t *testing.T) {
		s := newStream(true)
		s.BotTurnStopped("")
		if got := texts(s); len(got) != 0 {
			t.Fatalf("responses = %v, want none", got)
		}
	})
}

// The bot's speech is accounted for once nothing awaits transcription and no
// turn is open.
func TestSegmentsAndTurnsHoldTheWait(t *testing.T) {
	ctx := t.Context()
	s := newStream(true)
	if !s.WaitBotTranscribed(ctx, 10*time.Millisecond) {
		t.Fatal("a fresh stream's bot speech is not accounted for")
	}
	// A segment the VAD still hears is speech not yet accounted for.
	s.BotSegmentStarted()
	if s.WaitBotTranscribed(ctx, 10*time.Millisecond) {
		t.Fatal("a segment being heard is accounted for")
	}
	s.BotSegmentEnded()
	if s.WaitBotTranscribed(ctx, 10*time.Millisecond) {
		t.Fatal("a segment awaiting transcription is accounted for")
	}
	s.BotTurnStarted()
	s.BotSegmentTranscribed()
	// The transcript is in, but the turn it feeds is still open.
	if s.WaitBotTranscribed(ctx, 10*time.Millisecond) {
		t.Fatal("an open turn is accounted for")
	}
	s.BotTurnStopped("Hello there!")
	if !s.WaitBotTranscribed(ctx, 10*time.Millisecond) {
		t.Fatal("the closed turn is not accounted for")
	}
}

// Once the bot ends the call, a wait for its next event fails at once, after
// anything still on its way.
func TestBotEndedStream(t *testing.T) {
	ctx := t.Context()
	t.Run("a wait fails at once once the bot ended the call", func(t *testing.T) {
		s := newStream(false)
		s.FrameToEvent(frames.NewBotStartedSpeakingFrame())
		s.BotEnded()
		s.Append(map[string]any{keyType: EventBotEnded})
		if s.BotSpeaking() {
			t.Fatal("the bot is speaking after it ended the call")
		}
		started := time.Now()
		_, err := s.NextEvent(ctx, EventLLMResponse, time.Now().Add(5*time.Second), FromNow, nil)
		if !errors.Is(err, ErrBotEnded) {
			t.Fatalf("err = %v, want ErrBotEnded", err)
		}
		if time.Since(started) > time.Second {
			t.Fatal("the wait did not fail at once")
		}
	})
	t.Run("an event queued before the bot ended still matches", func(t *testing.T) {
		s := newStream(false)
		s.Append(map[string]any{keyType: EventLLMResponse, "text": "Goodbye!"})
		s.BotEnded()
		s.Append(map[string]any{keyType: EventBotEnded})
		event, err := s.NextEvent(ctx, EventLLMResponse, time.Now().Add(5*time.Second), FromNow, nil)
		if err != nil || event["text"] != "Goodbye!" {
			t.Fatalf("event = %v, %v", event, err)
		}
	})
	t.Run("a transcript landing after the bot ended still matches", func(t *testing.T) {
		s := newStream(true)
		s.BotTurnStarted()
		s.BotEnded()
		got := make(chan map[string]any, 1)
		go func() {
			event, _ := s.NextEvent(ctx, EventResponse, time.Now().Add(5*time.Second), FromNow, nil)
			got <- event
		}()
		select {
		case <-got:
			t.Fatal("the wait ended before the transcript landed")
		case <-time.After(50 * time.Millisecond):
		}
		s.BotTurnStopped("Goodbye!")
		if event := <-got; event["text"] != "Goodbye!" {
			t.Fatalf("event = %v", event)
		}
	})
	t.Run("dropping the queued bot_ended event keeps the call ended", func(t *testing.T) {
		s := newStream(false)
		s.BotEnded()
		s.Append(map[string]any{keyType: EventBotEnded})
		s.DropPendingBotOutput("before send")
		_, err := s.NextEvent(ctx, EventLLMResponse, time.Now().Add(5*time.Second), FromNow, nil)
		if !errors.Is(err, ErrBotEnded) {
			t.Fatalf("err = %v, want ErrBotEnded", err)
		}
	})
}

// A send drops the bot's queued output, a queued harness_error with it; the run
// stays failed.
func TestDroppingTheQueuedHarnessErrorKeepsTheRunFailed(t *testing.T) {
	s := newStream(false)
	s.HarnessFailed("STT failed")
	s.Append(map[string]any{keyType: EventHarnessError, "text": "STT failed"})
	s.DropPendingBotOutput("before send")
	_, err := s.NextEvent(context.Background(), EventLLMResponse, time.Now().Add(5*time.Second), FromNow, nil)
	var harnessErr *HarnessError
	if !errors.As(err, &harnessErr) || harnessErr.Reason != "STT failed" {
		t.Fatalf("err = %v, want the harness error", err)
	}
}

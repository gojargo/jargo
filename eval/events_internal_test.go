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

// rawMessage is one of the bot's RTVI messages, as it arrives. Nil data leaves
// the message without any.
func rawMessage(t *testing.T, msgType string, data any) []byte {
	t.Helper()
	message := map[string]any{"label": rtvi.MessageLabel, "type": msgType}
	if data != nil {
		message["data"] = data
	}
	raw, err := json.Marshal(message)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

// messageFrame is one of the bot's RTVI messages, as the frame the client
// receives it in: a raw transport message.
func messageFrame(t *testing.T, msgType string, data any) *frames.InputTransportMessageFrame {
	t.Helper()
	return frames.NewInputTransportMessageFrame(rawMessage(t, msgType, data))
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
	s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMStarted, nil))
	s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMText, map[string]any{"text": "Hello "}))
	s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMText, map[string]any{"text": "world"}))
	event := s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMStopped, nil))
	startedAt, ok := event["started_at"].(float64)
	if !ok || startedAt > s.Elapsed() {
		t.Fatalf("started_at = %v, want the time of the first token", event["started_at"])
	}
	s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMStarted, nil))
	if event := s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMStopped, nil)); event["started_at"] != nil {
		t.Fatalf("a response with no text carries a time: %v", event)
	}
}

func TestLLMMarkerIsItsOwnEvent(t *testing.T) {
	s := newStream(false)
	s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMStarted, nil))
	s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMText, map[string]any{"text": "Hello"}))
	markers := []string{"●", "◐", "○"}
	wantEvent(t, s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMMarker,
		map[string]any{"raw": "● Hello", "text": "●", "kind": "complete", "markers": markers})),
		map[string]any{keyType: EventLLMMarker, "text": "●", "kind": "complete", "raw": "● Hello", "markers": markers})
	// The marker is not part of the reply's text.
	wantEvent(t, s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMStopped, nil)),
		map[string]any{keyType: EventLLMResponse, "text": "Hello"})
}

func TestLLMMarkerBeforeTheReplyStartsIsDropped(t *testing.T) {
	s := newStream(false)
	s.InputSent()
	wantNoEvent(t, s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMMarker, map[string]any{"raw": "◐", "text": "◐"})))
	s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMStarted, nil))
	wantEvent(t, s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMMarker,
		map[string]any{"raw": "◐", "text": "◐", "kind": "short"})),
		map[string]any{keyType: EventLLMMarker, "text": "◐", "kind": "short", "raw": "◐", "markers": []string{}})
}

func TestLLMLifecycleAggregatesText(t *testing.T) {
	s := newStream(false)
	wantEvent(t, s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMStarted, nil)),
		map[string]any{keyType: EventLLMStarted})
	wantNoEvent(t, s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMText, map[string]any{"text": "Hello "})))
	wantNoEvent(t, s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMText, map[string]any{"text": "world"})))
	wantEvent(t, s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMStopped, nil)),
		map[string]any{keyType: EventLLMResponse, "text": "Hello world"})
}

func TestInterruptionSuppressesStraggler(t *testing.T) {
	s := newStream(false)
	s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMStarted, nil))
	s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMText, map[string]any{"text": "Tell me about Paris"}))
	wantEvent(t, s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotInterrupted, nil)),
		map[string]any{keyType: EventBotInterrupted})
	// Straggler from the interrupted response is dropped.
	wantNoEvent(t, s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMText, map[string]any{"text": " what would"})))
	wantNoEvent(t, s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMStopped, nil)))
	// The genuinely new response.
	s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMStarted, nil))
	s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMText, map[string]any{"text": "Tokyo"}))
	wantEvent(t, s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMStopped, nil)),
		map[string]any{keyType: EventLLMResponse, "text": "Tokyo"})
}

// The bot's reported user-input arrives as a raw message, not a
// TranscriptionFrame, which the harness keeps for the bot's own response.
func TestUserTranscriptionFromMessage(t *testing.T) {
	s := newStream(false)
	input := func(data map[string]any) map[string]any {
		return s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeUserInput, data))
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
		return s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotOutput, map[string]any{
			"text": "Mm-hmm.", "output_type": outputType, "spoken_status": status,
			"spoken_progress": map[string]any{"accumulated_text": accumulated, "remaining_text": ""},
		}))
	}
	wantNoEvent(t, output("backchannel", "new", ""))
	wantEvent(t, output("backchannel", "completed", "Mm-hmm."),
		map[string]any{keyType: EventBotBackchannel, "text": "Mm-hmm."})
	wantNoEvent(t, output("sentence", "completed", "Mm-hmm."))
	wantEvent(t, s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeUserInput,
		map[string]any{"text": "mhm", "input_type": "backchannel"})),
		map[string]any{keyType: EventUserBackchannel, "text": "mhm"})
	wantNoEvent(t, s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeUserInput,
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
		event := s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotOutput, decoded))
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

	s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMStarted, nil))
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
	s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMStarted, nil))
	want(words(9, []string{}, rtvi.SpokenInProgress), "")
}

// A bot that turned raw text on sends it alongside the bot-llm-text and
// bot-tts-text the stream reads, so its text is not counted twice.
func TestRawTextMessagesAreNoEvents(t *testing.T) {
	s := newStream(true)
	s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMStarted, nil))
	wantNoEvent(t, s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeLLMRawText, map[string]any{"text": "Hello"})))
	s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMText, map[string]any{"text": "Hello"}))
	wantEvent(t, s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMStopped, nil)),
		map[string]any{keyType: EventLLMResponse, "text": "Hello"})
	wantNoEvent(t, s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeTTSRawText, map[string]any{"text": "Hello"})))
}

// user-input carries the same transcriptions.
func TestTheDeprecatedUserTranscriptionIsNoEvent(t *testing.T) {
	wantNoEvent(t, newStream(false).RTVIMessageToEvent(
		rawMessage(t, rtvi.TypeUserTranscription, //nolint:staticcheck // sent
			map[string]any{"text": "hello", "final": true})))
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
		wantEvent(t, s.RTVIMessageToEvent(rawMessage(t, msgType, nil)), map[string]any{keyType: event})
	}
}

// A new user turn drops the bot's response in progress, and what the bot still
// says of it is not its reply. The queue is left as it is.
func TestUserStartedSpeakingMessageDiscardsTheUnfinishedResponse(t *testing.T) {
	s := newStream(true)
	s.textBuffer = []string{"greeting"}
	s.queue = []map[string]any{{keyType: EventUserTranscription, "transcript": "hi"}}
	wantEvent(t, s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeUserStartedSpeaking, nil)),
		map[string]any{keyType: EventUserStartedSpeaking})
	if len(s.textBuffer) != 0 {
		t.Fatalf("text buffer = %v, want it dropped", s.textBuffer)
	}
	wantNoEvent(t, s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMText, map[string]any{"text": " and welcome"})))
	wantNoEvent(t, s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMStopped, nil)))
	if got := queued(s); !reflect.DeepEqual(got, []map[string]any{{keyType: EventUserTranscription, "transcript": "hi"}}) {
		t.Fatalf("queue = %v, want only the transcription", got)
	}
}

// The bot's reports of the user's turn starting survive the interruption sent
// right behind them.
func TestInterruptionKeepsTheUsersTurn(t *testing.T) {
	// expect feeds the bot's messages as the sink does, back to back, to a turn
	// waiting for event.
	expect := func(t *testing.T, messages []string, event string) map[string]any {
		t.Helper()
		s := newStream(true)
		s.InputSent()
		type outcome struct {
			event map[string]any
			err   error
		}
		waiting := make(chan outcome, 1)
		go func() {
			matched, err := s.NextEvent(t.Context(), event, time.Now().Add(time.Second), FromNow, nil)
			waiting <- outcome{matched, err}
		}()
		for _, msgType := range messages {
			if reported := s.RTVIMessageToEvent(rawMessage(t, msgType, nil)); reported != nil {
				s.Append(reported)
			}
		}
		got := <-waiting
		if got.err != nil {
			t.Fatalf("waiting for %s: %v", event, got.err)
		}
		return got.event
	}
	t.Run("user_started_speaking survives the interruption behind it", func(t *testing.T) {
		matched := expect(t, []string{rtvi.TypeUserStartedSpeaking, rtvi.TypeBotInterrupted}, EventUserStartedSpeaking)
		if matched[keyType] != EventUserStartedSpeaking {
			t.Fatalf("matched %v", matched)
		}
	})
	t.Run("vad_user_started_speaking survives both reports behind it", func(t *testing.T) {
		matched := expect(t,
			[]string{rtvi.TypeVADUserStarted, rtvi.TypeUserStartedSpeaking, rtvi.TypeBotInterrupted},
			EventVADUserStartedSpeaking)
		if matched[keyType] != EventVADUserStartedSpeaking {
			t.Fatalf("matched %v", matched)
		}
	})
	t.Run("bot_interrupted still matches", func(t *testing.T) {
		matched := expect(t, []string{rtvi.TypeUserStartedSpeaking, rtvi.TypeBotInterrupted}, EventBotInterrupted)
		if matched[keyType] != EventBotInterrupted {
			t.Fatalf("matched %v", matched)
		}
	})
	// Left queued, they would match the next turn's expectation.
	t.Run("a send still drops the reports of the turn before", func(t *testing.T) {
		s := newStream(true)
		for _, kind := range []string{EventVADUserStartedSpeaking, EventUserStartedSpeaking, EventBotInterrupted} {
			s.queue = append(s.queue, map[string]any{keyType: kind})
		}
		s.DropPendingBotOutput("before send")
		if got := queued(s); len(got) != 0 {
			t.Fatalf("queue = %v, want it empty", got)
		}
	})
}

func TestFunctionCallEvent(t *testing.T) {
	s := newStream(false)
	message := rawMessage(t, rtvi.TypeLLMFunctionCall, map[string]any{
		"function_name": "get_weather", "tool_call_id": "c", "arguments": map[string]any{"city": "Paris"},
	})
	wantEvent(t, s.RTVIMessageToEvent(message),
		map[string]any{keyType: EventFunctionCall, "name": "get_weather", "args": map[string]any{"city": "Paris"}})
}

func TestTTSTextOnlyInAudioMode(t *testing.T) {
	tts := func(text string) []byte { return rawMessage(t, rtvi.TypeBotTTSText, map[string]any{"text": text}) }
	wantNoEvent(t, newStream(false).RTVIMessageToEvent(tts("x")))
	wantEvent(t, newStream(true).RTVIMessageToEvent(tts("spoken")),
		map[string]any{keyType: EventTTSResponse, "text": "spoken"})
}

// An interrupted response, with no text, still emits an empty llm_response; the
// matcher's aggregation decides whether that passes.
func TestEmptyResponseStillEmitted(t *testing.T) {
	s := newStream(false)
	wantEvent(t, s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMStarted, nil)),
		map[string]any{keyType: EventLLMStarted})
	wantEvent(t, s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMStopped, nil)),
		map[string]any{keyType: EventLLMResponse, "text": ""})
}

func TestUnmappedMessageIgnored(t *testing.T) {
	wantNoEvent(t, newStream(false).RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotReady, map[string]any{})))
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
		event := newStream(false).RTVIMessageToEvent(rawMessage(t, rtvi.TypeMetrics, data))
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

// The bot's flow entering a node is an event named by the node.
func TestFlowNodeEvent(t *testing.T) {
	t.Run("a node entered becomes an event with its data and the node as text", func(t *testing.T) {
		data := map[string]any{"node": "verify.ask", "previous_node": "start", "parent": "verify"}
		event := newStream(false).RTVIMessageToEvent(rawMessage(t, rtvi.TypeFlowNodeEntered, data))
		wantEvent(t, event, map[string]any{keyType: EventFlowNode, keyData: data, keyText: "verify.ask"})
	})
	t.Run("an expectation matches the node", func(t *testing.T) {
		m := newMatcher(nil, false)
		m.stream.Append(m.stream.RTVIMessageToEvent(
			rawMessage(t, rtvi.TypeFlowNodeEntered, map[string]any{"node": "verify.ask"})))
		exp := Expectation{Event: EventFlowNode, TextMatches: []string{"^verify[.]"}}
		if f := mustMatch(t, m, exp, 1000); f != nil {
			t.Fatalf("failure = %v", f)
		}
		if m.LastMatchText != "verify.ask" {
			t.Errorf("recorded %q, want the node's name", m.LastMatchText)
		}
	})
}

// A message of the bot's own making is an event a scenario can read.
func TestServerMessageEvent(t *testing.T) {
	t.Run("it becomes an event with its data and its text", func(t *testing.T) {
		data := map[string]any{"done": true, "transcript": "héllo"}
		event := newStream(false).RTVIMessageToEvent(rawMessage(t, rtvi.TypeServerMessage, data))
		wantEvent(t, event, map[string]any{
			keyType: EventServerMessage, keyData: data, keyText: `{"done": true, "transcript": "héllo"}`,
		})
	})
	t.Run("its text keeps the keys in the order the bot sent them", func(t *testing.T) {
		raw := []byte(`{"label":"rtvi-ai","type":"server-message","data":{"z":[1,2.5,null],"a":{"b":"x\"y"}}}`)
		event := newStream(false).RTVIMessageToEvent(raw)
		if got, want := event[keyText], `{"z": [1, 2.5, null], "a": {"b": "x\"y"}}`; got != want {
			t.Fatalf("text = %v, want %v", got, want)
		}
	})
	t.Run("string data is its own text", func(t *testing.T) {
		event := newStream(false).RTVIMessageToEvent(rawMessage(t, rtvi.TypeServerMessage, "finished"))
		if event[keyData] != "finished" || event[keyText] != "finished" {
			t.Fatalf("event = %v, want the string as data and text", event)
		}
	})
}

// An image the bot reports becomes an image event.
func TestBotImageBecomesImage(t *testing.T) {
	event := newStream(false).RTVIMessageToEvent(rawMessage(t, evalws.BotImageType,
		map[string]any{"width": 64, "height": 32, "format": "RGB"}))
	wantEvent(t, event, map[string]any{keyType: EventImage, "width": 64.0, "height": 32.0, "format": "RGB"})
}

// A call's end carries its id and whether it was canceled in its args, and its
// name only when the bot reported one.
func TestFunctionCallStoppedEvent(t *testing.T) {
	s := newStream(false)
	canceled := s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeLLMFunctionCallStop, map[string]any{
		"function_name": "run_tests", "tool_call_id": "toolu_1", keyCanceled: true,
	}))
	wantEvent(t, canceled, map[string]any{
		keyType: EventFunctionCallStopped, "name": "run_tests",
		"args": map[string]any{"tool_call_id": "toolu_1", keyCanceled: true},
	})
	// At the bot's default report level only the id is reported.
	done := s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeLLMFunctionCallStop, map[string]any{"tool_call_id": "toolu_2"}))
	wantEvent(t, done, map[string]any{
		keyType: EventFunctionCallStopped, "name": nil,
		"args": map[string]any{"tool_call_id": "toolu_2", keyCanceled: false},
	})
	started := s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeLLMFunctionCall, map[string]any{"tool_call_id": "toolu_3"}))
	wantEvent(t, started, map[string]any{keyType: EventFunctionCall, "name": nil, "args": map[string]any{}})
}

// The bot's speaking reports become events, and a wait for it to go quiet.
func TestTheBotsSpeakingReportsBecomeEventsAndAWait(t *testing.T) {
	ctx := t.Context()
	s := newStream(false)
	if s.BotSpeaking() || !s.WaitBotQuiet(ctx, 10*time.Millisecond) {
		t.Fatal("a fresh stream's bot is not quiet")
	}
	wantEvent(t, s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotStartedSpeaking, nil)),
		map[string]any{keyType: EventBotStartedSpeaking})
	if !s.BotSpeaking() || s.WaitBotQuiet(ctx, 50*time.Millisecond) {
		t.Fatal("the bot is not speaking after it started")
	}
	wantEvent(t, s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotStoppedSpeaking, nil)),
		map[string]any{keyType: EventBotStoppedSpeaking})
	if s.BotSpeaking() || !s.WaitBotQuiet(ctx, 10*time.Millisecond) {
		t.Fatal("the bot is speaking after it stopped")
	}
}

func TestAnInterruptionCountsAsTheBotGoingQuiet(t *testing.T) {
	s := newStream(false)
	s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotStartedSpeaking, nil))
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
		s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotLLMStarted, nil)) // the bot's new response
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
		s.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotStartedSpeaking, nil))
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

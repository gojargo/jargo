package eval

import (
	"context"
	"encoding/json"
	"maps"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
)

// clientFor is the client of a session of scenario against botURL.
func clientFor(t *testing.T, scenario *ScriptScenario, botURL string, params SessionParams) *Client {
	t.Helper()
	s, err := NewScriptSession(scenario, botURL, params, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	return s.client
}

// The harness signals the connection's flags on its URL, skip_tts in text mode.
func TestConnectURL(t *testing.T) {
	url := func(botAudio bool, base string) string {
		scenario := &ScriptScenario{Name: "t"}
		scenario.BotAudio = botAudio
		return clientFor(t, scenario, base, SessionParams{}).connectURL()
	}
	if got := url(false, "ws://localhost:7860"); got != "ws://localhost:7860?skip_tts=true" {
		t.Errorf("text mode = %q", got)
	}
	if got := url(true, "ws://localhost:7860"); got != "ws://localhost:7860" {
		t.Errorf("audio mode = %q", got)
	}
	if got := url(false, "ws://localhost:7860?x=1"); got != "ws://localhost:7860?x=1&skip_tts=true" {
		t.Errorf("an existing query = %q", got)
	}

	t.Run("a response expectation asks for the bot's audio", func(t *testing.T) {
		scenario := &ScriptScenario{Name: "t", Turns: []ScriptTurn{
			{User: "x", Expect: []Expectation{{Event: EventResponse, Eval: "ok"}}},
		}}
		scenario.BotAudio = true
		got := clientFor(t, scenario, "ws://localhost:7860", SessionParams{}).connectURL()
		if !strings.Contains(got, "capture_bot_audio=true") || strings.Contains(got, "skip_tts") {
			t.Fatalf("url = %q", got)
		}
	})
	t.Run("an image expectation asks for the bot's images", func(t *testing.T) {
		scenario := &ScriptScenario{Name: "t", Turns: []ScriptTurn{{Expect: []Expectation{{Event: EventImage}}}}}
		if got := clientFor(t, scenario, "ws://localhost:7860", SessionParams{}).connectURL(); !strings.Contains(got,
			"capture_bot_images=true") {
			t.Fatalf("url = %q", got)
		}
	})
	t.Run("the run's params reach the client", func(t *testing.T) {
		params := SessionParams{DefaultTimeoutMS: 1234}
		s := newSessionFor(t, &ScriptScenario{Name: "t"}, params)
		if s.script.defaultTimeoutMS != 1234 {
			t.Fatalf("timeout %d, url %q", s.script.defaultTimeoutMS, s.client.connectURL())
		}
	})
}

// The connection's flags go out first, in eval-connect, the same flags as the
// URL's.
func TestConnectionFlagsMatchTheURL(t *testing.T) {
	c := clientFor(t, &ScriptScenario{Name: "t"}, "ws://localhost:7860", SessionParams{})
	tr := c.newTransport()
	if got := tr.Flags(); got != (evalws.ConnectionFlags{SkipTTS: true}) {
		t.Fatalf("flags = %+v, want skip_tts only", got)
	}
}

// sentMessages captures what a client sends.
type sentMessages struct {
	mu   sync.Mutex
	sent []rtvi.Message
}

// capture runs a client with no pipeline, recording the messages it sends.
func capture(c *Client) *sentMessages {
	s := &sentMessages{}
	c.send = func(m rtvi.Message) {
		s.mu.Lock()
		s.sent = append(s.sent, m)
		s.mu.Unlock()
	}
	return s
}

// A dtmf turn sends one RTVI dtmf message with all its keys.
func TestDTMFSender(t *testing.T) {
	for keys, want := range map[string][]string{"12#": {"1", "2", "#"}, "1": {"1"}} {
		c := clientFor(t, &ScriptScenario{Name: "t"}, "ws://localhost:0", SessionParams{})
		sent := capture(c)
		c.SendDTMF(keys)
		if len(sent.sent) != 1 || sent.sent[0].Type != rtvi.TypeDTMF {
			t.Fatalf("%q: sent %+v, want one dtmf message", keys, sent.sent)
		}
		if got, _ := sent.sent[0].Data.(rtvi.DTMFData); !reflect.DeepEqual(got.Buttons, want) {
			t.Fatalf("%q: buttons = %v, want %v", keys, got, want)
		}
	}
}

// The client's messages are numbered in order, and a text turn asks for a
// spoken answer only in audio mode.
func TestMessagesAreNumberedAndTextTurnsSayHowToAnswer(t *testing.T) {
	c := clientFor(t, &ScriptScenario{Name: "t"}, "ws://localhost:0", SessionParams{})
	sent := capture(c)
	c.SendText("hello")
	c.SendDTMF("1")
	if len(sent.sent) != 2 || sent.sent[0].ID != "1" || sent.sent[1].ID != "2" {
		t.Fatalf("sent %+v, want ids 1 and 2", sent.sent)
	}
	data, _ := json.Marshal(sent.sent[0].Data)
	if want := `{"content":"hello","options":{"run_immediately":true,"audio_response":false}}`; string(data) != want {
		t.Fatalf("send-text data = %s, want %s", data, want)
	}
}

// Stopping the client ends the bot's session.
func TestStopEndsTheBotsSession(t *testing.T) {
	c := clientFor(t, &ScriptScenario{Name: "t"}, "ws://localhost:0", SessionParams{})
	sent := capture(c)
	c.Stop(t.Context())
	if len(sent.sent) != 1 || sent.sent[0].Type != rtvi.TypeClientMessage {
		t.Fatalf("sent %+v, want one eval-cancel", sent.sent)
	}
	if d, _ := sent.sent[0].Data.(map[string]any); d["t"] != evalws.CancelMessageType {
		t.Fatalf("sent %+v, want eval-cancel", d)
	}
}

// collector records what the processor before it pushes.
type collector struct {
	*processor.Base
	mu     sync.Mutex
	frames []frames.Frame
}

func newCollector() *collector {
	c := &collector{}
	c.Base = processor.New("Collector", c)
	return c
}

func (c *collector) QueueFrame(_ context.Context, f frames.Frame, _ processor.Direction) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.frames = append(c.frames, f)
	return nil
}

func (c *collector) seen() []frames.Frame {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]frames.Frame(nil), c.frames...)
}

// The sink is the boundary between the bot's side of the pipeline and the
// user's.
func TestBotFrameSink(t *testing.T) {
	ctx := t.Context()
	sinkWith := func(persona *Persona, personaHears bool) (*botFrameSink, *collector) {
		s := newBotFrameSink(newStream(true), persona, personaHears)
		if err := s.Setup(ctx, processor.Setup{}); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = s.Cleanup(context.WithoutCancel(ctx)) })
		next := newCollector()
		s.Link(next)
		return s, next
	}
	sink := func() (*botFrameSink, *collector) { return sinkWith(nil, false) }
	push := func(s *botFrameSink, fs ...frames.Frame) {
		for _, f := range fs {
			if err := s.ProcessFrame(ctx, f, processor.Downstream); err != nil {
				t.Fatal(err)
			}
		}
	}

	t.Run("inject pushes downstream", func(t *testing.T) {
		s, next := sink()
		f := frames.NewTTSSpeakFrame("hello")
		if err := s.inject(ctx, f); err != nil {
			t.Fatal(err)
		}
		if got := next.seen(); len(got) != 1 || got[0] != frames.Frame(f) {
			t.Fatalf("pushed %v", got)
		}
	})
	t.Run("frames that are not the bot's messages make no events", func(t *testing.T) {
		// The STT's transcriptions build the bot's turn in the aggregator, and
		// the aggregator's own VAD, speaking and interruption frames, computed
		// from the bot's audio, are internal plumbing: none of them is a
		// scenario event.
		s, _ := sink()
		push(s,
			frames.NewTranscriptionFrame("Paris", "bot", "t"),
			frames.NewVADUserStartedSpeakingFrame(0, time.Now()), frames.NewVADUserStoppedSpeakingFrame(0, time.Now()),
			frames.NewUserStartedSpeakingFrame(), frames.NewUserStoppedSpeakingFrame(),
			frames.NewInterruptionFrame())
		if got := s.stream.EventsSeen(); len(got) != 0 {
			t.Fatalf("events = %v, want none", got)
		}
	})
	t.Run("the bot's messages stop at the sink", func(t *testing.T) {
		s, next := sink()
		push(s, botResponse(t, "Hello")...)
		push(s,
			messageFrame(t, rtvi.TypeBotTTSText, map[string]any{"text": "Hello"}),
			messageFrame(t, rtvi.TypeBotStartedSpeaking, nil),
			messageFrame(t, rtvi.TypeLLMFunctionCall, map[string]any{"function_name": "f", "tool_call_id": "c"}),
			messageFrame(t, "x", nil))
		if got := next.seen(); len(got) != 0 {
			t.Fatalf("pushed %v", got)
		}
	})
	t.Run("a scripted run swallows the computed interruption", func(t *testing.T) {
		s, next := sink()
		push(s, frames.NewInterruptionFrame(), frames.NewLLMContextFrame(frames.NewLLMContext("")))
		if got := next.seen(); len(got) != 0 {
			t.Fatalf("pushed %v", got)
		}
	})
	t.Run("upstream frames are not the bot's", func(t *testing.T) {
		// The user side's own function call travels upstream through the sink;
		// it is neither an event nor held back.
		s, _ := sink()
		prev := newCollector()
		prev.Link(s)
		f := frames.NewFunctionCallInProgressFrame("c", "end_call", json.RawMessage(`{}`), false, "")
		if err := s.ProcessFrame(ctx, f, processor.Upstream); err != nil {
			t.Fatal(err)
		}
		if got := prev.seen(); len(got) != 1 || got[0] != frames.Frame(f) {
			t.Fatalf("pushed upstream %v", got)
		}
		if len(s.stream.EventsSeen()) != 0 {
			t.Fatalf("events = %v, want none", s.stream.EventsSeen())
		}
	})
	t.Run("a persona is interrupted and runs per context frame", func(t *testing.T) {
		persona := stubPersona()
		s, next := sinkWith(persona, false)
		interruption := frames.NewInterruptionFrame()
		run := frames.NewLLMContextFrame(persona.Context())
		push(s, interruption, run)
		if got := next.seen(); len(got) != 2 || got[0] != frames.Frame(interruption) || got[1] != frames.Frame(run) {
			t.Fatalf("pushed %v", got)
		}
	})
	t.Run("hang up silences the persona", func(t *testing.T) {
		persona := stubPersona()
		s, next := sinkWith(persona, true)
		persona.HangUp()
		push(s, frames.NewLLMContextFrame(persona.Context()))
		push(s, botResponse(t, "Anything else?")...)
		if got := next.seen(); len(got) != 0 {
			t.Fatalf("pushed %v", got)
		}
		if got := persona.Context().Messages(); len(got) != 0 {
			t.Fatalf("messages = %v", got)
		}
	})
	t.Run("text feed hands the bot's response to the persona", func(t *testing.T) {
		persona := stubPersona()
		s, next := sinkWith(persona, true)
		push(s, botResponse(t, "Hello ", "there")...)
		wantMessages(t, persona, "Hello there")
		got := next.seen()
		if len(got) != 1 {
			t.Fatalf("pushed %v", got)
		}
		if cf, ok := got[0].(*frames.LLMContextFrame); !ok || cf.Context != persona.Context() {
			t.Fatalf("pushed %v, want the persona's context", got[0])
		}
	})
	t.Run("text feed waits for the bot's function call", func(t *testing.T) {
		persona := stubPersona()
		s, next := sinkWith(persona, true)
		call := map[string]any{"function_name": "f", "tool_call_id": "c"}
		push(s,
			messageFrame(t, rtvi.TypeBotLLMStarted, nil),
			messageFrame(t, rtvi.TypeBotLLMText, map[string]any{"text": "Let me check."}),
			messageFrame(t, rtvi.TypeLLMFunctionCall, call),
			messageFrame(t, rtvi.TypeBotLLMStopped, nil))
		// Held: the call is still running.
		wantMessages(t, persona)
		push(s, messageFrame(t, rtvi.TypeLLMFunctionCallStop, call))
		push(s, botResponse(t, "It's sunny.")...)
		wantMessages(t, persona, "Let me check. It's sunny.")
		if got := next.seen(); len(got) != 1 {
			t.Fatalf("pushed %v", got)
		}
	})
	t.Run("text feed ignores an empty response", func(t *testing.T) {
		persona := stubPersona()
		s, next := sinkWith(persona, true)
		push(s, botResponse(t)...)
		wantMessages(t, persona)
		if got := next.seen(); len(got) != 0 {
			t.Fatalf("pushed %v", got)
		}
	})
}

// stubPersonaLLM is a persona LLM that only takes registrations: the sink and
// relay tests need only the persona's context and its hearing.
type stubPersonaLLM struct {
	*processor.Base
	mu       sync.Mutex
	handlers map[string]llm.FunctionCallHandler
}

func newStubPersonaLLM() *stubPersonaLLM {
	s := &stubPersonaLLM{handlers: map[string]llm.FunctionCallHandler{}}
	s.Base = processor.New("StubPersonaLLM", s)
	return s
}

func (s *stubPersonaLLM) RegisterFunction(name string, h llm.FunctionCallHandler, _ ...llm.RegisterOption) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.handlers[name] = h
}

func (s *stubPersonaLLM) handler(name string) llm.FunctionCallHandler {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.handlers[name]
}

func stubPersona() *Persona { return NewPersona("A caller.", "A goal.", newStubPersonaLLM()) }

// botResponse is one of the bot's LLM responses, as the raw reports the
// serializer keeps them as.
func botResponse(t *testing.T, texts ...string) []frames.Frame {
	t.Helper()
	out := []frames.Frame{messageFrame(t, rtvi.TypeBotLLMStarted, nil)}
	for _, text := range texts {
		out = append(out, messageFrame(t, rtvi.TypeBotLLMText, map[string]any{"text": text}))
	}
	return append(out, messageFrame(t, rtvi.TypeBotLLMStopped, nil))
}

// personaResponse is the frames of one LLM response, as the persona's LLM
// emits them.
func personaResponse(skipTTS bool, texts ...string) []frames.Frame {
	start, end := frames.NewLLMFullResponseStartFrame(), frames.NewLLMFullResponseEndFrame()
	start.SkipTTS, end.SkipTTS = &skipTTS, &skipTTS
	out := []frames.Frame{start}
	for _, text := range texts {
		f := frames.NewLLMTextFrame(text)
		f.SkipTTS = &skipTTS
		out = append(out, f)
	}
	return append(out, end)
}

// wantMessages asserts the persona's context holds one user message per text.
func wantMessages(t *testing.T, persona *Persona, texts ...string) {
	t.Helper()
	got := persona.Context().Messages()
	if len(got) != len(texts) {
		t.Fatalf("messages = %+v, want %q", got, texts)
	}
	for i, text := range texts {
		if m := got[i]; m.Role != frames.RoleUser || m.Text != text {
			t.Fatalf("message %d = %+v, want the user saying %q", i, m, text)
		}
	}
}

// untimed are the events without the arrival time the stream stamps on them.
func untimed(events []map[string]any) []map[string]any {
	out := make([]map[string]any, 0, len(events))
	for _, e := range events {
		c := maps.Clone(e)
		delete(c, "at")
		out = append(out, c)
	}
	return out
}

// The persona's text-mode responses become one send-text each.
func TestPersonaTurnRelay(t *testing.T) {
	ctx := t.Context()
	relayOf := func() (*personaTurnRelay, *collector, *Trace) {
		trace := NewTrace()
		r := newPersonaTurnRelay(func(text string) rtvi.Message {
			return rtvi.Message{Type: rtvi.TypeSendText, Data: text}
		}, trace, newStream(true))
		if err := r.Setup(ctx, processor.Setup{}); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = r.Cleanup(context.WithoutCancel(ctx)) })
		next := newCollector()
		r.Link(next)
		return r, next, trace
	}
	push := func(r *personaTurnRelay, fs ...frames.Frame) {
		for _, f := range fs {
			if err := r.ProcessFrame(ctx, f, processor.Downstream); err != nil {
				t.Fatal(err)
			}
		}
	}
	traced := func(trace *Trace, want string) bool {
		return slices.ContainsFunc(trace.Lines(), func(line string) bool { return strings.Contains(line, want) })
	}
	wantEvents := func(t *testing.T, r *personaTurnRelay, want ...map[string]any) {
		t.Helper()
		got := untimed(r.stream.EventsSeen())
		if len(want) == 0 && len(got) == 0 {
			return
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("events = %v, want %v", got, want)
		}
	}
	sentTexts := func(next *collector) []any {
		var sent []any
		for _, f := range next.seen() {
			if m, ok := f.(*frames.OutputTransportMessageUrgentFrame); ok {
				if message, ok := m.Message.(rtvi.Message); ok {
					sent = append(sent, message.Data)
				}
			}
		}
		return sent
	}

	t.Run("a skip_tts response is sent as one text turn", func(t *testing.T) {
		r, next, trace := relayOf()
		response := personaResponse(true, "Hi ", "there.")
		push(r, response...)
		if got := sentTexts(next); !reflect.DeepEqual(got, []any{"Hi there."}) {
			t.Fatalf("sent %v", got)
		}
		// The response's own frames go on, for the assistant aggregator.
		var passed []frames.Frame
		for _, f := range next.seen() {
			if slices.Contains(response, f) {
				passed = append(passed, f)
			}
		}
		if !slices.Equal(passed, response) {
			t.Fatalf("passed %v, want %v", passed, response)
		}
		if !traced(trace, `"Hi there." (persona, text)`) {
			t.Fatalf("trace = %v", trace.Lines())
		}
		wantEvents(t, r, map[string]any{"type": "persona_turn", "text": "Hi there."})
	})
	t.Run("a spoken response is only traced", func(t *testing.T) {
		r, next, trace := relayOf()
		// In audio mode the TTS speaks the text and forwards the end frame after
		// it.
		push(r, frames.NewLLMFullResponseStartFrame(), frames.NewLLMTextFrame("Hi"),
			frames.NewTTSTextFrame("Hi", frames.TextTypeSentence), frames.NewLLMFullResponseEndFrame())
		if got := sentTexts(next); len(got) != 0 {
			t.Fatalf("sent %v", got)
		}
		if !traced(trace, `"Hi" (persona, audio)`) {
			t.Fatalf("trace = %v", trace.Lines())
		}
		wantEvents(t, r, map[string]any{"type": "persona_turn", "text": "Hi"})
	})
	t.Run("a response without words is not a turn", func(t *testing.T) {
		// An end_call-only response, or one cut off before any word went out.
		r, _, _ := relayOf()
		push(r, frames.NewLLMFullResponseStartFrame(), frames.NewLLMFullResponseEndFrame())
		push(r, frames.NewLLMFullResponseStartFrame(), frames.NewInterruptionFrame(), frames.NewLLMFullResponseEndFrame())
		wantEvents(t, r)
	})
	t.Run("an end_call made while the response is spoken follows its turn", func(t *testing.T) {
		// In audio mode the persona calls end_call before the TTS has spoken the
		// response it came with.
		r, _, _ := relayOf()
		push(r, frames.NewLLMFullResponseStartFrame())
		r.reportEndCall(map[string]any{"type": "end_call", "text": "done"})
		wantEvents(t, r)
		push(r, frames.NewTTSTextFrame("Goodbye!", frames.TextTypeSentence), frames.NewLLMFullResponseEndFrame())
		wantEvents(t, r,
			map[string]any{"type": "persona_turn", "text": "Goodbye!"},
			map[string]any{"type": "end_call", "text": "done"})
	})
	t.Run("an end_call with no response going out is appended at once", func(t *testing.T) {
		r, _, _ := relayOf()
		// The response already went out.
		push(r, frames.NewLLMFullResponseStartFrame(), frames.NewLLMFullResponseEndFrame())
		r.reportEndCall(map[string]any{"type": "end_call", "text": "first"})
		// The bot's speech cut the response off.
		push(r, frames.NewLLMFullResponseStartFrame())
		r.reportEndCall(map[string]any{"type": "end_call", "text": "second"})
		push(r, frames.NewInterruptionFrame())
		wantEvents(t, r,
			map[string]any{"type": "end_call", "text": "first"},
			map[string]any{"type": "end_call", "text": "second"})
	})
	t.Run("a response interrupted before a word went out is not a turn", func(t *testing.T) {
		r, _, _ := relayOf()
		push(r, frames.NewLLMFullResponseStartFrame(), frames.NewLLMTextFrame("Could you"), frames.NewInterruptionFrame())
		wantEvents(t, r)
	})
	t.Run("an interrupted response is a turn of the words that went out", func(t *testing.T) {
		r, _, _ := relayOf()
		push(r, frames.NewLLMFullResponseStartFrame())
		r.reportEndCall(map[string]any{"type": "end_call", "text": "done"})
		push(r, frames.NewTTSTextFrame("No thanks,", frames.TextTypeSentence), frames.NewInterruptionFrame())
		wantEvents(t, r,
			map[string]any{"type": "persona_turn", "text": "No thanks,"},
			map[string]any{"type": "end_call", "text": "done"})
	})
}

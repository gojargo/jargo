package eval

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
)

// clientFor is the client of a session of scenario against botURL.
func clientFor(t *testing.T, scenario *ScriptScenario, botURL string, params SessionParams) *Client {
	t.Helper()
	s, err := NewScriptSession(scenario, botURL, params, nil, nil)
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
		params := SessionParams{DefaultTimeoutMS: 1234, TriggerDisconnect: true}
		s := newSessionFor(t, &ScriptScenario{Name: "t"}, params)
		if s.script.defaultTimeoutMS != 1234 || !strings.Contains(s.client.connectURL(), "trigger_disconnect=true") {
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
	sink := func() (*botFrameSink, *collector) {
		s := newBotFrameSink(newStream(true))
		if err := s.Setup(ctx, processor.Setup{}); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = s.Cleanup(context.WithoutCancel(ctx)) })
		next := newCollector()
		s.Link(next)
		return s, next
	}
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
	t.Run("the bot's frames stop at the sink", func(t *testing.T) {
		s, next := sink()
		push(s,
			frames.NewLLMFullResponseStartFrame(), frames.NewLLMTextFrame("Hello"), frames.NewLLMFullResponseEndFrame(),
			frames.NewTTSTextFrame("Hello", frames.TextTypeSentence), frames.NewBotStartedSpeakingFrame(),
			frames.NewFunctionCallInProgressFrame("c", "f", json.RawMessage(`{}`), false, ""),
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
}

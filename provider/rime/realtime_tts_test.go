package rime

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
)

func TestRealtimeTTSCodaSamplingParamsAreIncludedInWebsocketParams(t *testing.T) {
	_, s := newRealtimeTTS(RealtimeTTSConfig{
		APIKey:            "test-api-key",
		Model:             "coda",
		Speaker:           "luna",
		RepetitionPenalty: new(1.1),
		Temperature:       new(0.5),
		TopP:              new(0.9),
		TimeScaleFactor:   new(1.2),
	})

	params := s.wsParams()

	want := map[string]string{
		"modelId":            "coda",
		"speaker":            "luna",
		"repetition_penalty": "1.1",
		"temperature":        "0.5",
		"top_p":              "0.9",
		"timeScaleFactor":    "1.2",
	}
	for k, v := range want {
		if got := params.Get(k); got != v {
			t.Errorf("%s = %q, want %q", k, got, v)
		}
	}
}

func TestRealtimeTTSInlineSpeedListsOneSpeedPerBracketedSpan(t *testing.T) {
	b, s := newRealtimeTTS(RealtimeTTSConfig{APIKey: "test-api-key"})

	text := InlineSpeed(b, "slow", 0.5) + " and " + InlineSpeed(b, "fast", 1.5)
	msg := s.buildMsg(text, "")

	if msg["text"] != "[slow] and [fast]" {
		t.Errorf("text = %v, want the spans in brackets", msg["text"])
	}
	if msg["inlineSpeedAlpha"] != "0.5,1.5" {
		t.Errorf("inlineSpeedAlpha = %v, want 0.5,1.5", msg["inlineSpeedAlpha"])
	}
	// The fields go with one message only.
	if _, ok := s.buildMsg("next", "")["inlineSpeedAlpha"]; ok {
		t.Error("the inline speeds were sent again with the next message")
	}
}

func TestRealtimeTTSPronounceWrapsThePhonemeInBraces(t *testing.T) {
	b, s := newRealtimeTTS(RealtimeTTSConfig{APIKey: "test-api-key"})

	text := Pronounce(b, "Say hello", "hello", "h'El0")
	msg := s.buildMsg(text, "")

	if msg["text"] != "Say {h'El0}" {
		t.Errorf("text = %v, want the phoneme in braces", msg["text"])
	}
	if msg["phonemizeBetweenBrackets"] != true {
		t.Errorf("phonemizeBetweenBrackets = %v, want true", msg["phonemizeBetweenBrackets"])
	}
}

// TestRealtimeTTSMarkupHelpers pins the spelling of Rime's spell function and
// pause tag.
func TestRealtimeTTSMarkupHelpers(t *testing.T) {
	if got := Spell("ABC"); got != "spell(ABC)" {
		t.Errorf("Spell = %q", got)
	}
	if got := PauseTag(0.5); got != "<500.0>" {
		t.Errorf("PauseTag(0.5) = %q, want <500.0>", got)
	}
}

// The mist controls go to the mist models only, and the coda ones to coda only.
func TestRealtimeTTSMistParamsFollowTheModel(t *testing.T) {
	_, s := newRealtimeTTS(RealtimeTTSConfig{
		APIKey:                   "k",
		Model:                    "mistv2",
		Language:                 language.FrenchCA,
		Segment:                  "bySentence",
		SpeedAlpha:               new(1.0),
		ReduceLatency:            new(true),
		PauseBetweenBrackets:     new(true),
		PhonemizeBetweenBrackets: new(false),
		NoTextNormalization:      new(true),
		SaveOovs:                 new(false),
		Temperature:              new(0.5),
	})

	params := s.wsParams()

	want := url.Values{
		"modelId":                  {"mistv2"},
		"audioFormat":              {"pcm"},
		"samplingRate":             {"24000"},
		"lang":                     {"fra"},
		"segment":                  {"bySentence"},
		"speedAlpha":               {"1.0"},
		"reduceLatency":            {"True"},
		"pauseBetweenBrackets":     {"true"},
		"phonemizeBetweenBrackets": {"false"},
		"noTextNormalization":      {"true"},
		"saveOovs":                 {"false"},
	}
	if params.Encode() != want.Encode() {
		t.Errorf("params = %s, want %s", params.Encode(), want.Encode())
	}
}

func TestRealtimeTTSServiceLabel(t *testing.T) {
	providertest.Service(t, "RimeRealtimeTTS", NewRealtimeTTS(RealtimeTTSConfig{APIKey: "k"}))
}

// fakeRimeWS is a local Rime WebSocket JSON server. It records the query each
// connection was opened with and the messages it receives, and answers text
// with a chunk of audio and the timings of its words, and a flush with done.
type fakeRimeWS struct {
	t *testing.T
	// holdDone answers a flush with nothing, leaving the context open.
	holdDone bool
	url      string

	mu       sync.Mutex
	queries  []url.Values
	messages []map[string]any
}

func newFakeRimeWS(t *testing.T, f *fakeRimeWS) *fakeRimeWS {
	t.Helper()
	f.t = t
	srv := httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(srv.Close)
	f.url = "ws" + strings.TrimPrefix(srv.URL, "http")
	return f
}

func (f *fakeRimeWS) serve(w http.ResponseWriter, r *http.Request) {
	c, err := websocket.Accept(w, r, nil)
	if err != nil {
		return
	}
	defer func() { _ = c.CloseNow() }()
	f.mu.Lock()
	f.queries = append(f.queries, r.URL.Query())
	f.mu.Unlock()

	ctx := r.Context()
	send := func(msg map[string]any) {
		payload, _ := json.Marshal(msg)
		_ = c.Write(ctx, websocket.MessageText, payload)
	}
	last := ""
	for {
		_, data, err := c.Read(ctx)
		if err != nil {
			return
		}
		var msg map[string]any
		if err := json.Unmarshal(data, &msg); err != nil {
			f.t.Errorf("decoding a client message: %v", err)
			return
		}
		f.mu.Lock()
		f.messages = append(f.messages, msg)
		f.mu.Unlock()
		if text, _ := msg["text"].(string); text != "" {
			last, _ = msg["contextId"].(string)
			send(map[string]any{
				"type": "chunk", "contextId": last,
				"data": base64.StdEncoding.EncodeToString([]byte{1, 2, 3, 4}),
			})
			// Each word is timed, a comma as a word of its own, the way Rime
			// times punctuation.
			var words []string
			var starts, ends []float64
			for word := range strings.FieldsSeq(text) {
				parts := []string{word}
				if w, ok := strings.CutSuffix(word, ","); ok {
					parts = []string{w, ","}
				}
				for _, part := range parts {
					words = append(words, part)
					starts = append(starts, float64(len(starts))*0.1)
					ends = append(ends, float64(len(ends)+1)*0.1)
				}
			}
			send(map[string]any{
				"type": "timestamps", "contextId": last,
				"word_timestamps": map[string]any{"words": words, "start": starts, "end": ends},
			})
		}
		if msg["operation"] == "flush" && last != "" && !f.holdDone {
			send(map[string]any{"type": "done", "contextId": last})
		}
	}
}

func (f *fakeRimeWS) sent() []map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.messages)
}

func (f *fakeRimeWS) connectionQueries() []url.Values {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.queries)
}

// operations names each message sent: its operation, or "text".
func operations(sent []map[string]any) []string {
	out := make([]string, 0, len(sent))
	for _, m := range sent {
		if op, ok := m["operation"].(string); ok {
			out = append(out, op)
			continue
		}
		out = append(out, "text")
	}
	return out
}

func testRealtimeTTS(f *fakeRimeWS, cfg RealtimeTTSConfig) (*tts.Base, *realtimeSynthesizer) {
	cfg.APIKey = "test-key"
	cfg.URL = f.url
	return newRealtimeTTS(cfg)
}

// A turn is sent as text on its context, flushed once its text has all been
// sent, and its audio and word timings come back on that context. The stream is
// ended with eos when the pipeline ends.
func TestRealtimeTTSRoundTrip(t *testing.T) {
	f := newFakeRimeWS(t, &fakeRimeWS{})
	b, _ := testRealtimeTTS(f, RealtimeTTSConfig{Speaker: "luna"})

	down, up := runTTSTest(t, b,
		queue(frames.NewTTSSpeakFrame("Hello, world.")),
		pause(300*time.Millisecond),
		queue(frames.NewBotStoppedSpeakingFrame()),
	)

	if errs := framesOf[*frames.ErrorFrame](slices.Concat(down, up)); len(errs) > 0 {
		t.Errorf("errors = %v", errs)
	}
	sent := f.sent()
	ops := operations(sent)
	if len(ops) < 3 || ops[0] != "text" || ops[1] != "flush" || ops[len(ops)-1] != "eos" {
		t.Errorf("messages = %v, want text, flush, then eos at the end", ops)
	}
	if sent[0]["text"] != "Hello, world. " || sent[0]["contextId"] == "" {
		t.Errorf("text message = %v, want the sentence on its context", sent[0])
	}
	if got := audioBytes(down); !bytes.Equal(got, []byte{1, 2, 3, 4}) {
		t.Errorf("audio = % x", got)
	}
	// The punctuation Rime times as a word of its own is attached to the word
	// before it.
	if got, want := spokenTexts(down), []string{"Hello,", "world."}; !slices.Equal(got, want) {
		t.Errorf("spoken words = %q, want %q", got, want)
	}
	q := f.connectionQueries()
	if len(q) == 0 || q[0].Get("speaker") != "luna" || q[0].Get("modelId") != "coda" ||
		q[0].Get("audioFormat") != "pcm" || q[0].Get("samplingRate") != "24000" {
		t.Errorf("connection query = %v", q)
	}
}

// An interruption clears what Rime holds.
func TestRealtimeTTSInterruptionClearsTheContext(t *testing.T) {
	f := newFakeRimeWS(t, &fakeRimeWS{holdDone: true})
	b, _ := testRealtimeTTS(f, RealtimeTTSConfig{})

	runTTSTest(t, b,
		queue(frames.NewTTSSpeakFrame("Stop me now.")),
		pause(100*time.Millisecond),
		queue(frames.NewInterruptionFrame()),
		pause(100*time.Millisecond),
	)

	if ops := operations(f.sent()); !slices.Contains(ops, "clear") {
		t.Errorf("messages = %v, want a clear", ops)
	}
}

// Every setting is a query parameter of the connection, so a change reconnects
// with the new value.
func TestRealtimeTTSSettingsChangeReconnects(t *testing.T) {
	f := newFakeRimeWS(t, &fakeRimeWS{})
	b, _ := testRealtimeTTS(f, RealtimeTTSConfig{Speaker: "luna"})

	update := &TTSSettings{}
	update.Voice = settings.Set("astra")
	runTTSTest(t, b,
		pause(50*time.Millisecond),
		queue(frames.NewTTSUpdateSettingsFrame(update)),
		pause(100*time.Millisecond),
	)

	q := f.connectionQueries()
	if len(q) != 2 || q[1].Get("speaker") != "astra" {
		t.Errorf("connection queries = %v, want a second connection with the new voice", q)
	}
}

// Rime's own name for the voice is accepted in an update.
func TestRealtimeTTSSpeakerAlias(t *testing.T) {
	delta := &TTSSettings{}
	if err := settings.FromMap(delta, map[string]any{"speaker": "astra"}); err != nil {
		t.Fatalf("FromMap: %v", err)
	}
	if got := delta.Voice.Or(""); got != "astra" {
		t.Errorf("voice = %q, want astra", got)
	}
}

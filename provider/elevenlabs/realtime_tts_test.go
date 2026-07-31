package elevenlabs

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/providertest"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
)

// TestRealtimeTTSConfigValidate pins which fields the service requires.
func TestRealtimeTTSConfigValidate(t *testing.T) {
	providertest.Configs(t, []providertest.ConfigCase{
		{Name: "missing API key", Cfg: RealtimeTTSConfig{}, Valid: false},
		{Name: "API key only", Cfg: RealtimeTTSConfig{APIKey: "k"}, Valid: true},
	})
}

// TestNewRealtimeTTS checks the constructor returns a service under the label
// that identifies it in logs, metrics and traces.
func TestNewRealtimeTTS(t *testing.T) {
	providertest.Service(t, "ElevenLabsRealtimeTTS", NewRealtimeTTS(RealtimeTTSConfig{APIKey: "k"}))
}

// Word timings are only reported when they were asked for, so the base takes the
// word-aligned path solely then.
func TestRealtimeTTSWordTimestampsAreOptional(t *testing.T) {
	plain := &realtimeSynthesizer{cfg: RealtimeTTSConfig{APIKey: "k"}.withDefaults()}
	if _, ok := any(plain).(tts.WordTimestamps); ok {
		t.Fatal("the plain synthesizer must not implement tts.WordTimestamps")
	}
	timed := &timedRealtimeSynthesizer{realtimeSynthesizer: plain}
	if _, ok := any(timed).(tts.WordTimestamps); !ok {
		t.Fatal("the timed synthesizer must implement tts.WordTimestamps")
	}
}

// The connection is opened once and its fixed parameters live in the URL, since
// only the text varies from one sentence to the next.
func TestRealtimeTTSEndpoint(t *testing.T) {
	s := &realtimeSynthesizer{cfg: RealtimeTTSConfig{
		APIKey:     "k",
		SampleRate: 24000,
		Language:   language.French,
	}.withDefaults()}

	raw := s.endpoint()
	if !strings.HasPrefix(raw, defaultRealtimeTTSHost+"/v1/text-to-speech/"+defaultVoiceID+"/multi-stream-input?") {
		t.Fatalf("unexpected endpoint: %s", raw)
	}
	u, err := url.Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	for key, want := range map[string]string{
		"model_id":      defaultModel,
		"output_format": "pcm_24000",
		"language_code": "fr",
		// Omitting auto_mode leaves the server buffering text that closing the
		// context then discards, so a synthesis returns no audio whatsoever.
		"auto_mode": "true",
	} {
		if got := u.Query().Get(key); got != want {
			t.Errorf("%s = %q, want %q", key, got, want)
		}
	}
}

// TestRealtimeTTSAcceptsALargeAudioMessage checks a single message carrying a
// long sentence's audio is read rather than rejected. Generating a whole sentence
// at once puts well over a megabyte of base64 in one message, and a read limit
// tight enough to cut that off fails the synthesis part-way through a reply.
func TestRealtimeTTSAcceptsALargeAudioMessage(t *testing.T) {
	// Comfortably past the megabyte that used to be the ceiling.
	big := make([]byte, 2<<20)
	for i := range big {
		big[i] = byte(i)
	}
	host, _ := realtimeTTSServer(t, [][]map[string]any{{
		audioMessage(big),
		{"isFinal": true},
	}})

	s := &realtimeSynthesizer{cfg: RealtimeTTSConfig{
		APIKey: "k", Host: host, SampleRate: 24000,
	}.withDefaults()}
	defer func() { _ = s.Close() }()

	if total := len(speakOne(t, s, "Une phrase longue.")); total != len(big) {
		t.Errorf("delivered %d bytes, want %d", total, len(big))
	}
}

// TestRealtimeTTSAutoModeCanBeDisabled checks the default is a default and not a
// constant: a caller driving generation itself with explicit flushes can turn it
// off.
func TestRealtimeTTSAutoModeCanBeDisabled(t *testing.T) {
	s := &realtimeSynthesizer{cfg: RealtimeTTSConfig{
		APIKey:   "k",
		AutoMode: new(false),
	}.withDefaults()}

	u, err := url.Parse(s.endpoint())
	if err != nil {
		t.Fatal(err)
	}
	if got := u.Query().Get("auto_mode"); got != "false" {
		t.Errorf("auto_mode = %q, want %q", got, "false")
	}
}

// realtimeTTSServer starts a fake multi-stream-input endpoint that records the
// client's messages and replies with a scripted sequence per synthesis.
func realtimeTTSServer(t *testing.T, script [][]map[string]any) (host string, sent <-chan map[string]any) {
	t.Helper()
	messages := make(chan map[string]any, 32)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()

		// Auto mode makes the server generate as text arrives. Without it, text is
		// held until a flush, and closing the context discards what was never
		// flushed — the real server then answers with a final marker and no audio.
		autoMode := r.URL.Query().Get("auto_mode") == "true"

		for _, events := range script {
			// Each synthesis opens a context, sends its text, and closes it.
			var contextID string
			generate := autoMode
			for {
				_, data, err := c.Read(ctx)
				if err != nil {
					return
				}
				var msg map[string]any
				if json.Unmarshal(data, &msg) != nil {
					return
				}
				messages <- msg
				if id, ok := msg["context_id"].(string); ok {
					contextID = id
				}
				if flush, ok := msg["flush"].(bool); ok && flush {
					generate = true
				}
				if closed, ok := msg["close_context"].(bool); ok && closed {
					break
				}
			}
			if !generate {
				b, _ := json.Marshal(map[string]any{"contextId": contextID, "isFinal": true})
				if c.Write(ctx, websocket.MessageText, b) != nil {
					return
				}
				continue
			}
			for _, ev := range events {
				// The server stamps every message with the context it belongs to.
				if _, ok := ev["contextId"]; !ok {
					ev["contextId"] = contextID
				}
				b, _ := json.Marshal(ev)
				if c.Write(ctx, websocket.MessageText, b) != nil {
					return
				}
			}
		}
		for {
			if _, _, err := c.Read(ctx); err != nil {
				return
			}
		}
	}))
	t.Cleanup(srv.Close)

	return "ws" + strings.TrimPrefix(srv.URL, "http"), messages
}

func audioMessage(pcm []byte) map[string]any {
	return map[string]any{"audio": base64.StdEncoding.EncodeToString(pcm)}
}

// TestRealtimeTTSSynthesize checks the messages the service sends and the audio
// it emits.
func TestRealtimeTTSSynthesize(t *testing.T) {
	host, sent := realtimeTTSServer(t, [][]map[string]any{{
		audioMessage([]byte{1, 2, 3, 4}),
		audioMessage([]byte{5, 6}),
		{"isFinal": true},
	}})

	s := &realtimeSynthesizer{cfg: RealtimeTTSConfig{APIKey: "k", Host: host}.withDefaults()}
	t.Cleanup(func() { _ = s.Close() })

	if got, want := speakOne(t, s, "Bonjour."), []byte{1, 2, 3, 4, 5, 6}; string(got) != string(want) {
		t.Fatalf("delivered %v, want %v", got, want)
	}

	open := <-sent
	if open["text"] != " " {
		t.Errorf("the opening message should initialize without committing text, got %q", open["text"])
	}
	body := <-sent
	if body["text"] != "Bonjour." {
		t.Errorf("sent text = %q, want %q", body["text"], "Bonjour.")
	}
	// Closing the context is what makes the final marker arrive right after the
	// last audio byte instead of the server waiting for more text.
	closing := <-sent
	if closing["close_context"] != true {
		t.Errorf("the context should be closed after the text, got %v", closing)
	}
	if open["context_id"] != body["context_id"] || body["context_id"] != closing["context_id"] {
		t.Error("every message of a synthesis must carry the same context id")
	}
}

// A second sentence must reuse the open connection: not paying for a handshake
// per sentence is the whole point of this service over the HTTP one.
func TestRealtimeTTSReusesTheConnection(t *testing.T) {
	host, _ := realtimeTTSServer(t, [][]map[string]any{
		{audioMessage([]byte{1}), {"isFinal": true}},
		{audioMessage([]byte{2}), {"isFinal": true}},
	})

	s := &realtimeSynthesizer{cfg: RealtimeTTSConfig{APIKey: "k", Host: host}.withDefaults()}
	t.Cleanup(func() { _ = s.Close() })

	speakOne(t, s, "Un.")
	first := s.conn
	speakOne(t, s, "Deux.")
	if s.conn != first {
		t.Fatal("the second synthesis redialed instead of reusing the open connection")
	}
}

// Audio the server had already generated for an abandoned synthesis arrives
// after it. Attributing by context id is what keeps it out of the next sentence.
func TestRealtimeTTSIgnoresOtherContexts(t *testing.T) {
	host, _ := realtimeTTSServer(t, [][]map[string]any{{
		{"contextId": "stale", "audio": base64.StdEncoding.EncodeToString([]byte{9, 9, 9})},
		audioMessage([]byte{7}),
		{"isFinal": true},
	}})

	s := &realtimeSynthesizer{cfg: RealtimeTTSConfig{APIKey: "k", Host: host}.withDefaults()}
	t.Cleanup(func() { _ = s.Close() })

	if got, want := speakOne(t, s, "Bonjour."), []byte{7}; string(got) != string(want) {
		t.Fatalf("delivered %v, want %v: audio from another context leaked in", got, want)
	}
}

// ElevenLabs times every character; the words spoken are assembled from those.
func TestRealtimeTTSReportsWords(t *testing.T) {
	host, _ := realtimeTTSServer(t, [][]map[string]any{{
		{
			"audio": base64.StdEncoding.EncodeToString([]byte{1, 2}),
			"alignment": map[string]any{
				"chars":            []string{"S", "a", "l", "u", "t", " ", "t", "o", "i"},
				"charStartTimesMs": []float64{0, 50, 100, 150, 200, 250, 300, 350, 400},
			},
		},
		{"isFinal": true},
	}})

	s := &timedRealtimeSynthesizer{realtimeSynthesizer: &realtimeSynthesizer{
		cfg: RealtimeTTSConfig{APIKey: "k", Host: host, WordTimestamps: true}.withDefaults(),
	}}
	t.Cleanup(func() { _ = s.Close() })

	// Words are delivered to the host with the audio, not returned by the call.
	rec := &offsetHost{}
	s.SetAudioContextHost(rec)
	ctx := context.Background()
	if err := s.RunTTSTimed(ctx, "Salut toi", "c1", nil, nil); err != nil {
		t.Fatal(err)
	}
	s.OnTurnContextCompleted(ctx, "c1")
	rec.waitForClose(t)

	want := []float64{0, 0.3}
	got := rec.offsets()
	if len(got) != len(want) {
		t.Fatalf("reported %v offsets, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("word offsets = %v, want %v", got, want)
		}
	}
}

// fakeHost stands in for the tts.Base a provider appends its audio to.
type fakeHost struct {
	mu     sync.Mutex
	audio  []byte
	words  []string
	frames []string
	closed bool
}

func (h *fakeHost) AppendToAudioContext(_ string, f frames.Frame) {
	h.mu.Lock()
	defer h.mu.Unlock()
	switch fr := f.(type) {
	case *frames.TTSAudioRawFrame:
		h.audio = append(h.audio, fr.Audio...)
		h.frames = append(h.frames, "audio")
	case *frames.TTSStoppedFrame:
		h.frames = append(h.frames, "stopped")
	}
}

func (h *fakeHost) AppendWordToAudioContext(_, word string, _ float64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.words = append(h.words, word)
}

func (h *fakeHost) RemoveAudioContext(string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.closed = true
}

func (h *fakeHost) AudioContextAvailable(string) bool { return !h.closed }

func (h *fakeHost) snapshot() ([]byte, []string, []string, bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]byte(nil), h.audio...), append([]string(nil), h.words...),
		append([]string(nil), h.frames...), h.closed
}

// waitForClose blocks until the provider has closed the context, or fails.
func (h *fakeHost) waitForClose(t *testing.T) {
	t.Helper()
	deadline := time.After(3 * time.Second)
	for {
		if _, _, _, closed := h.snapshot(); closed {
			return
		}
		select {
		case <-deadline:
			t.Fatal("the provider never closed the audio context")
		case <-time.After(5 * time.Millisecond):
		}
	}
}

// turnTTSServer starts a fake multi-stream-input endpoint that holds one context
// open across a turn: it records every message and replies only once the context
// is closed. Nothing comes back before then, so a client that waited for a
// sentence's audio before sending the next one would hang here.
func turnTTSServer(t *testing.T, events []map[string]any) (host string, sent <-chan map[string]any) {
	t.Helper()
	messages := make(chan map[string]any, 64)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()

		var contextID string
		for {
			_, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			var msg map[string]any
			if json.Unmarshal(data, &msg) != nil {
				return
			}
			messages <- msg
			if id, ok := msg["context_id"].(string); ok {
				contextID = id
			}
			if closed, ok := msg["close_context"].(bool); ok && closed {
				break
			}
		}
		for _, ev := range events {
			if _, ok := ev["contextId"]; !ok {
				ev["contextId"] = contextID
			}
			b, _ := json.Marshal(ev)
			if c.Write(ctx, websocket.MessageText, b) != nil {
				return
			}
		}
		for {
			if _, _, err := c.Read(ctx); err != nil {
				return
			}
		}
	}))
	t.Cleanup(srv.Close)

	return "ws" + strings.TrimPrefix(srv.URL, "http"), messages
}

// Every sentence of a turn goes into one context, closed once at the end. A
// context per sentence would make the server generate each one cold, and the
// seam between them is what leaves the transport with nothing to send.
func TestRealtimeTTSTurnSharesOneContext(t *testing.T) {
	wsHost, sent := turnTTSServer(t, []map[string]any{
		audioMessage([]byte{1, 2}),
		audioMessage([]byte{3, 4}),
		{"isFinal": true},
	})

	s := &realtimeSynthesizer{cfg: RealtimeTTSConfig{APIKey: "k", Host: wsHost}.withDefaults()}
	t.Cleanup(func() { _ = s.Close() })
	host := &fakeHost{}
	s.SetAudioContextHost(host)

	ctx := context.Background()
	// Neither call waits for the other's audio: the server sends none until the
	// context closes, so a client that waited would never reach the second.
	if err := s.RunTTS(ctx, "Un.", "turn-1", nil); err != nil {
		t.Fatal(err)
	}
	if err := s.RunTTS(ctx, "Deux.", "turn-1", nil); err != nil {
		t.Fatal(err)
	}
	s.OnTurnContextCompleted(ctx, "turn-1")
	host.waitForClose(t)

	audio, _, seq, _ := host.snapshot()
	if want := []byte{1, 2, 3, 4}; string(audio) != string(want) {
		t.Fatalf("appended audio %v, want %v", audio, want)
	}
	if len(seq) == 0 || seq[len(seq)-1] != "stopped" {
		t.Fatalf("frame sequence = %v, want it to end on a stop frame", seq)
	}

	var texts []string
	var closes int
	ids := map[string]bool{}
	for range 4 {
		msg := <-sent
		if id, ok := msg["context_id"].(string); ok {
			ids[id] = true
		}
		if text, ok := msg["text"].(string); ok {
			texts = append(texts, text)
		}
		if closed, ok := msg["close_context"].(bool); ok && closed {
			closes++
		}
	}
	if len(ids) != 1 {
		t.Errorf("a turn's messages must share one context id, got %d ids", len(ids))
	}
	if closes != 1 {
		t.Errorf("the context should be closed once, at the end of the turn, got %d", closes)
	}
	want := []string{" ", "Un.", "Deux."}
	if len(texts) != len(want) {
		t.Fatalf("sent texts = %v, want %v", texts, want)
	}
	for i := range want {
		if texts[i] != want[i] {
			t.Fatalf("sent texts = %v, want %v", texts, want)
		}
	}
}

// ElevenLabs times each message from the start of the audio it carries, so the
// offsets are rebased onto the context's timeline. Without that, every message
// after the first reports its words near zero and they are released ahead of the
// audio that speaks them.
func TestRealtimeTTSRebasesWordOffsetsOntoTheContext(t *testing.T) {
	alignment := func() map[string]any {
		return map[string]any{
			"chars":            []string{"U", "n", " "},
			"charStartTimesMs": []float64{0, 100, 200},
			"charDurationsMs":  []float64{100, 100, 100},
		}
	}
	first := audioMessage([]byte{1, 2})
	first["alignment"] = alignment()
	second := audioMessage([]byte{3, 4})
	second["alignment"] = alignment()

	wsHost, _ := turnTTSServer(t, []map[string]any{first, second, {"isFinal": true}})

	cfg := RealtimeTTSConfig{APIKey: "k", Host: wsHost, WordTimestamps: true}.withDefaults()
	s := &realtimeSynthesizer{cfg: cfg}
	t.Cleanup(func() { _ = s.Close() })
	rec := &offsetHost{}
	s.SetAudioContextHost(rec)

	ctx := context.Background()
	if err := s.RunTTS(ctx, "Un un.", "turn-1", nil); err != nil {
		t.Fatal(err)
	}
	s.OnTurnContextCompleted(ctx, "turn-1")
	rec.waitForClose(t)

	// The first message ends at 200ms plus 100ms of duration, so the second
	// message's word starts there rather than back at zero.
	want := []float64{0, 0.3}
	got := rec.offsets()
	if len(got) != len(want) {
		t.Fatalf("word offsets = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("word offsets = %v, want %v", got, want)
		}
	}
}

// offsetHost records the offsets words are reported at.
type offsetHost struct {
	fakeHost
	mu   sync.Mutex
	offs []float64
}

func (h *offsetHost) AppendWordToAudioContext(_, _ string, offset float64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.offs = append(h.offs, offset)
}

func (h *offsetHost) offsets() []float64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	return append([]float64(nil), h.offs...)
}

// An interrupted context must not spill into the next one: audio the server had
// already generated for it arrives after the interruption and is dropped by id.
func TestRealtimeTTSInterruptedContextDropsItsAudio(t *testing.T) {
	s := &realtimeSynthesizer{cfg: RealtimeTTSConfig{APIKey: "k"}.withDefaults()}
	host := &fakeHost{}
	s.SetAudioContextHost(host)
	s.states = map[string]*contextState{"c1": {}}

	s.OnAudioContextInterrupted(context.Background(), "c1")

	stale := &realtimeTTSMessage{ContextID: "c1", Audio: base64.StdEncoding.EncodeToString([]byte{1, 2})}
	if err := s.dispatch(stale); err != nil {
		t.Fatal(err)
	}
	if audio, _, _, _ := host.snapshot(); len(audio) != 0 {
		t.Fatalf("audio from the interrupted context reached the host: %v", audio)
	}
}

// speakOne drives one synthesis on its own context and returns the audio the
// provider delivered to its host.
func speakOne(t *testing.T, s *realtimeSynthesizer, text string) []byte {
	t.Helper()
	host := &fakeHost{}
	s.SetAudioContextHost(host)
	ctx := context.Background()
	if err := s.RunTTS(ctx, text, "c1", nil); err != nil {
		t.Fatalf("synthesis failed: %v", err)
	}
	s.OnTurnContextCompleted(ctx, "c1")
	host.waitForClose(t)
	audio, _, _, _ := host.snapshot() //nolint:dogsled // only the audio matters here
	return audio
}

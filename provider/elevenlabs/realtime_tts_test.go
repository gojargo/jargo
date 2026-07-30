package elevenlabs

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/coder/websocket"
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

	total := 0
	err := s.Synthesize(context.Background(), "Une phrase longue.", func(pcm []byte) error {
		total += len(pcm)
		return nil
	})
	if err != nil {
		t.Fatalf("synthesis of a large message failed: %v", err)
	}
	if total != len(big) {
		t.Errorf("emitted %d bytes, want %d", total, len(big))
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

	var got []byte
	err := s.Synthesize(context.Background(), "Bonjour.", func(pcm []byte) error {
		got = append(got, pcm...)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := []byte{1, 2, 3, 4, 5, 6}; string(got) != string(want) {
		t.Fatalf("emitted %v, want %v", got, want)
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

	discard := func([]byte) error { return nil }
	if err := s.Synthesize(context.Background(), "Un.", discard); err != nil {
		t.Fatal(err)
	}
	first := s.conn
	if err := s.Synthesize(context.Background(), "Deux.", discard); err != nil {
		t.Fatal(err)
	}
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

	var got []byte
	err := s.Synthesize(context.Background(), "Bonjour.", func(pcm []byte) error {
		got = append(got, pcm...)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := []byte{7}; string(got) != string(want) {
		t.Fatalf("emitted %v, want %v: audio from another context leaked into this synthesis", got, want)
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

	type spoken struct {
		word   string
		offset float64
	}
	var words []spoken
	err := s.SynthesizeTimed(context.Background(), "Salut toi",
		func([]byte) error { return nil },
		func(text string, offset float64) error {
			words = append(words, spoken{text, offset})
			return nil
		})
	if err != nil {
		t.Fatal(err)
	}

	want := []spoken{{"Salut", 0}, {"toi", 0.3}}
	if len(words) != len(want) {
		t.Fatalf("reported %v, want %v", words, want)
	}
	for i, w := range want {
		if words[i].word != w.word || words[i].offset != w.offset {
			t.Errorf("word %d = %v, want %v", i, words[i], w)
		}
	}
}

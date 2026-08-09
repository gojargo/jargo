package deepgram

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/language"
)

// TestQueryDefaults checks the opinionated Deepgram params are omitted when
// unset, so Deepgram's own defaults apply.
func TestQueryDefaults(t *testing.T) {
	cfg := Config{
		APIKey:   "k",
		Model:    defaultSTTModel,
		Language: language.EnglishUS,
		Encoding: defaultEncoding,
		Channels: defaultChannels,
	}
	q := cfg.query(16000, newSettings(cfg))

	// Omitted when unset → Deepgram's own defaults apply.
	for _, key := range []string{"smart_format", "vad_events", "utterance_end_ms", "endpointing"} {
		if q.Has(key) {
			t.Errorf("%s should be omitted by default, got %q", key, q.Get(key))
		}
	}
	// These remain on.
	if q.Get("interim_results") != "true" || q.Get("punctuate") != "true" {
		t.Errorf("interim_results=%q punctuate=%q, want both true",
			q.Get("interim_results"), q.Get("punctuate"))
	}
}

// Everything the session yields but a transcript is dropped, and the reader
// carries on to the next message rather than ending the session over it.
func TestRecvSkipsWhatCarriesNoTranscript(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()
		for _, m := range []map[string]any{
			{"type": msgMetadata, "request_id": "abc"},
			{"type": msgSpeechStarted},
			{"type": msgUtteranceEnd},
			// Nothing here knows this one, so it is logged and skipped rather
			// than ending the session.
			{"type": "SomethingNew"},
			// A stretch of audio Deepgram recognized nothing in.
			{"type": msgResults, "is_final": true, "channel": map[string]any{
				"alternatives": []map[string]any{{"transcript": ""}},
			}},
			{"type": msgResults, "is_final": true, "channel": map[string]any{
				"alternatives": []map[string]any{{"transcript": "hello world"}},
			}},
		} {
			data, _ := json.Marshal(m)
			if err := c.Write(ctx, websocket.MessageText, data); err != nil {
				return
			}
		}
	}))
	defer srv.Close()

	conn := &connector{cfg: Config{APIKey: "k", BaseURL: srv.URL}, live: newSettings(Config{})}
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := conn.Connect(ctx, 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { cancel(); _ = stream.Close() }()

	res, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if len(res) != 1 || res[0].Text != "hello world" || !res[0].Final {
		t.Fatalf("result = %+v, want the one message that carried a transcript", res)
	}
	if res[0].FromFinalize {
		t.Fatal("a transcript Deepgram sent on its own does not answer a finalize")
	}
}

// Deepgram answers a finalize with nothing left to say when the transcript went
// out ahead of it. The answer still has to reach the service: it is what says
// the transcript already sent closed the utterance.
func TestRecvKeepsAnEmptyFinalizeAnswer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		data, _ := json.Marshal(map[string]any{
			"type": msgResults, "is_final": true, "from_finalize": true,
			"channel": map[string]any{"alternatives": []map[string]any{{"transcript": ""}}},
		})
		_ = c.Write(r.Context(), websocket.MessageText, data)
	}))
	defer srv.Close()

	conn := &connector{cfg: Config{APIKey: "k", BaseURL: srv.URL}, live: newSettings(Config{})}
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := conn.Connect(ctx, 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { cancel(); _ = stream.Close() }()

	res, err := stream.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if len(res) != 1 || res[0].Text != "" || !res[0].Final || !res[0].FromFinalize {
		t.Fatalf("result = %+v, want an empty result confirming the finalize", res)
	}
}

// A message that does not parse ends the session rather than being read past:
// the session is speaking in something other than what was agreed.
func TestRecvEndsTheSessionOnAMessageThatDoesNotParse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		_ = c.Write(r.Context(), websocket.MessageText, []byte("{not json"))
		<-r.Context().Done()
	}))
	defer srv.Close()

	conn := &connector{cfg: Config{APIKey: "k", BaseURL: srv.URL}, live: newSettings(Config{})}
	ctx, cancel := context.WithCancel(context.Background())
	stream, err := conn.Connect(ctx, 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { cancel(); _ = stream.Close() }()

	if _, err := stream.Recv(); err == nil {
		t.Fatal("Recv read past a message that does not parse")
	}
}

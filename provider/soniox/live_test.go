package soniox

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/service/stt"
)

// Tests for the live session. Soniox is dialed and then handed a configuration
// over the socket itself, which carries the key: nothing happens until that
// handshake lands, and none of it had coverage.

// sonioxSession is what the fake endpoint saw and what it will say.
type sonioxSession struct {
	handshake chan string
	audio     chan []byte
	send      chan string
}

// sonioxServer stands up a fake Soniox live endpoint.
func sonioxServer(t *testing.T) (endpoint string, got *sonioxSession) {
	t.Helper()
	got = &sonioxSession{
		handshake: make(chan string, 4),
		audio:     make(chan []byte, 8),
		send:      make(chan string, 8),
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		go func() {
			for msg := range got.send {
				if err := c.Write(r.Context(), websocket.MessageText, []byte(msg)); err != nil {
					return
				}
			}
		}()
		for {
			typ, data, err := c.Read(r.Context())
			if err != nil {
				return
			}
			if typ == websocket.MessageBinary {
				select {
				case got.audio <- data:
				default:
				}
				continue
			}
			select {
			case got.handshake <- string(data):
			default:
			}
		}
	}))
	t.Cleanup(srv.Close)
	return "ws" + strings.TrimPrefix(srv.URL, "http"), got
}

// dialSoniox opens a session against the fake endpoint.
func dialSoniox(t *testing.T, cfg Config) (stt.Stream, *sonioxSession) {
	t.Helper()
	endpoint, got := sonioxServer(t)
	cfg.URL = endpoint
	if cfg.APIKey == "" {
		cfg.APIKey = "test-key"
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	c := &connector{cfg: cfg, live: newSTTSettings(cfg)}

	s, err := c.Connect(t.Context(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s, got
}

// TestConnectSendsTheHandshake checks the socket is configured before anything
// else happens: the key travels in the handshake, not in a header, so a session
// that skipped it would be dialed and then refused.
func TestConnectSendsTheHandshake(t *testing.T) {
	_, got := dialSoniox(t, Config{Model: "stt-rt-v5"})

	var cfg map[string]any
	select {
	case msg := <-got.handshake:
		if err := json.Unmarshal([]byte(msg), &cfg); err != nil {
			t.Fatalf("the handshake was not JSON: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("no handshake reached the server")
	}

	if cfg["api_key"] != "test-key" {
		t.Errorf("handshake api_key = %v, want test-key", cfg["api_key"])
	}
	if cfg["model"] != "stt-rt-v5" {
		t.Errorf("handshake model = %v, want stt-rt-v5", cfg["model"])
	}
	if cfg["sample_rate"] != float64(16000) {
		t.Errorf("handshake sample_rate = %v, want the pipeline's 16000", cfg["sample_rate"])
	}
	if cfg["audio_format"] != "s16le" {
		t.Errorf("handshake audio_format = %v, want s16le", cfg["audio_format"])
	}
	if cfg["enable_endpoint_detection"] != true {
		t.Errorf("handshake enable_endpoint_detection = %v, want it on by default",
			cfg["enable_endpoint_detection"])
	}
}

// TestConnectOmitsUnsetSettings checks a setting nobody chose is left out rather
// than sent as a zero: a zero means something to Soniox, and sending one would
// override its own default with a value the caller never asked for.
func TestConnectOmitsUnsetSettings(t *testing.T) {
	_, got := dialSoniox(t, Config{})

	var cfg map[string]any
	select {
	case msg := <-got.handshake:
		_ = json.Unmarshal([]byte(msg), &cfg)
	case <-time.After(3 * time.Second):
		t.Fatal("no handshake reached the server")
	}

	for _, key := range []string{
		"language_hints", "context", "enable_speaker_diarization",
		"max_endpoint_delay_ms", "endpoint_sensitivity", "client_reference_id",
	} {
		if _, present := cfg[key]; present {
			t.Errorf("handshake carries %q, want it left out when nothing set it", key)
		}
	}
}

// TestSendWritesAudioAsBinary checks audio goes out as binary frames, the only
// shape Soniox reads PCM from.
func TestSendWritesAudioAsBinary(t *testing.T) {
	s, got := dialSoniox(t, Config{})
	if err := s.Send([]byte{9, 8, 7}); err != nil {
		t.Fatalf("Send: %v", err)
	}
	select {
	case audio := <-got.audio:
		if string(audio) != string([]byte{9, 8, 7}) {
			t.Errorf("the server saw %v, want the PCM that was sent", audio)
		}
	case <-time.After(3 * time.Second):
		t.Error("no audio reached the server")
	}
}

// TestRecvAccumulatesFinalTokensUntilTheTurnEnds checks the shape of Soniox's
// output: final tokens accumulate and are only flushed as a finished utterance
// when the end marker arrives, while provisional tokens surface as interim
// results in the meantime.
func TestRecvAccumulatesFinalTokensUntilTheTurnEnds(t *testing.T) {
	s, got := dialSoniox(t, Config{})

	got.send <- `{"tokens":[{"text":"bon","is_final":true},{"text":"soir","is_final":false}]}`
	interim := recvSoniox(t, s)
	if len(interim) != 1 || interim[0].Final || interim[0].Text != "bonsoir" {
		t.Fatalf("first result = %+v, want the interim \"bonsoir\"", interim)
	}

	got.send <- `{"tokens":[{"text":"soir","is_final":true},{"text":"<end>","is_final":true}]}`
	final := recvSoniox(t, s)
	if len(final) != 1 {
		t.Fatalf("second result = %+v, want one", final)
	}
	if !final[0].Final || !final[0].EndOfTurn || final[0].Text != "bonsoir" {
		t.Errorf("second result = %+v, want the finished utterance \"bonsoir\"", final[0])
	}
}

// TestRecvReportsAServerError checks an error message ends the read with the
// reason the server gave, rather than being read as an empty transcript.
func TestRecvReportsAServerError(t *testing.T) {
	s, got := dialSoniox(t, Config{})
	got.send <- `{"error_code":401,"error_message":"bad key"}`

	err := recvSonioxError(t, s)
	if err == nil || !strings.Contains(err.Error(), "bad key") {
		t.Errorf("Recv = %v, want the server's own reason", err)
	}
}

// TestRecvEndsOnFinished checks the finished marker closes the stream cleanly,
// which is what tells the base the session is over rather than broken.
func TestRecvEndsOnFinished(t *testing.T) {
	s, got := dialSoniox(t, Config{})
	got.send <- `{"finished":true}`

	if err := recvSonioxError(t, s); !errors.Is(err, io.EOF) {
		t.Errorf("Recv = %v, want io.EOF", err)
	}
}

// TestCloseSignalsEndOfAudio checks the socket is not simply dropped: an empty
// binary frame is what tells Soniox the audio is complete, so it finalizes what
// it is still holding.
func TestCloseSignalsEndOfAudio(t *testing.T) {
	s, got := dialSoniox(t, Config{})
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	select {
	case audio := <-got.audio:
		if len(audio) != 0 {
			t.Errorf("the server saw %v on close, want the empty end-of-audio frame", audio)
		}
	case <-time.After(3 * time.Second):
		t.Error("nothing was sent on close")
	}
}

// recvSonioxError reads until Recv fails and returns why. A read that never
// returns is the failure being tested going unnoticed, so it is bounded rather
// than left to hang until the whole suite times out.
func recvSonioxError(t *testing.T, s stt.Stream) error {
	t.Helper()
	done := make(chan error, 1)
	go func() {
		_, err := s.Recv()
		done <- err
	}()
	select {
	case err := <-done:
		return err
	case <-time.After(3 * time.Second):
		t.Fatal("Recv neither returned a result nor failed")
		return nil
	}
}

// recvSoniox reads one batch of results.
func recvSoniox(t *testing.T, s stt.Stream) []stt.Result {
	t.Helper()
	type outcome struct {
		results []stt.Result
		err     error
	}
	done := make(chan outcome, 1)
	go func() {
		r, err := s.Recv()
		done <- outcome{r, err}
	}()
	select {
	case o := <-done:
		if o.err != nil {
			t.Fatalf("Recv: %v", o.err)
		}
		return o.results
	case <-time.After(3 * time.Second):
		t.Fatal("Recv returned nothing")
		return nil
	}
}

package speechmatics

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/stt"
)

// Tests for the transcription session. Speechmatics acknowledges the start
// before audio may flow, sends interim partials as they come, and accumulates
// finalized segments that are emitted together when it reports the utterance
// has ended.

// session is what the fake endpoint saw, and the socket it is speaking on.
type session struct {
	header http.Header
	start  map[string]any
	audio  chan []byte
	closed map[string]any
}

// sttServer starts a fake Speechmatics endpoint. It acknowledges the start
// unless ackErr is set, then sends the scripted messages.
func sttServer(t *testing.T, ackErr string, reply []map[string]any) (endpoint string, seen func() *session) {
	t.Helper()
	sessions := make(chan *session, 1)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got := &session{header: r.Header.Clone(), audio: make(chan []byte, 8)}
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.Close(websocket.StatusNormalClosure, "") }()
		ctx := r.Context()

		_, data, err := c.Read(ctx)
		if err != nil {
			return
		}
		got.start = map[string]any{}
		if err := json.Unmarshal(data, &got.start); err != nil {
			t.Errorf("decoding StartRecognition: %v", err)
			return
		}
		select {
		case sessions <- got:
		default:
		}

		ack := map[string]any{"message": "RecognitionStarted"}
		if ackErr != "" {
			ack = map[string]any{"message": "Error", "reason": ackErr}
		}
		for _, m := range append([]map[string]any{ack}, reply...) {
			b, err := json.Marshal(m)
			if err != nil {
				t.Errorf("encoding a reply: %v", err)
				return
			}
			if c.Write(ctx, websocket.MessageText, b) != nil {
				return
			}
		}

		// Whatever else arrives is the audio, and finally the end of stream.
		for {
			typ, data, err := c.Read(ctx)
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
			end := map[string]any{}
			if json.Unmarshal(data, &end) == nil && end["message"] == "EndOfStream" {
				got.closed = end
			}
		}
	}))
	t.Cleanup(srv.Close)

	return "ws" + strings.TrimPrefix(srv.URL, "http"), func() *session {
		select {
		case s := <-sessions:
			return s
		case <-time.After(2 * time.Second):
			t.Fatal("the endpoint was never called")
			return nil
		}
	}
}

// transcript builds a transcript message of the given kind.
func transcript(kind, text string) map[string]any {
	return map[string]any{
		"message":  kind,
		"metadata": map[string]any{"transcript": text},
	}
}

// conn builds a connector pointed at endpoint with the defaults NewSTT fills in.
func conn(endpoint string, opts ...func(*Config)) *connector {
	cfg := Config{
		APIKey:         "test-key",
		URL:            endpoint,
		Language:       language.EnglishUS,
		OperatingPoint: defaultOperatingPoint,
		MaxDelay:       defaultMaxDelay,
	}
	for _, o := range opts {
		o(&cfg)
	}
	return &connector{cfg: cfg}
}

func TestConnectSendsStartRecognition(t *testing.T) {
	endpoint, seen := sttServer(t, "", nil)

	s, err := conn(endpoint).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	got := seen()
	if got.header.Get("Authorization") != "Bearer test-key" {
		t.Errorf("Authorization = %q, want the key as a bearer token", got.header.Get("Authorization"))
	}
	if got.start["message"] != "StartRecognition" {
		t.Fatalf("first message = %v, want StartRecognition", got.start)
	}

	format, _ := got.start["audio_format"].(map[string]any)
	if format["type"] != "raw" || format["encoding"] != "pcm_s16le" {
		t.Errorf("audio_format = %v, want raw pcm_s16le", format)
	}
	if format["sample_rate"] != float64(16000) {
		t.Errorf("sample_rate = %v, want the rate the pipeline runs at", format["sample_rate"])
	}

	tc, _ := got.start["transcription_config"].(map[string]any)
	if tc["language"] != "en" {
		t.Errorf("language = %v, want the base code", tc["language"])
	}
	if tc["operating_point"] != defaultOperatingPoint {
		t.Errorf("operating_point = %v, want %q", tc["operating_point"], defaultOperatingPoint)
	}
	if tc["enable_partials"] != true {
		t.Errorf("enable_partials = %v, want it on by default", tc["enable_partials"])
	}
	if tc["max_delay"] != defaultMaxDelay {
		t.Errorf("max_delay = %v, want %v", tc["max_delay"], defaultMaxDelay)
	}

	// End-of-utterance detection is on by default, and is what closes a turn.
	cc, _ := tc["conversation_config"].(map[string]any)
	if cc["end_of_utterance_silence_trigger"] != defaultEndOfUtteranceSilence {
		t.Errorf("conversation_config = %v, want the default silence trigger", cc)
	}
}

// TestConnectOmitsEndOfUtteranceWhenDisabled checks a zero silence leaves the
// whole conversation config out, so the provider does not end utterances itself.
func TestConnectOmitsEndOfUtteranceWhenDisabled(t *testing.T) {
	endpoint, seen := sttServer(t, "", nil)
	off := 0.0

	s, err := conn(endpoint, func(c *Config) { c.EndOfUtteranceSilence = &off }).
		Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	tc, _ := seen().start["transcription_config"].(map[string]any)
	if _, ok := tc["conversation_config"]; ok {
		t.Errorf("conversation_config = %v, want it omitted when disabled", tc["conversation_config"])
	}
}

// TestConnectDisablesPartials checks partials can be turned off, for a caller
// who wants only finalized text.
func TestConnectDisablesPartials(t *testing.T) {
	endpoint, seen := sttServer(t, "", nil)
	off := false

	s, err := conn(endpoint, func(c *Config) { c.EnablePartials = &off }).
		Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	tc, _ := seen().start["transcription_config"].(map[string]any)
	if tc["enable_partials"] != false {
		t.Errorf("enable_partials = %v, want it off", tc["enable_partials"])
	}
}

// TestConnectFailsWhenTheStartIsRefused checks the session is not handed back
// when the provider refuses the start, so audio is never sent into nothing.
func TestConnectFailsWhenTheStartIsRefused(t *testing.T) {
	endpoint, _ := sttServer(t, "invalid operating point", nil)

	s, err := conn(endpoint).Connect(context.Background(), 16000)
	if err == nil {
		_ = s.Close()
		t.Fatal("Connect accepted a refused start")
	}
	if !errors.Is(err, errServer) {
		t.Errorf("error = %v, want it reported as a server error", err)
	}
	if !strings.Contains(err.Error(), "invalid operating point") {
		t.Errorf("error = %v, want it to carry what the provider said", err)
	}
}

// TestRecvEmitsPartialsAsTheyCome checks an interim transcript is reported
// straight away, so the pipeline can react before the utterance ends.
func TestRecvEmitsPartialsAsTheyCome(t *testing.T) {
	endpoint, _ := sttServer(t, "", []map[string]any{
		transcript("AddPartialTranscript", "hello"),
	})

	s, err := conn(endpoint).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	got, err := s.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if len(got) != 1 || got[0].Text != "hello" || got[0].Final {
		t.Errorf("results = %+v, want one interim hello", got)
	}
}

// TestRecvJoinsFinalSegmentsAtEndOfUtterance checks finalized segments
// accumulate and are emitted together when the utterance ends. Emitting each on
// its own would end the turn on the first one.
func TestRecvJoinsFinalSegmentsAtEndOfUtterance(t *testing.T) {
	endpoint, _ := sttServer(t, "", []map[string]any{
		transcript("AddTranscript", "hello"),
		transcript("AddTranscript", "there"),
		{"message": "EndOfUtterance"},
	})

	s, err := conn(endpoint).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	got, err := s.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("results = %+v, want one", got)
	}
	if got[0].Text != "hello there" {
		t.Errorf("text = %q, want the segments joined with a space", got[0].Text)
	}
	if !got[0].Final || !got[0].EndOfTurn {
		t.Errorf("result = %+v, want it final and ending the turn", got[0])
	}
}

// TestRecvIgnoresAnEmptyUtterance checks an end-of-utterance with nothing
// accumulated reports no turn, so silence does not close one.
func TestRecvIgnoresAnEmptyUtterance(t *testing.T) {
	endpoint, _ := sttServer(t, "", []map[string]any{
		{"message": "EndOfUtterance"},
		transcript("AddPartialTranscript", "after"),
	})

	s, err := conn(endpoint).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	got, err := s.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if len(got) != 1 || got[0].Text != "after" {
		t.Errorf("results = %+v, want the empty utterance stepped over", got)
	}
}

// TestRecvEndsOnEndOfTranscript checks the session reports it is over the way
// the caller expects, so the read loop stops rather than treating it as a fault.
func TestRecvEndsOnEndOfTranscript(t *testing.T) {
	endpoint, _ := sttServer(t, "", []map[string]any{{"message": "EndOfTranscript"}})

	s, err := conn(endpoint).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	if _, err := s.Recv(); !errors.Is(err, io.EOF) {
		t.Errorf("Recv = %v, want io.EOF", err)
	}
}

// TestRecvReportsAServerError checks an error mid-session is surfaced carrying
// what the provider said.
func TestRecvReportsAServerError(t *testing.T) {
	endpoint, _ := sttServer(t, "", []map[string]any{
		{"message": "Error", "reason": "audio too quiet"},
	})

	s, err := conn(endpoint).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	_, err = s.Recv()
	if err == nil {
		t.Fatal("Recv accepted an error message")
	}
	if !errors.Is(err, errServer) || !strings.Contains(err.Error(), "audio too quiet") {
		t.Errorf("error = %v, want the server error carrying its reason", err)
	}
}

// TestSendWritesAudioAsBinary checks the audio goes as a binary frame, which is
// what Speechmatics reads it as.
func TestSendWritesAudioAsBinary(t *testing.T) {
	endpoint, seen := sttServer(t, "", nil)

	s, err := conn(endpoint).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	pcm := []byte{1, 2, 3, 4}
	if err := s.Send(pcm); err != nil {
		t.Fatalf("Send: %v", err)
	}

	select {
	case got := <-seen().audio:
		if string(got) != string(pcm) {
			t.Errorf("audio = %v, want %v", got, pcm)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("the audio never arrived")
	}
}

// TestMetadataReportsTheMeasuredLatency checks the transcript latency the turn
// strategies size their wait by, and that it can be overridden.
func TestMetadataReportsTheMeasuredLatency(t *testing.T) {
	if got := (&connector{}).Metadata().TTFSP99; got != stt.SpeechmaticsTTFSP99 {
		t.Errorf("TTFSP99 = %v, want the measured figure", got)
	}
	c := &connector{cfg: Config{TTFSP99: 300 * time.Millisecond}}
	if got := c.Metadata().TTFSP99; got != 300*time.Millisecond {
		t.Errorf("TTFSP99 = %v, want the override", got)
	}
}

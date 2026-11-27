package speechmatics

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/stt"
)

// Tests for the transcription session. Speechmatics acknowledges the start
// before audio may flow, sends interim partials as they come, and accumulates
// finalized segments that are emitted together when it reports the utterance
// has ended.

// session is what the fake endpoint saw, and the socket it is speaking on.
type session struct {
	header   http.Header
	path     string
	query    url.Values
	start    map[string]any
	audio    chan []byte
	messages chan map[string]any
	closed   map[string]any
}

// awaitAudio, as a scripted message, holds the rest of the script until the
// session has been sent audio.
const awaitAudio = "test: await audio"

// readUntilAudio reads what the client sends until a binary frame arrives,
// recording it, and reports whether one did.
func readUntilAudio(ctx context.Context, c *websocket.Conn, got *session) bool {
	for {
		typ, data, err := c.Read(ctx)
		if err != nil {
			return false
		}
		if typ == websocket.MessageBinary {
			select {
			case got.audio <- data:
			default:
			}
			return true
		}
	}
}

// sttServer starts a fake Speechmatics endpoint. It acknowledges the start
// unless ackErr is set, then sends the scripted messages.
func sttServer(t *testing.T, ackErr string, reply []map[string]any) (endpoint string, seen func() *session) {
	t.Helper()
	sessions := make(chan *session, 1)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got := &session{
			header: r.Header.Clone(), path: r.URL.Path, query: r.URL.Query(),
			audio: make(chan []byte, 8), messages: make(chan map[string]any, 8),
		}
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
			if m["message"] == awaitAudio {
				// Hold the rest of the script until the first audio arrives.
				if !readUntilAudio(ctx, c, got) {
					return
				}
				continue
			}
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
			m := map[string]any{}
			if json.Unmarshal(data, &m) != nil {
				continue
			}
			if m["message"] == "EndOfStream" {
				got.closed = m
				_ = c.Write(ctx, websocket.MessageText, []byte(`{"message":"EndOfTranscript"}`))
				continue
			}
			select {
			case got.messages <- m:
			default:
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

// segment builds a segment message of the given kind, attributed to speaker
// when one is given.
func segment(kind, text, speaker string) map[string]any {
	seg := map[string]any{"transcript": text}
	if speaker != "" {
		seg["speaker"] = speaker
	}
	return map[string]any{
		"message":  kind,
		"segment":  seg,
		"metadata": map[string]any{"start_time": 0.0, "end_time": 1.0},
	}
}

// conn builds a connector pointed at endpoint, with its settings built the way
// NewSTT builds them.
func conn(endpoint string, opts ...func(*Config)) *connector {
	cfg := Config{APIKey: "test-key", URL: endpoint}
	for _, o := range opts {
		o(&cfg)
	}
	return &connector{cfg: cfg, live: newSTTSettings(cfg)}
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
	if got.path != "/agent" {
		t.Errorf("path = %q, want the agent endpoint", got.path)
	}
	if got.query.Get("sm-app") != appName {
		t.Errorf("sm-app = %q, want %q", got.query.Get("sm-app"), appName)
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

	// The turn config is a sibling of the transcription config, and the
	// pipeline closes turns unless told otherwise.
	turn, _ := got.start["turn_config"].(map[string]any)
	if turn["turn_detection_mode"] != "external" {
		t.Errorf("turn_config = %v, want external turn detection by default", turn)
	}

	tc, _ := got.start["transcription_config"].(map[string]any)
	want := map[string]any{"language": "en", "model": defaultModel}
	if !reflect.DeepEqual(tc, want) {
		t.Errorf("transcription_config = %v, want only %v when nothing else is set", tc, want)
	}
}

// TestConnectSendsWhatIsConfigured checks every transcription setting reaches
// the session as Speechmatics names it, the diarization knobs inside the
// speaker diarization config.
func TestConnectSendsWhatIsConfigured(t *testing.T) {
	endpoint, seen := sttServer(t, "", nil)

	s, err := conn(endpoint, func(c *Config) {
		c.Language = language.EnglishGB
		c.Domain = "finance"
		c.Model = "linden-2"
		c.TurnDetectionMode = TurnDetectionVAD
		c.EnablePartials = new(false)
		c.EmitSentences = new(true)
		c.PunctuationOverrides = map[string]any{"permitted_marks": []any{".", ","}}
		c.TranscriptFilteringConfig = map[string]any{"remove_disfluencies": true}
		c.AdditionalVocab = []AdditionalVocabEntry{{Content: "jargo", SoundsLike: []string{"jar go"}}}
		c.EnableDiarization = new(true)
		c.MaxSpeakers = new(2)
		c.SpeakerSensitivity = new(0.7)
		c.PreferCurrentSpeaker = new(true)
		c.KnownSpeakers = []SpeakerIdentifier{{Label: "Agent", SpeakerIdentifiers: []string{"id1"}}}
	}).Connect(context.Background(), 8000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	got := seen()
	turn, _ := got.start["turn_config"].(map[string]any)
	if turn["turn_detection_mode"] != "vad" {
		t.Errorf("turn_config = %v, want vad", turn)
	}
	tc, _ := got.start["transcription_config"].(map[string]any)
	want := map[string]any{
		"language":                    "en",
		"output_locale":               "en-GB",
		"model":                       "linden-2",
		"domain":                      "finance",
		"enable_partials":             false,
		"emit_sentences":              true,
		"punctuation_overrides":       map[string]any{"permitted_marks": []any{".", ","}},
		"transcript_filtering_config": map[string]any{"remove_disfluencies": true},
		"additional_vocab":            []any{map[string]any{"content": "jargo", "sounds_like": []any{"jar go"}}},
		"diarization":                 "speaker",
		"speaker_diarization_config": map[string]any{
			"max_speakers":           float64(2),
			"speaker_sensitivity":    0.7,
			"prefer_current_speaker": true,
			"speakers":               []any{map[string]any{"label": "Agent", "speaker_identifiers": []any{"id1"}}},
		},
	}
	if !reflect.DeepEqual(tc, want) {
		t.Errorf("transcription_config =\n%v\nwant\n%v", tc, want)
	}
}

// TestConnectSendsFilipinoAsTagalog checks Filipino starts the session on the
// Tagalog pack, the one Speechmatics offers for it, rather than on fil, which
// it refuses.
func TestConnectSendsFilipinoAsTagalog(t *testing.T) {
	endpoint, seen := sttServer(t, "", nil)

	s, err := conn(endpoint, func(c *Config) { c.Language = language.Filipino }).
		Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	tc, _ := seen().start["transcription_config"].(map[string]any)
	if tc["language"] != "tl" {
		t.Errorf("language = %v, want tl", tc["language"])
	}
}

// TestSpeechmaticsLanguage checks languages resolve to the codes Speechmatics
// takes: Tagalog and Filipino, in any region, share the one pack, and a regional
// variant goes out as its base code.
func TestSpeechmaticsLanguage(t *testing.T) {
	cases := map[language.Language]string{
		language.Tagalog:    "tl",
		language.Filipino:   "tl",
		language.FilipinoPH: "tl",
		language.EnglishUS:  "en",
		language.Spanish:    "es",
	}
	for l, want := range cases {
		if got := speechmaticsLanguage(l); got != want {
			t.Errorf("speechmaticsLanguage(%q) = %q, want %q", l, got, want)
		}
	}
}

// TestConnectFailsWhenTheStartIsRefused checks the session is not handed back
// when the provider refuses the start, so audio is never sent into nothing.
func TestConnectFailsWhenTheStartIsRefused(t *testing.T) {
	endpoint, _ := sttServer(t, "invalid model", nil)

	s, err := conn(endpoint).Connect(context.Background(), 16000)
	if err == nil {
		_ = s.Close()
		t.Fatal("Connect accepted a refused start")
	}
	if !errors.Is(err, errServer) {
		t.Errorf("error = %v, want it reported as a server error", err)
	}
	if !strings.Contains(err.Error(), "invalid model") {
		t.Errorf("error = %v, want it to carry what the provider said", err)
	}
}

// TestRecvEmitsPartialSegmentsAsInterims checks a partial segment is reported
// straight away as an interim transcript, with its speaker.
func TestRecvEmitsPartialSegmentsAsInterims(t *testing.T) {
	endpoint, _ := sttServer(t, "", []map[string]any{
		segment(msgAddPartialSegment, "hello", "S1"),
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
	want := []stt.Result{{Text: "hello", Speaker: "S1"}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("results = %+v, want %+v", got, want)
	}
}

// TestRecvEmitsEachFinalSegment checks each final segment is its own final
// transcript, marked as the one answering a finalize when one was asked for,
// and that an empty segment is stepped over.
func TestRecvEmitsEachFinalSegment(t *testing.T) {
	endpoint, _ := sttServer(t, "", []map[string]any{
		segment(msgAddSegment, "", ""),
		segment(msgAddSegment, "hello there", "S2"),
		segment(msgAddSegment, "bye", ""),
	})

	s, err := conn(endpoint).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	for _, want := range []stt.Result{
		{Text: "hello there", Final: true, Speaker: "S2", FromFinalize: true},
		{Text: "bye", Final: true, FromFinalize: true},
	} {
		got, err := s.Recv()
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
		if !reflect.DeepEqual(got, []stt.Result{want}) {
			t.Errorf("results = %+v, want %+v", got, want)
		}
	}
}

// TestTurnEventsFollowTheTurnDetectionMode checks Speechmatics' turn events
// are reported as speech boundaries only when it closes turns itself, and that
// only then is the session left to close them: with external turn detection
// the session finalizes on the pipeline's VAD instead.
func TestTurnEventsFollowTheTurnDetectionMode(t *testing.T) {
	reply := []map[string]any{
		{"message": msgStartOfTurn},
		{"message": msgEndOfTurn},
		segment(msgAddSegment, "done", ""),
	}

	t.Run("vad", func(t *testing.T) {
		endpoint, _ := sttServer(t, "", reply)
		s, err := conn(endpoint, func(c *Config) { c.TurnDetectionMode = TurnDetectionVAD }).
			Connect(context.Background(), 16000)
		if err != nil {
			t.Fatalf("Connect: %v", err)
		}
		defer func() { _ = s.Close() }()
		if _, ok := s.(stt.Finalizer); ok {
			t.Error("the session finalizes on the pipeline's VAD, want the service to close turns")
		}
		for _, want := range []stt.SpeechState{stt.SpeechStarted, stt.SpeechStopped} {
			got, err := s.Recv()
			if err != nil {
				t.Fatalf("Recv: %v", err)
			}
			if len(got) != 1 || got[0].Speech != want {
				t.Errorf("results = %+v, want speech %v", got, want)
			}
		}
	})

	t.Run("external", func(t *testing.T) {
		endpoint, _ := sttServer(t, "", reply)
		s, err := conn(endpoint).Connect(context.Background(), 16000)
		if err != nil {
			t.Fatalf("Connect: %v", err)
		}
		defer func() { _ = s.Close() }()
		if _, ok := s.(stt.Finalizer); !ok {
			t.Error("the session does not finalize on the pipeline's VAD")
		}
		got, err := s.Recv()
		if err != nil {
			t.Fatalf("Recv: %v", err)
		}
		if len(got) != 1 || got[0].Text != "done" || got[0].Speech != stt.SpeechUnknown {
			t.Errorf("results = %+v, want the turn events stepped over", got)
		}
	})
}

// TestFinalizeForcesTheEndOfUtterance checks a finalize closes the turn at the
// point in the audio sent so far, as an audio timestamp.
func TestFinalizeForcesTheEndOfUtterance(t *testing.T) {
	endpoint, seen := sttServer(t, "", nil)
	s, err := conn(endpoint).Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = s.Close() }()

	// Half a second of 16 kHz 16-bit audio.
	if err := s.Send(make([]byte, 16000)); err != nil {
		t.Fatalf("Send: %v", err)
	}
	fin, ok := s.(stt.Finalizer)
	if !ok {
		t.Fatal("the session does not finalize")
	}
	if err := fin.Finalize(); err != nil {
		t.Fatalf("Finalize: %v", err)
	}

	got := seen()
	select {
	case m := <-got.messages:
		if m["message"] != "ForceEndOfUtterance" || m["timestamp"] != 0.5 {
			t.Errorf("message = %v, want ForceEndOfUtterance at 0.5s", m)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("the finalize never arrived")
	}
}

// TestRecvEndsOnEndOfTranscript checks the session reports it is over the way
// the caller expects, so the read loop stops rather than treating it as a fault.
func TestRecvEndsOnEndOfTranscript(t *testing.T) {
	endpoint, _ := sttServer(t, "", []map[string]any{{"message": msgEndOfTranscript}})

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
// what the provider said, and that a warning is not: the session continues.
func TestRecvReportsAServerError(t *testing.T) {
	endpoint, _ := sttServer(t, "", []map[string]any{
		{"message": msgWarning, "type": "duration_limit_exceeded", "reason": "soon"},
		{"message": msgError, "reason": "audio too quiet"},
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
	if got := *conn("").Metadata().TTFSP99; got != stt.SpeechmaticsTTFSP99 {
		t.Errorf("TTFSP99 = %v, want the measured figure", got)
	}
	c := conn("", func(c *Config) { c.TTFSP99 = new(300 * time.Millisecond) })
	if got := *c.Metadata().TTFSP99; got != 300*time.Millisecond {
		t.Errorf("TTFSP99 = %v, want the override", got)
	}
}

// erroringServer is a fake Speechmatics endpoint whose first session ends on a
// server Error of the given type, and that counts the sessions opened on it.
type erroringServer struct {
	mu       sync.Mutex
	sessions int
}

func newErroringServer(t *testing.T, errorType string) (*erroringServer, string) {
	t.Helper()
	f := &erroringServer{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		ctx := r.Context()
		if _, _, err := c.Read(ctx); err != nil {
			return
		}
		f.mu.Lock()
		f.sessions++
		first := f.sessions == 1
		f.mu.Unlock()
		reply := []map[string]any{{"message": "RecognitionStarted"}}
		if first {
			reply = append(reply, map[string]any{"message": "Error", "type": errorType, "reason": "no"})
		}
		for _, m := range reply {
			b, err := json.Marshal(m)
			if err != nil {
				t.Errorf("encoding a reply: %v", err)
				return
			}
			if c.Write(ctx, websocket.MessageText, b) != nil {
				return
			}
		}
		for {
			_, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			// EndOfStream is answered as Speechmatics answers it.
			if strings.Contains(string(data), `"EndOfStream"`) {
				_ = c.Write(ctx, websocket.MessageText, []byte(`{"message":"EndOfTranscript"}`))
			}
		}
	}))
	t.Cleanup(srv.Close)
	return f, "ws" + strings.TrimPrefix(srv.URL, "http")
}

func (f *erroringServer) opened() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.sessions
}

// serviceRun is a Speechmatics service running in a pipeline, recording the
// errors it reports.
type serviceRun struct {
	svc  *stt.StreamService
	task *pipeline.Worker
	done chan error

	mu     sync.Mutex
	errors []*frames.ErrorFrame
}

func runService(t *testing.T, endpoint string) *serviceRun {
	t.Helper()
	r := &serviceRun{done: make(chan error, 1)}
	r.svc = NewSTT(Config{APIKey: "test-key", URL: endpoint, SampleRate: 16000}).StreamService
	event.On(r.svc.Events(), processor.EventError, func(_ context.Context, ef *frames.ErrorFrame) {
		r.mu.Lock()
		defer r.mu.Unlock()
		r.errors = append(r.errors, ef)
	})
	r.task = pipeline.NewWorker(pipeline.New(r.svc), pipeline.WorkerConfig{IdleTimeout: -1})
	go func() { r.done <- r.task.Run(context.Background()) }()
	t.Cleanup(func() {
		r.task.StopWhenDone()
		select {
		case <-r.done:
		case <-time.After(5 * time.Second):
			t.Error("the pipeline did not shut down")
		}
	})
	return r
}

// reported is the errors reported so far.
func (r *serviceRun) reported() []*frames.ErrorFrame {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]*frames.ErrorFrame(nil), r.errors...)
}

// waitFor polls until cond holds, failing the test if it never does.
func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

// Ported from upstream (test_server_error_rejecting_the_request_is_permanent).
// A server Error that rejects the request or the account would fail a fresh
// session the same way, so it is reported as permanent: the service is no
// longer usable, and the read loop does not reconnect into it.
func TestAServerErrorRejectingTheRequestIsPermanent(t *testing.T) {
	server, endpoint := newErroringServer(t, errorTypeNotAuthorised)
	r := runService(t, endpoint)

	waitFor(t, "the error to be reported", func() bool { return len(r.reported()) > 0 })

	got := r.reported()[0]
	if !strings.Contains(got.Error, errorTypeNotAuthorised) || !strings.Contains(got.Error, "no") {
		t.Errorf("error = %q, want it to name the type and the reason", got.Error)
	}
	if r.svc.Usable() {
		t.Error("the service is still usable after a rejection that would recur")
	}
	if n := server.opened(); n != 1 {
		t.Errorf("sessions opened = %d, want the rejected one alone", n)
	}
}

// Ported from upstream (test_server_error_ending_the_session_reconnects). A
// server Error that only ends the session, such as a timeout, is reported as
// recoverable and followed by a new session.
func TestAServerErrorEndingTheSessionReconnects(t *testing.T) {
	server, endpoint := newErroringServer(t, "session_timeout")
	r := runService(t, endpoint)

	waitFor(t, "a second session", func() bool { return server.opened() >= 2 })

	errs := r.reported()
	if len(errs) == 0 {
		t.Fatal("the error that ended the session was not reported")
	}
	if !strings.Contains(errs[0].Error, "session_timeout") {
		t.Errorf("error = %q, want it to name the type", errs[0].Error)
	}
	if errs[0].Category.IsPermanent() {
		t.Errorf("category = %v, want a recoverable one", errs[0].Category)
	}
	if !r.svc.Usable() {
		t.Error("the service is unusable after an error a new session recovers from")
	}
}

// startingServer is a fake Speechmatics endpoint that answers the start of
// each session as answer says, given the session's number from 1, and counts
// the sessions opened on it. A nil answer drops the connection unanswered.
type startingServer struct {
	mu       sync.Mutex
	sessions int
}

func newStartingServer(t *testing.T, answer func(session int) map[string]any) (*startingServer, string) {
	t.Helper()
	f := &startingServer{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		ctx := r.Context()
		if _, _, err := c.Read(ctx); err != nil {
			return
		}
		f.mu.Lock()
		f.sessions++
		n := f.sessions
		f.mu.Unlock()
		reply := answer(n)
		if reply == nil {
			return
		}
		b, err := json.Marshal(reply)
		if err != nil {
			t.Errorf("encoding a reply: %v", err)
			return
		}
		if c.Write(ctx, websocket.MessageText, b) != nil {
			return
		}
		for {
			_, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			if strings.Contains(string(data), `"EndOfStream"`) {
				_ = c.Write(ctx, websocket.MessageText, []byte(`{"message":"EndOfTranscript"}`))
			}
		}
	}))
	t.Cleanup(srv.Close)
	return f, "ws" + strings.TrimPrefix(srv.URL, "http")
}

func (f *startingServer) opened() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.sessions
}

// Ported from upstream (test_connect_does_not_reconnect_after_rejection). An
// Error in place of RecognitionStarted is a rejected session, which a retry
// would meet the same way: it is reported as permanent, the service is no
// longer usable, and no further session is opened.
func TestAnErrorWhileTheSessionStartsIsPermanent(t *testing.T) {
	for _, kind := range []string{"invalid_config", errorTypeNotAuthorised} {
		t.Run(kind, func(t *testing.T) {
			server, endpoint := newStartingServer(t, func(int) map[string]any {
				return map[string]any{"message": "Error", "type": kind, "reason": "refused"}
			})
			r := runService(t, endpoint)

			waitFor(t, "the service to become unusable", func() bool { return !r.svc.Usable() })
			reported := r.reported()
			if len(reported) == 0 {
				t.Fatal("the rejection was not reported")
			}
			got := reported[0]
			if got.Err == nil || !strings.Contains(got.Err.Error(), "refused") {
				t.Errorf("error = %v, want it to carry the reason Speechmatics gave", got.Err)
			}
			// Permanent because the service was told so, whatever the category.
			if !errors.Is(got.Err, stt.ErrPermanent) {
				t.Errorf("error = %v, want it marked permanent", got.Err)
			}
			if n := server.opened(); n != 1 {
				t.Errorf("sessions opened = %d, want the rejected one alone", n)
			}
		})
	}
}

// Ported from upstream (test_open_connection_transient_drop_stays_retryable). A
// connection that drops before the session starts is not a rejection: the
// service stays usable and opens another session.
func TestADropWhileTheSessionStartsIsRetried(t *testing.T) {
	server, endpoint := newStartingServer(t, func(session int) map[string]any {
		if session == 1 {
			return nil
		}
		return map[string]any{"message": "RecognitionStarted"}
	})
	r := runService(t, endpoint)

	waitFor(t, "a second session", func() bool { return server.opened() >= 2 })
	waitFor(t, "the second session to start", r.svc.Connected)
	if !r.svc.Usable() {
		t.Error("the service is unusable after a drop a new session recovers from")
	}
}

// endingServer is a fake Speechmatics endpoint that records the EndOfStream
// messages each session receives, and answers one with EndOfTranscript unless
// told to stay silent.
type endingServer struct {
	silent bool
	// ended receives once for each session the client has closed.
	ended chan struct{}

	mu       sync.Mutex
	sessions int
	eos      []map[string]any
	audio    int
}

func newEndingServer(t *testing.T, silent bool) (*endingServer, string) {
	t.Helper()
	f := &endingServer{silent: silent, ended: make(chan struct{}, 4)}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		defer func() { f.ended <- struct{}{} }()
		ctx := r.Context()
		if _, _, err := c.Read(ctx); err != nil {
			return
		}
		f.mu.Lock()
		f.sessions++
		f.mu.Unlock()
		if c.Write(ctx, websocket.MessageText, []byte(`{"message":"RecognitionStarted"}`)) != nil {
			return
		}
		for {
			typ, data, err := c.Read(ctx)
			if err != nil {
				return
			}
			if typ == websocket.MessageBinary {
				f.mu.Lock()
				f.audio++
				f.mu.Unlock()
				continue
			}
			m := map[string]any{}
			if json.Unmarshal(data, &m) != nil || m["message"] != "EndOfStream" {
				continue
			}
			f.mu.Lock()
			f.eos = append(f.eos, m)
			f.mu.Unlock()
			if !f.silent {
				_ = c.Write(ctx, websocket.MessageText, []byte(`{"message":"EndOfTranscript"}`))
			}
		}
	}))
	t.Cleanup(srv.Close)
	return f, "ws" + strings.TrimPrefix(srv.URL, "http")
}

// endOfStreams is the EndOfStream messages received so far.
func (f *endingServer) endOfStreams() []map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]map[string]any(nil), f.eos...)
}

func (f *endingServer) counts() (sessions, audio int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.sessions, f.audio
}

// Ported from upstream (the session client's disconnect, which sends
// EndOfStream and waits for EndOfTranscript). Ending the pipeline tells
// Speechmatics the audio is over with one EndOfStream, carrying the number of
// audio chunks sent, while the socket is still open, and the teardown ends on
// the EndOfTranscript that answers it rather than waiting out its window.
func TestStopSendsEndOfStreamAndWaitsForEndOfTranscript(t *testing.T) {
	server, endpoint := newEndingServer(t, false)
	svc := NewSTT(Config{APIKey: "test-key", URL: endpoint, SampleRate: 16000})
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{IdleTimeout: -1})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	waitFor(t, "the session to open", svc.Connected)

	task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 640), 16000, 1))
	task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 640), 16000, 1))
	waitFor(t, "the audio to arrive", func() bool { _, audio := server.counts(); return audio == 2 })

	started := time.Now()
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("the pipeline did not shut down")
	}
	if took := time.Since(started); took >= endOfStreamTimeout {
		t.Errorf("the teardown took %v, want it to end on EndOfTranscript", took)
	}
	select {
	case <-server.ended:
	case <-time.After(3 * time.Second):
		t.Fatal("the session was never closed")
	}

	eos := server.endOfStreams()
	if len(eos) != 1 {
		t.Fatalf("EndOfStream messages = %d, want 1", len(eos))
	}
	if eos[0]["last_seq_no"] != float64(2) {
		t.Errorf("last_seq_no = %v, want the 2 chunks sent", eos[0]["last_seq_no"])
	}
	if sessions, _ := server.counts(); sessions != 1 {
		t.Errorf("sessions opened = %d, want 1", sessions)
	}
}

// Every session is ended with EndOfStream, not only one the pipeline stops: a
// canceled pipeline sends it too, once.
func TestCancelSendsEndOfStream(t *testing.T) {
	server, endpoint := newEndingServer(t, false)
	svc := NewSTT(Config{APIKey: "test-key", URL: endpoint, SampleRate: 16000})
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{IdleTimeout: -1})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	waitFor(t, "the session to open", svc.Connected)

	task.Cancel(context.Background(), "test")
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("the pipeline did not shut down")
	}
	select {
	case <-server.ended:
	case <-time.After(3 * time.Second):
		t.Fatal("the session was never closed")
	}
	if n := len(server.endOfStreams()); n != 1 {
		t.Errorf("EndOfStream messages = %d, want 1", n)
	}
}

// A session closed without being drained first, as one replaced after it
// failed, is still sent EndOfStream before its socket closes.
func TestCloseEndsAnUndrainedSession(t *testing.T) {
	server, endpoint := newEndingServer(t, true)
	c := conn(endpoint)
	s, err := c.Connect(t.Context(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	select {
	case <-server.ended:
	case <-time.After(3 * time.Second):
		t.Fatal("the session was never closed")
	}
	if n := len(server.endOfStreams()); n != 1 {
		t.Errorf("EndOfStream messages = %d, want 1", n)
	}
}

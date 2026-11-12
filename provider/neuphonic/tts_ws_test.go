package neuphonic

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"maps"
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
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
	uctx "github.com/gojargo/jargo/utils/context"
	"github.com/gojargo/jargo/utils/events"
)

// Tests for the connection held open for the call. Neuphonic is configured by
// the URL a connection is opened on, takes each sentence with a marker that has
// it generate the text at once, and streams base64 PCM that names no request.

// dialed is what the fake endpoint saw of one connection's handshake.
type dialed struct {
	path   string
	query  url.Values
	header http.Header
}

// fakeServer is a fake Neuphonic endpoint. It records every connection and
// every message the client sends, and answers each text with what reply
// returns.
type fakeServer struct {
	url  string
	sent chan map[string]any

	mu     sync.Mutex
	reply  func(msg map[string]any) []any
	refuse int
	conns  []*websocket.Conn
	dials  []dialed
}

func newFakeServer(t *testing.T, reply func(msg map[string]any) []any) *fakeServer {
	t.Helper()
	s := &fakeServer{sent: make(chan map[string]any, 256), reply: reply}
	srv := httptest.NewServer(http.HandlerFunc(s.serve))
	t.Cleanup(srv.Close)
	s.url = "ws" + strings.TrimPrefix(srv.URL, "http")
	return s
}

func (s *fakeServer) serve(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	refuse := s.refuse
	s.mu.Unlock()
	if refuse != 0 {
		http.Error(w, "refused", refuse)
		return
	}
	c, err := websocket.Accept(w, r, nil)
	if err != nil {
		return
	}
	s.mu.Lock()
	s.conns = append(s.conns, c)
	s.dials = append(s.dials, dialed{path: r.URL.Path, query: r.URL.Query(), header: r.Header.Clone()})
	reply := s.reply
	s.mu.Unlock()
	defer func() { _ = c.CloseNow() }()
	ctx := r.Context()
	for {
		_, data, err := c.Read(ctx)
		if err != nil {
			return
		}
		var msg map[string]any
		if json.Unmarshal(data, &msg) != nil {
			return
		}
		s.sent <- msg
		if reply == nil {
			continue
		}
		for _, ev := range reply(msg) {
			typ := websocket.MessageText
			var b []byte
			if raw, ok := ev.([]byte); ok {
				typ, b = websocket.MessageBinary, raw
			} else {
				b, _ = json.Marshal(ev)
			}
			if c.Write(ctx, typ, b) != nil {
				return
			}
		}
	}
}

func (s *fakeServer) connections() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.conns)
}

// dial is the handshake of the i-th connection.
func (s *fakeServer) dial(i int) dialed {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.dials[i]
}

// drop ends every connection without a closing handshake.
func (s *fakeServer) drop() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, c := range s.conns {
		_ = c.CloseNow()
	}
}

func (s *fakeServer) next(t *testing.T) map[string]any {
	t.Helper()
	select {
	case msg := <-s.sent:
		return msg
	case <-time.After(3 * time.Second):
		t.Fatal("the client sent nothing")
		return nil
	}
}

// quiet reports whether the client sends nothing for a moment.
func (s *fakeServer) quiet() bool {
	select {
	case <-s.sent:
		return false
	case <-time.After(50 * time.Millisecond):
		return true
	}
}

// onText answers every sentence sent with events. The keepalive and the flush
// carry no sentence and are not answered.
func onText(events ...any) func(map[string]any) []any {
	return func(msg map[string]any) []any {
		text, _ := msg["text"].(string)
		if strings.TrimSpace(strings.TrimSuffix(text, stopMarker)) != "" {
			return events
		}
		return nil
	}
}

// audioMessage is one chunk as Neuphonic sends it.
func audioMessage(pcm []byte) map[string]any {
	return map[string]any{"data": map[string]any{
		"audio": base64.StdEncoding.EncodeToString(pcm),
		"stop":  false,
	}}
}

// fakeSvc stands in for the TTS service the synthesizer asks about its audio
// contexts.
type fakeSvc struct {
	mu     sync.Mutex
	active string
}

func (f *fakeSvc) ActiveAudioContextID() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.active
}

func (f *fakeSvc) setActive(id string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.active = id
}

func (f *fakeSvc) StopAllMetrics(context.Context) {}

// fakeHost stands in for the tts.Base the provider appends its audio to.
type fakeHost struct {
	mu     sync.Mutex
	audio  map[string][]byte
	errors []string
}

func newFakeHost() *fakeHost { return &fakeHost{audio: map[string][]byte{}} }

func (h *fakeHost) AppendToAudioContext(id string, f frames.Frame) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if a, ok := f.(*frames.TTSAudioRawFrame); ok {
		if a.ContextID != id {
			h.errors = append(h.errors, "frame labeled "+a.ContextID+" appended to "+id)
		}
		h.audio[id] = append(h.audio[id], a.Audio...)
	}
}

func (h *fakeHost) AppendWordToAudioContext(string, string, float64) {}

func (h *fakeHost) AddWordTimestamps(string, []uctx.WordTiming, tts.WordTimingOptions) {}

func (h *fakeHost) RemoveAudioContext(string) {}

func (h *fakeHost) AudioContextAvailable(string) bool { return true }

func (h *fakeHost) RotateTurnContext(context.Context) {}

func (h *fakeHost) ReplaceConnection() {}

func (h *fakeHost) PushFrame(context.Context, frames.Frame, processor.Direction) error { return nil }

func (h *fakeHost) PushError(_ context.Context, msg string, _ error, _ bool, _ ...processor.ErrorOption) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.errors = append(h.errors, msg)
}

func (h *fakeHost) ResetActiveAudioContext() {}

func (h *fakeHost) audioFor(id string) []byte {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.audio[id])
}

func (h *fakeHost) reported() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.errors)
}

func waitUntil(t *testing.T, msg string, cond func() bool) {
	t.Helper()
	deadline := time.After(3 * time.Second)
	for !cond() {
		select {
		case <-deadline:
			t.Fatal(msg)
		case <-time.After(5 * time.Millisecond):
		}
	}
}

// connectedSynth builds a service for server, with the synthesizer's host and
// service replaced by fakes, and starts it.
func connectedSynth(t *testing.T, server *fakeServer, cfg Config) (*synthesizer, *fakeHost, *fakeSvc) {
	t.Helper()
	cfg.APIKey, cfg.URL = "test-key", server.url
	_, s := newTTS(cfg)
	host, svc := newFakeHost(), &fakeSvc{}
	s.host, s.svc = host, svc
	s.Start(context.Background())
	if !s.Connected() {
		t.Fatal("the service did not connect when it started")
	}
	t.Cleanup(func() { s.Disconnect(context.Background()) })
	return s, host, svc
}

// updateSettings applies delta the way the base does, then hands the change to
// the synthesizer.
func updateSettings(t *testing.T, s *synthesizer, delta *TTSSettings) {
	t.Helper()
	changed, err := settings.Apply(s.live, delta)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.UpdateSettings(context.Background(), changed); err != nil {
		t.Fatal(err)
	}
}

// The connection opens when the service starts, configured by its URL: the
// language in the path, the synthesis options in the query, and the key in a
// header. Options left unset are not sent.
func TestStartOpensAConnection(t *testing.T) {
	server := newFakeServer(t, nil)
	connectedSynth(t, server, Config{})

	d := server.dial(0)
	if d.header.Get("x-api-key") != "test-key" {
		t.Errorf("x-api-key = %q, want the configured key", d.header.Get("x-api-key"))
	}
	if d.path != "/speak/en" {
		t.Errorf("path = %s, want the language in the path", d.path)
	}
	for field, want := range map[string]string{
		"lang_code":     "en",
		"speed":         "1",
		"encoding":      defaultEncoding,
		"sampling_rate": "22050",
	} {
		if got := d.query.Get(field); got != want {
			t.Errorf("%s = %q, want %q", field, got, want)
		}
	}
	for _, field := range []string{"voice_id", "temperature", "ipa_key"} {
		if d.query.Has(field) {
			t.Errorf("%s was sent though none was configured", field)
		}
	}
	if !server.quiet() {
		t.Error("the service sent something on connecting; everything goes in the URL")
	}
}

// The options a caller sets reach the endpoint.
func TestConnectionCarriesTheConfiguredOptions(t *testing.T) {
	server := newFakeServer(t, nil)
	temperature := 0.4
	connectedSynth(t, server, Config{
		VoiceID:     "voice-9",
		Language:    language.French,
		Speed:       1.2,
		Temperature: &temperature,
		IPAKey:      "pt-br",
		SampleRate:  16000,
	})

	d := server.dial(0)
	if d.path != "/speak/fr" {
		t.Errorf("path = %s, want the language in the path", d.path)
	}
	for field, want := range map[string]string{
		"lang_code":     "fr",
		"speed":         "1.2",
		"temperature":   "0.4",
		"sampling_rate": "16000",
		"voice_id":      "voice-9",
		"ipa_key":       "pt-br",
	} {
		if got := d.query.Get(field); got != want {
			t.Errorf("%s = %q, want %q", field, got, want)
		}
	}
}

// Every sentence goes out on the one connection with the marker that has it
// generated at once, and its audio comes back labeled with the context playing.
func TestSentencesShareTheConnection(t *testing.T) {
	server := newFakeServer(t, onText(audioMessage([]byte{1, 2}), audioMessage([]byte{3})))
	s, host, svc := connectedSynth(t, server, Config{})

	ctx := context.Background()
	for _, id := range []string{"c1", "c2"} {
		svc.setActive(id)
		if err := s.RunTTS(ctx, "Hello "+id+".", id, nil); err != nil {
			t.Fatal(err)
		}
		if msg := server.next(t); msg["text"] != "Hello "+id+". <STOP>" {
			t.Errorf("sent %v, want the sentence with the stop marker", msg)
		}
		waitUntil(t, "the audio never arrived", func() bool {
			return bytes.Equal(host.audioFor(id), []byte{1, 2, 3})
		})
	}
	if n := server.connections(); n != 1 {
		t.Errorf("the server saw %d connections, want 1", n)
	}
	if errs := host.reported(); len(errs) != 0 {
		t.Errorf("errors reported: %v", errs)
	}
}

// Messages that carry no audio, and binary messages, are stepped over, and the
// connection carries on.
func TestMessagesWithoutAudioAreIgnored(t *testing.T) {
	server := newFakeServer(t, onText(
		map[string]any{"message": "connected"},
		map[string]any{"data": map[string]any{"audio": "", "stop": true}},
		[]byte{9, 9},
		audioMessage([]byte{7, 7}),
	))
	s, host, svc := connectedSynth(t, server, Config{})
	svc.setActive("c1")

	if err := s.RunTTS(context.Background(), "Hello.", "c1", nil); err != nil {
		t.Fatal(err)
	}
	waitUntil(t, "the audio never arrived", func() bool { return len(host.audioFor("c1")) > 0 })
	if got := host.audioFor("c1"); !bytes.Equal(got, []byte{7, 7}) {
		t.Errorf("audio = %v, want the one real chunk", got)
	}
	if n := server.connections(); n != 1 {
		t.Errorf("the server saw %d connections, want 1", n)
	}
}

// A message that is not JSON, or audio that is not base64, ends the reading as
// a failed connection rather than being played as rubbish.
func TestUnreadableMessagesFailTheConnection(t *testing.T) {
	s := newSynthesizer(Config{APIKey: "k"})
	host := newFakeHost()
	s.host, s.svc = host, &fakeSvc{active: "c1"}

	if err := s.handleMessage([]byte("not json")); err == nil {
		t.Error("a message that is not JSON was accepted")
	}
	if err := s.handleMessage([]byte(`{"data":{"audio":"not base64!!"}}`)); err == nil {
		t.Error("audio that is not base64 was accepted")
	}
	if got := host.audioFor("c1"); len(got) != 0 {
		t.Errorf("audio = %v, want none", got)
	}
}

// The keepalive is an empty text.
func TestKeepalive(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, Config{})

	if !s.KeepaliveReady() {
		t.Fatal("the keepalive is not ready on an open connection")
	}
	if err := s.SendKeepalive(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := server.next(t); !maps.Equal(got, map[string]any{"text": ""}) {
		t.Errorf("keepalive = %v, want an empty text", got)
	}
	if !s.ws.KeepaliveRunning() {
		t.Error("the keepalive loop is not running while connected")
	}
}

// A flush sends the stop marker alone, which has Neuphonic generate what it
// holds.
func TestFlushAudio(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, Config{})

	s.FlushAudio(context.Background(), "")
	if got := server.next(t); !maps.Equal(got, map[string]any{"text": stopMarker}) {
		t.Errorf("flush = %v, want the stop marker", got)
	}
}

// Every setting travels in the URL, so a change to any of them opens a new
// connection carrying it.
func TestSettingsChangeReconnects(t *testing.T) {
	for field, tc := range map[string]struct {
		delta *TTSSettings
		param string
		want  string
	}{
		"voice":       {&TTSSettings{Voice: settings.Set("v2")}, "voice_id", "v2"},
		"language":    {&TTSSettings{Language: settings.Set("es")}, "lang_code", "es"},
		"speed":       {&TTSSettings{Speed: settings.Set(1.5)}, "speed", "1.5"},
		"temperature": {&TTSSettings{Temperature: settings.Set(0.7)}, "temperature", "0.7"},
		"ipa_key":     {&TTSSettings{IPAKey: settings.Set("es-castilian")}, "ipa_key", "es-castilian"},
	} {
		t.Run(field, func(t *testing.T) {
			server := newFakeServer(t, nil)
			s, _, _ := connectedSynth(t, server, Config{VoiceID: "v1"})

			updateSettings(t, s, tc.delta)
			if n := server.connections(); n != 2 {
				t.Fatalf("%s must reconnect: the server saw %d connections", field, n)
			}
			if got := server.dial(1).query.Get(tc.param); got != tc.want {
				t.Errorf("new connection %s = %q, want %q", tc.param, got, tc.want)
			}
			if !s.Connected() {
				t.Error("not connected after the change")
			}
		})
	}
}

// Re-sending what the service already has changes nothing, and keeps the
// connection.
func TestUnchangedSettingsKeepTheConnection(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, Config{VoiceID: "v1"})

	updateSettings(t, s, &TTSSettings{Voice: settings.Set("v1")})
	if n := server.connections(); n != 1 {
		t.Errorf("an unchanged voice reconnected: %d connections", n)
	}
}

// A connection the server drops is restored by the receive loop.
func TestReconnectsWhenTheServerDrops(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, Config{Websocket: wsservice.Options{
		ReconnectBackoffMinWait: time.Millisecond,
		ReconnectBackoffMaxWait: time.Millisecond,
	}})

	server.drop()
	waitUntil(t, "the service never reconnected", func() bool {
		return server.connections() == 2 && s.Connected()
	})
}

// A sentence after the connection closed opens it again before it is sent.
func TestSentenceReopensAClosedConnection(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, Config{})
	s.Disconnect(context.Background())

	if err := s.RunTTS(context.Background(), "Hello.", "c1", nil); err != nil {
		t.Fatal(err)
	}
	if got := server.next(t); got["text"] != "Hello. <STOP>" {
		t.Errorf("sent %v, want the sentence", got)
	}
	if n := server.connections(); n != 2 {
		t.Errorf("the server saw %d connections, want the closed one opened again", n)
	}
}

// A send that fails stops the context's speech, is reported, and resets the
// connection.
func TestFailedSendResetsTheConnection(t *testing.T) {
	server := newFakeServer(t, nil)
	_, s := newTTS(Config{APIKey: "k", URL: server.url})
	s.host, s.svc = newFakeHost(), &fakeSvc{}
	t.Cleanup(func() { s.Disconnect(context.Background()) })

	// A connection that is already closed but not yet known to be: the send is
	// where the failure shows.
	conn, err := s.ws.Dial(context.Background(), server.url, http.Header{}, wsutil.DefaultReadLimit)
	if err != nil {
		t.Fatal(err)
	}
	_ = conn.CloseNow()
	s.conn = &wsConn{conn: conn, reads: make(chan wsRead), stop: make(chan struct{})}

	var got []frames.Frame
	err = s.RunTTS(context.Background(), "Hello.", "c1", func(f frames.Frame) error {
		got = append(got, f)
		return nil
	})
	if err == nil {
		t.Fatal("RunTTS accepted a failed send")
	}
	if len(got) != 1 {
		t.Fatalf("yielded %v, want one stop frame", got)
	}
	if stopped, ok := got[0].(*frames.TTSStoppedFrame); !ok || stopped.ContextID != "c1" {
		t.Errorf("yielded %v, want the context's speech stopped", got[0])
	}
	if !s.Connected() || server.connections() != 2 {
		t.Errorf("connected %v with %d connections, want a fresh one", s.Connected(), server.connections())
	}
}

// TestNeuphonicLanguage checks the codes the provider names languages by. Hindi
// is the one it spells in capitals; everything else is the base code, and an
// unset language is English.
func TestNeuphonicLanguage(t *testing.T) {
	tests := []struct {
		in   language.Language
		want string
	}{
		{in: "", want: "en"},
		{in: language.English, want: "en"},
		{in: language.EnglishUS, want: "en"},
		{in: language.Hindi, want: "HI"},
		{in: language.French, want: "fr"},
		{in: language.Spanish, want: "es"},
	}
	for _, tt := range tests {
		if got := neuphonicLanguage(tt.in); got != tt.want {
			t.Errorf("neuphonicLanguage(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

// TestSampleRate checks the rate the frames are stamped with is the one asked
// of the provider.
func TestSampleRate(t *testing.T) {
	s := newSynthesizer(Config{SampleRate: 16000})
	if got := s.SampleRate(); got != 16000 {
		t.Errorf("SampleRate() = %d, want the configured rate", got)
	}
}

// run is a running pipeline around the service.
type run struct {
	t    *testing.T
	task *pipeline.Worker
	done chan error

	mu   sync.Mutex
	down []frames.Frame
}

func runPipeline(t *testing.T, b *tts.Base) *run {
	t.Helper()
	r := &run{t: t, done: make(chan error, 1)}
	r.task = pipeline.NewWorker(pipeline.New(b), pipeline.WorkerConfig{ReachedDownstreamFilter: pipeline.AnyFrame})
	events.On(&r.task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		r.mu.Lock()
		r.down = append(r.down, f)
		r.mu.Unlock()
	})
	go func() { r.done <- r.task.Run(context.Background()) }()
	r.waitFor(func(f frames.Frame) bool { _, ok := f.(*frames.StartFrame); return ok })
	return r
}

func (r *run) waitFor(match func(frames.Frame) bool) {
	r.t.Helper()
	waitUntil(r.t, "the frame waited for never came", func() bool {
		r.mu.Lock()
		defer r.mu.Unlock()
		return slices.ContainsFunc(r.down, match)
	})
}

func (r *run) stop() {
	r.t.Helper()
	r.task.StopWhenDone()
	r.wait()
}

// cancel ends the pipeline at once, without waiting for audio still playing.
func (r *run) cancel() {
	r.t.Helper()
	r.task.Cancel(context.Background(), "")
	r.wait()
}

func (r *run) wait() {
	r.t.Helper()
	select {
	case <-r.done:
	case <-time.After(5 * time.Second):
		r.t.Fatal("the pipeline did not finish")
	}
}

// The service connects when the pipeline starts and its audio reaches the
// pipeline over that one connection, labeled with the context it was asked
// for.
func TestHoldsOneConnectionForThePipeline(t *testing.T) {
	server := newFakeServer(t, onText(audioMessage(make([]byte, 480))))
	b := NewTTS(Config{APIKey: "k", URL: server.url, SampleRate: 24000})
	r := runPipeline(t, b)
	if n := server.connections(); n != 1 {
		t.Fatalf("the server saw %d connections when the pipeline started, want 1", n)
	}

	r.task.QueueFrame(frames.NewTTSSpeakFrame("Hello there."))
	r.waitFor(func(f frames.Frame) bool {
		a, ok := f.(*frames.TTSAudioRawFrame)
		return ok && a.ContextID != ""
	})
	r.cancel()

	if n := server.connections(); n != 1 {
		t.Errorf("the server saw %d connections, want one for the whole pipeline", n)
	}
}

// A connection the server refuses is reported as a connection error.
func TestReportsAFailedConnection(t *testing.T) {
	server := newFakeServer(t, nil)
	server.refuse = http.StatusUnauthorized
	b := NewTTS(Config{APIKey: "k", URL: server.url})
	var mu sync.Mutex
	var reported []string
	events.On(b.Events(), service.EventConnectionError, func(_ context.Context, msg string) {
		mu.Lock()
		reported = append(reported, msg)
		mu.Unlock()
	})

	r := runPipeline(t, b)
	defer r.stop()
	waitUntil(t, "the refused connection was never reported", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(reported) > 0
	})
}

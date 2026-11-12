package asyncai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"maps"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsservice"
	uctx "github.com/gojargo/jargo/utils/context"
	"github.com/gojargo/jargo/utils/events"
)

func TestValidate(t *testing.T) {
	if err := (Config{APIKey: "k", Voice: "v"}).Validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	if err := (Config{Voice: "v"}).Validate(); err == nil {
		t.Fatal("config without APIKey should be rejected")
	}
	if err := (Config{APIKey: "k"}).Validate(); err == nil {
		t.Fatal("config without Voice should be rejected")
	}
}

// fakeServer is a fake Async endpoint. It records every message the client
// sends and answers each with what reply returns, stamping the context the
// message named.
type fakeServer struct {
	url  string
	sent chan map[string]any

	mu     sync.Mutex
	reply  func(msg map[string]any) []map[string]any
	refuse int
	conns  []*websocket.Conn
}

func newFakeServer(t *testing.T, reply func(msg map[string]any) []map[string]any) *fakeServer {
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
			ev = maps.Clone(ev)
			if _, ok := ev["context_id"]; !ok {
				ev["context_id"] = msg["context_id"]
			}
			b, _ := json.Marshal(ev)
			if c.Write(ctx, websocket.MessageText, b) != nil {
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

// waitFor reads the client's messages until one matches, returning it.
func (s *fakeServer) waitFor(t *testing.T, match func(map[string]any) bool) map[string]any {
	t.Helper()
	for {
		if msg := s.next(t); match(msg) {
			return msg
		}
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

func isInit(msg map[string]any) bool { _, ok := msg["model_id"]; return ok }

// onTranscript answers every sentence sent with events.
func onTranscript(events ...map[string]any) func(map[string]any) []map[string]any {
	return func(msg map[string]any) []map[string]any {
		if text, _ := msg["transcript"].(string); strings.TrimSpace(text) != "" {
			return events
		}
		return nil
	}
}

func audioMessage(pcm []byte) map[string]any {
	return map[string]any{"audio": base64.StdEncoding.EncodeToString(pcm)}
}

// fakeSvc stands in for the TTS service the synthesizer asks about its audio
// contexts.
type fakeSvc struct {
	mu     sync.Mutex
	active string
	warned []string
}

func (f *fakeSvc) ActiveAudioContextID() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.active
}

func (f *fakeSvc) HasActiveAudioContext() bool { return f.ActiveAudioContextID() != "" }

func (f *fakeSvc) StopAllMetrics(context.Context) {}

func (f *fakeSvc) WarnUnhandledUpdatedSettings(_ context.Context, unhandled []string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.warned = append(f.warned, unhandled...)
}

// fakeHost stands in for the tts.Base the provider appends its audio to. Only
// the contexts opened on it take frames.
type fakeHost struct {
	mu     sync.Mutex
	open   map[string]bool
	audio  map[string][]byte
	errors []string
}

func newFakeHost(ids ...string) *fakeHost {
	h := &fakeHost{open: map[string]bool{}, audio: map[string][]byte{}}
	for _, id := range ids {
		h.open[id] = true
	}
	return h
}

func (h *fakeHost) AppendToAudioContext(id string, f frames.Frame) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if a, ok := f.(*frames.TTSAudioRawFrame); ok {
		h.audio[id] = append(h.audio[id], a.Audio...)
	}
}

func (h *fakeHost) AppendWordToAudioContext(string, string, float64) {}

func (h *fakeHost) AddWordTimestamps(string, []uctx.WordTiming, tts.WordTimingOptions) {}

func (h *fakeHost) RemoveAudioContext(id string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	delete(h.open, id)
}

func (h *fakeHost) AudioContextAvailable(id string) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.open[id]
}

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
	cfg.APIKey, cfg.URL = "k", server.url
	if cfg.Voice == "" {
		cfg.Voice = "v1"
	}
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
func updateSettings(t *testing.T, s *synthesizer, delta *settings.TTS) {
	t.Helper()
	changed, err := settings.Apply(s.live, delta)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.UpdateSettings(context.Background(), changed); err != nil {
		t.Fatal(err)
	}
}

// The session opens when the service starts, with an init message carrying the
// model, the voice, the output format and the language.
func TestStartOpensASession(t *testing.T) {
	server := newFakeServer(t, nil)
	connectedSynth(t, server, Config{Model: "m1", Language: "fr", SampleRate: 16000})

	init := server.next(t)
	if init["model_id"] != "m1" || init["language"] != "fr" {
		t.Errorf("init = %v, want model m1 and language fr", init)
	}
	if voice, _ := init["voice"].(map[string]any); voice["id"] != "v1" || voice["mode"] != "id" {
		t.Errorf("init voice = %v, want id v1", init["voice"])
	}
	format, _ := init["output_format"].(map[string]any)
	if format["encoding"] != defaultEncoding || format["container"] != defaultContainer ||
		format["sample_rate"] != 16000.0 {
		t.Errorf("init output format = %v", format)
	}
}

// Every sentence goes out on the one connection, on its own context, and its
// audio comes back labeled with that context.
func TestSentencesShareTheConnection(t *testing.T) {
	server := newFakeServer(t, onTranscript(audioMessage([]byte{1, 2})))
	s, host, _ := connectedSynth(t, server, Config{})
	server.next(t)
	host.open["c1"], host.open["c2"] = true, true

	ctx := context.Background()
	for _, id := range []string{"c1", "c2"} {
		if err := s.RunTTS(ctx, "Hello "+id+".", id, nil); err != nil {
			t.Fatal(err)
		}
		msg := server.next(t)
		if msg["transcript"] != "Hello "+id+"." || msg["context_id"] != id || msg["force"] != true {
			t.Errorf("sent %v, want the sentence forced on %s", msg, id)
		}
	}
	waitUntil(t, "the audio never arrived", func() bool {
		return bytes.Equal(host.audioFor("c1"), []byte{1, 2}) && bytes.Equal(host.audioFor("c2"), []byte{1, 2})
	})
	if n := server.connections(); n != 1 {
		t.Errorf("the server saw %d connections, want 1", n)
	}
}

// Audio for a context that is no longer open is dropped, unless it is the
// active one, which the base opens again.
func TestAudioForAClosedContext(t *testing.T) {
	server := newFakeServer(t, nil)
	s, host, svc := connectedSynth(t, server, Config{})
	opened := host.AudioContextAvailable

	if err := s.handleMessage([]byte(`{"context_id":"gone","audio":"AQI="}`)); err != nil {
		t.Fatal(err)
	}
	if got := host.audioFor("gone"); len(got) != 0 || opened("gone") {
		t.Errorf("audio for a closed context reached the host: %v", got)
	}

	svc.active = "late"
	if err := s.handleMessage([]byte(`{"context_id":"late","audio":"AQI="}`)); err != nil {
		t.Fatal(err)
	}
	if got := host.audioFor("late"); !bytes.Equal(got, []byte{1, 2}) {
		t.Errorf("audio for the active context = %v, want it appended", got)
	}
}

// The final message arrives after the context has been closed, and is ignored.
func TestFinalMessageIsIgnored(t *testing.T) {
	s := newSynthesizer(Config{APIKey: "k", Voice: "v"})
	host := newFakeHost("c1")
	s.host, s.svc = host, &fakeSvc{}
	if err := s.handleMessage([]byte(`{"context_id":"c1","final":true,"audio":"AQI="}`)); err != nil {
		t.Fatal(err)
	}
	if got := host.audioFor("c1"); len(got) != 0 || !host.AudioContextAvailable("c1") {
		t.Errorf("final message acted on: audio %v, open %v", got, host.AudioContextAvailable("c1"))
	}
}

// The keepalive is a blank transcript, on the active context when there is one.
func TestKeepalive(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, svc := connectedSynth(t, server, Config{})
	server.next(t)
	ctx := context.Background()

	if err := s.SendKeepalive(ctx); err != nil {
		t.Fatal(err)
	}
	if got := server.next(t); !maps.Equal(got, map[string]any{"transcript": " "}) {
		t.Errorf("keepalive = %v, want a blank transcript naming no context", got)
	}
	svc.active = "c1"
	if err := s.SendKeepalive(ctx); err != nil {
		t.Fatal(err)
	}
	if got := server.next(t); !maps.Equal(got, map[string]any{"transcript": " ", "context_id": "c1"}) {
		t.Errorf("keepalive = %v, want a blank transcript on c1", got)
	}
}

// A flush forces out what the server holds for the context.
func TestFlushAudio(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, svc := connectedSynth(t, server, Config{})
	server.next(t)
	svc.active = "c1"

	s.FlushAudio(context.Background(), "")
	if got := server.next(t); !maps.Equal(got, map[string]any{"transcript": " ", "context_id": "c1", "force": true}) {
		t.Errorf("flush = %v, want a forced blank transcript on the active context", got)
	}
}

// A context is closed on the server when the user cuts it off, and when its
// audio has played: Async never says a context is exhausted.
func TestContextsAreClosed(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, Config{})
	server.next(t)
	ctx := context.Background()

	s.OnAudioContextInterrupted(ctx, "c1")
	s.OnAudioContextCompleted(ctx, "c2")
	for _, id := range []string{"c1", "c2"} {
		want := map[string]any{"context_id": id, "close_context": true, "transcript": ""}
		if got := server.next(t); !maps.Equal(got, want) {
			t.Errorf("sent %v, want %s closed", got, id)
		}
	}
}

// Disconnecting with a context still playing terminates the contexts first.
func TestDisconnectTerminatesAnActiveContext(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, svc := connectedSynth(t, server, Config{})
	server.next(t)
	svc.active = "c1"

	s.Disconnect(context.Background())
	if got := server.next(t); !maps.Equal(got, map[string]any{"terminate": true}) {
		t.Errorf("sent %v on disconnect, want terminate", got)
	}
	if s.Connected() {
		t.Error("still connected after disconnecting")
	}
}

// Without a context playing there is nothing to terminate.
func TestDisconnectWithoutAnActiveContext(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, Config{})
	server.next(t)

	s.Disconnect(context.Background())
	if !server.quiet() {
		t.Error("the service sent something on disconnecting with no context playing")
	}
}

// The model, the voice and the language are sent only in the init message, so
// a change to any of them starts a new session.
func TestSessionInitFieldChangeStartsANewSession(t *testing.T) {
	for field, delta := range map[string]*settings.TTS{
		"voice":    {Voice: settings.Set("v2")},
		"model":    {Base: settings.Base{Model: settings.Set("m2")}},
		"language": {Language: settings.Set("es")},
	} {
		t.Run(field, func(t *testing.T) {
			server := newFakeServer(t, nil)
			s, _, _ := connectedSynth(t, server, Config{Model: "m1"})
			server.next(t)

			updateSettings(t, s, delta)
			server.waitFor(t, isInit)
			if n := server.connections(); n != 2 {
				t.Errorf("%s must rebuild the session: the server saw %d connections", field, n)
			}
		})
	}
}

// Re-sending what the service already has changes nothing, and keeps the
// session.
func TestUnchangedSettingsKeepTheSession(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, Config{})
	server.next(t)

	updateSettings(t, s, &settings.TTS{Voice: settings.Set("v1")})
	if n := server.connections(); n != 1 || !server.quiet() {
		t.Errorf("an unchanged voice reconnected: %d connections", n)
	}
}

// The new session is only worth anything if its init message carries the new
// value, and the change opens exactly one.
func TestNewSessionCarriesTheUpdatedVoice(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, Config{})
	if voice, _ := server.next(t)["voice"].(map[string]any); voice["id"] != "v1" {
		t.Fatalf("first session voice = %v, want v1", voice)
	}

	updateSettings(t, s, &settings.TTS{Voice: settings.Set("v2")})
	init := server.next(t)
	if voice, _ := init["voice"].(map[string]any); voice["id"] != "v2" {
		t.Errorf("new session voice = %v, want v2", voice)
	}
	if !server.quiet() || server.connections() != 2 {
		t.Errorf("the change opened %d sessions in all, want exactly one new one", server.connections())
	}
}

// Anything outside the init message cannot be applied to a live session, so it
// is reported as not taking effect.
func TestAFieldThatIsNotInTheInitMessageStillWarns(t *testing.T) {
	s := newSynthesizer(Config{APIKey: "k", Voice: "v1"})
	svc := &fakeSvc{}
	s.svc = svc
	delta := &settings.TTS{}
	delta.Extra = map[string]any{"pace": 1.2}
	changed, err := settings.Apply(s.live, delta)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.UpdateSettings(context.Background(), changed); err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(svc.warned, "pace") {
		t.Errorf("warned about %v, want pace", svc.warned)
	}
}

// A connection the server drops is restored by the receive loop, with a fresh
// session.
func TestReconnectsWhenTheServerDrops(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, Config{Websocket: wsservice.Options{
		ReconnectBackoffMinWait: time.Millisecond,
		ReconnectBackoffMaxWait: time.Millisecond,
	}})
	server.next(t)

	server.drop()
	server.waitFor(t, isInit)
	waitUntil(t, "the service never reconnected", func() bool { return s.Connected() })
	if n := server.connections(); n != 2 {
		t.Errorf("the server saw %d connections, want 2", n)
	}
}

// A sentence after the connection closed opens it again before it is sent.
func TestSentenceReopensAClosedConnection(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, Config{})
	server.next(t)
	s.Disconnect(context.Background())

	if err := s.RunTTS(context.Background(), "Hello.", "c1", nil); err != nil {
		t.Fatal(err)
	}
	if !isInit(server.next(t)) {
		t.Fatal("the connection was not opened again with an init message")
	}
	if got := server.next(t); got["transcript"] != "Hello." {
		t.Errorf("sent %v, want the sentence", got)
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
// pipeline over that one connection.
func TestHoldsOneConnectionForThePipeline(t *testing.T) {
	server := newFakeServer(t, onTranscript(audioMessage(make([]byte, 480))))
	b := NewTTS(Config{APIKey: "k", Voice: "v1", URL: server.url, SampleRate: 24000})
	r := runPipeline(t, b)
	if n := server.connections(); n != 1 {
		t.Fatalf("the server saw %d connections when the pipeline started, want 1", n)
	}

	r.task.QueueFrame(frames.NewTTSSpeakFrame("Hello there."))
	r.waitFor(func(f frames.Frame) bool { _, ok := f.(*frames.TTSAudioRawFrame); return ok })
	r.cancel()

	if n := server.connections(); n != 1 {
		t.Errorf("the server saw %d connections, want one for the whole pipeline", n)
	}
}

// A connection the server refuses is reported as a connection error.
func TestReportsAFailedConnection(t *testing.T) {
	server := newFakeServer(t, nil)
	server.refuse = http.StatusUnauthorized
	b := NewTTS(Config{APIKey: "k", Voice: "v1", URL: server.url})
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

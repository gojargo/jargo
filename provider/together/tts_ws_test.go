package together

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
	uctx "github.com/gojargo/jargo/utils/context"
	"github.com/gojargo/jargo/utils/events"
)

// Tests for the connection held open for the call. Together streams 24 kHz PCM
// over an OpenAI-compatible realtime socket: the voice is pinned on the session
// once it opens, each sentence is appended to the text buffer and committed,
// and the audio comes back base64 in deltas naming no request, until the
// session says the request is done.

const ctxID = "ctx-1"

// fakeServer is a fake Together endpoint. It records every connection, and
// every message the client sends, and answers each with what reply returns.
type fakeServer struct {
	url  string
	sent chan map[string]any

	mu      sync.Mutex
	reply   func(msg map[string]any) []map[string]any
	refuse  int
	conns   []*websocket.Conn
	queries []url.Values
	headers []http.Header
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
	s.queries = append(s.queries, r.URL.Query())
	s.headers = append(s.headers, r.Header.Clone())
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
			if s.write(ctx, c, ev) != nil {
				return
			}
		}
	}
}

func (s *fakeServer) write(ctx context.Context, c *websocket.Conn, ev map[string]any) error {
	b, _ := json.Marshal(ev)
	return c.Write(ctx, websocket.MessageText, b)
}

// push sends ev on the latest connection.
func (s *fakeServer) push(t *testing.T, ev map[string]any) {
	t.Helper()
	s.mu.Lock()
	c := s.conns[len(s.conns)-1]
	s.mu.Unlock()
	if err := s.write(context.Background(), c, ev); err != nil {
		t.Fatal(err)
	}
}

func (s *fakeServer) connections() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.conns)
}

func (s *fakeServer) query(i int) url.Values {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.queries[i]
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

// speaking answers each commit with a chunk of audio and the done event.
func speaking(msg map[string]any) []map[string]any {
	if msg["type"] != "input_text_buffer.commit" {
		return nil
	}
	return []map[string]any{delta(make([]byte, 3200)), done()}
}

// delta is an audio delta as Together sends it.
func delta(pcm []byte) map[string]any {
	return map[string]any{
		"type":  "conversation.item.audio_output.delta",
		"delta": base64.StdEncoding.EncodeToString(pcm),
	}
}

// done ends the synthesis of a request.
func done() map[string]any {
	return map[string]any{"type": "conversation.item.audio_output.done", "item_id": "item-1"}
}

// fakeSvc stands in for the TTS service the synthesizer asks about its audio
// contexts.
type fakeSvc struct {
	mu      sync.Mutex
	active  string
	stopped int
}

func (f *fakeSvc) ActiveAudioContextID() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.active
}

func (f *fakeSvc) StopAllMetrics(context.Context) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.stopped++
}

func (f *fakeSvc) stops() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.stopped
}

// fakeHost stands in for the tts.Base the provider appends its audio to.
type fakeHost struct {
	mu       sync.Mutex
	appended map[string][]frames.Frame
	removed  []string
	pushed   []frames.Frame
	errors   []string
}

func newFakeHost() *fakeHost { return &fakeHost{appended: map[string][]frames.Frame{}} }

func (h *fakeHost) AppendToAudioContext(id string, f frames.Frame) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.appended[id] = append(h.appended[id], f)
}

func (h *fakeHost) AppendWordToAudioContext(string, string, float64) {}

func (h *fakeHost) AddWordTimestamps(string, []uctx.WordTiming, tts.WordTimingOptions) {}

func (h *fakeHost) RemoveAudioContext(id string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.removed = append(h.removed, id)
}

func (h *fakeHost) AudioContextAvailable(string) bool { return true }

func (h *fakeHost) RotateTurnContext(context.Context) {}

func (h *fakeHost) ReplaceConnection() {}

func (h *fakeHost) PushFrame(_ context.Context, f frames.Frame, _ processor.Direction) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.pushed = append(h.pushed, f)
	return nil
}

func (h *fakeHost) PushError(_ context.Context, msg string, _ error, _ bool, _ ...processor.ErrorOption) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.errors = append(h.errors, msg)
}

func (h *fakeHost) ResetActiveAudioContext() {}

func (h *fakeHost) framesFor(id string) []frames.Frame {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.appended[id])
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
// service replaced by fakes, starts it and takes the session update it sends
// on connecting.
func connectedSynth(t *testing.T, server *fakeServer, cfg TTSConfig) (*synthesizer, *fakeHost, *fakeSvc) {
	t.Helper()
	cfg.APIKey, cfg.URL = "test-key", server.url
	_, s := newTTS(cfg)
	host, svc := newFakeHost(), &fakeSvc{active: ctxID}
	s.host, s.svc = host, svc
	s.Start(context.Background())
	if !s.Connected() {
		t.Fatal("the service did not connect when it started")
	}
	t.Cleanup(func() { s.Disconnect(context.Background()) })
	if got := server.next(t); got["type"] != "tts_session.updated" {
		t.Fatalf("first message = %v, want the session update", got)
	}
	return s, host, svc
}

// handle hands an event to the synthesizer as the receive loop does.
func handle(t *testing.T, s *synthesizer, ev map[string]any) error {
	t.Helper()
	data, err := json.Marshal(ev)
	if err != nil {
		t.Fatal(err)
	}
	return s.handleMessage(context.Background(), data)
}

// The connection is opened when the service starts, authorized with the key,
// named the model, the voice and the language on its URL, and has the voice
// pinned on its session.
func TestConnectPinsTheVoice(t *testing.T) {
	server := newFakeServer(t, nil)
	cfg := TTSConfig{APIKey: "test-key", URL: server.url, ExtraParams: map[string]any{
		"pronunciation_dict": []any{"jargo/jar go"},
	}}
	_, s := newTTS(cfg)
	s.host, s.svc = newFakeHost(), &fakeSvc{}
	s.Start(context.Background())
	t.Cleanup(func() { s.Disconnect(context.Background()) })

	want := map[string]any{"type": "tts_session.updated", "session": map[string]any{
		"voice":        defaultTTSVoice,
		"extra_params": map[string]any{"pronunciation_dict": []any{"jargo/jar go"}},
	}}
	if got := server.next(t); !reflect.DeepEqual(got, want) {
		t.Errorf("session update = %v, want %v", got, want)
	}
	server.mu.Lock()
	header := server.headers[0]
	server.mu.Unlock()
	if auth := header.Get("Authorization"); auth != "Bearer test-key" {
		t.Errorf("Authorization = %q, want the key as a bearer token", auth)
	}
	q := server.query(0)
	if q.Get("model") != defaultTTSModel || q.Get("voice") != defaultTTSVoice || q.Get("language") != "en" {
		t.Errorf("query = %v, want the default model, voice and language", q)
	}
}

// Without extra options the session update carries only the voice.
func TestSessionUpdateWithoutExtraParams(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, TTSConfig{})
	var msg map[string]any
	if err := json.Unmarshal(s.sessionUpdate(), &msg); err != nil {
		t.Fatal(err)
	}
	session, _ := msg["session"].(map[string]any)
	if !reflect.DeepEqual(session, map[string]any{"voice": defaultTTSVoice}) {
		t.Errorf("session = %v, want only the voice", session)
	}
}

// A sentence is appended to the text buffer and committed on the open
// connection, and the call returns without waiting for its audio.
func TestSentenceIsAppendedAndCommitted(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, TTSConfig{})

	if err := s.RunTTS(context.Background(), "hello there", ctxID, nil); err != nil {
		t.Fatal(err)
	}
	if got := server.next(t); got["type"] != "input_text_buffer.append" || got["text"] != "hello there" {
		t.Errorf("sent %v, want the text appended", got)
	}
	if got := server.next(t); got["type"] != "input_text_buffer.commit" {
		t.Errorf("sent %v, want the commit", got)
	}
	if n := server.connections(); n != 1 {
		t.Errorf("the server saw %d connections, want the one opened at start", n)
	}
}

// Audio is labeled with the context playing, an empty delta is stepped over,
// and audio that is not base64 is reported without failing the connection.
func TestAudioDeltas(t *testing.T) {
	s := newSynthesizer(TTSConfig{APIKey: "k"})
	host, svc := newFakeHost(), &fakeSvc{active: ctxID}
	s.host, s.svc = host, svc

	if err := handle(t, s, delta([]byte{1, 2, 3, 4})); err != nil {
		t.Fatal(err)
	}
	if err := handle(t, s, map[string]any{"type": "conversation.item.audio_output.delta", "delta": ""}); err != nil {
		t.Fatal(err)
	}
	undecodable := map[string]any{"type": "conversation.item.audio_output.delta", "delta": "not base64!!"}
	if err := handle(t, s, undecodable); err != nil {
		t.Errorf("an undecodable delta failed the connection: %v", err)
	}
	got := host.framesFor(ctxID)
	if len(got) != 1 {
		t.Fatalf("appended %v, want the one real delta", got)
	}
	a, ok := got[0].(*frames.TTSAudioRawFrame)
	if !ok || string(a.Audio) != string([]byte{1, 2, 3, 4}) || a.ContextID != ctxID || a.SampleRate != ttsSampleRate {
		t.Errorf("appended %v, want the audio on the active context at 24 kHz", got[0])
	}
	if errs := host.reported(); len(errs) != 1 || !strings.HasPrefix(errs[0], "Error processing audio delta") {
		t.Errorf("reported %v, want the undecodable delta", errs)
	}
}

// The done event stops the context playing and ends it at once, and ends the
// metrics.
func TestDoneEndsTheContext(t *testing.T) {
	s := newSynthesizer(TTSConfig{APIKey: "k"})
	host, svc := newFakeHost(), &fakeSvc{active: ctxID}
	s.host, s.svc = host, svc

	if err := handle(t, s, done()); err != nil {
		t.Fatal(err)
	}
	got := host.framesFor(ctxID)
	if len(got) != 1 {
		t.Fatalf("appended %v, want a stop frame", got)
	}
	if stopped, ok := got[0].(*frames.TTSStoppedFrame); !ok || stopped.ContextID != ctxID {
		t.Errorf("appended %v, want the context stopped", got[0])
	}
	if !slices.Equal(host.removed, []string{ctxID}) {
		t.Errorf("removed %v, want the active context", host.removed)
	}
	if svc.stops() != 1 {
		t.Errorf("metrics stopped %d times, want once", svc.stops())
	}

	// With no context playing there is nothing to end.
	svc.active = ""
	if err := handle(t, s, done()); err != nil {
		t.Fatal(err)
	}
	if len(host.removed) != 1 {
		t.Errorf("removed %v, want nothing more without a context", host.removed)
	}
}

// A failed synthesis is reported with what Together said, stops the context
// playing and ends the metrics, and leaves the connection up.
func TestFailedSynthesisIsReported(t *testing.T) {
	s := newSynthesizer(TTSConfig{APIKey: "k"})
	host, svc := newFakeHost(), &fakeSvc{active: ctxID}
	s.host, s.svc = host, svc

	err := handle(t, s, map[string]any{
		"type":  "conversation.item.tts.failed",
		"error": map[string]any{"message": "voice not found", "code": "bad_voice"},
	})
	if err != nil {
		t.Errorf("a failed synthesis failed the connection: %v", err)
	}
	if errs := host.reported(); len(errs) != 1 || !strings.HasPrefix(errs[0], "TTS error: ") ||
		!strings.Contains(errs[0], "voice not found") {
		t.Errorf("reported %v, want what the provider said", errs)
	}
	if len(host.pushed) != 1 {
		t.Fatalf("pushed %v, want one stop frame", host.pushed)
	}
	if stopped, ok := host.pushed[0].(*frames.TTSStoppedFrame); !ok || stopped.ContextID != ctxID {
		t.Errorf("pushed %v, want the active context stopped", host.pushed[0])
	}
	if svc.stops() != 1 {
		t.Errorf("metrics stopped %d times, want once", svc.stops())
	}
}

// An error event is reported and fails the connection, so it is opened again.
func TestErrorEventFailsTheConnection(t *testing.T) {
	s := newSynthesizer(TTSConfig{APIKey: "k"})
	host := newFakeHost()
	s.host, s.svc = host, &fakeSvc{active: ctxID}

	err := handle(t, s, map[string]any{
		"type":  "error",
		"error": map[string]any{"message": "session expired", "code": "expired"},
	})
	if !errors.Is(err, errTTS) || !strings.Contains(err.Error(), "session expired") {
		t.Errorf("error = %v, want the session's error", err)
	}
	if errs := host.reported(); !slices.Equal(errs, []string{"Together AI TTS error [expired]: session expired"}) {
		t.Errorf("reported %v, want the session's error", errs)
	}
}

// What cannot be read, and the events that only inform, leave the connection
// up; the session id is kept.
func TestInformationalEvents(t *testing.T) {
	s := newSynthesizer(TTSConfig{APIKey: "k"})
	host := newFakeHost()
	s.host, s.svc = host, &fakeSvc{active: ctxID}

	if err := s.handleMessage(context.Background(), []byte("not json")); err != nil {
		t.Errorf("a message that is not JSON failed the connection: %v", err)
	}
	for _, ev := range []map[string]any{
		{"type": "session.created", "session": map[string]any{"id": "sess-1"}},
		{"type": "session.updated", "session": map[string]any{"voice": "af_heart"}},
		{"type": "conversation.item.input_text.received", "text": "hello"},
		{"type": "something.new"},
	} {
		if err := handle(t, s, ev); err != nil {
			t.Errorf("%v failed the connection: %v", ev["type"], err)
		}
	}
	if s.sessionID != "sess-1" {
		t.Errorf("session id = %q, want the one Together gave", s.sessionID)
	}
	if len(host.reported()) != 0 || len(host.framesFor(ctxID)) != 0 {
		t.Errorf("informational events produced %v and %v", host.reported(), host.framesFor(ctxID))
	}
}

// An error event has the receive loop open the connection again.
func TestErrorEventReconnects(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, TTSConfig{Websocket: wsservice.Options{
		ReconnectBackoffMinWait: time.Millisecond,
		ReconnectBackoffMaxWait: time.Millisecond,
	}})
	waitUntil(t, "the server never saw the connection", func() bool { return server.connections() == 1 })

	server.push(t, map[string]any{"type": "error", "error": map[string]any{"message": "boom", "code": "x"}})
	waitUntil(t, "the service never reconnected", func() bool {
		return server.connections() == 2 && s.Connected()
	})
}

// An interruption clears the text Together is still holding, and ends the
// metrics.
func TestInterruptionClearsTheTextBuffer(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, svc := connectedSynth(t, server, TTSConfig{})

	s.OnAudioContextInterrupted(context.Background(), ctxID)
	if got := server.next(t); got["type"] != "input_text_buffer.clear" {
		t.Errorf("sent %v, want the text buffer cleared", got)
	}
	if svc.stops() != 1 {
		t.Errorf("metrics stopped %d times, want once", svc.stops())
	}

	s.OnAudioContextInterrupted(context.Background(), "")
	if !server.quiet() {
		t.Error("an interruption without a context sent something")
	}
}

// Flushing commits the text buffer, and does nothing without a connection.
func TestFlushAudioCommits(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, TTSConfig{})

	s.FlushAudio(context.Background(), ctxID)
	if got := server.next(t); got["type"] != "input_text_buffer.commit" {
		t.Errorf("sent %v, want the commit", got)
	}

	s.Disconnect(context.Background())
	s.FlushAudio(context.Background(), ctxID)
	if !server.quiet() {
		t.Error("a flush without a connection sent something")
	}
}

// Any change of settings reconnects, with the new settings on the URL and the
// session.
func TestSettingsChangeReconnects(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, TTSConfig{})
	waitUntil(t, "the server never saw the connection", func() bool { return server.connections() == 1 })
	ctx := context.Background()

	update := &TTSSettings{MaxPartialLength: settings.Set(80)}
	update.Voice = settings.Set("af_bella")
	changed, err := settings.Apply(s.live, update)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.UpdateSettings(ctx, changed); err != nil {
		t.Fatal(err)
	}
	waitUntil(t, "the change did not reconnect", func() bool { return server.connections() == 2 })
	q := server.query(1)
	if q.Get("voice") != "af_bella" || q.Get("max_partial_length") != "80" {
		t.Errorf("query = %v, want the new settings", q)
	}
	got := server.next(t)
	if session, _ := got["session"].(map[string]any); session["voice"] != "af_bella" {
		t.Errorf("session update = %v, want the new voice", got)
	}

	// Nothing changed, nothing to reconnect for.
	if err := s.UpdateSettings(ctx, settings.Changed{}); err != nil {
		t.Fatal(err)
	}
	if !server.quiet() || server.connections() != 2 {
		t.Error("an empty change reconnected")
	}
}

// A connection the server drops is restored by the receive loop.
func TestReconnectsWhenTheServerDrops(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, TTSConfig{Websocket: wsservice.Options{
		ReconnectBackoffMinWait: time.Millisecond,
		ReconnectBackoffMaxWait: time.Millisecond,
	}})
	waitUntil(t, "the server never saw the connection", func() bool { return server.connections() == 1 })

	server.drop()
	waitUntil(t, "the service never reconnected", func() bool {
		return server.connections() == 2 && s.Connected()
	})
}

// A sentence after the connection closed opens it again before it is sent.
func TestSentenceReopensAClosedConnection(t *testing.T) {
	server := newFakeServer(t, nil)
	s, _, _ := connectedSynth(t, server, TTSConfig{})
	s.Disconnect(context.Background())

	if err := s.RunTTS(context.Background(), "Hello.", ctxID, nil); err != nil {
		t.Fatal(err)
	}
	if got := server.next(t); got["type"] != "tts_session.updated" {
		t.Errorf("sent %v, want the new session's update", got)
	}
	if got := server.next(t); got["text"] != "Hello." {
		t.Errorf("sent %v, want the sentence", got)
	}
	// The server counts a connection once its handshake is done, which can
	// trail the dial returning.
	waitUntil(t, "the closed connection was not opened again", func() bool { return server.connections() == 2 })
}

// A send that fails is reported, and the sentence is not failed for it.
func TestFailedSendIsReported(t *testing.T) {
	server := newFakeServer(t, nil)
	_, s := newTTS(TTSConfig{APIKey: "k", URL: server.url})
	host := newFakeHost()
	s.host, s.svc = host, &fakeSvc{}
	t.Cleanup(func() { s.Disconnect(context.Background()) })

	// A connection that is already closed but not yet known to be: the send is
	// where the failure shows.
	conn, err := s.ws.Dial(context.Background(), server.url, http.Header{}, wsutil.DefaultReadLimit)
	if err != nil {
		t.Fatal(err)
	}
	_ = conn.CloseNow()
	s.conn = &ttsConn{conn: conn, reads: make(chan ttsRead), stop: make(chan struct{})}

	if err := s.RunTTS(context.Background(), "Hello.", "c1", nil); err != nil {
		t.Errorf("RunTTS failed the sentence: %v", err)
	}
	errs := host.reported()
	if len(errs) != 2 || !strings.HasPrefix(errs[0], "Error sending message") {
		t.Errorf("reported %v, want the append and the commit failing", errs)
	}
}

// A dial that fails is reported and leaves no connection.
func TestFailedConnectionIsReported(t *testing.T) {
	server := newFakeServer(t, nil)
	server.refuse = http.StatusUnauthorized
	_, s := newTTS(TTSConfig{APIKey: "k", URL: server.url})
	host := newFakeHost()
	s.host, s.svc = host, &fakeSvc{}
	t.Cleanup(func() { s.Disconnect(context.Background()) })

	s.Start(context.Background())
	if s.Connected() {
		t.Error("connected though the server refused")
	}
	if errs := host.reported(); len(errs) != 1 || !strings.HasPrefix(errs[0], "Error connecting to Together AI TTS") {
		t.Errorf("reported %v, want the failed connection", errs)
	}
}

// TestSampleRate checks the fixed rate Together streams at, which is what the
// emitted frames are stamped with.
func TestSampleRate(t *testing.T) {
	if got := (&synthesizer{}).SampleRate(); got != ttsSampleRate {
		t.Errorf("SampleRate() = %d, want %d", got, ttsSampleRate)
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

// cancel ends the pipeline at once, without waiting for audio still playing.
func (r *run) cancel() {
	r.t.Helper()
	r.task.Cancel(context.Background(), "")
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
	server := newFakeServer(t, speaking)
	b := NewTTS(TTSConfig{APIKey: "k", URL: server.url})
	r := runPipeline(t, b)
	waitUntil(t, "the server saw no connection when the pipeline started", func() bool {
		return server.connections() == 1
	})

	r.task.QueueFrame(frames.NewTTSSpeakFrame("Hello there."))
	r.waitFor(func(f frames.Frame) bool {
		a, ok := f.(*frames.TTSAudioRawFrame)
		return ok && a.ContextID != ""
	})
	r.task.QueueFrame(frames.NewTTSSpeakFrame("And again."))
	waitUntil(t, "the second sentence was never spoken", func() bool {
		r.mu.Lock()
		defer r.mu.Unlock()
		stops := 0
		for _, f := range r.down {
			if _, ok := f.(*frames.TTSStoppedFrame); ok {
				stops++
			}
		}
		return stops >= 2
	})
	r.cancel()

	if n := server.connections(); n != 1 {
		t.Errorf("the server saw %d connections, want one for the whole pipeline", n)
	}
}

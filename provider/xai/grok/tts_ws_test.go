package grok

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"math"
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

// Tests for the streaming connection held open for the call. The audio
// parameters are on the URL, the sentences of a turn are streamed into one
// utterance closed by text.done, and the audio and the character timings come
// back naming no request.

const ctxID = "ctx-1"

// fakeTTSServer is a fake xAI streaming endpoint. It records every connection
// and every message the client sends, and answers each with what reply
// returns.
type fakeTTSServer struct {
	url  string
	sent chan map[string]any

	mu      sync.Mutex
	reply   func(msg map[string]any) []map[string]any
	refuse  int
	conns   []*websocket.Conn
	queries []url.Values
	headers []http.Header
}

func newFakeTTSServer(t *testing.T, reply func(msg map[string]any) []map[string]any) *fakeTTSServer {
	t.Helper()
	s := &fakeTTSServer{sent: make(chan map[string]any, 256), reply: reply}
	srv := httptest.NewServer(http.HandlerFunc(s.serve))
	t.Cleanup(srv.Close)
	s.url = wsURL(srv.URL)
	return s
}

func (s *fakeTTSServer) serve(w http.ResponseWriter, r *http.Request) {
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
			b, _ := json.Marshal(ev)
			if c.Write(ctx, websocket.MessageText, b) != nil {
				return
			}
		}
	}
}

func (s *fakeTTSServer) connections() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.conns)
}

func (s *fakeTTSServer) query(i int) url.Values {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.queries[i]
}

// drop ends every connection without a closing handshake.
func (s *fakeTTSServer) drop() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, c := range s.conns {
		_ = c.CloseNow()
	}
}

func (s *fakeTTSServer) next(t *testing.T) map[string]any {
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
func (s *fakeTTSServer) quiet() bool {
	select {
	case <-s.sent:
		return false
	case <-time.After(50 * time.Millisecond):
		return true
	}
}

// onDone answers each text.done with the events given.
func onDone(events ...map[string]any) func(msg map[string]any) []map[string]any {
	return func(msg map[string]any) []map[string]any {
		if msg["type"] != ttsClientDone {
			return nil
		}
		return events
	}
}

// audioDelta builds an audio.delta event carrying pcm and optional timings.
func audioDelta(pcm []byte, chars []string, times [][]float64) map[string]any {
	ev := map[string]any{"type": ttsEventDelta}
	if len(pcm) > 0 {
		ev["delta"] = base64.StdEncoding.EncodeToString(pcm)
	}
	if chars != nil {
		ev["audio_timestamps"] = map[string]any{"graph_chars": chars, "graph_times": times}
	}
	return ev
}

// charTimes times each character a tenth of a second after the one before.
func charTimes(chars []string) [][]float64 {
	times := make([][]float64, len(chars))
	for i := range chars {
		times[i] = []float64{float64(i) * 0.1, float64(i+1) * 0.1}
	}
	return times
}

func audioDone() map[string]any { return map[string]any{"type": ttsEventDone, "trace_id": "t"} }

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

// fakeHost stands in for the tts.Base the provider appends its audio and its
// word timings to.
type fakeHost struct {
	mu       sync.Mutex
	appended map[string][]frames.Frame
	words    []uctx.WordTiming
	removed  []string
	errors   []string
}

func newFakeHost() *fakeHost { return &fakeHost{appended: map[string][]frames.Frame{}} }

func (h *fakeHost) AppendToAudioContext(id string, f frames.Frame) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.appended[id] = append(h.appended[id], f)
}

func (h *fakeHost) AppendWordToAudioContext(string, string, float64) {}

func (h *fakeHost) AddWordTimestamps(_ string, words []uctx.WordTiming, _ tts.WordTimingOptions) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.words = append(h.words, words...)
}

func (h *fakeHost) RemoveAudioContext(id string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.removed = append(h.removed, id)
}

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

func (h *fakeHost) framesFor(id string) []frames.Frame {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.appended[id])
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
func connectedSynth(t *testing.T, server *fakeTTSServer, cfg TTSConfig) (*ttsSynthesizer, *fakeHost, *fakeSvc) {
	t.Helper()
	cfg.APIKey, cfg.URL = "test-key", server.url
	_, s := newTTS(cfg)
	host, svc := newFakeHost(), &fakeSvc{active: ctxID}
	s.host, s.svc = host, svc
	s.Start(context.Background())
	if !s.Connected() {
		t.Fatal("the service did not connect when it started")
	}
	// The server counts a connection once its handshake is done, which can
	// trail the dial returning.
	waitUntil(t, "the server never saw the connection", func() bool { return server.connections() == 1 })
	t.Cleanup(func() { s.Disconnect(context.Background()) })
	return s, host, svc
}

// detached builds a synthesizer with fakes for its host and service, to be
// handed messages directly.
func detached() (*ttsSynthesizer, *fakeHost, *fakeSvc) {
	s := newTTSSynthesizer(TTSConfig{APIKey: "k"})
	host, svc := newFakeHost(), &fakeSvc{active: ctxID}
	s.host, s.svc = host, svc
	return s, host, svc
}

// handle hands an event to the synthesizer as the receive loop does.
func handle(t *testing.T, s *ttsSynthesizer, ev map[string]any) error {
	t.Helper()
	data, err := json.Marshal(ev)
	if err != nil {
		t.Fatal(err)
	}
	return s.handleMessage(context.Background(), data)
}

// assertWords checks the timings reported, allowing for float rounding.
func assertWords(t *testing.T, got []uctx.WordTiming, want ...uctx.WordTiming) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("words = %v, want %v", got, want)
	}
	for i := range want {
		if got[i].Word != want[i].Word || math.Abs(got[i].Offset-want[i].Offset) > 1e-9 {
			t.Errorf("word %d = %v, want %v", i, got[i], want[i])
		}
	}
}

// Character timings become words timed from their first character, a word
// left unterminated at the end of one payload is finished by the next, and the
// last word of the utterance is flushed when its audio is done. xAI times are
// absolute across the utterance, so they are used as they are.
func TestWordTimesSplitAndCarryPartials(t *testing.T) {
	s, host, _ := detached()
	chars := []string{"H", "i", " ", "t", "h", "e", "r", "e"}

	if err := handle(t, s, audioDelta(nil, chars, charTimes(chars))); err != nil {
		t.Fatal(err)
	}
	assertWords(t, host.words, uctx.WordTiming{Word: "Hi", Offset: 0})

	chars2 := []string{"!", " ", "o", "k"}
	times2 := [][]float64{{2.0, 2.05}, {2.05, 2.1}, {2.1, 2.15}, {2.15, 2.2}}
	if err := handle(t, s, audioDelta(nil, chars2, times2)); err != nil {
		t.Fatal(err)
	}
	assertWords(t, host.words, uctx.WordTiming{Word: "Hi", Offset: 0}, uctx.WordTiming{Word: "there!", Offset: 0.3})

	if err := handle(t, s, audioDone()); err != nil {
		t.Fatal(err)
	}
	assertWords(t, host.words,
		uctx.WordTiming{Word: "Hi", Offset: 0},
		uctx.WordTiming{Word: "there!", Offset: 0.3},
		uctx.WordTiming{Word: "ok", Offset: 2.1})
}

// A payload whose characters and times do not pair up yields no words, does
// not fail the connection, and keeps the word being assembled.
func TestWordTimesLengthMismatchIsSafe(t *testing.T) {
	s, host, _ := detached()

	if err := handle(t, s, audioDelta(nil, []string{"x"}, [][]float64{{1.0, 1.1}})); err != nil {
		t.Fatal(err)
	}
	if err := handle(t, s, audioDelta(nil, []string{"a", "b"}, [][]float64{{0.0, 0.1}})); err != nil {
		t.Errorf("a length mismatch failed the connection: %v", err)
	}
	if len(host.words) != 0 {
		t.Fatalf("words = %v, want none from the mismatch", host.words)
	}
	if err := handle(t, s, audioDone()); err != nil {
		t.Fatal(err)
	}
	assertWords(t, host.words, uctx.WordTiming{Word: "x", Offset: 1.0})
}

// A word carried over from another context's utterance is dropped.
func TestWordTimesResetOnANewContext(t *testing.T) {
	s, host, svc := detached()

	if err := handle(t, s, audioDelta(nil, []string{"x"}, [][]float64{{1.0, 1.1}})); err != nil {
		t.Fatal(err)
	}
	svc.active = "ctx-2"
	if err := handle(t, s, audioDelta(nil, []string{"y", " "}, [][]float64{{0.0, 0.1}, {0.1, 0.2}})); err != nil {
		t.Fatal(err)
	}
	assertWords(t, host.words, uctx.WordTiming{Word: "y", Offset: 0})
}

// Audio is appended to the context playing, the done event stops it and ends
// it at once, and both end the metrics. With no context playing nothing is
// appended.
func TestAudioAndDone(t *testing.T) {
	s, host, svc := detached()

	if err := handle(t, s, audioDelta([]byte{1, 2, 3, 4}, nil, nil)); err != nil {
		t.Fatal(err)
	}
	if err := handle(t, s, audioDone()); err != nil {
		t.Fatal(err)
	}
	got := host.framesFor(ctxID)
	if len(got) != 2 {
		t.Fatalf("appended %v, want the audio then a stop frame", got)
	}
	if a, ok := got[0].(*frames.TTSAudioRawFrame); !ok || !bytes.Equal(a.Audio, []byte{1, 2, 3, 4}) ||
		a.ContextID != ctxID || a.SampleRate != defaultTTSSampleRate {
		t.Errorf("appended %v, want the audio on the active context", got[0])
	}
	if stopped, ok := got[1].(*frames.TTSStoppedFrame); !ok || stopped.ContextID != ctxID {
		t.Errorf("appended %v, want the context stopped", got[1])
	}
	if !slices.Equal(host.removed, []string{ctxID}) {
		t.Errorf("removed %v, want the active context", host.removed)
	}
	if svc.stops() != 1 {
		t.Errorf("metrics stopped %d times, want once", svc.stops())
	}

	svc.active = ""
	if err := handle(t, s, audioDelta([]byte{5, 6}, nil, nil)); err != nil {
		t.Fatal(err)
	}
	if err := handle(t, s, audioDone()); err != nil {
		t.Fatal(err)
	}
	if len(host.appended) != 1 || len(host.removed) != 1 {
		t.Errorf("without a context playing, appended %v and removed %v", host.appended, host.removed)
	}
}

// An error ends the context playing and is reported with what xAI said, while
// the connection stays up.
func TestServerErrorClosesTheContext(t *testing.T) {
	s, host, svc := detached()

	if err := handle(t, s, map[string]any{"type": ttsEventError, "message": "unknown voice"}); err != nil {
		t.Errorf("an error message failed the connection: %v", err)
	}
	got := host.framesFor(ctxID)
	if len(got) != 1 {
		t.Fatalf("appended %v, want a stop frame", got)
	}
	if _, ok := got[0].(*frames.TTSStoppedFrame); !ok {
		t.Errorf("appended %v, want the context stopped", got[0])
	}
	if !slices.Equal(host.removed, []string{ctxID}) {
		t.Errorf("removed %v, want the active context", host.removed)
	}
	if !slices.Equal(host.errors, []string{"xAI TTS error: unknown voice"}) {
		t.Errorf("reported %v, want the server's message", host.errors)
	}
	if svc.stops() != 1 {
		t.Errorf("metrics stopped %d times, want once", svc.stops())
	}

	if err := handle(t, s, map[string]any{"type": ttsEventError, "error": "rate limited"}); err != nil {
		t.Fatal(err)
	}
	if host.errors[1] != "xAI TTS error: rate limited" {
		t.Errorf("reported %q, want the error field when there is no message", host.errors[1])
	}
}

// What cannot be read, the acknowledgment of a clear and messages of other
// types leave the connection up; audio that cannot be decoded fails it.
func TestUnreadableMessages(t *testing.T) {
	s, host, _ := detached()

	if err := s.handleMessage(context.Background(), []byte("not json")); err != nil {
		t.Errorf("a message that is not JSON failed the connection: %v", err)
	}
	for _, ev := range []map[string]any{{"type": ttsEventClear}, {"type": "something.new"}} {
		if err := handle(t, s, ev); err != nil {
			t.Errorf("%v failed the connection: %v", ev["type"], err)
		}
	}
	if len(host.errors) != 0 {
		t.Errorf("reported %v, want nothing", host.errors)
	}
	if err := handle(t, s, map[string]any{"type": ttsEventDelta, "delta": "not base64!!"}); err == nil {
		t.Error("audio that is not base64 was accepted")
	}
}

// A sentence is streamed into the utterance on the open connection, and the
// flush closes the utterance.
func TestSentenceAndFlush(t *testing.T) {
	server := newFakeTTSServer(t, nil)
	s, _, _ := connectedSynth(t, server, TTSConfig{})
	ctx := context.Background()

	if err := s.RunTTS(ctx, "Hello there. ", ctxID, nil); err != nil {
		t.Fatal(err)
	}
	if got := server.next(t); got["type"] != ttsClientDelta || got["delta"] != "Hello there. " {
		t.Errorf("sent %v, want the text delta", got)
	}
	if !server.quiet() {
		t.Error("the sentence closed the utterance; the flush does that")
	}
	s.FlushAudio(ctx, ctxID)
	if got := server.next(t); got["type"] != ttsClientDone {
		t.Errorf("sent %v, want the utterance closed", got)
	}

	s.Disconnect(ctx)
	s.FlushAudio(ctx, ctxID)
	if !server.quiet() {
		t.Error("a flush without a connection sent something")
	}
}

// On barge-in the utterance is cleared and the connection kept.
func TestInterruptionSendsTextClear(t *testing.T) {
	server := newFakeTTSServer(t, nil)
	s, host, svc := connectedSynth(t, server, TTSConfig{})

	if err := handle(t, s, audioDelta(nil, []string{"x"}, [][]float64{{1.0, 1.1}})); err != nil {
		t.Fatal(err)
	}
	s.OnAudioContextInterrupted(context.Background(), ctxID)
	if got := server.next(t); got["type"] != ttsClientClear {
		t.Errorf("sent %v, want the utterance cleared", got)
	}
	if svc.stops() != 1 {
		t.Errorf("metrics stopped %d times, want once", svc.stops())
	}
	if n := server.connections(); n != 1 || !s.Connected() {
		t.Errorf("the interruption reconnected: %d connections", n)
	}
	// The word being assembled went with the utterance.
	if err := handle(t, s, audioDone()); err != nil {
		t.Fatal(err)
	}
	if len(host.words) != 0 {
		t.Errorf("words = %v, want the interrupted one dropped", host.words)
	}
}

// The settings are on the URL, booleans spelled the way xAI expects.
func TestWebsocketSettingsInURL(t *testing.T) {
	server := newFakeTTSServer(t, nil)
	speed, level, normalize, timestamps := 1.2, 2, true, false
	connectedSynth(t, server, TTSConfig{
		Speed:                    &speed,
		OptimizeStreamingLatency: &level,
		TextNormalization:        &normalize,
		WordTimestamps:           &timestamps,
	})

	q := server.query(0)
	for key, want := range map[string]string{
		"speed":                      "1.2",
		"optimize_streaming_latency": "2",
		"text_normalization":         "true",
		"with_timestamps":            "false",
	} {
		if got := q.Get(key); got != want {
			t.Errorf("%s = %q, want %q", key, got, want)
		}
	}
	server.mu.Lock()
	header := server.headers[0]
	server.mu.Unlock()
	if auth := header.Get("Authorization"); auth != "Bearer test-key" {
		t.Errorf("Authorization = %q, want the Bearer key", auth)
	}
}

// Any change of settings reconnects, with the new settings on the URL.
func TestSettingsChangeReconnects(t *testing.T) {
	server := newFakeTTSServer(t, nil)
	s, _, _ := connectedSynth(t, server, TTSConfig{})
	ctx := context.Background()

	update := &WebsocketTTSSettings{Speed: settings.Set(1.3)}
	update.Voice = settings.Set("atlas")
	update.Language = settings.Set("fr")
	changed, err := settings.Apply(s.live, update)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.UpdateSettings(ctx, changed); err != nil {
		t.Fatal(err)
	}
	waitUntil(t, "the change did not reconnect", func() bool { return server.connections() == 2 })
	q := server.query(1)
	if q.Get("voice") != "atlas" || q.Get("language") != "fr" || q.Get("speed") != "1.3" {
		t.Errorf("query = %v, want the new settings", q)
	}
	if !s.Connected() {
		t.Error("not connected after the change")
	}

	if err := s.UpdateSettings(ctx, settings.Changed{}); err != nil {
		t.Fatal(err)
	}
	if server.connections() != 2 {
		t.Error("an empty change reconnected")
	}
}

// A connection the server drops is restored by the receive loop.
func TestReconnectsWhenTheServerDrops(t *testing.T) {
	server := newFakeTTSServer(t, nil)
	s, _, _ := connectedSynth(t, server, TTSConfig{Websocket: wsservice.Options{
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
	server := newFakeTTSServer(t, nil)
	s, _, _ := connectedSynth(t, server, TTSConfig{})
	s.Disconnect(context.Background())

	if err := s.RunTTS(context.Background(), "Hello. ", ctxID, nil); err != nil {
		t.Fatal(err)
	}
	if got := server.next(t); got["delta"] != "Hello. " {
		t.Errorf("sent %v, want the sentence", got)
	}
	waitUntil(t, "the closed connection was not opened again", func() bool { return server.connections() == 2 })
}

// A send that fails stops the context's speech, is reported, and resets the
// connection.
func TestFailedSendResetsTheConnection(t *testing.T) {
	server := newFakeTTSServer(t, nil)
	_, s := newTTS(TTSConfig{APIKey: "k", URL: server.url})
	s.host, s.svc = newFakeHost(), &fakeSvc{}
	t.Cleanup(func() { s.Disconnect(context.Background()) })

	// A connection that is already closed but not yet known to be: the send is
	// where the failure shows.
	conn, err := s.ws.Dial(context.Background(), server.url, http.Header{}, wsutil.DefaultReadLimit)
	if err != nil {
		t.Fatal(err)
	}
	_ = conn.CloseNow()
	s.conn = &ttsConn{conn: conn, reads: make(chan ttsRead), stop: make(chan struct{})}

	var got []frames.Frame
	err = s.RunTTS(context.Background(), "Hello. ", "c1", func(f frames.Frame) error {
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
	waitUntil(t, "the connection was not reset", func() bool { return server.connections() == 2 })
	if !s.Connected() {
		t.Error("not connected after the reset")
	}
}

// run is a running pipeline around the service.
type run struct {
	t    *testing.T
	task *pipeline.Worker
	done chan error

	mu   sync.Mutex
	down []frames.Frame
	up   []frames.Frame
}

func runPipeline(t *testing.T, b *tts.Base) *run {
	t.Helper()
	r := &run{t: t, done: make(chan error, 1)}
	r.task = pipeline.NewWorker(pipeline.New(b), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
	})
	events.On(&r.task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		r.mu.Lock()
		r.down = append(r.down, f)
		r.mu.Unlock()
	})
	events.On(&r.task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		r.mu.Lock()
		r.up = append(r.up, f)
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

func (r *run) count(match func(frames.Frame) bool) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	n := 0
	for _, f := range r.down {
		if match(f) {
			n++
		}
	}
	return n
}

func (r *run) frames() (down, up []frames.Frame) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.down), slices.Clone(r.up)
}

func (r *run) stop() {
	r.t.Helper()
	r.task.StopWhenDone()
	select {
	case <-r.done:
	case <-time.After(5 * time.Second):
		r.t.Fatal("the pipeline did not finish")
	}
}

func isStopped(f frames.Frame) bool { _, ok := f.(*frames.TTSStoppedFrame); return ok }

// speak runs a pipeline around a service for server, speaks text and waits for
// its speech to stop, then ends the pipeline.
func speak(t *testing.T, server *fakeTTSServer, cfg TTSConfig, text string) *run {
	t.Helper()
	cfg.APIKey, cfg.URL = "test-key", server.url
	r := runPipeline(t, NewTTS(cfg))
	r.task.QueueFrame(frames.NewTTSSpeakFrame(text))
	r.waitFor(isStopped)
	r.stop()
	return r
}

// The streaming service sends the text then closes the utterance, and emits
// the audio between its started and stopped frames. The text is sent ending in
// a space, so the sentences of a turn do not run together.
func TestWebsocketTTSSpeaks(t *testing.T) {
	pcm := bytes.Repeat([]byte{0, 1, 2, 3}, 1024)
	server := newFakeTTSServer(t, onDone(audioDelta(pcm, nil, nil), audioDone()))
	off := false
	r := speak(t, server, TTSConfig{WordTimestamps: &off}, "Hello from xAI.")

	down, _ := r.frames()
	var audio []byte
	var started bool
	for _, f := range down {
		switch f := f.(type) {
		case *frames.TTSStartedFrame:
			started = true
		case *frames.TTSAudioRawFrame:
			if f.SampleRate != defaultTTSSampleRate || f.NumChannels != 1 {
				t.Errorf("audio frame at %d Hz, %d channels, want 24 kHz mono", f.SampleRate, f.NumChannels)
			}
			audio = append(audio, f.Audio...)
		}
	}
	if !started {
		t.Error("no TTSStartedFrame")
	}
	if !bytes.Equal(audio, pcm) {
		t.Errorf("emitted %d bytes of audio, want the %d sent", len(audio), len(pcm))
	}

	server.mu.Lock()
	header := server.headers[0]
	server.mu.Unlock()
	if auth := header.Get("Authorization"); auth != "Bearer test-key" {
		t.Errorf("Authorization = %q, want the Bearer key", auth)
	}
	q := server.query(0)
	for key, want := range map[string]string{
		"voice": defaultVoice, "language": ttsLanguageAuto, "codec": ttsCodec, "sample_rate": "24000",
	} {
		if got := q.Get(key); got != want {
			t.Errorf("%s = %q, want %q", key, got, want)
		}
	}
	if got := server.next(t); got["type"] != ttsClientDelta || got["delta"] != "Hello from xAI. " {
		t.Errorf("first message = %v, want the text with a trailing space", got)
	}
	if got := server.next(t); got["type"] != ttsClientDone {
		t.Errorf("second message = %v, want the utterance closed", got)
	}
	if n := server.connections(); n != 1 {
		t.Errorf("the server saw %d connections, want one", n)
	}
}

// An acknowledgment of a clear is handled without an error.
func TestAudioClearIsNotAnError(t *testing.T) {
	pcm := bytes.Repeat([]byte{0, 1, 2, 3}, 1024)
	server := newFakeTTSServer(t, onDone(map[string]any{"type": ttsEventClear}, audioDelta(pcm, nil, nil), audioDone()))
	r := speak(t, server, TTSConfig{}, "Hello from xAI.")

	down, up := r.frames()
	for _, f := range append(down, up...) {
		if _, ok := f.(*frames.ErrorFrame); ok {
			t.Errorf("got %v, want no error", f)
		}
	}
}

// spokenText joins the text of the word frames pushed.
func spokenText(r *run) string {
	down, _ := r.frames()
	var words []string
	for _, f := range down {
		if tf, ok := f.(*frames.TTSTextFrame); ok {
			words = append(words, tf.Text)
		}
	}
	return strings.Join(words, " ")
}

// With timings asked for, each word is pushed as a text frame.
func TestEmitsWordTimestamps(t *testing.T) {
	pcm := bytes.Repeat([]byte{0, 1, 2, 3}, 1024)
	chars := strings.Split("Hello world", "")
	server := newFakeTTSServer(t, onDone(audioDelta(pcm, chars, charTimes(chars)), audioDone()))
	r := speak(t, server, TTSConfig{}, "Hello world")

	if got := spokenText(r); !strings.Contains(got, "Hello") || !strings.Contains(got, "world") {
		t.Errorf("spoken text = %q, want both words", got)
	}
}

// Timings that arrive in a delta of their own, carrying no audio, still push
// their words.
func TestWordTimestampsFromAudiolessDelta(t *testing.T) {
	pcm := bytes.Repeat([]byte{0, 1, 2, 3}, 1024)
	chars := strings.Split("Hi there", "")
	server := newFakeTTSServer(t, onDone(
		audioDelta(pcm, nil, nil),
		map[string]any{"type": ttsEventDelta, "delta": "", "audio_timestamps": map[string]any{
			"graph_chars": chars, "graph_times": charTimes(chars),
		}},
		audioDone(),
	))
	r := speak(t, server, TTSConfig{}, "Hi there")

	if got := spokenText(r); !strings.Contains(got, "Hi") || !strings.Contains(got, "there") {
		t.Errorf("spoken text = %q, want both words", got)
	}
}

// Two turns are spoken over the one connection opened when the pipeline
// started.
func TestHoldsOneConnectionForThePipeline(t *testing.T) {
	server := newFakeTTSServer(t, onDone(audioDelta(make([]byte, 3200), nil, nil), audioDone()))
	r := runPipeline(t, NewTTS(TTSConfig{APIKey: "k", URL: server.url}))
	waitUntil(t, "the server saw no connection when the pipeline started", func() bool {
		return server.connections() == 1
	})

	r.task.QueueFrame(frames.NewTTSSpeakFrame("Hello there."))
	r.waitFor(isStopped)
	r.task.QueueFrame(frames.NewTTSSpeakFrame("And again."))
	waitUntil(t, "the second turn was never spoken", func() bool { return r.count(isStopped) >= 2 })
	r.stop()

	if n := server.connections(); n != 1 {
		t.Errorf("the server saw %d connections, want one for the whole pipeline", n)
	}
}

// A connection the server refuses is reported as a connection error.
func TestReportsAFailedConnection(t *testing.T) {
	server := newFakeTTSServer(t, nil)
	server.refuse = http.StatusUnauthorized
	b := NewTTS(TTSConfig{APIKey: "k", URL: server.url})
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

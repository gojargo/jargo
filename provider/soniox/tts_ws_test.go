package soniox

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
	"github.com/gojargo/jargo/service/wsutil"
	uctx "github.com/gojargo/jargo/utils/context"
	errs "github.com/gojargo/jargo/utils/errors"
	"github.com/gojargo/jargo/utils/events"
)

// Tests for the connection held open for the call. Each turn is a stream of
// its own on it, named by the turn's audio context: a configuration opens it,
// the text goes to it a sentence at a time, an empty text with text_end closes
// it, and Soniox answers with base64 audio and timings tagged with the stream,
// ending it with terminated.

// ttsFakeServer is a fake Soniox TTS endpoint. It records every connection and
// every message the client sends, and answers each with what reply returns.
type ttsFakeServer struct {
	url  string
	sent chan map[string]any

	mu      sync.Mutex
	reply   func(msg map[string]any) []any
	refuse  int
	conns   []*websocket.Conn
	headers []http.Header
}

func newTTSFakeServer(t *testing.T, reply func(msg map[string]any) []any) *ttsFakeServer {
	t.Helper()
	s := &ttsFakeServer{sent: make(chan map[string]any, 256), reply: reply}
	srv := httptest.NewServer(http.HandlerFunc(s.serve))
	t.Cleanup(srv.Close)
	s.url = "ws" + strings.TrimPrefix(srv.URL, "http")
	return s
}

func (s *ttsFakeServer) serve(w http.ResponseWriter, r *http.Request) {
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

func (s *ttsFakeServer) connections() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.conns)
}

func (s *ttsFakeServer) header(i int) http.Header {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.headers[i]
}

// drop ends every connection without a closing handshake.
func (s *ttsFakeServer) drop() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, c := range s.conns {
		_ = c.CloseNow()
	}
}

func (s *ttsFakeServer) next(t *testing.T) map[string]any {
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
func (s *ttsFakeServer) quiet() bool {
	select {
	case <-s.sent:
		return false
	case <-time.After(50 * time.Millisecond):
		return true
	}
}

// audioEvent is a chunk of audio for a stream, with timings when chars is set.
func audioEvent(stream string, pcm []byte, chars []string, starts []float64) map[string]any {
	ev := map[string]any{keyStreamID: stream}
	if len(pcm) > 0 {
		ev["audio"] = base64.StdEncoding.EncodeToString(pcm)
	}
	if chars != nil {
		ev["timestamps"] = map[string]any{
			"characters":                    chars,
			"character_start_times_seconds": starts,
		}
	}
	return ev
}

func terminated(stream string) map[string]any {
	return map[string]any{keyStreamID: stream, "terminated": true}
}

func streamNotFound(stream string) map[string]any {
	return map[string]any{
		keyStreamID:     stream,
		"error_code":    400,
		"error_type":    "invalid_stream_state",
		"error_message": "Stream " + stream + " not found. Send a start message first.",
	}
}

// ttsSvc stands in for the TTS service the synthesizer asks about its audio
// contexts.
type ttsSvc struct {
	mu      sync.Mutex
	active  string
	stopped int
}

func (f *ttsSvc) ActiveAudioContextID() string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.active
}

func (f *ttsSvc) StopAllMetrics(context.Context) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.stopped++
}

func (f *ttsSvc) metricsStopped() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.stopped
}

// ttsHost stands in for the tts.Base the provider appends its audio to. Only
// the contexts opened with open are available.
type ttsHost struct {
	mu       sync.Mutex
	open     map[string]bool
	audio    map[string][]byte
	appended map[string][]frames.Frame
	words    map[string][]uctx.WordTiming
	opts     map[string][]tts.WordTimingOptions
	removed  []string
	rotated  int
	errors   []string
}

func newTTSHost() *ttsHost {
	return &ttsHost{
		open:     map[string]bool{},
		audio:    map[string][]byte{},
		appended: map[string][]frames.Frame{},
		words:    map[string][]uctx.WordTiming{},
		opts:     map[string][]tts.WordTimingOptions{},
	}
}

func (h *ttsHost) openContext(ids ...string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, id := range ids {
		h.open[id] = true
	}
}

func (h *ttsHost) AppendToAudioContext(id string, f frames.Frame) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if a, ok := f.(*frames.TTSAudioRawFrame); ok {
		if a.ContextID != id {
			h.errors = append(h.errors, "frame labeled "+a.ContextID+" appended to "+id)
		}
		h.audio[id] = append(h.audio[id], a.Audio...)
	}
	h.appended[id] = append(h.appended[id], f)
}

func (h *ttsHost) AppendWordToAudioContext(string, string, float64) {}

func (h *ttsHost) AddWordTimestamps(id string, words []uctx.WordTiming, opts tts.WordTimingOptions) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.words[id] = append(h.words[id], words...)
	h.opts[id] = append(h.opts[id], opts)
}

func (h *ttsHost) RemoveAudioContext(id string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.removed = append(h.removed, id)
	delete(h.open, id)
}

func (h *ttsHost) AudioContextAvailable(id string) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.open[id]
}

func (h *ttsHost) RotateTurnContext(context.Context) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.rotated++
}

func (h *ttsHost) ReplaceConnection() {}

func (h *ttsHost) PushFrame(context.Context, frames.Frame, processor.Direction) error { return nil }

func (h *ttsHost) PushError(_ context.Context, msg string, _ error, _ bool, _ ...processor.ErrorOption) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.errors = append(h.errors, msg)
}

func (h *ttsHost) ResetActiveAudioContext() {}

func (h *ttsHost) audioFor(id string) []byte {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.audio[id])
}

func (h *ttsHost) appendedTo(id string) []frames.Frame {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.appended[id])
}

func (h *ttsHost) wordsFor(id string) []uctx.WordTiming {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.words[id])
}

func (h *ttsHost) wasRemoved(id string) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Contains(h.removed, id)
}

func (h *ttsHost) rotations() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.rotated
}

func (h *ttsHost) reported() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.errors)
}

func waitFor(t *testing.T, msg string, cond func() bool) {
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

// connectedTTS builds a service for server, with the synthesizer's host and
// service replaced by fakes, and starts it.
func connectedTTS(t *testing.T, server *ttsFakeServer, cfg TTSConfig) (*ttsSynthesizer, *ttsHost, *ttsSvc) {
	t.Helper()
	cfg.APIKey, cfg.URL = "test-key", server.url
	_, s := newTTS(cfg)
	host, svc := newTTSHost(), &ttsSvc{}
	s.host, s.svc = host, svc
	s.Start(context.Background())
	if !s.Connected() {
		t.Fatal("the service did not connect when it started")
	}
	// The server counts a connection once its handshake is done, which can
	// trail the dial returning.
	waitFor(t, "the server never saw the connection", func() bool { return server.connections() == 1 })
	t.Cleanup(func() { s.Disconnect(context.Background()) })
	return s, host, svc
}

func handle(t *testing.T, s *ttsSynthesizer, msgs ...map[string]any) {
	t.Helper()
	for _, m := range msgs {
		b, err := json.Marshal(m)
		if err != nil {
			t.Fatal(err)
		}
		if err := s.handleMessage(context.Background(), b); err != nil {
			t.Fatalf("handleMessage(%s): %v", b, err)
		}
	}
}

// The connection opens when the service starts, authorized by a bearer token
// in the handshake, and nothing is sent until a turn starts.
func TestTTSStartOpensAConnection(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	connectedTTS(t, server, TTSConfig{})

	if got := server.header(0).Get("Authorization"); got != "Bearer test-key" {
		t.Errorf("Authorization = %q, want the key as a bearer token", got)
	}
	if !server.quiet() {
		t.Error("the service sent something on connecting")
	}
}

// A turn's stream is opened as soon as the turn starts, once.
func TestTTSTurnStreamOpensWhenTheTurnStarts(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	s, _, _ := connectedTTS(t, server, TTSConfig{})
	ctx := context.Background()

	s.OnTurnContextCreated(ctx, "turn-1")
	if msg := server.next(t); msg[keyStreamID] != "turn-1" || msg["model"] != defaultTTSModel {
		t.Errorf("sent %v, want the stream's configuration", msg)
	}
	if err := s.RunTTS(ctx, "Hello.", "turn-1", nil); err != nil {
		t.Fatal(err)
	}
	msg := server.next(t)
	want := map[string]any{"text": "Hello.", "text_end": false, keyStreamID: "turn-1"}
	if !maps.Equal(msg, want) {
		t.Errorf("sent %v, want the sentence alone, the stream being open", msg)
	}
}

// A stream first seen on this connection (after a reconnect, say) is opened
// before its text, or Soniox would reject it, and opened only once.
func TestTTSRunTTSConfiguresAStreamItHasNotOpened(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	s, _, _ := connectedTTS(t, server, TTSConfig{})
	ctx := context.Background()

	for _, text := range []string{"Hello.", "Again."} {
		if err := s.RunTTS(ctx, text, "stream-1", nil); err != nil {
			t.Fatal(err)
		}
	}
	sent := []map[string]any{server.next(t), server.next(t), server.next(t)}
	for i, m := range sent {
		if m[keyStreamID] != "stream-1" {
			t.Errorf("message %d is for stream %v", i, m[keyStreamID])
		}
	}
	if _, ok := sent[0]["text"]; ok {
		t.Errorf("first message = %v, want the configuration", sent[0])
	}
	if sent[1]["text"] != "Hello." || sent[2]["text"] != "Again." {
		t.Errorf("texts = %v, %v", sent[1]["text"], sent[2]["text"])
	}
	if !server.quiet() {
		t.Error("the stream was configured twice")
	}
}

// Audio and timings go to the stream's context and nowhere else; terminated
// reports the word still being assembled, stops the context's speech and marks
// it done.
func TestTTSStreamAudioAndTimings(t *testing.T) {
	s := newTTSSynthesizer(TTSConfig{APIKey: "k"})
	host, svc := newTTSHost(), &ttsSvc{}
	s.host, s.svc = host, svc
	host.openContext("s1", "s2")

	handle(t, s,
		audioEvent("s1", []byte{1, 2}, []string{"h", "i", " ", "t", "h", "e"}, []float64{0, 0.1, 0.2, 0.3, 0.4, 0.5}),
		audioEvent("s2", []byte{9, 9}, nil, nil),
		audioEvent("gone", []byte{7, 7}, nil, nil),
		audioEvent("s1", nil, []string{"r", "e", ",", " ", "p", "a", "l"}, []float64{0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2}),
		map[string]any{keyStreamID: "s1", "audio_end": true},
		terminated("s1"),
	)

	if got := host.audioFor("s1"); !bytes.Equal(got, []byte{1, 2}) {
		t.Errorf("s1 audio = %v", got)
	}
	if got := host.audioFor("s2"); !bytes.Equal(got, []byte{9, 9}) {
		t.Errorf("s2 audio = %v", got)
	}
	if got := host.audioFor("gone"); len(got) != 0 {
		t.Errorf("audio for a closed context was appended: %v", got)
	}
	want := []uctx.WordTiming{{Word: "hi", Offset: 0}, {Word: "there,", Offset: 0.3}, {Word: "pal", Offset: 1.0}}
	if got := host.wordsFor("s1"); !slices.Equal(got, want) {
		t.Errorf("s1 words = %v, want %v", got, want)
	}
	appended := host.appendedTo("s1")
	if stopped, ok := appended[len(appended)-1].(*frames.TTSStoppedFrame); !ok || stopped.ContextID != "s1" {
		t.Errorf("last frame appended = %v, want the stream's stop frame", appended[len(appended)-1])
	}
	if !host.wasRemoved("s1") || host.wasRemoved("s2") {
		t.Errorf("removed %v, want s1 alone", host.removed)
	}
	if errs := host.reported(); len(errs) != 0 {
		t.Errorf("errors reported: %v", errs)
	}
}

// A language written without spaces reports its timings per character, and
// says the tokens carry their own spacing.
func TestTTSSpacelessTimingsCarryTheirOwnSpacing(t *testing.T) {
	for _, tc := range []struct {
		lang string
		want bool
	}{{"zh", true}, {"en", false}} {
		s := newTTSSynthesizer(TTSConfig{APIKey: "k"})
		s.live.Language = settings.Set(tc.lang)
		s.refreshSpaceless()
		host := newTTSHost()
		s.host, s.svc = host, &ttsSvc{}
		host.openContext("s1")
		handle(t, s, audioEvent("s1", nil, []string{"你", "好", " "}, []float64{0, 0.2, 0.3}))
		opts := host.opts["s1"]
		if len(opts) == 0 {
			t.Fatalf("%s: no timings reported", tc.lang)
		}
		if opts[0].IncludesInterFrameSpaces != tc.want {
			t.Errorf("%s: IncludesInterFrameSpaces = %v, want %v", tc.lang, opts[0].IncludesInterFrameSpaces, tc.want)
		}
	}
}

// Messages that are not JSON are stepped over, and audio that is not base64
// fails the connection rather than being played as rubbish.
func TestTTSUnreadableMessages(t *testing.T) {
	s := newTTSSynthesizer(TTSConfig{APIKey: "k"})
	host := newTTSHost()
	s.host, s.svc = host, &ttsSvc{}
	host.openContext("s1")

	if err := s.handleMessage(context.Background(), []byte("not json")); err != nil {
		t.Errorf("a message that is not JSON failed the connection: %v", err)
	}
	if err := s.handleMessage(context.Background(), []byte(`{"stream_id":"s1","audio":"not base64!!"}`)); err == nil {
		t.Error("audio that is not base64 was accepted")
	}
	if len(host.reported()) != 0 {
		t.Errorf("reported %v, want nothing", host.reported())
	}
}

// A stream's error is reported and ends that stream alone: its speech stops,
// it is forgotten, and the connection and the other streams carry on.
func TestTTSErrorEndsTheStream(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	s, host, _ := connectedTTS(t, server, TTSConfig{})
	host.openContext("s1", "s2")
	ctx := context.Background()
	for _, id := range []string{"s1", "s2"} {
		if err := s.RunTTS(ctx, "Hello.", id, nil); err != nil {
			t.Fatal(err)
		}
	}
	handle(t, s, map[string]any{
		keyStreamID: "s1", "error_code": 400, "error_type": "invalid_request", "error_message": "unknown voice",
	})

	const want = "Soniox TTS error 400 invalid_request (stream s1): unknown voice"
	if errs := host.reported(); len(errs) != 1 || errs[0] != want {
		t.Errorf("reported %v", errs)
	}
	appended := host.appendedTo("s1")
	if len(appended) == 0 {
		t.Fatal("nothing was appended to the failed stream")
	}
	if _, ok := appended[len(appended)-1].(*frames.TTSStoppedFrame); !ok || !host.wasRemoved("s1") {
		t.Error("the failed stream's speech was not stopped")
	}
	if host.wasRemoved("s2") {
		t.Error("the other stream was ended too")
	}
	s.stateMu.Lock()
	configured := s.configured["s1"]
	s.stateMu.Unlock()
	if configured {
		t.Error("the failed stream is still counted as open")
	}
	if !s.Connected() || server.connections() != 1 {
		t.Error("a stream's error ended the connection")
	}
}

// An interruption stops the metrics and cancels the context's stream, if it
// was opened, forgetting the word it was assembling.
func TestTTSInterruptionCancelsTheStream(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	s, _, svc := connectedTTS(t, server, TTSConfig{})
	ctx := context.Background()

	s.OnAudioContextInterrupted(ctx, "never-opened")
	if !server.quiet() {
		t.Error("a cancel went out for a stream Soniox never saw")
	}

	s.OnTurnContextCreated(ctx, "s1")
	server.next(t)
	s.toWordTimes("s1", timings("Hel", 0))
	s.OnAudioContextInterrupted(ctx, "s1")
	if got := server.next(t); !maps.Equal(got, map[string]any{keyStreamID: "s1", "cancel": true}) {
		t.Errorf("sent %v, want the stream canceled", got)
	}
	if svc.metricsStopped() == 0 {
		t.Error("the metrics were not stopped")
	}
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	if s.configured["s1"] || !s.canceled["s1"] {
		t.Errorf("configured %v, canceled %v", s.configured, s.canceled)
	}
	if _, ok := s.partials["s1"]; ok {
		t.Error("the stream's partial word was kept")
	}
}

// The stream ends on Soniox's side as the interruption cancels it: its
// terminated is on its way, and the cancel then finds no stream. That is the
// cancel done, not an error.
func TestTTSCancelCrossingTheStreamEndIsNotAnError(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	s, host, _ := connectedTTS(t, server, TTSConfig{})
	ctx := context.Background()

	s.OnTurnContextCreated(ctx, "stream-1")
	server.next(t)
	s.OnAudioContextInterrupted(ctx, "stream-1")
	if got := server.next(t); !maps.Equal(got, map[string]any{keyStreamID: "stream-1", "cancel": true}) {
		t.Errorf("sent %v, want the stream canceled", got)
	}
	handle(t, s, terminated("stream-1"), streamNotFound("stream-1"))
	if errs := host.reported(); len(errs) != 0 {
		t.Errorf("reported %v, want nothing", errs)
	}
}

// runTTSPipeline runs the service in a pipeline, collecting the errors it
// reports upstream and the frames reaching the end of it.
type ttsRun struct {
	t    *testing.T
	task *pipeline.Worker
	done chan error

	mu     sync.Mutex
	down   []frames.Frame
	errors []*frames.ErrorFrame
}

func runTTSPipeline(t *testing.T, b *tts.Base) *ttsRun {
	t.Helper()
	r := &ttsRun{t: t, done: make(chan error, 1)}
	r.task = pipeline.NewWorker(pipeline.New(b), pipeline.WorkerConfig{
		IdleTimeout:             -1,
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
	})
	events.On(&r.task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		r.mu.Lock()
		r.down = append(r.down, f)
		r.mu.Unlock()
	})
	events.On(&r.task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		if ef, ok := f.(*frames.ErrorFrame); ok {
			r.mu.Lock()
			r.errors = append(r.errors, ef)
			r.mu.Unlock()
		}
	})
	go func() { r.done <- r.task.Run(context.Background()) }()
	r.waitFor(func(f frames.Frame) bool { _, ok := f.(*frames.StartFrame); return ok })
	return r
}

func (r *ttsRun) waitFor(match func(frames.Frame) bool) {
	r.t.Helper()
	waitFor(r.t, "the frame waited for never came", func() bool {
		r.mu.Lock()
		defer r.mu.Unlock()
		return slices.ContainsFunc(r.down, match)
	})
}

func (r *ttsRun) frames() []frames.Frame {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.down)
}

func (r *ttsRun) reportedErrors() []*frames.ErrorFrame {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.errors)
}

func (r *ttsRun) stop() {
	r.t.Helper()
	r.task.StopWhenDone()
	select {
	case <-r.done:
	case <-time.After(5 * time.Second):
		r.t.Fatal("the pipeline did not finish")
	}
}

// The error code Soniox reports is an HTTP status, which gives the error its
// category; one that is not a number leaves it unknown. A stream that was not
// canceled and is not found is an invalid request.
func TestTTSErrorCodeSetsTheCategory(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	b, s := newTTS(TTSConfig{APIKey: "k", URL: server.url})
	r := runTTSPipeline(t, b)
	defer r.stop()

	handle(t, s,
		map[string]any{"error_code": 402, "error_type": "organization_balance_exhausted"},
		map[string]any{"error_code": 401, "error_type": "unauthorized"},
		map[string]any{"error_code": "oops"},
		streamNotFound("stream-1"),
	)
	var got []*frames.ErrorFrame
	waitFor(t, "the errors never reached the pipeline", func() bool {
		got = nil
		for _, ef := range r.reportedErrors() {
			if strings.HasPrefix(ef.Error, "Soniox TTS error") {
				got = append(got, ef)
			}
		}
		return len(got) == 4
	})
	want := []errs.Category{errs.Quota, errs.Authentication, errs.Unset, errs.InvalidRequest}
	for i, ef := range got {
		if ef.Category != want[i] && (want[i] != errs.Unset || ef.Category != errs.Unknown) {
			t.Errorf("error %d (%s) category = %q, want %q", i, ef.Error, ef.Category, want[i])
		}
	}
}

// The keepalive tells Soniox to keep the connection open without synthesizing
// anything, and runs while connected.
func TestTTSKeepalive(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	s, _, _ := connectedTTS(t, server, TTSConfig{})

	if !s.KeepaliveReady() {
		t.Fatal("the keepalive is not ready on an open connection")
	}
	if err := s.SendKeepalive(context.Background()); err != nil {
		t.Fatal(err)
	}
	if got := server.next(t); !maps.Equal(got, map[string]any{"keep_alive": true}) {
		t.Errorf("keepalive = %v", got)
	}
	if !s.ws.KeepaliveRunning() {
		t.Error("the keepalive loop is not running while connected")
	}
}

// A flush ends the named stream's text, or the playing one's when none is
// named.
func TestTTSFlushAudio(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	s, _, svc := connectedTTS(t, server, TTSConfig{})
	ctx := context.Background()

	s.FlushAudio(ctx, "s1")
	if got := server.next(t); !maps.Equal(got, map[string]any{"text": "", "text_end": true, keyStreamID: "s1"}) {
		t.Errorf("flush = %v", got)
	}
	s.FlushAudio(ctx, "")
	if !server.quiet() {
		t.Error("a flush went out with no stream to end")
	}
	svc.active = "s2"
	s.FlushAudio(ctx, "")
	if got := server.next(t); got[keyStreamID] != "s2" {
		t.Errorf("flush = %v, want the playing stream's", got)
	}
}

// A change to anything a stream is opened with closes the turn's stream out,
// so the rest of the turn opens a new one with the new values. The connection
// is kept.
func TestTTSSettingsChangeOpensANewStream(t *testing.T) {
	for name, delta := range map[string]*TTSSettings{
		"voice":               {TTS: settings.TTS{Voice: settings.Set("Emma")}},
		"model":               {TTS: settings.TTS{Base: settings.Base{Model: settings.Set("tts-rt-v1")}}},
		"language":            {TTS: settings.TTS{Language: settings.Set("fr")}},
		"speed":               {Speed: settings.Set(1.2)},
		"reduce_silence":      {ReduceSilence: settings.Set(true)},
		"client_reference_id": {ClientReferenceID: settings.Set("call-9")},
	} {
		t.Run(name, func(t *testing.T) {
			server := newTTSFakeServer(t, nil)
			s, host, _ := connectedTTS(t, server, TTSConfig{})
			changed, err := settings.Apply(s.live, delta)
			if err != nil {
				t.Fatal(err)
			}
			if err := s.UpdateSettings(context.Background(), changed); err != nil {
				t.Fatal(err)
			}
			if host.rotations() != 1 {
				t.Errorf("rotations = %d, want the turn's stream closed out", host.rotations())
			}
			cfg := s.buildConfigMsg("next")
			if !slices.ContainsFunc(slices.Collect(maps.Values(cfg)), func(v any) bool {
				return v == "Emma" || v == "tts-rt-v1" || v == "fr" || v == 1.2 || v == true || v == "call-9"
			}) {
				t.Errorf("the next stream's configuration %v does not carry the change", cfg)
			}
			if server.connections() != 1 {
				t.Error("a settings change reconnected")
			}
		})
	}
}

// A language change takes the timings of the new language from then on.
func TestTTSLanguageChangeMovesTheTimings(t *testing.T) {
	s := newTTSSynthesizer(TTSConfig{APIKey: "k"})
	s.host, s.svc = newTTSHost(), &ttsSvc{}
	delta := &TTSSettings{}
	delta.Language = settings.Set("ja")
	changed, err := settings.Apply(s.live, delta)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.UpdateSettings(context.Background(), changed); err != nil {
		t.Fatal(err)
	}
	if !s.spacelessLanguage() {
		t.Error("Japanese is not timed per character after the change")
	}
}

// A connection the server drops is restored by the receive loop.
func TestTTSReconnectsWhenTheServerDrops(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	s, _, _ := connectedTTS(t, server, TTSConfig{Websocket: wsservice.Options{
		ReconnectBackoffMinWait: time.Millisecond,
		ReconnectBackoffMaxWait: time.Millisecond,
	}})

	server.drop()
	waitFor(t, "the service never reconnected", func() bool {
		return server.connections() == 2 && s.Connected()
	})
}

// A sentence after the connection closed opens it again before it is sent.
func TestTTSSentenceReopensAClosedConnection(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	s, _, _ := connectedTTS(t, server, TTSConfig{})
	s.Disconnect(context.Background())

	if err := s.RunTTS(context.Background(), "Hello.", "s1", nil); err != nil {
		t.Fatal(err)
	}
	server.next(t)
	if got := server.next(t); got["text"] != "Hello." {
		t.Errorf("sent %v, want the sentence", got)
	}
	if n := server.connections(); n != 2 {
		t.Errorf("the server saw %d connections, want the closed one opened again", n)
	}
}

// A send that fails stops the context's speech, is reported with the failure,
// and resets the connection.
func TestTTSFailedSendResetsTheConnection(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	_, s := newTTS(TTSConfig{APIKey: "k", URL: server.url})
	s.host, s.svc = newTTSHost(), &ttsSvc{}
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
	err = s.RunTTS(context.Background(), "Hello.", "s1", func(f frames.Frame) error {
		got = append(got, f)
		return nil
	})
	if err == nil {
		t.Fatal("RunTTS accepted a failed send")
	}
	if len(got) != 1 {
		t.Fatalf("yielded %v, want one stop frame", got)
	}
	if stopped, ok := got[0].(*frames.TTSStoppedFrame); !ok || stopped.ContextID != "s1" {
		t.Errorf("yielded %v, want the context's speech stopped", got[0])
	}
	waitFor(t, "the connection was not reset", func() bool { return server.connections() == 2 })
	if !s.Connected() {
		t.Error("not connected after the reset")
	}
}

// A connection that cannot be opened fails the sentence with the failure.
func TestTTSConnectFailureFailsTheSentence(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	server.refuse = http.StatusServiceUnavailable
	_, s := newTTS(TTSConfig{APIKey: "k", URL: server.url})
	host := newTTSHost()
	s.host, s.svc = host, &ttsSvc{}

	var got []frames.Frame
	err := s.RunTTS(context.Background(), "Hello.", "s1", func(f frames.Frame) error {
		got = append(got, f)
		return nil
	})
	if err == nil {
		t.Fatal("RunTTS succeeded with no connection")
	}
	if len(got) != 1 {
		t.Fatalf("yielded %v, want the context's speech stopped", got)
	}
}

// The service connects when the pipeline starts, and a turn goes out over that
// connection as one stream: opened as the turn starts, its sentences sent to
// it, and closed by a flush once the text is complete. The stream's audio
// reaches the pipeline on the turn's context.
func TestTTSHoldsOneConnectionForThePipeline(t *testing.T) {
	server := newTTSFakeServer(t, func(msg map[string]any) []any {
		id, _ := msg[keyStreamID].(string)
		if msg["text_end"] == true {
			return []any{audioEvent(id, make([]byte, 480), []string{"H", "i", " "}, []float64{0, 0.1, 0.2}), terminated(id)}
		}
		return nil
	})
	b := NewTTS(TTSConfig{APIKey: "k", URL: server.url})
	r := runTTSPipeline(t, b)
	waitFor(t, "the server saw no connection when the pipeline started", func() bool {
		return server.connections() == 1
	})

	r.task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	r.task.QueueFrame(frames.NewLLMTextFrame("Hi. How are you? "))
	r.task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	r.waitFor(func(f frames.Frame) bool { _, ok := f.(*frames.TTSStoppedFrame); return ok })

	config := server.next(t)
	stream, _ := config[keyStreamID].(string)
	if stream == "" || config["model"] != defaultTTSModel {
		t.Fatalf("first message = %v, want the stream's configuration", config)
	}
	for _, want := range []map[string]any{
		{"text": "Hi.", "text_end": false, keyStreamID: stream},
		{"text": "How are you?", "text_end": false, keyStreamID: stream},
		{"text": "", "text_end": true, keyStreamID: stream},
	} {
		if got := server.next(t); !maps.Equal(got, want) {
			t.Errorf("sent %v, want %v", got, want)
		}
	}
	for _, f := range r.frames() {
		if a, ok := f.(*frames.TTSAudioRawFrame); ok && a.ContextID != stream {
			t.Errorf("audio on context %q, want the stream's %q", a.ContextID, stream)
		}
	}
	r.stop()

	if n := server.connections(); n != 1 {
		t.Errorf("the server saw %d connections, want one for the whole pipeline", n)
	}
}

// A settings change in the middle of a sentence still being streamed finalizes
// that sentence on the old stream, so the part of it already heard keeps its
// words, and the rest of the turn goes to a new stream opened with the new
// voice.
func TestTTSSettingsChangeMidSentenceFinalizesTheOldStream(t *testing.T) {
	server := newTTSFakeServer(t, func(msg map[string]any) []any {
		id, _ := msg[keyStreamID].(string)
		if msg["text_end"] == true {
			return []any{
				audioEvent(id, make([]byte, 480), []string{"H", "i", " ", "t", "h", "e", "r", "e"},
					[]float64{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}),
				terminated(id),
			}
		}
		return nil
	})
	b := NewTTS(TTSConfig{APIKey: "k", URL: server.url, TextAggregation: frames.TextTypeToken})
	r := runTTSPipeline(t, b)
	defer r.stop()

	r.task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	r.task.QueueFrame(frames.NewLLMTextFrame("Hi"))
	r.task.QueueFrame(frames.NewLLMTextFrame(" there"))
	voice := &TTSSettings{}
	voice.Voice = settings.Set("Emma")
	r.task.QueueFrame(frames.NewTTSUpdateSettingsFrame(voice))
	r.task.QueueFrame(frames.NewLLMTextFrame(" friend."))
	r.task.QueueFrame(frames.NewLLMFullResponseEndFrame())

	r.waitFor(func(f frames.Frame) bool {
		a, ok := f.(*frames.AggregatedTextFrame)
		return ok && a.Text == "Hi there"
	})
	r.waitFor(func(f frames.Frame) bool {
		w, ok := f.(*frames.TTSTextFrame)
		return ok && w.Text == "Hi"
	})

	old, _ := server.next(t)[keyStreamID].(string)
	var sawFlush bool
	var newStream string
	for newStream == "" {
		m := server.next(t)
		switch {
		case m[keyStreamID] == old && m["text_end"] == true:
			sawFlush = true
		case m[keyStreamID] != old && m["voice"] == "Emma":
			newStream, _ = m[keyStreamID].(string)
		}
	}
	if !sawFlush {
		t.Error("the old stream was not closed before the new one opened")
	}
}

// A connection the server refuses is reported as a connection error.
func TestTTSReportsAFailedConnection(t *testing.T) {
	server := newTTSFakeServer(t, nil)
	server.refuse = http.StatusUnauthorized
	b := NewTTS(TTSConfig{APIKey: "k", URL: server.url})
	var mu sync.Mutex
	var reported []string
	events.On(b.Events(), service.EventConnectionError, func(_ context.Context, msg string) {
		mu.Lock()
		reported = append(reported, msg)
		mu.Unlock()
	})

	r := runTTSPipeline(t, b)
	defer r.stop()
	waitFor(t, "the refused connection was never reported", func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(reported) > 0
	})
}

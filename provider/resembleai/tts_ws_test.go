package resembleai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
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
	"github.com/gojargo/jargo/utils/events"
)

// Tests for the connection held open for the call. Every sentence is a request
// of its own, numbered, and Resemble AI answers with base64 audio labeled with
// the request's number, ending each with an audio_end.

// fakeServer is a fake Resemble AI endpoint. It records every connection and
// every request the client sends, and answers each with what reply returns.
type fakeServer struct {
	url  string
	sent chan map[string]any

	mu      sync.Mutex
	reply   func(msg map[string]any) []any
	refuse  int
	conns   []*websocket.Conn
	headers []http.Header
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
			b, ok := ev.([]byte)
			if !ok {
				b, _ = json.Marshal(ev)
			}
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

func (s *fakeServer) header(i int) http.Header {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.headers[i]
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

// audioMessage is one chunk as Resemble AI sends it for request id.
func audioMessage(id any, audio []byte) map[string]any {
	return map[string]any{
		"type":          "audio",
		"request_id":    id,
		"audio_content": base64.StdEncoding.EncodeToString(audio),
	}
}

func audioEnd(id any) map[string]any {
	return map[string]any{"type": "audio_end", "request_id": id}
}

// speak answers every request with audio, in chunks of the sizes given, then
// its audio_end.
func speak(sizes ...int) func(map[string]any) []any {
	return func(msg map[string]any) []any {
		id := msg["request_id"]
		var out []any
		for _, n := range sizes {
			out = append(out, audioMessage(id, pcm(n)))
		}
		return append(out, audioEnd(id))
	}
}

// pcm is n bytes of a recognizable pattern.
func pcm(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i%250 + 1)
	}
	return b
}

// fakeSvc stands in for the TTS service the synthesizer reports through.
type fakeSvc struct {
	mu      sync.Mutex
	stopped int
}

func (f *fakeSvc) Name() string { return "ResembleTTS#0" }

func (f *fakeSvc) StopAllMetrics(context.Context) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.stopped++
}

func (f *fakeSvc) metricsStopped() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.stopped
}

// fakeHost stands in for the tts.Base the provider appends its audio to. Only
// the contexts opened with open are available.
type fakeHost struct {
	mu       sync.Mutex
	open     map[string]bool
	chunks   map[string][][]byte
	appended map[string][]frames.Frame
	words    map[string][]uctx.WordTiming
	removed  []string
	pushed   []frames.Frame
	errors   []string
}

func newFakeHost() *fakeHost {
	return &fakeHost{
		open:     map[string]bool{},
		chunks:   map[string][][]byte{},
		appended: map[string][]frames.Frame{},
		words:    map[string][]uctx.WordTiming{},
	}
}

func (h *fakeHost) openContext(ids ...string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, id := range ids {
		h.open[id] = true
	}
}

func (h *fakeHost) AppendToAudioContext(id string, f frames.Frame) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if a, ok := f.(*frames.TTSAudioRawFrame); ok {
		if a.ContextID != id {
			h.errors = append(h.errors, "frame labeled "+a.ContextID+" appended to "+id)
		}
		h.chunks[id] = append(h.chunks[id], a.Audio)
	}
	h.appended[id] = append(h.appended[id], f)
}

func (h *fakeHost) AppendWordToAudioContext(string, string, float64) {}

func (h *fakeHost) AddWordTimestamps(id string, words []uctx.WordTiming, _ tts.WordTimingOptions) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.words[id] = append(h.words[id], words...)
}

func (h *fakeHost) RemoveAudioContext(id string) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.removed = append(h.removed, id)
	delete(h.open, id)
}

func (h *fakeHost) AudioContextAvailable(id string) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.open[id]
}

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

func (h *fakeHost) chunksFor(id string) [][]byte {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.chunks[id])
}

func (h *fakeHost) audioFor(id string) []byte {
	return bytes.Join(h.chunksFor(id), nil)
}

func (h *fakeHost) appendedTo(id string) []frames.Frame {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.appended[id])
}

func (h *fakeHost) wordsFor(id string) []uctx.WordTiming {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.words[id])
}

func (h *fakeHost) wasRemoved(id string) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Contains(h.removed, id)
}

func (h *fakeHost) pushedFrames() []frames.Frame {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.pushed)
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
	if cfg.Voice == "" {
		cfg.Voice = "voice-1"
	}
	_, s := newTTS(cfg)
	host, svc := newFakeHost(), &fakeSvc{}
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

// A clean close from the server ends the reading without an error, so the
// base reconnects.
func TestReceiveReturnsWhenServerCloses(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		_ = c.Close(websocket.StatusNormalClosure, "")
	}))
	defer srv.Close()

	_, s := newTTS(Config{APIKey: "test-key", Voice: "test-voice", URL: "ws" + strings.TrimPrefix(srv.URL, "http")})
	s.host, s.svc = newFakeHost(), &fakeSvc{}
	ctx := context.Background()
	if err := s.ConnectWebsocket(ctx); err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- s.ReceiveMessages(ctx) }()
	select {
	case err := <-done:
		if err != nil {
			t.Errorf("ReceiveMessages = %v, want nil on a clean close", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("ReceiveMessages did not return when the server closed")
	}
	if err := s.DisconnectWebsocket(ctx); err != nil {
		t.Fatal(err)
	}
}

// The connection opens when the service starts, authorized by a bearer token,
// and nothing is sent until there is text.
func TestStartOpensAConnection(t *testing.T) {
	server := newFakeServer(t, nil)
	connectedSynth(t, server, Config{})

	if got := server.header(0).Get("Authorization"); got != "Bearer test-key" {
		t.Errorf("Authorization = %q, want the key as a bearer token", got)
	}
	select {
	case msg := <-server.sent:
		t.Errorf("sent %v on connecting, want nothing", msg)
	case <-time.After(50 * time.Millisecond):
	}
}

// Every sentence goes out on the one connection as a request of its own,
// numbered from zero, with the voice and the output format.
func TestSentencesAreNumberedRequests(t *testing.T) {
	server := newFakeServer(t, nil)
	s, host, _ := connectedSynth(t, server, Config{Voice: "voice-9", SampleRate: 16000, Precision: "PCM_24"})
	host.openContext("c1", "c2")

	ctx := context.Background()
	for i, id := range []string{"c1", "c2"} {
		if err := s.RunTTS(ctx, "Hello "+id+".", id, nil); err != nil {
			t.Fatal(err)
		}
		msg := server.next(t)
		want := map[string]any{
			"voice_uuid":                  "voice-9",
			"data":                        "Hello " + id + ".",
			"binary_response":             false,
			"request_id":                  float64(i),
			"output_format":               "wav",
			"sample_rate":                 float64(16000),
			"precision":                   "PCM_24",
			"no_audio_header":             true,
			"apply_custom_pronunciations": false,
		}
		for k, v := range want {
			if msg[k] != v {
				t.Errorf("request %d %s = %v, want %v", i, k, msg[k], v)
			}
		}
		if len(msg) != len(want) {
			t.Errorf("request %d = %v, want exactly %v", i, msg, want)
		}
	}
	if n := server.connections(); n != 1 {
		t.Errorf("the server saw %d connections, want 1", n)
	}
}

// A change to the settings takes effect from the next request, on the
// connection already open.
func TestSettingsChangeTheNextRequest(t *testing.T) {
	server := newFakeServer(t, nil)
	s, host, _ := connectedSynth(t, server, Config{OutputFormat: "mp3"})
	host.openContext("c1")

	delta := &TTSSettings{ApplyCustomPronunciations: settings.Set(true)}
	delta.Voice = settings.Set("voice-2")
	changed, err := settings.Apply(s.live, delta)
	if err != nil {
		t.Fatal(err)
	}
	if len(changed) != 2 {
		t.Errorf("changed = %v, want the voice and the pronunciations", changed)
	}
	if err := s.RunTTS(context.Background(), "Hello.", "c1", nil); err != nil {
		t.Fatal(err)
	}
	msg := server.next(t)
	if msg["voice_uuid"] != "voice-2" || msg["apply_custom_pronunciations"] != true || msg["output_format"] != "mp3" {
		t.Errorf("request = %v, want the new settings", msg)
	}
	if n := server.connections(); n != 1 {
		t.Errorf("the server saw %d connections, want the change to keep the connection", n)
	}
}

// Audio is held until the jitter buffer has filled, then passed on in chunks
// of the threshold size; audio_end passes on the rest, trimmed to whole
// samples, stops the context's speech and marks it done.
func TestJitterBufferAndChunking(t *testing.T) {
	server := newFakeServer(t, speak(jitterBufferBytes-1, 5000, 1000))
	s, host, _ := connectedSynth(t, server, Config{})
	host.openContext("c1")

	if err := s.RunTTS(context.Background(), "Hello.", "c1", nil); err != nil {
		t.Fatal(err)
	}
	waitUntil(t, "the context was never marked done", func() bool { return host.wasRemoved("c1") })

	total := jitterBufferBytes - 1 + 5000 + 1000
	chunks := host.chunksFor("c1")
	tail := chunks[len(chunks)-1]
	for i, c := range chunks[:len(chunks)-1] {
		if len(c) != bufferThresholdBytes {
			t.Fatalf("chunk %d is %d bytes, want %d", i, len(c), bufferThresholdBytes)
		}
	}
	if len(tail)%2 != 0 || len(tail) >= bufferThresholdBytes {
		t.Errorf("tail is %d bytes, want an even remainder", len(tail))
	}
	if got := len(host.audioFor("c1")); got != total-1 {
		t.Errorf("passed on %d bytes, want %d, the odd byte dropped", got, total-1)
	}
	want := bytes.Join([][]byte{pcm(jitterBufferBytes - 1), pcm(5000), pcm(1000)}, nil)[:total-1]
	if !bytes.Equal(host.audioFor("c1"), want) {
		t.Error("the audio passed on is not the audio received, in order")
	}
	appended := host.appendedTo("c1")
	if stopped, ok := appended[len(appended)-1].(*frames.TTSStoppedFrame); !ok || stopped.ContextID != "c1" {
		t.Errorf("last frame appended = %v, want the context's stop frame", appended[len(appended)-1])
	}
	for _, a := range host.chunksFor("c1") {
		if len(a) == 0 {
			t.Error("an empty chunk was passed on")
		}
	}
	if errs := host.reported(); len(errs) != 0 {
		t.Errorf("errors reported: %v", errs)
	}
}

// A request shorter than the jitter buffer plays all at once when it ends.
func TestShortRequestPlaysAtItsEnd(t *testing.T) {
	server := newFakeServer(t, func(msg map[string]any) []any {
		id := msg["request_id"]
		return []any{audioMessage(id, pcm(600)), audioMessage(id, pcm(400))}
	})
	s, host, _ := connectedSynth(t, server, Config{})
	host.openContext("c1")

	if err := s.RunTTS(context.Background(), "Hi.", "c1", nil); err != nil {
		t.Fatal(err)
	}
	server.next(t)
	time.Sleep(50 * time.Millisecond)
	if got := host.audioFor("c1"); len(got) != 0 {
		t.Fatalf("passed on %d bytes before the jitter buffer filled", len(got))
	}

	_, err := s.handleMessage(context.Background(), mustJSON(t, audioEnd(0)))
	if err != nil {
		t.Fatal(err)
	}
	if got := len(host.audioFor("c1")); got != 1000 {
		t.Errorf("passed on %d bytes at the end, want all 1000", got)
	}
	if !host.wasRemoved("c1") {
		t.Error("the context was not marked done")
	}
}

// Each request's audio goes to the context it was sent for, told apart by the
// request's number, however the requests interleave.
func TestAudioGoesToTheRequestsContext(t *testing.T) {
	server := newFakeServer(t, nil)
	s, host, _ := connectedSynth(t, server, Config{})
	host.openContext("c1", "c2")
	ctx := context.Background()
	for _, id := range []string{"c1", "c2"} {
		if err := s.RunTTS(ctx, "Hello.", id, nil); err != nil {
			t.Fatal(err)
		}
		server.next(t)
	}

	for _, m := range []map[string]any{
		audioMessage(1, []byte{2, 2}),
		audioMessage(0, []byte{1, 1}),
		audioEnd(1),
		audioEnd(0),
	} {
		if _, err := s.handleMessage(ctx, mustJSON(t, m)); err != nil {
			t.Fatal(err)
		}
	}
	if got := host.audioFor("c1"); !bytes.Equal(got, []byte{1, 1}) {
		t.Errorf("c1 audio = %v, want request 0's", got)
	}
	if got := host.audioFor("c2"); !bytes.Equal(got, []byte{2, 2}) {
		t.Errorf("c2 audio = %v, want request 1's", got)
	}
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	if len(s.requestToContext) != 0 || len(s.audioBuffers) != 0 || len(s.playbackStarted) != 0 {
		t.Errorf("bookkeeping left after both ended: %v %v %v",
			s.requestToContext, s.audioBuffers, s.playbackStarted)
	}
}

// Messages for a context that is no longer open, or for a request never sent,
// are dropped, an empty message is stepped over, and one that is not JSON is
// reported without ending the connection.
func TestMessagesForNoOpenContextAreDropped(t *testing.T) {
	server := newFakeServer(t, nil)
	s, host, _ := connectedSynth(t, server, Config{})
	ctx := context.Background()
	if err := s.RunTTS(ctx, "Hello.", "gone", nil); err != nil {
		t.Fatal(err)
	}
	server.next(t)

	for _, raw := range [][]byte{
		mustJSON(t, audioMessage(0, pcm(jitterBufferBytes))),
		mustJSON(t, audioEnd(0)),
		mustJSON(t, audioMessage(7, pcm(jitterBufferBytes))),
		mustJSON(t, map[string]any{"type": "error", "message": "no request"}),
		[]byte(`{}`),
	} {
		if redialed, err := s.handleMessage(ctx, raw); err != nil || redialed {
			t.Fatalf("handleMessage(%s) = %v, %v", raw, redialed, err)
		}
	}
	if _, err := s.handleMessage(ctx, []byte("not json")); err != nil {
		t.Fatalf("a message that is not JSON failed the connection: %v", err)
	}
	if errs := host.reported(); len(errs) != 1 || !strings.HasPrefix(errs[0], "Received invalid JSON: not json") {
		t.Errorf("reported %v, want only the invalid JSON", errs)
	}
	if host.wasRemoved("gone") || len(host.audioFor("gone")) != 0 {
		t.Error("a message for a closed context was acted on")
	}
}

// Audio that is not base64 fails the connection rather than being played as
// rubbish.
func TestUndecodableAudioFailsTheConnection(t *testing.T) {
	s := newSynthesizer(Config{APIKey: "k", Voice: "v"})
	host := newFakeHost()
	s.host, s.svc = host, &fakeSvc{}
	host.openContext("c1")
	s.requestToContext[0] = "c1"

	_, err := s.handleMessage(context.Background(),
		[]byte(`{"type":"audio","request_id":0,"audio_content":"not base64!!"}`))
	if err == nil {
		t.Error("audio that is not base64 was accepted")
	}
}

// The timings reported with the audio are passed on per character, at the
// start of each, once the request's audio has started playing.
func TestTimestampsAreReported(t *testing.T) {
	s := newSynthesizer(Config{APIKey: "k", Voice: "v"})
	host := newFakeHost()
	s.host, s.svc = host, &fakeSvc{}
	host.openContext("c1")
	s.requestToContext[0] = "c1"

	timed := func(audio []byte) []byte {
		m := audioMessage(0, audio)
		m["audio_timestamps"] = map[string]any{
			"graph_chars": []string{"H", "i", "!"},
			"graph_times": [][]float64{{0, 0.1}, {0.1, 0.2}, {0.2}},
		}
		return mustJSON(t, m)
	}
	ctx := context.Background()
	if _, err := s.handleMessage(ctx, timed(pcm(100))); err != nil {
		t.Fatal(err)
	}
	if got := host.wordsFor("c1"); len(got) != 0 {
		t.Errorf("timings %v reported while the jitter buffer was filling", got)
	}
	if _, err := s.handleMessage(ctx, timed(pcm(jitterBufferBytes))); err != nil {
		t.Fatal(err)
	}
	want := []uctx.WordTiming{{Word: "H", Offset: 0}, {Word: "i", Offset: 0.1}}
	if got := host.wordsFor("c1"); !slices.Equal(got, want) {
		t.Errorf("timings = %v, want %v", got, want)
	}
}

// An error message is reported twice, as the server worded it and as the
// service's, stops the request's speech at once and stops the metrics, and
// leaves the connection alone.
func TestErrorMessageIsReported(t *testing.T) {
	server := newFakeServer(t, func(msg map[string]any) []any {
		id := msg["request_id"]
		return []any{
			audioMessage(id, pcm(100)),
			map[string]any{
				"type": "error", "request_id": id,
				"error_name": "BadRequest", "message": "voice not found", "status_code": 400,
			},
		}
	})
	s, host, svc := connectedSynth(t, server, Config{})
	host.openContext("c1")

	if err := s.RunTTS(context.Background(), "Hello.", "c1", nil); err != nil {
		t.Fatal(err)
	}
	waitUntil(t, "the error was never reported", func() bool { return len(host.reported()) == 2 })
	errs := host.reported()
	if errs[0] != "Error: BadRequest (status 400): voice not found" {
		t.Errorf("first error = %q", errs[0])
	}
	if errs[1] != "ResembleTTS#0 error: BadRequest - voice not found" {
		t.Errorf("second error = %q", errs[1])
	}
	pushed := host.pushedFrames()
	if len(pushed) != 1 {
		t.Fatalf("pushed %v, want the stop frame", pushed)
	}
	if stopped, ok := pushed[0].(*frames.TTSStoppedFrame); !ok || stopped.ContextID != "c1" {
		t.Errorf("pushed %v, want the context's stop frame", pushed[0])
	}
	if svc.metricsStopped() == 0 {
		t.Error("the metrics were not stopped")
	}
	s.stateMu.Lock()
	buffered := len(s.audioBuffers)
	s.stateMu.Unlock()
	if buffered != 0 {
		t.Error("the request's audio is still buffered")
	}
	time.Sleep(50 * time.Millisecond)
	if n := server.connections(); n != 1 || !s.Connected() {
		t.Errorf("connections = %d, want the one kept", n)
	}
}

// Rejected credentials close the connection and open a new one.
func TestAuthErrorRedials(t *testing.T) {
	for _, status := range []int{401, 403} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			var once sync.Once
			server := newFakeServer(t, func(msg map[string]any) []any {
				var out []any
				once.Do(func() {
					out = []any{map[string]any{
						"type": "error", "request_id": msg["request_id"],
						"error_name": "Unauthorized", "message": "bad key", "status_code": status,
					}}
				})
				return out
			})
			s, host, _ := connectedSynth(t, server, Config{Websocket: wsservice.Options{
				ReconnectBackoffMinWait: time.Millisecond,
				ReconnectBackoffMaxWait: time.Millisecond,
			}})
			host.openContext("c1")

			if err := s.RunTTS(context.Background(), "Hello.", "c1", nil); err != nil {
				t.Fatal(err)
			}
			waitUntil(t, "the service never redialed", func() bool { return server.connections() >= 2 })
			waitUntil(t, "the service is not connected after the redial", s.Connected)
		})
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
	s, host, _ := connectedSynth(t, server, Config{})
	host.openContext("c1")
	s.Disconnect(context.Background())

	if err := s.RunTTS(context.Background(), "Hello.", "c1", nil); err != nil {
		t.Fatal(err)
	}
	if got := server.next(t); got["data"] != "Hello." {
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
	_, s := newTTS(Config{APIKey: "k", Voice: "v", URL: server.url})
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
	waitUntil(t, "the connection was not reset", func() bool { return server.connections() == 2 })
	if !s.Connected() {
		t.Error("not connected after the reset")
	}
}

// Closing the connection does not wait on a close acknowledgment Resemble AI
// never sends.
func TestUnacknowledgedCloseDoesNotDelay(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer func() { _ = c.CloseNow() }()
		// Deaf from here: the client's close frame is never answered.
		<-r.Context().Done()
	}))
	defer srv.Close()

	_, s := newTTS(Config{APIKey: "k", Voice: "v", URL: "ws" + strings.TrimPrefix(srv.URL, "http")})
	s.host, s.svc = newFakeHost(), &fakeSvc{}
	s.Start(context.Background())
	if !s.Connected() {
		t.Fatal("not connected")
	}
	start := time.Now()
	s.Disconnect(context.Background())
	if elapsed := time.Since(start); elapsed > 500*time.Millisecond {
		t.Errorf("Disconnect took %v, want it to return without waiting on the close", elapsed)
	}
}

func mustJSON(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
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

func (r *run) frames() []frames.Frame {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.down)
}

func (r *run) stop() {
	r.t.Helper()
	r.task.StopWhenDone()
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

// The service connects when the pipeline starts, and the sentences of one
// utterance go out over that connection as requests of their own, each spoken
// on a context of its own that its audio_end ends.
func TestEachSentenceHasAContextOfItsOwn(t *testing.T) {
	server := newFakeServer(t, speak(480))
	b := NewTTS(Config{APIKey: "k", Voice: "v", URL: server.url})
	r := runPipeline(t, b)
	waitUntil(t, "the server saw no connection when the pipeline started", func() bool {
		return server.connections() == 1
	})

	r.task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	r.task.QueueFrame(frames.NewLLMTextFrame("Hello there. How are you? "))
	r.task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	stops := func() []string {
		var ids []string
		for _, f := range r.frames() {
			if s, ok := f.(*frames.TTSStoppedFrame); ok {
				ids = append(ids, s.ContextID)
			}
		}
		return ids
	}
	waitUntil(t, "both sentences never finished", func() bool { return len(stops()) == 2 })
	ids := stops()
	if ids[0] == ids[1] || ids[0] == "" {
		t.Errorf("sentences stopped on contexts %v, want one each", ids)
	}
	var audio []string
	for _, f := range r.frames() {
		if a, ok := f.(*frames.TTSAudioRawFrame); ok && !slices.Contains(audio, a.ContextID) {
			audio = append(audio, a.ContextID)
		}
	}
	if !slices.Equal(audio, ids) {
		t.Errorf("audio came on contexts %v, want %v", audio, ids)
	}
	r.stop()

	if n := server.connections(); n != 1 {
		t.Errorf("the server saw %d connections, want one for the whole pipeline", n)
	}
}

// A connection the server refuses is reported as a connection error.
func TestReportsAFailedConnection(t *testing.T) {
	server := newFakeServer(t, nil)
	server.refuse = http.StatusUnauthorized
	b := NewTTS(Config{APIKey: "k", Voice: "v", URL: server.url})
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
